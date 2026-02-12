//! Runtime Sweep conversation manager for online next-edit prediction.
//!
//! This manager mirrors Sweep's training format at inference time by:
//! - Tracking file state, cursor location, and opened files
//! - Building a single Sweep-formatted system message for the current state
//! - Parsing model output (`<|file_sep|>updated/...`) back into line-based edits

use std::collections::{HashMap, HashSet};

use crate::conversation::{ConversationMessage, Role};
use crate::diff::compute_changed_block_lines;
use crate::helpers::{apply_change, floor_char_boundary};
use crate::sweep_format::{sweep_system_prompt, SweepHistoryCenterMode, SweepOpenedFileContextMode};
use crate::{COALESCE_RADIUS, Tokenizer};

const DEFAULT_SWEEP_VIEWPORT_LINES: usize = 21;
const DEFAULT_MAX_HISTORY_ENTRIES: usize = 64;

#[derive(Debug, Clone)]
struct SweepHistoryEntry {
    file_path: String,
    before_viewport: String,
    after_viewport: String,
}

#[derive(Debug, Clone, Copy)]
struct EditRegion {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
struct SweepPromptContext {
    target_file: String,
    window_start_line: usize,
    current_window: String,
}

/// Configuration for runtime Sweep state management.
#[derive(Debug, Clone)]
pub struct SweepConversationStateManagerConfig {
    /// Fixed viewport line count for target/history windows.
    pub viewport_lines: usize,
    /// How to include opened files in context.
    pub opened_file_context_mode: SweepOpenedFileContextMode,
    /// How to center history windows.
    pub history_center_mode: SweepHistoryCenterMode,
    /// Coalesce radius (lines) for grouping nearby content edits.
    pub coalesce_radius: usize,
    /// Maximum number of history entries to keep.
    pub max_history_entries: usize,
    /// System prompt prepended to Sweep context blocks.
    pub system_prompt: String,
    /// Special tokens added per user/system message by the chat template.
    pub special_tokens_per_user_message: usize,
    /// Special tokens added per assistant message by the chat template.
    pub special_tokens_per_assistant_message: usize,
    /// Fixed special tokens at conversation start.
    pub conversation_start_tokens: usize,
}

impl Default for SweepConversationStateManagerConfig {
    fn default() -> Self {
        Self {
            viewport_lines: DEFAULT_SWEEP_VIEWPORT_LINES,
            opened_file_context_mode: SweepOpenedFileContextMode::Full,
            history_center_mode: SweepHistoryCenterMode::ChangedBlock,
            coalesce_radius: COALESCE_RADIUS,
            max_history_entries: DEFAULT_MAX_HISTORY_ENTRIES,
            system_prompt: sweep_system_prompt(),
            special_tokens_per_user_message: 0,
            special_tokens_per_assistant_message: 0,
            conversation_start_tokens: 0,
        }
    }
}

/// Sweep prompt payload produced for the model.
#[derive(Debug, Clone)]
pub struct SweepRuntimePrompt {
    pub messages: Vec<ConversationMessage>,
    pub token_count: usize,
    pub target_file: String,
    pub window_start_line: usize,
    pub window_end_line: usize,
    pub current_window: String,
}

/// Parsed model edit kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepModelEditKind {
    Insert,
    Delete,
    Replace,
}

/// Parsed model edit in 1-based line coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SweepModelEdit {
    pub target_file: String,
    pub kind: SweepModelEditKind,
    /// 1-based start line in target file.
    pub start_line: usize,
    /// 1-based end line in target file for delete/replace (inclusive).
    pub end_line: Option<usize>,
    /// Replacement/insert text payload.
    pub text: Option<String>,
}

/// Runtime Sweep conversation state manager.
pub struct SweepConversationStateManager<T>
where
    T: Tokenizer,
{
    tokenizer: T,
    config: SweepConversationStateManagerConfig,
    file_states: HashMap<String, String>,
    opened_set: HashSet<String>,
    opened_order: Vec<String>,
    active_file: Option<String>,
    last_cursor_line_by_file: HashMap<String, usize>,
    pending_edits_before: HashMap<String, Option<String>>,
    pending_edit_regions: HashMap<String, Option<EditRegion>>,
    history: Vec<SweepHistoryEntry>,
    last_prompt_context: Option<SweepPromptContext>,
}

impl<T> SweepConversationStateManager<T>
where
    T: Tokenizer,
{
    pub fn new(tokenizer: T, config: SweepConversationStateManagerConfig) -> Self {
        Self {
            tokenizer,
            config,
            file_states: HashMap::new(),
            opened_set: HashSet::new(),
            opened_order: Vec::new(),
            active_file: None,
            last_cursor_line_by_file: HashMap::new(),
            pending_edits_before: HashMap::new(),
            pending_edit_regions: HashMap::new(),
            history: Vec::new(),
            last_prompt_context: None,
        }
    }

    /// Reset all runtime state.
    pub fn reset(&mut self) {
        self.file_states.clear();
        self.opened_set.clear();
        self.opened_order.clear();
        self.active_file = None;
        self.last_cursor_line_by_file.clear();
        self.pending_edits_before.clear();
        self.pending_edit_regions.clear();
        self.history.clear();
        self.last_prompt_context = None;
    }

    /// Get current content snapshot for a file.
    pub fn get_file_content(&self, file_path: &str) -> String {
        self.file_states.get(file_path).cloned().unwrap_or_default()
    }

    fn mark_opened(&mut self, file_path: &str) {
        if self.opened_set.insert(file_path.to_string()) {
            self.opened_order.push(file_path.to_string());
        }
    }

    /// Handle tab/file switch event.
    ///
    /// `text_content` should be provided for first-open snapshots.
    pub fn handle_tab_event(&mut self, file_path: &str, text_content: Option<&str>) {
        self.flush_all_pending_edits();
        self.active_file = Some(file_path.to_string());
        self.mark_opened(file_path);
        if let Some(text) = text_content {
            let content = text.replace("\\n", "\n").replace("\\r", "\r");
            self.file_states.insert(file_path.to_string(), content);
        }
    }

    /// Handle content change event.
    pub fn handle_content_event(
        &mut self,
        file_path: &str,
        offset: usize,
        length: usize,
        new_text: &str,
    ) {
        self.active_file = Some(file_path.to_string());
        self.mark_opened(file_path);

        let before = self.file_states.get(file_path).cloned().unwrap_or_default();
        let safe_offset = floor_char_boundary(&before, offset.min(before.len()));
        let safe_end = floor_char_boundary(&before, (offset + length).min(before.len()));
        let start_line_current = before[..safe_offset].matches('\n').count() + 1;
        let deleted_content = &before[safe_offset..safe_end];
        let lines_added = new_text.matches('\n').count();
        let lines_deleted = deleted_content.matches('\n').count();
        let region_start = start_line_current;
        let region_end = start_line_current + lines_added.max(lines_deleted);

        let current_region = self.pending_edit_regions.get(file_path).and_then(|r| *r);
        if let Some(region) = current_region {
            if region_start < region.start.saturating_sub(self.config.coalesce_radius)
                || region_start > region.end + self.config.coalesce_radius
            {
                self.flush_pending_edit_for_file(file_path);
            }
        }

        let after = apply_change(&before, offset, length, new_text);
        if self
            .pending_edits_before
            .get(file_path)
            .and_then(|v| v.as_ref())
            .is_none()
        {
            self.pending_edits_before
                .insert(file_path.to_string(), Some(before));
        }

        let current_region = self.pending_edit_regions.get(file_path).and_then(|r| *r);
        let new_region = if let Some(region) = current_region {
            EditRegion {
                start: region.start.min(region_start),
                end: region.end.max(region_end),
            }
        } else {
            EditRegion {
                start: region_start,
                end: region_start.max(region_end),
            }
        };
        self.pending_edit_regions
            .insert(file_path.to_string(), Some(new_region));
        self.file_states.insert(file_path.to_string(), after);
    }

    /// Handle selection event, updating cursor line for target context centering.
    pub fn handle_selection_event(&mut self, file_path: &str, offset: usize) {
        self.active_file = Some(file_path.to_string());
        self.mark_opened(file_path);
        let content = self.file_states.get(file_path).cloned().unwrap_or_default();
        let safe_offset = floor_char_boundary(&content, offset.min(content.len()));
        let target_line = content[..safe_offset].matches('\n').count() + 1;
        self.last_cursor_line_by_file
            .insert(file_path.to_string(), target_line.max(1));
    }

    /// Handle explicit cursor line updates (1-based).
    pub fn handle_cursor_by_line(&mut self, file_path: &str, line: usize) {
        self.active_file = Some(file_path.to_string());
        self.mark_opened(file_path);
        self.last_cursor_line_by_file
            .insert(file_path.to_string(), line.max(1));
    }

    /// Terminal events do not emit Sweep blocks, but command/focus boundaries flush pending edits.
    pub fn handle_terminal_command_event(&mut self, _command: &str) {
        self.flush_all_pending_edits();
    }

    pub fn handle_terminal_output_event(&mut self, _output: &str) {}

    pub fn handle_terminal_focus_event(&mut self) {
        self.flush_all_pending_edits();
    }

    pub fn handle_git_branch_checkout_event(&mut self, _branch_info: &str) {
        self.flush_all_pending_edits();
    }

    fn flush_pending_edit_for_file(&mut self, file_path: &str) {
        let before_snapshot = match self.pending_edits_before.get(file_path) {
            Some(Some(s)) => s.clone(),
            _ => return,
        };
        let after_snapshot = self.file_states.get(file_path).cloned().unwrap_or_default();

        if before_snapshot.trim_end_matches('\n') == after_snapshot.trim_end_matches('\n') {
            self.pending_edits_before.insert(file_path.to_string(), None);
            self.pending_edit_regions.insert(file_path.to_string(), None);
            return;
        }

        let Ok(changed) = compute_changed_block_lines(&before_snapshot, &after_snapshot) else {
            self.pending_edits_before.insert(file_path.to_string(), None);
            self.pending_edit_regions.insert(file_path.to_string(), None);
            return;
        };

        let (before_center, after_center) = match self.config.history_center_mode {
            SweepHistoryCenterMode::ChangedBlock => {
                (changed.start_before.max(1), changed.start_after.max(1))
            }
            SweepHistoryCenterMode::Cursor => {
                let cursor_line = self.last_cursor_line_by_file.get(file_path).copied();
                match cursor_line {
                    Some(line) if line > 0 => (line, line),
                    _ => (changed.start_before.max(1), changed.start_after.max(1)),
                }
            }
        };

        self.history.push(SweepHistoryEntry {
            file_path: file_path.to_string(),
            before_viewport: viewport_text(&before_snapshot, before_center, self.config.viewport_lines),
            after_viewport: viewport_text(&after_snapshot, after_center, self.config.viewport_lines),
        });

        if self.history.len() > self.config.max_history_entries {
            let to_drop = self.history.len() - self.config.max_history_entries;
            self.history.drain(0..to_drop);
        }

        self.pending_edits_before.insert(file_path.to_string(), None);
        self.pending_edit_regions.insert(file_path.to_string(), None);
    }

    fn flush_all_pending_edits(&mut self) {
        let files: Vec<String> = self.pending_edits_before.keys().cloned().collect();
        for file in files {
            self.flush_pending_edit_for_file(&file);
        }
    }

    fn choose_target_file(&self) -> Option<String> {
        if let Some(active) = &self.active_file {
            return Some(active.clone());
        }
        if let Some(last_opened) = self.opened_order.last() {
            return Some(last_opened.clone());
        }
        self.file_states.keys().next().cloned()
    }

    fn compute_token_count(&self, messages: &[ConversationMessage]) -> usize {
        self.config.conversation_start_tokens
            + messages
                .iter()
                .map(|m| {
                    let overhead = match m.role {
                        Role::Assistant => self.config.special_tokens_per_assistant_message,
                        Role::System | Role::User => self.config.special_tokens_per_user_message,
                    };
                    self.tokenizer.count_tokens(&m.content) + overhead
                })
                .sum::<usize>()
    }

    /// Build a Sweep-formatted prompt for current state.
    pub fn finalize_for_model(&mut self) -> SweepRuntimePrompt {
        self.flush_all_pending_edits();
        let Some(target_file) = self.choose_target_file() else {
            self.last_prompt_context = None;
            return SweepRuntimePrompt {
                messages: Vec::new(),
                token_count: 0,
                target_file: String::new(),
                window_start_line: 1,
                window_end_line: 0,
                current_window: String::new(),
            };
        };

        let target_content = self.file_states.get(&target_file).cloned().unwrap_or_default();
        let cursor_line = self
            .last_cursor_line_by_file
            .get(&target_file)
            .copied()
            .unwrap_or(1)
            .max(1);
        let target_line_count = split_lines_lossless(&target_content).len();
        let (window_start_line, window_end_line) =
            compute_window_bounds(target_line_count, cursor_line, self.config.viewport_lines);
        let current_window = viewport_text(&target_content, cursor_line, self.config.viewport_lines);

        let mut blocks = Vec::new();
        for opened_file in &self.opened_order {
            if opened_file == &target_file {
                continue;
            }
            let Some(content) = self.file_states.get(opened_file) else {
                continue;
            };
            let body = match self.config.opened_file_context_mode {
                SweepOpenedFileContextMode::Full => content.clone(),
                SweepOpenedFileContextMode::Viewport => {
                    let center = self
                        .last_cursor_line_by_file
                        .get(opened_file)
                        .copied()
                        .unwrap_or(1);
                    viewport_text(content, center, self.config.viewport_lines)
                }
            };
            blocks.push(format_file_sep_block(opened_file, &body));
        }

        for entry in &self.history {
            blocks.push(format_file_sep_block(
                &format!("{}.diff", entry.file_path),
                &format!(
                    "original:\n{}\nupdated:\n{}",
                    entry.before_viewport, entry.after_viewport
                ),
            ));
        }

        let scoped_original = scoped_target_path("original", &target_file);
        let scoped_current = scoped_target_path("current", &target_file);
        blocks.push(format_file_sep_block(&scoped_original, &current_window));
        blocks.push(format_file_sep_block(&scoped_current, &current_window));

        let system_content = format!("{}\n\n{}", self.config.system_prompt, blocks.join("\n"));
        let messages = vec![ConversationMessage {
            role: Role::System,
            content: system_content,
        }];
        let token_count = self.compute_token_count(&messages);

        self.last_prompt_context = Some(SweepPromptContext {
            target_file: target_file.clone(),
            window_start_line,
            current_window: current_window.clone(),
        });

        SweepRuntimePrompt {
            messages,
            token_count,
            target_file,
            window_start_line,
            window_end_line,
            current_window,
        }
    }

    /// Parse model output into a line-based edit against the last finalized prompt.
    ///
    /// Returns `Ok(None)` when no valid `updated/<target>` block is found or when
    /// the resulting window is unchanged.
    pub fn parse_model_response(&self, response: &str) -> Result<Option<SweepModelEdit>, String> {
        let Some(ctx) = &self.last_prompt_context else {
            return Err("No prompt context available. Call finalize_for_model() first.".to_string());
        };

        let Some(updated_window_raw) = extract_updated_window(response, &ctx.target_file) else {
            return Ok(None);
        };
        let updated_window = normalize_block_body(&updated_window_raw);
        let current_window = normalize_block_body(&ctx.current_window);
        if updated_window == current_window {
            return Ok(None);
        }

        let changed = match compute_changed_block_lines(&current_window, &updated_window) {
            Ok(changed) => changed,
            Err(_) => return Ok(None),
        };

        let abs_start = ctx.window_start_line + changed.start_before - 1;
        if changed.end_before < changed.start_before {
            let text = if changed.replacement_lines.is_empty() {
                None
            } else {
                Some(format!("{}\n", changed.replacement_lines.join("\n")))
            };
            return Ok(Some(SweepModelEdit {
                target_file: ctx.target_file.clone(),
                kind: SweepModelEditKind::Insert,
                start_line: abs_start,
                end_line: None,
                text,
            }));
        }

        let abs_end = ctx.window_start_line + changed.end_before - 1;
        if changed.replacement_lines.is_empty() {
            return Ok(Some(SweepModelEdit {
                target_file: ctx.target_file.clone(),
                kind: SweepModelEditKind::Delete,
                start_line: abs_start,
                end_line: Some(abs_end),
                text: None,
            }));
        }

        Ok(Some(SweepModelEdit {
            target_file: ctx.target_file.clone(),
            kind: SweepModelEditKind::Replace,
            start_line: abs_start,
            end_line: Some(abs_end),
            text: Some(format!("{}\n", changed.replacement_lines.join("\n"))),
        }))
    }
}

fn split_lines_lossless(content: &str) -> Vec<&str> {
    if content.is_empty() {
        Vec::new()
    } else {
        content.lines().collect()
    }
}

fn compute_window_bounds(
    total_lines: usize,
    center_line_one_based: usize,
    viewport_lines: usize,
) -> (usize, usize) {
    if total_lines == 0 {
        return (1, 0);
    }

    let viewport = viewport_lines.max(1).min(total_lines);
    let center = center_line_one_based.max(1).min(total_lines);

    let above = viewport / 2;
    let below = viewport - above - 1;

    let mut start = center.saturating_sub(above).max(1);
    let mut end = (center + below).min(total_lines);

    let current = end - start + 1;
    if current < viewport {
        let missing = viewport - current;
        let extend_up = missing.min(start.saturating_sub(1));
        start -= extend_up;
        let remaining = missing - extend_up;
        end = (end + remaining).min(total_lines);
    }

    (start, end)
}

fn viewport_text(content: &str, center_line_one_based: usize, viewport_lines: usize) -> String {
    let lines = split_lines_lossless(content);
    if lines.is_empty() {
        return String::new();
    }
    let (start, end) = compute_window_bounds(lines.len(), center_line_one_based, viewport_lines);
    lines[start - 1..end].join("\n")
}

fn format_file_sep_block(path: &str, body: &str) -> String {
    format!("<|file_sep|>{}\n{}", path, body)
}

fn scoped_target_path(scope: &str, target_file: &str) -> String {
    if target_file.starts_with('/') {
        format!("{}{}", scope, target_file)
    } else {
        format!("{}/{}", scope, target_file)
    }
}

fn normalize_block_body(body: &str) -> String {
    body.replace("\r\n", "\n")
        .replace('\r', "\n")
        .trim_end_matches('\n')
        .to_string()
}

fn extract_last_fenced_block(text: &str) -> Option<String> {
    let mut last_block: Option<String> = None;
    let mut search_start = 0usize;

    while let Some(rel_start) = text[search_start..].find("```") {
        let start = search_start + rel_start;
        let mut fence_end = start;
        while fence_end < text.len() && text.as_bytes()[fence_end] == b'`' {
            fence_end += 1;
        }
        let fence = &text[start..fence_end];
        let Some(line_break_rel) = text[fence_end..].find('\n') else {
            break;
        };
        let line_break = fence_end + line_break_rel;
        let closing_marker = format!("\n{}", fence);
        let Some(closing_rel) = text[line_break + 1..].find(&closing_marker) else {
            search_start = fence_end;
            continue;
        };
        let closing = line_break + 1 + closing_rel;
        last_block = Some(text[line_break + 1..closing].trim_end().to_string());
        search_start = closing + 1 + fence.len();
    }

    last_block
}

fn extract_updated_window_from_candidate(candidate: &str, target_file: &str) -> Option<String> {
    let scoped = scoped_target_path("updated", target_file);
    let marker = format!("<|file_sep|>{}", scoped);
    let generic_marker = "<|file_sep|>updated/";

    let marker_start = candidate
        .rfind(&marker)
        .or_else(|| candidate.rfind(generic_marker))?;

    let marker_line_end = candidate[marker_start..]
        .find('\n')
        .map(|idx| marker_start + idx)
        .unwrap_or(candidate.len());
    let body_start = if marker_line_end < candidate.len() {
        marker_line_end + 1
    } else {
        marker_line_end
    };
    let rest = &candidate[body_start..];
    let body_end = rest.find("\n<|file_sep|>").unwrap_or(rest.len());
    Some(rest[..body_end].to_string())
}

fn extract_updated_window(response: &str, target_file: &str) -> Option<String> {
    if let Some(block) = extract_last_fenced_block(response) {
        if let Some(parsed) = extract_updated_window_from_candidate(&block, target_file) {
            return Some(parsed);
        }
    }
    extract_updated_window_from_candidate(response, target_file)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CharApproxTokenizer;
    impl Tokenizer for CharApproxTokenizer {
        fn count_tokens(&self, text: &str) -> usize {
            text.len() / 4
        }

        fn truncate_to_max_tokens(&self, text: &str, max_tokens: usize) -> String {
            text.chars().take(max_tokens * 4).collect()
        }
    }

    fn new_manager() -> SweepConversationStateManager<CharApproxTokenizer> {
        SweepConversationStateManager::new(
            CharApproxTokenizer,
            SweepConversationStateManagerConfig::default(),
        )
    }

    #[test]
    fn test_finalize_for_model_builds_sweep_prompt() {
        let mut manager = new_manager();
        manager.handle_tab_event("/a.py", Some("a1\na2\na3"));
        manager.handle_selection_event("/a.py", 0);

        let prompt = manager.finalize_for_model();
        assert_eq!(prompt.messages.len(), 1);
        assert_eq!(prompt.messages[0].role, Role::System);
        assert!(prompt.messages[0].content.contains("<|file_sep|>original/a.py"));
        assert!(prompt.messages[0].content.contains("<|file_sep|>current/a.py"));
        assert_eq!(prompt.target_file, "/a.py");
        assert_eq!(prompt.window_start_line, 1);
        assert_eq!(prompt.window_end_line, 3);
        assert_eq!(prompt.current_window, "a1\na2\na3");
    }

    #[test]
    fn test_finalize_for_model_includes_edit_history() {
        let mut manager = new_manager();
        manager.handle_tab_event("/a.py", Some("a1\na2\na3"));

        // Replace line 2: "a2" -> "changed"
        // Offsets: "a1\n" is 3 chars, line2 length is 2.
        manager.handle_content_event("/a.py", 3, 2, "changed");
        manager.handle_selection_event("/a.py", 0);

        let prompt = manager.finalize_for_model();
        assert!(prompt.messages[0].content.contains("<|file_sep|>/a.py.diff"));
        assert!(prompt.messages[0].content.contains("original:"));
        assert!(prompt.messages[0].content.contains("updated:"));
    }

    #[test]
    fn test_parse_model_response_replace() {
        let mut manager = new_manager();
        manager.handle_tab_event("/a.py", Some("a1\na2\na3"));
        manager.handle_selection_event("/a.py", 0);
        let prompt = manager.finalize_for_model();

        let response = format!(
            "<|file_sep|>updated/a.py\n{}",
            prompt.current_window.replace("a2", "updated")
        );

        let edit = manager.parse_model_response(&response).unwrap().unwrap();
        assert_eq!(edit.target_file, "/a.py");
        assert_eq!(edit.kind, SweepModelEditKind::Replace);
        assert_eq!(edit.start_line, 2);
        assert_eq!(edit.end_line, Some(2));
        assert_eq!(edit.text.as_deref(), Some("updated\n"));
    }

    #[test]
    fn test_parse_model_response_no_change_returns_none() {
        let mut manager = new_manager();
        manager.handle_tab_event("/a.py", Some("a1\na2\na3"));
        manager.handle_selection_event("/a.py", 0);
        let prompt = manager.finalize_for_model();

        let response = format!("<|file_sep|>updated/a.py\n{}", prompt.current_window);
        let edit = manager.parse_model_response(&response).unwrap();
        assert!(edit.is_none());
    }

    #[test]
    fn test_parse_model_response_without_block_returns_none() {
        let mut manager = new_manager();
        manager.handle_tab_event("/a.py", Some("a1\na2\na3"));
        manager.handle_selection_event("/a.py", 0);
        manager.finalize_for_model();

        let edit = manager.parse_model_response("no sweep block").unwrap();
        assert!(edit.is_none());
    }
}
