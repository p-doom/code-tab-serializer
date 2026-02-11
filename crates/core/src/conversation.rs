//! Conversation state manager for serializing IDE events into conversation format.

use std::collections::{HashMap, HashSet};

use serde::{Serialize, Serializer};

use crate::diff::compute_changed_block_lines;
use crate::helpers::{
    clean_text, escape_single_quotes_for_sed, fenced_block, floor_char_boundary,
    line_numbered_output, normalize_terminal_output, serialize_compute_viewport, Viewport,
};
use crate::Tokenizer;
use crate::{
    COALESCE_RADIUS, MAX_TOKENS_PER_MESSAGE, MAX_TOKENS_PER_TERMINAL_OUTPUT, VIEWPORT_RADIUS,
};

/// Role in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Serialize for Role {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Role::System => serializer.serialize_str("system"),
            Role::User => serializer.serialize_str("user"),
            Role::Assistant => serializer.serialize_str("assistant"),
        }
    }
}

/// A single message in the conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationMessage {
    pub role: Role,
    pub content: String,
}

impl ConversationMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Configuration for the ConversationStateManager.
#[derive(Debug, Clone)]
pub struct ConversationStateManagerConfig {
    pub viewport_radius: usize,
    pub coalesce_radius: usize,
    pub max_tokens_per_message: usize,
    pub max_tokens_per_terminal_output: usize,
    /// Maximum tokens per conversation chunk (for preprocessing). None = no chunking.
    pub max_tokens_per_conversation: Option<usize>,
    /// Minimum messages required to keep a conversation chunk.
    pub min_conversation_messages: usize,
    /// System prompt to include when counting conversation tokens.
    /// This ensures accurate token limits that account for the system prompt.
    pub system_prompt: Option<String>,
    /// Special tokens added per user/system message by the chat template.
    pub special_tokens_per_user_message: usize,
    /// Special tokens added per assistant message by the chat template.
    pub special_tokens_per_assistant_message: usize,
    /// Fixed special tokens at conversation start.
    pub conversation_start_tokens: usize,
}

impl Default for ConversationStateManagerConfig {
    fn default() -> Self {
        Self {
            viewport_radius: VIEWPORT_RADIUS,
            coalesce_radius: COALESCE_RADIUS,
            max_tokens_per_message: MAX_TOKENS_PER_MESSAGE,
            max_tokens_per_terminal_output: MAX_TOKENS_PER_TERMINAL_OUTPUT,
            max_tokens_per_conversation: None, // No chunking by default (for extension)
            min_conversation_messages: 5,
            system_prompt: None,
            special_tokens_per_user_message: 0,
            special_tokens_per_assistant_message: 0,
            conversation_start_tokens: 0,
        }
    }
}

/// A finalized conversation with its token count.
#[derive(Debug, Clone)]
pub struct FinalizedConversation {
    pub messages: Vec<ConversationMessage>,
    pub token_count: usize,
}

/// Edit region tracking for coalescing nearby edits.
#[derive(Debug, Clone, Copy)]
struct EditRegion {
    start: usize,
    end: usize,
}

/// Manages conversation state for serializing IDE events.
///
/// The tokenizer is provided externally, allowing the caller to use
/// either a character-based approximation (for runtime) or an accurate tokenizer
/// (for preprocessing).
pub struct ConversationStateManager<T>
where
    T: Tokenizer,
{
    tokenizer: T,
    config: ConversationStateManagerConfig,
    // Current conversation being built
    messages: Vec<ConversationMessage>,
    current_tokens: usize,
    // Finalized conversations (for chunking mode)
    finalized_conversations: Vec<FinalizedConversation>,
    // File state tracking
    file_states: HashMap<String, String>,
    per_file_viewport: HashMap<String, Option<Viewport>>,
    files_opened_in_conversation: HashSet<String>,
    terminal_output_buffer: Vec<String>,
    pending_edits_before: HashMap<String, Option<String>>,
    pending_edit_regions: HashMap<String, Option<EditRegion>>,
}

impl<T> ConversationStateManager<T>
where
    T: Tokenizer,
{
    /// Create a new ConversationStateManager with the given tokenizer.
    pub fn new(tokenizer: T, config: ConversationStateManagerConfig) -> Self {
        // Start with conversation overhead + system prompt tokens
        let mut start_tokens = config.conversation_start_tokens;
        if let Some(ref system_prompt) = config.system_prompt {
            let system_tokens = tokenizer.count_tokens(system_prompt);
            start_tokens += system_tokens + config.special_tokens_per_user_message;
        }

        Self {
            tokenizer,
            config,
            messages: Vec::new(),
            current_tokens: start_tokens,
            finalized_conversations: Vec::new(),
            file_states: HashMap::new(),
            per_file_viewport: HashMap::new(),
            files_opened_in_conversation: HashSet::new(),
            terminal_output_buffer: Vec::new(),
            pending_edits_before: HashMap::new(),
            pending_edit_regions: HashMap::new(),
        }
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.messages.clear();
        self.current_tokens = 0;
        self.finalized_conversations.clear();
        self.file_states.clear();
        self.per_file_viewport.clear();
        self.files_opened_in_conversation.clear();
        self.terminal_output_buffer.clear();
        self.pending_edits_before.clear();
        self.pending_edit_regions.clear();
    }

    /// Finalize the current conversation and start a new one.
    /// This is called automatically when conversation token limit is exceeded.
    fn finalize_current_conversation(&mut self) {
        if self.messages.is_empty() {
            return;
        }

        // Check if conversation meets minimum requirements
        let is_long_enough = self.messages.len() >= self.config.min_conversation_messages;
        let has_user = self.messages.iter().any(|m| m.role == Role::User);
        let has_assistant = self.messages.iter().any(|m| m.role == Role::Assistant);

        if is_long_enough && has_user && has_assistant {
            self.finalized_conversations.push(FinalizedConversation {
                messages: std::mem::take(&mut self.messages),
                token_count: self.current_tokens,
            });
        } else {
            self.messages.clear();
        }

        // Reset with conversation start overhead + system prompt for next chunk
        self.current_tokens = self.config.conversation_start_tokens;
        if let Some(ref system_prompt) = self.config.system_prompt {
            let system_tokens = self.tokenizer.count_tokens(system_prompt);
            self.current_tokens += system_tokens + self.config.special_tokens_per_user_message;
        }
        self.files_opened_in_conversation.clear();
    }

    /// Get all finalized conversations with their token counts.
    /// Call this after processing all events.
    pub fn get_conversations(&mut self) -> Vec<FinalizedConversation> {
        // Finalize any remaining conversation
        self.flush_all_pending_edits();
        self.flush_terminal_output_buffer();
        self.finalize_current_conversation();

        std::mem::take(&mut self.finalized_conversations)
    }

    /// Get a copy of all messages.
    pub fn get_messages(&self) -> Vec<ConversationMessage> {
        self.messages.clone()
    }

    /// Get the current content of a file.
    pub fn get_file_content(&self, file_path: &str) -> String {
        self.file_states.get(file_path).cloned().unwrap_or_default()
    }

    /// Append a message, truncating if it exceeds token limits.
    /// If chunking is enabled and conversation limit would be exceeded,
    /// finalizes current conversation and starts a new one.
    fn append_message(&mut self, mut message: ConversationMessage) {
        let mut tokens = self.tokenizer.count_tokens(&message.content);

        if tokens > self.config.max_tokens_per_message {
            message.content = self
                .tokenizer
                .truncate_to_max_tokens(&message.content, self.config.max_tokens_per_message);
            tokens = self.config.max_tokens_per_message;
        }

        // Add special tokens overhead from chat template (role-specific)
        let overhead = match message.role {
            Role::Assistant => self.config.special_tokens_per_assistant_message,
            _ => self.config.special_tokens_per_user_message,
        };
        let tokens_with_overhead = tokens + overhead;

        // Check if we need to start a new conversation (chunking mode)
        if let Some(max_tokens) = self.config.max_tokens_per_conversation {
            if self.current_tokens + tokens_with_overhead > max_tokens && !self.messages.is_empty()
            {
                self.finalize_current_conversation();
                // After starting a new conversation, we need to re-capture file states
                // This will happen naturally as files are accessed
            }
        }

        self.messages.push(message);
        self.current_tokens += tokens_with_overhead;
    }

    /// Capture file contents if not already shown in this conversation.
    fn maybe_capture_file_contents(&mut self, file_path: &str, content: &str) {
        if self.files_opened_in_conversation.contains(file_path) {
            return;
        }
        let cmd = format!("cat -n {}", file_path);
        self.append_message(ConversationMessage::assistant(fenced_block(
            Some("bash"),
            &clean_text(&cmd),
        )));
        let output = line_numbered_output(content, None, None);
        self.append_message(ConversationMessage::user(format!(
            "<stdout>\n{}\n</stdout>",
            output
        )));
        self.files_opened_in_conversation
            .insert(file_path.to_string());
    }

    /// Flush buffered terminal output.
    pub fn flush_terminal_output_buffer(&mut self) {
        if self.terminal_output_buffer.is_empty() {
            return;
        }
        let aggregated: String = self.terminal_output_buffer.join("");
        let out = normalize_terminal_output(&aggregated);
        let mut cleaned = clean_text(&out);

        let tokens = self.tokenizer.count_tokens(&cleaned);
        if tokens > self.config.max_tokens_per_terminal_output {
            let truncated = self
                .tokenizer
                .truncate_to_max_tokens(&cleaned, self.config.max_tokens_per_terminal_output);
            cleaned = format!("{}\n... [truncated]", truncated);
        }

        if !cleaned.trim().is_empty() {
            self.append_message(ConversationMessage::user(format!(
                "<stdout>\n{}\n</stdout>",
                cleaned
            )));
        }
        self.terminal_output_buffer.clear();
    }

    /// Flush pending edits for a specific file.
    pub fn flush_pending_edit_for_file(&mut self, target_file: &str) {
        let before_snapshot = match self.pending_edits_before.get(target_file) {
            Some(Some(s)) => s.clone(),
            _ => return,
        };

        let after_state = self
            .file_states
            .get(target_file)
            .cloned()
            .unwrap_or_default();

        if before_snapshot.trim_end_matches('\n') == after_state.trim_end_matches('\n') {
            self.pending_edits_before
                .insert(target_file.to_string(), None);
            self.pending_edit_regions
                .insert(target_file.to_string(), None);
            return;
        }

        let changed = compute_changed_block_lines(&before_snapshot, &after_state)
            .expect("Failed to compute changed block lines");

        let before_total_lines = before_snapshot.split('\n').count();
        let sed_cmd: String;

        if changed.end_before < changed.start_before {
            // Pure insertion
            let escaped_lines: Vec<String> = changed
                .replacement_lines
                .iter()
                .map(|line| escape_single_quotes_for_sed(line))
                .collect();
            let sed_payload = escaped_lines.join("\\\n");
            if changed.start_before <= before_total_lines.max(1) {
                sed_cmd = format!(
                    "sed -i '{}i\\\n{}' {}",
                    changed.start_before, sed_payload, target_file
                );
            } else {
                sed_cmd = format!("sed -i '$a\\\n{}' {}", sed_payload, target_file);
            }
        } else if changed.replacement_lines.is_empty() {
            // Pure deletion
            sed_cmd = format!(
                "sed -i '{},{}d' {}",
                changed.start_before, changed.end_before, target_file
            );
        } else {
            // Replacement
            let escaped_lines: Vec<String> = changed
                .replacement_lines
                .iter()
                .map(|line| escape_single_quotes_for_sed(line))
                .collect();
            let sed_payload = escaped_lines.join("\\\n");
            sed_cmd = format!(
                "sed -i '{},{}c\\\n{}' {}",
                changed.start_before, changed.end_before, sed_payload, target_file
            );
        }

        let total_lines = after_state.split('\n').count();
        let center = (changed.start_after + changed.end_after) / 2;
        let vp = serialize_compute_viewport(total_lines, center, self.config.viewport_radius);
        self.per_file_viewport
            .insert(target_file.to_string(), Some(vp));

        self.maybe_capture_file_contents(target_file, &before_snapshot);

        let chained_cmd = format!(
            "{} && cat -n {} | sed -n '{},{}p'",
            sed_cmd, target_file, vp.start, vp.end
        );
        self.append_message(ConversationMessage::assistant(fenced_block(
            Some("bash"),
            &clean_text(&chained_cmd),
        )));

        let viewport_output = line_numbered_output(&after_state, Some(vp.start), Some(vp.end));
        self.append_message(ConversationMessage::user(format!(
            "<stdout>\n{}\n</stdout>",
            viewport_output
        )));

        self.pending_edits_before
            .insert(target_file.to_string(), None);
        self.pending_edit_regions
            .insert(target_file.to_string(), None);
    }

    /// Flush all pending edits.
    pub fn flush_all_pending_edits(&mut self) {
        let files: Vec<String> = self.pending_edits_before.keys().cloned().collect();
        for file in files {
            self.flush_pending_edit_for_file(&file);
        }
    }

    /// Handle a tab (file switch) event.
    pub fn handle_tab_event(&mut self, file_path: &str, text_content: Option<&str>) {
        self.flush_all_pending_edits();
        self.flush_terminal_output_buffer();

        if let Some(text) = text_content {
            let content = text.replace("\\n", "\n").replace("\\r", "\r");
            self.file_states
                .insert(file_path.to_string(), content.clone());

            let cmd = format!("cat -n {}", file_path);
            self.append_message(ConversationMessage::assistant(fenced_block(
                Some("bash"),
                &clean_text(&cmd),
            )));
            let output = line_numbered_output(&content, None, None);
            self.append_message(ConversationMessage::user(format!(
                "<stdout>\n{}\n</stdout>",
                output
            )));
            self.files_opened_in_conversation
                .insert(file_path.to_string());
        } else {
            // File switch without content snapshot: show current viewport only
            let content = self.file_states.get(file_path).cloned().unwrap_or_default();
            let total_lines = content.split('\n').count();
            let vp = self
                .per_file_viewport
                .get(file_path)
                .and_then(|v| *v)
                .filter(|v| v.end > 0)
                .unwrap_or_else(|| {
                    let new_vp =
                        serialize_compute_viewport(total_lines, 1, self.config.viewport_radius);
                    self.per_file_viewport
                        .insert(file_path.to_string(), Some(new_vp));
                    new_vp
                });

            if vp.end >= vp.start {
                self.maybe_capture_file_contents(file_path, &content);
                let cmd = format!("cat -n {} | sed -n '{},{}p'", file_path, vp.start, vp.end);
                self.append_message(ConversationMessage::assistant(fenced_block(
                    Some("bash"),
                    &clean_text(&cmd),
                )));
                let viewport_output = line_numbered_output(&content, Some(vp.start), Some(vp.end));
                self.append_message(ConversationMessage::user(format!(
                    "<stdout>\n{}\n</stdout>",
                    viewport_output
                )));
            }
        }
    }

    /// Handle a content change event.
    pub fn handle_content_event(
        &mut self,
        file_path: &str,
        offset: usize,
        length: usize,
        new_text: &str,
    ) {
        self.flush_terminal_output_buffer();

        let before = self.file_states.get(file_path).cloned().unwrap_or_default();
        let new_text_str = new_text;

        // Approximate current edit region in line space
        let safe_offset = floor_char_boundary(&before, offset.min(before.len()));
        let safe_end = floor_char_boundary(&before, (offset + length).min(before.len()));
        let start_line_current = before[..safe_offset].matches('\n').count() + 1;
        let deleted_content = &before[safe_offset..safe_end];
        let lines_added = new_text_str.matches('\n').count();
        let lines_deleted = deleted_content.matches('\n').count();
        let region_start = start_line_current;
        let region_end = start_line_current + lines_added.max(lines_deleted);

        // Flush pending edits if this edit is far from the pending region
        let current_region = self.pending_edit_regions.get(file_path).and_then(|r| *r);
        if let Some(region) = current_region {
            if region_start < region.start.saturating_sub(self.config.coalesce_radius)
                || region_start > region.end + self.config.coalesce_radius
            {
                self.flush_pending_edit_for_file(file_path);
            }
        }

        let after = crate::helpers::apply_change(&before, offset, length, new_text);

        if self
            .pending_edits_before
            .get(file_path)
            .and_then(|v| v.as_ref())
            .is_none()
        {
            self.pending_edits_before
                .insert(file_path.to_string(), Some(before));
        }

        // Update/initialize region union
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

    /// Handle a pre-coalesced edit transition.
    ///
    /// This is used by the shared CSV coalescing layer so all CSV-derived
    /// output formats can reuse the same edit-burst boundaries.
    pub fn handle_coalesced_edit_event(&mut self, file_path: &str, before: &str, after: &str) {
        self.flush_terminal_output_buffer();

        if before.trim_end_matches('\n') == after.trim_end_matches('\n') {
            self.file_states
                .insert(file_path.to_string(), after.to_string());
            return;
        }

        // Ensure the serializer computes the exact sed command from the supplied
        // before/after snapshots.
        self.file_states
            .insert(file_path.to_string(), before.to_string());
        self.pending_edits_before
            .insert(file_path.to_string(), Some(before.to_string()));
        self.file_states
            .insert(file_path.to_string(), after.to_string());
        self.flush_pending_edit_for_file(file_path);
    }

    /// Handle a selection event.
    pub fn handle_selection_event(&mut self, file_path: &str, offset: usize) {
        // During an edit burst (pending edits), suppress viewport emissions
        if self
            .pending_edits_before
            .get(file_path)
            .and_then(|v| v.as_ref())
            .is_some()
        {
            return;
        }

        self.flush_terminal_output_buffer();

        let content = self.file_states.get(file_path).cloned().unwrap_or_default();
        let safe_offset = floor_char_boundary(&content, offset.min(content.len()));
        let target_line = content[..safe_offset].matches('\n').count() + 1;

        self.emit_viewport_for_line(file_path, target_line);
    }

    pub fn handle_cursor_by_line(&mut self, file_path: &str, line: usize) {
        if self
            .pending_edits_before
            .get(file_path)
            .and_then(|v| v.as_ref())
            .is_some()
        {
            return;
        }

        self.flush_terminal_output_buffer();
        self.emit_viewport_for_line(file_path, line);
    }

    fn emit_viewport_for_line(&mut self, file_path: &str, target_line: usize) {
        let content = self.file_states.get(file_path).cloned().unwrap_or_default();
        let total_lines = content.split('\n').count();

        let current_vp = self.per_file_viewport.get(file_path).and_then(|v| *v);
        let mut should_emit = false;

        let vp = if let Some(vp) = current_vp.filter(|v| v.end > 0) {
            if target_line < vp.start || target_line > vp.end {
                let new_vp = serialize_compute_viewport(
                    total_lines,
                    target_line,
                    self.config.viewport_radius,
                );
                self.per_file_viewport
                    .insert(file_path.to_string(), Some(new_vp));
                should_emit = true;
                new_vp
            } else {
                vp
            }
        } else {
            let new_vp =
                serialize_compute_viewport(total_lines, target_line, self.config.viewport_radius);
            self.per_file_viewport
                .insert(file_path.to_string(), Some(new_vp));
            should_emit = true;
            new_vp
        };

        if should_emit && vp.end >= vp.start {
            self.maybe_capture_file_contents(file_path, &content);
            let cmd = format!("cat -n {} | sed -n '{},{}p'", file_path, vp.start, vp.end);
            self.append_message(ConversationMessage::assistant(fenced_block(
                Some("bash"),
                &clean_text(&cmd),
            )));
            let viewport_output = line_numbered_output(&content, Some(vp.start), Some(vp.end));
            self.append_message(ConversationMessage::user(format!(
                "<stdout>\n{}\n</stdout>",
                viewport_output
            )));
        }
    }

    /// Handle a terminal command event.
    pub fn handle_terminal_command_event(&mut self, command: &str) {
        self.flush_all_pending_edits();
        self.flush_terminal_output_buffer();

        let command_str = command.replace("\\n", "\n").replace("\\r", "\r");
        self.append_message(ConversationMessage::assistant(fenced_block(
            Some("bash"),
            &clean_text(&command_str),
        )));
    }

    /// Handle a terminal output event.
    pub fn handle_terminal_output_event(&mut self, output: &str) {
        let raw_output = output.replace("\\n", "\n").replace("\\r", "\r");
        self.terminal_output_buffer.push(raw_output);
    }

    /// Handle a terminal focus event.
    pub fn handle_terminal_focus_event(&mut self) {
        self.flush_all_pending_edits();
        self.flush_terminal_output_buffer();
        // No-op for bash transcript; focus changes don't emit commands/output
    }

    /// Handle a git branch checkout event.
    pub fn handle_git_branch_checkout_event(&mut self, branch_info: &str) {
        self.flush_all_pending_edits();
        self.flush_terminal_output_buffer();

        let branch_str = branch_info.replace("\\n", "\n").replace("\\r", "\r");
        let cleaned = clean_text(&branch_str);

        // Extract branch name from "to 'branch_name'" pattern
        let re = regex::Regex::new(r"to '([^']+)'").unwrap();
        let branch_name = match re.captures(&cleaned) {
            Some(caps) => caps.get(1).map(|m| m.as_str().trim().to_string()),
            None => {
                eprintln!(
                    "[crowd-pilot] Could not extract branch name from git checkout message: {}",
                    cleaned
                );
                return;
            }
        };

        let mut branch_name = match branch_name {
            Some(b) => b,
            None => return,
        };

        // Safe-quote branch if it contains special characters
        let special_chars = regex::Regex::new(r"[^A-Za-z0-9._/\\-]").unwrap();
        if special_chars.is_match(&branch_name) {
            branch_name = format!("'{}'", branch_name.replace('\'', "'\"'\"'"));
        }

        let cmd = format!("git checkout {}", branch_name);
        self.append_message(ConversationMessage::assistant(fenced_block(
            Some("bash"),
            &clean_text(&cmd),
        )));
    }

    /// Set file state directly (for YAML adapter).
    ///
    /// This allows setting file content without providing incremental edit details.
    /// The diff will be computed when the edit is flushed.
    ///
    /// For new files, call `handle_tab_event` first to show the initial content.
    pub fn set_file_state(&mut self, file_path: &str, content: String) {
        self.flush_terminal_output_buffer();

        // If file already exists and content differs, set up pending edit
        if let Some(old_content) = self.file_states.get(file_path) {
            if old_content != &content {
                // Only set pending_edits_before if not already tracking an edit
                if self
                    .pending_edits_before
                    .get(file_path)
                    .and_then(|v| v.as_ref())
                    .is_none()
                {
                    self.pending_edits_before
                        .insert(file_path.to_string(), Some(old_content.clone()));
                }
            }
        }

        self.file_states.insert(file_path.to_string(), content);
    }

    /// Check if a file exists in the current state.
    pub fn has_file(&self, file_path: &str) -> bool {
        self.file_states.contains_key(file_path)
    }

    /// Finalize and get conversation ready for model.
    pub fn finalize_for_model(&mut self) -> Vec<ConversationMessage> {
        self.flush_all_pending_edits();
        self.flush_terminal_output_buffer();
        self.get_messages()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Character-based approximate tokenizer for tests.
    struct CharApproxTokenizer;

    impl Tokenizer for CharApproxTokenizer {
        fn count_tokens(&self, text: &str) -> usize {
            text.len() / 4
        }

        fn truncate_to_max_tokens(&self, text: &str, max_tokens: usize) -> String {
            text.chars().take(max_tokens * 4).collect()
        }
    }

    #[test]
    fn test_basic_tab_event() {
        let mut manager = ConversationStateManager::new(
            CharApproxTokenizer,
            ConversationStateManagerConfig::default(),
        );

        manager.handle_tab_event(
            "/test/file.rs",
            Some("fn main() {\n    println!(\"hello\");\n}"),
        );

        let messages = manager.finalize_for_model();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::Assistant);
        assert!(messages[0].content.contains("cat -n /test/file.rs"));
        assert_eq!(messages[1].role, Role::User);
        assert!(messages[1].content.contains("<stdout>"));
    }

    #[test]
    fn test_content_event() {
        let mut manager = ConversationStateManager::new(
            CharApproxTokenizer,
            ConversationStateManagerConfig::default(),
        );

        manager.handle_tab_event("/test/file.rs", Some("line1\nline2\nline3"));
        manager.handle_content_event("/test/file.rs", 6, 5, "modified");

        let messages = manager.finalize_for_model();
        // Should have: cat (open file), stdout, sed (edit), stdout
        assert!(messages.len() >= 4);
    }

    #[test]
    fn test_terminal_command() {
        let mut manager = ConversationStateManager::new(
            CharApproxTokenizer,
            ConversationStateManagerConfig::default(),
        );

        manager.handle_terminal_command_event("cargo build");
        manager.handle_terminal_output_event("Compiling...\n");
        manager.handle_terminal_output_event("Finished\n");

        let messages = manager.finalize_for_model();
        assert_eq!(messages.len(), 2);
        assert!(messages[0].content.contains("cargo build"));
        assert!(messages[1].content.contains("Compiling"));
    }
}
