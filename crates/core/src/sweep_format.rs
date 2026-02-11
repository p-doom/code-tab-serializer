//! Sweep-style format for next-edit prediction.
//!
//! This module follows Sweep's key ideas:
//! - Use Qwen-style `<|file_sep|>` blocks
//! - Represent history edits as `original` / `updated` blocks (not unified diffs)
//! - Rewrite a fixed sliding window around the cursor (default 21 lines)

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::conversation::{ConversationMessage, Role};
use crate::diff::compute_changed_block_lines;
use crate::yaml_adapter::{Cursor, State, Task};

const DEFAULT_VIEWPORT_LINES: usize = 21;

/// Default Sweep-style system prompt.
pub fn sweep_system_prompt() -> String {
    r#"You are a next-edit prediction model.

Predict the user's next code edit by rewriting the updated target window.
Keep edits precise and consistent with recent changes.

Input format:
- `<|file_sep|>{file}` blocks for opened-file context.
- `<|file_sep|>{file}.diff` blocks with `original:` and `updated:` recent edit windows.
- `<|file_sep|>original/{target_file}` target window before the next edit.
- `<|file_sep|>current/{target_file}` current target window shown to the user.

Output format:
- Output exactly one block:
  `<|file_sep|>updated/{target_file}`
  followed by the rewritten target window."#
        .to_string()
}

/// Opened-file context strategy for non-target files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepOpenedFileContextMode {
    /// Include full file content for opened files.
    Full,
    /// Include only a fixed viewport for opened files.
    Viewport,
}

/// Strategy for centering history `original/updated` windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepHistoryCenterMode {
    /// Center history windows on the changed block (current behavior).
    ChangedBlock,
    /// Center history windows on the cursor line at that state (same-file only).
    Cursor,
}

/// Sampling policy when generating Sweep examples from task states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepSamplingMode {
    /// Emit examples only for states tagged with `eval: EVAL`.
    EvalOnly,
    /// Emit examples for every state transition.
    EveryTransition,
}

/// Configuration for Sweep format conversion.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    /// Total lines in fixed viewport windows.
    pub viewport_lines: usize,
    /// How to include opened files in context.
    pub opened_file_context_mode: SweepOpenedFileContextMode,
    /// How to center history windows (`*.diff` blocks).
    pub history_center_mode: SweepHistoryCenterMode,
    /// Hard max token budget per generated conversation.
    /// When set, Sweep trims oldest history first to fit this budget.
    pub max_tokens_per_conversation: Option<usize>,
    /// System prompt used in the system message.
    pub system_prompt: String,
    /// Special tokens added per user/system message by the chat template.
    pub special_tokens_per_user_message: usize,
    /// Special tokens added per assistant message by the chat template.
    pub special_tokens_per_assistant_message: usize,
    /// Fixed special tokens at conversation start.
    pub conversation_start_tokens: usize,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            viewport_lines: DEFAULT_VIEWPORT_LINES,
            opened_file_context_mode: SweepOpenedFileContextMode::Full,
            history_center_mode: SweepHistoryCenterMode::ChangedBlock,
            max_tokens_per_conversation: None,
            system_prompt: sweep_system_prompt(),
            special_tokens_per_user_message: 0,
            special_tokens_per_assistant_message: 0,
            conversation_start_tokens: 0,
        }
    }
}

/// Sweep conversation message.
#[derive(Debug, Clone)]
pub struct SweepConversation {
    pub messages: Vec<ConversationMessage>,
    pub token_count: usize,
    /// Target file path for the example.
    pub target_file: String,
}

#[derive(Debug, Clone)]
struct SweepHistoryEntry {
    file_path: String,
    before_viewport: String,
    after_viewport: String,
}

fn should_sample_state(state: &State, state_idx: usize, mode: SweepSamplingMode) -> bool {
    match mode {
        SweepSamplingMode::EvalOnly => state.eval_tag.as_deref() == Some("EVAL"),
        SweepSamplingMode::EveryTransition => state_idx > 0,
    }
}

fn message_overhead_tokens(role: Role, config: &SweepConfig) -> usize {
    match role {
        Role::Assistant => config.special_tokens_per_assistant_message,
        Role::System | Role::User => config.special_tokens_per_user_message,
    }
}

fn mark_opened(file_path: &str, opened_set: &mut HashSet<String>, opened_order: &mut Vec<String>) {
    if opened_set.insert(file_path.to_string()) {
        opened_order.push(file_path.to_string());
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
        return (0, 0);
    }

    let viewport = viewport_lines.max(1).min(total_lines);
    let center = center_line_one_based.max(1).min(total_lines);

    let above = viewport / 2;
    let below = viewport - above - 1;

    let mut start = center.saturating_sub(above);
    if start == 0 {
        start = 1;
    }
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

fn compute_state_history_entries(
    prev_files: &HashMap<String, String>,
    curr_files: &HashMap<String, String>,
    viewport_lines: usize,
    state_cursor: Option<&Cursor>,
    history_center_mode: SweepHistoryCenterMode,
) -> Vec<SweepHistoryEntry> {
    let mut paths: Vec<&str> = curr_files
        .keys()
        .filter_map(|p| {
            prev_files
                .get(p)
                .filter(|old| *old != curr_files.get(p).expect("path exists in curr"))
                .map(|_| p.as_str())
        })
        .collect();
    paths.sort_unstable();

    let mut entries = Vec::new();
    for path in paths {
        let old_content = prev_files
            .get(path)
            .expect("changed path must exist in prev files");
        let new_content = curr_files
            .get(path)
            .expect("changed path must exist in curr files");

        if let Ok(changed) = compute_changed_block_lines(old_content, new_content) {
            let (before_center, after_center) = match history_center_mode {
                SweepHistoryCenterMode::ChangedBlock => {
                    (changed.start_before.max(1), changed.start_after.max(1))
                }
                SweepHistoryCenterMode::Cursor => {
                    let cursor_line = state_cursor
                        .filter(|cursor| cursor.line > 0 && cursor.file.as_deref() == Some(path))
                        .map(|cursor| cursor.line);
                    match cursor_line {
                        Some(line) => (line, line),
                        None => (changed.start_before.max(1), changed.start_after.max(1)),
                    }
                }
            };

            let before_viewport = viewport_text(old_content, before_center, viewport_lines);
            let after_viewport = viewport_text(new_content, after_center, viewport_lines);
            entries.push(SweepHistoryEntry {
                file_path: path.to_string(),
                before_viewport,
                after_viewport,
            });
        }
    }

    entries
}

fn format_file_sep_block(path: &str, body: &str) -> String {
    format!("<|file_sep|>{}\n{}", path, body)
}

fn build_sweep_conversation<T: crate::Tokenizer>(
    target_file: &str,
    input_content: &str,
    expected_content: &str,
    cursor_line_one_based: usize,
    opened_context_blocks: &[String],
    history: &[SweepHistoryEntry],
    tokenizer: &T,
    config: &SweepConfig,
) -> Option<SweepConversation> {
    let original_window =
        viewport_text(input_content, cursor_line_one_based, config.viewport_lines);
    let current_window = viewport_text(input_content, cursor_line_one_based, config.viewport_lines);
    let updated_window = viewport_text(
        expected_content,
        cursor_line_one_based,
        config.viewport_lines,
    );

    let history_blocks = history
        .iter()
        .map(|entry| {
            format_file_sep_block(
                &format!("{}.diff", entry.file_path),
                &format!(
                    "original:\n{}\nupdated:\n{}",
                    entry.before_viewport, entry.after_viewport
                ),
            )
        })
        .collect::<Vec<_>>();

    let original_block =
        format_file_sep_block(&format!("original/{}", target_file), &original_window);
    let current_block = format_file_sep_block(&format!("current/{}", target_file), &current_window);
    let assistant_content =
        format_file_sep_block(&format!("updated/{}", target_file), &updated_window);

    let build_messages_from_history_start = |history_start: usize| -> Vec<ConversationMessage> {
        let mut blocks = Vec::with_capacity(
            opened_context_blocks.len() + (history_blocks.len() - history_start) + 2,
        );
        blocks.extend(opened_context_blocks.iter().cloned());
        blocks.extend(history_blocks[history_start..].iter().cloned());
        blocks.push(original_block.clone());
        blocks.push(current_block.clone());

        let system_content = format!("{}\n\n{}", config.system_prompt, blocks.join("\n"));
        vec![
            ConversationMessage {
                role: Role::System,
                content: system_content,
            },
            ConversationMessage {
                role: Role::Assistant,
                content: assistant_content.clone(),
            },
        ]
    };

    let compute_tokens = |messages: &[ConversationMessage]| -> usize {
        config.conversation_start_tokens
            + messages
                .iter()
                .map(|m| {
                    tokenizer.count_tokens(&m.content) + message_overhead_tokens(m.role, config)
                })
                .sum::<usize>()
    };

    let (messages, token_count) = if let Some(max_tokens) = config.max_tokens_per_conversation {
        let full_messages = build_messages_from_history_start(0);
        let full_tokens = compute_tokens(&full_messages);
        if full_tokens <= max_tokens {
            (full_messages, full_tokens)
        } else {
            let no_history_start = history_blocks.len();
            let no_history_messages = build_messages_from_history_start(no_history_start);
            let no_history_tokens = compute_tokens(&no_history_messages);
            if no_history_tokens > max_tokens {
                return None;
            }

            // Keep as much (most recent) history as possible while fitting token budget.
            let mut low = 0usize;
            let mut high = no_history_start;
            while low < high {
                let mid = (low + high) / 2;
                let mid_messages = build_messages_from_history_start(mid);
                let mid_tokens = compute_tokens(&mid_messages);
                if mid_tokens <= max_tokens {
                    high = mid;
                } else {
                    low = mid + 1;
                }
            }

            let fitted_messages = build_messages_from_history_start(low);
            let fitted_tokens = compute_tokens(&fitted_messages);
            (fitted_messages, fitted_tokens)
        }
    } else {
        let messages = build_messages_from_history_start(0);
        let token_count = compute_tokens(&messages);
        (messages, token_count)
    };

    Some(SweepConversation {
        messages,
        token_count,
        target_file: target_file.to_string(),
    })
}

/// Process task states to Sweep format conversations.
pub fn process_task_sweep<T: crate::Tokenizer>(
    task: &Task,
    tokenizer: &T,
    config: &SweepConfig,
    sampling_mode: SweepSamplingMode,
) -> Result<Vec<SweepConversation>, String> {
    let mut conversations = Vec::new();
    let mut history: Vec<SweepHistoryEntry> = Vec::new();
    let mut prev_files: HashMap<String, String> = HashMap::new();

    let mut opened_set: HashSet<String> = HashSet::new();
    let mut opened_order: Vec<String> = Vec::new();
    let mut last_cursor_line_by_file: HashMap<String, usize> = HashMap::new();

    for (state_idx, state) in task.states.iter().enumerate() {
        if let Some(cursor) = &state.cursor {
            if let Some(file) = &cursor.file {
                mark_opened(file, &mut opened_set, &mut opened_order);
                if cursor.line > 0 {
                    last_cursor_line_by_file.insert(file.clone(), cursor.line);
                }
            }
        }

        let curr_files_opt = state.files.as_ref();
        let state_history = curr_files_opt
            .map(|curr| {
                compute_state_history_entries(
                    &prev_files,
                    curr,
                    config.viewport_lines,
                    state.cursor.as_ref(),
                    config.history_center_mode,
                )
            })
            .unwrap_or_default();

        for entry in &state_history {
            mark_opened(&entry.file_path, &mut opened_set, &mut opened_order);
        }

        if should_sample_state(state, state_idx, sampling_mode) {
            let curr_files = curr_files_opt.ok_or_else(|| {
                format!(
                    "State {} in task '{}' is missing files required for Sweep sampling",
                    state_idx, task.task_id
                )
            })?;

            let cursor = state.cursor.as_ref().ok_or_else(|| {
                format!(
                    "State {} in task '{}' is missing cursor required for Sweep sampling",
                    state_idx, task.task_id
                )
            })?;

            let cursor_file = cursor.file.as_ref().ok_or_else(|| {
                format!(
                    "State {} in task '{}' has cursor position but missing cursor.file",
                    state_idx, task.task_id
                )
            })?;

            if cursor.line == 0 {
                return Err(format!(
                    "State {} in task '{}' has invalid cursor.line=0; cursor lines must be 1-based",
                    state_idx, task.task_id
                ));
            }

            let expected_content = curr_files.get(cursor_file).ok_or_else(|| {
                format!(
                    "State {} in task '{}' references cursor.file '{}' that is not present in files",
                    state_idx, task.task_id, cursor_file
                )
            })?;

            let input_content = prev_files
                .get(cursor_file)
                .map(|s| s.as_str())
                .unwrap_or(expected_content);

            let mut opened_context_blocks = Vec::new();
            for file_path in &opened_order {
                if file_path == cursor_file {
                    continue;
                }

                let content = curr_files
                    .get(file_path)
                    .or_else(|| prev_files.get(file_path))
                    .map(|s| s.as_str());
                let Some(content) = content else {
                    continue;
                };

                let body = match config.opened_file_context_mode {
                    SweepOpenedFileContextMode::Full => content.to_string(),
                    SweepOpenedFileContextMode::Viewport => {
                        let center = last_cursor_line_by_file
                            .get(file_path)
                            .copied()
                            .unwrap_or(1);
                        viewport_text(content, center, config.viewport_lines)
                    }
                };

                opened_context_blocks.push(format_file_sep_block(file_path, &body));
            }

            let Some(conv) = build_sweep_conversation(
                cursor_file,
                input_content,
                expected_content,
                cursor.line,
                &opened_context_blocks,
                &history,
                tokenizer,
                config,
            ) else {
                // Even with zero history, required context does not fit token budget.
                continue;
            };
            conversations.push(conv);
        }

        history.extend(state_history);

        if let Some(curr_files) = curr_files_opt {
            for (path, content) in curr_files {
                prev_files.insert(path.clone(), content.clone());
            }
        }
    }

    Ok(conversations)
}

/// Process YAML task to Sweep format conversations.
///
/// Emits examples only for `eval: EVAL` states.
pub fn process_yaml_task_sweep<T: crate::Tokenizer>(
    task: &Task,
    tokenizer: &T,
    config: &SweepConfig,
) -> Result<Vec<SweepConversation>, String> {
    process_task_sweep(task, tokenizer, config, SweepSamplingMode::EvalOnly)
}

/// Convert YAML to Sweep format conversations.
pub fn convert_yaml_to_sweep<T: crate::Tokenizer>(
    yaml_content: &str,
    tokenizer: &T,
    config: &SweepConfig,
) -> Result<Vec<SweepConversation>, String> {
    let task = crate::yaml_adapter::parse_yaml_task(yaml_content)?;
    process_yaml_task_sweep(&task, tokenizer, config)
}

/// Convert one CSV session file to Sweep format conversations.
///
/// CSV transitions are sampled on every transition.
pub fn convert_csv_to_sweep_session<T: crate::Tokenizer>(
    csv_path: &Path,
    tokenizer: &T,
    coalesce_radius: usize,
    config: &SweepConfig,
) -> Result<Vec<SweepConversation>, String> {
    let task = crate::csv_adapter::csv_session_to_task(csv_path, coalesce_radius).map_err(|e| {
        format!(
            "Failed to convert CSV session {:?} to task: {}",
            csv_path, e
        )
    })?;
    process_task_sweep(&task, tokenizer, config, SweepSamplingMode::EveryTransition)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CharTokenizer;
    impl crate::Tokenizer for CharTokenizer {
        fn count_tokens(&self, text: &str) -> usize {
            text.len() / 4
        }

        fn truncate_to_max_tokens(&self, text: &str, max: usize) -> String {
            text.chars().take(max * 4).collect()
        }
    }

    #[test]
    fn test_viewport_text_uses_fixed_window_size() {
        let content = (1..=40)
            .map(|i| format!("line{}", i))
            .collect::<Vec<_>>()
            .join("\n");

        let vp = viewport_text(&content, 20, 21);
        assert_eq!(vp.lines().count(), 21);
        assert!(vp.contains("line10"));
        assert!(vp.contains("line30"));
    }

    #[test]
    fn test_process_task_sweep_every_transition() {
        let yaml = r#"
task_id: sweep_transition_sampling
states:
  - step: 0
    eval: NO_EVAL
    files:
      test.py: |
        a = 1
    cursor:
      file: test.py
      line: 1
      column: 0
  - step: 1
    eval: NO_EVAL
    files:
      test.py: |
        a = 2
    cursor:
      file: test.py
      line: 1
      column: 0
  - step: 2
    eval: NO_EVAL
    files:
      test.py: |
        a = 3
    cursor:
      file: test.py
      line: 1
      column: 0
"#;
        let task = crate::yaml_adapter::parse_yaml_task(yaml).unwrap();
        let convs = process_task_sweep(
            &task,
            &CharTokenizer,
            &SweepConfig::default(),
            SweepSamplingMode::EveryTransition,
        )
        .unwrap();
        assert_eq!(convs.len(), 2);
    }

    #[test]
    fn test_process_task_sweep_errors_on_missing_cursor_file() {
        let yaml = r#"
task_id: missing_cursor_file
states:
  - step: 0
    eval: NO_EVAL
    files:
      test.py: |
        a = 1
    cursor:
      file: test.py
      line: 1
      column: 0
  - step: 1
    eval: EVAL
    files:
      test.py: |
        a = 2
    cursor:
      line: 1
      column: 0
"#;
        let task = crate::yaml_adapter::parse_yaml_task(yaml).unwrap();
        let err = process_yaml_task_sweep(&task, &CharTokenizer, &SweepConfig::default())
            .expect_err("expected strict cursor.file validation error");
        assert!(err.contains("missing cursor.file"));
    }

    #[test]
    fn test_history_blocks_use_original_updated_labels() {
        let yaml = r#"
task_id: history_labels
states:
  - step: 0
    eval: NO_EVAL
    files:
      t.py: |
        x = 1
    cursor:
      file: t.py
      line: 1
      column: 0
  - step: 1
    eval: NO_EVAL
    files:
      t.py: |
        x = 2
    cursor:
      file: t.py
      line: 1
      column: 0
  - step: 2
    eval: EVAL
    files:
      t.py: |
        x = 3
    cursor:
      file: t.py
      line: 1
      column: 0
"#;
        let task = crate::yaml_adapter::parse_yaml_task(yaml).unwrap();
        let convs =
            process_yaml_task_sweep(&task, &CharTokenizer, &SweepConfig::default()).unwrap();
        assert_eq!(convs.len(), 1);
        let system = &convs[0].messages[0].content;
        assert!(system.contains(".diff"));
        assert!(system.contains("original:"));
        assert!(system.contains("updated:"));
    }

    #[test]
    fn test_history_blocks_can_center_on_cursor_line() {
        let yaml = r#"
task_id: history_centering
states:
  - step: 0
    eval: NO_EVAL
    files:
      t.py: |
        l1
        l2
        l3
        l4
        l5
        l6
        l7
        l8
    cursor:
      file: t.py
      line: 8
      column: 0
  - step: 1
    eval: NO_EVAL
    files:
      t.py: |
        l1_edit1
        l2
        l3
        l4
        l5
        l6
        l7
        l8
    cursor:
      file: t.py
      line: 7
      column: 0
  - step: 2
    eval: EVAL
    files:
      t.py: |
        l1_edit1
        l2_edit2
        l3
        l4
        l5
        l6
        l7
        l8
    cursor:
      file: t.py
      line: 2
      column: 0
"#;
        let task = crate::yaml_adapter::parse_yaml_task(yaml).unwrap();

        let changed_cfg = SweepConfig {
            viewport_lines: 3,
            history_center_mode: SweepHistoryCenterMode::ChangedBlock,
            ..SweepConfig::default()
        };
        let changed_convs = process_yaml_task_sweep(&task, &CharTokenizer, &changed_cfg).unwrap();
        let changed_system = &changed_convs[0].messages[0].content;
        assert!(changed_system
            .contains("<|file_sep|>t.py.diff\noriginal:\nl1\nl2\nl3\nupdated:\nl1_edit1\nl2\nl3"));

        let cursor_cfg = SweepConfig {
            viewport_lines: 3,
            history_center_mode: SweepHistoryCenterMode::Cursor,
            ..SweepConfig::default()
        };
        let cursor_convs = process_yaml_task_sweep(&task, &CharTokenizer, &cursor_cfg).unwrap();
        let cursor_system = &cursor_convs[0].messages[0].content;
        assert!(cursor_system
            .contains("<|file_sep|>t.py.diff\noriginal:\nl6\nl7\nl8\nupdated:\nl6\nl7\nl8"));
    }

    #[test]
    fn test_opened_file_context_mode_viewport_limits_non_target_context() {
        let yaml = r#"
task_id: opened_context
states:
  - step: 0
    eval: NO_EVAL
    files:
      a.py: |
        a1
        a2
        a3
        a4
      b.py: |
        b1
        b2
        b3
        b4
    cursor:
      file: b.py
      line: 2
      column: 0
  - step: 1
    eval: EVAL
    files:
      a.py: |
        a1
        a2_changed
        a3
        a4
      b.py: |
        b1
        b2
        b3
        b4
    cursor:
      file: a.py
      line: 2
      column: 0
"#;

        let task = crate::yaml_adapter::parse_yaml_task(yaml).unwrap();

        let full_cfg = SweepConfig::default();
        let full_convs = process_yaml_task_sweep(&task, &CharTokenizer, &full_cfg).unwrap();
        let full_system = &full_convs[0].messages[0].content;
        assert!(full_system.contains("b1\nb2\nb3\nb4"));

        let viewport_cfg = SweepConfig {
            viewport_lines: 2,
            opened_file_context_mode: SweepOpenedFileContextMode::Viewport,
            ..SweepConfig::default()
        };
        let viewport_convs = process_yaml_task_sweep(&task, &CharTokenizer, &viewport_cfg).unwrap();
        let viewport_system = &viewport_convs[0].messages[0].content;
        assert!(viewport_system.contains("<|file_sep|>b.py"));
        assert!(!viewport_system.contains("b1\nb2\nb3\nb4"));
    }

    #[test]
    fn test_sweep_token_cap_trims_oldest_history_first() {
        let yaml = r#"
task_id: trim_history_to_fit
states:
  - step: 0
    eval: NO_EVAL
    files:
      main.py: |
        value = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      other.py: |
        persistent_opened_context = "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    cursor:
      file: other.py
      line: 1
      column: 0
  - step: 1
    eval: NO_EVAL
    files:
      main.py: |
        value = "1111111111111111111111111111111111111111"
      other.py: |
        persistent_opened_context = "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    cursor:
      file: main.py
      line: 1
      column: 0
  - step: 2
    eval: NO_EVAL
    files:
      main.py: |
        value = "2222222222222222222222222222222222222222"
      other.py: |
        persistent_opened_context = "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    cursor:
      file: main.py
      line: 1
      column: 0
  - step: 3
    eval: NO_EVAL
    files:
      main.py: |
        value = "3333333333333333333333333333333333333333"
      other.py: |
        persistent_opened_context = "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    cursor:
      file: main.py
      line: 1
      column: 0
  - step: 4
    eval: EVAL
    files:
      main.py: |
        value = "4444444444444444444444444444444444444444"
      other.py: |
        persistent_opened_context = "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    cursor:
      file: main.py
      line: 1
      column: 0
"#;

        let task = crate::yaml_adapter::parse_yaml_task(yaml).unwrap();

        let full_cfg = SweepConfig::default();
        let full_convs = process_yaml_task_sweep(&task, &CharTokenizer, &full_cfg).unwrap();
        assert_eq!(full_convs.len(), 1);
        let full = &full_convs[0];
        let full_diff_count = full.messages[0].content.matches(".diff").count();
        assert!(full_diff_count >= 2);

        let capped_cfg = SweepConfig {
            max_tokens_per_conversation: Some(full.token_count.saturating_sub(1)),
            ..SweepConfig::default()
        };
        let capped_convs = process_yaml_task_sweep(&task, &CharTokenizer, &capped_cfg).unwrap();
        assert_eq!(capped_convs.len(), 1);
        let capped = &capped_convs[0];
        assert!(capped.token_count <= full.token_count.saturating_sub(1));

        let capped_system = &capped.messages[0].content;
        let capped_diff_count = capped_system.matches(".diff").count();
        assert!(capped_diff_count < full_diff_count);
        // Optional history should be trimmed first; opened context should remain present.
        assert!(capped_system.contains("<|file_sep|>other.py"));
    }

    #[test]
    fn test_sweep_token_cap_drops_sample_if_zero_history_does_not_fit() {
        let yaml = r#"
task_id: drop_if_required_context_too_large
states:
  - step: 0
    eval: NO_EVAL
    files:
      main.py: |
        x = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      opened.py: |
        y = "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    cursor:
      file: opened.py
      line: 1
      column: 0
  - step: 1
    eval: EVAL
    files:
      main.py: |
        x = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
      opened.py: |
        y = "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    cursor:
      file: main.py
      line: 1
      column: 0
"#;

        let task = crate::yaml_adapter::parse_yaml_task(yaml).unwrap();

        let capped_cfg = SweepConfig {
            max_tokens_per_conversation: Some(5),
            ..SweepConfig::default()
        };
        let convs = process_yaml_task_sweep(&task, &CharTokenizer, &capped_cfg).unwrap();
        assert_eq!(convs.len(), 0);
    }

    #[test]
    fn test_sweep_token_count_includes_chat_template_overhead() {
        let yaml = r#"
task_id: sweep_template_tokens
states:
  - step: 0
    eval: NO_EVAL
    files:
      test.py: |
        a = 1
    cursor:
      file: test.py
      line: 1
      column: 0
  - step: 1
    eval: EVAL
    files:
      test.py: |
        a = 2
    cursor:
      file: test.py
      line: 1
      column: 0
"#;
        let task = crate::yaml_adapter::parse_yaml_task(yaml).unwrap();

        let base = process_yaml_task_sweep(&task, &CharTokenizer, &SweepConfig::default()).unwrap();

        let mut with_template = SweepConfig::default();
        with_template.special_tokens_per_user_message = 5;
        with_template.special_tokens_per_assistant_message = 9;
        with_template.conversation_start_tokens = 2;
        let with_template = process_yaml_task_sweep(&task, &CharTokenizer, &with_template).unwrap();

        assert_eq!(base.len(), 1);
        assert_eq!(with_template.len(), 1);
        assert_eq!(with_template[0].messages.len(), 2);
        assert_eq!(base[0].messages.len(), 2);
        assert_eq!(with_template[0].token_count, base[0].token_count + 16);
    }
}
