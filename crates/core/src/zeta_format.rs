//! Zeta-style format for code completion tasks.
//!
//! This module implements the Zeta format which:
//! - Shows edit history as unified diffs
//! - Presents an editable region around the cursor
//! - Uses token budget for region expansion

use std::collections::HashMap;

use crate::conversation::{ConversationMessage, Role};
use crate::diff::compute_unified_diff;
use crate::yaml_adapter::Task;

// Constants from the extension
const BYTES_PER_TOKEN_GUESS: usize = 3;
const MAX_EDITABLE_TOKENS: usize = 180;
const MAX_CONTEXT_TOKENS: usize = 350;
const DIFF_CONTEXT_LINES: usize = 3;

const EDITABLE_REGION_START: &str = "<|editable_region_start|>";
const EDITABLE_REGION_END: &str = "<|editable_region_end|>";
const USER_CURSOR_MARKER: &str = "<|user_cursor|>";

/// Zeta-style system prompt template.
pub fn zeta_system_prompt() -> String {
    r#"# Instructions

You are an edit prediction assistant in a code editor. Your task is to predict the next edit to a given region of code surrounding the user's cursor.

1. Analyze the edit history to understand what the programmer is trying to achieve
2. Identify any incomplete refactoring or changes that need to be finished
3. Make the remaining edits that a human programmer would logically make next (by rewriting the code around their cursor)

## Focus on

- Completing any partially-applied changes made
- Ensuring consistency with the programming style and patterns already established
- Making edits that maintain or improve code quality

## Rules

- Do not just mechanically apply patterns - reason about what changes make sense given the context and the programmer's apparent goals.
- Do not just fix syntax errors - look for the broader refactoring pattern and apply it systematically throughout the code.
- Keep existing formatting unless it's absolutely necessary
- Don't write a lot of code if you're not sure what to do

# Input Format

You will be provided with:
1. The user's *edit history*, in chronological order. Use this to infer the user's trajectory and predict the next most logical edit.
2. A set of *related excerpts* from the user's codebase. Some of these may be needed for correctly predicting the next edit.
  - `…` may appear within a related file to indicate that some code has been skipped.
3. An excerpt from the user's *current file*.
    - Within the user's current file, there is an *editable region* delimited by the `<|editable_region_start|>` and `<|editable_region_end|>` tags. You can only predict edits in this region.
    - The `<|user_cursor|>` tag marks the user's current cursor position, as it stands after the last edit in the history.

# Output Format

- Briefly explain the user's current intent based on the edit history and their current cursor location.
- Output the entire editable region **including** the `<|editable_region_start|>` and `<|editable_region_end|>` markers.
- Keep the `<|user_cursor|>` marker in the output (it can move if needed).
- If you're unsure some portion of the next edit, you may still predict the surrounding code (such as a function definition, `for` loop, etc) and place the `<|user_cursor|>` within it for the user to fill in.
- Wrap the edited code in a codeblock with exactly five backticks.

## Example

### Input

`````
struct Product {
    name: String,
    price: u32,
}

fn calculate_total(products: &[Product]) -> u32 {
<|editable_region_start|>
    let mut total = 0;
    for product in products {
        total += <|user_cursor|>;
    }
    total
<|editable_region_end|>
}
`````

### Output

The user is computing a sum based on a list of products. The only numeric field on `Product` is `price`, so they must intend to sum the prices.

`````
<|editable_region_start|>
    let mut total = 0;
    for product in products {
        total += product.price;
    }
    total
<|editable_region_end|>
`````"#
        .to_string()
}

/// Line range for editable/context regions.
#[derive(Debug, Clone, Copy)]
pub struct LineRange {
    pub start: usize,
    pub end: usize,
}

/// Guess token count for a line.
fn line_token_guess(line: &str) -> usize {
    let bytes = line.len();
    std::cmp::max(1, bytes / BYTES_PER_TOKEN_GUESS)
}

/// Clamp to the nearest valid UTF-8 char boundary at or before `idx`.
fn floor_char_boundary(text: &str, mut idx: usize) -> usize {
    idx = idx.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Insert cursor marker using a UTF-8-safe index.
fn insert_cursor_marker(line: &str, cursor_column: usize) -> String {
    let char_idx = floor_char_boundary(line, cursor_column);
    format!(
        "{}{}{}",
        &line[..char_idx],
        USER_CURSOR_MARKER,
        &line[char_idx..]
    )
}

/// Expand a line range within token budget.
fn expand_line_range(lines: &[&str], base: LineRange, token_limit: usize) -> LineRange {
    let mut start = base.start.min(lines.len().saturating_sub(1));
    let mut end = base.end.min(lines.len().saturating_sub(1));

    // Count tokens in base range
    let mut remaining = token_limit as i64;
    for i in start..=end {
        if i < lines.len() {
            remaining -= line_token_guess(lines[i]) as i64;
        }
    }
    remaining = remaining.max(0);

    // Expand outward
    while remaining > 0 {
        let mut expanded = false;

        if start > 0 && remaining > 0 {
            start -= 1;
            remaining -= line_token_guess(lines[start]) as i64;
            expanded = true;
        }

        if end < lines.len().saturating_sub(1) && remaining > 0 {
            end += 1;
            remaining -= line_token_guess(lines[end]) as i64;
            expanded = true;
        }

        if !expanded {
            break;
        }
    }

    LineRange { start, end }
}

/// Compute editable and context ranges from cursor position.
pub fn compute_editable_and_context_ranges(
    lines: &[&str],
    cursor_line: usize,
) -> (LineRange, LineRange) {
    compute_editable_and_context_ranges_with_limits(
        lines,
        cursor_line,
        MAX_EDITABLE_TOKENS,
        MAX_CONTEXT_TOKENS,
    )
}

/// Compute editable and context ranges with caller-provided token budgets.
fn compute_editable_and_context_ranges_with_limits(
    lines: &[&str],
    cursor_line: usize,
    max_editable_tokens: usize,
    max_context_tokens: usize,
) -> (LineRange, LineRange) {
    if lines.is_empty() {
        let empty = LineRange { start: 0, end: 0 };
        return (empty, empty);
    }

    let clamped = cursor_line.min(lines.len().saturating_sub(1));
    let cursor_range = LineRange {
        start: clamped,
        end: clamped,
    };

    let editable = expand_line_range(lines, cursor_range, max_editable_tokens);
    let context = expand_line_range(lines, editable, max_context_tokens);

    (editable, context)
}

/// Format file excerpt with editable region markers and cursor.
pub fn format_cursor_excerpt(
    file_path: &str,
    content: &str,
    editable: LineRange,
    context: LineRange,
    cursor_line: usize,
    cursor_column: usize,
) -> String {
    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        return format!(
            "`````{}\n{}\n{}\n{}\n`````",
            file_path, EDITABLE_REGION_START, USER_CURSOR_MARKER, EDITABLE_REGION_END
        );
    }

    let context_start = context.start.min(lines.len().saturating_sub(1));
    let context_end = context.end.min(lines.len().saturating_sub(1));

    let mut excerpt_lines: Vec<String> = lines[context_start..=context_end]
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Insert cursor marker
    let cursor_index = cursor_line
        .saturating_sub(context_start)
        .min(excerpt_lines.len().saturating_sub(1));
    let line = &excerpt_lines[cursor_index];
    excerpt_lines[cursor_index] = insert_cursor_marker(line, cursor_column);

    // Insert editable region markers
    let editable_start_idx = editable.start.saturating_sub(context_start);
    let editable_end_idx = editable.end.saturating_sub(context_start);

    // Insert end marker first (so indices don't shift)
    let end_insert = (editable_end_idx + 1).min(excerpt_lines.len());
    excerpt_lines.insert(end_insert, EDITABLE_REGION_END.to_string());

    let start_insert = editable_start_idx.min(excerpt_lines.len());
    excerpt_lines.insert(start_insert, EDITABLE_REGION_START.to_string());

    format!("`````{}\n{}\n`````", file_path, excerpt_lines.join("\n"))
}

/// Build unified diff between two file contents.
pub fn build_unified_diff(
    old_path: &str,
    new_path: &str,
    old_content: &str,
    new_content: &str,
    context_lines: usize,
) -> String {
    let diff = compute_unified_diff(old_content, new_content, context_lines);
    if diff.is_empty() {
        return String::new();
    }
    format!("--- a/{}\n+++ b/{}\n{}", old_path, new_path, diff)
}

/// Edit history entry.
#[derive(Debug, Clone)]
pub struct EditHistoryEntry {
    pub old_path: String,
    pub new_path: String,
    pub diff: String,
}

/// Format edit history for Zeta prompt.
pub fn format_edit_history(entries: &[EditHistoryEntry]) -> String {
    if entries.is_empty() {
        return "(No edit history)".to_string();
    }

    entries
        .iter()
        .map(|e| format!("--- a/{}\n+++ b/{}\n{}", e.old_path, e.new_path, e.diff))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Configuration for Zeta format conversion.
#[derive(Debug, Clone)]
pub struct ZetaConfig {
    pub max_editable_tokens: usize,
    pub max_context_tokens: usize,
    pub diff_context_lines: usize,
}

impl Default for ZetaConfig {
    fn default() -> Self {
        Self {
            max_editable_tokens: MAX_EDITABLE_TOKENS,
            max_context_tokens: MAX_CONTEXT_TOKENS,
            diff_context_lines: DIFF_CONTEXT_LINES,
        }
    }
}

/// Zeta conversation message.
#[derive(Debug, Clone)]
pub struct ZetaConversation {
    pub messages: Vec<ConversationMessage>,
    pub token_count: usize,
    pub editable_range: LineRange,
}

/// Process YAML task to Zeta format conversations.
///
/// For each EVAL state, generates a conversation with:
/// - Edit history as diffs leading up to that point
/// - Current file with editable region around cursor
/// - Expected output (edited editable region)
pub fn process_yaml_task_zeta<T: crate::Tokenizer>(
    task: &Task,
    tokenizer: &T,
    config: &ZetaConfig,
) -> Vec<ZetaConversation> {
    let mut conversations = Vec::new();
    let mut edit_history: Vec<EditHistoryEntry> = Vec::new();
    let mut prev_files: HashMap<String, String> = HashMap::new();

    for state in &task.states {
        let edit_history_len_before_state = edit_history.len();

        // Track file changes as edit history
        if let Some(curr_files) = &state.files {
            for (path, new_content) in curr_files {
                if let Some(old_content) = prev_files.get(path) {
                    if old_content != new_content {
                        let diff = compute_unified_diff(
                            old_content,
                            new_content,
                            config.diff_context_lines,
                        );
                        if !diff.is_empty() {
                            edit_history.push(EditHistoryEntry {
                                old_path: path.clone(),
                                new_path: path.clone(),
                                diff,
                            });
                        }
                    }
                }
            }

            // Check if this is an EVAL state
            if state
                .eval_tag
                .as_ref()
                .map(|t| t == "EVAL")
                .unwrap_or(false)
            {
                // Find the current file and cursor
                let cursor = state.cursor.as_ref();
                let cursor_file = cursor.and_then(|c| c.file.as_ref());

                // Find the file to edit
                let edit_file = cursor_file
                    .map(|s| s.as_str())
                    .or_else(|| curr_files.keys().next().map(|s| s.as_str()));

                if let Some(file_path) = edit_file {
                    if let Some(content) = curr_files.get(file_path) {
                        // Get previous content for expected output
                        let prev_content = prev_files.get(file_path);

                        let cursor_line = cursor.map(|c| c.line).unwrap_or(0);
                        let cursor_col = cursor.map(|c| c.column).unwrap_or(0);

                        // Use prev_content for input, curr_content for expected output
                        let input_content = prev_content.map(|s| s.as_str()).unwrap_or(content);

                        let conv = build_zeta_conversation(
                            file_path,
                            input_content,
                            content,
                            cursor_line,
                            cursor_col,
                            &edit_history[..edit_history_len_before_state],
                            tokenizer,
                            config,
                        );

                        conversations.push(conv);
                    }
                }
            }

            // Update prev_files
            for (path, content) in curr_files {
                prev_files.insert(path.clone(), content.clone());
            }
        }
    }

    conversations
}

/// Build a Zeta conversation for a single EVAL point.
fn build_zeta_conversation<T: crate::Tokenizer>(
    file_path: &str,
    input_content: &str,
    expected_content: &str,
    cursor_line: usize,
    cursor_col: usize,
    edit_history: &[EditHistoryEntry],
    tokenizer: &T,
    config: &ZetaConfig,
) -> ZetaConversation {
    let lines: Vec<&str> = input_content.lines().collect();
    let (editable, context) = compute_editable_and_context_ranges_with_limits(
        &lines,
        cursor_line,
        config.max_editable_tokens,
        config.max_context_tokens,
    );

    // Build system prompt with context
    let edit_history_text = format_edit_history(edit_history);
    let cursor_excerpt = format_cursor_excerpt(
        file_path,
        input_content,
        editable,
        context,
        cursor_line,
        cursor_col,
    );

    let system_content = format!(
        "{}\n\n# 1. User Edits History\n\n`````\n{}\n`````\n\n# 2. Related excerpts\n\n(No context)\n\n# 3. Current File\n\n{}",
        zeta_system_prompt(),
        edit_history_text,
        cursor_excerpt
    );

    // Build expected output with editable region markers and cursor marker
    let expected_lines: Vec<&str> = expected_content.lines().collect();
    let expected_editable = if expected_lines.is_empty() {
        // Empty content: just include cursor marker between region markers
        format!(
            "{}\n{}\n{}",
            EDITABLE_REGION_START, USER_CURSOR_MARKER, EDITABLE_REGION_END
        )
    } else {
        let editable_start = editable.start.min(expected_lines.len().saturating_sub(1));
        let editable_end = editable.end.min(expected_lines.len().saturating_sub(1));
        let mut editable_region_lines: Vec<String> = expected_lines[editable_start..=editable_end]
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Insert cursor marker at the appropriate position within the editable region
        if !editable_region_lines.is_empty() {
            let cursor_index = cursor_line
                .saturating_sub(editable_start)
                .min(editable_region_lines.len().saturating_sub(1));
            let line = &editable_region_lines[cursor_index];
            editable_region_lines[cursor_index] = insert_cursor_marker(line, cursor_col);
        }

        format!(
            "{}\n{}\n{}",
            EDITABLE_REGION_START,
            editable_region_lines.join("\n"),
            EDITABLE_REGION_END
        )
    };

    let expected_output = format!(
        "Based on the edit history and cursor position, here's the predicted edit:\n\n`````\n{}\n`````",
        expected_editable
    );

    let messages = vec![
        ConversationMessage {
            role: Role::System,
            content: system_content.clone(),
        },
        ConversationMessage {
            role: Role::Assistant,
            content: expected_output.clone(),
        },
    ];

    let token_count =
        tokenizer.count_tokens(&system_content) + tokenizer.count_tokens(&expected_output);

    ZetaConversation {
        messages,
        token_count,
        editable_range: editable,
    }
}

/// Convert YAML to Zeta format conversations.
pub fn convert_yaml_to_zeta<T: crate::Tokenizer>(
    yaml_content: &str,
    tokenizer: &T,
    config: &ZetaConfig,
) -> Result<Vec<ZetaConversation>, String> {
    let task = crate::yaml_adapter::parse_yaml_task(yaml_content)?;
    Ok(process_yaml_task_zeta(&task, tokenizer, config))
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
    fn test_compute_editable_range() {
        let lines: Vec<&str> = (0..50).map(|_| "some code here").collect();
        let (editable, context) = compute_editable_and_context_ranges(&lines, 25);

        assert!(editable.start <= 25);
        assert!(editable.end >= 25);
        assert!(context.start <= editable.start);
        assert!(context.end >= editable.end);
    }

    #[test]
    fn test_format_cursor_excerpt() {
        let content = "line1\nline2\nline3\nline4\nline5";
        let editable = LineRange { start: 1, end: 3 };
        let context = LineRange { start: 0, end: 4 };

        let result = format_cursor_excerpt("test.py", content, editable, context, 2, 0);

        assert!(result.contains(EDITABLE_REGION_START));
        assert!(result.contains(EDITABLE_REGION_END));
        assert!(result.contains(USER_CURSOR_MARKER));
    }

    #[test]
    fn test_format_cursor_excerpt_empty_content() {
        let result = format_cursor_excerpt(
            "empty.py",
            "",
            LineRange { start: 0, end: 0 },
            LineRange { start: 0, end: 0 },
            0,
            0,
        );

        assert!(result.contains(EDITABLE_REGION_START));
        assert!(result.contains(EDITABLE_REGION_END));
        assert!(result.contains(USER_CURSOR_MARKER));
    }

    #[test]
    fn test_format_cursor_excerpt_utf8_cursor_column() {
        let content = "héllo\nline2";
        let editable = LineRange { start: 0, end: 1 };
        let context = LineRange { start: 0, end: 1 };

        let result = format_cursor_excerpt("utf8.py", content, editable, context, 0, 2);
        assert!(result.contains("h<|user_cursor|>éllo"));
    }

    #[test]
    fn test_compute_editable_range_respects_config_limits() {
        let lines: Vec<&str> = (0..20).map(|_| "abcdefghij").collect();
        let (default_editable, _) = compute_editable_and_context_ranges(&lines, 10);
        let (small_editable, _) = compute_editable_and_context_ranges_with_limits(&lines, 10, 1, 1);

        let default_width = default_editable.end.saturating_sub(default_editable.start);
        let small_width = small_editable.end.saturating_sub(small_editable.start);
        assert!(small_width <= default_width);
    }

    #[test]
    fn test_zeta_conversion() {
        let yaml = r#"
task_id: test
states:
  - step: 0
    eval: NO_EVAL
    files:
      test.py: |
        def hello():
            pass
    cursor:
      file: test.py
      line: 1
      column: 0
  - step: 1
    eval: EVAL
    files:
      test.py: |
        def hello():
            print("hello")
    cursor:
      file: test.py
      line: 1
      column: 0
"#;
        let config = ZetaConfig::default();
        let convs = convert_yaml_to_zeta(yaml, &CharTokenizer, &config).unwrap();

        assert!(!convs.is_empty());
        assert!(!convs[0].messages.is_empty());
    }
}
