//! Convert YAML state-based evaluation files to markdown+sed format.
//!
//! This module provides the inverse operation of the serializer: given a YAML file
//! containing file states at each step, it derives the sed commands that would
//! produce those state transitions.

use std::collections::HashMap;

use serde::Deserialize;

use crate::diff::compute_changed_block_lines;
use crate::helpers::{
    escape_single_quotes_for_sed, fenced_block, line_numbered_output, serialize_compute_viewport,
};
use crate::VIEWPORT_RADIUS;

/// A cursor position in a file.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Cursor {
    pub file: Option<String>,
    pub line: usize,
    pub column: usize,
}

/// Terminal state (command and output).
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Terminal {
    pub command: Option<String>,
    pub output: Option<String>,
}

/// A single state snapshot in the YAML format.
#[derive(Debug, Clone, Deserialize)]
pub struct State {
    pub files: Option<HashMap<String, String>>,
    pub cursor: Option<Cursor>,
    pub terminal: Option<Terminal>,
    #[serde(rename = "eval")]
    pub eval_tag: Option<String>,
    pub judge_assertions: Option<String>,
}

/// A task containing multiple states.
#[derive(Debug, Clone, Deserialize)]
pub struct Task {
    pub task_id: String,
    pub description: Option<String>,
    pub states: Vec<State>,
}

/// Action types that can be derived from state transitions.
#[derive(Debug, Clone, PartialEq)]
pub enum ActionType {
    Navigation,
    Edit,
    Terminal,
}

/// A derived action with its command and output.
#[derive(Debug, Clone)]
pub struct Action {
    pub action_type: ActionType,
    pub command: String,
    pub output: Option<String>,
    pub eval_tag: String,
    pub assertions: Option<String>,
}

/// Apply heuristic to handle blank lines at insert boundaries more naturally.
///
/// When humans code, they typically "keep old blank, insert after" rather than
/// "insert new blank, push old down". This adjusts the changed block to match
/// human intent.
fn apply_blank_line_heuristic(
    mut start_before: usize,
    end_before: usize,
    mut start_after: usize,
    mut end_after: usize,
    mut replacement_lines: Vec<String>,
    old_lines: &[&str],
) -> (usize, usize, usize, usize, Vec<String>) {
    let is_insert = end_before < start_before;

    // Handle leading blank lines
    while !replacement_lines.is_empty() && replacement_lines[0].trim().is_empty() {
        if is_insert {
            // Insert before line start_before - check if that line is blank
            let check_idx = start_before.saturating_sub(1);
            if check_idx < old_lines.len() && old_lines[check_idx].trim().is_empty() {
                replacement_lines.remove(0);
                start_after += 1;
                start_before += 1;
            } else {
                break;
            }
        } else {
            // For replacement, check line before the replaced region
            let check_idx = start_before.saturating_sub(2);
            if check_idx < old_lines.len() && old_lines[check_idx].trim().is_empty() {
                replacement_lines.remove(0);
                start_after += 1;
            } else {
                break;
            }
        }
    }

    // Handle trailing blank lines
    while !replacement_lines.is_empty() && replacement_lines.last().unwrap().trim().is_empty() {
        let check_idx = if is_insert {
            start_before.saturating_sub(1)
        } else {
            end_before
        };

        if check_idx < old_lines.len() && old_lines[check_idx].trim().is_empty() {
            replacement_lines.pop();
            end_after = end_after.saturating_sub(1);
        } else {
            break;
        }
    }

    (
        start_before,
        end_before,
        start_after,
        end_after,
        replacement_lines,
    )
}

/// Generate a sed command for the given state change.
fn generate_sed_command(
    file_path: &str,
    start_before: usize,
    end_before: usize,
    replacement_lines: &[String],
    before_total_lines: usize,
    start_after: usize,
    end_after: usize,
    after_total_lines: usize,
) -> String {
    // Escape each line for sed
    let escaped_lines: Vec<String> = replacement_lines
        .iter()
        .map(|line| escape_single_quotes_for_sed(line))
        .collect();
    let sed_payload = escaped_lines.join("\\\n");

    let sed_cmd = if end_before < start_before {
        // Pure insertion (no lines deleted)
        if start_before <= before_total_lines.max(1) {
            format!("sed -i '{}i\\\n{}' {}", start_before, sed_payload, file_path)
        } else {
            // Append at end of file
            format!("sed -i '$a\\\n{}' {}", sed_payload, file_path)
        }
    } else if replacement_lines.is_empty() {
        // Pure deletion
        format!(
            "sed -i '{},{}d' {}",
            start_before, end_before, file_path
        )
    } else {
        // Replacement
        format!(
            "sed -i '{},{}c\\\n{}' {}",
            start_before, end_before, sed_payload, file_path
        )
    };

    // Compute viewport centered on the change
    let center = (start_after + end_after) / 2;
    let vp = serialize_compute_viewport(after_total_lines, center, VIEWPORT_RADIUS);

    format!(
        "{} && cat -n {} | sed -n '{},{}p'",
        sed_cmd, file_path, vp.start, vp.end
    )
}

/// Generate cat output in the standard format.
fn generate_cat_output(file_content: &str, start: usize, end: usize) -> String {
    line_numbered_output(file_content, Some(start), Some(end))
}

/// Derive an action from a state transition.
pub fn derive_action(prev_state: Option<&State>, curr_state: &State) -> Action {
    let eval_tag = curr_state
        .eval_tag
        .clone()
        .unwrap_or_else(|| "NO_EVAL".to_string());
    let assertions = curr_state.judge_assertions.clone();

    // Check if it's a terminal command
    if let Some(terminal) = &curr_state.terminal {
        if let Some(cmd) = &terminal.command {
            return Action {
                action_type: ActionType::Terminal,
                command: cmd.clone(),
                output: terminal.output.clone(),
                eval_tag,
                assertions,
            };
        }
    }

    // Get file states
    let prev_files = prev_state
        .and_then(|s| s.files.as_ref())
        .cloned()
        .unwrap_or_default();
    let curr_files = curr_state.files.as_ref().cloned().unwrap_or_default();

    // Check for new files (navigation) and changed files (edit)
    let mut new_files = Vec::new();
    let mut changed_files = Vec::new();

    for path in curr_files.keys() {
        if !prev_files.contains_key(path) {
            new_files.push(path.clone());
        } else if prev_files.get(path) != curr_files.get(path) {
            changed_files.push(path.clone());
        }
    }

    // If there are changed files, handle as edit
    if !changed_files.is_empty() {
        let changed_file = &changed_files[0];
        let prev_content = prev_files.get(changed_file).map(|s| s.as_str()).unwrap_or("");
        let curr_content = curr_files.get(changed_file).map(|s| s.as_str()).unwrap_or("");

        let prev_lines: Vec<&str> = prev_content.lines().collect();
        let curr_lines: Vec<&str> = curr_content.lines().collect();

        if let Ok(changed_block) = compute_changed_block_lines(prev_content, curr_content) {
            // Apply blank line heuristic
            let (start_before, end_before, start_after, end_after, replacement_lines) =
                apply_blank_line_heuristic(
                    changed_block.start_before,
                    changed_block.end_before,
                    changed_block.start_after,
                    changed_block.end_after,
                    changed_block.replacement_lines.clone(),
                    &prev_lines,
                );

            let command = generate_sed_command(
                changed_file,
                start_before,
                end_before,
                &replacement_lines,
                prev_lines.len(),
                start_after,
                end_after,
                curr_lines.len(),
            );

            // Compute viewport for output
            let center = (start_after + end_after) / 2;
            let vp = serialize_compute_viewport(curr_lines.len(), center, VIEWPORT_RADIUS);
            let output = generate_cat_output(curr_content, vp.start, vp.end);

            return Action {
                action_type: ActionType::Edit,
                command,
                output: Some(output),
                eval_tag,
                assertions,
            };
        }
    }

    // If there are new files, handle as navigation (first open)
    if !new_files.is_empty() {
        let new_file = &new_files[0];
        let content = curr_files.get(new_file).map(|s| s.as_str()).unwrap_or("");
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let command = format!("cat -n {}", new_file);
        let output = line_numbered_output(content, Some(1), Some(total_lines));

        return Action {
            action_type: ActionType::Navigation,
            command,
            output: Some(output),
            eval_tag,
            assertions,
        };
    }

    // Cursor movement without file change
    if let Some(cursor) = &curr_state.cursor {
        // Get current file from cursor, or fall back to first file
        let current_file = cursor.file.clone()
            .or_else(|| curr_files.keys().next().cloned())
            .unwrap_or_default();
        let content = curr_files.get(&current_file).map(|s| s.as_str()).unwrap_or("");
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let vp = serialize_compute_viewport(total_lines, cursor.line.max(1), VIEWPORT_RADIUS);
        let command = format!(
            "cat -n {} | sed -n '{},{}p'",
            current_file, vp.start, vp.end
        );
        let output = generate_cat_output(content, vp.start, vp.end);

        return Action {
            action_type: ActionType::Navigation,
            command,
            output: Some(output),
            eval_tag,
            assertions,
        };
    }

    // Fallback: empty action
    Action {
        action_type: ActionType::Navigation,
        command: String::new(),
        output: None,
        eval_tag,
        assertions,
    }
}

/// Format an action as markdown.
fn action_to_markdown(action: &Action) -> String {
    let mut lines = Vec::new();

    // Eval tag
    let eval_str = if action.eval_tag == "EVAL" {
        " <EVAL>"
    } else {
        " <NO_EVAL>"
    };
    lines.push(format!("# Assistant{}", eval_str));
    
    // Command in bash code block
    lines.push("```bash".to_string());
    lines.push(action.command.clone());
    lines.push("```".to_string());

    // Assertions (if present, with blank line before)
    if let Some(assertions) = &action.assertions {
        lines.push(String::new()); // blank line
        lines.push("<assertions>".to_string());
        lines.push(assertions.clone());
        lines.push("</assertions>".to_string());
    }

    // User turn with output (if present, with blank line before)
    if let Some(output) = &action.output {
        lines.push(String::new()); // blank line
        lines.push("# User".to_string());
        lines.push("<stdout>".to_string());
        if !output.is_empty() {
            lines.push(output.clone());
        }
        lines.push("</stdout>".to_string());
    }

    lines.join("\n")
}

/// Convert a YAML task to markdown format.
pub fn yaml_to_markdown(task: &Task) -> String {
    let mut md_parts = Vec::new();

    // Process each state, using the previous state to derive file diffs
    let mut prev_state: Option<&State> = None;
    
    for curr_state in &task.states {
        let action = derive_action(prev_state, curr_state);
        
        // Skip empty actions
        if !action.command.is_empty() {
            md_parts.push(action_to_markdown(&action));
        }
        
        prev_state = Some(curr_state);
    }

    // Join with double newlines and ensure trailing newline
    md_parts.join("\n\n") + "\n"
}

/// Parse a YAML file and convert to markdown.
pub fn convert_yaml_file(yaml_content: &str) -> Result<String, String> {
    let task: Task =
        serde_yaml::from_str(yaml_content).map_err(|e| format!("Failed to parse YAML: {}", e))?;

    Ok(yaml_to_markdown(&task))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_quotes() {
        assert_eq!(
            escape_single_quotes_for_sed("hello 'world'"),
            "hello '\"'\"'world'\"'\"'"
        );
    }

    #[test]
    fn test_simple_replacement() {
        let yaml = r#"
task_id: test
states:
  - files:
      test.py: |
        line1
        line2
        line3
  - files:
      test.py: |
        line1
        modified
        line3
    eval: EVAL
"#;

        let md = convert_yaml_file(yaml).unwrap();
        assert!(md.contains("sed -i '2,2c\\"));
        assert!(md.contains("modified"));
        assert!(md.contains("<EVAL>"));
    }

    #[test]
    fn test_terminal_command() {
        let yaml = r#"
task_id: test
states:
  - files: {}
  - terminal:
      command: "git status"
      output: "On branch main"
    eval: NO_EVAL
"#;

        let md = convert_yaml_file(yaml).unwrap();
        assert!(md.contains("git status"));
        assert!(md.contains("On branch main"));
    }
}
