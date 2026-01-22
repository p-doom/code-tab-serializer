//! Convert YAML state-based evaluation files directly to test cases JSONL format.
//!
//! This module converts YAML files containing file states at each step directly
//! to the test cases format used for evaluation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::diff::compute_changed_block_lines;
use crate::helpers::{escape_single_quotes_for_sed, line_numbered_output, serialize_compute_viewport};
use crate::VIEWPORT_RADIUS;

// ============================================================================
// YAML Types
// ============================================================================

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

// ============================================================================
// Action Types
// ============================================================================

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

// ============================================================================
// Test Case Types
// ============================================================================

/// A message in the test case context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCaseMessage {
    pub role: String,
    pub content: String,
    pub eval_tag: Option<String>,
    pub assertions: Option<String>,
}

/// A test case for evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub task_id: String,
    pub context: Vec<TestCaseMessage>,
    pub expected_final_response: String,
    pub assertions: Option<String>,
}

// ============================================================================
// Helpers
// ============================================================================

/// Apply heuristic to handle blank lines at insert boundaries more naturally.
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
            let check_idx = start_before.saturating_sub(1);
            if check_idx < old_lines.len() && old_lines[check_idx].trim().is_empty() {
                replacement_lines.remove(0);
                start_after += 1;
                start_before += 1;
            } else {
                break;
            }
        } else {
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

    (start_before, end_before, start_after, end_after, replacement_lines)
}

/// Generate a sed command for a file edit.
fn generate_sed_command(
    filepath: &str,
    old_content: &str,
    new_content: &str,
) -> Option<(String, String)> {
    let old_lines: Vec<&str> = old_content.lines().collect();

    let block = compute_changed_block_lines(old_content, new_content).ok()?;

    let (start_before, end_before, start_after, end_after, replacement_lines) =
        apply_blank_line_heuristic(
            block.start_before,
            block.end_before,
            block.start_after,
            block.end_after,
            block.replacement_lines,
            &old_lines,
        );

    let is_insert = end_before < start_before;
    let is_delete = end_after < start_after;

    let sed_cmd = if is_delete {
        format!("sed -i '{},{}d' {}", start_before, end_before, filepath)
    } else if is_insert {
        let escaped_lines: Vec<String> = replacement_lines
            .iter()
            .map(|l| escape_single_quotes_for_sed(l))
            .collect();
        let content = escaped_lines.join("\\\n");
        format!(
            "sed -i '{}i\\\n{}\n' {}",
            start_before, content, filepath
        )
    } else {
        let escaped_lines: Vec<String> = replacement_lines
            .iter()
            .map(|l| escape_single_quotes_for_sed(l))
            .collect();
        let content = escaped_lines.join("\\\n");
        format!(
            "sed -i '{},{}c\\\n{}\n' {}",
            start_before, end_before, content, filepath
        )
    };

    // Compute viewport
    let center_line = if is_insert {
        start_before + replacement_lines.len() / 2
    } else if is_delete {
        start_before
    } else {
        start_before + replacement_lines.len() / 2
    };

    let total_lines = if is_delete {
        old_lines.len() - (end_before - start_before + 1)
    } else if is_insert {
        old_lines.len() + replacement_lines.len()
    } else {
        old_lines.len() - (end_before - start_before + 1) + replacement_lines.len()
    };

    let viewport = serialize_compute_viewport(center_line, total_lines, VIEWPORT_RADIUS);
    let cat_cmd = format!(
        "cat -n {} | sed -n '{},{}p'",
        filepath, viewport.start, viewport.end
    );

    let full_cmd = format!("{} && {}", sed_cmd, cat_cmd);

    // Generate expected output
    let output = line_numbered_output(new_content, Some(viewport.start), Some(viewport.end));

    Some((full_cmd, output))
}

/// Generate cat command output for a file.
fn generate_cat_output(filepath: &str, content: &str, cursor_line: usize) -> (String, String) {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();

    let viewport = serialize_compute_viewport(cursor_line, total_lines, VIEWPORT_RADIUS);

    let cmd = if viewport.start == 1 && viewport.end >= total_lines {
        format!("cat -n {}", filepath)
    } else {
        format!(
            "cat -n {} | sed -n '{},{}p'",
            filepath, viewport.start, viewport.end
        )
    };

    let output = line_numbered_output(content, Some(viewport.start), Some(viewport.end));

    (cmd, output)
}

/// Derive an action from a state transition.
fn derive_action(prev_state: Option<&State>, curr_state: &State) -> Action {
    let eval_tag = curr_state
        .eval_tag
        .clone()
        .unwrap_or_else(|| "NO_EVAL".to_string());
    let assertions = curr_state.judge_assertions.clone();

    // Check for terminal command
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

    // Check for file changes
    let prev_files = prev_state.and_then(|s| s.files.as_ref());
    let curr_files = curr_state.files.as_ref();

    if let Some(curr) = curr_files {
        for (filepath, new_content) in curr {
            let old_content = prev_files
                .and_then(|f| f.get(filepath))
                .map(|s| s.as_str())
                .unwrap_or("");

            if old_content != new_content {
                // File was edited
                if let Some((cmd, output)) =
                    generate_sed_command(filepath, old_content, new_content)
                {
                    return Action {
                        action_type: ActionType::Edit,
                        command: cmd,
                        output: Some(output),
                        eval_tag,
                        assertions,
                    };
                }
            } else if old_content.is_empty() && !new_content.is_empty() {
                // New file opened
                let cursor_line = curr_state
                    .cursor
                    .as_ref()
                    .map(|c| c.line)
                    .unwrap_or(1);
                let (cmd, output) = generate_cat_output(filepath, new_content, cursor_line);
                return Action {
                    action_type: ActionType::Navigation,
                    command: cmd,
                    output: Some(output),
                    eval_tag,
                    assertions,
                };
            }
        }

        // Check for navigation (cursor file specified, no edit)
        if let Some(cursor) = &curr_state.cursor {
            if let Some(file) = &cursor.file {
                if let Some(content) = curr.get(file) {
                    let (cmd, output) = generate_cat_output(file, content, cursor.line);
                    return Action {
                        action_type: ActionType::Navigation,
                        command: cmd,
                        output: Some(output),
                        eval_tag,
                        assertions,
                    };
                }
            }
        }
    }

    // Empty action
    Action {
        action_type: ActionType::Navigation,
        command: String::new(),
        output: None,
        eval_tag,
        assertions,
    }
}

// ============================================================================
// Conversion Functions
// ============================================================================

/// Format an action's command as a bash code block.
fn format_command(command: &str) -> String {
    format!("```bash\n{}\n```", command)
}

/// Format output as stdout block.
fn format_output(output: &str) -> String {
    if output.is_empty() {
        "<stdout>\n</stdout>".to_string()
    } else {
        format!("<stdout>\n{}\n</stdout>", output)
    }
}

/// Convert a YAML task to test cases.
///
/// Creates one test case for each EVAL-tagged assistant turn, with all preceding
/// turns as context.
pub fn yaml_to_testcases(task: &Task) -> Vec<TestCase> {
    let mut test_cases = Vec::new();
    let mut all_messages: Vec<TestCaseMessage> = Vec::new();
    let mut eval_count = 0;

    let mut prev_state: Option<&State> = None;

    for curr_state in &task.states {
        let action = derive_action(prev_state, curr_state);

        // Skip empty actions
        if action.command.is_empty() {
            prev_state = Some(curr_state);
            continue;
        }

        // Add assistant message
        let assistant_msg = TestCaseMessage {
            role: "assistant".to_string(),
            content: format_command(&action.command),
            eval_tag: Some(action.eval_tag.clone()),
            assertions: action.assertions.clone(),
        };

        // Add user message (output)
        let user_msg = if let Some(output) = &action.output {
            Some(TestCaseMessage {
                role: "user".to_string(),
                content: format_output(output),
                eval_tag: None,
                assertions: None,
            })
        } else {
            None
        };

        // If this is an EVAL turn, create a test case
        if action.eval_tag == "EVAL" {
            let context = all_messages.clone();

            let test_case = TestCase {
                task_id: format!("{}/{}", task.task_id, eval_count),
                context,
                expected_final_response: format_command(&action.command),
                assertions: action.assertions.clone(),
            };
            test_cases.push(test_case);
            eval_count += 1;
        }

        // Add messages to running list for future contexts
        all_messages.push(assistant_msg);
        if let Some(msg) = user_msg {
            all_messages.push(msg);
        }

        prev_state = Some(curr_state);
    }

    test_cases
}

/// Parse a YAML file and convert to test cases.
pub fn convert_yaml_to_testcases(yaml_content: &str) -> Result<Vec<TestCase>, String> {
    let task: Task =
        serde_yaml::from_str(yaml_content).map_err(|e| format!("Failed to parse YAML: {}", e))?;

    Ok(yaml_to_testcases(&task))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_testcase() {
        let yaml = r#"
task_id: test_task
states:
  - files:
      test.py: |
        line1
        line2
    eval: NO_EVAL
    terminal:
      command: "cat -n test.py"
      output: |
             1	line1
             2	line2
  - files:
      test.py: |
        line1
        modified
    eval: EVAL
    judge_assertions: "Check the edit is correct"
"#;

        let test_cases = convert_yaml_to_testcases(yaml).unwrap();
        assert_eq!(test_cases.len(), 1);
        assert_eq!(test_cases[0].task_id, "test_task/0");
        assert!(test_cases[0].context.len() >= 1);
        assert_eq!(
            test_cases[0].assertions,
            Some("Check the edit is correct".to_string())
        );
    }

    #[test]
    fn test_terminal_command() {
        let yaml = r#"
task_id: terminal_test
states:
  - files: {}
    terminal:
      command: "git status"
      output: "On branch main"
    eval: NO_EVAL
  - files: {}
    terminal:
      command: "git commit -m 'test'"
      output: "committed"
    eval: EVAL
"#;

        let test_cases = convert_yaml_to_testcases(yaml).unwrap();
        assert_eq!(test_cases.len(), 1);
        assert!(test_cases[0]
            .expected_final_response
            .contains("git commit"));
    }
}
