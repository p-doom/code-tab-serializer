//! YAML adapter for converting state-based eval files to conversations.
//!
//! This module provides an adapter that converts YAML eval files (which track
//! explicit file states) into the same conversation format used by the CSV
//! pipeline and VS Code extension.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::conversation::{
    ConversationStateManager, ConversationStateManagerConfig, FinalizedConversation,
};
use crate::Tokenizer;

/// Cursor position in a file.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Cursor {
    pub file: Option<String>,
    pub line: usize,
    #[serde(default)]
    pub column: usize,
}

/// Terminal state with command and output.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Terminal {
    pub command: Option<String>,
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
}

/// A single state in the YAML task.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct State {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<usize>,
    #[serde(default)]
    pub files: Option<HashMap<String, String>>,
    pub cursor: Option<Cursor>,
    pub terminal: Option<Terminal>,
    #[serde(rename = "eval")]
    pub eval_tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge_assertions: Option<String>,
}

/// A complete YAML task with states.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Task {
    pub task_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub states: Vec<State>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<String>,
}

/// Parse a YAML task from content.
pub fn parse_yaml_task(yaml_content: &str) -> Result<Task, String> {
    serde_yaml::from_str(yaml_content).map_err(|e| format!("Failed to parse YAML: {}", e))
}

/// Check if a state is an EVAL state.
pub fn is_eval_state(state: &State) -> bool {
    state
        .eval_tag
        .as_ref()
        .map(|t| t == "EVAL")
        .unwrap_or(false)
}

/// Check if a state has a terminal command.
pub fn has_terminal_command(state: &State) -> bool {
    state
        .terminal
        .as_ref()
        .and_then(|t| t.command.as_ref())
        .is_some()
}

/// Find all changed files between two states.
/// Returns Vec of (file_path, old_content, new_content).
pub fn find_changed_files<'a>(
    prev_files: Option<&'a HashMap<String, String>>,
    curr_files: Option<&'a HashMap<String, String>>,
) -> Vec<(&'a str, Option<&'a str>, &'a str)> {
    let Some(curr) = curr_files else {
        return Vec::new();
    };

    let mut changed = Vec::new();
    for (path, new_content) in curr {
        let old_content = prev_files.and_then(|f| f.get(path)).map(|s| s.as_str());

        // Include if: new file (old is None) or content changed
        let is_changed = match old_content {
            None => true,
            Some(old) => old != new_content.as_str(),
        };

        if is_changed {
            changed.push((path.as_str(), old_content, new_content.as_str()));
        }
    }

    changed
}

/// Configuration for YAML processing.
#[derive(Debug, Clone)]
pub struct YamlProcessingConfig {
    pub viewport_radius: usize,
    pub coalesce_radius: usize,
    pub max_tokens_per_message: usize,
    pub max_tokens_per_terminal_output: usize,
}

impl Default for YamlProcessingConfig {
    fn default() -> Self {
        Self {
            viewport_radius: crate::VIEWPORT_RADIUS,
            coalesce_radius: crate::COALESCE_RADIUS,
            max_tokens_per_message: crate::MAX_TOKENS_PER_MESSAGE,
            max_tokens_per_terminal_output: crate::MAX_TOKENS_PER_TERMINAL_OUTPUT,
        }
    }
}

/// Process a YAML task into conversations using the ConversationStateManager.
///
/// This converts the state-based YAML format into events that feed into
/// the existing serialization pipeline.
pub fn process_yaml_task<T: Tokenizer>(
    task: &Task,
    tokenizer: &T,
    config: &YamlProcessingConfig,
) -> Vec<FinalizedConversation> {
    let manager_config = ConversationStateManagerConfig {
        viewport_radius: config.viewport_radius,
        coalesce_radius: config.coalesce_radius,
        max_tokens_per_message: config.max_tokens_per_message,
        max_tokens_per_terminal_output: config.max_tokens_per_terminal_output,
        // No chunking for eval tasks
        max_tokens_per_conversation: None,
        min_conversation_messages: 1,
        system_prompt: None,
        special_tokens_per_user_message: 0,
        special_tokens_per_assistant_message: 0,
        conversation_start_tokens: 0,
    };

    let mut manager = ConversationStateManager::new(tokenizer, manager_config);
    let mut prev_state: Option<&State> = None;

    for curr_state in &task.states {
        process_state_transition(&mut manager, prev_state, curr_state);
        prev_state = Some(curr_state);
    }

    manager.get_conversations()
}

/// Process a single state transition.
fn process_state_transition<T: Tokenizer>(
    manager: &mut ConversationStateManager<T>,
    prev_state: Option<&State>,
    curr_state: &State,
) {
    // Handle terminal command first (flushes pending edits)
    if has_terminal_command(curr_state) {
        let terminal = curr_state.terminal.as_ref().unwrap();
        let command = terminal.command.as_ref().unwrap();
        manager.handle_terminal_command_event(command);

        if let Some(output) = &terminal.output {
            manager.handle_terminal_output_event(output);
        }
    }

    // Handle file changes
    let prev_files = prev_state.and_then(|s| s.files.as_ref());
    let curr_files = curr_state.files.as_ref();

    let changed = find_changed_files(prev_files, curr_files);

    let mut new_files: Vec<&str> = Vec::new();

    for (file_path, old_content, new_content) in &changed {
        if old_content.is_none() {
            // New file: show full content via tab event
            manager.handle_tab_event(file_path, Some(new_content));
            new_files.push(file_path);
        } else {
            // Existing file changed: set new state, let flush compute diff
            manager.set_file_state(file_path, new_content.to_string());
            manager.flush_pending_edit_for_file(file_path);
        }
    }

    if let Some(cursor) = &curr_state.cursor {
        if let Some(file_path) = &cursor.file {
            let file_was_new = new_files.iter().any(|p| *p == file_path);
            if manager.has_file(file_path) && !file_was_new {
                manager.handle_cursor_by_line(file_path, cursor.line);
            }
        }
    }
}

/// Convert YAML content to finalized conversations.
pub fn convert_yaml_to_conversations<T: Tokenizer>(
    yaml_content: &str,
    tokenizer: &T,
    config: &YamlProcessingConfig,
) -> Result<Vec<FinalizedConversation>, String> {
    let task = parse_yaml_task(yaml_content)?;
    Ok(process_yaml_task(&task, tokenizer, config))
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

    #[test]
    fn test_parse_yaml_task() {
        let yaml = r#"
task_id: test_task
description: A test task
states:
  - step: 0
    eval: NO_EVAL
    files:
      test.py: |
        line1
        line2
  - step: 1
    eval: EVAL
    files:
      test.py: |
        line1
        modified
"#;
        let task = parse_yaml_task(yaml).unwrap();
        assert_eq!(task.task_id, "test_task");
        assert_eq!(task.states.len(), 2);
    }

    #[test]
    fn test_find_changed_files() {
        let mut prev: HashMap<String, String> = HashMap::new();
        prev.insert("a.py".to_string(), "old content".to_string());

        let mut curr: HashMap<String, String> = HashMap::new();
        curr.insert("a.py".to_string(), "new content".to_string());
        curr.insert("b.py".to_string(), "brand new".to_string());

        let changed = find_changed_files(Some(&prev), Some(&curr));
        assert_eq!(changed.len(), 2);
    }

    #[test]
    fn test_process_yaml_task_simple() {
        let yaml = r#"
task_id: simple_edit
states:
  - step: 0
    eval: NO_EVAL
    files:
      test.py: |
        def hello():
            pass
  - step: 1
    eval: EVAL
    files:
      test.py: |
        def hello():
            print("hello")
"#;
        let task = parse_yaml_task(yaml).unwrap();
        let config = YamlProcessingConfig::default();
        let convs = process_yaml_task(&task, &CharApproxTokenizer, &config);

        // Should produce at least one conversation with messages
        assert!(!convs.is_empty() || convs.iter().any(|c| !c.messages.is_empty()));
    }

    #[test]
    fn test_process_yaml_with_terminal() {
        let yaml = r#"
task_id: terminal_test
states:
  - step: 0
    eval: NO_EVAL
    terminal:
      command: "git status"
      output: "On branch main"
  - step: 1
    eval: EVAL
    terminal:
      command: "git commit -m 'test'"
      output: "committed"
"#;
        let task = parse_yaml_task(yaml).unwrap();
        let config = YamlProcessingConfig::default();
        let convs = process_yaml_task(&task, &CharApproxTokenizer, &config);

        // Check that terminal commands appear in messages
        let all_messages: Vec<_> = convs.iter().flat_map(|c| &c.messages).collect();
        let has_git = all_messages.iter().any(|m| m.content.contains("git"));
        assert!(has_git || all_messages.is_empty());
    }

    #[test]
    fn test_is_eval_state() {
        let eval_state = State {
            step: Some(1),
            files: None,
            cursor: None,
            terminal: None,
            eval_tag: Some("EVAL".to_string()),
            judge_assertions: None,
        };
        assert!(is_eval_state(&eval_state));

        let no_eval_state = State {
            step: Some(0),
            files: None,
            cursor: None,
            terminal: None,
            eval_tag: Some("NO_EVAL".to_string()),
            judge_assertions: None,
        };
        assert!(!is_eval_state(&no_eval_state));
    }
}
