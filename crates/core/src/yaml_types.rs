//! Common YAML types for eval conversion modules.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Cursor {
    pub file: Option<String>,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Terminal {
    pub command: Option<String>,
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct State {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<usize>,
    pub files: Option<HashMap<String, String>>,
    pub cursor: Option<Cursor>,
    pub terminal: Option<Terminal>,
    #[serde(rename = "eval")]
    pub eval_tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge_assertions: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Task {
    pub task_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub states: Vec<State>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<String>,
}

/// Returns (file_path, old_content, new_content) for first changed file.
pub fn find_changed_file<'a>(
    prev_files: Option<&'a HashMap<String, String>>,
    curr_files: Option<&'a HashMap<String, String>>,
) -> Option<(&'a str, &'a str, &'a str)> {
    find_all_changed_files(prev_files, curr_files).into_iter().next()
}

/// Returns (file_path, old_content, new_content) for each changed file.
/// Only includes modified or new files - deletions are not supported by SED/Zeta formats.
pub fn find_all_changed_files<'a>(
    prev_files: Option<&'a HashMap<String, String>>,
    curr_files: Option<&'a HashMap<String, String>>,
) -> Vec<(&'a str, &'a str, &'a str)> {
    let Some(curr) = curr_files else {
        return Vec::new();
    };
    
    let mut changed = Vec::new();
    for (path, new_content) in curr {
        let old_content = prev_files
            .and_then(|f| f.get(path))
            .map(|s| s.as_str())
            .unwrap_or("");
        
        if old_content != new_content.as_str() {
            changed.push((path.as_str(), old_content, new_content.as_str()));
        }
    }
    
    changed
}

pub fn parse_yaml_task(yaml_content: &str) -> Result<Task, String> {
    serde_yaml::from_str(yaml_content)
        .map_err(|e| format!("Failed to parse YAML: {}", e))
}

pub fn is_eval_state(state: &State) -> bool {
    state.eval_tag.as_ref().map(|t| t == "EVAL").unwrap_or(false)
}

pub fn has_terminal_command(state: &State) -> bool {
    state.terminal.as_ref().and_then(|t| t.command.as_ref()).is_some()
}
