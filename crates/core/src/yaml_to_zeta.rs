//! Convert YAML evals to Zeta's diff-based format with editable region markers.

use serde::{Deserialize, Serialize};

use crate::diff::compute_changed_block_lines;
use crate::yaml_types::{find_all_changed_files, has_terminal_command, is_eval_state, parse_yaml_task, State, Task};
use crate::VIEWPORT_RADIUS;

const EDITABLE_START: &str = "<|editable_region_start|>";
const EDITABLE_END: &str = "<|editable_region_end|>";
const CURSOR_MARKER: &str = "<|user_cursor_is_here|>";

#[derive(Debug, Clone, Serialize)]
pub struct ZetaTestCase {
    pub events: String,
    pub input: String,
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assertions: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZetaTestCaseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZetaEvalTestCase {
    pub task_id: String,
    pub context: Vec<ZetaTestCaseMessage>,
    pub expected_final_response: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assertions: Option<String>,
}

fn generate_unified_diff(file_path: &str, old_content: &str, new_content: &str) -> String {
    let old_lines: Vec<&str> = old_content.lines().collect();
    let new_lines: Vec<&str> = new_content.lines().collect();

    let mut first_diff = 0;
    while first_diff < old_lines.len()
        && first_diff < new_lines.len()
        && old_lines[first_diff] == new_lines[first_diff]
    {
        first_diff += 1;
    }

    let mut last_diff_old = old_lines.len();
    let mut last_diff_new = new_lines.len();
    while last_diff_old > first_diff
        && last_diff_new > first_diff
        && old_lines[last_diff_old - 1] == new_lines[last_diff_new - 1]
    {
        last_diff_old -= 1;
        last_diff_new -= 1;
    }

    if first_diff == last_diff_old && first_diff == last_diff_new {
        return String::new();
    }

    let mut diff = format!("User edited file: \"{}\":\n\n```diff\n", file_path);

    let context_start = first_diff.saturating_sub(3);
    let old_start = context_start + 1;
    let new_start = context_start + 1;
    let old_count = last_diff_old - context_start + 3.min(old_lines.len().saturating_sub(last_diff_old));
    let new_count = last_diff_new - context_start + 3.min(new_lines.len().saturating_sub(last_diff_new));

    diff.push_str(&format!("@@ -{},{} +{},{} @@\n", old_start, old_count, new_start, new_count));

    for i in context_start..first_diff {
        if i < old_lines.len() {
            diff.push_str(&format!(" {}\n", old_lines[i]));
        }
    }
    for i in first_diff..last_diff_old {
        diff.push_str(&format!("-{}\n", old_lines[i]));
    }
    for i in first_diff..last_diff_new {
        diff.push_str(&format!("+{}\n", new_lines[i]));
    }
    for i in last_diff_old..(last_diff_old + 3).min(old_lines.len()) {
        diff.push_str(&format!(" {}\n", old_lines[i]));
    }

    diff.push_str("```");
    diff
}

fn insert_editable_markers(
    content: &str,
    edit_start: usize,
    edit_end: usize,
    cursor_line: usize,
    cursor_col: usize,
) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    if total == 0 {
        return format!("{}\n{}\n{}", EDITABLE_START, CURSOR_MARKER, EDITABLE_END);
    }

    let region_start = edit_start.saturating_sub(VIEWPORT_RADIUS).max(1);
    let region_end = (edit_end + VIEWPORT_RADIUS).min(total);

    let mut result = Vec::new();

    for i in 0..region_start.saturating_sub(1) {
        result.push(lines[i].to_string());
    }

    result.push(EDITABLE_START.to_string());

    for i in (region_start - 1)..region_end {
        let line_num = i + 1;
        let line = lines.get(i).unwrap_or(&"");
        if line_num == cursor_line {
            let col = cursor_col.min(line.len());
            let before: String = line.chars().take(col).collect();
            let after: String = line.chars().skip(col).collect();
            result.push(format!("{}{}{}", before, CURSOR_MARKER, after));
        } else {
            result.push(line.to_string());
        }
    }

    result.push(EDITABLE_END.to_string());

    for i in region_end..total {
        result.push(lines[i].to_string());
    }

    result.join("\n")
}

/// Tries all changed files until one produces a valid diff block.
pub fn derive_zeta_from_states(prev: &State, curr: &State) -> Option<ZetaTestCase> {
    let changed_files = find_all_changed_files(prev.files.as_ref(), curr.files.as_ref());
    
    if changed_files.is_empty() && has_terminal_command(curr) {
        return None;
    }
    
    for (file_path, old_content, new_content) in changed_files {
        let block = match compute_changed_block_lines(old_content, new_content) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let events = generate_unified_diff(file_path, old_content, new_content);

        let cursor = curr.cursor.as_ref();
        let cursor_line = cursor.map(|c| c.line).unwrap_or(block.start_after);
        let cursor_col = cursor.map(|c| c.column).unwrap_or(0);

        let input_marked = insert_editable_markers(
            old_content,
            block.start_before,
            block.end_before.max(block.start_before),
            cursor_line,
            cursor_col,
        );
        let output_marked = insert_editable_markers(
            new_content,
            block.start_after,
            block.end_after.max(block.start_after),
            cursor_line,
            cursor_col,
        );

        return Some(ZetaTestCase {
            events,
            input: format!("```{}\n{}\n```", file_path, input_marked),
            output: format!("```{}\n{}\n```", file_path, output_marked),
            assertions: curr.judge_assertions.clone(),
        });
    }

    None
}

pub fn yaml_to_zeta_testcases(task: &Task) -> Vec<ZetaTestCase> {
    let mut test_cases = Vec::new();
    let mut prev_state: Option<&State> = None;

    for curr_state in &task.states {
        if is_eval_state(curr_state) {
            if let Some(prev) = prev_state {
                if let Some(tc) = derive_zeta_from_states(prev, curr_state) {
                    test_cases.push(tc);
                }
            }
        }
        prev_state = Some(curr_state);
    }

    test_cases
}

pub fn format_zeta_markdown(tc: &ZetaTestCase) -> String {
    let mut parts = vec![
        "<events>".to_string(),
        tc.events.clone(),
        "</events>".to_string(),
        "".to_string(),
        "<input>".to_string(),
        tc.input.clone(),
        "</input>".to_string(),
        "".to_string(),
        "<output>".to_string(),
        tc.output.clone(),
        "</output>".to_string(),
    ];

    if let Some(assertions) = &tc.assertions {
        parts.extend(vec![
            "".to_string(),
            "<assertions>".to_string(),
            assertions.clone(),
            "</assertions>".to_string(),
        ]);
    }

    parts.join("\n")
}

pub fn convert_yaml_to_zeta(yaml_content: &str) -> Result<Vec<ZetaTestCase>, String> {
    let task = parse_yaml_task(yaml_content)?;
    Ok(yaml_to_zeta_testcases(&task))
}

fn format_zeta_prompt(tc: &ZetaTestCase) -> String {
    let parts = vec![
        "<events>".to_string(),
        tc.events.clone(),
        "</events>".to_string(),
        "".to_string(),
        "<input>".to_string(),
        tc.input.clone(),
        "</input>".to_string(),
    ];
    parts.join("\n")
}

fn format_zeta_response(tc: &ZetaTestCase) -> String {
    format!("<output>\n{}\n</output>", tc.output)
}

pub fn yaml_to_zeta_eval_testcases(task: &Task) -> Vec<ZetaEvalTestCase> {
    let zeta_cases = yaml_to_zeta_testcases(task);
    let mut eval_cases = Vec::new();

    for (i, tc) in zeta_cases.iter().enumerate() {
        let prompt = format_zeta_prompt(tc);
        let response = format_zeta_response(tc);

        eval_cases.push(ZetaEvalTestCase {
            task_id: format!("{}/{}", task.task_id, i),
            context: vec![ZetaTestCaseMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            expected_final_response: response,
            assertions: tc.assertions.clone(),
        });
    }

    eval_cases
}

pub fn convert_yaml_to_zeta_eval(yaml_content: &str) -> Result<Vec<ZetaEvalTestCase>, String> {
    let task = parse_yaml_task(yaml_content)?;
    Ok(yaml_to_zeta_eval_testcases(&task))
}
