//! CSV adapter and shared coalescing logic.
//!
//! This module parses VS Code-style CSV action logs into normalized events,
//! coalesces nearby edit bursts, and can build state-based `Task` snapshots.

use std::collections::{HashMap, VecDeque};
use std::path::Path;

use serde::Deserialize;

use crate::diff::compute_changed_block_lines;
use crate::helpers::{apply_change, floor_char_boundary};
use crate::yaml_adapter::{Cursor, State, Task, Terminal};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct CsvRow {
    #[serde(rename = "Sequence")]
    _sequence: Option<i64>,
    #[serde(rename = "Time")]
    _time: Option<String>,
    file: String,
    range_offset: Option<i64>,
    range_length: Option<i64>,
    text: Option<String>,
    #[serde(rename = "Language")]
    _language: Option<String>,
    #[serde(rename = "Type")]
    event_type: String,
}

/// Raw event decoded from a CSV row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsvRawEvent {
    Tab {
        file_path: String,
        text_content: Option<String>,
    },
    Content {
        file_path: String,
        offset: usize,
        length: usize,
        new_text: String,
    },
    Selection {
        file_path: String,
        offset: usize,
    },
    TerminalCommand {
        command: String,
    },
    TerminalOutput {
        output: String,
    },
    TerminalFocus,
    GitBranchCheckout {
        branch_info: String,
    },
}

/// Coalesced event stream shared by CSV-derived output formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoalescedCsvEvent {
    Tab {
        file_path: String,
        text_content: Option<String>,
    },
    Edit {
        file_path: String,
        before: String,
        after: String,
        cursor_line: usize,
        cursor_column: usize,
    },
    Selection {
        file_path: String,
        offset: usize,
    },
    TerminalCommand {
        command: String,
    },
    TerminalOutput {
        output: String,
    },
    TerminalFocus,
    GitBranchCheckout {
        branch_info: String,
    },
}

#[derive(Debug, Clone, Copy)]
struct EditRegion {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
struct PendingEdit {
    before: String,
    region: EditRegion,
}

fn decode_escaped_text(text: &str) -> String {
    text.replace("\\n", "\n").replace("\\r", "\r")
}

fn trim_trailing_newlines(s: &str) -> &str {
    s.trim_end_matches('\n')
}

/// Parse one CSV session file into raw events.
pub fn parse_csv_session(csv_path: &Path) -> Result<Vec<CsvRawEvent>, Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(csv_path)?;
    let mut out = Vec::new();

    for result in reader.deserialize() {
        let row: CsvRow = result?;

        match row.event_type.as_str() {
            "tab" => {
                out.push(CsvRawEvent::Tab {
                    file_path: row.file,
                    text_content: row.text.map(|s| decode_escaped_text(&s)),
                });
            }
            "content" => {
                let offset = row
                    .range_offset
                    .ok_or_else(|| format!("content event missing RangeOffset in {:?}", csv_path))?
                    as usize;
                let length = row
                    .range_length
                    .ok_or_else(|| format!("content event missing RangeLength in {:?}", csv_path))?
                    as usize;
                let new_text = decode_escaped_text(row.text.as_deref().unwrap_or(""));

                out.push(CsvRawEvent::Content {
                    file_path: row.file,
                    offset,
                    length,
                    new_text,
                });
            }
            "selection_command" | "selection_mouse" | "selection_keyboard" => {
                let offset = row.range_offset.ok_or_else(|| {
                    format!("selection event missing RangeOffset in {:?}", csv_path)
                })? as usize;
                out.push(CsvRawEvent::Selection {
                    file_path: row.file,
                    offset,
                });
            }
            "terminal_command" => {
                out.push(CsvRawEvent::TerminalCommand {
                    command: decode_escaped_text(row.text.as_deref().unwrap_or("")),
                });
            }
            "terminal_output" => {
                out.push(CsvRawEvent::TerminalOutput {
                    output: decode_escaped_text(row.text.as_deref().unwrap_or("")),
                });
            }
            "terminal_focus" => {
                out.push(CsvRawEvent::TerminalFocus);
            }
            "git_branch_checkout" => {
                out.push(CsvRawEvent::GitBranchCheckout {
                    branch_info: decode_escaped_text(row.text.as_deref().unwrap_or("")),
                });
            }
            other => {
                eprintln!("Warning: Unknown event type '{}' in {:?}", other, csv_path);
            }
        }
    }

    Ok(out)
}

struct CsvCoalescer {
    coalesce_radius: usize,
    file_states: HashMap<String, String>,
    pending_edits: HashMap<String, PendingEdit>,
    pending_order: VecDeque<String>,
    events: Vec<CoalescedCsvEvent>,
}

impl CsvCoalescer {
    fn new(coalesce_radius: usize) -> Self {
        Self {
            coalesce_radius,
            file_states: HashMap::new(),
            pending_edits: HashMap::new(),
            pending_order: VecDeque::new(),
            events: Vec::new(),
        }
    }

    fn push_pending_order_if_absent(&mut self, file_path: &str) {
        if !self.pending_order.iter().any(|p| p == file_path) {
            self.pending_order.push_back(file_path.to_string());
        }
    }

    fn remove_pending_order(&mut self, file_path: &str) {
        if let Some(idx) = self.pending_order.iter().position(|p| p == file_path) {
            self.pending_order.remove(idx);
        }
    }

    fn flush_pending_edit_for_file(&mut self, file_path: &str) {
        let Some(pending) = self.pending_edits.remove(file_path) else {
            return;
        };
        self.remove_pending_order(file_path);

        let after = self.file_states.get(file_path).cloned().unwrap_or_default();
        if trim_trailing_newlines(&pending.before) == trim_trailing_newlines(&after) {
            return;
        }

        let changed = compute_changed_block_lines(&pending.before, &after)
            .expect("Failed to compute changed block lines");

        self.events.push(CoalescedCsvEvent::Edit {
            file_path: file_path.to_string(),
            before: pending.before,
            after,
            cursor_line: changed.start_after.max(1),
            cursor_column: 0,
        });
    }

    fn flush_all_pending_edits(&mut self) {
        let files: Vec<String> = self.pending_order.iter().cloned().collect();
        for file in files {
            self.flush_pending_edit_for_file(&file);
        }
    }

    fn handle_content_event(
        &mut self,
        file_path: &str,
        offset: usize,
        length: usize,
        new_text: &str,
    ) {
        let before = self.file_states.get(file_path).cloned().unwrap_or_default();

        let safe_offset = floor_char_boundary(&before, offset.min(before.len()));
        let safe_end = floor_char_boundary(&before, (offset + length).min(before.len()));

        let start_line_current = before[..safe_offset].matches('\n').count() + 1;
        let deleted_content = &before[safe_offset..safe_end];
        let lines_added = new_text.matches('\n').count();
        let lines_deleted = deleted_content.matches('\n').count();
        let region_start = start_line_current;
        let region_end = start_line_current + lines_added.max(lines_deleted);

        if let Some(pending) = self.pending_edits.get(file_path) {
            if region_start < pending.region.start.saturating_sub(self.coalesce_radius)
                || region_start > pending.region.end + self.coalesce_radius
            {
                self.flush_pending_edit_for_file(file_path);
            }
        }

        let after = apply_change(&before, offset, length, new_text);

        if !self.pending_edits.contains_key(file_path) {
            self.push_pending_order_if_absent(file_path);
            self.pending_edits.insert(
                file_path.to_string(),
                PendingEdit {
                    before: before.clone(),
                    region: EditRegion {
                        start: region_start,
                        end: region_start.max(region_end),
                    },
                },
            );
        }

        let pending = self
            .pending_edits
            .get_mut(file_path)
            .expect("Pending edit must exist after insertion");

        pending.region = EditRegion {
            start: pending.region.start.min(region_start),
            end: pending.region.end.max(region_end),
        };

        self.file_states.insert(file_path.to_string(), after);
    }

    fn handle_event(&mut self, event: CsvRawEvent) {
        match event {
            CsvRawEvent::Tab {
                file_path,
                text_content,
            } => {
                self.flush_all_pending_edits();
                if let Some(content) = &text_content {
                    self.file_states.insert(file_path.clone(), content.clone());
                }
                self.events.push(CoalescedCsvEvent::Tab {
                    file_path,
                    text_content,
                });
            }
            CsvRawEvent::Content {
                file_path,
                offset,
                length,
                new_text,
            } => {
                self.handle_content_event(&file_path, offset, length, &new_text);
            }
            CsvRawEvent::Selection { file_path, offset } => {
                if self.pending_edits.contains_key(&file_path) {
                    // Match ConversationStateManager behavior: suppress viewport emissions
                    // for the same file while an edit burst is still pending.
                    return;
                }
                self.events
                    .push(CoalescedCsvEvent::Selection { file_path, offset });
            }
            CsvRawEvent::TerminalCommand { command } => {
                self.flush_all_pending_edits();
                self.events
                    .push(CoalescedCsvEvent::TerminalCommand { command });
            }
            CsvRawEvent::TerminalOutput { output } => {
                self.events
                    .push(CoalescedCsvEvent::TerminalOutput { output });
            }
            CsvRawEvent::TerminalFocus => {
                self.flush_all_pending_edits();
                self.events.push(CoalescedCsvEvent::TerminalFocus);
            }
            CsvRawEvent::GitBranchCheckout { branch_info } => {
                self.flush_all_pending_edits();
                self.events
                    .push(CoalescedCsvEvent::GitBranchCheckout { branch_info });
            }
        }
    }

    fn finish(mut self) -> Vec<CoalescedCsvEvent> {
        self.flush_all_pending_edits();
        self.events
    }
}

/// Coalesce a raw CSV event stream into shared transitions.
pub fn coalesce_csv_events(
    raw_events: &[CsvRawEvent],
    coalesce_radius: usize,
) -> Vec<CoalescedCsvEvent> {
    let mut coalescer = CsvCoalescer::new(coalesce_radius);
    for event in raw_events {
        coalescer.handle_event(event.clone());
    }
    coalescer.finish()
}

/// Parse and coalesce one CSV session file.
pub fn parse_and_coalesce_csv_session(
    csv_path: &Path,
    coalesce_radius: usize,
) -> Result<Vec<CoalescedCsvEvent>, Box<dyn std::error::Error>> {
    let raw = parse_csv_session(csv_path)?;
    Ok(coalesce_csv_events(&raw, coalesce_radius))
}

/// Convert a character offset into (line, column) where line is 1-based.
pub fn offset_to_line_column(content: &str, offset: usize) -> (usize, usize) {
    let safe_offset = floor_char_boundary(content, offset.min(content.len()));
    let line = content[..safe_offset].matches('\n').count() + 1;
    let col = content[..safe_offset]
        .rsplit('\n')
        .next()
        .map(|s| s.len())
        .unwrap_or(0);
    (line, col)
}

/// Build a state-based Task from coalesced CSV transitions.
///
/// States are tagged with `NO_EVAL` by default for CSV-derived data.
pub fn coalesced_events_to_task(task_id: String, events: &[CoalescedCsvEvent]) -> Task {
    let mut files: HashMap<String, String> = HashMap::new();
    let mut cursor: Option<Cursor> = None;

    let mut states = Vec::with_capacity(events.len() + 1);
    states.push(State {
        step: Some(0),
        files: Some(files.clone()),
        cursor: None,
        terminal: None,
        eval_tag: Some("NO_EVAL".to_string()),
        judge_assertions: None,
    });

    for (idx, event) in events.iter().enumerate() {
        let mut terminal: Option<Terminal> = None;

        match event {
            CoalescedCsvEvent::Tab {
                file_path,
                text_content,
            } => {
                if let Some(content) = text_content {
                    files.insert(file_path.clone(), content.clone());
                } else {
                    files.entry(file_path.clone()).or_default();
                }
                cursor = Some(Cursor {
                    file: Some(file_path.clone()),
                    line: 1,
                    column: 0,
                });
            }
            CoalescedCsvEvent::Edit {
                file_path,
                before,
                after,
                cursor_line,
                cursor_column,
            } => {
                if files.get(file_path).map(|s| s.as_str()) != Some(before.as_str()) {
                    files.insert(file_path.clone(), before.clone());
                }
                files.insert(file_path.clone(), after.clone());
                cursor = Some(Cursor {
                    file: Some(file_path.clone()),
                    line: (*cursor_line).max(1),
                    column: *cursor_column,
                });
            }
            CoalescedCsvEvent::Selection { file_path, offset } => {
                let content = files.get(file_path).cloned().unwrap_or_default();
                let (line, column) = offset_to_line_column(&content, *offset);
                cursor = Some(Cursor {
                    file: Some(file_path.clone()),
                    line,
                    column,
                });
            }
            CoalescedCsvEvent::TerminalCommand { command } => {
                terminal = Some(Terminal {
                    command: Some(command.clone()),
                    output: None,
                    exit_code: None,
                    cwd: None,
                });
            }
            CoalescedCsvEvent::TerminalOutput { output } => {
                terminal = Some(Terminal {
                    command: None,
                    output: Some(output.clone()),
                    exit_code: None,
                    cwd: None,
                });
            }
            CoalescedCsvEvent::TerminalFocus => {}
            CoalescedCsvEvent::GitBranchCheckout { branch_info } => {
                terminal = Some(Terminal {
                    command: Some(branch_info.clone()),
                    output: None,
                    exit_code: None,
                    cwd: None,
                });
            }
        }

        states.push(State {
            step: Some(idx + 1),
            files: Some(files.clone()),
            cursor: cursor.clone(),
            terminal,
            eval_tag: Some("NO_EVAL".to_string()),
            judge_assertions: None,
        });
    }

    Task {
        task_id,
        description: None,
        states,
        labels: None,
    }
}

/// Convert a CSV session file to a state-based Task.
pub fn csv_session_to_task(
    csv_path: &Path,
    coalesce_radius: usize,
) -> Result<Task, Box<dyn std::error::Error>> {
    let events = parse_and_coalesce_csv_session(csv_path, coalesce_radius)?;
    let task_id = csv_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("csv_session")
        .to_string();
    Ok(coalesced_events_to_task(task_id, &events))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offset_to_line_column_one_based() {
        let content = "a\nbcd\nef";
        let (line, col) = offset_to_line_column(content, 4); // inside "bcd"
        assert_eq!(line, 2);
        assert_eq!(col, 2);
    }

    #[test]
    fn test_coalesce_nearby_edits_into_single_transition() {
        let events = vec![
            CsvRawEvent::Tab {
                file_path: "/t.py".to_string(),
                text_content: Some("line1\nline2\nline3".to_string()),
            },
            CsvRawEvent::Content {
                file_path: "/t.py".to_string(),
                offset: 6,
                length: 0,
                new_text: "A".to_string(),
            },
            CsvRawEvent::Content {
                file_path: "/t.py".to_string(),
                offset: 7,
                length: 0,
                new_text: "B".to_string(),
            },
            CsvRawEvent::TerminalFocus,
        ];

        let coalesced = coalesce_csv_events(&events, 5);
        let edit_count = coalesced
            .iter()
            .filter(|e| matches!(e, CoalescedCsvEvent::Edit { .. }))
            .count();
        assert_eq!(edit_count, 1);
    }
}
