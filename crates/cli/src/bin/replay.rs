use std::{
    collections::HashMap,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Terminal,
};
use serde::Deserialize;
use serde_json::Value;
use vt100::Parser as VtParser;
use flate2::read::GzDecoder;
use tar::Archive;

#[derive(Parser, Debug)]
#[command(name = "crowd-pilot-replay")]
#[command(author, version, about = "Replay crowd-code 2.0 sessions", long_about = None)]
struct Args {
    /// Path to a directory containing source_part_00x.tar.gz recordings
    #[arg(long)]
    session: PathBuf,

    /// Terminal columns for VT rendering
    #[arg(long)]
    terminal_cols: u16,

    /// Terminal rows for VT rendering
    #[arg(long)]
    terminal_rows: u16,

    /// Maximum delay in milliseconds between observations (0 = real-time)
    #[arg(long, default_value = "100")]
    delay_ms: u64,
}

#[derive(Debug, Deserialize)]
struct RecordingSession {
    version: String,
    #[serde(rename = "sessionId")]
    _session_id: String,
    #[serde(rename = "startTime")]
    _start_time: u64,
    events: Vec<RecordingEvent>,
}

#[derive(Debug, Deserialize)]
struct RecordingChunk {
    version: String,
    #[serde(rename = "sessionId")]
    session_id: String,
    #[serde(rename = "startTime")]
    start_time: u64,
    #[serde(rename = "chunkIndex", default)]
    chunk_index: Option<u64>,
    events: Vec<RecordingEvent>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum RecordingEvent {
    #[serde(rename = "observation")]
    Observation {
        sequence: u64,
        timestamp: u64,
        observation: Observation,
    },
    #[serde(rename = "action")]
    Action {
        #[serde(rename = "sequence")]
        _sequence: u64,
        #[serde(rename = "timestamp")]
        _timestamp: u64,
        action: Value,
    },
    #[serde(rename = "workspace_snapshot")]
    WorkspaceSnapshot {
        #[serde(rename = "sequence")]
        _sequence: u64,
        #[serde(rename = "timestamp")]
        _timestamp: u64,
        #[serde(rename = "snapshotId")]
        _snapshot_id: String,
    },
}

#[derive(Debug, Deserialize, Clone)]
struct Observation {
    viewport: Option<ViewportState>,
    #[serde(rename = "activeTerminal")]
    active_terminal: Option<TerminalViewport>,
}

#[derive(Debug, Deserialize, Clone)]
struct ViewportState {
    file: String,
    #[serde(rename = "startLine")]
    start_line: u64,
    #[serde(rename = "endLine")]
    end_line: u64,
    content: String,
    #[serde(rename = "cursorPosition")]
    cursor_position: Option<CursorPosition>,
}

#[derive(Debug, Deserialize, Clone)]
struct CursorPosition {
    line: u64,
    character: u64,
}

#[derive(Debug, Deserialize, Clone)]
struct TerminalViewport {
    id: String,
    name: String,
    viewport: Vec<String>,
}

struct TerminalState {
    name: String,
    parser: VtParser,
}

struct ReplayState {
    terminals: HashMap<String, TerminalState>,
    active_terminal_id: Option<String>,
}

impl ReplayState {
    fn new() -> Self {
        Self {
            terminals: HashMap::new(),
            active_terminal_id: None,
        }
    }

    fn ensure_terminal(&mut self, id: &str, name: Option<&str>, rows: u16, cols: u16) {
        let entry = self
            .terminals
            .entry(id.to_string())
            .or_insert_with(|| TerminalState {
                name: name.unwrap_or("terminal").to_string(),
                parser: VtParser::new(rows, cols, 0),
            });
        if let Some(name) = name {
            entry.name = name.to_string();
        }
    }

    fn set_active(&mut self, id: &str, name: Option<&str>, rows: u16, cols: u16) {
        self.ensure_terminal(id, name, rows, cols);
        self.active_terminal_id = Some(id.to_string());
    }
}

fn get_field(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|k| value.get(*k)?.as_str())
        .map(String::from)
}

fn handle_action(action: &Value, state: &mut ReplayState, rows: u16, cols: u16) {
    let Some(kind) = action.get("kind").and_then(|v| v.as_str()) else {
        return;
    };

    let terminal_id = get_field(action, &["terminalId", "terminal_id"]);
    let terminal_name = get_field(action, &["terminalName", "terminal_name"]);

    match kind {
        "terminal_focus" | "terminal_command" => {
            if let Some(id) = terminal_id {
                state.set_active(&id, terminal_name.as_deref(), rows, cols);
            }
        }
        "terminal_output" => {
            let id = terminal_id
                .or_else(|| state.active_terminal_id.clone())
                .unwrap_or_else(|| "terminal-unknown".to_string());

            state.ensure_terminal(&id, terminal_name.as_deref(), rows, cols);

            if let Some(payload) = get_field(action, &["text", "output", "data"]) {
                if let Some(terminal) = state.terminals.get_mut(&id) {
                    terminal.parser.process(payload.as_bytes());
                }
            }
        }
        _ => {}
    }
}

const TAB_WIDTH: usize = 4;

fn expand_tabs(s: &str) -> String {
    s.replace('\t', "    ")
}

/// Convert a column position from tab-based to expanded-spaces-based
fn expand_column(line: &str, col: usize) -> usize {
    let mut expanded_col = 0;
    for (i, c) in line.chars().enumerate() {
        if i >= col {
            break;
        }
        expanded_col += if c == '\t' { TAB_WIDTH } else { 1 };
    }
    expanded_col
}

fn format_line(start_line: u64, idx: usize, content: &str) -> String {
    format!("{:>6} | {}", start_line + idx as u64, expand_tabs(content))
}

fn split_at_char(s: &str, col: usize) -> (&str, &str) {
    if col == 0 {
        return ("", s);
    }
    match s.char_indices().nth(col) {
        Some((idx, _)) => s.split_at(idx),
        None => (s, ""),
    }
}

/// Render unified diff lines with syntax highlighting
fn render_diff_lines(diff: &str) -> Vec<Line<'static>> {
    diff.lines()
        .map(|line| {
            if line.starts_with("+++") || line.starts_with("---") {
                // File headers - dim
                Line::from(Span::styled(
                    line.to_string(),
                    Style::default().fg(Color::DarkGray),
                ))
            } else if line.starts_with("@@") {
                // Hunk headers - cyan
                Line::from(Span::styled(
                    line.to_string(),
                    Style::default().fg(Color::Cyan),
                ))
            } else if line.starts_with('+') {
                // Added lines - green
                Line::from(Span::styled(
                    line.to_string(),
                    Style::default().fg(Color::Green),
                ))
            } else if line.starts_with('-') {
                // Removed lines - red
                Line::from(Span::styled(
                    line.to_string(),
                    Style::default().fg(Color::Red),
                ))
            } else {
                // Context lines - default
                Line::from(line.to_string())
            }
        })
        .collect()
}

fn render_editor_lines(viewport: &ViewportState) -> Vec<Line<'static>> {
    let lines: Vec<&str> = viewport.content.split('\n').collect();
    let start = viewport.start_line;

    let Some(cursor) = &viewport.cursor_position else {
        return lines
            .iter()
            .enumerate()
            .map(|(i, l)| Line::from(format_line(start, i, l)))
            .collect();
    };

    let start_offset = start.saturating_sub(1);
    let rel_line = cursor.line.saturating_sub(start_offset) as usize;

    if rel_line >= lines.len() {
        return lines
            .iter()
            .enumerate()
            .map(|(i, l)| Line::from(format_line(start, i, l)))
            .collect();
    }

    lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            if i != rel_line {
                return Line::from(format_line(start, i, line));
            }

            let expanded = expand_tabs(line);
            let col = expand_column(line, cursor.character as usize);
            let expanded_chars = expanded.chars().count();
            let line_str = if col > expanded_chars {
                format!("{}{}", expanded, " ".repeat(col - expanded_chars))
            } else {
                expanded
            };

            let (prefix, rest) = split_at_char(&line_str, col);
            let mut chars = rest.chars();
            let cursor_char = chars.next().unwrap_or(' ');

            Line::from(vec![
                Span::raw(format!("{:>6} | ", start + i as u64)),
                Span::raw(prefix.to_string()),
                Span::styled(
                    cursor_char.to_string(),
                    Style::default().fg(Color::Black).bg(Color::Red),
                ),
                Span::raw(chars.collect::<String>()),
            ])
        })
        .collect()
}

/// Represents an agent edit diff for display
#[derive(Clone)]
struct AgentDiff {
    file: String,
    change_type: String,
    diff: String,
}

#[derive(Clone)]
struct Frame {
    sequence: u64,
    timestamp: u64,
    observation: Observation,
    terminal_title: String,
    terminal_lines: Vec<String>,
    action_log: Vec<String>,
    agent_diffs: Vec<AgentDiff>,
}

impl Frame {
    fn editor_title(&self) -> String {
        match &self.observation.viewport {
            Some(v) => format!("editor: {} (lines {}-{})", v.file, v.start_line, v.end_line),
            None => "editor: <no active editor>".to_string(),
        }
    }

    fn is_terminal_empty(&self) -> bool {
        self.terminal_lines.is_empty()
            || self
                .terminal_lines
                .iter()
                .all(|l| l.trim().is_empty() || l.trim() == "<no active terminal>")
    }

    fn has_agent_diffs(&self) -> bool {
        !self.agent_diffs.is_empty()
    }
}

/// Extract agent diff from a file_change action if source is "agent"
fn extract_agent_diff(action: &Value) -> Option<AgentDiff> {
    let kind = action.get("kind")?.as_str()?;
    if kind != "file_change" {
        return None;
    }
    
    let source = action.get("source").and_then(|v| v.as_str())?;
    if source != "agent" {
        return None;
    }
    
    let file = action.get("file").and_then(|v| v.as_str())?.to_string();
    let change_type = action
        .get("changeType")
        .and_then(|v| v.as_str())
        .unwrap_or("change")
        .to_string();
    let diff = action
        .get("diff")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    
    Some(AgentDiff {
        file,
        change_type,
        diff,
    })
}

fn render_ui<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    frame: &Frame,
    frame_index: usize,
    frame_count: usize,
    paused: bool,
) -> io::Result<()> {
    terminal.clear()?;
    terminal.draw(|f| {
        let size = f.size();

        let constraints = [
            Constraint::Length(3),
            Constraint::Min(5),
            if frame.is_terminal_empty() {
                Constraint::Length(1)
            } else {
                Constraint::Min(5)
            },
        ];
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(size);

        // Header
        let status = if paused { "paused" } else { "playing" };
        let header = Paragraph::new(format!(
            "crowd-code 2.0 replay  |  frame {}/{}  |  seq: {}  ts: {}  |  {}  |  q quit  space pause  ←/→ step",
            frame_index + 1, frame_count, frame.sequence, frame.timestamp, status
        ))
        .block(Block::default().borders(Borders::ALL).title("session"))
        .wrap(Wrap { trim: true });
        f.render_widget(header, layout[0]);

        // Editor
        f.render_widget(Clear, layout[1]);
        let editor_content = match &frame.observation.viewport {
            Some(v) => Text::from(render_editor_lines(v)),
            None => Text::from("<no active editor>"),
        };
        let editor = Paragraph::new(editor_content)
            .block(Block::default().borders(Borders::ALL).title(frame.editor_title()))
            .wrap(Wrap { trim: false });
        f.render_widget(editor, layout[1]);

        // Terminal
        f.render_widget(Clear, layout[2]);
        let term_lines: Vec<Line> = frame.terminal_lines.iter().map(|l| Line::from(expand_tabs(l))).collect();
        let term = Paragraph::new(Text::from(term_lines))
            .block(Block::default().borders(Borders::ALL).title(frame.terminal_title.clone()))
            .wrap(Wrap { trim: false });
        f.render_widget(term, layout[2]);

        // Overlay: show diff panel if agent diffs present, otherwise show actions
        if frame.has_agent_diffs() {
            // Agent diff panel - larger to show diff content
            let overlay_width: u16 = 60.min(size.width.saturating_sub(4));
            let overlay_height: u16 = 15.min(size.height.saturating_sub(4));
            
            if size.width > overlay_width + 2 && size.height > overlay_height + 1 {
                let rect = Rect::new(
                    size.width - overlay_width - 1,
                    1,
                    overlay_width,
                    overlay_height,
                );
                f.render_widget(Clear, rect);
                
                // Show the most recent agent diff
                let latest_diff = frame.agent_diffs.last().unwrap();
                let (added, removed) = count_diff_lines(&latest_diff.diff);
                let title = format!(
                    "agent {}: {} (+{}/-{})",
                    latest_diff.change_type, latest_diff.file, added, removed
                );
                
                let diff_lines = render_diff_lines(&latest_diff.diff);
                let overlay = Paragraph::new(Text::from(diff_lines))
                    .block(Block::default().borders(Borders::ALL).title(title))
                    .wrap(Wrap { trim: false });
                f.render_widget(overlay, rect);
            }
        } else {
            // Actions overlay (original behavior)
            let overlay_width: u16 = 48;
            let overlay_height: u16 = (frame.action_log.len() as u16).clamp(1, 6) + 2;
            if size.width > overlay_width + 2 && size.height > overlay_height + 1 {
                let rect = Rect::new(size.width - overlay_width - 1, 1, overlay_width, overlay_height.min(size.height - 1));
                f.render_widget(Clear, rect);
                let actions: Vec<Line> = if frame.action_log.is_empty() {
                    vec![Line::from("<no recent actions>")]
                } else {
                    frame.action_log.iter().rev().take(5).map(|l| Line::from(l.clone())).collect()
                };
                let overlay = Paragraph::new(Text::from(actions))
                    .block(Block::default().borders(Borders::ALL).title("recent actions"))
                    .wrap(Wrap { trim: true });
                f.render_widget(overlay, rect);
            }
        }
    })?;
    Ok(())
}

/// Count added/removed lines in a unified diff string
fn count_diff_lines(diff: &str) -> (usize, usize) {
    let mut added = 0;
    let mut removed = 0;
    for line in diff.lines() {
        if line.starts_with('+') && !line.starts_with("+++") {
            added += 1;
        } else if line.starts_with('-') && !line.starts_with("---") {
            removed += 1;
        }
    }
    (added, removed)
}

fn summarize_action(action: &Value) -> Option<String> {
    let kind = action.get("kind")?.as_str()?;

    Some(match kind {
        "terminal_command" => {
            let command = action.get("command").and_then(|v| v.as_str()).unwrap_or("<command?>");
            if let Some(term) = get_field(action, &["terminalName", "terminal_name"]) {
                format!("cmd ({}): {}", term, command.trim())
            } else {
                format!("cmd: {}", command.trim())
            }
        }
        "selection" => {
            let file = action.get("file").and_then(|v| v.as_str()).unwrap_or("<file?>");
            let pos = |key: &str| {
                action.get(key).and_then(|v| {
                    Some((v.get("line")?.as_u64()?, v.get("character")?.as_u64()?))
                })
            };
            match (pos("selectionStart"), pos("selectionEnd")) {
                (Some((ls, cs)), Some((le, ce))) if ls == le && cs == ce => {
                    format!("cursor: {} {}:{}", file, ls + 1, cs + 1)
                }
                (Some((ls, cs)), Some((le, ce))) => {
                    format!("sel: {} {}:{}-{}:{}", file, ls + 1, cs + 1, le + 1, ce + 1)
                }
                _ => format!("selection: {}", file),
            }
        }
        "edit" => {
            let file = action.get("file").and_then(|v| v.as_str()).unwrap_or("<file?>");
            let source = action.get("source").and_then(|v| v.as_str()).unwrap_or("?");
            let diff = action.get("diff");
            let chars_added = diff
                .and_then(|d| d.get("text"))
                .and_then(|t| t.as_str())
                .map(|t| t.len())
                .unwrap_or(0);
            let chars_removed = diff
                .and_then(|d| d.get("rangeLength"))
                .and_then(|l| l.as_u64())
                .unwrap_or(0) as usize;
            format!("edit ({}): {} +{}/-{} chars", source, file, chars_added, chars_removed)
        }
        "file_change" => {
            let file = action.get("file").and_then(|v| v.as_str()).unwrap_or("<file?>");
            let source = action.get("source").and_then(|v| v.as_str()).unwrap_or("?");
            let change_type = action.get("changeType").and_then(|v| v.as_str()).unwrap_or("change");
            let diff_str = action.get("diff").and_then(|v| v.as_str());
            
            let diff_summary = if let Some(d) = diff_str {
                let (added, removed) = count_diff_lines(d);
                format!(" +{}/-{}", added, removed)
            } else {
                String::new()
            };
            
            match source {
                "agent" => format!("agent {}: {}{}", change_type, file, diff_summary),
                "git" | "git_checkout" => format!("git: {} ({})", file, change_type),
                _ => format!("{}: {} ({}){}", source, file, change_type, diff_summary),
            }
        }
        "tab_switch" => {
            let file = action.get("file").and_then(|v| v.as_str()).unwrap_or("<file?>");
            let prev = action.get("previousFile").and_then(|v| v.as_str());
            match prev {
                Some(p) => format!("tab: {} → {}", p, file),
                None => format!("tab: → {}", file),
            }
        }
        "terminal_focus" => {
            let term = get_field(action, &["terminalName", "terminal_name"]).unwrap_or_else(|| "<terminal?>".to_string());
            format!("focus: {}", term)
        }
        _ => {
            if let Some(file) = action.get("file").and_then(|v| v.as_str()) {
                format!("{}: {}", kind, file)
            } else if let Some(term) = get_field(action, &["terminalName", "terminal_name"]) {
                format!("{}: {}", kind, term)
            } else if let Some(text) = action.get("text").and_then(|v| v.as_str()) {
                format!("{}: {}", kind, text.trim())
            } else {
                kind.to_string()
            }
        }
    })
}

fn build_frames(session: &RecordingSession, rows: u16, cols: u16) -> Vec<Frame> {
    let mut state = ReplayState::new();
    let mut frames = Vec::new();
    let mut action_log: Vec<String> = Vec::new();
    let mut agent_diffs: Vec<AgentDiff> = Vec::new();

    for event in &session.events {
        match event {
            RecordingEvent::Action { action, .. } => {
                handle_action(action, &mut state, rows, cols);
                
                // Track agent diffs separately
                if let Some(agent_diff) = extract_agent_diff(action) {
                    agent_diffs.push(agent_diff);
                    // Keep only recent agent diffs (max 10)
                    if agent_diffs.len() > 10 {
                        agent_diffs.drain(0..agent_diffs.len() - 10);
                    }
                }
                
                if let Some(summary) = summarize_action(action) {
                    action_log.push(summary);
                    if action_log.len() > 50 {
                        action_log.drain(0..action_log.len() - 50);
                    }
                }
            }
            RecordingEvent::Observation { sequence, timestamp, observation } => {
                if let Some(t) = &observation.active_terminal {
                    state.set_active(&t.id, Some(&t.name), rows, cols);
                }

                let (terminal_title, terminal_lines) = match &observation.active_terminal {
                    Some(t) if !t.viewport.is_empty() => {
                        let mut parser = VtParser::new(rows, cols, 0);
                        for line in &t.viewport {
                            parser.process(line.as_bytes());
                            parser.process(b"\n");
                        }
                        let content = parser.screen().contents();
                        let lines: Vec<String> = content
                            .lines()
                            .map(String::from)
                            .collect();
                        (format!("terminal: {} ({})", t.name, t.id), lines)
                    }
                    _ => match &state.active_terminal_id {
                        Some(id) => match state.terminals.get(id) {
                            Some(t) => {
                                let content = t.parser.screen().contents();
                                let lines: Vec<String> = content.lines().map(String::from).collect();
                                if lines.iter().all(|l| l.trim().is_empty()) {
                                    (format!("terminal: {} ({})", t.name, id), vec![])
                                } else {
                                    (format!("terminal: {} ({})", t.name, id), lines)
                                }
                            }
                            None => (format!("terminal: ({})", id), vec![]),
                        },
                        None => ("terminal: <none>".to_string(), vec![]),
                    },
                };

                frames.push(Frame {
                    sequence: *sequence,
                    timestamp: *timestamp,
                    observation: observation.clone(),
                    terminal_title,
                    terminal_lines,
                    action_log: action_log.clone(),
                    agent_diffs: agent_diffs.clone(),
                });
                
                // Clear agent diffs after including them in a frame
                // (so each frame shows diffs that occurred since last observation)
                agent_diffs.clear();
            }
            RecordingEvent::WorkspaceSnapshot { .. } => {}
        }
    }

    frames
}

fn is_chunk_entry(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    name.starts_with("chunk_") && name.ends_with(".json")
}

fn load_tar_chunks(path: &Path) -> Result<Vec<(String, RecordingChunk)>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    let mut chunks = Vec::new();

    for entry in archive.entries()? {
        let mut entry = entry?;
        let entry_path = entry.path()?.to_path_buf();
        let entry_path_str = entry_path.to_string_lossy().to_string();
        let is_snapshot = entry_path_str.contains("/snapshots/") || entry_path_str.contains("\\snapshots\\");
        if is_snapshot || !is_chunk_entry(&entry_path) {
            continue;
        }

        let mut payload = String::new();
        entry.read_to_string(&mut payload)?;
        let chunk: RecordingChunk = serde_json::from_str(&payload)?;
        let key = format!("{}:{}", path.display(), entry_path_str);
        chunks.push((key, chunk));
    }

    Ok(chunks)
}

fn load_session(path: &Path) -> Result<RecordingSession, Box<dyn std::error::Error>> {
    let mut entries: Vec<PathBuf> = Vec::new();
    if path.is_dir() {
        entries = fs::read_dir(path)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|entry| {
                entry
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.ends_with(".tar.gz"))
                    .unwrap_or(false)
            })
            .collect();
    } else {
        entries.push(path.to_path_buf());
    }

    if entries.is_empty() {
        return Err("No source_part_*.tar.gz files found in directory".into());
    }

    let mut chunks: Vec<(String, RecordingChunk)> = Vec::new();
    for entry in entries {
        chunks.extend(load_tar_chunks(&entry)?);
    }

    let has_chunk_index = chunks.iter().all(|(_, chunk)| chunk.chunk_index.is_some());
    if has_chunk_index {
        chunks.sort_by_key(|(_, chunk)| (chunk.chunk_index.unwrap_or(0), chunk.start_time));
    } else {
        chunks.sort_by_key(|(key, _)| key.clone());
    }

    let mut session: Option<RecordingSession> = None;
    let mut events = Vec::new();

    for (entry, chunk) in chunks {
        if let Some(base) = &session {
            if base.version != chunk.version {
                eprintln!(
                    "Warning: version mismatch in {} ({} vs {})",
                    entry, chunk.version, base.version
                );
            }
        } else {
            session = Some(RecordingSession {
                version: chunk.version.clone(),
                _session_id: chunk.session_id.clone(),
                _start_time: chunk.start_time,
                events: Vec::new(),
            });
        }

        events.extend(chunk.events);
    }

    let mut session = session.ok_or("No chunk_*.json files found in source parts")?;
    session.events = events;
    Ok(session)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let session = load_session(&args.session)?;

    if session.version != "2.0" {
        eprintln!("Warning: expected version 2.0, got {}", session.version);
    }

    let frames = build_frames(&session, args.terminal_rows, args.terminal_cols);
    if frames.is_empty() {
        eprintln!("No observation frames found in session.");
        return Ok(());
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut frame_index = 0usize;
    let mut paused = false;
    let mut frame_start = Instant::now();
    let frame_count = frames.len();

    let render = |t: &mut Terminal<_>, idx: usize, p: bool| render_ui(t, &frames[idx], idx, frame_count, p);

    render(&mut terminal, frame_index, paused)?;

    loop {
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char(' ') => {
                        paused = !paused;
                        frame_start = Instant::now();
                        render(&mut terminal, frame_index, paused)?;
                    }
                    KeyCode::Right if frame_index + 1 < frame_count => {
                        frame_index += 1;
                        paused = true;
                        render(&mut terminal, frame_index, paused)?;
                    }
                    KeyCode::Left if frame_index > 0 => {
                        frame_index -= 1;
                        paused = true;
                        render(&mut terminal, frame_index, paused)?;
                    }
                    _ => {}
                }
            }
        }

        if paused || frame_index + 1 >= frame_count {
            if frame_index + 1 >= frame_count {
                paused = true;
            }
            continue;
        }

        let delay = frames[frame_index + 1].timestamp.saturating_sub(frames[frame_index].timestamp);
        let sleep_ms = if args.delay_ms == 0 { delay } else { delay.min(args.delay_ms) };

        if frame_start.elapsed() >= Duration::from_millis(sleep_ms) {
            frame_index += 1;
            frame_start = Instant::now();
            render(&mut terminal, frame_index, paused)?;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
