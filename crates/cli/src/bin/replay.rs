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

#[derive(Parser, Debug)]
#[command(name = "crowd-pilot-replay")]
#[command(author, version, about = "Replay crowd-code 2.0 sessions", long_about = None)]
struct Args {
    /// Path to a crowd-code 2.0 recording JSON.gz file or directory of chunks
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
            let line_str = if col > expanded.len() {
                format!("{}{}", expanded, " ".repeat(col - expanded.len()))
            } else {
                expanded
            };

            let (prefix, rest) = line_str.split_at(col);
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

#[derive(Clone)]
struct Frame {
    sequence: u64,
    timestamp: u64,
    observation: Observation,
    terminal_title: String,
    terminal_lines: Vec<String>,
    action_log: Vec<String>,
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

        // Actions overlay
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
    })?;
    Ok(())
}

fn summarize_action(action: &Value) -> Option<String> {
    let kind = action.get("kind")?.as_str()?;

    Some(match kind {
        "terminal_command" => {
            let command = action.get("command").and_then(|v| v.as_str()).unwrap_or("<command?>");
            if let Some(term) = get_field(action, &["terminalName", "terminal_name"]) {
                format!("terminal command ({}): {}", term, command.trim())
            } else {
                format!("terminal command: {}", command.trim())
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

    for event in &session.events {
        match event {
            RecordingEvent::Action { action, .. } => {
                handle_action(action, &mut state, rows, cols);
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
                });
            }
            RecordingEvent::WorkspaceSnapshot { .. } => {}
        }
    }

    frames
}

fn read_session_file(path: &Path) -> Result<RecordingChunk, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let mut decoder = GzDecoder::new(file);
    let mut payload = String::new();
    decoder.read_to_string(&mut payload)?;
    Ok(serde_json::from_str(&payload)?)
}

fn load_session(path: &Path) -> Result<RecordingSession, Box<dyn std::error::Error>> {
    if path.is_dir() {
        let entries: Vec<PathBuf> = fs::read_dir(path)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|entry| {
                entry
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.ends_with(".json.gz"))
                    .unwrap_or(false)
            })
            .collect();

        if entries.is_empty() {
            return Err("No session files found in directory".into());
        }

        let mut chunks: Vec<(PathBuf, RecordingChunk)> = Vec::new();
        for entry in entries {
            chunks.push((entry.clone(), read_session_file(&entry)?));
        }

        let has_chunk_index = chunks.iter().all(|(_, chunk)| chunk.chunk_index.is_some());
        if has_chunk_index {
            chunks.sort_by_key(|(_, chunk)| (chunk.chunk_index.unwrap_or(0), chunk.start_time));
        } else {
            chunks.sort_by_key(|(path, _)| path.clone());
        }

        let mut session: Option<RecordingSession> = None;
        let mut events = Vec::new();

        for (entry, chunk) in chunks {
            if let Some(base) = &session {
                if base.version != chunk.version {
                    eprintln!(
                        "Warning: version mismatch in {:?} ({} vs {})",
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

        let mut session = session.ok_or("No session files found in directory")?;
        session.events = events;
        Ok(session)
    } else {
        if path.extension().and_then(|ext| ext.to_str()) != Some("gz") {
            return Err("Only .json.gz files are supported for --session".into());
        }
        let chunk = read_session_file(path)?;
        Ok(RecordingSession {
            version: chunk.version,
            _session_id: chunk.session_id,
            _start_time: chunk.start_time,
            events: chunk.events,
        })
    }
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
