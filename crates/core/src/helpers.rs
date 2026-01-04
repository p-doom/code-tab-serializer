//! Helper functions for text processing and serialization.

use regex::Regex;
use std::sync::LazyLock;

// ANSI escape sequence patterns
static ANSI_CSI_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x1b\[[0-9;?]*[ -/]*[@-~]").unwrap());
static ANSI_OSC_TERMINATED_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x1b\][\s\S]*?(?:\x07|\x1b\\)").unwrap());
static ANSI_OSC_LINE_FALLBACK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x1b\][^\n]*$").unwrap());

/// Find the largest valid UTF-8 char boundary <= index.
///
/// This function prevents panics when the index lands inside a multi-byte
/// character by walking back at most 3 bytes to find a valid boundary.
/// # Approximation Warning
///
/// This is a pragmatic fix for handling offsets from VS Code, which provides
/// UTF-16 code unit offsets, while Rust strings use UTF-8 byte offsets.
///
/// - For ASCII/BMP text: UTF-16 offset ≈ UTF-8 byte offset (works correctly)
/// - For text with emojis/astral chars: UTF-16 offset < UTF-8 byte offset
///   (we undercount, slicing earlier in the file than intended)
///
/// The correct solution would be UTF-16 → UTF-8 offset conversion (O(n)),
/// but this O(1) approximation is acceptable for line counting where small
/// inaccuracies in viewport position don't significantly impact the output.
pub fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        s.len()
    } else if s.is_char_boundary(index) {
        index
    } else {
        // Walk backwards to find a char boundary
        let mut i = index;
        while i > 0 && !s.is_char_boundary(i) {
            i -= 1;
        }
        i
    }
}

/// Clean text by normalizing line endings and trimming trailing whitespace.
pub fn clean_text(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .trim_end()
        .to_string()
}

/// Create a fenced code block with optional language tag.
pub fn fenced_block(language: Option<&str>, content: &str) -> String {
    let lang = language.unwrap_or("").to_lowercase();
    format!("```{}\n{}\n```\n", lang, content)
}

/// Apply a text change at the given offset.
pub fn apply_change(content: &str, offset: usize, length: usize, new_text: &str) -> String {
    let mut base = content.to_string();

    // Handle escaped newlines in new_text
    let text = new_text.replace("\\n", "\n").replace("\\r", "\r");

    // Pad with spaces if offset is beyond content length
    if offset > base.len() {
        base.push_str(&" ".repeat(offset - base.len()));
    }

    // Get character indices (handle UTF-8)
    let chars: Vec<char> = base.chars().collect();
    let safe_offset = offset.min(chars.len());
    let safe_length = length.min(chars.len().saturating_sub(safe_offset));

    let before: String = chars[..safe_offset].iter().collect();
    let after: String = chars[safe_offset + safe_length..].iter().collect();

    format!("{}{}{}", before, text, after)
}

/// Apply backspace characters (\x08) to text.
pub fn apply_backspaces(text: &str) -> String {
    let mut out: Vec<char> = Vec::new();
    for ch in text.chars() {
        if ch == '\x08' {
            out.pop();
        } else {
            out.push(ch);
        }
    }
    out.into_iter().collect()
}

/// Normalize terminal output by removing ANSI sequences and handling carriage returns.
pub fn normalize_terminal_output(raw: &str) -> String {
    if raw.is_empty() {
        return raw.to_string();
    }

    // Apply backspaces
    let mut s = apply_backspaces(raw);

    // Remove OSC sequences that are properly terminated (BEL or ST)
    s = ANSI_OSC_TERMINATED_RE.replace_all(&s, "").to_string();

    // Fallback: drop any unterminated OSC up to end-of-line
    s = s
        .split('\n')
        .map(|line| ANSI_OSC_LINE_FALLBACK_RE.replace_all(line, "").to_string())
        .collect::<Vec<_>>()
        .join("\n");

    // Resolve carriage returns per line
    let resolved_lines: Vec<String> = s
        .split('\n')
        .map(|seg| {
            let parts: Vec<&str> = seg.split('\r').collect();
            // Pick last non-empty part if available; else last part
            parts
                .iter()
                .rev()
                .find(|p| !p.is_empty())
                .unwrap_or(parts.last().unwrap_or(&""))
                .to_string()
        })
        .collect();

    s = resolved_lines.join("\n");

    // Strip ANSI CSI escape sequences
    s = ANSI_CSI_RE.replace_all(&s, "").to_string();

    // Remove any remaining BEL beeps
    s = s.replace('\x07', "");

    s
}

/// Generate line-numbered output matching `cat -n` format.
///
/// Lines are numbered with 6-character right-aligned numbers followed by a tab.
pub fn line_numbered_output(content: &str, start_line: Option<usize>, end_line: Option<usize>) -> String {
    let lines: Vec<&str> = content.split('\n').collect();
    let total = if lines.len() == 1 && lines[0].is_empty() {
        0
    } else {
        lines.len()
    };

    if total == 0 {
        return String::new();
    }

    let s = start_line.map(|l| l.max(1).min(total)).unwrap_or(1);
    let e = end_line.map(|l| l.max(1).min(total)).unwrap_or(total);

    let mut buf = Vec::new();
    for idx in s..=e {
        let line_text = lines.get(idx - 1).unwrap_or(&"");
        buf.push(format!("{:6}\t{}", idx, line_text));
    }
    buf.join("\n")
}

/// Viewport with start and end line numbers (1-based, inclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Viewport {
    pub start: usize,
    pub end: usize,
}

/// Compute a viewport centered around a given line.
pub fn serialize_compute_viewport(total_lines: usize, center_line: usize, radius: usize) -> Viewport {
    if total_lines == 0 {
        return Viewport { start: 1, end: 0 };
    }
    let start = center_line.saturating_sub(radius).max(1);
    let end = (center_line + radius).min(total_lines);
    Viewport { start, end }
}

/// Escape special characters for use in sed replacement text.
///
/// Escapes backslashes (doubled) and single quotes (shell quote-switching technique).
pub fn escape_single_quotes_for_sed(text: &str) -> String {
    // 1. Escape backslashes first: \ -> \\
    // 2. Escape single quotes: ' -> '"'"' (close quote, add escaped quote via double quotes, reopen)
    text.replace('\\', "\\\\")
        .replace('\'', "'\"'\"'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text() {
        assert_eq!(clean_text("hello\r\nworld\r"), "hello\nworld");
        // trimEnd() removes all trailing whitespace including newlines
        assert_eq!(clean_text("test  \n  "), "test");
        assert_eq!(clean_text("test\n"), "test");
        assert_eq!(clean_text("test  "), "test");
    }

    #[test]
    fn test_fenced_block() {
        assert_eq!(
            fenced_block(Some("bash"), "echo hello"),
            "```bash\necho hello\n```\n"
        );
        assert_eq!(fenced_block(None, "code"), "```\ncode\n```\n");
    }

    #[test]
    fn test_apply_change() {
        assert_eq!(apply_change("hello", 5, 0, " world"), "hello world");
        assert_eq!(apply_change("hello", 0, 5, "hi"), "hi");
        // offset=2 (after "he"), length=2 (delete "ll"), insert "y" → "he" + "y" + "o"
        assert_eq!(apply_change("hello", 2, 2, "y"), "heyo");
        // Test with length=1
        assert_eq!(apply_change("hello", 2, 1, "y"), "heylo");
    }

    #[test]
    fn test_line_numbered_output() {
        let content = "line1\nline2\nline3";
        let output = line_numbered_output(content, Some(2), Some(3));
        assert!(output.contains("     2\tline2"));
        assert!(output.contains("     3\tline3"));
    }

    #[test]
    fn test_viewport() {
        let vp = serialize_compute_viewport(100, 50, 10);
        assert_eq!(vp.start, 40);
        assert_eq!(vp.end, 60);

        let vp = serialize_compute_viewport(100, 5, 10);
        assert_eq!(vp.start, 1);
        assert_eq!(vp.end, 15);
    }
}

