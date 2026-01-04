//! Core serialization logic for crowd-pilot IDE interaction data.
//!
//! This crate provides the `ConversationStateManager` which converts IDE events
//! (tab switches, edits, terminal commands, etc.) into conversation format
//! suitable for training language models.

/// Trait for tokenization operations.
/// 
/// Implementors provide token counting and truncation capabilities.
/// For exact tokenization (preprocessing), use a real tokenizer.
/// For approximate tokenization (runtime), use character-based estimation.
pub trait Tokenizer {
    /// Count the number of tokens in the given text.
    fn count_tokens(&self, text: &str) -> usize;
    
    /// Truncate text to at most `max_tokens` tokens.
    /// Returns the truncated text.
    fn truncate_to_max_tokens(&self, text: &str, max_tokens: usize) -> String;
}

// Blanket implementation for references to Tokenizers
impl<T: Tokenizer + ?Sized> Tokenizer for &T {
    fn count_tokens(&self, text: &str) -> usize {
        (*self).count_tokens(text)
    }
    
    fn truncate_to_max_tokens(&self, text: &str, max_tokens: usize) -> String {
        (*self).truncate_to_max_tokens(text, max_tokens)
    }
}

mod conversation;
mod diff;
mod helpers;
pub mod pipeline;

pub use conversation::{ConversationMessage, ConversationStateManager, ConversationStateManagerConfig, FinalizedConversation};
pub use pipeline::{
    discover_csv_files, process_all_sessions, process_session, write_jsonl_output,
    NemoMessage, NemoRecord, PipelineConfig, PipelineResult, SessionResult,
};
pub use diff::{compute_changed_block_lines, ChangedBlock};
pub use helpers::{
    apply_backspaces, apply_change, clean_text, escape_single_quotes_for_sed, fenced_block,
    line_numbered_output, normalize_terminal_output, serialize_compute_viewport, Viewport,
};

/// Default viewport radius (lines above/below cursor to show)
pub const VIEWPORT_RADIUS: usize = 10;

/// Default coalesce radius for grouping nearby edits
pub const COALESCE_RADIUS: usize = 5;

/// Default maximum tokens per message (approximate)
pub const MAX_TOKENS_PER_MESSAGE: usize = 2048;

/// Default maximum tokens per terminal output
pub const MAX_TOKENS_PER_TERMINAL_OUTPUT: usize = 256;

/// Generate the default system prompt with the given viewport radius.
pub fn default_system_prompt(viewport_radius: usize) -> String {
    let viewport_lines = 2 * viewport_radius + 1;
    format!(
        r#"You are a helpful assistant that interacts with a computer shell to solve programming tasks.
Your goal is to predict the next bash command a developer would most likely execute, given their editing and navigation history.

=== CONVERSATION FORMAT ===
The conversation history alternates between:
- Assistant messages: bash commands in fenced code blocks
- User messages: command output wrapped in <stdout>...</stdout> tags

After each edit, you should show the resulting file contents using `cat -n FILE | sed -n 'START,ENDp'`, which produces 6-character right-aligned line numbers followed by a tab, e.g.:
     1	first line
     2	second line

The chained cat command should show {viewport_lines} lines around the edited region.

=== RESPONSE FORMAT ===
Your response must contain exactly ONE bash code block with one command or two commands connected with &&.

<format_example>
```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected.

=== EDIT COMMAND FORMAT (IMPORTANT) ===
When you want to EDIT a file, you MUST encode the edit using line-based sed commands in ONE of the following forms, and you MUST NOT use substitution commands like "Ns/old/new/g".

Assume all line numbers are 1-based and paths are absolute.
Allowed edit encodings (choose exactly one per response):

1) Replace a contiguous block of lines:
   sed -i 'START,ENDc\
NEW_LINE_1\
NEW_LINE_2\
...
' /abs/path/to/file && cat -n /abs/path/to/file | sed -n 'VSTART,VENDp'

2) Delete a contiguous block of lines:
   sed -i 'START,ENDd' /abs/path/to/file && cat -n /abs/path/to/file | sed -n 'VSTART,VENDp'

3) Insert new lines BEFORE a given line:
   sed -i 'STARTi\
NEW_LINE_1\
NEW_LINE_2\
...
' /abs/path/to/file && cat -n /abs/path/to/file | sed -n 'VSTART,VENDp'

4) Append new lines at the END of the file:
   sed -i '$a\
NEW_LINE_1\
NEW_LINE_2\
...
' /abs/path/to/file && cat -n /abs/path/to/file | sed -n 'VSTART,VENDp'

Where VSTART and VEND specify a small viewport around the edited region.

Do NOT emit commands like "3s/print/print()/g" or any other "s/old/new/" style sed substitution; instead, always rewrite the affected lines using one of the line-based forms above.

When you are NOT editing files (e.g., running tests, git commands, tools, etc.), you may emit arbitrary bash commands."#
    )
}

