//! Node.js bindings for the crowd-pilot serializer.
//!
//! For the VS Code extension, we use character-based token approximation
//! since accurate tokenization is not required for runtime inference.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Mutex;

use crowd_pilot_serializer_core::{
    ConversationMessage as CoreMessage, ConversationStateManager as CoreManager,
    ConversationStateManagerConfig, Tokenizer,
};

/// A message in the conversation.
#[napi(object)]
pub struct ConversationMessage {
    pub from: String,
    pub value: String,
}

impl From<CoreMessage> for ConversationMessage {
    fn from(msg: CoreMessage) -> Self {
        Self {
            from: msg.from,
            value: msg.value,
        }
    }
}

/// Configuration options for the ConversationStateManager.
/// All fields are optional; unspecified values use core defaults.
#[napi(object)]
pub struct ConversationStateManagerOptions {
    /// Viewport radius (lines above/below cursor to show).
    pub viewport_radius: Option<u32>,
    /// Coalesce radius for grouping nearby edits.
    pub coalesce_radius: Option<u32>,
    /// Maximum tokens per message.
    pub max_tokens_per_message: Option<u32>,
    /// Maximum tokens per terminal output.
    pub max_tokens_per_terminal_output: Option<u32>,
}

/// Character-based approximate tokenizer (~4 chars per token).
/// Used for the VS Code extension runtime where exact tokenization is not required.
struct CharApproxTokenizer;

impl Tokenizer for CharApproxTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        text.len() / 4
    }

    fn truncate_to_max_tokens(&self, text: &str, max_tokens: usize) -> String {
        text.chars().take(max_tokens * 4).collect()
    }
}

/// Manages conversation state for serializing IDE events.
///
/// Uses character-based token approximation for the VS Code extension runtime.
/// For accurate tokenization during preprocessing, use the CLI with Python bindings.
#[napi]
pub struct ConversationStateManager {
    inner: Mutex<CoreManager<CharApproxTokenizer>>,
}

#[napi]
impl ConversationStateManager {
    /// Create a new ConversationStateManager with default character-based token approximation.
    ///
    /// @param options - Optional configuration options.
    #[napi(constructor)]
    pub fn new(options: Option<ConversationStateManagerOptions>) -> Result<Self> {
        let defaults = ConversationStateManagerConfig::default();
        
        let config = match options {
            Some(opts) => ConversationStateManagerConfig {
                viewport_radius: opts.viewport_radius.map(|v| v as usize).unwrap_or(defaults.viewport_radius),
                coalesce_radius: opts.coalesce_radius.map(|v| v as usize).unwrap_or(defaults.coalesce_radius),
                max_tokens_per_message: opts.max_tokens_per_message.map(|v| v as usize).unwrap_or(defaults.max_tokens_per_message),
                max_tokens_per_terminal_output: opts.max_tokens_per_terminal_output.map(|v| v as usize).unwrap_or(defaults.max_tokens_per_terminal_output),
                // Extension-specific: no chunking (single ongoing conversation)
                max_tokens_per_conversation: None,
                min_conversation_messages: defaults.min_conversation_messages,
            },
            None => ConversationStateManagerConfig {
                // Extension-specific: no chunking
                max_tokens_per_conversation: None,
                ..defaults
            },
        };

        Ok(Self {
            inner: Mutex::new(CoreManager::new(CharApproxTokenizer, config)),
        })
    }

    /// Reset all state.
    #[napi]
    pub fn reset(&self) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.reset();
        Ok(())
    }

    /// Get a copy of all messages.
    #[napi]
    pub fn get_messages(&self) -> Result<Vec<ConversationMessage>> {
        let inner = self.inner.lock().map_err(|_| Error::from_reason("Lock poisoned"))?;
        Ok(inner.get_messages().into_iter().map(Into::into).collect())
    }

    /// Get the current content of a file.
    #[napi]
    pub fn get_file_content(&self, file_path: String) -> Result<String> {
        let inner = self.inner.lock().map_err(|_| Error::from_reason("Lock poisoned"))?;
        Ok(inner.get_file_content(&file_path))
    }

    /// Handle a tab (file switch) event.
    ///
    /// @param filePath - The path to the file.
    /// @param textContent - The file contents, or null if switching to an already-open file.
    #[napi]
    pub fn handle_tab_event(&self, file_path: String, text_content: Option<String>) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_tab_event(&file_path, text_content.as_deref());
        Ok(())
    }

    /// Handle a content change event.
    ///
    /// @param filePath - The path to the file.
    /// @param offset - The character offset where the change starts.
    /// @param length - The number of characters being replaced.
    /// @param newText - The new text being inserted.
    #[napi]
    pub fn handle_content_event(
        &self,
        file_path: String,
        offset: u32,
        length: u32,
        new_text: String,
    ) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_content_event(&file_path, offset as usize, length as usize, &new_text);
        Ok(())
    }

    /// Handle a selection event.
    ///
    /// @param filePath - The path to the file.
    /// @param offset - The character offset of the selection start.
    #[napi]
    pub fn handle_selection_event(&self, file_path: String, offset: u32) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_selection_event(&file_path, offset as usize);
        Ok(())
    }

    /// Handle a terminal command event.
    ///
    /// @param command - The command that was executed.
    #[napi]
    pub fn handle_terminal_command_event(&self, command: String) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_terminal_command_event(&command);
        Ok(())
    }

    /// Handle a terminal output event.
    ///
    /// @param output - The terminal output.
    #[napi]
    pub fn handle_terminal_output_event(&self, output: String) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_terminal_output_event(&output);
        Ok(())
    }

    /// Handle a terminal focus event.
    #[napi]
    pub fn handle_terminal_focus_event(&self) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_terminal_focus_event();
        Ok(())
    }

    /// Handle a git branch checkout event.
    ///
    /// @param branchInfo - The git checkout message containing the branch name.
    #[napi]
    pub fn handle_git_branch_checkout_event(&self, branch_info: String) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_git_branch_checkout_event(&branch_info);
        Ok(())
    }

    /// Finalize and get conversation ready for model.
    #[napi]
    pub fn finalize_for_model(&self) -> Result<Vec<ConversationMessage>> {
        let mut inner = self.inner.lock().map_err(|_| Error::from_reason("Lock poisoned"))?;
        Ok(inner.finalize_for_model().into_iter().map(Into::into).collect())
    }
}

/// Helper function: estimate tokens using character approximation.
/// Uses ~4 characters per token as a rough approximation.
#[napi]
pub fn estimate_tokens(text: String) -> u32 {
    (text.len() / 4) as u32
}

/// Helper function: clean text by normalizing line endings.
#[napi]
pub fn clean_text(text: String) -> String {
    crowd_pilot_serializer_core::clean_text(&text)
}

/// Helper function: create a fenced code block.
#[napi]
pub fn fenced_block(language: Option<String>, content: String) -> String {
    crowd_pilot_serializer_core::fenced_block(language.as_deref(), &content)
}

/// Helper function: normalize terminal output.
#[napi]
pub fn normalize_terminal_output(raw: String) -> String {
    crowd_pilot_serializer_core::normalize_terminal_output(&raw)
}

/// Helper function: generate line-numbered output.
#[napi]
pub fn line_numbered_output(
    content: String,
    start_line: Option<u32>,
    end_line: Option<u32>,
) -> String {
    crowd_pilot_serializer_core::line_numbered_output(
        &content,
        start_line.map(|v| v as usize),
        end_line.map(|v| v as usize),
    )
}

/// Get the default system prompt for the model.
///
/// This returns the same system prompt used during preprocessing, ensuring
/// consistency between training and deployment.
///
/// @param viewportRadius - Viewport radius (lines above/below cursor).
#[napi]
pub fn get_default_system_prompt(viewport_radius: u32) -> String {
    crowd_pilot_serializer_core::default_system_prompt(viewport_radius as usize)
}
