//! Node.js bindings for the crowd-pilot serializer.
//!
//! For the VS Code extension, we use character-based token approximation
//! since accurate tokenization is not required for runtime inference.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Mutex;

use crowd_pilot_serializer_core::{
    convert_yaml_to_sweep as core_convert_yaml_to_sweep,
    sweep_system_prompt as core_sweep_system_prompt, ConversationMessage as CoreMessage,
    ConversationStateManager as CoreManager, ConversationStateManagerConfig, Role, SweepConfig,
    SweepConversationStateManager as CoreSweepManager, SweepConversationStateManagerConfig,
    SweepHistoryCenterMode, SweepModelEdit as CoreSweepModelEdit,
    SweepModelEditKind as CoreSweepModelEditKind, SweepOpenedFileContextMode,
    SweepRuntimePrompt as CoreSweepRuntimePrompt, Tokenizer,
};

/// A message in the conversation.
#[napi(object)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

impl From<CoreMessage> for ConversationMessage {
    fn from(msg: CoreMessage) -> Self {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        Self {
            role: role.to_string(),
            content: msg.content,
        }
    }
}

/// Sweep YAML conversion options.
#[napi(object)]
pub struct SweepConversionOptions {
    /// Fixed viewport lines for target/history windows.
    pub viewport_lines: Option<u32>,
    /// Opened-file context mode: `full` or `viewport`.
    pub opened_file_context: Option<String>,
    /// History centering mode for `*.diff` windows: `changed` or `cursor`.
    pub history_center: Option<String>,
    /// Hard max tokens per conversation. History is trimmed first to fit.
    pub max_tokens_per_conversation: Option<u32>,
    /// Optional custom system prompt. Uses Sweep default when omitted.
    pub system_prompt: Option<String>,
}

/// Sweep conversation output.
#[napi(object)]
pub struct SweepConversation {
    pub messages: Vec<ConversationMessage>,
    pub token_count: u32,
    pub target_file: String,
}

/// Configuration options for the SweepConversationStateManager.
#[napi(object)]
pub struct SweepConversationStateManagerOptions {
    /// Fixed viewport line count for target/history windows.
    pub viewport_lines: Option<u32>,
    /// Opened-file context mode: `full` or `viewport`.
    pub opened_file_context: Option<String>,
    /// History centering mode for `*.diff` windows: `changed` or `cursor`.
    pub history_center: Option<String>,
    /// Coalesce radius (line-based) for grouping nearby edits.
    pub coalesce_radius: Option<u32>,
    /// Maximum number of history entries kept in context.
    pub max_history_entries: Option<u32>,
    /// Optional custom Sweep system prompt.
    pub system_prompt: Option<String>,
}

/// Runtime Sweep prompt payload.
#[napi(object)]
pub struct SweepRuntimePrompt {
    pub messages: Vec<ConversationMessage>,
    pub token_count: u32,
    pub target_file: String,
    pub window_start_line: u32,
    pub window_end_line: u32,
    pub current_window: String,
}

impl From<CoreSweepRuntimePrompt> for SweepRuntimePrompt {
    fn from(value: CoreSweepRuntimePrompt) -> Self {
        Self {
            messages: value.messages.into_iter().map(Into::into).collect(),
            token_count: value.token_count as u32,
            target_file: value.target_file,
            window_start_line: value.window_start_line as u32,
            window_end_line: value.window_end_line as u32,
            current_window: value.current_window,
        }
    }
}

/// Parsed Sweep model edit.
#[napi(object)]
pub struct SweepModelEdit {
    pub target_file: String,
    /// One of: `insert`, `delete`, `replace`.
    pub kind: String,
    /// 1-based start line in target file.
    pub start_line: u32,
    /// 1-based end line for delete/replace (inclusive).
    pub end_line: Option<u32>,
    /// Replacement/insert payload when applicable.
    pub text: Option<String>,
}

impl From<CoreSweepModelEdit> for SweepModelEdit {
    fn from(value: CoreSweepModelEdit) -> Self {
        let kind = match value.kind {
            CoreSweepModelEditKind::Insert => "insert",
            CoreSweepModelEditKind::Delete => "delete",
            CoreSweepModelEditKind::Replace => "replace",
        };
        Self {
            target_file: value.target_file,
            kind: kind.to_string(),
            start_line: value.start_line as u32,
            end_line: value.end_line.map(|v| v as u32),
            text: value.text,
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
                viewport_radius: opts
                    .viewport_radius
                    .map(|v| v as usize)
                    .unwrap_or(defaults.viewport_radius),
                coalesce_radius: opts
                    .coalesce_radius
                    .map(|v| v as usize)
                    .unwrap_or(defaults.coalesce_radius),
                max_tokens_per_message: opts
                    .max_tokens_per_message
                    .map(|v| v as usize)
                    .unwrap_or(defaults.max_tokens_per_message),
                max_tokens_per_terminal_output: opts
                    .max_tokens_per_terminal_output
                    .map(|v| v as usize)
                    .unwrap_or(defaults.max_tokens_per_terminal_output),
                // Extension-specific: no chunking (single ongoing conversation)
                max_tokens_per_conversation: None,
                min_conversation_messages: defaults.min_conversation_messages,
                // Extension doesn't need chat template counting
                system_prompt: None,
                special_tokens_per_user_message: 0,
                special_tokens_per_assistant_message: 0,
                conversation_start_tokens: 0,
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
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.reset();
        Ok(())
    }

    /// Get a copy of all messages.
    #[napi]
    pub fn get_messages(&self) -> Result<Vec<ConversationMessage>> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        Ok(inner.get_messages().into_iter().map(Into::into).collect())
    }

    /// Get the current content of a file.
    #[napi]
    pub fn get_file_content(&self, file_path: String) -> Result<String> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        Ok(inner.get_file_content(&file_path))
    }

    /// Handle a tab (file switch) event.
    ///
    /// @param filePath - The path to the file.
    /// @param textContent - The file contents, or null if switching to an already-open file.
    #[napi]
    pub fn handle_tab_event(&self, file_path: String, text_content: Option<String>) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
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
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_content_event(&file_path, offset as usize, length as usize, &new_text);
        Ok(())
    }

    /// Handle a selection event.
    ///
    /// @param filePath - The path to the file.
    /// @param offset - The character offset of the selection start.
    #[napi]
    pub fn handle_selection_event(&self, file_path: String, offset: u32) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_selection_event(&file_path, offset as usize);
        Ok(())
    }

    /// Handle a terminal command event.
    ///
    /// @param command - The command that was executed.
    #[napi]
    pub fn handle_terminal_command_event(&self, command: String) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_terminal_command_event(&command);
        Ok(())
    }

    /// Handle a terminal output event.
    ///
    /// @param output - The terminal output.
    #[napi]
    pub fn handle_terminal_output_event(&self, output: String) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_terminal_output_event(&output);
        Ok(())
    }

    /// Handle a terminal focus event.
    #[napi]
    pub fn handle_terminal_focus_event(&self) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_terminal_focus_event();
        Ok(())
    }

    /// Handle a git branch checkout event.
    ///
    /// @param branchInfo - The git checkout message containing the branch name.
    #[napi]
    pub fn handle_git_branch_checkout_event(&self, branch_info: String) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_git_branch_checkout_event(&branch_info);
        Ok(())
    }

    /// Finalize and get conversation ready for model.
    #[napi]
    pub fn finalize_for_model(&self) -> Result<Vec<ConversationMessage>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        Ok(inner
            .finalize_for_model()
            .into_iter()
            .map(Into::into)
            .collect())
    }
}

/// Manages runtime Sweep state for next-edit prompting/parsing.
#[napi]
pub struct SweepConversationStateManager {
    inner: Mutex<CoreSweepManager<CharApproxTokenizer>>,
}

#[napi]
impl SweepConversationStateManager {
    /// Create a new SweepConversationStateManager.
    #[napi(constructor)]
    pub fn new(options: Option<SweepConversationStateManagerOptions>) -> Result<Self> {
        let mut config = SweepConversationStateManagerConfig::default();

        if let Some(opts) = options {
            if let Some(lines) = opts.viewport_lines {
                config.viewport_lines = lines as usize;
            }
            if let Some(radius) = opts.coalesce_radius {
                config.coalesce_radius = radius as usize;
            }
            if let Some(max_history) = opts.max_history_entries {
                config.max_history_entries = max_history as usize;
            }
            if let Some(prompt) = opts.system_prompt {
                config.system_prompt = prompt;
            }

            if let Some(mode) = opts.opened_file_context {
                config.opened_file_context_mode = parse_opened_file_context_mode(&mode)?;
            }
            if let Some(mode) = opts.history_center {
                config.history_center_mode = parse_history_center_mode(&mode)?;
            }
        }

        Ok(Self {
            inner: Mutex::new(CoreSweepManager::new(CharApproxTokenizer, config)),
        })
    }

    /// Reset all Sweep runtime state.
    #[napi]
    pub fn reset(&self) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.reset();
        Ok(())
    }

    /// Get current content snapshot for a file.
    #[napi]
    pub fn get_file_content(&self, file_path: String) -> Result<String> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        Ok(inner.get_file_content(&file_path))
    }

    /// Handle a tab/file switch event.
    #[napi]
    pub fn handle_tab_event(&self, file_path: String, text_content: Option<String>) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_tab_event(&file_path, text_content.as_deref());
        Ok(())
    }

    /// Handle a content change event.
    #[napi]
    pub fn handle_content_event(
        &self,
        file_path: String,
        offset: u32,
        length: u32,
        new_text: String,
    ) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_content_event(&file_path, offset as usize, length as usize, &new_text);
        Ok(())
    }

    /// Handle a selection event.
    #[napi]
    pub fn handle_selection_event(&self, file_path: String, offset: u32) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_selection_event(&file_path, offset as usize);
        Ok(())
    }

    /// Handle explicit 1-based cursor line updates.
    #[napi]
    pub fn handle_cursor_by_line(&self, file_path: String, line: u32) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_cursor_by_line(&file_path, line as usize);
        Ok(())
    }

    #[napi]
    pub fn handle_terminal_command_event(&self, command: String) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_terminal_command_event(&command);
        Ok(())
    }

    #[napi]
    pub fn handle_terminal_output_event(&self, output: String) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_terminal_output_event(&output);
        Ok(())
    }

    #[napi]
    pub fn handle_terminal_focus_event(&self) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_terminal_focus_event();
        Ok(())
    }

    #[napi]
    pub fn handle_git_branch_checkout_event(&self, branch_info: String) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        inner.handle_git_branch_checkout_event(&branch_info);
        Ok(())
    }

    /// Finalize current state into a Sweep prompt payload for the model.
    #[napi]
    pub fn finalize_for_model(&self) -> Result<SweepRuntimePrompt> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        Ok(inner.finalize_for_model().into())
    }

    /// Parse model output from the last finalized prompt into a line-based edit.
    #[napi]
    pub fn parse_model_response(&self, response: String) -> Result<Option<SweepModelEdit>> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| Error::from_reason("Lock poisoned"))?;
        let parsed = inner
            .parse_model_response(&response)
            .map_err(Error::from_reason)?;
        Ok(parsed.map(Into::into))
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

/// Get the default system prompt for Sweep format.
#[napi]
pub fn get_default_sweep_system_prompt() -> String {
    core_sweep_system_prompt()
}

/// Convert YAML content to Sweep-format conversations.
#[napi]
pub fn convert_yaml_to_sweep(
    yaml_content: String,
    options: Option<SweepConversionOptions>,
) -> Result<Vec<SweepConversation>> {
    let mut config = SweepConfig::default();

    if let Some(opts) = options {
        if let Some(lines) = opts.viewport_lines {
            config.viewport_lines = lines as usize;
        }

        if let Some(mode) = opts.opened_file_context {
            config.opened_file_context_mode = parse_opened_file_context_mode(&mode)?;
        }

        if let Some(mode) = opts.history_center {
            config.history_center_mode = parse_history_center_mode(&mode)?;
        }

        if let Some(max_tokens) = opts.max_tokens_per_conversation {
            config.max_tokens_per_conversation = Some(max_tokens as usize);
        }

        if let Some(prompt) = opts.system_prompt {
            config.system_prompt = prompt;
        }
    }

    let conversations = core_convert_yaml_to_sweep(&yaml_content, &CharApproxTokenizer, &config)
        .map_err(Error::from_reason)?;

    Ok(conversations
        .into_iter()
        .map(|conv| SweepConversation {
            messages: conv.messages.into_iter().map(Into::into).collect(),
            token_count: conv.token_count as u32,
            target_file: conv.target_file,
        })
        .collect())
}

fn parse_opened_file_context_mode(mode: &str) -> Result<SweepOpenedFileContextMode> {
    match mode.to_lowercase().as_str() {
        "full" => Ok(SweepOpenedFileContextMode::Full),
        "viewport" => Ok(SweepOpenedFileContextMode::Viewport),
        other => Err(Error::from_reason(format!(
            "Invalid opened_file_context '{}'. Supported: full, viewport",
            other
        ))),
    }
}

fn parse_history_center_mode(mode: &str) -> Result<SweepHistoryCenterMode> {
    match mode.to_lowercase().as_str() {
        "changed" => Ok(SweepHistoryCenterMode::ChangedBlock),
        "cursor" => Ok(SweepHistoryCenterMode::Cursor),
        other => Err(Error::from_reason(format!(
            "Invalid history_center '{}'. Supported: changed, cursor",
            other
        ))),
    }
}
