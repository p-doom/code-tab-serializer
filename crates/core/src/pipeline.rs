//! Pipeline for processing CSV sessions into conversations.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::conversation::{ConversationStateManager, ConversationStateManagerConfig, FinalizedConversation, Role};
use crate::Tokenizer;

/// A row from the CSV file.
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

/// Configuration for the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub max_tokens_per_conversation: usize,
    pub max_tokens_per_message: usize,
    pub min_conversation_messages: usize,
    pub viewport_radius: usize,
    pub coalesce_radius: usize,
    pub val_ratio: f64,
    pub system_prompt: Option<String>,
    /// Special tokens added per user/system message by the chat template.
    pub special_tokens_per_user_message: usize,
    /// Special tokens added per assistant message by the chat template.
    pub special_tokens_per_assistant_message: usize,
    /// Fixed special tokens at conversation start.
    pub conversation_start_tokens: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_conversation: 8192,
            max_tokens_per_message: 2048,
            min_conversation_messages: 5,
            viewport_radius: 10,
            coalesce_radius: 5,
            val_ratio: 0.1,
            system_prompt: None,
            special_tokens_per_user_message: 0,
            special_tokens_per_assistant_message: 0,
            conversation_start_tokens: 0,
        }
    }
}

/// Result of processing a single session.
#[derive(Debug)]
pub struct SessionResult {
    pub conversations: Vec<FinalizedConversation>,
    pub source_path: String,
}

/// Result of processing all sessions.
#[derive(Debug, Serialize)]
pub struct PipelineResult {
    pub total_sessions: usize,
    pub total_conversations: usize,
    pub train_conversations: usize,
    pub val_conversations: usize,
    pub total_messages: usize,
    pub total_tokens: usize,
}

/// Miles conversation record format.
#[derive(Debug, Serialize)]
pub struct MilesRecord {
    pub messages: Vec<MilesMessage>,
}

/// A message in Miles format.
#[derive(Debug, Serialize)]
pub struct MilesMessage {
    pub role: Role,
    pub content: String,
}

/// Discover all CSV files in a directory.
pub fn discover_csv_files(root: &Path) -> Vec<std::path::PathBuf> {
    let mut paths: Vec<std::path::PathBuf> = WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "csv"))
        .map(|e| e.path().to_path_buf())
        .collect();
    paths.sort();
    paths
}

/// Process a single CSV session file.
pub fn process_session<T>(
    csv_path: &Path,
    tokenizer: &T,
    config: &PipelineConfig,
) -> Result<Vec<FinalizedConversation>, Box<dyn std::error::Error>>
where
    T: Tokenizer,
{
    let manager_config = ConversationStateManagerConfig {
        viewport_radius: config.viewport_radius,
        coalesce_radius: config.coalesce_radius,
        max_tokens_per_message: config.max_tokens_per_message,
        max_tokens_per_terminal_output: 256,
        max_tokens_per_conversation: Some(config.max_tokens_per_conversation),
        min_conversation_messages: config.min_conversation_messages,
        system_prompt: config.system_prompt.clone(),
        special_tokens_per_user_message: config.special_tokens_per_user_message,
        special_tokens_per_assistant_message: config.special_tokens_per_assistant_message,
        conversation_start_tokens: config.conversation_start_tokens,
    };

    let mut manager = ConversationStateManager::new(tokenizer, manager_config);

    let mut reader = csv::Reader::from_path(csv_path)?;
    
    for result in reader.deserialize() {
        let row: CsvRow = result?;
        
        match row.event_type.as_str() {
            "tab" => {
                manager.handle_tab_event(&row.file, row.text.as_deref());
            }
            "content" => {
                let offset = row.range_offset.expect("content event missing RangeOffset") as usize;
                let length = row.range_length.expect("content event missing RangeLength") as usize;
                let text = row.text.as_deref().unwrap_or("");
                manager.handle_content_event(&row.file, offset, length, text);
            }
            "selection_command" | "selection_mouse" | "selection_keyboard" => {
                let offset = row.range_offset.expect("selection event missing RangeOffset") as usize;
                manager.handle_selection_event(&row.file, offset);
            }
            "terminal_command" => {
                let command = row.text.as_deref().unwrap_or_else(|| {
                    eprintln!("Warning: terminal_command event missing Text in {:?}", csv_path);
                    ""
                });
                manager.handle_terminal_command_event(command);
            }
            "terminal_output" => {
                let output = row.text.as_deref().unwrap_or_else(|| {
                    eprintln!("Warning: terminal_output event missing Text in {:?}", csv_path);
                    ""
                });
                manager.handle_terminal_output_event(output);
            }
            "terminal_focus" => {
                manager.handle_terminal_focus_event();
            }
            "git_branch_checkout" => {
                let branch_info = row.text.as_deref().unwrap_or_else(|| {
                    eprintln!("Warning: git_branch_checkout event missing Text in {:?}", csv_path);
                    ""
                });
                manager.handle_git_branch_checkout_event(branch_info);
            }
            other => {
                eprintln!("Warning: Unknown event type '{}' in {:?}", other, csv_path);
            }
        }
    }

    Ok(manager.get_conversations())
}

/// Process all CSV sessions in a directory in parallel.
///
/// Uses rayon for parallel processing. The tokenizer must be `Sync + Send`
/// to be shared across threads.
pub fn process_all_sessions<T>(
    csv_root: &Path,
    tokenizer: &T,
    config: &PipelineConfig,
) -> Result<Vec<SessionResult>, Box<dyn std::error::Error>>
where
    T: Tokenizer + Sync + Send,
{
    let csv_files = discover_csv_files(csv_root);

    if csv_files.is_empty() {
        return Err(format!("No CSV files found under {:?}", csv_root).into());
    }

    let total_files = csv_files.len();
    let processed_count = AtomicUsize::new(0);
    let error_count = AtomicUsize::new(0);

    let results: Vec<SessionResult> = csv_files
        .into_par_iter()
        .filter_map(|csv_path| {
            let result = process_session(&csv_path, tokenizer, config);
            let count = processed_count.fetch_add(1, Ordering::Relaxed) + 1;

            match result {
                Ok(conversations) => {
                    if count % 100 == 0 || count == total_files {
                        eprintln!("Processed {}/{} sessions...", count, total_files);
                    }
                    Some(SessionResult {
                        conversations,
                        source_path: csv_path.to_string_lossy().to_string(),
                    })
                }
                Err(e) => {
                    error_count.fetch_add(1, Ordering::Relaxed);
                    eprintln!("Error processing {:?}: {}", csv_path, e);
                    None
                }
            }
        })
        .collect();

    let errors = error_count.load(Ordering::Relaxed);
    if errors > 0 {
        eprintln!("Warning: {} sessions failed to process", errors);
    }

    Ok(results)
}

/// Write conversations to JSONL files (training and validation).
pub fn write_jsonl_output(
    session_results: Vec<SessionResult>,
    output_dir: &Path,
    val_ratio: f64,
    system_prompt: &str,
) -> Result<PipelineResult, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    std::fs::create_dir_all(output_dir)?;

    // Shuffle sessions for train/val split (using simple deterministic shuffle)
    let mut sessions: Vec<_> = session_results.into_iter().enumerate().collect();
    // Simple deterministic shuffle based on index
    sessions.sort_by(|(i, a), (j, b)| {
        let hash_a = (i * 2654435761) % 1000;
        let hash_b = (j * 2654435761) % 1000;
        hash_a.cmp(&hash_b).then_with(|| a.source_path.cmp(&b.source_path))
    });

    let total_sessions = sessions.len();
    let val_count = (total_sessions as f64 * val_ratio).round() as usize;
    let train_count = total_sessions - val_count;

    let train_path = output_dir.join("training.jsonl");
    let val_path = output_dir.join("validation.jsonl");

    let mut train_file = BufWriter::new(File::create(&train_path)?);
    let mut val_file = BufWriter::new(File::create(&val_path)?);

    let mut train_conversations = 0;
    let mut val_conversations = 0;
    let mut total_messages = 0;
    let mut total_tokens = 0;

    for (idx, (_, session)) in sessions.into_iter().enumerate() {
        let is_validation = idx >= train_count;
        
        for conv in session.conversations {
            let mut messages = vec![MilesMessage {
                role: Role::System,
                content: system_prompt.to_string(),
            }];
            
            messages.extend(conv.messages.iter().map(|m| MilesMessage {
                role: m.role,
                content: m.content.clone(),
            }));

            let record = MilesRecord { messages };
            let json_line = serde_json::to_string(&record)?;
            
            if is_validation {
                writeln!(val_file, "{}", json_line)?;
                val_conversations += 1;
            } else {
                writeln!(train_file, "{}", json_line)?;
                train_conversations += 1;
            }

            total_messages += conv.messages.len();
            total_tokens += conv.token_count;
        }
    }

    train_file.flush()?;
    val_file.flush()?;

    Ok(PipelineResult {
        total_sessions,
        total_conversations: train_conversations + val_conversations,
        train_conversations,
        val_conversations,
        total_messages,
        total_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    /// Character-based approximate tokenizer for tests.
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
    fn test_discover_csv_files() {
        let temp = TempDir::new().unwrap();
        let csv1 = temp.path().join("session1.csv");
        let csv2 = temp.path().join("subdir/session2.csv");
        
        std::fs::create_dir_all(temp.path().join("subdir")).unwrap();
        std::fs::write(&csv1, "header\n").unwrap();
        std::fs::write(&csv2, "header\n").unwrap();

        let files = discover_csv_files(temp.path());
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_process_session() {
        let temp = TempDir::new().unwrap();
        let csv_path = temp.path().join("test.csv");
        
        let mut file = std::fs::File::create(&csv_path).unwrap();
        writeln!(file, "Sequence,Time,File,RangeOffset,RangeLength,Text,Language,Type").unwrap();
        writeln!(file, "1,2024-01-01,/test/file.rs,0,0,\"fn main() {{}}\",rust,tab").unwrap();
        writeln!(file, "2,2024-01-01,/test/file.rs,0,0,echo hello,bash,terminal_command").unwrap();
        writeln!(file, "3,2024-01-01,/test/file.rs,0,0,hello world,bash,terminal_output").unwrap();

        let config = PipelineConfig {
            min_conversation_messages: 2,
            ..Default::default()
        };

        let tokenizer = CharApproxTokenizer;
        let conversations = process_session(&csv_path, &tokenizer, &config).unwrap();
        
        // Should have at least one conversation with messages
        assert!(!conversations.is_empty() || conversations.iter().any(|c| !c.messages.is_empty()));
    }
}

