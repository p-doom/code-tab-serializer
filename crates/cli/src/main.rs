//! CLI tool for serializing crowd-pilot IDE interaction data.
//!
//! This tool processes CSV session files and outputs JSONL format for:
//! - SED command-prediction training
//! - Zeta edit-prediction training
//! - Sweep next-edit training

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use tokenizers::Tokenizer as HfTokenizer;

use crowd_pilot_serializer_core::{
    convert_csv_to_sweep_session, convert_csv_to_zeta_session, default_system_prompt,
    discover_csv_files,
    pipeline::{MilesMessage, MilesRecord, PipelineConfig, PipelineResult},
    process_all_sessions, sweep_system_prompt, write_jsonl_output, SweepConfig, SweepConversation,
    SweepHistoryCenterMode, SweepOpenedFileContextMode, Tokenizer, ZetaConfig, ZetaConversation,
};

/// Special token counts for known chat templates: (user/system, assistant, conversation_start)
fn chat_template_overhead(name: &str) -> Option<(usize, usize, usize)> {
    match name.to_lowercase().as_str() {
        // Qwen3: <|im_start|>role\ncontent<|im_end|> per message
        // Assistant messages get extra <think> block (4 tokens)
        "qwen3" => Some((5, 9, 0)),
        // GLM-4.5: [gMASK]<sop> at start (2 tokens), then <|role|>\n per message (2 tokens)
        // Assistant messages get extra <think></think>\n (3 tokens)
        "glm45" => Some((2, 5, 2)),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    Sed,
    Zeta,
    Sweep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SweepOpenedFileContextArg {
    Full,
    Viewport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SweepHistoryCenterArg {
    Changed,
    Cursor,
}

/// Serialize crowd-pilot CSV sessions to JSONL format.
#[derive(Parser, Debug)]
#[command(name = "crowd-pilot-serialize")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Output format
    #[arg(long, value_enum, default_value_t = OutputFormat::Sed)]
    output_format: OutputFormat,

    /// Root directory containing CSV session files
    #[arg(long)]
    csv_root: PathBuf,

    /// Output directory for JSONL files
    #[arg(long)]
    output_dir: PathBuf,

    /// HuggingFace tokenizer model name or path
    #[arg(long)]
    tokenizer: String,

    /// Chat template format for accurate token counting (required).
    /// Supported: qwen3, glm45
    #[arg(long)]
    chat_template: Option<String>,

    /// Maximum tokens per conversation chunk (SED/Sweep)
    #[arg(long, default_value = "8192")]
    max_tokens_per_conversation: usize,

    /// Maximum tokens per message (SED only)
    #[arg(long, default_value = "2048")]
    max_tokens_per_message: usize,

    /// Minimum messages required to keep a conversation (SED only)
    #[arg(long, default_value = "5")]
    min_conversation_messages: usize,

    /// Viewport radius (SED only)
    #[arg(long, default_value = "10")]
    viewport_radius: usize,

    /// Coalesce radius for grouping nearby edits (shared)
    #[arg(long, default_value = "5")]
    coalesce_radius: usize,

    /// Fraction of sessions for validation (0.0-1.0)
    #[arg(long, default_value = "0.1")]
    val_ratio: f64,

    /// Custom system prompt (SED/Sweep only)
    #[arg(long)]
    system_prompt: Option<String>,

    /// Zeta max editable tokens
    #[arg(long, default_value = "180")]
    zeta_max_editable_tokens: usize,

    /// Zeta max context tokens
    #[arg(long, default_value = "350")]
    zeta_max_context_tokens: usize,

    /// Zeta unified diff context lines
    #[arg(long, default_value = "3")]
    zeta_diff_context_lines: usize,

    /// Sweep fixed viewport lines
    #[arg(long, default_value = "21")]
    sweep_viewport_lines: usize,

    /// Sweep opened-file context mode
    #[arg(long, value_enum, default_value_t = SweepOpenedFileContextArg::Full)]
    sweep_opened_file_context: SweepOpenedFileContextArg,

    /// Sweep history window centering mode
    #[arg(long, value_enum, default_value_t = SweepHistoryCenterArg::Changed)]
    sweep_history_center: SweepHistoryCenterArg,
}

/// Wrapper around HuggingFace tokenizers for token counting and truncation.
///
/// This uses the Rust-native tokenizers library, which is `Send + Sync`
/// and enables true parallel tokenization without the Python GIL.
struct RustTokenizer {
    inner: HfTokenizer,
}

impl RustTokenizer {
    /// Load a HuggingFace tokenizer from a model name or path.
    fn load(model_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = HfTokenizer::from_pretrained(model_name, None)
            .map_err(|e| e as Box<dyn std::error::Error>)?;

        Ok(Self { inner })
    }
}

impl Tokenizer for RustTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        self.inner
            .encode(text, false)
            .expect("Failed to encode text with tokenizer")
            .get_ids()
            .len()
    }

    fn truncate_to_max_tokens(&self, text: &str, max_tokens: usize) -> String {
        let encoding = self
            .inner
            .encode(text, false)
            .expect("Failed to encode text with tokenizer");

        let ids = encoding.get_ids();
        if ids.len() <= max_tokens {
            return text.to_string();
        }

        // Use offsets to slice original string precisely
        let offsets = encoding.get_offsets();
        let (_, end) = offsets[max_tokens - 1];
        text[..end].to_string()
    }
}

#[derive(Debug)]
struct ZetaSessionResult {
    conversations: Vec<ZetaConversation>,
    source_path: String,
}

#[derive(Debug)]
struct SweepSessionResult {
    conversations: Vec<SweepConversation>,
    source_path: String,
}

fn write_zeta_jsonl_output(
    session_results: Vec<ZetaSessionResult>,
    output_dir: &Path,
    val_ratio: f64,
) -> Result<PipelineResult, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;

    let mut sessions: Vec<_> = session_results.into_iter().enumerate().collect();
    sessions.sort_by(|(i, a), (j, b)| {
        let hash_a = (i * 2654435761) % 1000;
        let hash_b = (j * 2654435761) % 1000;
        hash_a
            .cmp(&hash_b)
            .then_with(|| a.source_path.cmp(&b.source_path))
    });

    let total_sessions = sessions.len();
    let val_count = (total_sessions as f64 * val_ratio).round() as usize;
    let train_count = total_sessions.saturating_sub(val_count);

    let train_path = output_dir.join("training.jsonl");
    let val_path = output_dir.join("validation.jsonl");

    let mut train_file = BufWriter::new(std::fs::File::create(&train_path)?);
    let mut val_file = BufWriter::new(std::fs::File::create(&val_path)?);

    let mut train_conversations = 0;
    let mut val_conversations = 0;
    let mut total_messages = 0;
    let mut total_tokens = 0;

    for (idx, (_, session)) in sessions.into_iter().enumerate() {
        let is_validation = idx >= train_count;
        for conv in session.conversations {
            let messages = conv
                .messages
                .iter()
                .map(|m| MilesMessage {
                    role: m.role,
                    content: m.content.clone(),
                })
                .collect::<Vec<_>>();

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

fn write_sweep_jsonl_output(
    session_results: Vec<SweepSessionResult>,
    output_dir: &Path,
    val_ratio: f64,
) -> Result<PipelineResult, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;

    let mut sessions: Vec<_> = session_results.into_iter().enumerate().collect();
    sessions.sort_by(|(i, a), (j, b)| {
        let hash_a = (i * 2654435761) % 1000;
        let hash_b = (j * 2654435761) % 1000;
        hash_a
            .cmp(&hash_b)
            .then_with(|| a.source_path.cmp(&b.source_path))
    });

    let total_sessions = sessions.len();
    let val_count = (total_sessions as f64 * val_ratio).round() as usize;
    let train_count = total_sessions.saturating_sub(val_count);

    let train_path = output_dir.join("training.jsonl");
    let val_path = output_dir.join("validation.jsonl");

    let mut train_file = BufWriter::new(std::fs::File::create(&train_path)?);
    let mut val_file = BufWriter::new(std::fs::File::create(&val_path)?);

    let mut train_conversations = 0;
    let mut val_conversations = 0;
    let mut total_messages = 0;
    let mut total_tokens = 0;

    for (idx, (_, session)) in sessions.into_iter().enumerate() {
        let is_validation = idx >= train_count;
        for conv in session.conversations {
            let messages = conv
                .messages
                .iter()
                .map(|m| MilesMessage {
                    role: m.role,
                    content: m.content.clone(),
                })
                .collect::<Vec<_>>();

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

fn run_sed(
    args: &Args,
    tokenizer: &RustTokenizer,
) -> Result<(PipelineResult, serde_json::Value), Box<dyn std::error::Error>> {
    let chat_template = args
        .chat_template
        .as_deref()
        .ok_or_else(|| "--chat-template is required when --output-format sed".to_string())?;

    let (special_tokens_per_user, special_tokens_per_assistant, conversation_start_tokens) =
        chat_template_overhead(chat_template).ok_or_else(|| {
            format!(
                "Unknown chat template: '{}'. Supported: qwen3, glm45",
                chat_template
            )
        })?;

    println!("  Chat template: {}", chat_template);
    println!(
        "  Per user/system message overhead: {} tokens",
        special_tokens_per_user
    );
    println!(
        "  Per assistant message overhead: {} tokens",
        special_tokens_per_assistant
    );
    println!(
        "  Conversation start overhead: {} tokens",
        conversation_start_tokens
    );

    let default_prompt = default_system_prompt(args.viewport_radius);
    let system_prompt = args
        .system_prompt
        .clone()
        .unwrap_or_else(|| default_prompt.clone());

    let config = PipelineConfig {
        max_tokens_per_conversation: args.max_tokens_per_conversation,
        max_tokens_per_message: args.max_tokens_per_message,
        min_conversation_messages: args.min_conversation_messages,
        viewport_radius: args.viewport_radius,
        coalesce_radius: args.coalesce_radius,
        val_ratio: args.val_ratio,
        system_prompt: Some(system_prompt.clone()),
        special_tokens_per_user_message: special_tokens_per_user,
        special_tokens_per_assistant_message: special_tokens_per_assistant,
        conversation_start_tokens,
    };

    println!("Processing CSV files from {:?}...", args.csv_root);
    let session_results = process_all_sessions(&args.csv_root, tokenizer, &config)?;
    println!("Processed {} sessions", session_results.len());

    println!("Writing output to {:?}...", args.output_dir);
    let result = write_jsonl_output(
        session_results,
        &args.output_dir,
        args.val_ratio,
        &system_prompt,
    )?;

    let metadata = serde_json::json!({
        "mode": "sed",
        "config": {
            "csv_root": args.csv_root.to_string_lossy(),
            "output_dir": args.output_dir.to_string_lossy(),
            "tokenizer": args.tokenizer,
            "chat_template": chat_template,
            "max_tokens_per_conversation": args.max_tokens_per_conversation,
            "max_tokens_per_message": args.max_tokens_per_message,
            "min_conversation_messages": args.min_conversation_messages,
            "viewport_radius": args.viewport_radius,
            "coalesce_radius": args.coalesce_radius,
            "val_ratio": args.val_ratio,
        },
    });

    Ok((result, metadata))
}

fn run_zeta(
    args: &Args,
    tokenizer: &RustTokenizer,
) -> Result<(PipelineResult, serde_json::Value), Box<dyn std::error::Error>> {
    let chat_template = args
        .chat_template
        .as_deref()
        .ok_or_else(|| "--chat-template is required when --output-format zeta".to_string())?;

    let (special_tokens_per_user, special_tokens_per_assistant, conversation_start_tokens) =
        chat_template_overhead(chat_template).ok_or_else(|| {
            format!(
                "Unknown chat template: '{}'. Supported: qwen3, glm45",
                chat_template
            )
        })?;

    println!("  Chat template: {}", chat_template);
    println!(
        "  Per user/system message overhead: {} tokens",
        special_tokens_per_user
    );
    println!(
        "  Per assistant message overhead: {} tokens",
        special_tokens_per_assistant
    );
    println!(
        "  Conversation start overhead: {} tokens",
        conversation_start_tokens
    );

    if args.system_prompt.is_some() {
        eprintln!("Warning: --system-prompt is ignored for --output-format zeta");
    }

    let zeta_config = ZetaConfig {
        max_editable_tokens: args.zeta_max_editable_tokens,
        max_context_tokens: args.zeta_max_context_tokens,
        diff_context_lines: args.zeta_diff_context_lines,
        special_tokens_per_user_message: special_tokens_per_user,
        special_tokens_per_assistant_message: special_tokens_per_assistant,
        conversation_start_tokens,
    };

    println!("Processing CSV files from {:?}...", args.csv_root);
    let csv_files = discover_csv_files(&args.csv_root);
    if csv_files.is_empty() {
        return Err(format!("No CSV files found under {:?}", args.csv_root).into());
    }

    let total_files = csv_files.len();
    let mut session_results = Vec::with_capacity(total_files);
    let mut error_count = 0usize;
    for (idx, csv_path) in csv_files.into_iter().enumerate() {
        let conversations = match convert_csv_to_zeta_session(
            &csv_path,
            tokenizer,
            args.coalesce_radius,
            &zeta_config,
        ) {
            Ok(conversations) => conversations,
            Err(e) => {
                error_count += 1;
                eprintln!("Warning: Error processing {:?}: {}", csv_path, e);
                continue;
            }
        };

        if (idx + 1) % 100 == 0 || idx + 1 == total_files {
            eprintln!("Processed {}/{} sessions...", idx + 1, total_files);
        }

        session_results.push(ZetaSessionResult {
            conversations,
            source_path: csv_path.to_string_lossy().to_string(),
        });
    }

    if error_count > 0 {
        eprintln!("Warning: {} sessions failed to process", error_count);
    }

    println!("Writing output to {:?}...", args.output_dir);
    let result = write_zeta_jsonl_output(session_results, &args.output_dir, args.val_ratio)?;

    let metadata = serde_json::json!({
        "mode": "zeta",
        "config": {
            "csv_root": args.csv_root.to_string_lossy(),
            "output_dir": args.output_dir.to_string_lossy(),
            "tokenizer": args.tokenizer,
            "chat_template": chat_template,
            "special_tokens_per_user_message": special_tokens_per_user,
            "special_tokens_per_assistant_message": special_tokens_per_assistant,
            "conversation_start_tokens": conversation_start_tokens,
            "coalesce_radius": args.coalesce_radius,
            "val_ratio": args.val_ratio,
            "zeta_max_editable_tokens": args.zeta_max_editable_tokens,
            "zeta_max_context_tokens": args.zeta_max_context_tokens,
            "zeta_diff_context_lines": args.zeta_diff_context_lines,
        },
    });

    Ok((result, metadata))
}

fn run_sweep(
    args: &Args,
    tokenizer: &RustTokenizer,
) -> Result<(PipelineResult, serde_json::Value), Box<dyn std::error::Error>> {
    let chat_template = args
        .chat_template
        .as_deref()
        .ok_or_else(|| "--chat-template is required when --output-format sweep".to_string())?;

    let (special_tokens_per_user, special_tokens_per_assistant, conversation_start_tokens) =
        chat_template_overhead(chat_template).ok_or_else(|| {
            format!(
                "Unknown chat template: '{}'. Supported: qwen3, glm45",
                chat_template
            )
        })?;

    println!("  Chat template: {}", chat_template);
    println!(
        "  Per user/system message overhead: {} tokens",
        special_tokens_per_user
    );
    println!(
        "  Per assistant message overhead: {} tokens",
        special_tokens_per_assistant
    );
    println!(
        "  Conversation start overhead: {} tokens",
        conversation_start_tokens
    );

    let opened_file_context_mode = match args.sweep_opened_file_context {
        SweepOpenedFileContextArg::Full => SweepOpenedFileContextMode::Full,
        SweepOpenedFileContextArg::Viewport => SweepOpenedFileContextMode::Viewport,
    };
    let history_center_mode = match args.sweep_history_center {
        SweepHistoryCenterArg::Changed => SweepHistoryCenterMode::ChangedBlock,
        SweepHistoryCenterArg::Cursor => SweepHistoryCenterMode::Cursor,
    };

    let default_prompt = sweep_system_prompt();
    let system_prompt = args
        .system_prompt
        .clone()
        .unwrap_or_else(|| default_prompt.clone());

    let sweep_config = SweepConfig {
        viewport_lines: args.sweep_viewport_lines,
        opened_file_context_mode,
        history_center_mode,
        max_tokens_per_conversation: Some(args.max_tokens_per_conversation),
        system_prompt: system_prompt.clone(),
        special_tokens_per_user_message: special_tokens_per_user,
        special_tokens_per_assistant_message: special_tokens_per_assistant,
        conversation_start_tokens,
    };

    println!("Processing CSV files from {:?}...", args.csv_root);
    let csv_files = discover_csv_files(&args.csv_root);
    if csv_files.is_empty() {
        return Err(format!("No CSV files found under {:?}", args.csv_root).into());
    }

    let total_files = csv_files.len();
    let mut session_results = Vec::with_capacity(total_files);
    let mut error_count = 0usize;
    for (idx, csv_path) in csv_files.into_iter().enumerate() {
        let conversations = match convert_csv_to_sweep_session(
            &csv_path,
            tokenizer,
            args.coalesce_radius,
            &sweep_config,
        ) {
            Ok(conversations) => conversations,
            Err(e) => {
                error_count += 1;
                eprintln!("Warning: Error processing {:?}: {}", csv_path, e);
                continue;
            }
        };

        if (idx + 1) % 100 == 0 || idx + 1 == total_files {
            eprintln!("Processed {}/{} sessions...", idx + 1, total_files);
        }

        session_results.push(SweepSessionResult {
            conversations,
            source_path: csv_path.to_string_lossy().to_string(),
        });
    }

    if error_count > 0 {
        eprintln!("Warning: {} sessions failed to process", error_count);
    }

    println!("Writing output to {:?}...", args.output_dir);
    let result = write_sweep_jsonl_output(session_results, &args.output_dir, args.val_ratio)?;

    let metadata = serde_json::json!({
        "mode": "sweep",
        "config": {
            "csv_root": args.csv_root.to_string_lossy(),
            "output_dir": args.output_dir.to_string_lossy(),
            "tokenizer": args.tokenizer,
            "chat_template": chat_template,
            "system_prompt": system_prompt,
            "special_tokens_per_user_message": special_tokens_per_user,
            "special_tokens_per_assistant_message": special_tokens_per_assistant,
            "conversation_start_tokens": conversation_start_tokens,
            "max_tokens_per_conversation": args.max_tokens_per_conversation,
            "coalesce_radius": args.coalesce_radius,
            "val_ratio": args.val_ratio,
            "sweep_viewport_lines": args.sweep_viewport_lines,
            "sweep_opened_file_context": match args.sweep_opened_file_context {
                SweepOpenedFileContextArg::Full => "full",
                SweepOpenedFileContextArg::Viewport => "viewport",
            },
            "sweep_history_center": match args.sweep_history_center {
                SweepHistoryCenterArg::Changed => "changed",
                SweepHistoryCenterArg::Cursor => "cursor",
            },
        },
    });

    Ok((result, metadata))
}

fn write_metadata(
    args: &Args,
    result: &PipelineResult,
    extra: serde_json::Value,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let metadata_path = args.output_dir.join("metadata.json");
    let metadata = serde_json::json!({
        "config": extra["config"],
        "mode": extra["mode"],
        "counts": {
            "total_sessions": result.total_sessions,
            "total_conversations": result.total_conversations,
            "train_conversations": result.train_conversations,
            "val_conversations": result.val_conversations,
        },
        "stats": {
            "total_messages": result.total_messages,
            "total_tokens": result.total_tokens,
            "avg_messages_per_conversation": if result.total_conversations > 0 {
                result.total_messages as f64 / result.total_conversations as f64
            } else {
                0.0
            },
            "avg_tokens_per_conversation": if result.total_conversations > 0 {
                result.total_tokens as f64 / result.total_conversations as f64
            } else {
                0.0
            },
        },
        "files": {
            "train_path": args.output_dir.join("training.jsonl").to_string_lossy(),
            "val_path": args.output_dir.join("validation.jsonl").to_string_lossy(),
        },
    });
    std::fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;
    Ok(metadata_path)
}

fn print_summary(output_dir: &Path, metadata_path: &Path, result: &PipelineResult) {
    println!("\n[summary]");
    println!("  Total sessions processed: {}", result.total_sessions);
    println!("  Train conversations: {}", result.train_conversations);
    println!("  Val conversations: {}", result.val_conversations);
    println!("  Total messages: {}", result.total_messages);
    println!("  Total tokens: {}", result.total_tokens);
    println!("  Output: {:?}/{{training,validation}}.jsonl", output_dir);
    println!("  Metadata: {:?}", metadata_path);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading tokenizer from {}...", args.tokenizer);
    let tokenizer = RustTokenizer::load(&args.tokenizer)?;

    let (result, metadata_extra) = match args.output_format {
        OutputFormat::Sed => run_sed(&args, &tokenizer)?,
        OutputFormat::Zeta => run_zeta(&args, &tokenizer)?,
        OutputFormat::Sweep => run_sweep(&args, &tokenizer)?,
    };

    let metadata_path = write_metadata(&args, &result, metadata_extra)?;
    print_summary(&args.output_dir, &metadata_path, &result);
    Ok(())
}
