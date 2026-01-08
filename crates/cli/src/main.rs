//! CLI tool for serializing crowd-pilot IDE interaction data.
//!
//! This tool processes CSV session files and outputs JSONL format suitable for
//! Miles SFT training. It uses the HuggingFace tokenizers Rust library for
//! accurate token counting.

use std::path::PathBuf;

use clap::Parser;
use tokenizers::Tokenizer as HfTokenizer;

use crowd_pilot_serializer_core::{
    default_system_prompt,
    pipeline::{PipelineConfig, PipelineResult},
    process_all_sessions, write_jsonl_output, Tokenizer,
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

/// Serialize crowd-pilot CSV sessions to Miles JSONL format.
#[derive(Parser, Debug)]
#[command(name = "crowd-pilot-serialize")]
#[command(author, version, about, long_about = None)]
struct Args {
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
    chat_template: String,

    /// Maximum tokens per conversation chunk
    #[arg(long, default_value = "8192")]
    max_tokens_per_conversation: usize,

    /// Maximum tokens per message
    #[arg(long, default_value = "2048")]
    max_tokens_per_message: usize,

    /// Minimum messages required to keep a conversation
    #[arg(long, default_value = "5")]
    min_conversation_messages: usize,

    /// Viewport radius (lines above/below cursor)
    #[arg(long, default_value = "10")]
    viewport_radius: usize,

    /// Coalesce radius for grouping nearby edits
    #[arg(long, default_value = "5")]
    coalesce_radius: usize,

    /// Fraction of sessions for validation (0.0-1.0)
    #[arg(long, default_value = "0.1")]
    val_ratio: f64,

    /// Custom system prompt (optional)
    #[arg(long)]
    system_prompt: Option<String>,
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
        let encoding = self.inner
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let (special_tokens_per_user, special_tokens_per_assistant, conversation_start_tokens) =
        chat_template_overhead(&args.chat_template).ok_or_else(|| {
            format!(
                "Unknown chat template: '{}'. Supported: qwen3, glm45",
                args.chat_template
            )
        })?;

    println!("Loading tokenizer from {}...", args.tokenizer);
    let tokenizer = RustTokenizer::load(&args.tokenizer)?;

    println!("  Chat template: {}", args.chat_template);
    println!("  Per user/system message overhead: {} tokens", special_tokens_per_user);
    println!("  Per assistant message overhead: {} tokens", special_tokens_per_assistant);
    println!("  Conversation start overhead: {} tokens", conversation_start_tokens);

    let default_prompt = default_system_prompt(args.viewport_radius);
    let system_prompt = args.system_prompt.clone().unwrap_or(default_prompt.clone());

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
    let session_results = process_all_sessions(
        &args.csv_root,
        &tokenizer,
        &config,
    )?;

    let total_sessions = session_results.len();
    println!("Processed {} sessions", total_sessions);

    println!("Writing output to {:?}...", args.output_dir);
    let result: PipelineResult = write_jsonl_output(
        session_results,
        &args.output_dir,
        args.val_ratio,
        &system_prompt,
    )?;

    let metadata_path = args.output_dir.join("metadata.json");
    let metadata = serde_json::json!({
        "config": {
            "csv_root": args.csv_root.to_string_lossy(),
            "output_dir": args.output_dir.to_string_lossy(),
            "tokenizer": args.tokenizer,
            "max_tokens_per_conversation": args.max_tokens_per_conversation,
            "max_tokens_per_message": args.max_tokens_per_message,
            "min_conversation_messages": args.min_conversation_messages,
            "viewport_radius": args.viewport_radius,
            "coalesce_radius": args.coalesce_radius,
            "val_ratio": args.val_ratio,
        },
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

    println!("\n[summary]");
    println!("  Total sessions processed: {}", result.total_sessions);
    println!("  Train conversations: {}", result.train_conversations);
    println!("  Val conversations: {}", result.val_conversations);
    println!("  Total messages: {}", result.total_messages);
    println!("  Total tokens: {}", result.total_tokens);
    println!("  Output: {:?}/{{training,validation}}.jsonl", args.output_dir);
    println!("  Metadata: {:?}", metadata_path);

    Ok(())
}
