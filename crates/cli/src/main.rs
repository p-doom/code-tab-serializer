//! CLI tool for serializing crowd-pilot IDE interaction data.
//!
//! This tool processes CSV session files and outputs JSONL format suitable for
//! NeMo SFT training. It uses the HuggingFace tokenizers Rust library for
//! accurate token counting.

use std::path::PathBuf;

use clap::Parser;
use tokenizers::Tokenizer as HfTokenizer;

use crowd_pilot_serializer_core::{
    default_system_prompt,
    pipeline::{PipelineConfig, PipelineResult},
    process_all_sessions, write_jsonl_output, Tokenizer,
};

/// Serialize crowd-pilot CSV sessions to NeMo JSONL format.
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
        
        let truncated_ids: Vec<u32> = ids[..max_tokens].to_vec();
        self.inner
            .decode(&truncated_ids, true)
            .expect("Failed to decode truncated tokens")
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading tokenizer from {}...", args.tokenizer);
    let tokenizer = RustTokenizer::load(&args.tokenizer)?;

    let config = PipelineConfig {
        max_tokens_per_conversation: args.max_tokens_per_conversation,
        max_tokens_per_message: args.max_tokens_per_message,
        min_conversation_messages: args.min_conversation_messages,
        viewport_radius: args.viewport_radius,
        coalesce_radius: args.coalesce_radius,
        val_ratio: args.val_ratio,
    };

    println!("Processing CSV files from {:?}...", args.csv_root);
    let session_results = process_all_sessions(
        &args.csv_root,
        &tokenizer,
        &config,
    )?;

    let total_sessions = session_results.len();
    println!("Processed {} sessions", total_sessions);

    let default_prompt = default_system_prompt(args.viewport_radius);
    let system_prompt = args.system_prompt.as_deref().unwrap_or(&default_prompt);

    println!("Writing output to {:?}...", args.output_dir);
    let result: PipelineResult = write_jsonl_output(
        session_results,
        &args.output_dir,
        args.val_ratio,
        system_prompt,
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
