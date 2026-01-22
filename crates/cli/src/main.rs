//! CLI tool for serializing crowd-pilot IDE interaction data.
//!
//! This tool processes CSV session files and outputs JSONL format suitable for
//! Miles SFT training. It uses the HuggingFace tokenizers Rust library for
//! accurate token counting.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tokenizers::Tokenizer as HfTokenizer;

use crowd_pilot_serializer_core::{
    convert_yaml_to_testcases, default_system_prompt,
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
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    // Legacy args for backwards compatibility (when no subcommand is used)
    /// Root directory containing CSV session files
    #[arg(long)]
    csv_root: Option<PathBuf>,

    /// Output directory for JSONL files
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// HuggingFace tokenizer model name or path
    #[arg(long)]
    tokenizer: Option<String>,

    /// Chat template format for accurate token counting (required).
    /// Supported: qwen3, glm45
    #[arg(long)]
    chat_template: Option<String>,

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

#[derive(Subcommand, Debug)]
enum Commands {
    /// Convert YAML state-based eval files directly to test cases JSONL
    YamlToTestcases {
        /// Input YAML file or directory containing YAML files
        #[arg(long)]
        input: PathBuf,

        /// Output JSONL file
        #[arg(long)]
        output: PathBuf,
    },
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

fn run_yaml_to_testcases(input: &PathBuf, output: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut all_test_cases = Vec::new();

    if input.is_dir() {
        let entries: Vec<_> = std::fs::read_dir(input)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "yaml" || ext == "yml")
                    .unwrap_or(false)
            })
            .collect();

        println!("Converting {} YAML files to test cases...", entries.len());

        for entry in entries {
            let yaml_path = entry.path();
            let yaml_content = std::fs::read_to_string(&yaml_path)?;
            let test_cases = convert_yaml_to_testcases(&yaml_content)
                .map_err(|e| format!("Error converting {:?}: {}", yaml_path, e))?;
            
            println!("  {} -> {} test cases", yaml_path.display(), test_cases.len());
            all_test_cases.extend(test_cases);
        }
    } else {
        // Process single file
        let yaml_content = std::fs::read_to_string(input)?;
        let test_cases = convert_yaml_to_testcases(&yaml_content)?;
        println!("Converted {} -> {} test cases", input.display(), test_cases.len());
        all_test_cases.extend(test_cases);
    }

    // Write all test cases to output JSONL file
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    let mut output_file = std::fs::File::create(output)?;
    use std::io::Write;
    for test_case in &all_test_cases {
        let json = serde_json::to_string(test_case)?;
        writeln!(output_file, "{}", json)?;
    }

    println!("Wrote {} test cases to {}", all_test_cases.len(), output.display());
    Ok(())
}

fn run_serialize(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let csv_root = cli
        .csv_root
        .as_ref()
        .ok_or("--csv-root is required for serialization")?;
    let output_dir = cli
        .output_dir
        .as_ref()
        .ok_or("--output-dir is required for serialization")?;
    let tokenizer_name = cli
        .tokenizer
        .as_ref()
        .ok_or("--tokenizer is required for serialization")?;
    let chat_template = cli
        .chat_template
        .as_ref()
        .ok_or("--chat-template is required for serialization")?;

    let (special_tokens_per_user, special_tokens_per_assistant, conversation_start_tokens) =
        chat_template_overhead(chat_template).ok_or_else(|| {
            format!(
                "Unknown chat template: '{}'. Supported: qwen3, glm45",
                chat_template
            )
        })?;

    println!("Loading tokenizer from {}...", tokenizer_name);
    let tokenizer = RustTokenizer::load(tokenizer_name)?;

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

    let default_prompt = default_system_prompt(cli.viewport_radius);
    let system_prompt = cli
        .system_prompt
        .clone()
        .unwrap_or_else(|| default_prompt.clone());

    let config = PipelineConfig {
        max_tokens_per_conversation: cli.max_tokens_per_conversation,
        max_tokens_per_message: cli.max_tokens_per_message,
        min_conversation_messages: cli.min_conversation_messages,
        viewport_radius: cli.viewport_radius,
        coalesce_radius: cli.coalesce_radius,
        val_ratio: cli.val_ratio,
        system_prompt: Some(system_prompt.clone()),
        special_tokens_per_user_message: special_tokens_per_user,
        special_tokens_per_assistant_message: special_tokens_per_assistant,
        conversation_start_tokens,
    };

    println!("Processing CSV files from {:?}...", csv_root);
    let session_results = process_all_sessions(csv_root, &tokenizer, &config)?;

    let total_sessions = session_results.len();
    println!("Processed {} sessions", total_sessions);

    println!("Writing output to {:?}...", output_dir);
    let result: PipelineResult =
        write_jsonl_output(session_results, output_dir, cli.val_ratio, &system_prompt)?;

    let metadata_path = output_dir.join("metadata.json");
    let metadata = serde_json::json!({
        "config": {
            "csv_root": csv_root.to_string_lossy(),
            "output_dir": output_dir.to_string_lossy(),
            "tokenizer": tokenizer_name,
            "max_tokens_per_conversation": cli.max_tokens_per_conversation,
            "max_tokens_per_message": cli.max_tokens_per_message,
            "min_conversation_messages": cli.min_conversation_messages,
            "viewport_radius": cli.viewport_radius,
            "coalesce_radius": cli.coalesce_radius,
            "val_ratio": cli.val_ratio,
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
            "train_path": output_dir.join("training.jsonl").to_string_lossy(),
            "val_path": output_dir.join("validation.jsonl").to_string_lossy(),
        },
    });
    std::fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;

    println!("\n[summary]");
    println!("  Total sessions processed: {}", result.total_sessions);
    println!("  Train conversations: {}", result.train_conversations);
    println!("  Val conversations: {}", result.val_conversations);
    println!("  Total messages: {}", result.total_messages);
    println!("  Total tokens: {}", result.total_tokens);
    println!(
        "  Output: {:?}/{{training,validation}}.jsonl",
        output_dir
    );
    println!("  Metadata: {:?}", metadata_path);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::YamlToTestcases { input, output }) => {
            run_yaml_to_testcases(input, output)?;
        }
        None => {
            // Legacy mode: run serialization
            run_serialize(&cli)?;
        }
    }

    Ok(())
}
