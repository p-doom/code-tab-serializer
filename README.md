# crowd-pilot-serializer

Core serialization library for crowd-pilot's IDE interaction data. Converts IDE events (tab switches, edits, terminal commands, etc.) into conversation format for training language models.

## Architecture

This is a Rust library with:
- **Node.js/TypeScript bindings** (via napi-rs) - for the VS Code extension (uses character approximation for token counting)
- **CLI binary** - for batch preprocessing (uses HuggingFace tokenizer via embedded Python)

The serialization logic is the single source of truth, ensuring consistency between runtime inference and training data preprocessing.

## Crates

- `crates/core` - Core serialization logic
- `crates/napi` - Node.js bindings (`@crowd-pilot/serializer` npm package)
- `crates/cli` - CLI binary for preprocessing (`crowd-pilot-serialize`)

## Building

### Prerequisites

- Rust 1.70+
- Node.js 18+ (for napi bindings)
- Python 3.9+ with `transformers` installed (for CLI tokenizer)

### Build all

```bash
cargo build --release
```

### Build Node.js bindings

```bash
cd crates/napi
npm install
npm run build
```

### Build CLI

```bash
cargo build --release -p crowd-pilot-serialize
```

## Usage

### Node.js/TypeScript (Usage in crowd-pilot-extension)

```typescript
import { ConversationStateManager } from '@crowd-pilot/serializer';

const manager = new ConversationStateManager({
  viewportRadius: 10,
  coalesceRadius: 5,
  maxTokensPerMessage: 2048,
  maxTokensPerTerminalOutput: 256,
});

manager.handleTabEvent('/path/to/file.ts', 'file contents...');
manager.handleContentEvent('/path/to/file.ts', 10, 0, 'inserted text');

const messages = manager.finalizeForModel();
```

### CLI (Preprocessing)

#### SED format (command prediction)

```bash
crowd-pilot-serialize \
    --output-format sed \
    --csv-root ./data/sessions \
    --output-dir ./output \
    --tokenizer "Qwen/Qwen2-7B" \
    --chat-template qwen3 \
    --max-tokens-per-conversation 8192 \
    --max-tokens-per-message 2048 \
    --val-ratio 0.1
```

This uses shared CSV coalescing and outputs `training.jsonl` and `validation.jsonl` in SED conversation format.

#### Zeta format (edit prediction)

```bash
crowd-pilot-serialize \
    --output-format zeta \
    --csv-root ./data/sessions \
    --output-dir ./output-zeta \
    --tokenizer "Qwen/Qwen2-7B" \
    --chat-template qwen3 \
    --zeta-max-editable-tokens 180 \
    --zeta-max-context-tokens 350 \
    --zeta-diff-context-lines 3
```

This uses shared CSV coalescing and converts each CSV transition into a Zeta training example.

#### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-format` | `sed` | Output format: `sed` or `zeta` |
| `--csv-root` | required | Root directory containing per-session CSV files |
| `--output-dir` | required | Output directory for JSONL files |
| `--tokenizer` | required | HuggingFace tokenizer name or path |
| `--chat-template` | required | Chat template: `qwen3` or `glm45` |
| `--max-tokens-per-conversation` | 8192 | Maximum tokens per conversation chunk (SED only) |
| `--max-tokens-per-message` | 2048 | Maximum tokens per message (SED only) |
| `--min-conversation-messages` | 5 | Minimum messages to keep a conversation (SED only) |
| `--viewport-radius` | 10 | Lines above/below cursor to show (SED only) |
| `--coalesce-radius` | 5 | Radius for grouping nearby edits |
| `--val-ratio` | 0.10 | Fraction of sessions for validation |
| `--zeta-max-editable-tokens` | 180 | Editable-region token budget (Zeta only) |
| `--zeta-max-context-tokens` | 350 | Context-region token budget (Zeta only) |
| `--zeta-diff-context-lines` | 3 | Unified-diff context lines (Zeta only) |

### CLI (Replay crowd-code 2.0)

```bash
crowd-pilot-replay \
    --session ./recordings \
    --terminal-cols 120 \
    --terminal-rows 30 \
    --delay-ms 100
```

This replays crowd-code 2.0 recordings and renders editor + terminal viewports. The `--session` path must be a single `source_part_*.tar.gz` file or a directory containing those `.tar.gz` parts. Terminal output is VT-rendered from terminal bytestream events when available; otherwise it falls back to the recorded terminal viewport lines.

## License

Apache 2.0
