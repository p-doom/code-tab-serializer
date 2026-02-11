//! Python bindings for the crowd-pilot serializer.
//!
//! Provides functions for converting YAML eval files to conversations.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crowd_pilot_serializer_core::{
    convert_yaml_to_conversations as core_convert, convert_yaml_to_sweep as core_convert_sweep,
    convert_yaml_to_zeta as core_convert_zeta, default_system_prompt as core_default_system_prompt,
    parse_yaml_task as core_parse_yaml, sweep_system_prompt as core_sweep_system_prompt,
    zeta_system_prompt as core_zeta_system_prompt, ConversationMessage, FinalizedConversation,
    Role, SweepConfig, SweepConversation, SweepHistoryCenterMode, SweepOpenedFileContextMode,
    Tokenizer, YamlProcessingConfig, ZetaConfig, ZetaConversation,
};

/// Character-based approximate tokenizer (~4 chars per token).
struct CharApproxTokenizer;

impl Tokenizer for CharApproxTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        text.len() / 4
    }

    fn truncate_to_max_tokens(&self, text: &str, max_tokens: usize) -> String {
        text.chars().take(max_tokens * 4).collect()
    }
}

fn role_to_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

fn message_to_dict(py: Python<'_>, msg: &ConversationMessage) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("role", role_to_str(msg.role))?;
    dict.set_item("content", &msg.content)?;
    Ok(dict.into())
}

fn conversation_to_dict(py: Python<'_>, conv: &FinalizedConversation) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);

    let messages: Vec<Py<PyDict>> = conv
        .messages
        .iter()
        .map(|m| message_to_dict(py, m))
        .collect::<PyResult<_>>()?;

    dict.set_item("messages", PyList::new(py, messages)?)?;
    dict.set_item("token_count", conv.token_count)?;

    Ok(dict.into())
}

/// Convert YAML content to conversations.
///
/// Args:
///     yaml_content: The YAML file content as a string.
///     viewport_radius: Number of lines above/below cursor to show (default: 10).
///
/// Returns:
///     List of conversation dictionaries, each with:
///     - messages: List of {role, content} dicts
///     - token_count: Approximate token count
#[pyfunction]
#[pyo3(signature = (yaml_content, viewport_radius=10))]
fn convert_yaml_to_conversations(
    py: Python<'_>,
    yaml_content: String,
    viewport_radius: usize,
) -> PyResult<Vec<Py<PyDict>>> {
    let config = YamlProcessingConfig {
        viewport_radius,
        ..Default::default()
    };

    let conversations = core_convert(&yaml_content, &CharApproxTokenizer, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    conversations
        .iter()
        .map(|c| conversation_to_dict(py, c))
        .collect()
}

/// Parse a YAML task file and return structured data.
///
/// Args:
///     yaml_content: The YAML file content as a string.
///
/// Returns:
///     Dictionary with task_id, description, and states.
#[pyfunction]
fn parse_yaml_task(py: Python<'_>, yaml_content: String) -> PyResult<Py<PyDict>> {
    let task = core_parse_yaml(&yaml_content)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    let dict = PyDict::new(py);
    dict.set_item("task_id", &task.task_id)?;

    if let Some(desc) = &task.description {
        dict.set_item("description", desc)?;
    }

    // Convert states to list of dicts
    let states: Vec<Py<PyDict>> = task
        .states
        .iter()
        .map(|s| {
            let state_dict = PyDict::new(py);
            if let Some(step) = s.step {
                state_dict.set_item("step", step)?;
            }
            if let Some(eval_tag) = &s.eval_tag {
                state_dict.set_item("eval", eval_tag)?;
            }
            if let Some(files) = &s.files {
                let files_dict = PyDict::new(py);
                for (k, v) in files {
                    files_dict.set_item(k, v)?;
                }
                state_dict.set_item("files", files_dict)?;
            }
            if let Some(cursor) = &s.cursor {
                let cursor_dict = PyDict::new(py);
                if let Some(file) = &cursor.file {
                    cursor_dict.set_item("file", file)?;
                }
                cursor_dict.set_item("line", cursor.line)?;
                cursor_dict.set_item("column", cursor.column)?;
                state_dict.set_item("cursor", cursor_dict)?;
            }
            if let Some(terminal) = &s.terminal {
                let term_dict = PyDict::new(py);
                if let Some(cmd) = &terminal.command {
                    term_dict.set_item("command", cmd)?;
                }
                if let Some(out) = &terminal.output {
                    term_dict.set_item("output", out)?;
                }
                if let Some(exit_code) = terminal.exit_code {
                    term_dict.set_item("exit_code", exit_code)?;
                }
                state_dict.set_item("terminal", term_dict)?;
            }
            if let Some(assertions) = &s.judge_assertions {
                state_dict.set_item("judge_assertions", assertions)?;
            }
            Ok(state_dict.into())
        })
        .collect::<PyResult<_>>()?;

    dict.set_item("states", PyList::new(py, states)?)?;

    if let Some(labels) = &task.labels {
        dict.set_item("labels", labels)?;
    }

    Ok(dict.into())
}

/// Get the default system prompt for SED-format models.
///
/// Args:
///     viewport_radius: Number of lines above/below cursor to show.
///
/// Returns:
///     The system prompt string.
#[pyfunction]
#[pyo3(signature = (viewport_radius=10))]
fn default_system_prompt(viewport_radius: usize) -> String {
    core_default_system_prompt(viewport_radius)
}

fn zeta_conversation_to_dict(py: Python<'_>, conv: &ZetaConversation) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);

    let messages: Vec<Py<PyDict>> = conv
        .messages
        .iter()
        .map(|m| message_to_dict(py, m))
        .collect::<PyResult<_>>()?;

    dict.set_item("messages", PyList::new(py, messages)?)?;
    dict.set_item("token_count", conv.token_count)?;

    // Add editable range info
    let range_dict = PyDict::new(py);
    range_dict.set_item("start", conv.editable_range.start)?;
    range_dict.set_item("end", conv.editable_range.end)?;
    dict.set_item("editable_range", range_dict)?;

    Ok(dict.into())
}

fn sweep_conversation_to_dict(py: Python<'_>, conv: &SweepConversation) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);

    let messages: Vec<Py<PyDict>> = conv
        .messages
        .iter()
        .map(|m| message_to_dict(py, m))
        .collect::<PyResult<_>>()?;

    dict.set_item("messages", PyList::new(py, messages)?)?;
    dict.set_item("token_count", conv.token_count)?;
    dict.set_item("target_file", &conv.target_file)?;
    Ok(dict.into())
}

/// Convert YAML content to Zeta-format conversations.
///
/// Args:
///     yaml_content: The YAML file content as a string.
///
/// Returns:
///     List of Zeta conversation dictionaries, each with:
///     - messages: List of {role, content} dicts
///     - token_count: Approximate token count
///     - editable_range: {start, end} line numbers
#[pyfunction]
fn convert_yaml_to_zeta(py: Python<'_>, yaml_content: String) -> PyResult<Vec<Py<PyDict>>> {
    let config = ZetaConfig::default();

    let conversations = core_convert_zeta(&yaml_content, &CharApproxTokenizer, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    conversations
        .iter()
        .map(|c| zeta_conversation_to_dict(py, c))
        .collect()
}

/// Get the Zeta-format system prompt.
///
/// Returns:
///     The Zeta system prompt string.
#[pyfunction]
fn zeta_system_prompt() -> String {
    core_zeta_system_prompt()
}

/// Convert YAML content to Sweep-format conversations.
///
/// Args:
///     yaml_content: The YAML file content as a string.
///     viewport_lines: Fixed window size for target/history viewports (default: 21).
///     opened_file_context: Context strategy for opened files: `"full"` or `"viewport"`.
///     system_prompt: Optional custom system prompt. Defaults to Sweep prompt when None.
///     history_center: History window centering mode: `"changed"` or `"cursor"`.
///     max_tokens_per_conversation: Optional hard max token budget for a sample.
///         History is trimmed first to fit; sample is dropped if zero-history still does not fit.
///
/// Returns:
///     List of Sweep conversation dictionaries with:
///     - messages: List of {role, content} dicts
///     - token_count: Approximate token count
///     - target_file: Target file path
#[pyfunction]
#[pyo3(signature = (yaml_content, viewport_lines=21, opened_file_context="full".to_string(), system_prompt=None, history_center="changed".to_string(), max_tokens_per_conversation=None))]
fn convert_yaml_to_sweep(
    py: Python<'_>,
    yaml_content: String,
    viewport_lines: usize,
    opened_file_context: String,
    system_prompt: Option<String>,
    history_center: String,
    max_tokens_per_conversation: Option<usize>,
) -> PyResult<Vec<Py<PyDict>>> {
    let opened_mode = match opened_file_context.to_lowercase().as_str() {
        "full" => SweepOpenedFileContextMode::Full,
        "viewport" => SweepOpenedFileContextMode::Viewport,
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid opened_file_context '{}'. Supported: full, viewport",
                other
            )));
        }
    };

    let history_center_mode = match history_center.to_lowercase().as_str() {
        "changed" => SweepHistoryCenterMode::ChangedBlock,
        "cursor" => SweepHistoryCenterMode::Cursor,
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid history_center '{}'. Supported: changed, cursor",
                other
            )));
        }
    };

    let config = SweepConfig {
        viewport_lines,
        opened_file_context_mode: opened_mode,
        history_center_mode,
        max_tokens_per_conversation,
        system_prompt: system_prompt.unwrap_or_else(core_sweep_system_prompt),
        ..SweepConfig::default()
    };

    let conversations = core_convert_sweep(&yaml_content, &CharApproxTokenizer, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    conversations
        .iter()
        .map(|c| sweep_conversation_to_dict(py, c))
        .collect()
}

/// Get the Sweep-format default system prompt.
#[pyfunction]
fn sweep_system_prompt() -> String {
    core_sweep_system_prompt()
}

/// Python module for crowd-pilot serializer.
#[pymodule]
fn crowd_pilot_serializer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(convert_yaml_to_conversations, m)?)?;
    m.add_function(wrap_pyfunction!(convert_yaml_to_zeta, m)?)?;
    m.add_function(wrap_pyfunction!(convert_yaml_to_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(parse_yaml_task, m)?)?;
    m.add_function(wrap_pyfunction!(default_system_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(zeta_system_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(sweep_system_prompt, m)?)?;
    Ok(())
}
