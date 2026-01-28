//! Python bindings for the crowd-pilot serializer.
//!
//! Exposes YAML conversion functions for evaluation pipelines.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crowd_pilot_serializer_core::{
    convert_yaml_to_testcases as core_yaml_to_sed,
    convert_yaml_to_zeta_eval as core_yaml_to_zeta,
    default_system_prompt as core_default_system_prompt,
    TestCase, TestCaseMessage, ZetaEvalTestCase, ZetaTestCaseMessage,
};

fn test_case_message_to_dict(py: Python<'_>, msg: &TestCaseMessage) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("role", &msg.role)?;
    dict.set_item("content", &msg.content)?;
    if let Some(ref tag) = msg.eval_tag {
        dict.set_item("eval_tag", tag)?;
    }
    if let Some(ref assertions) = msg.assertions {
        dict.set_item("assertions", assertions)?;
    }
    Ok(dict.into())
}

fn test_case_to_dict(py: Python<'_>, tc: &TestCase) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("task_id", &tc.task_id)?;
    
    let context: Vec<Py<PyDict>> = tc.context
        .iter()
        .map(|msg| test_case_message_to_dict(py, msg))
        .collect::<PyResult<_>>()?;
    dict.set_item("context", context)?;
    
    dict.set_item("expected_final_response", &tc.expected_final_response)?;
    
    if let Some(ref assertions) = tc.assertions {
        dict.set_item("assertions", assertions)?;
    }
    
    Ok(dict.into())
}

fn zeta_message_to_dict(py: Python<'_>, msg: &ZetaTestCaseMessage) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("role", &msg.role)?;
    dict.set_item("content", &msg.content)?;
    Ok(dict.into())
}

fn zeta_test_case_to_dict(py: Python<'_>, tc: &ZetaEvalTestCase) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("task_id", &tc.task_id)?;
    
    let context: Vec<Py<PyDict>> = tc.context
        .iter()
        .map(|msg| zeta_message_to_dict(py, msg))
        .collect::<PyResult<_>>()?;
    dict.set_item("context", context)?;
    
    dict.set_item("expected_final_response", &tc.expected_final_response)?;
    
    if let Some(ref assertions) = tc.assertions {
        dict.set_item("assertions", assertions)?;
    }
    
    Ok(dict.into())
}

/// Convert YAML content to SED-format test cases.
///
/// Args:
///     yaml_content: The YAML file content as a string.
///
/// Returns:
///     List of test case dictionaries with keys:
///     - task_id: str
///     - context: List[dict] with role/content
///     - expected_final_response: str
///     - assertions: Optional[str]
#[pyfunction]
fn convert_yaml_to_testcases(py: Python<'_>, yaml_content: String) -> PyResult<Vec<Py<PyDict>>> {
    let test_cases = core_yaml_to_sed(&yaml_content)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    test_cases
        .iter()
        .map(|tc| test_case_to_dict(py, tc))
        .collect()
}

/// Convert YAML content to Zeta-format test cases.
///
/// Args:
///     yaml_content: The YAML file content as a string.
///
/// Returns:
///     List of test case dictionaries with keys:
///     - task_id: str
///     - context: List[dict] with role/content
///     - expected_final_response: str
///     - assertions: Optional[str]
#[pyfunction]
fn convert_yaml_to_zeta_eval(py: Python<'_>, yaml_content: String) -> PyResult<Vec<Py<PyDict>>> {
    let test_cases = core_yaml_to_zeta(&yaml_content)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    test_cases
        .iter()
        .map(|tc| zeta_test_case_to_dict(py, tc))
        .collect()
}

/// Get the default system prompt for SED-format models.
///
/// Args:
///     viewport_radius: Number of lines above/below cursor to show.
///
/// Returns:
///     The system prompt string.
#[pyfunction]
fn default_system_prompt(viewport_radius: usize) -> String {
    core_default_system_prompt(viewport_radius)
}

/// Python module for crowd-pilot serializer.
#[pymodule]
fn crowd_pilot_serializer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(convert_yaml_to_testcases, m)?)?;
    m.add_function(wrap_pyfunction!(convert_yaml_to_zeta_eval, m)?)?;
    m.add_function(wrap_pyfunction!(default_system_prompt, m)?)?;
    Ok(())
}
