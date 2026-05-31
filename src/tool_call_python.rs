use std::{ffi::CString, time::Duration};

use pyo3::{prelude::*, types::PyDict};
use research_utility::log_message::log_warning;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::{
    llm_model::{LlmModelMarker, MyTokenizer},
    token_array::TokenArray,
};

pub fn extract_python_tool_call(response: String) -> Option<String> {
    let python_fence = "```python";
    let Some(python_start_position) = response.find(python_fence) else {
        return None;
    };

    let start_position = python_start_position;
    let end_position = {
        let search_start = python_start_position + python_fence.len();
        let after_python_fence = &response[search_start..];
        if let Some(end_relative) = after_python_fence.find("```\n") {
            search_start + end_relative + "```\n".len()
        } else if let Some(end_relative) = after_python_fence.find("```") {
            let mut end_position = search_start + end_relative + "```".len();
            if after_python_fence[end_relative + "```".len()..].starts_with('\n') {
                end_position += 1;
            }
            end_position
        } else {
            return None;
        }
    };
    Some(response[start_position..end_position].to_string())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PythonToolResponse {
    PythonSuccess(String),
    PythonError(String),
    // EmptyMessageHint,
}

impl PythonToolResponse {
    pub fn with_multi_turn_chat_template<M: LlmModelMarker>(
        &self,
        enable_thinking: bool,
    ) -> TokenArray<M> {
        let raw_python_response = match self {
            PythonToolResponse::PythonSuccess(output) => output.clone(),
            PythonToolResponse::PythonError(error) => format!("Python error: {}", error),
        };
        M::Tokenizer::apply_python_response_template_and_tokenize(
            raw_python_response,
            enable_thinking,
        )
    }
}

fn blocking_python_code_task(code: String) -> PyResult<String> {
    Python::attach(|py| -> PyResult<String> {
        // Persistent REPL namespace
        let globals = PyDict::new(py);

        let io_redirect_code = r#"
import ast
import contextlib
import io
import sys
import numpy as np
import scipy
import sympy as sp
from sympy import *
buf = io.StringIO()

class _NullWriter:
    def write(self, _text):
        return 0

    def flush(self):
        return None

_stderr_sink = _NullWriter()

def _execute_with_trailing_expression(code_text: str, namespace: dict):
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(_stderr_sink):
        tree = ast.parse(code_text, mode='exec')
        if len(tree.body) == 0:
            return

        last_stmt = tree.body[-1]
        if isinstance(last_stmt, ast.Expr):
            prefix_module = ast.Module(body=tree.body[:-1], type_ignores=[])
            ast.fix_missing_locations(prefix_module)
            if len(prefix_module.body) > 0:
                exec(compile(prefix_module, '<tool>', 'exec'), namespace, namespace)

            expr = ast.Expression(last_stmt.value)
            ast.fix_missing_locations(expr)
            expr_value = eval(compile(expr, '<tool>', 'eval'), namespace, namespace)
            if expr_value is not None:
                print(repr(expr_value))
        else:
            exec(compile(tree, '<tool>', 'exec'), namespace, namespace)
"#;

        py.run(
            CString::new(io_redirect_code).unwrap().as_c_str(),
            Some(&globals),
            Some(&globals),
        )?;
        globals.set_item("__code_input", code)?;
        py.run(
            CString::new("_execute_with_trailing_expression(__code_input, globals())")
                .unwrap()
                .as_c_str(),
            Some(&globals),
            Some(&globals),
        )?;
        // use eval to get the result of the last expression
        let output = py
            .eval(
                CString::new("buf.getvalue()").unwrap().as_c_str(),
                Some(&globals),
                Some(&globals),
            )?
            .extract::<String>()?;
        Ok(output)
    })
}

const MAX_TOOL_OUTPUT_CHARS: usize = 2000;
static PYTHON_EXECUTION_SEMAPHORE: Semaphore = Semaphore::const_new(1);

fn format_limited_output(output: String, max_chars: usize) -> String {
    assert!(max_chars > 0, "max_chars must be greater than zero");
    let output_len = output.chars().count();
    if output_len <= max_chars {
        return output;
    }
    log_warning(format!(
        "Python output length limit exceeded, truncated to {} characters.",
        max_chars
    ));

    let truncated: String = output.chars().take(max_chars).collect();
    let omitted_len = output_len - max_chars;
    format!(
        "{}\n[Output truncated: original_length={}, shown={}, omitted={}]",
        truncated, output_len, max_chars, omitted_len
    )
}

async fn execute_python_code(code: String) -> PythonToolResponse {
    let _python_permit = PYTHON_EXECUTION_SEMAPHORE
        .acquire()
        .await
        .expect("python execution semaphore closed");
    let task = tokio::task::spawn_blocking(move || blocking_python_code_task(code));
    let result = tokio::time::timeout(Duration::from_millis(5000), task).await;
    match result {
        Ok(join_result) => match join_result {
            Ok(Ok(output)) => {
                if output.trim().is_empty() {
                    log_warning("Python interpreter did not return any output.");
                    return PythonToolResponse::PythonSuccess(
                        "Python interpreter did not return any output. Please use print statements to retrieve results.".to_string(),
                    );
                }
                PythonToolResponse::PythonSuccess(format_limited_output(
                    output,
                    MAX_TOOL_OUTPUT_CHARS,
                ))
            }
            Ok(Err(err)) => PythonToolResponse::PythonError(err.to_string()),
            Err(_) => PythonToolResponse::PythonError(
                "Sorry, unexpected error occurred. Please try again.".to_string(),
            ),
        },
        Err(_) => PythonToolResponse::PythonError("Python code execution timed out.".to_string()),
    }
}

pub async fn execute_python_tool_call(tool_call: &str) -> PythonToolResponse {
    let trimmed_tool_call = tool_call.trim_start().to_string();
    assert!(
        trimmed_tool_call.starts_with("```python"),
        "Tool call not properly formatted: {}",
        tool_call
    );
    let code_start = trimmed_tool_call
        .find('\n')
        .map(|idx| idx + 1)
        .unwrap_or("```python".len());
    let fence_end_index = trimmed_tool_call[code_start..]
        .find("```")
        .map(|relative_idx| code_start + relative_idx);
    let Some(fence_end_index) = fence_end_index else {
        return PythonToolResponse::PythonError(
            "Tool call markdown code block not properly closed.".to_string(),
        );
    };
    if fence_end_index < code_start {
        return PythonToolResponse::PythonError(
            "Tool call markdown code block not properly formatted.".to_string(),
        );
    }
    let code = &trimmed_tool_call[code_start..fence_end_index];
    execute_python_code(code.to_string()).await
}

#[cfg(test)]
mod tests {
    use super::format_limited_output;

    #[test]
    fn no_truncation_when_below_limit() {
        let output = "hello".to_string();
        let result = format_limited_output(output.clone(), 2000);
        assert_eq!(result, output);
    }

    #[test]
    fn no_truncation_when_equal_to_limit() {
        let output = "a".repeat(2000);
        let result = format_limited_output(output.clone(), 2000);
        assert_eq!(result, output);
    }

    #[test]
    fn truncates_when_exceeding_limit() {
        let output = "b".repeat(2005);
        let result = format_limited_output(output, 2000);
        let expected_prefix = "b".repeat(2000);
        assert!(result.starts_with(&expected_prefix));
        assert!(
            result.ends_with("[Output truncated: original_length=2005, shown=2000, omitted=5]")
        );
    }

    #[test]
    fn truncates_unicode_by_character_count() {
        let output = "你".repeat(2001);
        let result = format_limited_output(output, 2000);
        let expected_prefix = "你".repeat(2000);
        assert!(result.starts_with(&expected_prefix));
        assert!(
            result.ends_with("[Output truncated: original_length=2001, shown=2000, omitted=1]")
        );
    }
}
