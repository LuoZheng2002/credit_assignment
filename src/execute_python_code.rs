use std::{ffi::CString, time::Duration};

use pyo3::{prelude::*, types::PyDict};

fn blocking_python_code_task(code: String) -> PyResult<String> {
    Python::attach(|py| -> PyResult<String> {
        // Persistent REPL namespace
        let globals = PyDict::new(py);

        let io_redirect_code = r#"
import ast
import io
import sys
buf = io.StringIO()
sys.stdout = buf

def _execute_with_trailing_expression(code_text: str, namespace: dict):
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

fn format_limited_output(output: String, max_chars: usize) -> String {
    assert!(max_chars > 0, "max_chars must be greater than zero");
    let output_len = output.chars().count();
    if output_len <= max_chars {
        return output;
    }

    println!(
        "[Warning]: python output length limit exceeded, truncated to {} characters.",
        max_chars
    );

    let truncated: String = output.chars().take(max_chars).collect();
    let omitted_len = output_len - max_chars;
    format!(
        "{}\n[Output truncated: original_length={}, shown={}, omitted={}]",
        truncated, output_len, max_chars, omitted_len
    )
}

pub async fn execute_python_code(code: String) -> String {
    let task = tokio::task::spawn_blocking(move || blocking_python_code_task(code));
    let result = tokio::time::timeout(Duration::from_millis(5000), task).await;
    match result {
        Ok(join_result) => match join_result {
            Ok(Ok(output)) => {
                if output.trim().is_empty() {
                    // panic!("Python interpreter did not return any output. Please use print statements to retrieve results.");
                    println!("[Warning]: Python interpreter did not return any output.");
                    return "Python interpreter did not return any output. Please use print statements to retrieve results.".to_string();
                }
                format_limited_output(output, MAX_TOOL_OUTPUT_CHARS)
            }
            Ok(Err(err)) => format!("Python error: {}", err),
            Err(_) => "Sorry, unexpected error occurred. Please try again.".to_string(),
        },
        Err(_) => "Python code execution timed out.".to_string(),
    }
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
