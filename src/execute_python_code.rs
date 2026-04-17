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

pub async fn execute_python_code(code: String) -> String {
    let task = tokio::task::spawn_blocking(move || blocking_python_code_task(code));
    let result = tokio::time::timeout(Duration::from_millis(5000), task).await;
    match result {
        Ok(join_result) => match join_result {
            Ok(Ok(output)) => {
                if output.trim().is_empty() {
                    panic!("Python interpreter did not return any output. Please use print statements to retrieve results.");
                    // return "Python interpreter did not return any output. Please use print statements to retrieve results.".to_string();
                }
                output
            }
            Ok(Err(err)) => format!("Python error: {}", err),
            Err(_) => "Sorry, unexpected error occurred. Please try again.".to_string(),
        },
        Err(_) => "Python code execution timed out.".to_string(),
    }
}
