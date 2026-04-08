use std::{ffi::CString, time::Duration};

use pyo3::{prelude::*, types::PyDict};

fn blocking_python_code_task(code: String) -> PyResult<String> {
    Python::attach(|py| -> PyResult<String> {
        // Persistent REPL namespace
        let globals = PyDict::new(py);

        let io_redirect_code = r#"
import io, sys
buf = io.StringIO()
sys.stdout = buf
"#;

        py.run(
            CString::new(io_redirect_code).unwrap().as_c_str(),
            Some(&globals),
            Some(&globals),
        )?;
        let code = CString::new(code).unwrap();
        py.run(code.as_c_str(), Some(&globals), Some(&globals))?;
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
            Ok(Ok(output)) => output,
            Ok(Err(err)) => format!("Python error: {}", err),
            Err(_) => "Sorry, unexpected error occurred. Please try again.".to_string(),
        },
        Err(_) => "Python code execution timed out.".to_string(),
    }
}
