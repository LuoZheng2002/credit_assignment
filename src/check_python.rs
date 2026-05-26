use pyo3::{PyResult, Python, types::PyAnyMethods};

pub fn check_sympy_availability() -> Result<(), String> {
    let result = Python::attach(|py| -> PyResult<Result<(), String>> {
        let sys = py.import("sys")?;
        let executable: String = sys.getattr("executable")?.extract()?;

        let result = py.import("sympy");
        match result {
            Ok(_) => Ok(Ok(())),
            Err(err) => Ok(Err(format!(
                "Error importing sympy: {}. The python environment does not have sympy installed. Python executable is: {}",
                err, executable
            ))),
        }
    });
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(format!("Error importing sympy: {}", err)),
        Err(err) => Err(format!(
            "Error acquiring GIL or executing Python code: {}",
            err
        )),
    }
}
