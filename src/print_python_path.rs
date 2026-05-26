use pyo3::{PyResult, Python, types::PyAnyMethods};

pub fn print_python_path() -> PyResult<String> {
    Python::attach(|py| -> PyResult<String> {
        let sys = py.import("sys")?;

        let executable: String = sys.getattr("executable")?.extract()?;
        let prefix: String = sys.getattr("prefix")?.extract()?;
        let base_prefix: String = sys.getattr("base_prefix")?.extract()?;
        let path: Vec<String> = sys.getattr("path")?.extract()?;

        let mut result = String::new();
        result.push_str(&format!("sys.executable = {}\n", executable));
        result.push_str(&format!("sys.prefix      = {}\n", prefix));
        result.push_str(&format!("sys.base_prefix = {}\n", base_prefix));
        result.push_str("sys.path =\n");
        for p in path {
            result.push_str(&format!("  {}\n", p));
        }

        Ok(result)
    })
}
