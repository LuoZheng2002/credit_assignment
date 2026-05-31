use std::process::Command;

pub fn check_sympy_availability() -> Result<(), String> {
    let output = Command::new("uv")
        .arg("run")
        .arg("--project")
        .arg("pyprojects/common")
        .arg("python")
        .arg("-c")
        .arg("import sympy, numpy, scipy")
        .output()
        .map_err(|error| {
            format!(
                "Failed to run python dependency check command for tool server runtime: {}",
                error
            )
        })?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        Err(format!(
            "Python tool server environment check failed. Command: `uv run --project pyprojects/common python -c \"import sympy, numpy, scipy\"`. stderr: {}",
            if stderr.is_empty() {
                "<empty>"
            } else {
                &stderr
            }
        ))
    }
}
