use credit_assignment::execute_python_code::execute_python_code;
use pyo3::Python;




#[tokio::test]
async fn test_execute_python_code() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        std::process::abort();
    }));
    Python::initialize();
    let code = r#"
a = 1 + 2
print(a)
import time
time.sleep(6000)
"#.to_string();
    let output = execute_python_code(code).await;
    assert_eq!(output.trim(), "3");
}