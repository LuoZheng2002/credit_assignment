use credit_assignment::execute_python_code::execute_python_code;
use pyo3::Python;

#[tokio::test]
async fn test_python_environment() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        std::process::abort();
    }));
    dotenvy::dotenv().ok();
    println!("PYTHONPATH: {:?}", std::env::var("PYTHONPATH"));
    Python::initialize();
    let code = r#"
import sympy
print("Hello world!")
"#
    .to_string();
    let output = execute_python_code(code).await;
    assert_eq!(output.trim(), "Hello world!");
}
