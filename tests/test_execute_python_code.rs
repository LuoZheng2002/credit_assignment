// use credit_assignment::{
//     agent::trajectory_action_types::ToolResponse, execute_python_code::execute_python_code,
// };
// use pyo3::Python;

// #[tokio::test]
// async fn test_execute_python_code() {
//     std::panic::set_hook(Box::new(|info| {
//         eprintln!("panic occurred: {}", info);
//         std::process::abort();
//     }));
//     Python::initialize();
//     let code = r#"
// a = 1 + 2
// print(a)
// import time
// time.sleep(6000)
// "#
//     .to_string();
//     let output = execute_python_code(code).await;
//     let ToolResponse::PythonSuccess(output) = output else {
//         panic!("Expected PythonSuccess, got PythonError");
//     };
//     assert_eq!(output.trim(), "3");
// }
