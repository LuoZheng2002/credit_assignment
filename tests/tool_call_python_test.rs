use credit_assignment::tool_call_python::{
    PythonToolResponse, PythonToolServerPool, PYTHON_TOOL_REQUEST_TIMEOUT_MS,
    PYTHON_TOOL_SERVER_RESPONSE_TIMEOUT_MS,
};
use tokio::time::{Duration, Instant, timeout};

#[tokio::test]
async fn python_tool_executes_simple_code() {
    let pool = PythonToolServerPool::new(1).await.unwrap();
    let response = pool.execute_code("print(1 + 1)".to_string()).await;
    assert_eq!(
        response,
        PythonToolResponse::PythonSuccess("2\n".to_string())
    );
}

#[tokio::test]
async fn python_tool_state_does_not_persist_across_requests() {
    let pool = PythonToolServerPool::new(1).await.unwrap();
    let first = pool
        .execute_code("x = 41\nprint(\"set\")".to_string())
        .await;
    assert_eq!(
        first,
        PythonToolResponse::PythonSuccess("set\n".to_string())
    );

    let second = pool.execute_code("print(x)".to_string()).await;
    match second {
        PythonToolResponse::PythonError(message) => {
            assert!(
                message.contains("name 'x' is not defined") || message.contains("NameError"),
                "unexpected error message: {}",
                message
            );
        }
        other => panic!("expected state-isolation error, got {:?}", other),
    }
}

#[tokio::test]
async fn python_tool_blocks_writing_files_to_disk() {
    let pool = PythonToolServerPool::new(1).await.unwrap();
    let response = pool
        .execute_code(
            "with open('blocked.txt', 'w') as handle:\n    handle.write('x')".to_string(),
        )
        .await;
    match response {
        PythonToolResponse::PythonError(message) => {
            assert!(
                message.contains("sandbox") || message.contains("forbids"),
                "unexpected error message: {}",
                message
            );
        }
        other => panic!("expected sandbox violation, got {:?}", other),
    }
}

#[tokio::test]
async fn python_tool_timeout_returns_promptly_and_pool_stays_usable() {
    let pool = PythonToolServerPool::new(1).await.unwrap();
    let code = r#"
import time
while True:
    pass
"#;
    let started = Instant::now();
    let response = timeout(
        Duration::from_millis(PYTHON_TOOL_SERVER_RESPONSE_TIMEOUT_MS + 2000),
        pool.execute_code(code.to_string()),
    )
    .await
    .expect("tool execution should return promptly after timeout handling");
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_millis(PYTHON_TOOL_SERVER_RESPONSE_TIMEOUT_MS + 1000),
        "tool execution took too long: {:?}",
        elapsed
    );
    match response {
        PythonToolResponse::PythonError(message) => {
            assert!(
                message.contains(&format!(
                    "timed out after {} ms",
                    PYTHON_TOOL_REQUEST_TIMEOUT_MS
                )),
                "unexpected error message: {}",
                message
            );
        }
        other => panic!("expected timeout error, got {:?}", other),
    }

    let recovery = pool.execute_code("print(6 * 7)".to_string()).await;
    assert_eq!(
        recovery,
        PythonToolResponse::PythonSuccess("42\n".to_string())
    );
}
