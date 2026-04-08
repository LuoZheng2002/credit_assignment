use credit_assignment::multi_agent::rollout::{
    execute_planner_tool_call, split_tool_call_segments_for_test,
};
use credit_assignment::multi_agent::session::ModelOperation;
use pyo3::Python;

#[test]
fn split_mixed_tool_calls_preserves_order_and_wrappers() {
    let response = r#"Reasoning before first call
<tool_call>{"name": "python", "code": "print(1)"}</tool_call>
Additional reasoning between calls
```python
print(2)
```
Final reasoning after markdown call
<tool_call>{"name": "sub_agent", "request": "Compute something"}</tool_call>
"#;

    let operations = split_tool_call_segments_for_test(response);
    assert_eq!(operations.len(), 6, "expected 6 operations in total");

    assert!(matches!(
        &operations[0],
        ModelOperation::PlannerReasoning(reasoning)
            if reasoning == "Reasoning before first call"
    ));
    assert!(matches!(
        &operations[1],
        ModelOperation::PlannerToolCall(tool_call)
            if tool_call == "<tool_call>{\"name\": \"python\", \"code\": \"print(1)\"}</tool_call>"
    ));
    assert!(matches!(
        &operations[2],
        ModelOperation::PlannerReasoning(reasoning)
            if reasoning == "Additional reasoning between calls"
    ));
    assert!(matches!(
        &operations[3],
        ModelOperation::PlannerToolCall(tool_call)
            if tool_call.starts_with("```python") && tool_call.contains("print(2)")
    ));
    assert!(matches!(
        &operations[4],
        ModelOperation::PlannerReasoning(reasoning)
            if reasoning == "Final reasoning after markdown call"
    ));
    assert!(matches!(
        &operations[5],
        ModelOperation::PlannerToolCall(tool_call)
            if tool_call == "<tool_call>{\"name\": \"sub_agent\", \"request\": \"Compute something\"}</tool_call>"
    ));
}

#[tokio::test]
async fn execute_markdown_planner_tool_call_runs_python() {
    Python::initialize();
    let markdown_tool_call = r#"```python
from math import comb

def evaluate_expression(n):

```"#;
    let response = execute_planner_tool_call(markdown_tool_call).await;
    assert!(response.contains("<tool_response>5</tool_response>"));
}

#[tokio::test]
async fn execute_json_planner_tool_call_runs_python() {
    Python::initialize();
    let json_tool_call = r#"<tool_call>{"name": "python", "code": "print(7)"}</tool_call>"#;
    let response = execute_planner_tool_call(json_tool_call).await;
    assert!(response.contains("<tool_response>7</tool_response>"));
}
