use crate::agent::{tool_call_python::execute_python_code, trajectory_action_types::ToolResponse};

pub async fn execute_planner_tool_call(tool_call: &str) -> ToolResponse {
    let mut trimmed_tool_call = tool_call.trim_start().to_string();
    // trim <tool_wait>
    if trimmed_tool_call.starts_with("<tool_wait>") {
        trimmed_tool_call = trimmed_tool_call["<tool_wait>".len()..]
            .trim_start()
            .to_string();
    }
    assert!(
        trimmed_tool_call.starts_with("```python"),
        "Tool call not properly formatted: {}",
        tool_call
    );
    let Some(fence_end_index) = trimmed_tool_call.rfind("```") else {
        return ToolResponse::PythonError(
            "Tool call markdown code block not properly closed.".to_string(),
        );
    };
    let code_start = trimmed_tool_call
        .find('\n')
        .map(|idx| idx + 1)
        .unwrap_or("```python".len());
    if fence_end_index < code_start {
        return ToolResponse::PythonError(
            "Tool call markdown code block not properly formatted.".to_string(),
        );
    }
    let code = &trimmed_tool_call[code_start..fence_end_index];
    execute_python_code(code.to_string()).await
}

pub const MAX_NUM_TRAJECTORIES: usize = 16;
