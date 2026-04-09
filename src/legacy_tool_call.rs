use crate::execute_python_code::execute_python_code;


async fn execute_tool_call_content(tool_call_content: &str) -> String {
    // first check for valid json format
    let tool_call_json: serde_json::Value = match serde_json::from_str(tool_call_content) {
        Ok(json) => json,
        Err(_) => {
            return "<tool_response>Tool call content is not in valid json format.</tool_response>"
                .to_string();
        }
    };
    // tool call must be an object with a "name" field
    let tool_name = match tool_call_json.get("name") {
        Some(name) => match name.as_str() {
            Some("python") => "python",
            Some("sub_agent") => "sub_agent",
            _ => {
                return "<tool_response>Tool name not recognized.</tool_response>".to_string();
            }
        },
        None => {
            return "<tool_response>Tool call json must have a \"name\" field.</tool_response>"
                .to_string();
        }
    };
    match tool_name {
        "python" => {
            // for python tool, we expect a "code" field
            match tool_call_json.get("code") {
                Some(code) => match code.as_str() {
                    Some(code_str) => {
                        let python_code_result = execute_python_code(code_str.to_string()).await;
                        return format!(
                            "<tool_response>{}</tool_response>",
                            python_code_result.trim()
                        );
                    }
                    None => {
                        return "<tool_response>\"code\" field in tool call json must be a string.</tool_response>".to_string();
                    }
                },
                None => {
                    return "<tool_response>Python tool call json must have a \"code\" field.</tool_response>".to_string();
                }
            }
        }
        "sub_agent" => {
            // for sub_agent tool, we expect a "request" field
            match tool_call_json.get("request") {
                Some(request) => match request.as_str() {
                    Some(_request_str) => {
                        // for now we just return a placeholder response for the sub-agent tool, as implementing a full sub-agent is out of scope for this project
                        // return format!("<tool_response>Sub-agent received request: {}</tool_response>", request_str);
                        // TODO: implement sub-agent tool by calling rollout recursively with the request as the question
                        return format!(
                            "<tool_response>Sorry, sub-agent tool currently unavailable.</tool_response>"
                        );
                    }
                    None => {
                        return "<tool_response>\"request\" field in tool call json must be a string.</tool_response>".to_string();
                    }
                },
                None => {
                    return "<tool_response>Sub-agent tool call json must have a \"request\" field.</tool_response>".to_string();
                }
            }
        }
        _ => {
            return "<tool_response>Tool name not recognized.</tool_response>".to_string();
        }
    }
}

pub async fn execute_legacy_tool_call(tool_call: &str) -> String {
    assert!(
            tool_call.starts_with("<tool_call>"),
            "Tool call not properly formatted: {}",
            tool_call
        );
        let tool_call_content_end_index = if let Some(end_index) = tool_call.find("</tool_call>") {
            end_index
        } else {
            tool_call.len()
        };
        let tool_call_content = &tool_call["<tool_call>".len()..tool_call_content_end_index];
        execute_tool_call_content(tool_call_content).await
}