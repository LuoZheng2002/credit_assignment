// todo: fill the tool information
pub fn prompt_with_tool_call(question: String) -> String {
    format!(
        "\
You are a helpful agent that solves the following problem.\n\
Question: {question}\n\
You can use the following tools to help you solve the problem:\n\
"
    )
}

pub fn prompt_without_tool_call(question: String) -> String {
    format!(
        "\
You are a helpful agent that solves the following problem.\n\
Question: {question}\n\
Please reason step by step, and put the final answer in a \\boxed{{}}.\n\
"
    )
}
