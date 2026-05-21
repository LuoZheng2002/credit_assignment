// todo: fill the tool information
pub fn prompt_with_tool_call(question: String) -> String {
    format!(
        "\
You are a helpful agent that solves the following problem.\n\
Question: {question}\n\
You are encouraged to use the Python code executor to help you do calculations to ensure correctness. \
You can invoke python code using the syntax similar to the following example:\n\
\n\
<tool_wait>\n\
```python
import numpy as np
A = np.array([[2, 1], [1, -1]])
b = np.array([5, 1])
solution = np.linalg.solve(A, b)
solution
```
</tool_wait>\n\
\n\
Make sure to wrap the python code with <tool_wait> and </tool_wait> tags.\n\
You should reason step by step, and put the final answer in a \\boxed{{}}.\n\
Begin your reasoning:"
    )
}

pub fn prompt_without_tool_call(question: String) -> String {
    format!(
        "\
You are a helpful agent that solves the following problem.\n\
Question: {question}\n\
You should reason step by step, and put the final answer in a \\boxed{{}}.\n\
Begin your reasoning:"
    )
}
