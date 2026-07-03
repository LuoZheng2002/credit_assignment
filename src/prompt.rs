pub fn prompt_with_tool_call(question: String) -> String {
    format!(
        "\
You are a helpful agent that solves the following problem.\n\
Question: {question}\n\
You are encouraged to use the Python code executor to help you do calculations to ensure correctness. \
You can invoke python code by emitting a Python markdown code block like this:\n\
\n\
```python
import numpy as np
A = np.array([[2, 1], [1, -1]])
b = np.array([5, 1])
solution = np.linalg.solve(A, b)
solution
```
\n\
You should reason step by step, and put the final answer in a \\boxed{{}}. \
If the question asks for a yes/no judgment, put exactly Yes or No in the \\boxed{{}}.\n\
Begin your reasoning:"
    )
}

pub fn prompt_without_tool_call(question: String) -> String {
    format!(
        "\
You are a helpful agent that solves the following problem.\n\
Question: {question}\n\
You should reason step by step, and put the final answer in a \\boxed{{}}. \
If the question asks for a yes/no judgment, put exactly Yes or No in the \\boxed{{}}.\n\
Begin your reasoning:"
    )
}
