import argparse
import ast
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sympy as sp


MAX_TOOL_OUTPUT_CHARS = 2000


def _load_dotenv_if_present(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists() or not path.is_file():
        return
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=path, override=False)


def create_request_namespace() -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "np": np,
        "scipy": scipy,
        "sp": sp,
    }
    for symbol_name in sp.__all__:
        namespace[symbol_name] = getattr(sp, symbol_name)
    return namespace


def execute_with_trailing_expression(code_text: str, namespace: dict[str, Any]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tree = ast.parse(code_text, mode="exec")
        if len(tree.body) == 0:
            return ""

        last_stmt = tree.body[-1]
        if isinstance(last_stmt, ast.Expr):
            prefix_module = ast.Module(body=tree.body[:-1], type_ignores=[])
            ast.fix_missing_locations(prefix_module)
            if len(prefix_module.body) > 0:
                exec(compile(prefix_module, "<tool>", "exec"), namespace, namespace)

            expr = ast.Expression(last_stmt.value)
            ast.fix_missing_locations(expr)
            expr_value = eval(compile(expr, "<tool>", "eval"), namespace, namespace)
            if expr_value is not None:
                print(repr(expr_value))
        else:
            exec(compile(tree, "<tool>", "exec"), namespace, namespace)
    return buf.getvalue()


def format_limited_output(output: str, max_chars: int) -> str:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    output_len = len(output)
    if output_len <= max_chars:
        return output

    truncated = output[:max_chars]
    omitted_len = output_len - max_chars
    return (
        f"{truncated}\n"
        "[Output truncated: "
        f"original_length={output_len}, shown={max_chars}, omitted={omitted_len}]"
    )


def execute_single_shot(code: str) -> dict[str, Any]:
    try:
        namespace = create_request_namespace()
        output = execute_with_trailing_expression(code, namespace)
        output = format_limited_output(output, MAX_TOOL_OUTPUT_CHARS)
        return {"ok": True, "output": output}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "error": str(error)}


def main() -> int:
    _load_dotenv_if_present()

    parser = argparse.ArgumentParser(description="Python tool single-shot executor")
    parser.add_argument("--single-shot", action="store_true")
    args = parser.parse_args()

    if not args.single_shot:
        sys.stderr.write("--single-shot must be set\n")
        return 2

    code = sys.stdin.read()
    response = execute_single_shot(code)
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
