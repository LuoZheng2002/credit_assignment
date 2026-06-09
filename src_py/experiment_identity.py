from __future__ import annotations


def experiment_key(model_cli_name: str, config_nickname: str) -> str:
    model = model_cli_name.strip()
    nickname = config_nickname.strip()
    if not model:
        raise ValueError("model_cli_name cannot be empty")
    if not nickname:
        raise ValueError("config_nickname cannot be empty")
    return f"{model}_{nickname}"


def modal_function_name(prefix: str, model_cli_name: str, config_nickname: str) -> str:
    name_prefix = prefix.strip()
    if not name_prefix:
        raise ValueError("function name prefix cannot be empty")
    return f"{name_prefix}__{experiment_key(model_cli_name, config_nickname)}"
