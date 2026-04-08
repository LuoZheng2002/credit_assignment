from pathlib import Path
import json

from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen2.5-7B-Instruct"
out_dir = Path("./qwen2.5-7b-instruct_assets")
out_dir.mkdir(parents=True, exist_ok=True)

# Download the tokenizer files as concrete files in out_dir
snapshot_download(
    repo_id=repo_id,
    revision="main",
    local_dir=str(out_dir),
    local_dir_use_symlinks=False,
    allow_patterns=[
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ],
)

# Extract the chat template into its own file for Minijinja or other Jinja engines
tokenizer_config_path = out_dir / "tokenizer_config.json"
cfg = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))

chat_template = cfg.get("chat_template")
if not chat_template:
    raise RuntimeError("No chat_template found in tokenizer_config.json")

(out_dir / "chat_template.jinja").write_text(chat_template, encoding="utf-8")

print(f"Saved files to: {out_dir.resolve()}")
print(f"Chat template: {(out_dir / 'chat_template.jinja').resolve()}")