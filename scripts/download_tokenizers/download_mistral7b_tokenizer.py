from pathlib import Path

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3", trust_remote_code=True
)
output_dir = Path("tokenizers/mistral7b")
tokenizer.save_pretrained(output_dir)

chat_template = tokenizer.chat_template
if chat_template is None:
    raise RuntimeError(
        "Tokenizer chat_template is missing for mistralai/Mistral-7B-Instruct-v0.3"
    )

(output_dir / "chat_template.jinja").write_text(chat_template)
