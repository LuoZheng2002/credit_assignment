from pathlib import Path

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True
)
output_dir = Path("tokenizers/llama31")
tokenizer.save_pretrained(output_dir)

chat_template = tokenizer.chat_template
if chat_template is None:
    raise RuntimeError("Tokenizer chat_template is missing for meta-llama/Llama-3.1-8B-Instruct")

(output_dir / "chat_template.jinja").write_text(chat_template)
