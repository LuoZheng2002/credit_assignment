import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-4b-it", trust_remote_code=True
)
output_dir = Path("tokenizers/gemma3")
tokenizer.save_pretrained(output_dir)

chat_template = tokenizer.chat_template
if chat_template is None:
    raise RuntimeError("Tokenizer chat_template is missing for google/gemma-3-4b-it")

(output_dir / "chat_template.jinja").write_text(chat_template)
