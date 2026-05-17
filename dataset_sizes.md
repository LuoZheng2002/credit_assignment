# Dataset sizes and split info (Hugging Face)

Source used for counts: Hugging Face Datasets Server `size` endpoint (`https://datasets-server.huggingface.co/size?dataset=<dataset_id>`), checked on 2026-05-16.

| Dataset name | Hugging Face dataset ID | Total rows | Train rows | Test rows | Split note |
|---|---|---:|---:|---:|---|
| DeepMath | `mlfoundations-dev/deepmath` | 309,330 | 309,330 | 0 | Only a `train` split is published |
| MATH | `EleutherAI/hendrycks_math` | 12,500 | 7,500 | 5,000 | 7 subject configs; each has `train` + `test` |
| GSM8K | `openai/gsm8k` | 17,584 | 14,946 | 2,638 | Two configs (`main`, `socratic`), each has 7,473 train / 1,319 test |
| AIME25 | `math-ai/aime25` | 30 | 0 | 30 | Only a `test` split is published |
| AMC23 | `math-ai/amc23` | 40 | 0 | 40 | Only a `test` split is published |

## Train-test split ratios

- DeepMath: `100% train / 0% test` (no test split on HF for this dataset ID)
- MATH: `60% train / 40% test`
- GSM8K: `85% train / 15% test` (same ratio in both `main` and `socratic` configs)
- AIME25: `0% train / 100% test`
- AMC23: `0% train / 100% test`
