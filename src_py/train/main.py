from __future__ import annotations

import argparse

from .engine import TrainConfig, train_with_deepspeed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train causal LM with DeepSpeed ZeRO-3")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--tokenized-sqlite-path", type=str, required=True)
    parser.add_argument("--batch-sqlite-path", type=str, required=True)
    parser.add_argument("--deepspeed-config-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--pad-token-id", type=int, required=True)
    parser.add_argument("--advantage-clip", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--grad-accum-steps", type=int, required=True)
    parser.add_argument("--log-interval-steps", type=int, required=True)
    parser.add_argument("--save-interval-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = TrainConfig(
        model_name_or_path=args.model_name_or_path,
        tokenized_sqlite_path=args.tokenized_sqlite_path,
        batch_sqlite_path=args.batch_sqlite_path,
        deepspeed_config_path=args.deepspeed_config_path,
        output_dir=args.output_dir,
        pad_token_id=args.pad_token_id,
        advantage_clip=args.advantage_clip,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        grad_accum_steps=args.grad_accum_steps,
        log_interval_steps=args.log_interval_steps,
        save_interval_steps=args.save_interval_steps,
        seed=args.seed,
    )
    train_with_deepspeed(config)


if __name__ == "__main__":
    main()
