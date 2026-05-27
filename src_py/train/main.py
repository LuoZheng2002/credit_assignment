from __future__ import annotations

import argparse

from .engine import TrainConfig, train_with_deepspeed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train causal LM with LoRA or FSDP")
    parser.add_argument("--training-plan", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--training-trajectory-sqlite-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--advantage-clip", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--grad-accum-steps", type=int, required=True)
    parser.add_argument("--log-interval-steps", type=int, required=True)
    parser.add_argument("--save-interval-steps", type=int, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    parser.add_argument("--lora-dropout", type=float, required=True)
    parser.add_argument("--lora-target-modules-csv", type=str, required=True)
    parser.add_argument("--resume-checkpoint-tag", type=str, required=False, default="auto")
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = TrainConfig(
        training_plan=args.training_plan,
        model_name_or_path=args.model_name_or_path,
        training_trajectory_sqlite_path=args.training_trajectory_sqlite_path,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        advantage_clip=args.advantage_clip,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        grad_accum_steps=args.grad_accum_steps,
        log_interval_steps=args.log_interval_steps,
        save_interval_steps=args.save_interval_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules_csv=args.lora_target_modules_csv,
        resume_checkpoint_tag=args.resume_checkpoint_tag,
        seed=args.seed,
    )
    train_with_deepspeed(config)


if __name__ == "__main__":
    main()
