from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a histogram from DirectTrainingSetStatistics JSON and mark "
            "average_absolute_advantage_cutoff."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to DirectTrainingSetStatistics JSON file",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to output image file (e.g. .png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    average_absolute_advantages_sorted = data["average_absolute_advantages_sorted"]
    average_absolute_advantage_cutoff = data["average_absolute_advantage_cutoff"]

    if not isinstance(average_absolute_advantages_sorted, list):
        raise ValueError("average_absolute_advantages_sorted must be a list")

    absolute_advantages = [float(x) for x in average_absolute_advantages_sorted]
    cutoff = float(average_absolute_advantage_cutoff)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    if absolute_advantages:
        num_bins = min(100, max(10, int(len(absolute_advantages) ** 0.5)))
        plt.hist(absolute_advantages, bins=num_bins, edgecolor="black", alpha=0.8)
    else:
        plt.text(
            0.5,
            0.5,
            "No data in average_absolute_advantages_sorted",
            ha="center",
            va="center",
        )

    plt.axvline(
        cutoff,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"absolute cutoff={cutoff:.6g}",
    )
    plt.xlabel("Average Absolute Segment Advantage")
    plt.ylabel("Count")
    plt.title("Histogram of Average Absolute Segment Advantage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_file)
    plt.close()

    print(f"Saved histogram to {args.output_file}")


if __name__ == "__main__":
    main()
