"""
Create a dumbbell plot that focuses only on stealth injections: for each model
we show how accuracy moves from the clean baseline (GT) to the stealth injected
result and the stealth-immune recovery.

Usage:
    python accuracy_dumbbell_plot.py --output Results/Plots/accuracy_dumbbell_plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "Results" / "Plots" / "ALL_MODELS_FULL_METRICS.csv"
DEFAULT_OUTPUT = ROOT / "Results" / "Plots" / "accuracy_dumbbell_plot.png"

# Model order chosen to match storyboard references (Gemini 3 at the top, etc.)
DISPLAY_MODELS = [
    "Gemini 3 Pro Preview",
    "Phi-4 Multimodal Instruct",
    "Qwen 3 VL 8B Thinking",
    "GPT-5",
    "GPT-4o mini",
    "GPT-5 nano",
    "Gemini 2.5 Flash",
    "Nemotron Nano 12B 2 VL",
    "Claude Sonnet 4.5",
    "MedGemma 4B IT",
]

CSV_MODEL_MAP = {
    "Claude Sonnet 4.5": "CLAUDE",
    "GPT-4o mini": "GPT4MINI",
    "GPT-5": "GPT5",
    "GPT-5 nano": "GPT5NANO",
    "Gemini 2.5 Flash": "Gemini 2.5 Flash",
    "Gemini 3 Pro Preview": "GEMINI",
    "Nemotron Nano 12B 2 VL": "NEMOTRON",
    "Phi-4 Multimodal Instruct": "PHI",
    "Qwen 3 VL 8B Thinking": "QWEN",
    "MedGemma 4B IT": "MEDGEMMA",
}



def get_value(frame: pd.DataFrame, scenario: str, column: str) -> float:
    rows = frame.loc[frame["Scenario"] == scenario, column]
    if rows.empty:
        raise ValueError(f"No data for scenario '{scenario}' in column '{column}'.")
    return float(rows.iloc[0])


def build_plot(df: pd.DataFrame, metric: str, output_path: Path | None) -> None:
    sns.set_theme(style="whitegrid")

    baseline_color = "#17223b"
    stealth_injected_color = "#e76f51"
    stealth_immune_color = "#2a9d8f"
    line_color = "#9aa5b1"
    label_config = {
        "baseline":         {"dx": 0.02, "dy": -0.06, "ha": "left", "va": "bottom", "weight": "bold"},
        "stealth_injected": {"dx": 0.01, "dy": 0.18, "ha": "right", "va": "bottom", "weight": "bold"},
        "stealth_immune":   {"dx": 0.0, "dy": -0.18, "ha": "center", "va": "top", "weight": "bold"},
    }

    fig, ax = plt.subplots(figsize=(18, 10))

    y_positions = list(range(len(DISPLAY_MODELS)))
    y_ticks = []
    y_labels = []

    for idx, model in enumerate(DISPLAY_MODELS):
        key = CSV_MODEL_MAP[model]
        subset = df[df["Model"] == key]

        baseline = get_value(subset, "Gt", metric)
        stealth_injected = get_value(subset, "Stealth Injected", metric)
        stealth_immune = get_value(subset, "Stealth Injection Immune", metric)

        y = len(DISPLAY_MODELS) - 1 - idx

        # connector baseline -> stealth immune (solid) and immune -> stealth injected (dashed)
        ax.hlines(y, min(baseline, stealth_immune), max(baseline, stealth_immune), color=line_color, linewidth=2.5)
        ax.hlines(y, min(stealth_injected, stealth_immune), max(stealth_injected, stealth_immune), color=line_color, linewidth=2.5, linestyles="--")

        ax.scatter(baseline, y, s=120, color=baseline_color, edgecolor="white", linewidth=1.0, zorder=3)
        ax.scatter(stealth_immune, y, s=130, marker="D", color=stealth_immune_color, edgecolor="white", linewidth=1.0, zorder=4)
        ax.scatter(stealth_injected, y, s=140, marker="X", color=stealth_injected_color, edgecolor="white", linewidth=1.2, zorder=5)

        def annotate(value: float, color: str, key: str):
            cfg = label_config[key]
            ax.text(
                value + cfg["dx"],
                y + cfg["dy"],
                f"{value:.2f}",
                fontsize=18,
                color=color,
                ha=cfg["ha"],
                va=cfg["va"],
                fontweight=cfg["weight"],
            )

        if baseline > 0:
            annotate(baseline, baseline_color, "baseline")
        if stealth_injected > 0:
            annotate(stealth_injected, stealth_injected_color, "stealth_injected")
        if stealth_immune > 0:
            annotate(stealth_immune, stealth_immune_color, "stealth_immune")

        y_ticks.append(y)
        y_labels.append(model)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontweight="bold", fontsize=18)
    ax.set_xlabel("Accuracy", fontsize=18)
    ax.set_title("Impact of Stealth Injection on Accuracy", fontsize=22, fontweight="bold", pad=18)
    ax.set_xlim(0.25, 0.80)
    ax.tick_params(axis='x', labelsize=18)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=baseline_color, markeredgecolor="white", markersize=12, label="Baseline (GT)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=stealth_injected_color, markeredgecolor="white", markersize=12, label="Stealth Injected"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=stealth_immune_color, markeredgecolor="white", markersize=12, label="Stealth Immune"),
        Line2D([0], [0], color=line_color, linewidth=2.5, label="Baseline ↔ Stealth Immune"),
        Line2D([0], [0], color=line_color, linewidth=2.5, linestyle="--", label="Immune ↔ Stealth Injected"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=16)

    fig.tight_layout()

    output = output_path or DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved stealth-only accuracy plot to {output}")

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stealth-only accuracy dumbbell plot.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="Path to ALL_MODELS_FULL_METRICS.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to save the generated figure (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    build_plot(df, metric="Accuracy", output_path=args.output)


if __name__ == "__main__":
    main()


