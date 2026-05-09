"""
Create a sensitivity-specificity displacement plot that mimics the storyboard
shared by the team: two panels showing how each model moves in the
sensitivity-specificity plane when moving from baseline (GT) to visible or
stealth injection scenarios.

Usage:
    python sensitivity_specificity_displacement.py \
        --output Results/Plots/sensitivity_specificity_displacement.png
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
DEFAULT_OUTPUT = ROOT / "Results" / "Plots" / "sensitivity_specificity_displacement.png"
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

def build_plot(df: pd.DataFrame, output_path: Path | None) -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.linewidth'] = 0.8
    plt.rcParams['grid.alpha'] = 0.3

    baseline_color = "#1a237e"  # Deep indigo
    visible_color = "#e53935"   # Vibrant red
    stealth_color = "#ffa726"   # Warm orange
    connector_color = "#424242"  # Dark gray
    inversion_fill = "#ffebee"   # Light pink
    text_color = "#212121"       # Almost black

    records = []

    def scenario_values(frame: pd.DataFrame, scenario: str) -> tuple[float, float]:
        rows = frame.loc[frame["Scenario"] == scenario, ["Specificity", "Sensitivity"]]
        if rows.empty:
            raise ValueError(f"No data for scenario '{scenario}'.")
        spec = float(rows.iloc[0]["Specificity"])
        sens = float(rows.iloc[0]["Sensitivity"])
        return spec, sens

    for idx, model in enumerate(sorted(DISPLAY_MODELS), start=1):
        key = CSV_MODEL_MAP[model]
        subset = df[df["Model"] == key]
        baseline_spec, baseline_sens = scenario_values(subset, "Gt")
        visible_spec, visible_sens = scenario_values(subset, "Injected")
        stealth_spec, stealth_sens = scenario_values(subset, "Stealth Injected")

        records.append(
            {
                "index": idx,
                "name": model,
                "baseline_spec": baseline_spec,
                "baseline_sens": baseline_sens,
                "visible_spec": visible_spec,
                "visible_sens": visible_sens,
                "stealth_spec": stealth_spec,
                "stealth_sens": stealth_sens,
            }
        )

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()  # Convert to 1D array for easier indexing

    def configure_axis(ax, panel_title: str, show_xlabel: bool = True):
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        if show_xlabel:
            ax.set_xlabel("Specificity", fontsize=14, fontweight="semibold", color=text_color, labelpad=8)
        else:
            ax.set_xlabel("", fontsize=14, fontweight="semibold", color=text_color, labelpad=8)
        ax.set_ylabel("Sensitivity", fontsize=14, fontweight="semibold", color=text_color, labelpad=8)
        ax.set_title(panel_title, fontsize=16, fontweight="bold", pad=8, color=text_color)
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.4, color="#9e9e9e")
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.0, length=5)

        x_coords = [0.0, 1.0]
        y_coords = [1.0, 0.0]
        ax.fill_between(x_coords, y_coords, 0, color=inversion_fill, alpha=0.6, zorder=0)
        ax.plot(x_coords, y_coords, linestyle="--", color="#757575", linewidth=2.0, 
                label="Random Guessing", alpha=0.8, zorder=0)
        ax.text(
            0.5,
            0.08,
            "Adversarial Inversion\n(Worse than Random)",
            color="#c62828",
            fontsize=11,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#c62828", 
                     linewidth=1.5, alpha=0.9),
            zorder=10,
        )

    def plot_panel(ax, injection_key: str, color: str, marker: str, title: str, show_xlabel: bool = True):
        configure_axis(ax, title, show_xlabel)

        def baseline_label_offset(index: int) -> tuple[float, float]:
            offsets = [
                (0.015, 0.025),
                (-0.018, 0.025),
                (0.025, -0.018),
                (-0.025, -0.025),
                (0.018, 0.018),
                (-0.020, -0.020),
            ]
            return offsets[index % len(offsets)]

        for entry in records:
            base_spec = entry["baseline_spec"]
            base_sens = entry["baseline_sens"]
            inj_spec = entry[f"{injection_key}_spec"]
            inj_sens = entry[f"{injection_key}_sens"]

            # Add small offset for zero values to make them visible
            offset_spec = inj_spec if inj_spec > 0.001 else 0.01
            offset_sens = inj_sens if inj_sens > 0.001 else 0.01

            ax.plot(
                [base_spec, offset_spec],
                [base_sens, offset_sens],
                color=connector_color,
                linewidth=2.2,
                zorder=1,
                alpha=0.7,
            )

            ax.scatter(
                base_spec,
                base_sens,
                s=100,
                color=baseline_color,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
                alpha=0.95,
            )
            
            ax.scatter(
                offset_spec,
                offset_sens,
                s=120,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=1.5,
                zorder=4,
                alpha=0.95,
            )

            dx, dy = baseline_label_offset(entry["index"])
            ax.text(
                base_spec + dx,
                base_sens + dy,
                str(entry["index"]),
                fontsize=11,
                fontweight="bold",
                color="#e53935",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", 
                         edgecolor="#e53935", linewidth=1.0, alpha=0.9),
                zorder=5,
            )

    plot_panel(axes[0], "visible", visible_color, "^", "(a) Baseline → Visible Injection", show_xlabel=False)
    plot_panel(axes[1], "stealth", stealth_color, "v", "(b) Baseline → Stealth Injection", show_xlabel=True)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=baseline_color, 
               markeredgecolor="white", markersize=13, label="Baseline (GT)", markeredgewidth=1.5),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=visible_color, 
               markeredgecolor="white", markersize=13, label="Visible Injected", markeredgewidth=1.5),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=stealth_color, 
               markeredgecolor="white", markersize=13, label="Stealth Injected", markeredgewidth=1.5),
        Line2D([0], [0], color=connector_color, linewidth=2.5, label="Attack Displacement"),
        Line2D([0], [0], linestyle="--", color="#757575", linewidth=2.0, label="Random Guessing"),
    ]
    
    index_text = "\n".join(f"{entry['index']}: {entry['name']}" for entry in records)
    
    # Place legend in the top panel's upper right corner, slightly shifted left
    legend = axes[0].legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.78, 1.0), 
                           frameon=True, fontsize=13, fancybox=True, shadow=True, 
                           title="Model Index:\n" + index_text, title_fontsize=12,
                           edgecolor="#bdbdbd", facecolor="white", framealpha=0.95)
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_color(text_color)

    fig.suptitle(
        "Attack Displacement in Sensitivity-Specificity Space",
        fontsize=20,
        fontweight="bold",
        y=0.995,
        color=text_color,
    )
    plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.08, hspace=0.3)

    output_path = output_path or DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved displacement plot to {output_path}")

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create sensitivity-specificity displacement plot.")
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
    build_plot(df, output_path=args.output)


if __name__ == "__main__":
    main()


