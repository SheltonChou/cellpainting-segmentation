import json
import numpy as np
import pandas as pd
from pathlib import Path
from project_paths import RESULTS_ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = RESULTS_ROOT
VIS_DIR    = RESULT_DIR / "visualisations"

def main():
    full_met = json.load(open(RESULT_DIR / "full_metrics.json"))
    map_full = json.load(open(RESULT_DIR / "map_results_full.json"))
    baseline = json.load(open(RESULT_DIR / "baseline/summary.json"))
    baseline_full = json.load(open(RESULT_DIR / "baseline/cellpose_full_670_summary.json"))

    # Published results from Table 1 in planning report
    published = [
        {"Method": "CellProfiler",  "Year": 2018, "F1@0.5": 0.72,  "IoU": None, "mAP": None, "Type": "Published"},
        {"Method": "U-Net*",        "Year": 2015, "F1@0.5": 0.84,  "IoU": None, "mAP": None, "Type": "Published"},
        {"Method": "StarDist",      "Year": 2018, "F1@0.5": 0.83,  "IoU": None, "mAP": None, "Type": "Published"},
        {"Method": "Cellpose 2.0",  "Year": 2022, "F1@0.5": 0.861, "IoU": None, "mAP": None, "Type": "Published"},
    ]

    # Our results
    ours = [
        {
            "Method": "Cellpose (ours, DSB2018 test)",
            "Year": 2024,
            "F1@0.5": round(baseline["f1_05_mean"], 3),
            "IoU":    round(baseline["mean_iou"], 3),
            "mAP":    round(baseline.get("mAP", 0), 3),
            "Type": "Ours"
        },
        {
            "Method": "Cellpose (ours, DSB2018 full 670)",
            "Year": 2024,
            "F1@0.5": round(baseline_full["f1_05_mean"], 3),
            "IoU":    round(baseline_full["mean_iou"], 3),
            "mAP":    round(baseline_full["mAP"], 3),
            "Type": "Ours"
        },
        {
            "Method": "U-Net No Aug",
            "Year": 2024,
            "F1@0.5": full_met["aug_no_augmentation"]["f1_05"],
            "IoU":    full_met["aug_no_augmentation"]["mean_iou"],
            "mAP":    map_full["aug_no_augmentation"]["mAP"],
            "Type": "Ours"
        },
        {
            "Method": "U-Net Geo Aug",
            "Year": 2024,
            "F1@0.5": full_met["aug_geometric_only"]["f1_05"],
            "IoU":    full_met["aug_geometric_only"]["mean_iou"],
            "mAP":    map_full["aug_geometric_only"]["mAP"],
            "Type": "Ours"
        },
        {
            "Method": "U-Net Geo+Int Aug",
            "Year": 2024,
            "F1@0.5": full_met["aug_geometric_and_intensity"]["f1_05"],
            "IoU":    full_met["aug_geometric_and_intensity"]["mean_iou"],
            "mAP":    map_full["aug_geometric_and_intensity"]["mAP"],
            "Type": "Ours"
        },
        {
            "Method": "U-Net Full FT (500)",
            "Year": 2024,
            "F1@0.5": full_met["tl_full_finetune_budget500"]["f1_05"],
            "IoU":    full_met["tl_full_finetune_budget500"]["mean_iou"],
            "mAP":    map_full["tl_full_finetune_budget500"]["mAP"],
            "Type": "Ours"
        },
    ]

    df = pd.DataFrame(published + ours)
    df.to_csv(RESULT_DIR / "literature_comparison_full.csv", index=False)

    print("\n===== FULL LITERATURE COMPARISON =====")
    print(df.to_string(index=False))

    # Plot F1, IoU, mAP side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    colors = ["#AAAAAA" if t == "Published" else "#4C72B0" for t in df["Type"]]

    for ax, metric, ylabel in zip(axes,
        ["F1@0.5", "IoU", "mAP"],
        ["F1 Score (IoU=0.5)", "Test IoU", "mAP (IoU 0.5:0.95)"]):

        vals = df[metric].fillna(0)
        bars = ax.bar(range(len(df)), vals, color=colors, edgecolor='black', width=0.7)
        for i, (bar, v, t) in enumerate(zip(bars, df[metric], df["Type"])):
            if pd.notna(v) and v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f'{v:.3f}', ha='center', fontsize=7.5,
                        fontweight='bold' if t == "Ours" else 'normal')
            else:
                ax.text(bar.get_x()+bar.get_width()/2, 0.02,
                        'N/A', ha='center', fontsize=7, color='gray')

        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["Method"], rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} Comparison", fontsize=12)
        ax.set_ylim(0, 1.08)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.861, color='orange', linestyle='--', alpha=0.5, linewidth=1)

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(color='#AAAAAA', label='Published methods'),
        Patch(color='#4C72B0', label='Our methods (this work)'),
    ], loc='upper center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle("Full Comparison with Published Methods", fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "literature_comparison_full.png", dpi=150, bbox_inches='tight')
    print("\nSaved literature_comparison_full.png")

if __name__ == "__main__":
    main()
