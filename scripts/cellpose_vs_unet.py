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
    # Load baseline results
    baseline = json.load(open(RESULT_DIR / "baseline/summary.json"))
    full_met = json.load(open(RESULT_DIR / "full_metrics.json"))
    map_full = json.load(open(RESULT_DIR / "map_results_full.json"))

    # Methods to compare
    methods = {
        "Cellpose\n(baseline)": {
            "IoU":   baseline["mean_iou"],
            "F1@0.5": baseline["f1_05_mean"],
            "F1@0.7": baseline["f1_07_mean"],
            "mAP":   baseline.get("mAP", None),
            "color": "#888888",
        },
        "U-Net\n(No Aug)": {
            "IoU":   full_met["aug_no_augmentation"]["mean_iou"],
            "F1@0.5": full_met["aug_no_augmentation"]["f1_05"],
            "F1@0.7": full_met["aug_no_augmentation"]["f1_07"],
            "mAP":   map_full["aug_no_augmentation"]["mAP"],
            "color": "#4C72B0",
        },
        "U-Net\n(Geo Aug)": {
            "IoU":   full_met["aug_geometric_only"]["mean_iou"],
            "F1@0.5": full_met["aug_geometric_only"]["f1_05"],
            "F1@0.7": full_met["aug_geometric_only"]["f1_07"],
            "mAP":   map_full["aug_geometric_only"]["mAP"],
            "color": "#55A868",
        },
        "U-Net\n(Geo+Int Aug)": {
            "IoU":   full_met["aug_geometric_and_intensity"]["mean_iou"],
            "F1@0.5": full_met["aug_geometric_and_intensity"]["f1_05"],
            "F1@0.7": full_met["aug_geometric_and_intensity"]["f1_07"],
            "mAP":   map_full["aug_geometric_and_intensity"]["mAP"],
            "color": "#C44E52",
        },
        "U-Net\n(Full FT 500)": {
            "IoU":   full_met["tl_full_finetune_budget500"]["mean_iou"],
            "F1@0.5": full_met["tl_full_finetune_budget500"]["f1_05"],
            "F1@0.7": full_met["tl_full_finetune_budget500"]["f1_07"],
            "mAP":   map_full["tl_full_finetune_budget500"]["mAP"],
            "color": "#8172B2",
        },
    }

    labels = list(methods.keys())
    colors = [v["color"] for v in methods.values()]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("IoU",   "Test IoU",         axes[0,0]),
        ("F1@0.5","F1 Score (IoU=0.5)",axes[0,1]),
        ("F1@0.7","F1 Score (IoU=0.7)",axes[1,0]),
        ("mAP",   "mAP (IoU 0.5:0.95)",axes[1,1]),
    ]

    for metric, ylabel, ax in metrics:
        values = [v[metric] for v in methods.values()]
        # Handle None for Cellpose mAP
        plot_values = [v if v is not None else 0 for v in values]
        bars = ax.bar(labels, plot_values, color=colors, edgecolor='black', width=0.6)
        for bar, v in zip(bars, values):
            if v is not None:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
            else:
                ax.text(bar.get_x()+bar.get_width()/2, 0.02,
                        'N/A', ha='center', fontsize=10, color='gray')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} Comparison", fontsize=13)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        # Set ylim
        valid = [v for v in plot_values if v > 0]
        if valid:
            ax.set_ylim(min(valid)*0.9, max(valid)*1.08)

    plt.suptitle("Cellpose Baseline vs U-Net Configurations", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "cellpose_vs_unet.png", dpi=150, bbox_inches='tight')
    print("Saved cellpose_vs_unet.png")

    # Print summary table
    print("\n===== CELLPOSE vs U-NET COMPARISON =====")
    print(f"{'Method':<25} {'IoU':>6} {'F1@0.5':>7} {'F1@0.7':>7} {'mAP':>6}")
    print("-" * 55)
    for name, v in methods.items():
        name_clean = name.replace("\n", " ")
        mAP = f"{v['mAP']:.3f}" if v['mAP'] else "  N/A"
        print(f"{name_clean:<25} {v['IoU']:>6.3f} {v['F1@0.5']:>7.3f} {v['F1@0.7']:>7.3f} {mAP:>6}")

if __name__ == "__main__":
    main()
