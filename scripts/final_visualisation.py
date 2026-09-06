import json
import numpy as np
import pandas as pd
from pathlib import Path
from project_paths import RESULTS_ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULT_DIR = RESULTS_ROOT
VIS_DIR    = RESULT_DIR / "visualisations"

def main():
    # Load all results
    unet     = json.load(open(RESULT_DIR / "unet_all_results.json"))
    map_full = json.load(open(RESULT_DIR / "map_results_full.json"))
    full_met = json.load(open(RESULT_DIR / "full_metrics.json"))
    ms       = json.load(open(RESULT_DIR / "multi_seed" / "multi_seed_results.json"))
    pseudo   = json.load(open(RESULT_DIR / "pseudo_label" / "pseudo_label_results.json"))
    indiv    = json.load(open(RESULT_DIR / "aug_individual" / "individual_aug_results.json"))

    # ── Figure 1: Complete augmentation comparison ────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Individual ops
    ops    = list(indiv.keys())
    ious   = list(indiv.values())
    colors = ["#888888","#4C72B0","#55A868","#C44E52","#8172B2","#CCB974","#64B5CD"]
    bars = axes[0].bar(ops, ious, color=colors, edgecolor='black')
    for bar, v in zip(bars, ious):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                     f'{v:.4f}', ha='center', fontsize=9)
    axes[0].set_ylabel("Validation IoU", fontsize=12)
    axes[0].set_title("Individual Augmentation Operations", fontsize=13)
    axes[0].set_ylim(0.855, 0.875)
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 3-group ablation with error bars from multi_seed
    groups = ["no_augmentation", "geometric_only", "geometric_and_intensity"]
    labels = ["No Aug", "Geo Aug", "Geo+Int Aug"]
    means  = [ms[f"aug_{g}"]["mean"] for g in groups]
    stds   = [ms[f"aug_{g}"]["std"]  for g in groups]
    colors3 = ["#4C72B0","#55A868","#C44E52"]
    bars = axes[1].bar(labels, means, yerr=stds, capsize=6,
                       color=colors3, edgecolor='black', width=0.5)
    for bar, v in zip(bars, means):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                     f'{v:.4f}', ha='center', fontsize=11)
    axes[1].set_ylabel("Validation IoU (mean ± std, 8 seeds)", fontsize=12)
    axes[1].set_title("Augmentation Ablation (8-seed)", fontsize=13)
    axes[1].set_ylim(0.82, 0.875)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "fig_augmentation_complete.png", dpi=150)
    print("Saved fig_augmentation_complete.png")

    # ── Figure 2: Transfer learning complete ─────────
    budgets = [40, 100, 250, 500]
    tl_cfgs = [
        ("full_finetune",    "Full Finetune",    "#4C72B0"),
        ("frozen_backbone",  "Frozen Backbone",  "#55A868"),
        ("partial_finetune", "Partial Finetune", "#C44E52"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # IoU learning curves with error bars
    for cfg, label, color in tl_cfgs:
        means = [ms.get(f"tl_{cfg}_budget{b}", {}).get("mean", None) for b in budgets]
        stds  = [ms.get(f"tl_{cfg}_budget{b}", {}).get("std",  None) for b in budgets]
        valid = [(b, m, s) for b, m, s in zip(budgets, means, stds) if m is not None]
        if valid:
            bs, ms_, ss = zip(*valid)
            axes[0].errorbar(bs, ms_, yerr=ss, marker='o', label=label,
                             color=color, linewidth=2, markersize=8, capsize=5)
    axes[0].set_xlabel("Annotation Budget", fontsize=12)
    axes[0].set_ylabel("Validation IoU (mean ± std, 8 seeds)", fontsize=12)
    axes[0].set_title("Transfer Learning: IoU vs Budget", fontsize=13)
    axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(budgets)

    # mAP learning curves
    for cfg, label, color in tl_cfgs:
        maps = [map_full.get(f"tl_{cfg}_budget{b}", {}).get("mAP", None) for b in budgets]
        valid = [(b, m) for b, m in zip(budgets, maps) if m is not None]
        if valid:
            bs, ms_ = zip(*valid)
            axes[1].plot(bs, ms_, marker='s', label=label,
                         color=color, linewidth=2, markersize=8)
    axes[1].set_xlabel("Annotation Budget", fontsize=12)
    axes[1].set_ylabel("mAP (IoU 0.5:0.95)", fontsize=12)
    axes[1].set_title("Transfer Learning: mAP vs Budget", fontsize=13)
    axes[1].legend(fontsize=11); axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(budgets)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "fig_transfer_learning_complete.png", dpi=150)
    print("Saved fig_transfer_learning_complete.png")

    # ── Figure 3: Pseudo-label threshold curve ────────
    thresholds = [int(k) for k in pseudo.keys()]
    val_ious   = [pseudo[k]["val_iou"] for k in pseudo.keys()]
    accept_rates = [pseudo[k]["accept_rate"] for k in pseudo.keys()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(thresholds, val_ious, marker='o', linewidth=2,
                 markersize=8, color="#4C72B0")
    axes[0].axhline(y=unet["aug_geometric_only"], color='red',
                    linestyle='--', label=f'Geo Aug baseline ({unet["aug_geometric_only"]:.4f})')
    for x, y in zip(thresholds, val_ious):
        axes[0].annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=9)
    axes[0].set_xlabel("Min cells threshold", fontsize=12)
    axes[0].set_ylabel("Validation IoU", fontsize=12)
    axes[0].set_title("Pseudo-label: Threshold vs Performance", fontsize=13)
    axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3)

    axes[1].bar([str(t) for t in thresholds], accept_rates,
                color="#55A868", edgecolor='black')
    axes[1].set_xlabel("Min cells threshold", fontsize=12)
    axes[1].set_ylabel("Accept Rate (%)", fontsize=12)
    axes[1].set_title("Pseudo-label: Accept Rate vs Threshold", fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "fig_pseudo_label_complete.png", dpi=150)
    print("Saved fig_pseudo_label_complete.png")

    print("\nAll visualisations saved.")

if __name__ == "__main__":
    main()
