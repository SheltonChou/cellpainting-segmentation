#!/usr/bin/env python3
"""Recreate the three final result figures that use Route B metrics.

This script reads the archived final evaluation rather than older intermediate
metric files. It therefore requires neither the raw images nor model weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from project_paths import RESULTS_ROOT


ROUTE_DIR = RESULTS_ROOT / "routeB_instance_eval"
VIS_DIR = RESULTS_ROOT / "visualisations"
MODEL_LABELS = {
    "aug_no_augmentation": "No augmentation",
    "aug_geometric_only": "Geometric",
    "aug_geometric_and_intensity": "Geometric + intensity",
    "tl_full_finetune_budget500": "Full fine-tuning, full",
}
TL_LABELS = {
    "full_finetune": "Full fine-tuning",
    "frozen_backbone": "Frozen backbone",
    "partial_finetune": "Partial fine-tuning",
}
COLORS = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728"]


def plot_ap_thresholds(payload: dict) -> None:
    summaries = payload["summaries"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for (model, label), color in zip(MODEL_LABELS.items(), COLORS):
        points = summaries[model]["ap_per_threshold"]
        x = np.array([float(key) for key in points])
        y = np.array(list(points.values()))
        ax.plot(x, y, marker="o", linewidth=2, color=color, label=label)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Average precision")
    ax.set_xticks(np.arange(0.50, 1.00, 0.05))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIS_DIR / "routeB_ap_thresholds.png", dpi=150)
    plt.close(fig)


def plot_transfer_map(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    budgets = [40, 100, 250, 500]
    for (method, label), color in zip(TL_LABELS.items(), COLORS):
        values = []
        for budget in budgets:
            model = f"tl_{method}_budget{budget}"
            values.append(float(summary.loc[model, "map_50_95"]))
        ax.plot(budgets, values, marker="o", linewidth=2, color=color, label=label)
    ax.set_xlabel("Labelled training images")
    ax.set_ylabel("Mask mAP (IoU 0.50-0.95)")
    ax.set_xticks(budgets)
    ax.set_xticklabels(["40", "100", "250", "Full\n(n=468)"])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIS_DIR / "tl_routeB_map.png", dpi=150)
    plt.close(fig)


def density_record(data: pd.DataFrame) -> dict:
    x = data["n_gt_instances"].to_numpy()
    y = data["binary_iou"].to_numpy()
    labels = ["1-5", "6-15", "16-30", "31-50", "51-100", "100+"]
    data["density_bin"] = pd.cut(
        data["n_gt_instances"],
        bins=[0, 5, 15, 30, 50, 100, np.inf],
        labels=labels,
    )

    grouped = data.groupby("density_bin", observed=False)
    means = grouped[["n_gt_instances", "binary_iou"]].mean()
    counts = grouped.size()
    correlation, p_value = stats.pearsonr(x, y)
    fisher_z = np.arctanh(correlation)
    fisher_se = 1.0 / np.sqrt(len(data) - 3)
    ci = np.tanh([fisher_z - 1.96 * fisher_se, fisher_z + 1.96 * fisher_se])
    return {
        "n_test_images": int(len(data)),
        "pearson_r": float(correlation),
        "p_value": float(p_value),
        "fisher_ci95": [float(ci[0]), float(ci[1])],
        "bin_mean_iou": {label: float(means.loc[label, "binary_iou"]) for label in labels},
        "bin_counts": {label: int(counts.loc[label]) for label in labels},
    }


def plot_density(per_image: pd.DataFrame) -> dict:
    records = {}
    for model in [
        "aug_no_augmentation",
        "aug_geometric_only",
        "tl_full_finetune_budget500",
    ]:
        records[model] = density_record(per_image[per_image["model"] == model].copy())

    data = per_image[per_image["model"] == "aug_geometric_only"].copy()
    x = data["n_gt_instances"].to_numpy()
    y = data["binary_iou"].to_numpy()
    labels = ["1-5", "6-15", "16-30", "31-50", "51-100", "100+"]
    data["density_bin"] = pd.cut(
        data["n_gt_instances"], bins=[0, 5, 15, 30, 50, 100, np.inf], labels=labels
    )
    grouped = data.groupby("density_bin", observed=False)
    means = grouped[["n_gt_instances", "binary_iou"]].mean()
    counts = grouped.size()

    fig, ax = plt.subplots(figsize=(10, 6))
    points = ax.scatter(x, y, alpha=0.55, s=30, label="Test images")
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(x.min(), x.max(), 200)
    line, = ax.plot(line_x, slope * line_x + intercept, linewidth=2, label="Linear fit")
    bins = ax.scatter(
        means["n_gt_instances"], means["binary_iou"], marker="D", s=140,
        color="#FF7F0E", label="Bin mean", zorder=4,
    )
    for label in labels:
        row = means.loc[label]
        ax.annotate(
            f"{label} (n={counts.loc[label]})", (row["n_gt_instances"], row["binary_iou"]),
            xytext=(6, 5), textcoords="offset points",
        )
    ax.set_xlabel("Ground-truth nuclei per image")
    ax.set_ylabel("Binary mask IoU")
    ax.set_ylim(0, 1.02)
    ax.grid(False)
    ax.legend([points, line, bins], ["Test images", "Linear fit", "Bin mean"])
    fig.tight_layout()
    fig.savefig(VIS_DIR / "density_test_only.png", dpi=150)
    plt.close(fig)
    return {"split_seed": 42, "n_test_images": int(len(data)), "models": records}


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads((ROUTE_DIR / "routeB_results.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(ROUTE_DIR / "routeB_summary.csv").set_index("model")
    per_image = pd.read_csv(ROUTE_DIR / "routeB_per_image_metrics.csv")
    plot_ap_thresholds(payload)
    plot_transfer_map(summary)
    density_summary = plot_density(per_image)
    output = RESULTS_ROOT / "density_analysis" / "density_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(density_summary, indent=2), encoding="utf-8")
    print("Recreated Route B AP, transfer learning and test density figures.")


if __name__ == "__main__":
    main()
