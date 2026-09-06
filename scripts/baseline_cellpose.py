import os, sys, glob, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from project_paths import DSB_DIR, RESULTS_ROOT
from tqdm import tqdm
from cellpose import models, io
from skimage import io as skio, color
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 路径 ──────────────────────────────────────────────
DATA_DIR   = DSB_DIR
RESULT_DIR = RESULTS_ROOT / "baseline"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── 工具函数 ──────────────────────────────────────────
def load_image(path):
    img = skio.imread(str(path))
    if img.ndim == 3:
        img = img[..., :3]
        img = color.rgb2gray(img)
    return (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

def load_gt_masks(mask_dir):
    masks = sorted(glob.glob(str(mask_dir / "*.png")))
    if not masks:
        return np.zeros((1,1), dtype=np.int32)
    ref = skio.imread(masks[0])
    combined = np.zeros(ref.shape[:2], dtype=np.int32)
    for i, m in enumerate(masks, 1):
        arr = skio.imread(m)
        if arr.ndim == 3: arr = arr[..., 0]
        combined[arr > 0] = i
    return combined

def compute_iou_matrix(pred, gt):
    pred_ids = np.unique(pred[pred > 0])
    gt_ids   = np.unique(gt[gt > 0])
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return np.zeros((len(pred_ids), len(gt_ids)))
    iou = np.zeros((len(pred_ids), len(gt_ids)))
    for i, pi in enumerate(pred_ids):
        pm = pred == pi
        for j, gi in enumerate(gt_ids):
            gm = gt == gi
            inter = (pm & gm).sum()
            union = (pm | gm).sum()
            iou[i, j] = inter / union if union > 0 else 0
    return iou

def compute_metrics(pred, gt, iou_thresh=0.5):
    iou = compute_iou_matrix(pred, gt)
    if iou.size == 0:
        n_gt = len(np.unique(gt[gt>0]))
        return 0.0, 0.0, 0.0, n_gt, 0
    matched = iou >= iou_thresh
    tp = min(matched.any(axis=1).sum(), matched.any(axis=0).sum())
    fp = (~matched.any(axis=1)).sum()
    fn = (~matched.any(axis=0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
    n_pred = len(np.unique(pred[pred>0]))
    n_gt   = len(np.unique(gt[gt>0]))
    return float(prec), float(rec), float(f1), n_gt, n_pred

def compute_mean_iou(pred, gt):
    iou = compute_iou_matrix(pred, gt)
    if iou.size == 0: return 0.0
    return float(iou.max(axis=0).mean()) if iou.shape[1] > 0 else 0.0

# ── 主实验 ────────────────────────────────────────────
def main():
    print("Loading Cellpose model...")
    model = models.CellposeModel(gpu=torch.cuda.is_available(), model_type="cyto2")

    image_ids = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(image_ids)} images")

    records = []
    for img_id in tqdm(image_ids, desc="Evaluating"):
        img_path  = list((DATA_DIR / img_id / "images").glob("*.png"))[0]
        mask_dir  = DATA_DIR / img_id / "masks"

        image  = load_image(img_path)
        gt     = load_gt_masks(mask_dir)

        masks, flows, styles = model.eval(
            image, diameter=None, channels=[0,0],
            flow_threshold=0.4, cellprob_threshold=0.0
        )

        miou = compute_mean_iou(masks, gt)
        p5, r5, f5, n_gt, n_pred = compute_metrics(masks, gt, 0.5)
        p7, r7, f7, _,    _      = compute_metrics(masks, gt, 0.7)

        records.append({
            "image_id": img_id,
            "mean_iou": miou,
            "precision_05": p5, "recall_05": r5, "f1_05": f5,
            "precision_07": p7, "recall_07": r7, "f1_07": f7,
            "n_gt_cells": n_gt, "n_pred_cells": n_pred,
        })

    df = pd.DataFrame(records)
    df.to_csv(RESULT_DIR / "cellpose_baseline_results.csv", index=False)

    summary = {
        "mean_iou":    round(df.mean_iou.mean(), 4),
        "std_iou":     round(df.mean_iou.std(),  4),
        "f1_05_mean":  round(df.f1_05.mean(),    4),
        "f1_05_std":   round(df.f1_05.std(),     4),
        "f1_07_mean":  round(df.f1_07.mean(),    4),
        "f1_07_std":   round(df.f1_07.std(),     4),
        "n_images":    len(df),
    }
    with open(RESULT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== BASELINE RESULTS =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 保存图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(axes,
        ["mean_iou", "f1_05", "f1_07"],
        ["Mean IoU", "F1 (IoU=0.5)", "F1 (IoU=0.7)"]):
        ax.hist(df[col], bins=20, edgecolor='black')
        ax.axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean={df[col].mean():.3f}')
        ax.set_title(title); ax.set_xlabel("Score"); ax.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "baseline_distribution.png", dpi=150)
    print(f"\nResults saved to {RESULT_DIR}")

if __name__ == "__main__":
    main()
