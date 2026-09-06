import os, json, glob
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from project_paths import RESULTS_ROOT, RXRX1_IMAGES_DIR
from tqdm import tqdm
from cellpose import models as cp_models
from skimage import io as skio, color
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RXRX1_DIR  = RXRX1_IMAGES_DIR
RESULT_DIR = RESULTS_ROOT / "baseline_rxrx1"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Use 200 images for baseline evaluation
MAX_IMAGES = 200

def load_image(path):
    img = skio.imread(str(path))
    if img.ndim == 3: img = img[..., 0]
    return img.astype(np.uint8)

def compute_metrics(pred, gt, iou_thresh=0.5):
    pred_ids = np.unique(pred[pred > 0])
    gt_ids   = np.unique(gt[gt > 0])
    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return 0.0, 0.0, 0.0
    iou_mat = np.zeros((len(pred_ids), len(gt_ids)))
    for i, pi in enumerate(pred_ids):
        pm = pred == pi
        for j, gi in enumerate(gt_ids):
            gm = gt == gi
            inter = (pm & gm).sum()
            union = (pm | gm).sum()
            iou_mat[i,j] = inter / union if union > 0 else 0
    matched = iou_mat >= iou_thresh
    tp = min(matched.any(axis=1).sum(), matched.any(axis=0).sum())
    fp = (~matched.any(axis=1)).sum()
    fn = (~matched.any(axis=0)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2*prec*rec / (prec+rec+1e-8)
    return float(prec), float(rec), float(f1)

def compute_mean_iou(pred, gt):
    pred_ids = np.unique(pred[pred > 0])
    gt_ids   = np.unique(gt[gt > 0])
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return 0.0
    ious = []
    for gi in gt_ids:
        gm = gt == gi
        best = 0.0
        for pi in pred_ids:
            pm = pred == pi
            inter = (pm & gm).sum()
            union = (pm | gm).sum()
            best = max(best, inter/union if union > 0 else 0)
        ious.append(best)
    return float(np.mean(ious))

def main():
    # Get w1 (nucleus) images
    rxrx1_paths = sorted(RXRX1_DIR.rglob("*_w1.png"))[:MAX_IMAGES]
    print(f"Found {len(rxrx1_paths)} RxRx1 images")

    model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), model_type='nuclei')
    records = []

    for path in tqdm(rxrx1_paths, desc="Cellpose on RxRx1"):
        img = load_image(path)
        masks, flows, _ = model.eval(img, diameter=15, channels=[0,0])
        n_cells = len(np.unique(masks)) - 1
        fg_frac = float((masks > 0).mean())

        # No GT masks available for RxRx1, so report detection stats
        records.append({
            "image_id":   path.name,
            "experiment": path.parent.parent.name,
            "n_cells":    n_cells,
            "fg_fraction": fg_frac,
        })

    df = pd.DataFrame(records)
    df.to_csv(RESULT_DIR / "cellpose_rxrx1_detection.csv", index=False)

    summary = {
        "n_images":       len(df),
        "mean_cells":     round(df.n_cells.mean(), 2),
        "std_cells":      round(df.n_cells.std(), 2),
        "mean_fg_frac":   round(df.fg_fraction.mean(), 4),
        "experiments":    df.experiment.nunique(),
    }
    with open(RESULT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot cell count distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df.n_cells, bins=30, edgecolor='black', color='steelblue')
    axes[0].set_xlabel("Number of cells detected"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"Cell count distribution (mean={summary['mean_cells']:.0f})")
    axes[1].hist(df.fg_fraction, bins=30, edgecolor='black', color='coral')
    axes[1].set_xlabel("Foreground fraction"); axes[1].set_ylabel("Count")
    axes[1].set_title(f"Foreground fraction (mean={summary['mean_fg_frac']:.3f})")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "rxrx1_detection_stats.png", dpi=150)

    print("\n===== RxRx1 CELLPOSE BASELINE =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
