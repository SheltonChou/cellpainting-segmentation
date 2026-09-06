import json, glob, random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from project_paths import DSB_DIR, RESULTS_ROOT
from tqdm import tqdm
from skimage import io as skio, color, exposure
from skimage.transform import resize
from cellpose import models as cp_models

SEED = 42
random.seed(SEED); np.random.seed(SEED)

RESULT_DIR = RESULTS_ROOT / "baseline"
IMG_SIZE   = 256
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

def load_image(path):
    img = skio.imread(str(path))
    if img.ndim == 3:
        img = img[..., :3]
        img = color.rgb2gray(img)
    return exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)

def load_gt_mask(mask_dir):
    mask_paths = sorted(glob.glob(str(mask_dir / "*.png")))
    if not mask_paths: return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    ref = skio.imread(mask_paths[0])
    combined = np.zeros(ref.shape[:2], dtype=np.float32)
    for mp in mask_paths:
        m = skio.imread(mp)
        if m.ndim == 3: m = m[..., 0]
        combined[m > 0] = 1.0
    return combined

def compute_metrics(pred, target, iou_thresh=0.5):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    iou   = inter / (union + 1e-8)
    tp = 1.0 if iou >= iou_thresh else 0.0
    fp = 1.0 - tp; fn = 1.0 - tp
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2*prec*rec/(prec+rec+1e-8)
    return float(iou), float(f1)

def main():
    all_ids = sorted([d.name for d in DSB_DIR.iterdir() if d.is_dir()])
    print(f"Total images: {len(all_ids)}")

    model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), model_type='cyto')
    records = []

    for img_id in tqdm(all_ids, desc="Cellpose full evaluation"):
        img_path = list((DSB_DIR / img_id / "images").glob("*.png"))[0]
        img  = load_image(img_path)
        mask = load_gt_mask(DSB_DIR / img_id / "masks")
        mask_r = resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)

        masks_pred, _, _ = model.eval(img, diameter=None, channels=[0,0])
        pred = resize((masks_pred > 0).astype(np.float32),
                      (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)

        iou, f1_05 = compute_metrics(pred, mask_r, 0.5)
        _,   f1_07 = compute_metrics(pred, mask_r, 0.7)

        records.append({
            "image_id": img_id,
            "mean_iou": iou,
            "f1_05":    f1_05,
            "f1_07":    f1_07,
        })

    df = pd.DataFrame(records)
    df.to_csv(RESULT_DIR / "cellpose_full_670_results.csv", index=False)

    # Compute mAP
    aps = []
    for iou_thresh in IOU_THRESHOLDS:
        f1s = []
        for img_id in tqdm(all_ids, desc=f"mAP IoU={iou_thresh:.2f}", leave=False):
            img_path = list((DSB_DIR / img_id / "images").glob("*.png"))[0]
            img  = load_image(img_path)
            mask = load_gt_mask(DSB_DIR / img_id / "masks")
            mask_r = resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
            masks_pred, _, _ = model.eval(img, diameter=None, channels=[0,0])
            pred = resize((masks_pred > 0).astype(np.float32),
                          (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
            _, f1 = compute_metrics(pred, mask_r, iou_thresh)
            f1s.append(f1)
        aps.append(float(np.mean(f1s)))

    mAP = float(np.mean(aps))

    summary_full = {
        "n_images":    len(df),
        "mean_iou":    round(float(df.mean_iou.mean()), 4),
        "std_iou":     round(float(df.mean_iou.std()),  4),
        "f1_05_mean":  round(float(df.f1_05.mean()),    4),
        "f1_05_std":   round(float(df.f1_05.std()),     4),
        "f1_07_mean":  round(float(df.f1_07.mean()),    4),
        "f1_07_std":   round(float(df.f1_07.std()),     4),
        "mAP":         round(mAP, 4),
    }

    with open(RESULT_DIR / "cellpose_full_670_summary.json", "w") as f:
        json.dump(summary_full, f, indent=2)

    print("\n===== CELLPOSE FULL 670 RESULTS =====")
    for k, v in summary_full.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
