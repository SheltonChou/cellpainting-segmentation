import json, glob, random
import numpy as np
import torch
from pathlib import Path
from project_paths import DSB_DIR, RESULTS_ROOT
from tqdm import tqdm
from skimage import io as skio, color, exposure
from skimage.transform import resize
from cellpose import models as cp_models

SEED = 42
random.seed(SEED); np.random.seed(SEED)

RESULT_DIR = RESULTS_ROOT
IMG_SIZE   = 256
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

def load_image_for_cellpose(path):
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

def compute_f1_at_threshold(pred, target, iou_thresh):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    iou   = inter / (union + 1e-8)
    tp = 1.0 if iou >= iou_thresh else 0.0
    fp = 1.0 - tp; fn = 1.0 - tp
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return 2*prec*rec/(prec+rec+1e-8)

def main():
    all_ids = sorted([d.name for d in DSB_DIR.iterdir() if d.is_dir()])
    random.seed(SEED); random.shuffle(all_ids)
    n = len(all_ids)
    test_ids = all_ids[int(n*0.85):]
    print(f"Test set: {len(test_ids)} images")

    model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), model_type='cyto')
    aps = []
    for iou_thresh in IOU_THRESHOLDS:
        f1s = []
        for img_id in tqdm(test_ids, desc=f"IoU={iou_thresh:.2f}"):
            img_path = list((DSB_DIR / img_id / "images").glob("*.png"))[0]
            img    = load_image_for_cellpose(img_path)
            mask   = load_gt_mask(DSB_DIR / img_id / "masks")
            mask_r = resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
            masks_pred, _, _ = model.eval(img, diameter=None, channels=[0,0])
            pred = resize((masks_pred > 0).astype(np.float32),
                          (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
            f1s.append(compute_f1_at_threshold(pred, mask_r, iou_thresh))
        aps.append(float(np.mean(f1s)))
        print(f"IoU={iou_thresh:.2f}: AP={aps[-1]:.4f}")

    mAP = float(np.mean(aps))
    print(f"\nCellpose mAP: {mAP:.4f}")

    summary = json.load(open(RESULT_DIR / "baseline/summary.json"))
    summary["mAP"] = mAP
    with open(RESULT_DIR / "baseline/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    map_full = json.load(open(RESULT_DIR / "map_results_full.json"))
    map_full["cellpose_baseline"] = {"mAP": mAP, "AP_per_threshold": aps}
    with open(RESULT_DIR / "map_results_full.json", "w") as f:
        json.dump(map_full, f, indent=2)

    print("Updated results. Done.")

if __name__ == "__main__":
    main()
