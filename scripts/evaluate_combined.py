import os, json, random, glob
import numpy as np
from pathlib import Path
from project_paths import CHECKPOINTS_DIR, DSB_DIR, RESULTS_ROOT
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from skimage import io as skio, color
from skimage.transform import resize
import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 42
random.seed(SEED); np.random.seed(SEED)

CKPT_DIR   = CHECKPOINTS_DIR
RESULT_DIR = RESULTS_ROOT
IMG_SIZE  = 256
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def load_image(path):
    img = skio.imread(str(path))
    if img.ndim == 3:
        img = img[..., :3]
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def load_gt_mask(mask_dir):
    mask_paths = sorted(glob.glob(str(mask_dir / "*.png")))
    if not mask_paths:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    ref = skio.imread(mask_paths[0])
    combined = np.zeros(ref.shape[:2], dtype=np.float32)
    for mp in mask_paths:
        m = skio.imread(mp)
        if m.ndim == 3: m = m[..., 0]
        combined[m > 0] = 1.0
    return combined

def iou_np(pred, target):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return float(inter / (union + 1e-8))

def compute_f1_at_threshold(pred_binary, gt_binary, iou_thresh):
    inter = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - inter
    iou = inter / (union + 1e-8)
    tp = 1.0 if iou >= iou_thresh else 0.0
    fp = 1.0 - tp; fn = 1.0 - tp
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return 2*prec*rec / (prec + rec + 1e-8)

def get_test_ids():
    all_ids = sorted([d.name for d in DSB_DIR.iterdir()
                      if d.is_dir() and (d / "images").exists()])
    random.seed(SEED); random.shuffle(all_ids)
    n = len(all_ids)
    test_ids = all_ids[int(n*0.85):]
    return test_ids

def evaluate_checkpoint(ckpt_path, test_ids):
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                     in_channels=1, classes=1).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()])

    ious = []
    f1_05_list = []
    f1_07_list = []
    ap_per_thresh = []

    with torch.no_grad():
        for img_id in test_ids:
            img_path = list((DSB_DIR / img_id / "images").glob("*.png"))[0]
            img  = load_image(img_path)
            mask = load_gt_mask(DSB_DIR / img_id / "masks")
            mask_r = resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)

            x = tf(image=img[..., np.newaxis])["image"].unsqueeze(0).float().to(device)
            prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
            pred = (prob > 0.5).astype(np.float32)

            ious.append(iou_np(pred, mask_r))
            f1_05_list.append(compute_f1_at_threshold(pred, mask_r, 0.5))
            f1_07_list.append(compute_f1_at_threshold(pred, mask_r, 0.7))

    # mAP
    ap_vals = []
    with torch.no_grad():
        for thresh in IOU_THRESHOLDS:
            f1s = []
            for img_id in test_ids:
                img_path = list((DSB_DIR / img_id / "images").glob("*.png"))[0]
                img  = load_image(img_path)
                mask = load_gt_mask(DSB_DIR / img_id / "masks")
                mask_r = resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
                x = tf(image=img[..., np.newaxis])["image"].unsqueeze(0).float().to(device)
                prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
                pred = (prob > 0.5).astype(np.float32)
                f1s.append(compute_f1_at_threshold(pred, mask_r, thresh))
            ap_vals.append(float(np.mean(f1s)))

    return {
        "mean_iou":  round(float(np.mean(ious)), 4),
        "f1_05":     round(float(np.mean(f1_05_list)), 4),
        "f1_07":     round(float(np.mean(f1_07_list)), 4),
        "mAP":       round(float(np.mean(ap_vals)), 4),
        "AP_per_threshold": [round(v, 4) for v in ap_vals],
    }

def main():
    test_ids = get_test_ids()
    print(f"Test set: {len(test_ids)} images")

    configs = {
        "geo_aug_only":                  "aug_geometric_only.pth",
        "combined_geometric_only":       "combined_geometric_only.pth",
        "combined_geometric_and_intensity": "combined_geometric_and_intensity.pth",
    }

    results = {}
    for name, ckpt_file in configs.items():
        ckpt_path = CKPT_DIR / ckpt_file
        if not ckpt_path.exists():
            print(f"MISSING: {ckpt_path}")
            continue
        print(f"\nEvaluating: {name}")
        metrics = evaluate_checkpoint(ckpt_path, test_ids)
        results[name] = metrics
        print(f"  IoU={metrics['mean_iou']}, F1@0.5={metrics['f1_05']}, "
              f"F1@0.7={metrics['f1_07']}, mAP={metrics['mAP']}")

    with open(RESULT_DIR / "combined_eval_unified.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved combined_eval_unified.json")
    print("\n===== FINAL UNIFIED RESULTS =====")
    for name, m in results.items():
        print(f"{name}: IoU={m['mean_iou']}, F1@0.5={m['f1_05']}, "
              f"F1@0.7={m['f1_07']}, mAP={m['mAP']}")

if __name__ == "__main__":
    main()
