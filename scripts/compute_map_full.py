import json, glob, random
import numpy as np
import pandas as pd
from pathlib import Path
from project_paths import CHECKPOINTS_DIR, DSB_DIR, RESULTS_ROOT
import torch
import segmentation_models_pytorch as smp
from skimage import io as skio, color
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 42
random.seed(SEED); np.random.seed(SEED)

RESULT_DIR = RESULTS_ROOT
CKPT_DIR   = CHECKPOINTS_DIR
VIS_DIR    = RESULT_DIR / "visualisations"

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

def load_image(path):
    img = skio.imread(str(path))
    if img.ndim == 3:
        img = img[..., :3]
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

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

def load_model(ckpt_name):
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                     in_channels=1, classes=1).to(DEVICE)
    ckpt = CKPT_DIR / f"{ckpt_name}.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
    model.eval()
    return model

def predict(model, img_array):
    tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()])
    x  = tf(image=img_array[..., np.newaxis])["image"].unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model(x)).squeeze().cpu().numpy()
    return (pred > 0.5).astype(np.float32)

def compute_map(model, test_ids):
    aps = []
    for iou_thresh in IOU_THRESHOLDS:
        f1s = []
        for img_id in test_ids:
            img_path = list((DSB_DIR / img_id / "images").glob("*.png"))[0]
            img  = load_image(img_path)
            mask = load_gt_mask(DSB_DIR / img_id / "masks")
            mask_r = resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
            pred = predict(model, img)
            inter = (pred * mask_r).sum()
            union = pred.sum() + mask_r.sum() - inter
            iou   = inter / (union + 1e-8)
            tp = 1.0 if iou >= iou_thresh else 0.0
            fp = 1.0 - tp; fn = 1.0 - tp
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1s.append(2*prec*rec/(prec+rec+1e-8))
        aps.append(float(np.mean(f1s)))
    return float(np.mean(aps)), aps

def main():
    all_ids = sorted([d.name for d in DSB_DIR.iterdir() if d.is_dir()])
    random.seed(SEED); random.shuffle(all_ids)
    n = len(all_ids)
    test_ids = all_ids[int(n*0.85):]
    print(f"Test set: {len(test_ids)} images")

    # All 12 TL configs + 3 aug configs + combined
    configs = {}
    for budget in [40, 100, 250, 500]:
        for cfg in ["full_finetune", "frozen_backbone", "partial_finetune"]:
            name = f"tl_{cfg}_budget{budget}"
            configs[name] = name
    for aug in ["no_augmentation", "geometric_only", "geometric_and_intensity"]:
        configs[f"aug_{aug}"] = f"aug_{aug}"
    configs["combined_geometric_only"] = "combined_geometric_only"
    configs["combined_geometric_and_intensity"] = "combined_geometric_and_intensity"

    map_results = {}
    for name, ckpt in configs.items():
        if not (CKPT_DIR / f"{ckpt}.pth").exists():
            print(f"Skipping {name} (no checkpoint)")
            continue
        model = load_model(ckpt)
        mAP, aps = compute_map(model, test_ids)
        map_results[name] = {"mAP": round(mAP, 4), "AP_per_threshold": [round(a,4) for a in aps]}
        print(f"{name}: mAP={mAP:.4f}")

    with open(RESULT_DIR / "map_results_full.json", "w") as f:
        json.dump(map_results, f, indent=2)

    # Plot TL mAP learning curves
    budgets = [40, 100, 250, 500]
    tl_cfgs = [
        ("full_finetune",    "Full Finetune",    "#4C72B0"),
        ("frozen_backbone",  "Frozen Backbone",  "#55A868"),
        ("partial_finetune", "Partial Finetune", "#C44E52"),
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    for cfg, label, color in tl_cfgs:
        maps = [map_results.get(f"tl_{cfg}_budget{b}", {}).get("mAP", None) for b in budgets]
        if any(m is not None for m in maps):
            ax.plot(budgets, maps, marker='o', color=color, linewidth=2,
                    markersize=8, label=label)
    ax.set_xlabel("Annotation Budget", fontsize=13)
    ax.set_ylabel("mAP (IoU 0.5:0.95)", fontsize=13)
    ax.set_title("Transfer Learning: mAP vs Annotation Budget", fontsize=14)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_xticks(budgets)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "tl_map_curves.png", dpi=150)

    print("\nAll done.")
    print("\n===== FULL mAP RESULTS =====")
    for k, v in map_results.items():
        print(f"  {k}: {v['mAP']}")

if __name__ == "__main__":
    main()
