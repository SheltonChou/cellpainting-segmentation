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

def compute_metrics(pred, target, iou_thresh=0.5):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    iou   = inter / (union + 1e-8)
    tp = 1.0 if iou >= iou_thresh else 0.0
    fp = 1.0 - tp; fn = 1.0 - tp
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2*prec*rec / (prec+rec+1e-8)
    return float(iou), float(f1)

def evaluate_model(model, test_ids):
    ious, f1s_05, f1s_07 = [], [], []
    for img_id in test_ids:
        img_path = list((DSB_DIR / img_id / "images").glob("*.png"))[0]
        img  = load_image(img_path)
        mask = load_gt_mask(DSB_DIR / img_id / "masks")
        mask_r = resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
        pred = predict(model, img)
        iou, f1_05 = compute_metrics(pred, mask_r, 0.5)
        _,   f1_07 = compute_metrics(pred, mask_r, 0.7)
        ious.append(iou); f1s_05.append(f1_05); f1s_07.append(f1_07)
    return {
        "mean_iou": round(float(np.mean(ious)),  4),
        "std_iou":  round(float(np.std(ious)),   4),
        "f1_05":    round(float(np.mean(f1s_05)),4),
        "f1_07":    round(float(np.mean(f1s_07)),4),
    }

def main():
    all_ids = sorted([d.name for d in DSB_DIR.iterdir() if d.is_dir()])
    random.seed(SEED); random.shuffle(all_ids)
    n = len(all_ids)
    test_ids = all_ids[int(n*0.85):]
    print(f"Test set: {len(test_ids)} images")

    configs = []
    for aug in ["no_augmentation", "geometric_only", "geometric_and_intensity"]:
        configs.append(f"aug_{aug}")
    for budget in [40, 100, 250, 500]:
        for cfg in ["full_finetune", "frozen_backbone", "partial_finetune"]:
            configs.append(f"tl_{cfg}_budget{budget}")
    configs.append("combined_geometric_only")
    configs.append("combined_geometric_and_intensity")

    all_results = {}
    for name in configs:
        if not (CKPT_DIR / f"{name}.pth").exists():
            print(f"Skipping {name}")
            continue
        model = load_model(name)
        metrics = evaluate_model(model, test_ids)
        all_results[name] = metrics
        print(f"{name}: IoU={metrics['mean_iou']:.4f} F1@0.5={metrics['f1_05']:.4f} F1@0.7={metrics['f1_07']:.4f}")

    with open(RESULT_DIR / "full_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # F1 comparison plot
    aug_names  = [k for k in all_results if k.startswith("aug_")]
    aug_labels = [k.replace("aug_","") for k in aug_names]
    aug_f1_05  = [all_results[k]["f1_05"] for k in aug_names]
    aug_f1_07  = [all_results[k]["f1_07"] for k in aug_names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(aug_labels)); w = 0.35
    axes[0].bar(x-w/2, aug_f1_05, w, label="F1@IoU=0.5", color="#4C72B0", edgecolor='black')
    axes[0].bar(x+w/2, aug_f1_07, w, label="F1@IoU=0.7", color="#55A868", edgecolor='black')
    axes[0].set_xticks(x); axes[0].set_xticklabels(aug_labels, rotation=15)
    axes[0].set_ylabel("F1 Score"); axes[0].set_title("Augmentation F1 Comparison")
    axes[0].legend(); axes[0].set_ylim(0.7, 1.0)
    tl_names  = [f"tl_{c}_budget500" for c in ["full_finetune","frozen_backbone","partial_finetune"]]
    tl_labels = ["Full Finetune","Frozen Backbone","Partial Finetune"]
    tl_f1_05  = [all_results.get(k,{}).get("f1_05", 0) for k in tl_names]
    tl_f1_07  = [all_results.get(k,{}).get("f1_07", 0) for k in tl_names]
    x2 = np.arange(len(tl_labels))
    axes[1].bar(x2-w/2, tl_f1_05, w, label="F1@IoU=0.5", color="#4C72B0", edgecolor='black')
    axes[1].bar(x2+w/2, tl_f1_07, w, label="F1@IoU=0.7", color="#55A868", edgecolor='black')
    axes[1].set_xticks(x2); axes[1].set_xticklabels(tl_labels, rotation=15)
    axes[1].set_ylabel("F1 Score"); axes[1].set_title("Transfer Learning F1 (Budget=500)")
    axes[1].legend(); axes[1].set_ylim(0.7, 1.0)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "f1_comparison.png", dpi=150)

    # Pseudo-label curve
    pseudo_file = RESULT_DIR / "pseudo_label" / "pseudo_label_results.json"
    if pseudo_file.exists():
        with open(pseudo_file) as f:
            pseudo_data = json.load(f)
        thresholds = [int(k) for k in pseudo_data.keys()]
        val_ious   = [pseudo_data[k]["val_iou"] for k in pseudo_data.keys()]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, val_ious, marker='o', linewidth=2, markersize=8, color="#4C72B0")
        ax.axhline(y=0.8745, color='red', linestyle='--', label='Geo Aug baseline (0.8745)')
        ax.set_xlabel("Min cells threshold", fontsize=13)
        ax.set_ylabel("Validation IoU", fontsize=13)
        ax.set_title("Pseudo-label: Threshold vs Performance", fontsize=14)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(VIS_DIR / "pseudo_label_curve.png", dpi=150)

    # Combined comparison
    combined_data = {
        "Geo Aug\n(no pseudo)":  all_results.get("aug_geometric_only", {}).get("mean_iou", 0),
        "Geo Aug\n+ Pseudo":     all_results.get("combined_geometric_only", {}).get("mean_iou", 0),
        "Geo+Int Aug\n+ Pseudo": all_results.get("combined_geometric_and_intensity", {}).get("mean_iou", 0),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(combined_data.keys(), combined_data.values(),
                  color=["#4C72B0","#55A868","#C44E52"], edgecolor='black', width=0.5)
    for bar, v in zip(bars, combined_data.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f'{v:.4f}', ha='center', fontsize=11)
    ax.set_ylabel("Test IoU", fontsize=13)
    ax.set_title("Combined Experiment: Pseudo-label + Augmentation", fontsize=14)
    ax.set_ylim(0.78, 0.90)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "combined_comparison.png", dpi=150)
    print("All done.")

if __name__ == "__main__":
    main()
