import json, glob, random
import numpy as np
import pandas as pd
from pathlib import Path
from project_paths import CHECKPOINTS_DIR, DSB_DIR, RESULTS_ROOT
from tqdm import tqdm
import torch
import segmentation_models_pytorch as smp
from skimage import io as skio, color, measure
from skimage.transform import resize
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

SEED = 42
random.seed(SEED); np.random.seed(SEED)

RESULT_DIR = RESULTS_ROOT / "density_analysis"
VIS_DIR    = RESULTS_ROOT / "visualisations"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

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

def count_gt_cells(mask_dir):
    return len(sorted(glob.glob(str(mask_dir / "*.png"))))

def load_model(ckpt_name):
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                     in_channels=1, classes=1).to(DEVICE)
    ckpt = CHECKPOINTS_DIR / f"{ckpt_name}.pth"
    model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
    model.eval()
    return model

def predict(model, img):
    tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()])
    x  = tf(image=img[..., np.newaxis])["image"].unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model(x)).squeeze().cpu().numpy()
    return (pred > 0.5).astype(np.float32)

def iou_np(pred, target):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return float(inter / (union + 1e-8))

def main():
    all_ids = sorted([d.name for d in DSB_DIR.iterdir() if d.is_dir()])
    random.seed(SEED); random.shuffle(all_ids)
    print(f"Total images: {len(all_ids)}")

    models = {
        "U-Net (No Aug)":  "aug_no_augmentation",
        "U-Net (Geo Aug)": "aug_geometric_only",
        "U-Net (Full FT)": "tl_full_finetune_budget500",
    }
    loaded_models = {k: load_model(v) for k, v in models.items()}

    records = []
    for img_id in tqdm(all_ids, desc="Analysing density vs performance"):
        img_path  = list((DSB_DIR / img_id / "images").glob("*.png"))[0]
        mask_dir  = DSB_DIR / img_id / "masks"

        img    = load_image(img_path)
        mask   = load_gt_mask(mask_dir)
        mask_r = resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
        n_cells = count_gt_cells(mask_dir)
        fg_frac = float(mask_r.mean())

        row = {
            "image_id": img_id,
            "n_gt_cells": n_cells,
            "fg_fraction": fg_frac,
        }

        for model_name, model in loaded_models.items():
            pred = predict(model, img)
            iou  = iou_np(pred, mask_r)
            row[f"iou_{model_name}"] = iou

        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(RESULT_DIR / "density_performance.csv", index=False)

    # Bin by cell count
    df["density_bin"] = pd.cut(df["n_gt_cells"],
                                bins=[0, 5, 15, 30, 50, 100, 500],
                                labels=["1-5","6-15","16-30","31-50","51-100","100+"])

    print("\n===== DENSITY VS PERFORMANCE =====")
    for model_name in loaded_models.keys():
        col = f"iou_{model_name}"
        corr, pval = stats.pearsonr(df["n_gt_cells"], df[col])
        print(f"\n{model_name}:")
        print(f"  Pearson r={corr:.3f}, p={pval:.4f}")
        print(df.groupby("density_bin")[col].mean().round(4))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    # 1. Scatter: n_cells vs IoU per model
    for (model_name, _), color in zip(models.items(), colors):
        col = f"iou_{model_name}"
        axes[0,0].scatter(df["n_gt_cells"], df[col],
                          alpha=0.3, s=20, color=color, label=model_name)
        # Trend line
        z = np.polyfit(df["n_gt_cells"], df[col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df["n_gt_cells"].min(), df["n_gt_cells"].max(), 100)
        axes[0,0].plot(x_line, p(x_line), color=color, linewidth=2)
    axes[0,0].set_xlabel("Number of GT Cells", fontsize=12)
    axes[0,0].set_ylabel("IoU", fontsize=12)
    axes[0,0].set_title("Cell Density vs IoU", fontsize=13)
    axes[0,0].legend(fontsize=9); axes[0,0].grid(True, alpha=0.3)

    # 2. Bar: mean IoU per density bin
    bin_means = df.groupby("density_bin")[[f"iou_{m}" for m in models.keys()]].mean()
    bin_means.columns = list(models.keys())
    bin_means.plot(kind='bar', ax=axes[0,1], color=colors, edgecolor='black')
    axes[0,1].set_xlabel("Cell Density Bin", fontsize=12)
    axes[0,1].set_ylabel("Mean IoU", fontsize=12)
    axes[0,1].set_title("Mean IoU by Cell Density", fontsize=13)
    axes[0,1].legend(fontsize=9); axes[0,1].grid(True, alpha=0.3, axis='y')
    axes[0,1].tick_params(axis='x', rotation=30)

    # 3. Foreground fraction distribution
    axes[1,0].hist(df["fg_fraction"], bins=30, edgecolor='black', color="#4C72B0")
    axes[1,0].axvline(df["fg_fraction"].mean(), color='red', linestyle='--',
                      label=f'Mean={df["fg_fraction"].mean():.3f}')
    axes[1,0].set_xlabel("Foreground Fraction", fontsize=12)
    axes[1,0].set_ylabel("Count", fontsize=12)
    axes[1,0].set_title("DSB2018: Cell Coverage Distribution", fontsize=13)
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    # 4. n_cells distribution
    axes[1,1].hist(df["n_gt_cells"], bins=30, edgecolor='black', color="#55A868")
    axes[1,1].axvline(df["n_gt_cells"].mean(), color='red', linestyle='--',
                      label=f'Mean={df["n_gt_cells"].mean():.1f}')
    axes[1,1].set_xlabel("Number of GT Cells", fontsize=12)
    axes[1,1].set_ylabel("Count", fontsize=12)
    axes[1,1].set_title("DSB2018: Cell Count Distribution", fontsize=13)
    axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    plt.suptitle("Cell Density vs Segmentation Performance Analysis",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "density_vs_performance.png", dpi=150, bbox_inches='tight')
    print("\nSaved density_vs_performance.png")

if __name__ == "__main__":
    main()
