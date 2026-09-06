import json
import numpy as np
import pandas as pd
from pathlib import Path
from project_paths import CHECKPOINTS_DIR, DSB_DIR, RESULTS_ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = RESULTS_ROOT
VIS_DIR    = RESULT_DIR / "visualisations"

def plot_ap_curves():
    """AP vs IoU threshold curves for all methods."""
    map_full = json.load(open(RESULT_DIR / "map_results_full.json"))
    baseline = json.load(open(RESULT_DIR / "baseline/summary.json"))
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Aug ablation curves
    aug_configs = {
        "Cellpose (baseline)": "cellpose_baseline",
        "U-Net No Aug":        "aug_no_augmentation",
        "U-Net Geo Aug":       "aug_geometric_only",
        "U-Net Geo+Int Aug":   "aug_geometric_and_intensity",
    }
    colors = ["#888888","#4C72B0","#55A868","#C44E52"]
    for (label, key), color in zip(aug_configs.items(), colors):
        if key in map_full and "AP_per_threshold" in map_full[key]:
            aps = map_full[key]["AP_per_threshold"]
            axes[0].plot(iou_thresholds, aps, marker='o', label=label,
                        color=color, linewidth=2, markersize=6)
    axes[0].set_xlabel("IoU Threshold", fontsize=12)
    axes[0].set_ylabel("Average Precision", fontsize=12)
    axes[0].set_title("AP vs IoU Threshold\nAugmentation Ablation", fontsize=13)
    axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(iou_thresholds)
    axes[0].set_xticklabels([f"{t:.2f}" for t in iou_thresholds], rotation=45)

    # TL budget 500 curves
    tl_configs = {
        "Full Finetune 500":    "tl_full_finetune_budget500",
        "Frozen Backbone 500":  "tl_frozen_backbone_budget500",
        "Partial Finetune 500": "tl_partial_finetune_budget500",
    }
    colors_tl = ["#4C72B0","#55A868","#C44E52"]
    for (label, key), color in zip(tl_configs.items(), colors_tl):
        if key in map_full and "AP_per_threshold" in map_full[key]:
            aps = map_full[key]["AP_per_threshold"]
            axes[1].plot(iou_thresholds, aps, marker='s', label=label,
                        color=color, linewidth=2, markersize=6)
    axes[1].set_xlabel("IoU Threshold", fontsize=12)
    axes[1].set_ylabel("Average Precision", fontsize=12)
    axes[1].set_title("AP vs IoU Threshold\nTransfer Learning (Budget=500)", fontsize=13)
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(iou_thresholds)
    axes[1].set_xticklabels([f"{t:.2f}" for t in iou_thresholds], rotation=45)

    plt.tight_layout()
    plt.savefig(VIS_DIR / "ap_threshold_curves.png", dpi=150)
    print("Saved ap_threshold_curves.png")

def plot_failure_cases():
    """Find and plot worst-performing test images."""
    import glob, random
    import torch
    import segmentation_models_pytorch as smp
    from skimage import io as skio, color
    from skimage.transform import resize
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    CKPT_DIR = CHECKPOINTS_DIR
    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 256

    def load_img(path):
        img = skio.imread(str(path))
        if img.ndim == 3: img = img[..., :3]; img = color.rgb2gray(img)
        img = img.astype(np.float32)
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    def load_mask(mask_dir):
        mask_paths = sorted(glob.glob(str(mask_dir / "*.png")))
        if not mask_paths: return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        ref = skio.imread(mask_paths[0])
        combined = np.zeros(ref.shape[:2], dtype=np.float32)
        for mp in mask_paths:
            m = skio.imread(mp)
            if m.ndim == 3: m = m[..., 0]
            combined[m > 0] = 1.0
        return combined

    all_ids = sorted([d.name for d in DSB_DIR.iterdir() if d.is_dir()])
    random.seed(42); random.shuffle(all_ids)
    n = len(all_ids)
    test_ids = all_ids[int(n*0.85):]

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                     in_channels=1, classes=1).to(DEVICE)
    model.load_state_dict(torch.load(str(CKPT_DIR/"aug_geometric_only.pth"),
                                     map_location=DEVICE))
    model.eval()

    tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()])
    scores = []
    for img_id in test_ids:
        img_path = list((DSB_DIR / img_id / "images").glob("*.png"))[0]
        img  = load_img(img_path)
        mask = load_mask(DSB_DIR / img_id / "masks")
        mask_r = resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
        x = tf(image=img[..., np.newaxis])["image"].unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            pred = (torch.sigmoid(model(x)).squeeze().cpu().numpy() > 0.5).astype(np.float32)
        inter = (pred * mask_r).sum()
        union = pred.sum() + mask_r.sum() - inter
        iou = float(inter / (union + 1e-8))
        scores.append((iou, img_id))

    scores.sort()
    worst  = scores[:4]   # 4 worst
    best   = scores[-4:]  # 4 best

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    for row, (iou, img_id) in enumerate(worst):
        img_path = list((DSB_DIR / img_id / "images").glob("*.png"))[0]
        img  = load_img(img_path)
        mask = load_mask(DSB_DIR / img_id / "masks")
        mask_r = resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
        img_r  = resize(img,  (IMG_SIZE, IMG_SIZE))
        x = tf(image=img[..., np.newaxis])["image"].unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            pred = (torch.sigmoid(model(x)).squeeze().cpu().numpy() > 0.5).astype(np.float32)

        axes[row,0].imshow(img_r, cmap='gray')
        axes[row,0].set_ylabel(f"IoU={iou:.3f}", fontsize=9)
        axes[row,1].imshow(mask_r, cmap='gray')
        axes[row,2].imshow(pred, cmap='gray')
        # Error map
        err = np.zeros((*pred.shape, 3))
        err[..., 0] = pred * (1 - mask_r)  # FP = red
        err[..., 1] = (1-pred) * mask_r     # FN = green
        axes[row,3].imshow(err)
        if row == 0:
            for ax, t in zip(axes[row], ["Input","GT","Prediction","Error (R=FP,G=FN)",]):
                ax.set_title(t, fontsize=9)
        for ax in axes[row]: ax.axis('off')

    for row, (iou, img_id) in enumerate(best):
        img_path = list((DSB_DIR / img_id / "images").glob("*.png"))[0]
        img  = load_img(img_path)
        mask = load_mask(DSB_DIR / img_id / "masks")
        mask_r = resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
        img_r  = resize(img,  (IMG_SIZE, IMG_SIZE))
        x = tf(image=img[..., np.newaxis])["image"].unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            pred = (torch.sigmoid(model(x)).squeeze().cpu().numpy() > 0.5).astype(np.float32)

        axes[row,4].imshow(img_r, cmap='gray')
        axes[row,4].set_title("Best cases" if row==0 else "", fontsize=9)
        axes[row,5].imshow(pred, cmap='gray')
        axes[row,5].set_xlabel(f"IoU={iou:.3f}", fontsize=9)
        for ax in axes[row, 4:]: ax.axis('off')

    plt.suptitle("Failure Analysis: Worst vs Best Segmentation Cases", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "failure_analysis.png", dpi=150, bbox_inches='tight')
    print("Saved failure_analysis.png")

def plot_literature_comparison():
    """Compare with published results from Table 1 in planning report."""
    data = {
        "Method": ["CellProfiler", "U-Net*", "StarDist", "Cellpose 2.0",
                   "Our U-Net\n(No Aug)", "Our U-Net\n(Geo Aug)", "Our U-Net\n(Full FT)"],
        "F1@0.5": [0.72, 0.84, 0.83, 0.861, 0.960, 0.970, 0.970],
        "Published": [True, True, True, True, False, False, False],
        "Year": [2018, 2015, 2018, 2022, 2024, 2024, 2024],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#AAAAAA" if p else "#4C72B0" for p in df["Published"]]
    bars = ax.bar(df["Method"], df["F1@0.5"], color=colors, edgecolor='black', width=0.6)

    for bar, v, pub in zip(bars, df["F1@0.5"], df["Published"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{v:.3f}', ha='center', fontsize=10,
                fontweight='bold' if not pub else 'normal')

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#AAAAAA', label='Published methods'),
        Patch(color='#4C72B0', label='Our methods (this work)'),
    ], fontsize=11)

    ax.axhline(y=0.861, color='orange', linestyle='--', alpha=0.7,
               label='Cellpose 2.0 benchmark (0.861)')
    ax.set_ylabel("F1 Score (IoU=0.5)", fontsize=13)
    ax.set_title("Comparison with Published Methods (Table 1)", fontsize=14)
    ax.set_ylim(0.65, 1.02)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=9)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "literature_comparison.png", dpi=150)
    print("Saved literature_comparison.png")

def main():
    plot_ap_curves()
    plot_failure_cases()
    plot_literature_comparison()
    print("\nAll plots saved.")

if __name__ == "__main__":
    main()
