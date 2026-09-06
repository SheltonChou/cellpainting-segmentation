import json, glob, random
import numpy as np
import pandas as pd
from pathlib import Path
from project_paths import CHECKPOINTS_DIR, RESULTS_ROOT, RXRX1_IMAGES_DIR, RXRX1_METADATA
from tqdm import tqdm
import torch
import segmentation_models_pytorch as smp
from skimage import io as skio, color
from skimage.transform import resize
from cellpose import models as cp_models
from skimage import measure
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED)

RXRX1_DIR  = RXRX1_IMAGES_DIR
META_PATH  = RXRX1_METADATA
RESULT_DIR = RESULTS_ROOT / "cell_type_bias"
VIS_DIR    = RESULTS_ROOT / "visualisations"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
N_PER_CELL_TYPE = 100  # images per cell type

def load_image(path):
    img = skio.imread(str(path))
    if img.ndim == 3: img = img[..., 0]
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def load_unet():
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                     in_channels=1, classes=1).to(DEVICE)
    ckpt = CHECKPOINTS_DIR / "aug_geometric_only.pth"
    model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
    model.eval()
    return model

def predict_unet(model, img):
    tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()])
    x  = tf(image=img[..., np.newaxis])["image"].unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model(x)).squeeze().cpu().numpy()
    return (pred > 0.5).astype(np.float32)

def compute_cell_density(pred_mask):
    """Estimate cell density from predicted mask."""
    labeled = measure.label(pred_mask > 0.5)
    n_cells = labeled.max()
    fg_frac  = float((pred_mask > 0.5).mean())
    return n_cells, fg_frac

def main():
    import pandas as pd
    meta = pd.read_csv(META_PATH)
    print(f"Metadata loaded: {len(meta)} records")
    print(f"Cell types: {meta['cell_type'].unique()}")

    # Sample images per cell type
    cp_model   = cp_models.CellposeModel(gpu=torch.cuda.is_available(), model_type='nuclei')
    unet_model = load_unet()

    records = []
    for cell_type in meta['cell_type'].unique():
        ct_meta = meta[meta['cell_type'] == cell_type].sample(
            n=min(N_PER_CELL_TYPE, len(meta[meta['cell_type']==cell_type])),
            random_state=SEED)
        print(f"\nProcessing {cell_type} ({len(ct_meta)} images)...")

        for _, row in tqdm(ct_meta.iterrows(), total=len(ct_meta)):
            w1_path = (RXRX1_DIR / row['experiment'] /
                      f"Plate{row['plate']}" /
                      f"{row['well']}_s{row['site']}_w1.png")
            if not w1_path.exists(): continue

            img = load_image(w1_path)
            img_r = resize(img, (IMG_SIZE, IMG_SIZE))

            # UNet prediction
            unet_pred = predict_unet(unet_model, img)
            n_cells_unet, fg_unet = compute_cell_density(unet_pred)

            # Cellpose prediction
            from skimage import exposure
            img_cp = exposure.rescale_intensity(
                img, out_range=(0,255)).astype(np.uint8)
            cp_masks, _, _ = cp_model.eval(img_cp, diameter=15, channels=[0,0])
            n_cells_cp = len(np.unique(cp_masks)) - 1
            fg_cp = float((cp_masks > 0).mean())

            records.append({
                "cell_type":     cell_type,
                "experiment":    row['experiment'],
                "n_cells_unet":  n_cells_unet,
                "fg_frac_unet":  fg_unet,
                "n_cells_cp":    n_cells_cp,
                "fg_frac_cp":    fg_cp,
                "unet_pred_mean": float(unet_pred.mean()),
            })

    df = pd.DataFrame(records)
    df.to_csv(RESULT_DIR / "cell_type_bias_results.csv", index=False)

    # Summary stats per cell type
    summary = df.groupby("cell_type").agg({
        "n_cells_unet": ["mean","std"],
        "fg_frac_unet": ["mean","std"],
        "n_cells_cp":   ["mean","std"],
        "fg_frac_cp":   ["mean","std"],
    }).round(3)
    print("\n===== CELL TYPE BIAS ANALYSIS =====")
    print(summary)
    summary.to_csv(RESULT_DIR / "cell_type_summary.csv")

    # Plot
    cell_types = df["cell_type"].unique()
    colors = ["#4C72B0","#55A868","#C44E52","#8172B2"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cell density per cell type (UNet)
    means = [df[df.cell_type==ct]["n_cells_unet"].mean() for ct in cell_types]
    stds  = [df[df.cell_type==ct]["n_cells_unet"].std()  for ct in cell_types]
    axes[0,0].bar(cell_types, means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black')
    axes[0,0].set_title("Cells Detected per Image (U-Net)", fontsize=12)
    axes[0,0].set_ylabel("Number of Cells"); axes[0,0].grid(True, alpha=0.3, axis='y')

    # 2. Foreground fraction per cell type
    means = [df[df.cell_type==ct]["fg_frac_unet"].mean() for ct in cell_types]
    stds  = [df[df.cell_type==ct]["fg_frac_unet"].std()  for ct in cell_types]
    axes[0,1].bar(cell_types, means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black')
    axes[0,1].set_title("Foreground Fraction per Cell Type (U-Net)", fontsize=12)
    axes[0,1].set_ylabel("Foreground Fraction"); axes[0,1].grid(True, alpha=0.3, axis='y')

    # 3. Cellpose vs UNet cell count comparison
    x = np.arange(len(cell_types)); w = 0.35
    cp_means   = [df[df.cell_type==ct]["n_cells_cp"].mean()   for ct in cell_types]
    unet_means = [df[df.cell_type==ct]["n_cells_unet"].mean() for ct in cell_types]
    axes[1,0].bar(x-w/2, cp_means,   w, label="Cellpose", color="#888888", edgecolor='black')
    axes[1,0].bar(x+w/2, unet_means, w, label="U-Net",    color="#4C72B0", edgecolor='black')
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(cell_types)
    axes[1,0].set_title("Cellpose vs U-Net: Cells Detected", fontsize=12)
    axes[1,0].set_ylabel("Number of Cells"); axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3, axis='y')

    # 4. Cell density distribution boxplot
    data_by_ct = [df[df.cell_type==ct]["n_cells_unet"].values for ct in cell_types]
    bp = axes[1,1].boxplot(data_by_ct, labels=cell_types, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[1,1].set_title("Cell Density Distribution by Cell Type", fontsize=12)
    axes[1,1].set_ylabel("Cells per Image"); axes[1,1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Cell Type Bias Analysis: Segmentation Performance Across Cell Lines",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "cell_type_bias.png", dpi=150, bbox_inches='tight')
    print("\nSaved cell_type_bias.png")

    # Cell density vs performance analysis
    fig, ax = plt.subplots(figsize=(9, 6))
    for ct, color in zip(cell_types, colors):
        ct_df = df[df.cell_type == ct]
        ax.scatter(ct_df["n_cells_unet"], ct_df["fg_frac_unet"],
                   label=ct, color=color, alpha=0.6, s=50)
    ax.set_xlabel("Number of Cells Detected", fontsize=12)
    ax.set_ylabel("Foreground Fraction", fontsize=12)
    ax.set_title("Cell Density vs Coverage by Cell Type", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "cell_density_vs_coverage.png", dpi=150)
    print("Saved cell_density_vs_coverage.png")

if __name__ == "__main__":
    main()
