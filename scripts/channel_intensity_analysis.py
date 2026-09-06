import json
import numpy as np
from pathlib import Path
from project_paths import RESULTS_ROOT, RXRX1_IMAGES_DIR
from skimage import io as skio
import albumentations as A
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RXRX1_DIR  = RXRX1_IMAGES_DIR
RESULT_DIR = RESULTS_ROOT / "channel_analysis"
VIS_DIR    = RESULTS_ROOT / "visualisations"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

N_IMAGES   = 200
N_AUGMENTS = 20
CHANNEL_NAMES = ["Hoechst(w1)","ConA(w2)","Phalloidin(w3)",
                 "Syto14(w4)","MitoTracker(w5)","WGA(w6)"]

def load_multichannel(exp_dir, well, site):
    channels = []
    for ch in range(1, 7):
        fpath = exp_dir / f"{well}_{site}_w{ch}.png"
        if fpath.exists():
            img = skio.imread(str(fpath)).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            channels.append(img)
    return np.stack(channels, axis=-1) if len(channels) == 6 else None

def channel_ratios(img_6ch):
    means = img_6ch.mean(axis=(0,1))
    return means / (means.sum() + 1e-8)

def main():
    all_w1 = sorted(RXRX1_DIR.rglob("*_w1.png"))[:N_IMAGES]
    print(f"Analysing {len(all_w1)} images...")

    orig_list, unif_list, pc_list = [], [], []

    aug_uniform = A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.2, p=1.0)

    for w1_path in all_w1:
        parts  = w1_path.stem.split("_")
        well, site = parts[0], parts[1]
        img = load_multichannel(w1_path.parent, well, site)
        if img is None: continue

        orig_list.append(channel_ratios(img))

        # Uniform aug: same perturbation applied to all channels equally
        u_ratios = []
        for _ in range(N_AUGMENTS):
            scale = np.random.uniform(0.8, 1.2)
            aug_img = np.clip(img * scale, 0, 1)
            u_ratios.append(channel_ratios(aug_img))
        unif_list.append(np.mean(u_ratios, axis=0))

        # Per-channel aug: independent perturbation per channel
        pc_ratios = []
        for _ in range(N_AUGMENTS):
            scales = np.random.uniform(0.8, 1.2, size=6)
            aug_img = np.clip(img * scales[np.newaxis, np.newaxis, :], 0, 1)
            pc_ratios.append(channel_ratios(aug_img))
        pc_list.append(np.mean(pc_ratios, axis=0))

    orig = np.array(orig_list)    # N x 6
    unif = np.array(unif_list)    # N x 6
    pc   = np.array(pc_list)      # N x 6

    unif_dist = np.abs(unif - orig)   # N x 6
    pc_dist   = np.abs(pc   - orig)   # N x 6

    print(f"\n{'Channel':<18} {'Uniform Aug':>12} {'Per-channel Aug':>16}")
    print("-" * 48)
    for i, ch in enumerate(CHANNEL_NAMES):
        print(f"{ch:<18} {unif_dist[:,i].mean():>12.4f} {pc_dist[:,i].mean():>16.4f}")

    results = {
        "original_mean_ratios":        orig.mean(axis=0).tolist(),
        "uniform_aug_distortion":      unif_dist.mean(axis=0).tolist(),
        "perchannel_aug_distortion":   pc_dist.mean(axis=0).tolist(),
        "channel_names":               CHANNEL_NAMES,
    }
    with open(RESULT_DIR / "channel_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    x = np.arange(6); w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors6 = ["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974","#64B5CD"]

    axes[0].bar(x, orig.mean(axis=0), yerr=orig.std(axis=0),
                capsize=5, color=colors6, edgecolor='black')
    axes[0].set_xticks(x); axes[0].set_xticklabels(CHANNEL_NAMES, rotation=20, ha='right')
    axes[0].set_ylabel("Mean Intensity Ratio"); axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_title("Original Channel Intensity Ratios\n(mean ± std)", fontsize=12)

    axes[1].bar(x-w/2, unif_dist.mean(axis=0), w,
                label="Uniform Aug", color="#4C72B0", edgecolor='black')
    axes[1].bar(x+w/2, pc_dist.mean(axis=0), w,
                label="Per-channel Aug", color="#C44E52", edgecolor='black')
    axes[1].set_xticks(x); axes[1].set_xticklabels(CHANNEL_NAMES, rotation=20, ha='right')
    axes[1].set_ylabel("Mean Absolute Ratio Distortion")
    axes[1].set_title("Channel Ratio Distortion\n(lower = less biological signal distortion)", fontsize=12)
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Per-channel Intensity Analysis: Biological Signal Preservation",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "channel_intensity_analysis.png", dpi=150, bbox_inches='tight')
    print("\nSaved channel_intensity_analysis.png")

if __name__ == "__main__":
    main()
