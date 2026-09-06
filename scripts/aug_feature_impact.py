import json
import numpy as np
import pandas as pd
from pathlib import Path
from project_paths import RESULTS_ROOT, RXRX1_IMAGES_DIR
from tqdm import tqdm
from skimage import io as skio, measure
from skimage.transform import resize
import albumentations as A
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

RXRX1_DIR  = RXRX1_IMAGES_DIR
RESULT_DIR = RESULTS_ROOT / "aug_feature_impact"
VIS_DIR    = RESULTS_ROOT / "visualisations"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

N_IMAGES   = 100
N_AUGMENTS = 10
CHANNEL_NAMES = ["Hoechst","ConA","Phalloidin","Syto14","MitoTracker","WGA"]

def load_multichannel(exp_dir, well, site):
    channels = []
    for ch in range(1, 7):
        fpath = exp_dir / f"{well}_{site}_w{ch}.png"
        if fpath.exists():
            img = skio.imread(str(fpath)).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            channels.append(img)
    return np.stack(channels, axis=-1) if len(channels) == 6 else None

def extract_morphological_features(img_6ch):
    """Extract Cell Painting-style morphological features from each channel."""
    features = {}
    for i, ch_name in enumerate(CHANNEL_NAMES):
        ch = img_6ch[..., i]
        # Intensity features
        features[f"{ch_name}_mean"]     = float(ch.mean())
        features[f"{ch_name}_std"]      = float(ch.std())
        features[f"{ch_name}_max"]      = float(ch.max())
        # Texture: coefficient of variation
        features[f"{ch_name}_cv"]       = float(ch.std() / (ch.mean() + 1e-8))
        # Channel ratio relative to Hoechst (nuclear stain)
        hoechst_mean = img_6ch[..., 0].mean()
        features[f"{ch_name}_ratio_to_hoechst"] = float(ch.mean() / (hoechst_mean + 1e-8))
    return features

def apply_augmentation(img_6ch, aug_type):
    if aug_type == "none":
        return img_6ch.copy()
    elif aug_type == "flip":
        return img_6ch[::-1, :, :].copy()
    elif aug_type == "rotation":
        return np.rot90(img_6ch, k=1).copy()
    elif aug_type == "uniform_intensity":
        scale = np.random.uniform(0.8, 1.2)
        return np.clip(img_6ch * scale, 0, 1)
    elif aug_type == "perchannel_intensity":
        scales = np.random.uniform(0.8, 1.2, size=6)
        return np.clip(img_6ch * scales[np.newaxis, np.newaxis, :], 0, 1)
    elif aug_type == "elastic":
        aug = A.ElasticTransform(alpha=120, sigma=6, p=1.0)
        channels = []
        for i in range(6):
            result = aug(image=(img_6ch[..., i]*255).astype(np.uint8))
            channels.append(result["image"].astype(np.float32) / 255.0)
        return np.stack(channels, axis=-1)

def feature_distance(feat1, feat2):
    """Compute normalized distance between feature vectors."""
    v1 = np.array(list(feat1.values()))
    v2 = np.array(list(feat2.values()))
    return float(np.linalg.norm(v1 - v2) / (np.linalg.norm(v1) + 1e-8))

def main():
    all_w1 = sorted(RXRX1_DIR.rglob("*_w1.png"))[:N_IMAGES]
    print(f"Analysing {len(all_w1)} images...")

    aug_types = ["none", "flip", "rotation", "elastic",
                 "uniform_intensity", "perchannel_intensity"]

    records = []
    for w1_path in tqdm(all_w1):
        parts = w1_path.stem.split("_")
        well, site = parts[0], parts[1]
        img = load_multichannel(w1_path.parent, well, site)
        if img is None: continue

        orig_feat = extract_morphological_features(img)

        for aug_type in aug_types:
            if aug_type == "none":
                dist = 0.0
                ratio_distortion = 0.0
            else:
                dists = []
                ratio_dists = []
                for _ in range(N_AUGMENTS):
                    aug_img = apply_augmentation(img, aug_type)
                    aug_feat = extract_morphological_features(aug_img)
                    dists.append(feature_distance(orig_feat, aug_feat))
                    # Channel ratio distortion
                    orig_ratios = np.array([img[..., i].mean() for i in range(6)])
                    orig_ratios = orig_ratios / (orig_ratios.sum() + 1e-8)
                    aug_ratios  = np.array([aug_img[..., i].mean() for i in range(6)])
                    aug_ratios  = aug_ratios / (aug_ratios.sum() + 1e-8)
                    ratio_dists.append(float(np.abs(orig_ratios - aug_ratios).mean()))
                dist = float(np.mean(dists))
                ratio_distortion = float(np.mean(ratio_dists))

            records.append({
                "aug_type": aug_type,
                "feature_distance": dist,
                "ratio_distortion": ratio_distortion,
            })

    df = pd.DataFrame(records)
    df.to_csv(RESULT_DIR / "aug_feature_impact.csv", index=False)

    summary = df.groupby("aug_type").agg({
        "feature_distance": ["mean","std"],
        "ratio_distortion": ["mean","std"],
    }).round(4)
    print("\n===== AUGMENTATION FEATURE IMPACT =====")
    print(summary)

    # Plot
    aug_order = ["none","flip","rotation","elastic",
                 "uniform_intensity","perchannel_intensity"]
    aug_labels = ["None","Flip","Rotation","Elastic",
                  "Uniform\nIntensity","Per-channel\nIntensity"]
    colors = ["#888888","#4C72B0","#55A868","#C44E52","#8172B2","#CCB974"]

    feat_means = [df[df.aug_type==a]["feature_distance"].mean() for a in aug_order]
    feat_stds  = [df[df.aug_type==a]["feature_distance"].std()  for a in aug_order]
    ratio_means = [df[df.aug_type==a]["ratio_distortion"].mean() for a in aug_order]
    ratio_stds  = [df[df.aug_type==a]["ratio_distortion"].std()  for a in aug_order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(aug_labels, feat_means, yerr=feat_stds, capsize=5,
                color=colors, edgecolor='black')
    axes[0].set_ylabel("Normalised Feature Distance", fontsize=12)
    axes[0].set_title("Augmentation Impact on\nMorphological Feature Space", fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, (v, s) in enumerate(zip(feat_means, feat_stds)):
        axes[0].text(i, v+s+0.001, f'{v:.4f}', ha='center', fontsize=9)

    axes[1].bar(aug_labels, ratio_means, yerr=ratio_stds, capsize=5,
                color=colors, edgecolor='black')
    axes[1].set_ylabel("Mean Channel Ratio Distortion", fontsize=12)
    axes[1].set_title("Augmentation Impact on\nChannel Intensity Ratios", fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, (v, s) in enumerate(zip(ratio_means, ratio_stds)):
        axes[1].text(i, v+s+0.0001, f'{v:.4f}', ha='center', fontsize=9)

    plt.suptitle("Impact of Augmentation Operations on Cell Painting Feature Space",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "aug_feature_impact.png", dpi=150, bbox_inches='tight')
    print("\nSaved aug_feature_impact.png")

if __name__ == "__main__":
    main()
