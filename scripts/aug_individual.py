import os, json, random
import numpy as np
import pandas as pd
from pathlib import Path
from project_paths import CHECKPOINTS_DIR, DSB_DIR, RESULTS_ROOT
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from skimage import io as skio, color
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

RESULT_DIR = RESULTS_ROOT / "aug_individual"
CKPT_DIR   = CHECKPOINTS_DIR
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 256
BATCH_SIZE = 8
EPOCHS     = 60
LR         = 1e-4
PATIENCE   = 15

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

class DSBDataset(Dataset):
    def __init__(self, image_ids, transform=None):
        self.ids = image_ids
        self.transform = transform
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_path = list((DSB_DIR / img_id / "images").glob("*.png"))[0]
        img  = load_image(img_path)
        mask = load_gt_mask(DSB_DIR / img_id / "masks")
        if self.transform:
            aug  = self.transform(image=img[..., np.newaxis], mask=mask)
            img  = aug["image"]
            mask = aug["mask"].unsqueeze(0).float()
        else:
            img  = torch.from_numpy(img).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

def iou_score(pred, target, thresh=0.5):
    pred  = (torch.sigmoid(pred) > thresh).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter / (union + 1e-8)).item()

# Individual augmentation configs
AUG_CONFIGS = {
    "none": [],
    "flip_only":      [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)],
    "rotation_only":  [A.RandomRotate90(p=0.5)],
    "elastic_only":   [A.ElasticTransform(alpha=120, sigma=6, p=0.5)],
    "blur_only":      [A.GaussianBlur(blur_limit=(3,7), p=0.5)],
    "intensity_only": [A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)],
    "geometric_only": [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                       A.RandomRotate90(p=0.5), A.ElasticTransform(alpha=120, sigma=6, p=0.3)],
}

def get_transform(ops):
    base = [A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()]
    return A.Compose(ops + base, additional_targets={"mask":"mask"})

def train_one(config_name, ops):
    print(f"\n{'='*50}\n{config_name}\n{'='*50}")
    all_ids = sorted([d.name for d in DSB_DIR.iterdir() if d.is_dir()])
    random.seed(SEED); random.shuffle(all_ids)
    n = len(all_ids)
    train_ids = all_ids[:int(n*0.7)]
    val_ids   = all_ids[int(n*0.7):int(n*0.85)]

    train_tf = get_transform(ops)
    val_tf   = get_transform([])
    train_dl = DataLoader(DSBDataset(train_ids, train_tf), batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(DSBDataset(val_ids,   val_tf),   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet",
                     in_channels=1, classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    bce  = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary")

    best_iou = 0.0; patience_cnt = 0; history = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = bce(model(imgs), masks) + dice(model(imgs), masks)
            loss.backward(); optimizer.step()
        model.eval()
        v_iou = 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                v_iou += iou_score(model(imgs), masks)
        v_iou /= len(val_dl)
        history.append({"epoch": epoch, "val_iou": v_iou})
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | val_iou={v_iou:.4f}")
        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), CKPT_DIR / f"aug_indiv_{config_name}.pth")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE: break

    pd.DataFrame(history).to_csv(RESULT_DIR / f"{config_name}_history.csv", index=False)
    return best_iou

def main():
    results = {}
    for name, ops in AUG_CONFIGS.items():
        iou = train_one(name, ops)
        results[name] = round(iou, 4)
        print(f"  {name}: {iou:.4f}")

    with open(RESULT_DIR / "individual_aug_results.json", "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#888888","#4C72B0","#55A868","#C44E52","#8172B2","#CCB974","#64B5CD"]
    bars = ax.bar(results.keys(), results.values(),
                  color=colors[:len(results)], edgecolor='black')
    for bar, v in zip(bars, results.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f'{v:.4f}', ha='center', fontsize=10)
    ax.set_ylabel("Validation IoU", fontsize=13)
    ax.set_title("Individual Augmentation Operations Ablation", fontsize=14)
    ax.set_ylim(0.82, 0.90)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / "visualisations/individual_aug_ablation.png", dpi=150)
    print("\nAll done.", results)

if __name__ == "__main__":
    main()
