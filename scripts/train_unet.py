import os, json, random
import numpy as np
import pandas as pd
from pathlib import Path
from project_paths import CHECKPOINTS_DIR, DSB_DIR, RESULTS_ROOT
from tqdm import tqdm
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

DATA_DIR   = DSB_DIR
RESULT_DIR = RESULTS_ROOT
CKPT_DIR   = CHECKPOINTS_DIR
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 256
BATCH_SIZE = 8
EPOCHS     = 80
LR         = 1e-4
PATIENCE   = 15

# ── Dataset ──────────────────────────────────────────
class DSB2018Dataset(Dataset):
    def __init__(self, image_ids, data_dir, transform=None):
        self.ids = image_ids
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = list((self.data_dir / img_id / "images").glob("*.png"))[0]
        mask_paths = sorted(glob.glob(str(self.data_dir / img_id / "masks" / "*.png")))

        img = skio.imread(str(img_path))
        if img.ndim == 3:
            img = img[..., :3]
            img = color.rgb2gray(img)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        mask = np.zeros(img.shape[:2], dtype=np.float32)
        for mp in mask_paths:
            m = skio.imread(mp)
            if m.ndim == 3: m = m[..., 0]
            mask[m > 0] = 1.0

        if self.transform:
            aug = self.transform(image=img[..., np.newaxis], mask=mask)
            img  = aug["image"]
            mask = aug["mask"].unsqueeze(0).float()
        else:
            img  = torch.from_numpy(img).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

def get_transforms(aug_group="no_augmentation"):
    base = [A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()]
    geo  = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5), A.ElasticTransform(alpha=120, sigma=6, p=0.3)]
    intn = [A.GaussianBlur(blur_limit=(3,7), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)]
    if aug_group == "no_augmentation":   ops = base
    elif aug_group == "geometric_only":  ops = geo + base
    else:                                ops = geo + intn + base
    return (A.Compose(ops, additional_targets={"mask":"mask"}),
            A.Compose(base, additional_targets={"mask":"mask"}))

# ── Metrics ───────────────────────────────────────────
def dice_score(pred, target, thresh=0.5):
    pred = (torch.sigmoid(pred) > thresh).float()
    inter = (pred * target).sum()
    return (2 * inter / (pred.sum() + target.sum() + 1e-8)).item()

def iou_score(pred, target, thresh=0.5):
    pred = (torch.sigmoid(pred) > thresh).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter / (union + 1e-8)).item()

# ── Training ──────────────────────────────────────────
def train_one_config(config_name, aug_group, annotation_budget=None, 
                     freeze_encoder=False, freeze_late=False):
    print(f"\n{'='*50}")
    print(f"Config: {config_name} | Aug: {aug_group} | Budget: {annotation_budget}")
    print(f"{'='*50}")

    all_ids = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    random.shuffle(all_ids)
    n = len(all_ids)
    train_ids = all_ids[:int(n*0.7)]
    val_ids   = all_ids[int(n*0.7):int(n*0.85)]

    if annotation_budget is not None:
        train_ids = train_ids[:annotation_budget]

    train_tf, val_tf = get_transforms(aug_group)
    train_ds = DSB2018Dataset(train_ids, DATA_DIR, train_tf)
    val_ds   = DSB2018Dataset(val_ids,   DATA_DIR, val_tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet",
                     in_channels=1, classes=1).to(DEVICE)

    if freeze_encoder:
        for p in model.encoder.parameters(): p.requires_grad = False
    elif freeze_late:
        # freeze only first 3 blocks
        for name, p in model.encoder.named_parameters():
            if any(f"layer{i}" in name for i in [1,2]):
                p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    bce  = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary")

    best_val_iou = 0.0
    patience_cnt = 0
    history = []

    for epoch in range(1, EPOCHS+1):
        # Train
        model.train()
        t_loss = 0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = bce(out, masks) + dice(out, masks)
            loss.backward(); optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        # Val
        model.eval()
        v_iou = v_dice = 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                out = model(imgs)
                v_iou  += iou_score(out, masks)
                v_dice += dice_score(out, masks)

        t_loss /= len(train_dl)
        v_iou  /= len(val_dl)
        v_dice /= len(val_dl)
        history.append({"epoch":epoch, "train_loss":t_loss, "val_iou":v_iou, "val_dice":v_dice})

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | loss={t_loss:.4f} | val_iou={v_iou:.4f} | val_dice={v_dice:.4f}")

        if v_iou > best_val_iou:
            best_val_iou = v_iou
            torch.save(model.state_dict(), CKPT_DIR / f"{config_name}.pth")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(RESULT_DIR / f"{config_name}_history.csv", index=False)
    print(f"Best val IoU: {best_val_iou:.4f}")
    return best_val_iou

def main():
    results = {}

    # Obj 3: Augmentation ablation (full dataset)
    for aug in ["no_augmentation", "geometric_only", "geometric_and_intensity"]:
        name = f"aug_{aug}"
        results[name] = train_one_config(name, aug)

    # Obj 4: Transfer learning configs x annotation budgets
    tl_configs = [
        ("full_finetune",    False, False),
        ("frozen_backbone",  True,  False),
        ("partial_finetune", False, True),
    ]
    for budget in [40, 100, 250, 500]:
        for cfg_name, freeze_enc, freeze_late in tl_configs:
            name = f"tl_{cfg_name}_budget{budget}"
            results[name] = train_one_config(
                name, "geometric_only", 
                annotation_budget=budget,
                freeze_encoder=freeze_enc,
                freeze_late=freeze_late
            )

    with open(RESULT_DIR / "unet_all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll done. Results:", results)

if __name__ == "__main__":
    main()
