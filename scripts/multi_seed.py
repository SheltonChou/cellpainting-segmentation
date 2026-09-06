import os, json, random
import numpy as np
import pandas as pd
from pathlib import Path
from project_paths import CHECKPOINTS_DIR, DSB_DIR, RESULTS_ROOT
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from skimage import io as skio, color
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

RESULT_DIR = RESULTS_ROOT / "multi_seed"
CKPT_DIR   = CHECKPOINTS_DIR
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 256
BATCH_SIZE = 8
EPOCHS     = 60
LR         = 1e-4
PATIENCE   = 15
SEEDS      = [42, 123, 456, 789, 1024, 2048, 4096, 8192]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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
        img_id = self.ids[idx]
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

def get_transforms(aug_group):
    base = [A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()]
    geo  = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5), A.ElasticTransform(alpha=120, sigma=6, p=0.3)]
    intn = [A.GaussianBlur(blur_limit=(3,7), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)]
    if aug_group == "no_augmentation":       ops = base
    elif aug_group == "geometric_only":      ops = geo + base
    else:                                    ops = geo + intn + base
    return (A.Compose(ops,  additional_targets={"mask":"mask"}),
            A.Compose(base, additional_targets={"mask":"mask"}))

def iou_score(pred, target, thresh=0.5):
    pred  = (torch.sigmoid(pred) > thresh).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter / (union + 1e-8)).item()

def train_one(config_name, aug_group, budget, seed, freeze_encoder=False, freeze_late=False):
    set_seed(seed)
    all_ids = sorted([d.name for d in DSB_DIR.iterdir() if d.is_dir()])
    random.shuffle(all_ids)
    n = len(all_ids)
    train_ids = all_ids[:int(n*0.7)]
    val_ids   = all_ids[int(n*0.7):int(n*0.85)]
    if budget: train_ids = train_ids[:budget]

    train_tf, val_tf = get_transforms(aug_group)
    train_dl = DataLoader(DSBDataset(train_ids, train_tf), batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(DSBDataset(val_ids,   val_tf),   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet",
                     in_channels=1, classes=1).to(DEVICE)
    if freeze_encoder:
        for p in model.encoder.parameters(): p.requires_grad = False
    elif freeze_late:
        for name, p in model.encoder.named_parameters():
            if any(f"layer{i}" in name for i in [1,2]): p.requires_grad = False

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=LR)
    bce  = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary")

    best_iou = 0.0; patience_cnt = 0
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
        if v_iou > best_iou:
            best_iou = v_iou; patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE: break
    return best_iou

def main():
    experiments = [
        # Obj 3: augmentation ablation
        ("aug_no_augmentation",        "no_augmentation",        None,  False, False),
        ("aug_geometric_only",         "geometric_only",         None,  False, False),
        ("aug_geometric_and_intensity","geometric_and_intensity", None,  False, False),
        # Obj 4: transfer learning key configs
        ("tl_full_finetune_budget40",    "geometric_only", 40,  False, False),
        ("tl_frozen_backbone_budget40",  "geometric_only", 40,  True,  False),
        ("tl_partial_finetune_budget40", "geometric_only", 40,  False, True),
        ("tl_full_finetune_budget100",   "geometric_only", 100, False, False),
        ("tl_frozen_backbone_budget100", "geometric_only", 100, True,  False),
        ("tl_partial_finetune_budget100","geometric_only", 100, False, True),
        ("tl_full_finetune_budget250",   "geometric_only", 250, False, False),
        ("tl_frozen_backbone_budget250", "geometric_only", 250, True,  False),
        ("tl_partial_finetune_budget250","geometric_only", 250, False, True),
        ("tl_full_finetune_budget500",   "geometric_only", 500, False, False),
        ("tl_frozen_backbone_budget500", "geometric_only", 500, True,  False),
        ("tl_partial_finetune_budget500","geometric_only", 500, False, True),
    ]

    all_results = {}
    for cfg_name, aug, budget, freeze_enc, freeze_late in experiments:
        seed_ious = []
        print(f"\n{'='*50}\n{cfg_name}\n{'='*50}")
        for seed in SEEDS:
            iou = train_one(cfg_name, aug, budget, seed, freeze_enc, freeze_late)
            seed_ious.append(iou)
            print(f"  seed={seed}: IoU={iou:.4f}")
        all_results[cfg_name] = {
            "ious": seed_ious,
            "mean": float(np.mean(seed_ious)),
            "std":  float(np.std(seed_ious)),
        }

    # Wilcoxon tests: geometric_only vs no_augmentation
    aug_no  = all_results["aug_no_augmentation"]["ious"]
    aug_geo = all_results["aug_geometric_only"]["ious"]
    aug_int = all_results["aug_geometric_and_intensity"]["ious"]
    stat1, p1 = stats.wilcoxon(aug_geo, aug_no)
    stat2, p2 = stats.wilcoxon(aug_int, aug_no)

    all_results["wilcoxon_geo_vs_no"]  = {"statistic": stat1, "p_value": p1}
    all_results["wilcoxon_int_vs_no"]  = {"statistic": stat2, "p_value": p2}

    with open(RESULT_DIR / "multi_seed_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n===== SUMMARY =====")
    for k, v in all_results.items():
        if "ious" in v:
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")
        else:
            print(f"  {k}: p={v['p_value']:.4f}")

if __name__ == "__main__":
    main()
