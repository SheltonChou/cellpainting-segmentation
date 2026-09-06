import os, json, random, glob
import numpy as np
from pathlib import Path
from project_paths import CHECKPOINTS_DIR, DSB_DIR, RESULTS_ROOT, RXRX1_IMAGES_DIR
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import segmentation_models_pytorch as smp
from cellpose import models as cp_models
from skimage import io as skio, color
from skimage.transform import resize
import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

RXRX1_DIR  = RXRX1_IMAGES_DIR
RESULT_DIR = RESULTS_ROOT / "pseudo_volume"
CKPT_DIR   = CHECKPOINTS_DIR
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 256
BATCH_SIZE = 8
EPOCHS     = 50
LR         = 1e-4
PATIENCE   = 10
VOLUMES    = [0, 100, 200, 300, 400, 500]  # 0 = no pseudo-labels

def load_image(path):
    img = skio.imread(str(path))
    if img.ndim == 3:
        img = img[..., :3]
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def load_gt_mask(mask_dir):
    mask_paths = sorted(glob.glob(str(mask_dir / "*.png")))
    if not mask_paths:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    ref = skio.imread(mask_paths[0])
    combined = np.zeros(ref.shape[:2], dtype=np.float32)
    for mp in mask_paths:
        m = skio.imread(mp)
        if m.ndim == 3: m = m[..., 0]
        combined[m > 0] = 1.0
    return combined

def get_transforms():
    geo  = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5), A.ElasticTransform(alpha=120, sigma=6, p=0.3)]
    base = [A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()]
    return (A.Compose(geo + base, additional_targets={"mask":"mask"}, is_check_shapes=False),
            A.Compose(base,        additional_targets={"mask":"mask"}, is_check_shapes=False))

def iou_score(pred, target, thresh=0.5):
    pred  = (torch.sigmoid(pred) > thresh).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter / (union + 1e-8)).item()

class LabelledDataset(Dataset):
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

class PseudoDataset(Dataset):
    def __init__(self, image_paths, pseudo_masks, transform=None):
        self.paths  = image_paths
        self.masks  = pseudo_masks
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img  = load_image(self.paths[idx])
        mask = self.masks[idx].astype(np.float32)
        if self.transform:
            aug  = self.transform(image=img[..., np.newaxis], mask=mask)
            img  = aug["image"]
            mask = aug["mask"].unsqueeze(0).float()
        else:
            img  = torch.from_numpy(img).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

def generate_pseudo_labels(rxrx1_paths):
    print("Generating pseudo-labels from all 500 RxRx1 images...")
    cp_model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), model_type='nuclei')
    all_paths, all_masks = [], []
    for path in tqdm(rxrx1_paths):
        img = skio.imread(str(path))
        if img.ndim == 3: img = img[..., 0]
        masks, _, _ = cp_model.eval(img.astype(np.uint8), diameter=15, channels=[0,0])
        if len(np.unique(masks)) > 2:
            binary = resize((masks > 0).astype(np.float32),
                            (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
            all_paths.append(str(path))
            all_masks.append(binary)
    print(f"Generated {len(all_paths)} pseudo-labels")
    return all_paths, all_masks

def train_with_volume(labelled_ids, val_ids, pseudo_paths, pseudo_masks, n_pseudo, config_name):
    train_tf, val_tf = get_transforms()
    lab_ds  = LabelledDataset(labelled_ids, train_tf)
    val_ds  = LabelledDataset(val_ids, val_tf)

    if n_pseudo > 0 and len(pseudo_paths) > 0:
        n = min(n_pseudo, len(pseudo_paths))
        pseudo_ds = PseudoDataset(pseudo_paths[:n], pseudo_masks[:n], train_tf)
        train_dl  = DataLoader(ConcatDataset([lab_ds, pseudo_ds]),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    else:
        train_dl = DataLoader(lab_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                     in_channels=1, classes=1).to(DEVICE)
    ckpt = CKPT_DIR / "aug_geometric_only.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))

    optimizer = optim.Adam(model.parameters(), lr=LR)
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
            best_iou = v_iou
            torch.save(model.state_dict(), CKPT_DIR / f"{config_name}.pth")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE: break
    return best_iou

def main():
    rxrx1_paths = sorted(RXRX1_DIR.rglob("*_w1.png"))[:500]
    all_ids = sorted([d.name for d in DSB_DIR.iterdir() if d.is_dir()])
    random.shuffle(all_ids)
    n = len(all_ids)
    labelled_ids = all_ids[:int(n*0.7)]
    val_ids      = all_ids[int(n*0.7):int(n*0.85)]

    # Generate all pseudo-labels once
    pseudo_paths, pseudo_masks = generate_pseudo_labels(rxrx1_paths)

    results = {}
    for vol in VOLUMES:
        print(f"\n{'='*50}\nVolume: {vol} pseudo-labels\n{'='*50}")
        config_name = f"pseudo_vol_{vol}"
        val_iou = train_with_volume(
            labelled_ids, val_ids,
            pseudo_paths, pseudo_masks,
            vol, config_name
        )
        results[vol] = {"n_pseudo": vol, "val_iou": val_iou}
        print(f"  n_pseudo={vol}: val_iou={val_iou:.4f}")

    with open(RESULT_DIR / "volume_results.json", "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    vols    = list(results.keys())
    val_ious = [results[v]["val_iou"] for v in vols]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(vols, val_ious, marker='o', linewidth=2, markersize=8, color="#4C72B0")
    ax.axhline(y=results[0]["val_iou"], color='red', linestyle='--',
               label=f'No pseudo-label baseline ({results[0]["val_iou"]:.4f})')
    for x, y in zip(vols, val_ious):
        ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)
    ax.set_xlabel("Number of Pseudo-labelled Images", fontsize=13)
    ax.set_ylabel("Validation IoU", fontsize=13)
    ax.set_title("Pseudo-label: Volume of Unlabelled Data vs Performance", fontsize=14)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_xticks(vols)
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / "visualisations/pseudo_volume_curve.png", dpi=150)
    print("\nAll done.", results)

if __name__ == "__main__":
    main()
