import os, json, random, glob
import numpy as np
import pandas as pd
from pathlib import Path
from project_paths import CHECKPOINTS_DIR, DSB_DIR, RESULTS_ROOT, RXRX1_IMAGES_DIR
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import segmentation_models_pytorch as smp
from skimage import io as skio, color
from skimage.transform import resize
import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

RXRX1_DIR  = RXRX1_IMAGES_DIR
RESULT_DIR = RESULTS_ROOT / "combined"
CKPT_DIR   = CHECKPOINTS_DIR
PSEUDO_DIR = RESULTS_ROOT / "pseudo_label"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 256
BATCH_SIZE = 8
EPOCHS     = 80
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
    if not mask_paths:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    ref = skio.imread(mask_paths[0])
    combined = np.zeros(ref.shape[:2], dtype=np.float32)
    for mp in mask_paths:
        m = skio.imread(mp)
        if m.ndim == 3: m = m[..., 0]
        combined[m > 0] = 1.0
    return combined

def iou_score(pred, target, thresh=0.5):
    pred  = (torch.sigmoid(pred) > thresh).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter / (union + 1e-8)).item()

def f1_score_np(pred, target):
    inter = (pred * target).sum()
    prec  = inter / (pred.sum() + 1e-8)
    rec   = inter / (target.sum() + 1e-8)
    return float(2 * prec * rec / (prec + rec + 1e-8))

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

def get_transforms(aug_group="geometric_only"):
    base = [A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()]
    geo  = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5), A.ElasticTransform(alpha=120, sigma=6, p=0.3)]
    intn = [A.GaussianBlur(blur_limit=(3,7), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)]
    if aug_group == "geometric_only":           ops = geo + base
    elif aug_group == "geometric_and_intensity": ops = geo + intn + base
    else:                                        ops = base
    return (A.Compose(ops,  additional_targets={"mask":"mask"}, is_check_shapes=False),
            A.Compose(base, additional_targets={"mask":"mask"}, is_check_shapes=False))

def load_best_pseudo_masks():
    """Load pseudo-label results and find best threshold."""
    results_file = PSEUDO_DIR / "pseudo_label_results.json"
    if not results_file.exists():
        print("Pseudo-label results not found, waiting...")
        return None, None, None

    with open(results_file) as f:
        results = json.load(f)

    best_thresh = None
    best_iou    = 0.0
    for k, v in results.items():
        if v.get("val_iou") and v["val_iou"] > best_iou:
            best_iou    = v["val_iou"]
            best_thresh = k

    print(f"Best pseudo-label threshold: {best_thresh} (val_iou={best_iou:.4f})")
    return best_thresh, best_iou, results

def generate_pseudo_masks_for_combined(min_cells=20):
    """Re-generate pseudo masks using best threshold."""
    from cellpose import models as cp_models
    rxrx1_paths = sorted(RXRX1_DIR.rglob("*_w1.png"))[:500]
    cp_model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), model_type='nuclei')
    accepted_paths, pseudo_masks = [], []
    print(f"Generating pseudo-labels for combined experiment (min_cells={min_cells})...")
    for path in tqdm(rxrx1_paths):
        img = skio.imread(str(path))
        if img.ndim == 3: img = img[..., 0]
        masks, _, _ = cp_model.eval(img.astype(np.uint8), diameter=15, channels=[0,0])
        n_cells = len(np.unique(masks)) - 1
        if n_cells >= min_cells:
            binary = (masks > 0).astype(np.float32)
            binary = resize(binary, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
            pseudo_masks.append(binary)
            accepted_paths.append(str(path))
    print(f"Accepted {len(accepted_paths)}/500 images")
    return accepted_paths, pseudo_masks

def train_combined(labelled_ids, val_ids, test_ids,
                   pseudo_paths, pseudo_masks, aug_group, config_name):
    train_tf, val_tf = get_transforms(aug_group)
    lab_ds    = LabelledDataset(labelled_ids, train_tf)
    pseudo_ds = PseudoDataset(pseudo_paths, pseudo_masks, train_tf)
    val_ds    = LabelledDataset(val_ids,   val_tf)
    test_ds   = LabelledDataset(test_ids,  val_tf)

    train_dl = DataLoader(ConcatDataset([lab_ds, pseudo_ds]),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                     in_channels=1, classes=1).to(DEVICE)
    # Start from best augmentation checkpoint
    ckpt = CKPT_DIR / "aug_geometric_only.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    bce  = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary")

    best_iou = 0.0; patience_cnt = 0; history = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        t_loss = 0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = bce(out, masks) + dice(out, masks)
            loss.backward(); optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        model.eval()
        v_iou = 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                v_iou += iou_score(model(imgs), masks)
        v_iou  /= len(val_dl)
        t_loss /= len(train_dl)
        history.append({"epoch": epoch, "train_loss": t_loss, "val_iou": v_iou})

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | loss={t_loss:.4f} | val_iou={v_iou:.4f}")

        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), CKPT_DIR / f"{config_name}.pth")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE: break

    # Test set evaluation
    model.load_state_dict(torch.load(str(CKPT_DIR / f"{config_name}.pth"), map_location=DEVICE))
    model.eval()
    test_iou = 0
    with torch.no_grad():
        for imgs, masks in test_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            test_iou += iou_score(model(imgs), masks)
    test_iou /= len(test_dl)

    pd.DataFrame(history).to_csv(RESULT_DIR / f"{config_name}_history.csv", index=False)
    return best_iou, test_iou

def main():
    # Data splits
    all_ids = sorted([d.name for d in DSB_DIR.iterdir() if d.is_dir()])
    random.shuffle(all_ids)
    n = len(all_ids)
    train_ids = all_ids[:int(n*0.7)]
    val_ids   = all_ids[int(n*0.7):int(n*0.85)]
    test_ids  = all_ids[int(n*0.85):]

    # Load pseudo-label results to find best threshold
    best_thresh, best_pseudo_iou, pseudo_results = load_best_pseudo_masks()

    # Generate pseudo masks (use min_cells=20 as middle threshold)
    pseudo_paths, pseudo_masks = generate_pseudo_masks_for_combined(min_cells=20)

    results = {}

    # Baseline: best aug only (no pseudo)
    print("\n" + "="*50)
    print("Config: geo_aug_only (no pseudo)")
    train_tf, val_tf = get_transforms("geometric_only")
    lab_ds   = LabelledDataset(train_ids, train_tf)
    val_ds   = LabelledDataset(val_ids,   val_tf)
    test_ds  = LabelledDataset(test_ids,  val_tf)
    train_dl = DataLoader(lab_ds,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                     in_channels=1, classes=1).to(DEVICE)
    ckpt = CKPT_DIR / "aug_geometric_only.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
    model.eval()
    v_iou = t_iou = 0
    with torch.no_grad():
        for imgs, masks in val_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            v_iou += iou_score(model(imgs), masks)
        for imgs, masks in test_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            t_iou += iou_score(model(imgs), masks)
    results["geo_aug_only"] = {
        "val_iou":  v_iou / len(val_dl),
        "test_iou": t_iou / len(test_dl),
        "n_pseudo": 0,
    }
    print(f"  val_iou={results['geo_aug_only']['val_iou']:.4f} "
          f"test_iou={results['geo_aug_only']['test_iou']:.4f}")

    # Combined: best aug + pseudo
    for aug_group in ["geometric_only", "geometric_and_intensity"]:
        config_name = f"combined_{aug_group}"
        print(f"\n{'='*50}")
        print(f"Config: {config_name} + pseudo ({len(pseudo_paths)} images)")
        val_iou, test_iou = train_combined(
            train_ids, val_ids, test_ids,
            pseudo_paths, pseudo_masks,
            aug_group, config_name
        )
        results[config_name] = {
            "val_iou":  val_iou,
            "test_iou": test_iou,
            "n_pseudo": len(pseudo_paths),
            "aug_group": aug_group,
        }
        print(f"  Best val_iou={val_iou:.4f} | test_iou={test_iou:.4f}")

    with open(RESULT_DIR / "combined_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== COMBINED EXPERIMENT RESULTS =====")
    for k, v in results.items():
        print(f"  {k}: val={v['val_iou']:.4f} test={v['test_iou']:.4f}")

    # Compare against individual approaches
    print("\n===== COMPARISON =====")
    print(f"  Geo aug only:          val={results['geo_aug_only']['val_iou']:.4f}")
    if "combined_geometric_only" in results:
        print(f"  Geo aug + pseudo:      val={results['combined_geometric_only']['val_iou']:.4f}")
    if "combined_geometric_and_intensity" in results:
        print(f"  Geo+Int aug + pseudo:  val={results['combined_geometric_and_intensity']['val_iou']:.4f}")

if __name__ == "__main__":
    main()
