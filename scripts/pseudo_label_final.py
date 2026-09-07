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
from skimage import io as skio, color
from skimage.transform import resize
import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

RXRX1_DIR  = RXRX1_IMAGES_DIR
RESULT_DIR = RESULTS_ROOT / "pseudo_label_final"
CKPT_DIR   = CHECKPOINTS_DIR
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE       = 256
BATCH_SIZE     = 8
EPOCHS         = 50
LR             = 1e-4
PATIENCE       = 10
MAX_ROUNDS     = 3
MAX_UNLABELLED = 500
THRESHOLDS     = [0.5, 0.6, 0.7, 0.8, 0.9]

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

def iou_np(pred, target):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return float(inter / (union + 1e-8))

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

def generate_pseudo_labels(model, rxrx1_paths, confidence_threshold):
    """
    Use UNet model prediction probability as confidence score.
    confidence = mean probability of foreground pixels.
    Accept image if confidence >= threshold.
    """
    tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()])
    accepted_paths, pseudo_masks, confidences = [], [], []
    model.eval()
    with torch.no_grad():
        for path in tqdm(rxrx1_paths, desc=f"Pseudo-labelling (thresh={confidence_threshold})"):
            img = load_image(path)
            x   = tf(image=img[..., np.newaxis])["image"].unsqueeze(0).float().to(DEVICE)
            prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
            # Confidence = mean probability of predicted foreground
            fg_mask = prob > 0.5
            if fg_mask.sum() > 0:
                confidence = float(prob[fg_mask].mean())
            else:
                confidence = 0.0
            confidences.append(confidence)
            if confidence >= confidence_threshold:
                binary = fg_mask.astype(np.float32)
                pseudo_masks.append(binary)
                accepted_paths.append(str(path))

    accept_rate = len(accepted_paths) / len(rxrx1_paths) * 100
    print(f"  Accepted {len(accepted_paths)}/{len(rxrx1_paths)} ({accept_rate:.1f}%)")
    print(f"  Confidence: mean={np.mean(confidences):.3f} std={np.std(confidences):.3f}")
    return accepted_paths, pseudo_masks, accept_rate, float(np.mean(confidences))

def compute_label_quality(model, monitor_ids):
    """Agreement rate between model predictions and GT on held-out annotated subset."""
    tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()])
    model.eval()
    ious = []
    with torch.no_grad():
        for img_id in monitor_ids:
            img_path = list((DSB_DIR / img_id / "images").glob("*.png"))[0]
            img  = load_image(img_path)
            mask = load_gt_mask(DSB_DIR / img_id / "masks")
            mask_r = resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True)
            x    = tf(image=img[..., np.newaxis])["image"].unsqueeze(0).float().to(DEVICE)
            pred = (torch.sigmoid(model(x)).squeeze().cpu().numpy() > 0.5).astype(np.float32)
            ious.append(iou_np(pred, mask_r))
    return float(np.mean(ious))

def train_round(labelled_ids, val_ids, pseudo_paths, pseudo_masks, config_name):
    train_tf, val_tf = get_transforms()
    lab_ds   = LabelledDataset(labelled_ids, train_tf)
    val_ds   = LabelledDataset(val_ids, val_tf)

    if pseudo_paths:
        pseudo_ds = PseudoDataset(pseudo_paths, pseudo_masks, train_tf)
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
            out  = model(imgs)
            loss = bce(out, masks) + dice(out, masks)
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

    model.load_state_dict(torch.load(str(CKPT_DIR / f"{config_name}.pth"), map_location=DEVICE))
    return model, best_iou

def run_threshold_experiment(threshold, labelled_ids, val_ids, monitor_ids, rxrx1_paths):
    """Run 3-round iterative pseudo-labelling for one threshold."""
    print(f"\n{'='*60}")
    print(f"THRESHOLD: {threshold}")
    print(f"{'='*60}")

    rounds_results = []
    pseudo_paths, pseudo_masks = [], []

    # Get initial model (round 0 = no pseudo labels)
    config_name = f"pseudo_final_thresh{str(threshold).replace('.','')}_round0"
    model, val_iou = train_round(labelled_ids, val_ids, [], [], config_name)
    agreement = compute_label_quality(model, monitor_ids)
    print(f"  Round 0 (baseline): val_iou={val_iou:.4f} agreement={agreement:.4f}")
    rounds_results.append({
        "round": 0, "val_iou": val_iou,
        "agreement_rate": agreement, "n_pseudo": 0,
        "accept_rate": 0.0
    })

    for round_num in range(1, MAX_ROUNDS + 1):
        # Generate pseudo-labels using current model
        pseudo_paths, pseudo_masks, accept_rate, mean_conf = generate_pseudo_labels(
            model, rxrx1_paths, threshold)

        if len(pseudo_paths) < 10:
            print(f"  Round {round_num}: too few pseudo-labels ({len(pseudo_paths)}), stopping")
            break

        config_name = f"pseudo_final_thresh{str(threshold).replace('.','')}_round{round_num}"
        model, val_iou = train_round(
            labelled_ids, val_ids, pseudo_paths, pseudo_masks, config_name)
        agreement = compute_label_quality(model, monitor_ids)

        print(f"  Round {round_num}: val_iou={val_iou:.4f} agreement={agreement:.4f} "
              f"n_pseudo={len(pseudo_paths)} accept={accept_rate:.1f}%")

        rounds_results.append({
            "round": round_num,
            "val_iou": val_iou,
            "agreement_rate": agreement,
            "n_pseudo": len(pseudo_paths),
            "accept_rate": accept_rate,
            "mean_confidence": mean_conf,
        })

    return rounds_results

def main():
    rxrx1_paths = sorted(RXRX1_DIR.rglob("*_w1.png"))[:MAX_UNLABELLED]
    print(f"Found {len(rxrx1_paths)} RxRx1 images")

    all_ids = sorted([d.name for d in DSB_DIR.iterdir() if d.is_dir()])
    random.shuffle(all_ids)
    n = len(all_ids)
    labelled_ids = all_ids[:int(n*0.7)]
    val_ids      = all_ids[int(n*0.7):int(n*0.85)]
    monitor_ids  = all_ids[int(n*0.85):int(n*0.85)+50]
    print(f"Labelled: {len(labelled_ids)}, Val: {len(val_ids)}, Monitor: {len(monitor_ids)}")

    all_results = {}
    for thresh in THRESHOLDS:
        rounds = run_threshold_experiment(
            thresh, labelled_ids, val_ids, monitor_ids, rxrx1_paths)
        all_results[str(thresh)] = rounds

    with open(RESULT_DIR / "pseudo_final_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974"]
    for (thresh, rounds), color in zip(all_results.items(), colors):
        rounds_list = [r["round"] for r in rounds]
        val_ious    = [r["val_iou"] for r in rounds]
        agreements  = [r["agreement_rate"] for r in rounds]
        axes[0].plot(rounds_list, val_ious, marker='o', label=f"thresh={thresh}",
                     color=color, linewidth=2)
        axes[1].plot(rounds_list, agreements, marker='s', label=f"thresh={thresh}",
                     color=color, linewidth=2)

    axes[0].set_xlabel("Retraining Round"); axes[0].set_ylabel("Validation IoU")
    axes[0].set_title("Pseudo-label Performance across Rounds"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Retraining Round"); axes[1].set_ylabel("Agreement Rate (IoU)")
    axes[1].set_title("In-domain Agreement Monitoring"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / "visualisations/pseudo_label_final.png", dpi=150)

    print("\n===== FINAL PSEUDO-LABEL RESULTS =====")
    for thresh, rounds in all_results.items():
        best = max(rounds, key=lambda x: x["val_iou"])
        print(f"  thresh={thresh}: best_val_iou={best['val_iou']:.4f} at round={best['round']}")

if __name__ == "__main__":
    main()
