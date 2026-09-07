#!/usr/bin/env python3
"""
Route B evaluation for Shuaiyu Zhou MSc project.

Purpose
-------
Evaluate the existing DSB2018 test checkpoints with the final instance metric
definitions, without retraining:

1) threshold sigmoid foreground probability at 0.5;
2) recover predicted instances by 8-connected components;
3) assign each predicted component an instance confidence equal to the
   mean sigmoid probability within that component;
4) use Hungarian bipartite matching for instance precision/recall/F1;
5) compute single-class COCO-style mask AP from ranked instance confidences
   at IoU thresholds 0.50:0.05:0.95 (101-point interpolated AP);
6) compute paired bootstrap uncertainty from test images for selected comparisons.

This script uses the original individual DSB2018 masks as ground-truth
instances. Earlier metric files are retained only as records of the project.

Run from the repository root, e.g.
    python scripts/routeB_instance_eval.py --scope all

Outputs are written to:
    results/routeB_instance_eval/
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from scipy.optimize import linear_sum_assignment
from skimage import io as skio, color, measure
from skimage.transform import resize

IMG_SIZE = 256
SPLIT_SEED = 42
FG_THRESHOLD = 0.5
IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dsb_dir(project_root: Path, explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates += [
        project_root / "data/dsb2018/stage1_train",
        project_root / "data-science-bowl-2018/stage1_train",
        project_root / "dsb2018_extracted",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            # sanity check: at least one image-id directory with images/masks
            dirs = [d for d in p.iterdir() if d.is_dir()]
            if dirs:
                return p
    raise FileNotFoundError(
        "Could not find DSB2018 training directory. Tried:\n  " +
        "\n  ".join(str(x) for x in candidates)
    )


def set_split_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def make_test_ids(dsb_dir: Path):
    ids = sorted([d.name for d in dsb_dir.iterdir() if d.is_dir()])
    random.seed(SPLIT_SEED)
    random.shuffle(ids)
    n = len(ids)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return ids[val_end:], {
        "n_total": n,
        "n_train_by_code": train_end,
        "n_val_by_code": val_end - train_end,
        "n_test_by_code": n - val_end,
        "train_end": train_end,
        "val_end": val_end,
    }


def load_image(img_path: Path):
    img = skio.imread(str(img_path))
    if img.ndim == 3:
        img = img[..., :3]
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def load_gt_instances(mask_dir: Path):
    masks = []
    for mp in sorted(mask_dir.glob("*.png")):
        m = skio.imread(str(mp))
        if m.ndim == 3:
            m = m[..., 0]
        mr = resize(
            (m > 0).astype(np.uint8),
            (IMG_SIZE, IMG_SIZE),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) > 0.5
        if mr.any():
            masks.append(mr)
    return masks


def load_model(ckpt_path: Path, device):
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    ).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_probability(model, img, device):
    tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()])
    x = tf(image=img[..., np.newaxis])["image"].unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    return prob.astype(np.float32)


def components_from_probability(prob, threshold=FG_THRESHOLD):
    binary = prob > threshold
    # 8-connectivity in 2D
    labels = measure.label(binary, connectivity=2)
    pred_masks = []
    pred_scores = []
    for lab in range(1, labels.max() + 1):
        m = labels == lab
        if not m.any():
            continue
        pred_masks.append(m)
        pred_scores.append(float(prob[m].mean()))
    return pred_masks, np.asarray(pred_scores, dtype=float), binary


def binary_iou(pred_binary, gt_instances):
    if gt_instances:
        gt = np.logical_or.reduce(gt_instances)
    else:
        gt = np.zeros_like(pred_binary, dtype=bool)
    inter = np.logical_and(pred_binary, gt).sum()
    union = np.logical_or(pred_binary, gt).sum()
    return float(inter / union) if union > 0 else 1.0


def iou_matrix(pred_masks, gt_masks):
    P, G = len(pred_masks), len(gt_masks)
    mat = np.zeros((P, G), dtype=np.float32)
    if P == 0 or G == 0:
        return mat
    gt_areas = np.array([g.sum() for g in gt_masks], dtype=np.float32)
    for i, p in enumerate(pred_masks):
        p_area = float(p.sum())
        for j, g in enumerate(gt_masks):
            inter = float(np.logical_and(p, g).sum())
            union = p_area + gt_areas[j] - inter
            if union > 0:
                mat[i, j] = inter / union
    return mat


def hungarian_counts(iou_mat, threshold):
    """
    Maximise number of matches meeting threshold; break ties by total IoU.
    """
    P, G = iou_mat.shape
    if P == 0:
        return 0, 0, G
    if G == 0:
        return 0, P, 0

    valid = iou_mat >= threshold
    # Strongly reward every valid edge, then reward higher IoU.
    cost = np.where(valid, -1_000_000.0 - iou_mat, 0.0)
    rows, cols = linear_sum_assignment(cost)
    matched = 0
    for r, c in zip(rows, cols):
        if valid[r, c]:
            matched += 1
    tp = matched
    fp = P - tp
    fn = G - tp
    return tp, fp, fn


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return precision, recall, f1


def hungarian_dataset_metrics(records, indices=None):
    if indices is None:
        indices = range(len(records))
    by_t = {}
    for t in IOU_THRESHOLDS:
        tp = fp = fn = 0
        for idx in indices:
            a, b, c = hungarian_counts(records[idx]["iou_mat"], float(t))
            tp += a; fp += b; fn += c
        p, r, f1 = prf(tp, fp, fn)
        by_t[f"{t:.2f}"] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(p), "recall": float(r), "f1": float(f1),
        }

    f1s = [by_t[f"{t:.2f}"]["f1"] for t in IOU_THRESHOLDS]
    return {
        "f1_50": by_t["0.50"]["f1"],
        "f1_70": by_t["0.70"]["f1"],
        "mean_f1_50_95": float(np.mean(f1s)),
        "per_threshold": by_t,
    }


def interpolated_ap(recalls, precisions):
    """
    COCO-style 101-point interpolated AP for one class and one IoU threshold.
    """
    recall_grid = np.linspace(0.0, 1.0, 101)
    vals = []
    for r in recall_grid:
        mask = recalls >= r
        vals.append(float(np.max(precisions[mask])) if np.any(mask) else 0.0)
    return float(np.mean(vals))


def ranked_ap(records, threshold, indices=None):
    """
    Single-class mask AP across images. Predictions are ranked globally by
    mean component foreground probability. Matching is performed per image in
    descending confidence order, as in standard AP evaluation.
    """
    if indices is None:
        indices = list(range(len(records)))
    else:
        indices = list(indices)

    # Treat repeated bootstrap image occurrences as distinct pseudo-images.
    total_gt = 0
    predictions = []
    for pseudo_img_id, idx in enumerate(indices):
        rec = records[idx]
        total_gt += rec["n_gt"]
        for p_idx, score in enumerate(rec["scores"]):
            predictions.append(
                (float(score), pseudo_img_id, idx, p_idx)
            )

    if total_gt == 0:
        return 1.0
    if not predictions:
        return 0.0

    predictions.sort(key=lambda x: x[0], reverse=True)
    matched_gt = {
        pseudo_img_id: set() for pseudo_img_id in range(len(indices))
    }

    tps = np.zeros(len(predictions), dtype=np.float64)
    fps = np.zeros(len(predictions), dtype=np.float64)

    for k, (_, pseudo_img_id, idx, p_idx) in enumerate(predictions):
        mat = records[idx]["iou_mat"]
        if mat.shape[1] == 0:
            fps[k] = 1.0
            continue

        row = mat[p_idx]
        order = np.argsort(row)[::-1]
        match = None
        for g_idx in order:
            if g_idx in matched_gt[pseudo_img_id]:
                continue
            if row[g_idx] >= threshold:
                match = int(g_idx)
                break
        if match is not None:
            tps[k] = 1.0
            matched_gt[pseudo_img_id].add(match)
        else:
            fps[k] = 1.0

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recalls = cum_tp / total_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # Precision envelope
    if len(precisions):
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    return interpolated_ap(recalls, precisions)


def ap_dataset_metrics(records, indices=None):
    aps = {}
    for t in IOU_THRESHOLDS:
        aps[f"{t:.2f}"] = ranked_ap(records, float(t), indices)
    return {
        "ap_50": aps["0.50"],
        "ap_75": aps["0.75"],
        "map_50_95": float(np.mean(list(aps.values()))),
        "ap_per_threshold": aps,
    }


def summary_metrics(records, indices=None):
    if indices is None:
        indices = list(range(len(records)))
    else:
        indices = list(indices)

    binary = np.array([records[i]["binary_iou"] for i in indices], dtype=float)
    h = hungarian_dataset_metrics(records, indices)
    a = ap_dataset_metrics(records, indices)
    return {
        "n_images": len(indices),
        "binary_iou_mean": float(binary.mean()),
        "binary_iou_sd": float(binary.std(ddof=1)) if len(binary) > 1 else 0.0,
        "instance_f1_50": h["f1_50"],
        "instance_f1_70": h["f1_70"],
        "mean_instance_f1_50_95": h["mean_f1_50_95"],
        "ap_50": a["ap_50"],
        "ap_75": a["ap_75"],
        "map_50_95": a["map_50_95"],
        "hungarian_per_threshold": h["per_threshold"],
        "ap_per_threshold": a["ap_per_threshold"],
    }


def percentile_ci(values, alpha=0.05):
    return [
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    ]


def bootstrap_model(records, B, rng):
    n = len(records)
    keys = [
        "binary_iou_mean",
        "instance_f1_50",
        "instance_f1_70",
        "mean_instance_f1_50_95",
        "ap_50",
        "ap_75",
        "map_50_95",
    ]
    store = {k: [] for k in keys}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        s = summary_metrics(records, idx)
        for k in keys:
            store[k].append(s[k])
    return {k: percentile_ci(v) for k, v in store.items()}


def bootstrap_pair(records_a, records_b, B, rng):
    assert len(records_a) == len(records_b)
    n = len(records_a)
    keys = [
        "binary_iou_mean",
        "instance_f1_50",
        "instance_f1_70",
        "mean_instance_f1_50_95",
        "ap_50",
        "map_50_95",
    ]
    diffs = {k: [] for k in keys}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        sa = summary_metrics(records_a, idx)
        sb = summary_metrics(records_b, idx)
        for k in keys:
            diffs[k].append(sa[k] - sb[k])
    return {
        k: {
            "difference_mean_bootstrap": float(np.mean(v)),
            "ci95": percentile_ci(v),
        }
        for k, v in diffs.items()
    }


def evaluate_checkpoint(name, ckpt_path, dsb_dir, test_ids, device):
    print(f"\nEvaluating {name}")
    print(f"  checkpoint: {ckpt_path}")
    model = load_model(ckpt_path, device)
    records = []

    for i, img_id in enumerate(test_ids, start=1):
        img_paths = list((dsb_dir / img_id / "images").glob("*.png"))
        if not img_paths:
            raise FileNotFoundError(f"No image found for {img_id}")
        img = load_image(img_paths[0])
        gt_masks = load_gt_instances(dsb_dir / img_id / "masks")
        prob = predict_probability(model, img, device)
        pred_masks, scores, binary = components_from_probability(prob)
        mat = iou_matrix(pred_masks, gt_masks)
        records.append({
            "image_id": img_id,
            "n_pred": len(pred_masks),
            "n_gt": len(gt_masks),
            "scores": scores,
            "iou_mat": mat,
            "binary_iou": binary_iou(binary, gt_masks),
        })
        if i % 20 == 0 or i == len(test_ids):
            print(f"  {i}/{len(test_ids)} images")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def get_configs(ckpt_dir, scope):
    preferred = [
        "aug_no_augmentation",
        "aug_geometric_only",
        "aug_geometric_and_intensity",
        "tl_full_finetune_budget500",
        "tl_frozen_backbone_budget500",
        "tl_partial_finetune_budget500",
        "combined_geometric_only",
        "combined_geometric_and_intensity",
    ]
    if scope == "main":
        names = preferred
    else:
        names = []
        names += [
            "aug_no_augmentation",
            "aug_geometric_only",
            "aug_geometric_and_intensity",
        ]
        for budget in [40, 100, 250, 500]:
            for cfg in ["full_finetune", "frozen_backbone", "partial_finetune"]:
                names.append(f"tl_{cfg}_budget{budget}")
        names += ["combined_geometric_only", "combined_geometric_and_intensity"]

    found = {}
    for name in names:
        p = ckpt_dir / f"{name}.pth"
        if p.exists():
            found[name] = p
        else:
            print(f"Warning: missing checkpoint {p}")
    return found


def write_per_image(out_dir, all_records):
    rows = []
    for model_name, records in all_records.items():
        for r in records:
            # per-image Hungarian F1 at 0.5 and 0.7
            row = {
                "model": model_name,
                "image_id": r["image_id"],
                "binary_iou": r["binary_iou"],
                "n_pred_instances": r["n_pred"],
                "n_gt_instances": r["n_gt"],
            }
            for t in [0.50, 0.70]:
                tp, fp, fn = hungarian_counts(r["iou_mat"], t)
                p, rec, f1 = prf(tp, fp, fn)
                row[f"instance_precision_{int(t*100)}"] = p
                row[f"instance_recall_{int(t*100)}"] = rec
                row[f"instance_f1_{int(t*100)}"] = f1
                row[f"tp_{int(t*100)}"] = tp
                row[f"fp_{int(t*100)}"] = fp
                row[f"fn_{int(t*100)}"] = fn
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "routeB_per_image_metrics.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".", help="Project root containing results/")
    ap.add_argument("--dsb-dir", default=None, help="Optional explicit DSB2018 directory")
    ap.add_argument("--scope", choices=["main", "all"], default="all")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="Paired image bootstrap replicates (default 2000)")
    ap.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    ckpt_dir = project_root / "results/checkpoints"
    out_dir = project_root / "results/routeB_instance_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    dsb_dir = resolve_dsb_dir(project_root, args.dsb_dir)
    test_ids, split_info = make_test_ids(dsb_dir)
    device = choose_device()

    print("Project root:", project_root)
    print("DSB2018:", dsb_dir)
    print("Device:", device)
    print("Split reconstructed from code:", split_info)
    print("Test images:", len(test_ids))

    configs = get_configs(ckpt_dir, args.scope)
    if not configs:
        raise RuntimeError("No requested checkpoints found.")

    all_records = {}
    summaries = {}

    for name, path in configs.items():
        records = evaluate_checkpoint(name, path, dsb_dir, test_ids, device)
        all_records[name] = records
        summaries[name] = summary_metrics(records)
        s = summaries[name]
        print(
            f"  {name}: binary IoU={s['binary_iou_mean']:.4f}, "
            f"F1@.50={s['instance_f1_50']:.4f}, "
            f"F1@.70={s['instance_f1_70']:.4f}, "
            f"mF1={s['mean_instance_f1_50_95']:.4f}, "
            f"AP50={s['ap_50']:.4f}, mAP={s['map_50_95']:.4f}"
        )

    write_per_image(out_dir, all_records)

    # Bootstrap CIs
    rng = np.random.default_rng(args.bootstrap_seed)
    model_cis = {}
    print(f"\nBootstrapping {args.bootstrap} resamples per model...")
    for name, records in all_records.items():
        print("  CI:", name)
        model_cis[name] = bootstrap_model(records, args.bootstrap, rng)

    # Key paired comparisons, direction = first minus second
    comparisons_to_try = [
        ("aug_geometric_only", "aug_no_augmentation"),
        ("aug_geometric_and_intensity", "aug_geometric_only"),
        ("tl_full_finetune_budget500", "tl_frozen_backbone_budget500"),
        ("tl_full_finetune_budget500", "tl_partial_finetune_budget500"),
        ("tl_full_finetune_budget500", "aug_geometric_only"),
        ("combined_geometric_only", "aug_geometric_only"),
        ("combined_geometric_and_intensity", "aug_geometric_only"),
    ]
    pairwise = {}
    for a, b in comparisons_to_try:
        if a in all_records and b in all_records:
            print(f"  paired bootstrap: {a} - {b}")
            pairwise[f"{a}__minus__{b}"] = bootstrap_pair(
                all_records[a], all_records[b], args.bootstrap, rng
            )

    # Flat summary CSV
    flat_rows = []
    for name, s in summaries.items():
        row = {
            "model": name,
            "n_images": s["n_images"],
            "binary_iou_mean": s["binary_iou_mean"],
            "binary_iou_sd": s["binary_iou_sd"],
            "instance_f1_50": s["instance_f1_50"],
            "instance_f1_70": s["instance_f1_70"],
            "mean_instance_f1_50_95": s["mean_instance_f1_50_95"],
            "ap_50": s["ap_50"],
            "ap_75": s["ap_75"],
            "map_50_95": s["map_50_95"],
        }
        for metric, ci in model_cis[name].items():
            row[f"{metric}_ci_low"] = ci[0]
            row[f"{metric}_ci_high"] = ci[1]
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(out_dir / "routeB_summary.csv", index=False)

    serializable_summaries = {}
    for name, s in summaries.items():
        serializable_summaries[name] = s

    payload = {
        "method": {
            "foreground_threshold": FG_THRESHOLD,
            "predicted_instances": "8-connected components",
            "instance_confidence": "mean sigmoid probability within predicted component",
            "hungarian_matching": True,
            "ap_definition": "single-class 101-point interpolated mask AP",
            "iou_thresholds": [float(x) for x in IOU_THRESHOLDS],
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_unit": "test image",
            "split_seed": SPLIT_SEED,
        },
        "paths": {
            "project_root": str(project_root),
            "dsb_dir": str(dsb_dir),
            "checkpoint_dir": str(ckpt_dir),
        },
        "split_info": split_info,
        "summaries": serializable_summaries,
        "bootstrap_ci95": model_cis,
        "paired_bootstrap_differences": pairwise,
    }
    with open(out_dir / "routeB_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    # Human-readable report
    with open(out_dir / "routeB_report.txt", "w") as f:
        f.write("ROUTE B INSTANCE-LEVEL RE-EVALUATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"DSB directory: {dsb_dir}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Split: {split_info}\n\n")
        for name, s in summaries.items():
            f.write(f"{name}\n")
            f.write(f"  binary IoU mean: {s['binary_iou_mean']:.4f}\n")
            f.write(f"  instance F1@0.50: {s['instance_f1_50']:.4f}\n")
            f.write(f"  instance F1@0.70: {s['instance_f1_70']:.4f}\n")
            f.write(f"  mean instance F1@0.50:0.95: {s['mean_instance_f1_50_95']:.4f}\n")
            f.write(f"  AP@0.50: {s['ap_50']:.4f}\n")
            f.write(f"  AP@0.75: {s['ap_75']:.4f}\n")
            f.write(f"  mAP@0.50:0.95: {s['map_50_95']:.4f}\n\n")

    print("\nDONE.")
    print("Outputs:", out_dir)
    print("Generated result files:")
    print("  routeB_summary.csv")
    print("  routeB_results.json")
    print("  routeB_report.txt")
    print("  routeB_per_image_metrics.csv")


if __name__ == "__main__":
    main()
