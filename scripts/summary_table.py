import json
import pandas as pd
from pathlib import Path
from project_paths import RESULTS_ROOT

RESULT_DIR = RESULTS_ROOT

def main():
    # Load all results
    unet     = json.load(open(RESULT_DIR / "unet_all_results.json"))
    full_met = json.load(open(RESULT_DIR / "full_metrics.json"))
    map_full = json.load(open(RESULT_DIR / "map_results_full.json"))
    baseline = json.load(open(RESULT_DIR / "baseline/summary.json"))

    rows = []

    # Cellpose baseline
    rows.append({
        "Method": "Cellpose (baseline)",
        "Config": "-",
        "Budget": "N/A",
        "Val IoU": baseline["mean_iou"],
        "Test IoU": "-",
        "F1@0.5": baseline["f1_05_mean"],
        "F1@0.7": baseline["f1_07_mean"],
        "mAP": "-",
    })

    # Aug ablation
    for aug in ["no_augmentation", "geometric_only", "geometric_and_intensity"]:
        k = f"aug_{aug}"
        rows.append({
            "Method": "U-Net + ResNet50",
            "Config": aug,
            "Budget": "Full",
            "Val IoU": round(unet[k], 4),
            "Test IoU": full_met.get(k, {}).get("mean_iou", "-"),
            "F1@0.5": full_met.get(k, {}).get("f1_05", "-"),
            "F1@0.7": full_met.get(k, {}).get("f1_07", "-"),
            "mAP": map_full.get(k, {}).get("mAP", "-"),
        })

    # TL configs
    for budget in [40, 100, 250, 500]:
        for cfg in ["full_finetune", "frozen_backbone", "partial_finetune"]:
            k = f"tl_{cfg}_budget{budget}"
            rows.append({
                "Method": "U-Net + ResNet50",
                "Config": cfg,
                "Budget": budget,
                "Val IoU": round(unet[k], 4),
                "Test IoU": full_met.get(k, {}).get("mean_iou", "-"),
                "F1@0.5": full_met.get(k, {}).get("f1_05", "-"),
                "F1@0.7": full_met.get(k, {}).get("f1_07", "-"),
                "mAP": map_full.get(k, {}).get("mAP", "-"),
            })

    # Combined
    for k in ["combined_geometric_only", "combined_geometric_and_intensity"]:
        rows.append({
            "Method": "U-Net + Pseudo",
            "Config": k.replace("combined_", ""),
            "Budget": "Full",
            "Val IoU": "-",
            "Test IoU": full_met.get(k, {}).get("mean_iou", "-"),
            "F1@0.5": full_met.get(k, {}).get("f1_05", "-"),
            "F1@0.7": full_met.get(k, {}).get("f1_07", "-"),
            "mAP": map_full.get(k, {}).get("mAP", "-"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "full_results_table.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved to {RESULT_DIR}/full_results_table.csv")

if __name__ == "__main__":
    main()
