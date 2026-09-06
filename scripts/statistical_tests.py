import json
import numpy as np
from pathlib import Path
from project_paths import RESULTS_ROOT
from scipy import stats

RESULT_DIR = RESULTS_ROOT

def main():
    # Load multi_seed results if available
    ms_file = RESULT_DIR / "multi_seed" / "multi_seed_results.json"
    
    results = {}
    
    if ms_file.exists():
        with open(ms_file) as f:
            ms = json.load(f)
        
        # Augmentation Wilcoxon tests
        aug_no  = ms["aug_no_augmentation"]["ious"]
        aug_geo = ms["aug_geometric_only"]["ious"]
        aug_int = ms["aug_geometric_and_intensity"]["ious"]
        
        stat1, p1 = stats.wilcoxon(aug_geo, aug_no)
        stat2, p2 = stats.wilcoxon(aug_int, aug_no)
        stat3, p3 = stats.wilcoxon(aug_geo, aug_int)
        
        results["wilcoxon_geo_vs_no"]  = {"statistic": stat1, "p_value": round(p1,4), "significant": bool(p1 < 0.05)}
        results["wilcoxon_int_vs_no"]  = {"statistic": stat2, "p_value": round(p2,4), "significant": bool(p2 < 0.05)}
        results["wilcoxon_geo_vs_int"] = {"statistic": stat3, "p_value": round(p3,4), "significant": bool(p3 < 0.05)}
        
        # TL Wilcoxon tests at budget 500
        for budget in [250, 500]:
            full    = ms.get(f"tl_full_finetune_budget{budget}",    {}).get("ious", [])
            frozen  = ms.get(f"tl_frozen_backbone_budget{budget}",  {}).get("ious", [])
            partial = ms.get(f"tl_partial_finetune_budget{budget}", {}).get("ious", [])
            
            if full and frozen:
                s, p = stats.wilcoxon(full, frozen)
                results[f"wilcoxon_full_vs_frozen_budget{budget}"] = {
                    "statistic": s, "p_value": round(p,4), "significant": bool(p < 0.05)}
            if full and partial:
                s, p = stats.wilcoxon(full, partial)
                results[f"wilcoxon_full_vs_partial_budget{budget}"] = {
                    "statistic": s, "p_value": round(p,4), "significant": bool(p < 0.05)}
        
        # Mean ± std summary
        print("\n===== MULTI-SEED SUMMARY =====")
        for k, v in ms.items():
            if "ious" in v:
                print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")
    
    else:
        print("multi_seed results not ready yet, skipping Wilcoxon tests")
        # Use single-run results as proxy
        unet_results = json.load(open(RESULT_DIR / "unet_all_results.json"))
        print("\n===== SINGLE-RUN RESULTS =====")
        for k, v in unet_results.items():
            print(f"  {k}: {v:.4f}")
    
    with open(RESULT_DIR / "statistical_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n===== STATISTICAL TESTS =====")
    for k, v in results.items():
        sig = "SIGNIFICANT" if v.get("significant") else "not significant"
        print(f"  {k}: p={v['p_value']} ({sig})")

if __name__ == "__main__":
    main()
