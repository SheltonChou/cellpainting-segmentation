#!/usr/bin/env python3
"""Check the internal consistency of the archived result record.

This script intentionally uses only the Python standard library so that a
fresh clone can be audited before the training environment is installed.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192]
EXPECTED_MAP = {
    "aug_no_augmentation": 0.3430936019254671,
    "aug_geometric_only": 0.38243387567364817,
    "aug_geometric_and_intensity": 0.3662567768225382,
    "tl_full_finetune_budget500": 0.39279602254269763,
    "tl_frozen_backbone_budget500": 0.37726868022886284,
    "tl_partial_finetune_budget500": 0.3817350160124483,
    "combined_geometric_only": 0.3810058711270886,
    "combined_geometric_and_intensity": 0.3851591396238186,
}
EXPECTED_FINAL = {
    "aug_no_augmentation": (0.8214, 0.5670, 0.3431),
    "aug_geometric_only": (0.8473, 0.6059, 0.3824),
    "aug_geometric_and_intensity": (0.8339, 0.5881, 0.3663),
    "tl_full_finetune_budget40": (0.7591, 0.4793, 0.2576),
    "tl_frozen_backbone_budget40": (0.7220, 0.4239, 0.2109),
    "tl_partial_finetune_budget40": (0.7500, 0.4593, 0.2257),
    "tl_full_finetune_budget100": (0.8097, 0.5631, 0.3120),
    "tl_frozen_backbone_budget100": (0.7940, 0.5390, 0.3189),
    "tl_partial_finetune_budget100": (0.8141, 0.5608, 0.3164),
    "tl_full_finetune_budget250": (0.8284, 0.5790, 0.3596),
    "tl_frozen_backbone_budget250": (0.8220, 0.5709, 0.3491),
    "tl_partial_finetune_budget250": (0.8272, 0.5669, 0.3522),
    "tl_full_finetune_budget500": (0.8512, 0.6264, 0.3928),
    "tl_frozen_backbone_budget500": (0.8367, 0.5948, 0.3773),
    "tl_partial_finetune_budget500": (0.8410, 0.5968, 0.3817),
    "combined_geometric_only": (0.8273, 0.5850, 0.3810),
    "combined_geometric_and_intensity": (0.8320, 0.5944, 0.3852),
}
EXPECTED_TEST_IDS = 101


class CheckFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def read_script_seeds() -> list[int]:
    path = ROOT / "scripts" / "multi_seed.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SEEDS":
                    value = ast.literal_eval(node.value)
                    require(isinstance(value, list), "SEEDS must be a list")
                    return value
    raise CheckFailure("SEEDS assignment not found in scripts/multi_seed.py")


def check_multi_seed_results() -> int:
    script_seeds = read_script_seeds()
    require(script_seeds == EXPECTED_SEEDS, f"unexpected seed list: {script_seeds}")

    path = ROOT / "results" / "multi_seed" / "multi_seed_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    checked = 0
    for name, record in payload.items():
        if "ious" not in record:
            continue
        values = record["ious"]
        require(len(values) == len(EXPECTED_SEEDS), f"{name} has {len(values)} values")
        require(
            math.isclose(statistics.fmean(values), record["mean"], abs_tol=1e-12),
            f"{name} mean does not match its eight values",
        )
        require(
            math.isclose(statistics.pstdev(values), record["std"], abs_tol=1e-12),
            f"{name} standard deviation does not match its eight values",
        )
        checked += 1
    require(checked == 15, f"expected 15 repeated-run configurations, found {checked}")
    return checked


def check_route_b() -> int:
    path = ROOT / "results" / "routeB_instance_eval" / "routeB_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["model"]: row for row in csv.DictReader(handle)}

    require(set(rows) == set(EXPECTED_FINAL), "Route B model list does not contain 17 final models")
    for model, expected_metrics in EXPECTED_FINAL.items():
        require(model in rows, f"Route B summary is missing {model}")
        observed_metrics = (
            float(rows[model]["binary_iou_mean"]),
            float(rows[model]["instance_f1_70"]),
            float(rows[model]["map_50_95"]),
        )
        for label, observed, expected in zip(
            ("binary IoU", "F1 at 0.70", "mAP"), observed_metrics, expected_metrics
        ):
            require(
                math.isclose(observed, expected, abs_tol=0.00005),
                f"{model} {label} changed: expected {expected}, found {observed}",
            )
        require(
            int(rows[model]["n_images"]) == EXPECTED_TEST_IDS,
            f"{model} does not use {EXPECTED_TEST_IDS} images",
        )

    for model, expected in EXPECTED_MAP.items():
        require(
            math.isclose(float(rows[model]["map_50_95"]), expected, abs_tol=1e-12),
            f"{model} full precision mAP changed",
        )

    detail_path = ROOT / "results" / "routeB_instance_eval" / "routeB_per_image_metrics.csv"
    with detail_path.open(newline="", encoding="utf-8") as handle:
        detail = list(csv.DictReader(handle))
    require(
        len(detail) == len(EXPECTED_FINAL) * EXPECTED_TEST_IDS,
        "Route B per-image table has an unexpected row count",
    )
    id_sets = {}
    for row in detail:
        id_sets.setdefault(row["model"], set()).add(row["image_id"])
    require(set(id_sets) == set(EXPECTED_FINAL), "per-image model list differs from summary")
    reference_ids = id_sets["aug_no_augmentation"]
    require(len(reference_ids) == EXPECTED_TEST_IDS, "test image identifiers are not unique")
    require(all(ids == reference_ids for ids in id_sets.values()), "models use different test images")
    return len(EXPECTED_FINAL)


def check_pseudo_volume() -> int:
    path = ROOT / "results" / "pseudo_volume" / "volume_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_volumes = ["0", "100", "200", "300", "400", "500"]
    require(list(payload) == expected_volumes, "unexpected pseudo-label volume sequence")
    values = [payload[key]["val_iou"] for key in expected_volumes]
    require(
        all(left > right for left, right in zip(values, values[1:])),
        "archived pseudo-label volume IoU values are not strictly decreasing",
    )
    return len(values)


def check_dataset_records() -> int:
    split = json.loads(
        (ROOT / "results" / "rxrx1_analysis" / "split_info.json").read_text(encoding="utf-8")
    )
    require(
        (split["n_train"], split["n_val"], split["n_test"]) == (81212, 22142, 22156),
        "RxRx1 grouped split counts changed",
    )
    baseline = json.loads(
        (ROOT / "results" / "baseline_rxrx1" / "summary.json").read_text(encoding="utf-8")
    )
    require(baseline["n_images"] == 200, "RxRx1 Cellpose sample size changed")
    require(math.isclose(baseline["mean_cells"], 446.18, abs_tol=1e-12), "mean cell count changed")
    require(math.isclose(baseline["std_cells"], 172.77, abs_tol=1e-12), "cell count SD changed")
    return 2


def check_statistical_records() -> int:
    payload = json.loads(
        (ROOT / "results" / "statistical_tests.json").read_text(encoding="utf-8")
    )
    expected = {
        "wilcoxon_geo_vs_no": 0.0078,
        "wilcoxon_int_vs_no": 0.0078,
        "wilcoxon_geo_vs_int": 0.0391,
        "wilcoxon_full_vs_frozen_budget250": 0.0078,
        "wilcoxon_full_vs_partial_budget250": 0.0078,
        "wilcoxon_full_vs_frozen_budget500": 0.1094,
        "wilcoxon_full_vs_partial_budget500": 0.0781,
    }
    require(set(payload) == set(expected), "Wilcoxon comparison list changed")
    for name, p_value in expected.items():
        require(
            math.isclose(payload[name]["p_value"], p_value, abs_tol=1e-12),
            f"{name} p-value changed",
        )
    return len(expected)


def check_pseudo_thresholds() -> int:
    payload = json.loads(
        (ROOT / "results" / "pseudo_label_final" / "pseudo_final_results.json").read_text(
            encoding="utf-8"
        )
    )
    require(list(payload) == ["0.5", "0.6", "0.7", "0.8", "0.9"], "threshold list changed")
    for threshold, rounds in payload.items():
        require([row["round"] for row in rounds] == [0, 1, 2, 3], f"rounds changed at {threshold}")
        for row in rounds[1:]:
            require(row["n_pseudo"] == 500, f"pseudo-label count changed at {threshold}")
            require(row["accept_rate"] == 100.0, f"acceptance rate changed at {threshold}")
    return len(payload)


def check_density_record() -> int:
    path = ROOT / "results" / "density_analysis" / "density_performance.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == EXPECTED_TEST_IDS, "density table must contain only 101 test images")
    require(len({row["image_id"] for row in rows}) == EXPECTED_TEST_IDS, "density image IDs repeat")

    summary = json.loads(
        (ROOT / "results" / "density_analysis" / "density_summary.json").read_text(encoding="utf-8")
    )
    require(summary["split_seed"] == 42, "density split seed changed")
    require(summary["n_test_images"] == EXPECTED_TEST_IDS, "density summary size changed")
    expected_r = {
        "aug_no_augmentation": -0.2343567119,
        "aug_geometric_only": -0.2453265950,
        "tl_full_finetune_budget500": -0.2899975942,
    }
    require(set(summary["models"]) == set(expected_r), "density model list changed")
    for model, expected in expected_r.items():
        observed = summary["models"][model]["pearson_r"]
        require(math.isclose(observed, expected, abs_tol=1e-7), f"{model} density r changed")
    geo = summary["models"]["aug_geometric_only"]
    require(
        geo["bin_counts"] == {"1-5": 5, "6-15": 24, "16-30": 29, "31-50": 20, "51-100": 18, "100+": 5},
        "density bin counts changed",
    )
    return len(rows)


def main() -> int:
    try:
        repeated = check_multi_seed_results()
        route_b = check_route_b()
        volumes = check_pseudo_volume()
        datasets = check_dataset_records()
        tests = check_statistical_records()
        thresholds = check_pseudo_thresholds()
        density = check_density_record()
    except (CheckFailure, FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Verified {repeated} configurations, each with eight seeds.")
    print(f"Verified all reported metrics for {route_b} final models on 101 common test images.")
    print(f"Verified {volumes} pseudo-label volume records.")
    print(f"Verified {datasets} dataset and baseline records.")
    print(f"Verified {tests} paired Wilcoxon comparisons and {thresholds} pseudo-label thresholds.")
    print(f"Verified density analysis on {density} reserved test images.")
    print("All archived result checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
