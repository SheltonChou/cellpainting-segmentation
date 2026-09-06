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

    for model, expected in EXPECTED_MAP.items():
        require(model in rows, f"Route B summary is missing {model}")
        observed = float(rows[model]["map_50_95"])
        require(
            math.isclose(observed, expected, abs_tol=1e-12),
            f"{model} mAP changed: expected {expected}, found {observed}",
        )
        require(int(rows[model]["n_images"]) == 101, f"{model} does not use 101 images")
    return len(EXPECTED_MAP)


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


def main() -> int:
    try:
        repeated = check_multi_seed_results()
        route_b = check_route_b()
        volumes = check_pseudo_volume()
    except (CheckFailure, FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Verified {repeated} eight-seed configurations.")
    print(f"Verified {route_b} final Route B mAP values on 101 test images.")
    print(f"Verified {volumes} pseudo-label volume records.")
    print("All archived-result checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

