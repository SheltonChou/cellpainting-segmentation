"""Portable project paths shared by the reproduction scripts.

Paths default to folders inside the repository and can be overridden with
environment variables.  This keeps the archived experiments runnable without
editing source files or relying on a particular user account.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


DATA_ROOT = _path_from_env("CELLPAINT_DATA_ROOT", REPO_ROOT / "data")
DSB_DIR = _path_from_env("DSB2018_DIR", DATA_ROOT / "dsb2018" / "stage1_train")
RXRX1_ROOT = _path_from_env("RXRX1_ROOT", DATA_ROOT / "rxrx1" / "rxrx1")
RXRX1_IMAGES_DIR = _path_from_env("RXRX1_IMAGES_DIR", RXRX1_ROOT / "images")
RXRX1_METADATA = _path_from_env("RXRX1_METADATA", RXRX1_ROOT / "metadata.csv")
RESULTS_ROOT = _path_from_env("CELLPAINT_RESULTS_DIR", REPO_ROOT / "results")
CHECKPOINTS_DIR = _path_from_env(
    "CELLPAINT_CHECKPOINT_DIR", RESULTS_ROOT / "checkpoints"
)


def ensure_output_dirs() -> None:
    """Create the standard writable output directories."""

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "visualisations").mkdir(parents=True, exist_ok=True)
