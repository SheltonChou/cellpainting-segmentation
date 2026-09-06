#!/usr/bin/env python3
"""Validate the local DSB2018 and RxRx1 directory layouts."""

from __future__ import annotations

import sys
from pathlib import Path

from project_paths import DSB_DIR, RXRX1_IMAGES_DIR, RXRX1_METADATA


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)


def check_dsb() -> bool:
    if not DSB_DIR.is_dir():
        fail(f"DSB2018 directory not found: {DSB_DIR}")
        return False

    image_ids = sorted(path for path in DSB_DIR.iterdir() if path.is_dir())
    malformed = []
    for image_dir in image_ids:
        if not list((image_dir / "images").glob("*.png")):
            malformed.append(f"{image_dir.name}: missing image")
        if not list((image_dir / "masks").glob("*.png")):
            malformed.append(f"{image_dir.name}: missing masks")

    print(f"DSB2018 directory: {DSB_DIR}")
    print(f"DSB2018 image-ID directories: {len(image_ids)}")
    if len(image_ids) != 670:
        fail("expected 670 labelled DSB2018 image-ID directories")
    if malformed:
        for item in malformed[:10]:
            fail(item)
    return len(image_ids) == 670 and not malformed


def check_rxrx1() -> bool:
    ok = True
    print(f"RxRx1 image directory: {RXRX1_IMAGES_DIR}")
    print(f"RxRx1 metadata: {RXRX1_METADATA}")
    if not RXRX1_IMAGES_DIR.is_dir():
        fail(f"RxRx1 image directory not found: {RXRX1_IMAGES_DIR}")
        return False
    if not RXRX1_METADATA.is_file():
        fail(f"RxRx1 metadata file not found: {RXRX1_METADATA}")
        ok = False

    channel_one = sorted(RXRX1_IMAGES_DIR.rglob("*_w1.png"))
    print(f"RxRx1 channel-1 images: {len(channel_one)}")
    if len(channel_one) < 500:
        fail("at least 500 channel-1 images are required")
        ok = False
    return ok


def main() -> int:
    ok = check_dsb() and check_rxrx1()
    if ok:
        print("Data layout checks passed.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

