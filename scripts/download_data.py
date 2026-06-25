#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""
Download + preprocess MovieLens/Amazon datasets into $GR_DATA_ROOT.

Reads .env for GR_DATA_ROOT via python-dotenv, then runs the research
preprocessor's `preprocess_rating()` for the requested datasets. The
preprocessor hardcodes `tmp/<dataset>/...` paths, so this script chdirs
into the repo and symlinks `tmp/` to GR_DATA_ROOT.

Usage:
  python3 scripts/download_data.py                     # default: ml-1m
  python3 scripts/download_data.py ml-1m ml-20m
  python3 scripts/download_data.py --all               # ml-1m, ml-20m, amzn-books
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED = ["ml-1m", "ml-20m", "amzn-books"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["ml-1m"],
        help=f"datasets to preprocess (default: ml-1m). Supported: {SUPPORTED}",
    )
    parser.add_argument(
        "--all", action="store_true", help="preprocess all supported datasets"
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    dataset_dir = os.environ.get("GR_DATA_ROOT")
    if not dataset_dir:
        sys.exit("GR_DATA_ROOT not set (define it in .env)")
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # The preprocessor writes to `./tmp/<dataset>/`. Symlink repo/tmp -> GR_DATA_ROOT
    # so the output lands on the shared filesystem.
    tmp_link = REPO_ROOT / "tmp"
    if tmp_link.is_symlink():
        if tmp_link.readlink() != dataset_dir:
            sys.exit(
                f"{tmp_link} -> {tmp_link.readlink()} but expected {dataset_dir}. "
                "Remove or fix the symlink."
            )
    elif tmp_link.exists():
        sys.exit(f"{tmp_link} exists and is not a symlink; remove it and re-run.")
    else:
        tmp_link.symlink_to(dataset_dir)

    os.chdir(REPO_ROOT)

    # Import after chdir so any relative paths inside preprocessor resolve correctly.
    from generative_recommenders.research.data.preprocessor import (
        get_common_preprocessors,
    )

    requested = SUPPORTED if args.all else args.datasets
    unknown = [d for d in requested if d not in SUPPORTED]
    if unknown:
        sys.exit(f"unknown dataset(s): {unknown}. Supported: {SUPPORTED}")

    preprocessors = get_common_preprocessors()
    for name in requested:
        print(f"[download_data] preprocessing {name} -> {dataset_dir}/{name}")
        preprocessors[name].preprocess_rating()
    print("[download_data] done")


if __name__ == "__main__":
    main()
