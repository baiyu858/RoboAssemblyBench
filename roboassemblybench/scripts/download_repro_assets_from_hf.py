#!/usr/bin/env python3
"""Download Hugging Face reproducibility assets into their expected local paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ID = "baiyu858/InternUtopia-repro-assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="Optional Hugging Face endpoint, for example https://hf-mirror.com.",
    )
    parser.add_argument("--local-dir", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Optional include pattern. Example: third_part/Fabrica/**",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_patterns = args.include or [
        "third_part/Fabrica/**",
        "third_part/factory_dual_franka_peg_transfer/**",
        "IsaacLab/**",
        "recordings/**",
        "manifest.json",
    ]
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=include_patterns,
        endpoint=args.endpoint,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {args.repo_id} into {args.local_dir}")


if __name__ == "__main__":
    main()
