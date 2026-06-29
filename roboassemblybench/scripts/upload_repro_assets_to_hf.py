#!/usr/bin/env python3
"""Upload local reproducibility assets that should not live in the GitHub repo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ID = "baiyu858/InternUtopia-repro-assets"
DEFAULT_ASSETS = {
    "third_part/Fabrica": REPO_ROOT / "third_part/Fabrica",
    "third_part/factory_dual_franka_peg_transfer": REPO_ROOT / "third_part/factory_dual_franka_peg_transfer",
    "IsaacLab": REPO_ROOT / "IsaacLab",
    "recordings": REPO_ROOT / "recordings",
}
IGNORE_PATTERNS = [
    ".git",
    ".git/*",
    "**/.git/**",
    "__pycache__",
    "__pycache__/*",
    "**/__pycache__/**",
    ".pytest_cache",
    ".pytest_cache/*",
    "**/.pytest_cache/**",
    ".cache",
    ".cache/*",
    "**/.cache/**",
]


def _dir_summary(path: Path) -> dict[str, int]:
    file_count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_bytes += item.stat().st_size
    return {"file_count": file_count, "total_bytes": total_bytes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="Optional Hugging Face endpoint, for example https://hf-mirror.com.",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--asset",
        action="append",
        choices=sorted(DEFAULT_ASSETS),
        help="Asset key to upload. Repeat to upload a subset. Defaults to every existing asset.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.asset or list(DEFAULT_ASSETS)
    api = HfApi(endpoint=args.endpoint)

    manifest = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "endpoint": args.endpoint,
        "assets": {},
        "ignore_patterns": IGNORE_PATTERNS,
    }
    upload_items: list[tuple[str, Path]] = []
    for key in selected:
        path = DEFAULT_ASSETS[key]
        if not path.exists():
            manifest["assets"][key] = {"exists": False, "local_path": str(path)}
            continue
        manifest["assets"][key] = {"exists": True, "local_path": str(path), **_dir_summary(path)}
        upload_items.append((key, path))

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
    for key, path in upload_items:
        print(f"Uploading {path} -> {args.repo_id}/{key}")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=path,
            path_in_repo=key,
            ignore_patterns=IGNORE_PATTERNS,
            commit_message=f"Upload {key} reproducibility assets",
        )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_or_fileobj=json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
        path_in_repo="manifest.json",
        commit_message="Update reproducibility asset manifest",
    )


if __name__ == "__main__":
    main()
