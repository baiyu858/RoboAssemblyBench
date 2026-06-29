#!/usr/bin/env python3
"""Download Hugging Face reproducibility assets into their expected local paths."""

from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from urllib.parse import quote

import requests
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tqdm.auto import tqdm


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
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--no-mirror-fallback",
        action="store_true",
        help="Disable direct resolve-url fallback for older huggingface_hub versions against hf-mirror.com.",
    )
    return parser.parse_args()


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _resolve_url(endpoint: str, repo_id: str, repo_type: str, revision: str, path: str) -> str:
    repo_prefix = "" if repo_type == "model" else f"{repo_type}s/"
    return (
        f"{endpoint.rstrip('/')}/{repo_prefix}{repo_id}/resolve/"
        f"{quote(revision, safe='')}/{quote(path, safe='/')}"
    )


def _download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        with temp_path.open("wb") as handle:
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=str(output_path),
                disable=total == 0,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    progress.update(len(chunk))
    temp_path.replace(output_path)


def _download_with_resolve_urls(args: argparse.Namespace, include_patterns: list[str]) -> None:
    endpoint = args.endpoint or "https://huggingface.co"
    api = HfApi(endpoint=endpoint)
    info = api.repo_info(repo_id=args.repo_id, repo_type=args.repo_type, revision=args.revision)
    revision = args.revision or info.sha
    if revision is None:
        revision = "main"
    if info.siblings is None:
        raise RuntimeError(f"Could not list files for {args.repo_id}")

    paths = [
        sibling.rfilename
        for sibling in info.siblings
        if _matches_any(sibling.rfilename, include_patterns)
    ]
    for path in paths:
        url = _resolve_url(endpoint, args.repo_id, args.repo_type, revision, path)
        _download_file(url, args.local_dir / path)


def main() -> None:
    args = parse_args()
    include_patterns = args.include or [
        "third_part/Fabrica/**",
        "third_part/factory_dual_franka_peg_transfer/**",
        "IsaacLab/**",
        "recordings/**",
        "manifest.json",
    ]
    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            local_dir=args.local_dir,
            allow_patterns=include_patterns,
            endpoint=args.endpoint,
            max_workers=args.max_workers,
            local_dir_use_symlinks=False,
        )
    except LocalEntryNotFoundError:
        if args.no_mirror_fallback or args.endpoint != "https://hf-mirror.com":
            raise
        print("snapshot_download failed against hf-mirror.com; falling back to direct resolve-url download.")
        _download_with_resolve_urls(args, include_patterns)
    print(f"Downloaded {args.repo_id} into {args.local_dir}")


if __name__ == "__main__":
    main()
