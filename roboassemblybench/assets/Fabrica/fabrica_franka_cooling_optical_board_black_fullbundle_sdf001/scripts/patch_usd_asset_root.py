#!/usr/bin/env python3
"""Patch Isaac official asset root references in packaged USD files."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ROOT = "/data/hjj/geniesim/isaac_assets_5.1/extracted/Assets/Isaac/5.1/Isaac"


def patch_file(path: Path, old_root: str, new_root: str, dry_run: bool) -> int:
    text = path.read_text(encoding="utf-8")
    count = text.count(old_root)
    if count and not dry_run:
        path.write_text(text.replace(old_root, new_root), encoding="utf-8")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-asset-root", required=True, help="Path to the Isaac official asset root.")
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--old-root", default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    new_root = str(Path(args.isaac_asset_root).resolve())
    files = [
        package_root / "scene" / "scene.usda",
        package_root / "scene" / "clean_packing_table.usda",
    ]

    total = 0
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        count = patch_file(path, args.old_root, new_root, args.dry_run)
        total += count
        action = "would patch" if args.dry_run else "patched"
        print(f"{action} {count} reference(s): {path}")

    print(f"total references patched: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
