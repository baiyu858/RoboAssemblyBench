#!/usr/bin/env python3
"""Resume the InternUtopia full asset download and safely extract archives.

This script is intentionally non-interactive:
- resumes the Hugging Face dataset download into the assets directory;
- extracts normal zip files with path safety checks;
- skips multipart zip sets until all numbered parts are present;
- extracts ready multipart zip sets via a temporary combined archive.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


DEFAULT_REPO_ID = "InternRobotics/GRScenes"
DEFAULT_REPO_TYPE = "dataset"
MULTIPART_PART_RE = re.compile(r"^(?P<stem>.+)\.z(?P<index>\d+)$", re.IGNORECASE)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_assets_dir() -> Path:
    return repo_root() / "internutopia" / "assets"


def run_command(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    logging.info("Running command: %s", shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, env=env)
    return proc.returncode


def resume_hf_download(assets_dir: Path, repo_id: str, repo_type: str, retries: int, retry_delay: int) -> bool:
    cmd = [
        "huggingface-cli",
        "download",
        repo_id,
        "--repo-type",
        repo_type,
        "--local-dir",
        str(assets_dir),
        "--resume-download",
    ]
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    for attempt in range(1, retries + 1):
        rc = run_command(cmd, env=env)
        if rc == 0:
            logging.info("Hugging Face download finished successfully.")
            return True
        if attempt < retries:
            logging.warning(
                "Hugging Face download failed on attempt %d/%d with exit code %d; retrying in %ds.",
                attempt,
                retries,
                rc,
                retry_delay,
            )
            time.sleep(retry_delay)
        else:
            logging.warning(
                "Hugging Face download failed on attempt %d/%d with exit code %d; continuing to extraction.",
                attempt,
                retries,
                rc,
            )
    return False


def safe_commonpath(base: Path, candidate: Path) -> bool:
    base_str = str(base.resolve())
    cand_str = str(candidate.resolve(strict=False))
    try:
        return os.path.commonpath([base_str, cand_str]) == base_str
    except ValueError:
        return False


def safe_extract_regular_zip(zip_path: Path, output_dir: Path) -> None:
    logging.info("Extracting regular zip: %s", zip_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (output_dir / info.filename).resolve(strict=False)
            if not safe_commonpath(output_dir, target):
                raise RuntimeError(f"Refusing to extract path outside destination: {info.filename}")

            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def split_archive_parts(zip_path: Path) -> dict[int, Path]:
    stem = zip_path.stem
    parts: dict[int, Path] = {0: zip_path}
    if not zip_path.parent.is_dir():
        return parts

    for entry in zip_path.parent.iterdir():
        if not entry.is_file():
            continue
        match = MULTIPART_PART_RE.match(entry.name)
        if not match or match.group("stem") != stem:
            continue
        parts[int(match.group("index"))] = entry
    return parts


def multipart_status(zip_path: Path) -> tuple[bool, list[int], dict[int, Path]]:
    parts = split_archive_parts(zip_path)
    numbered = sorted(index for index in parts if index > 0)
    if not numbered:
        return False, [], parts

    max_index = numbered[-1]
    missing = [idx for idx in range(1, max_index + 1) if idx not in parts]
    ready = zip_path.exists() and not missing
    return ready, missing, parts


def extract_split_zip(zip_path: Path, output_dir: Path) -> None:
    ready, missing, parts = multipart_status(zip_path)
    if not ready:
        if missing:
            logging.info("Skipping multipart zip until complete: %s (missing parts: %s)", zip_path, missing)
        else:
            logging.info("Skipping multipart zip without numbered parts: %s", zip_path)
        return

    part_list = [parts[index] for index in sorted(parts) if index > 0]
    latest_source_mtime = max(p.stat().st_mtime for p in [zip_path, *part_list])
    logging.info("Extracting multipart zip: %s", zip_path)

    with tempfile.TemporaryDirectory(prefix="internutopia_zip_") as tempdir:
        combined_zip = Path(tempdir) / f"{zip_path.stem}_combined.zip"
        if combined_zip.exists() and combined_zip.stat().st_mtime >= latest_source_mtime:
            pass
        else:
            rc = run_command(
                ["zip", "-FF", zip_path.name, "--out", str(combined_zip)],
                cwd=zip_path.parent,
            )
            if rc != 0:
                raise subprocess.CalledProcessError(rc, ["zip", "-FF", zip_path.name, "--out", str(combined_zip)])

        safe_extract_regular_zip(combined_zip, output_dir)


def find_zip_archives(root: Path) -> list[Path]:
    archives = []
    for path in root.rglob("*.zip"):
        if path.name.startswith("."):
            continue
        if path.is_file():
            archives.append(path)
    return sorted(set(archives))


def rename_asset_dirs(root: Path) -> None:
    rename_map = {
        "target_69_new": "home_scenes",
        "target_30_new": "commercial_scenes",
    }
    for dir_path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        replacement = rename_map.get(dir_path.name)
        if not replacement:
            continue
        new_path = dir_path.with_name(replacement)
        if new_path.exists():
            logging.info("Not renaming %s because %s already exists.", dir_path, new_path)
            continue
        logging.info("Renaming %s -> %s", dir_path, new_path)
        dir_path.rename(new_path)


def extract_archives(root: Path, max_passes: int) -> None:
    processed: set[Path] = set()
    for pass_index in range(1, max_passes + 1):
        archives = find_zip_archives(root)
        if not archives:
            logging.info("No zip archives found on pass %d.", pass_index)
            return

        logging.info("Archive extraction pass %d found %d zip archive(s).", pass_index, len(archives))
        progress = False
        for zip_path in archives:
            resolved_zip = zip_path.resolve()
            if resolved_zip in processed:
                continue
            try:
                ready, missing, parts = multipart_status(zip_path)
                if len([idx for idx in parts if idx > 0]) > 0:
                    if not ready:
                        logging.info(
                            "Skipping multipart zip until complete: %s (missing parts: %s)",
                            zip_path,
                            missing,
                        )
                        continue
                    extract_split_zip(zip_path, zip_path.parent)
                else:
                    safe_extract_regular_zip(zip_path, zip_path.parent)
                logging.info("Extracted: %s", zip_path)
                processed.add(resolved_zip)
                progress = True
            except zipfile.BadZipFile as exc:
                logging.warning("Failed to extract %s as a zip archive: %s", zip_path, exc)
            except subprocess.CalledProcessError as exc:
                logging.warning("External unzip/repair command failed for %s: %s", zip_path, exc)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to extract %s: %s", zip_path, exc)

        rename_asset_dirs(root)
        if not progress:
            logging.info("No extraction progress on pass %d; stopping.", pass_index)
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume InternUtopia full asset download and extraction.")
    parser.add_argument("--assets-dir", type=Path, default=default_assets_dir(), help="Assets directory to resume.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo ID.")
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE, help="Hugging Face repo type.")
    parser.add_argument("--download-retries", type=int, default=3, help="How many times to retry the download.")
    parser.add_argument("--retry-delay", type=int, default=30, help="Seconds to wait between download retries.")
    parser.add_argument("--extract-passes", type=int, default=4, help="Maximum extraction passes over nested zips.")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    assets_dir = args.assets_dir.resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Repo root: %s", repo_root())
    logging.info("Assets dir: %s", assets_dir)

    resume_hf_download(
        assets_dir=assets_dir,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        retries=max(1, args.download_retries),
        retry_delay=max(1, args.retry_delay),
    )
    extract_archives(assets_dir, max_passes=max(1, args.extract_passes))
    logging.info("Resume run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
