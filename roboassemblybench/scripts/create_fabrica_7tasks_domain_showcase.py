#!/usr/bin/env python3
"""Create a front-camera mosaic for the seven Fabrica tasks and five profiles.

The script reads either raw collection directories or converted LeRobot v3
directories.  One successful episode is selected for every task/profile pair;
missing pairs remain visible in the mosaic as a labelled pending tile.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


TASKS = (
    "beam",
    "car",
    "cooling_manifold",
    "duct",
    "gamepad",
    "plumbers_block",
    "stool_circular",
)
PROFILES = (
    "object_distractors",
    "texture",
    "lighting",
    "table_color",
    "scene",
)
FRONT_CAMERA_KEY = "observation.images.front"
DEFAULT_ROOT = Path("/data/a17/baiyongjie/data/fabrica_7tasks_50k_lerobot_v3")
DEFAULT_OUTPUT = Path(
    "/data/a17/baiyongjie/data/fabrica_7tasks_50k_lerobot_v3/showcase/"
    "fabrica_7tasks_domain_randomization_front_7x5.mp4"
)


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_candidates(raw_dir: Path) -> list[Path]:
    manifest_path = raw_dir / "collection_manifest.json"
    candidates: list[Path] = []
    if manifest_path.is_file():
        try:
            manifest = load_json(manifest_path)
            for item in (manifest.get("successful_episodes") or {}).values():
                metadata_path = Path(str(item.get("metadata_path") or ""))
                if metadata_path.is_file():
                    candidates.append(metadata_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    candidates.extend(raw_dir.rglob("episode_*_cartesian_raw/metadata.json"))
    return list(dict.fromkeys(path.resolve() for path in candidates))


def successful_metadata(raw_dir: Path, converted_dir: Path) -> list[Path]:
    candidates = _metadata_candidates(raw_dir)
    conversion_path = converted_dir / "roboassemblybench_conversion_manifest.json"
    if conversion_path.is_file():
        try:
            conversion = load_json(conversion_path)
            for item in conversion.get("episodes") or []:
                metadata_path = Path(str(item.get("source_metadata") or ""))
                if metadata_path.is_file():
                    candidates.append(metadata_path.resolve())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    valid: list[Path] = []
    seen: set[Path] = set()
    for metadata_path in candidates:
        if metadata_path in seen or not metadata_path.is_file():
            continue
        seen.add(metadata_path)
        try:
            metadata = load_json(metadata_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        if not bool((metadata.get("metrics") or {}).get("success", False)):
            continue
        videos = metadata.get("videos") or {}
        front = Path(str(videos.get(FRONT_CAMERA_KEY) or ""))
        if front.is_file() and front.stat().st_size > 0:
            valid.append(metadata_path)
    return sorted(
        valid,
        key=lambda path: (
            int((load_json(path) or {}).get("layout_seed", -1)),
            int((load_json(path) or {}).get("seed", -1)),
            str(path),
        ),
    )


def choose_episode(raw_dir: Path, converted_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    candidates = successful_metadata(raw_dir, converted_dir)
    if not candidates:
        return None, {"status": "pending", "raw_dir": str(raw_dir)}
    # The middle candidate avoids systematically showing only the first layout
    # while remaining deterministic as the collection grows.
    metadata_path = candidates[(len(candidates) - 1) // 2]
    metadata = load_json(metadata_path)
    assert isinstance(metadata, dict)
    return metadata_path, {
        "status": "available",
        "metadata_path": str(metadata_path),
        "video_path": str((metadata.get("videos") or {}).get(FRONT_CAMERA_KEY)),
        "seed": int(metadata.get("seed", -1)),
        "layout_seed": int(metadata.get("layout_seed", metadata.get("seed", -1))),
        "frame_count": int(metadata.get("frame_count", 0)),
        "source_fps": float(metadata.get("fps", 30)),
        "domain_randomization": metadata.get("domain_randomization") or {},
    }


def ffmpeg_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def tile_filter(
    index: int,
    label: str,
    tile_width: int,
    tile_height: int,
    duration: float,
    fps: int,
    available: bool,
    source_fps: float,
    frame_count: int,
) -> str:
    if available:
        # The input is deliberately accelerated to a fixed review duration.
        # tpad keeps a short episode from shortening the xstack output.
        source_duration = max(frame_count / max(source_fps, 1.0), 1e-6)
        timestamp_scale = duration / source_duration
        chain = (
            f"setpts={timestamp_scale:.12f}*(PTS-STARTPTS),fps={fps},"
            f"trim=duration={duration:.3f},tpad=stop_mode=clone:stop_duration={duration:.3f},"
        )
    else:
        chain = ""
    return (
        f"[{index}:v]{chain}"
        f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,"
        f"pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        "drawbox=x=0:y=0:w=iw:h=45:color=black@0.72:t=fill,"
        f"drawtext=text='{ffmpeg_text(label)}':x=7:y=7:fontsize=14:fontcolor=white:"
        f"box=0[tile{index}]"
    )


def build_command(
    selected: list[dict[str, Any]],
    output: Path,
    *,
    columns: int,
    width: int,
    height: int,
    duration: float,
    fps: int,
    font_path: Path,
) -> list[str]:
    rows = math.ceil(len(selected) / columns)
    tile_width = width // columns
    tile_height = height // rows
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]
    filters: list[str] = []
    for index, item in enumerate(selected):
        if item["status"] == "available":
            command.extend(["-i", item["video_path"]])
        else:
            command.extend(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c=0x20252b:s={tile_width}x{tile_height}:r={fps}:d={duration:.3f}",
                ]
            )
        filters.append(
            tile_filter(
                index,
                item["label"],
                tile_width,
                tile_height,
                duration,
                fps,
                item["status"] == "available",
                float(item.get("source_fps", 30)),
                int(item.get("frame_count", 0)),
            )
        )

    layout = "|".join(
        f"{(index % columns) * tile_width}_{(index // columns) * tile_height}"
        for index in range(len(selected))
    )
    inputs = "".join(f"[tile{index}]" for index in range(len(selected)))
    filters.append(f"{inputs}xstack=inputs={len(selected)}:layout={layout}:fill=black,format=yuv420p[outv]")
    command.extend(
        [
            "-filter_complex_threads",
            "8",
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-an",
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1512)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--font-path",
        type=Path,
        default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found")
    if args.columns != 5 or len(TASKS) != 7 or args.width % args.columns:
        raise ValueError("The default showcase must be a 7x5 grid with divisible width")
    if args.height % len(TASKS):
        raise ValueError("Output height must be divisible by seven rows")
    if not args.font_path.is_file():
        raise FileNotFoundError(args.font_path)

    root = args.root.expanduser().resolve()
    selected: list[dict[str, Any]] = []
    for task in TASKS:
        for profile in PROFILES:
            raw_dir = root / task / profile / "raw"
            converted_dir = root / task / profile / "lerobot_v3"
            metadata_path, item = choose_episode(raw_dir, converted_dir)
            item.update(
                {
                    "task": task,
                    "profile": profile,
                    "label": f"{task} | {profile}"
                    + (
                        f" | seed {item['seed']} layout {item['layout_seed']}"
                        if metadata_path is not None
                        else " | PENDING"
                    ),
                }
            )
            selected.append(item)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    selection_path = output.with_suffix(".selection.json")
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": "roboassemblybench_fabrica_7tasks_domain_showcase_v1",
                "root": str(root),
                "camera": FRONT_CAMERA_KEY,
                "grid": {"columns": 5, "rows": 7, "width": args.width, "height": args.height},
                "duration_seconds": args.duration_seconds,
                "fps": args.fps,
                "episodes": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    command = build_command(
        selected,
        output,
        columns=args.columns,
        width=args.width,
        height=args.height,
        duration=args.duration_seconds,
        fps=args.fps,
        font_path=args.font_path.resolve(),
    )
    print(f"Available samples: {sum(item['status'] == 'available' for item in selected)}/35")
    print(f"Selection manifest: {selection_path}")
    subprocess.run(command, check=True)
    print(f"Created: {output}")


if __name__ == "__main__":
    main()
