from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio

from toolkits.factory_dual_franka_assembly.render_fabrica_official_motion_isaac import (
    _build_env,
    _camera_pose,
    _encode_mp4,
    _flush_world_for_capture,
    _to_uint8_rgba,
)
from toolkits.factory_dual_franka_assembly.scene_builder import build_dual_franka_assembly_episode


REPO_ROOT = Path(__file__).resolve().parents[2]


def _preview_camera_pose(
    task_cfg,
    *,
    option: str,
    object_prefix: str,
    include_robots: bool,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not include_robots:
        return _camera_pose(task_cfg, option=option, object_prefix=object_prefix)

    import numpy as np

    positions = [
        np.asarray(metadata["sampled_position"], dtype=float)
        for metadata in task_cfg.object_metadata
        if str(metadata.get("name", "")).startswith(f"{object_prefix}_")
        or metadata.get("name") in {"optical_board", "assembled_manifold_preview", "assembled_plumbers_preview", "fabrica_fixture"}
    ]
    positions.extend(
        np.asarray(metadata["position"], dtype=float)
        for metadata in task_cfg.robot_metadata
        if metadata.get("position") is not None
    )
    center = np.mean(positions, axis=0) if positions else np.asarray([0.5, 0.0, 1.05])
    if option == "front":
        position = center + np.asarray([2.50, -2.80, 1.80], dtype=float)
    elif option == "right":
        position = center + np.asarray([2.30, 2.35, 1.65], dtype=float)
    elif option == "official_like":
        position = center + np.asarray([2.55, -2.85, 1.95], dtype=float)
    else:
        raise ValueError(f"Unsupported camera option: {option}")
    look_at = center + np.asarray([0.0, 0.02, 0.10], dtype=float)
    return tuple(position.tolist()), tuple(look_at.tolist())


def render_task_scene_preview(
    *,
    recipe: str,
    scene_profile: str,
    seed: int,
    output_path: Path,
    frames_dir: Path,
    object_prefix: str,
    camera_option: str,
    width: int,
    height: int,
    fps: int,
    frame_count: int,
    warmup_steps: int,
    include_robots_in_camera: bool,
    headless: bool,
) -> dict:
    task_cfg = build_dual_franka_assembly_episode(
        recipe=recipe,
        seed=seed,
        episode_idx=0,
        scene_profile=scene_profile,
    )

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    env = _build_env(task_cfg, headless=headless)
    env.runner.render_interval = 0

    try:
        env.reset()

        import omni.replicator.core as rep

        camera_position, look_at = _preview_camera_pose(
            task_cfg,
            option=camera_option,
            object_prefix=object_prefix,
            include_robots=include_robots_in_camera,
        )
        camera = rep.create.camera(position=camera_position, look_at=look_at)
        render_product = rep.create.render_product(camera, (width, height))
        annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
        annotator.attach([render_product])

        rep.orchestrator.set_capture_on_play(False)
        for _ in range(max(warmup_steps, 0)):
            _flush_world_for_capture(env)
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)

        written_frames = []
        for frame_index in range(max(frame_count, 1)):
            _flush_world_for_capture(env)
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)
            frame_rgba = _to_uint8_rgba(annotator.get_data())
            frame_path = frames_dir / f"rgb_{frame_index:05d}.png"
            imageio.imwrite(frame_path, frame_rgba)
            written_frames.append(str(frame_path))

        rep.orchestrator.wait_until_complete()
        png_paths = _encode_mp4(frames_dir=frames_dir, output_path=output_path, fps=fps)

        summary = {
            "mode": "isaacsim_task_scene_preview",
            "recipe": recipe,
            "scene_profile": scene_profile,
            "seed": seed,
            "output_path": str(output_path),
            "frames_dir": str(frames_dir),
            "captured_frame_count": len(written_frames),
            "written_png_count": len(png_paths),
            "fps": fps,
            "camera_width": width,
            "camera_height": height,
            "camera_option": camera_option,
            "camera_position": camera_position,
            "camera_look_at": look_at,
            "include_robots_in_camera": include_robots_in_camera,
            "object_prefix": object_prefix,
            "limitations": [
                "This is a scene preview MP4 for the requested Isaac Sim task.",
                "It does not replay Fabrica Franka joints on UR5e and does not perform UR5e retargeted assembly motion.",
            ],
        }
        output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        try:
            env.close()
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="fabrica_plumbers_block_ur5e")
    parser.add_argument("--scene-profile", default="taoyuan_grscenes_tabletop")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/fabrica_plumbers_block_ur5e/plumbers_block_ur5e_scene_preview.mp4",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=REPO_ROOT / "outputs/fabrica_plumbers_block_ur5e/plumbers_block_ur5e_scene_preview_frames",
    )
    parser.add_argument("--object-prefix", default="fabrica_plumbers_block")
    parser.add_argument("--camera-option", choices=("front", "right", "official_like"), default="official_like")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-count", type=int, default=120)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--include-robots-in-camera", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    summary = render_task_scene_preview(
        recipe=args.recipe,
        scene_profile=args.scene_profile,
        seed=args.seed,
        output_path=args.output,
        frames_dir=args.frames_dir,
        object_prefix=args.object_prefix,
        camera_option=args.camera_option,
        width=args.width,
        height=args.height,
        fps=args.fps,
        frame_count=args.frame_count,
        warmup_steps=args.warmup_steps,
        include_robots_in_camera=bool(args.include_robots_in_camera),
        headless=bool(args.headless),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
