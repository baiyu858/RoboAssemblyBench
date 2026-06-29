from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from toolkits.factory_dual_franka_assembly.render_fabrica_official_motion_isaac import (
    _build_env,
    _encode_mp4,
    _flush_world_for_capture,
    _to_uint8_rgba,
)
from toolkits.factory_dual_franka_assembly.render_fabrica_traj_replay_isaac import (
    DEFAULT_ASSET_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_OUTPUT,
    DEFAULT_ASSEMBLY_DIR,
    UNIT_SCALE,
    _add_all_replay_prims,
    _camera_pose,
    _load_fabrica_traj,
    _parse_vector3,
    _set_replay_prim,
)
from toolkits.factory_dual_franka_assembly.scene_builder import build_dual_franka_assembly_episode


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_ENV_OUTPUT = (
    REPO_ROOT
    / "outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.mp4"
)


def _hide_prim(stage, prim_path: str) -> bool:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    try:
        UsdGeom.Imageable(prim).MakeInvisible()
    except Exception:
        return False
    return True


def _candidate_task_prim_paths(task_cfg) -> list[str]:
    paths: list[str] = []

    for robot_spec in task_cfg.robot_metadata:
        name = str(robot_spec.get("name", ""))
        prim_path = str(robot_spec.get("prim_path", ""))
        prim_leaf = prim_path.rsplit("/", maxsplit=1)[-1] if prim_path else ""
        for token in (name, prim_leaf):
            if not token:
                continue
            paths.extend(
                [
                    f"/World/env_0/robots/{token}",
                    f"/World/env_0/{token}",
                    f"/World/{token}",
                    f"/{token}",
                ]
            )

    task_object_names = {
        str(metadata.get("name", ""))
        for metadata in task_cfg.object_metadata
        if str(metadata.get("name", "")).startswith("fabrica_plumbers_block_")
        or str(metadata.get("name", ""))
        in {
            "optical_board",
            "fabrica_fixture",
            "assembled_plumbers_preview",
        }
    }
    for object_name in sorted(name for name in task_object_names if name):
        paths.extend(
            [
                f"/World/env_0/objects/{object_name}",
                f"/World/env_0/{object_name}",
                f"/World/{object_name}",
                f"/{object_name}",
            ]
        )

    deduped = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _hide_task_replay_overlaps(stage, task_cfg) -> list[str]:
    hidden_paths = []
    for prim_path in _candidate_task_prim_paths(task_cfg):
        if _hide_prim(stage, prim_path):
            hidden_paths.append(prim_path)
    return hidden_paths


def render_fabrica_traj_replay_in_task_env(
    *,
    recipe: str,
    scene_profile: str,
    seed: int,
    log_dir: Path,
    assembly_dir: Path,
    asset_dir: Path,
    output_path: Path,
    frames_dir: Path,
    width: int,
    height: int,
    fps: int,
    stride: int,
    max_frames: int | None,
    camera_option: str,
    world_offset: str | tuple[float, float, float] | list[float],
    warmup_steps: int,
    hide_task_replay_overlaps: bool,
    headless: bool,
    webrtc: bool = False,
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

    env = _build_env(task_cfg, headless=headless, webrtc=webrtc)
    env.runner.render_interval = 0

    try:
        print("[render_fabrica_task_env] resetting task environment...", flush=True)
        env.reset()
        print("[render_fabrica_task_env] task environment reset complete.", flush=True)

        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World/replay")

        hidden_paths = _hide_task_replay_overlaps(stage, task_cfg) if hide_task_replay_overlaps else []
        replay_prims = _add_all_replay_prims(stage, assembly_dir=assembly_dir, asset_dir=asset_dir, log_dir=log_dir)

        world_offset_m = _parse_vector3(world_offset)
        camera_position, look_at = _camera_pose(camera_option, world_offset_m=world_offset_m)
        camera = rep.create.camera(position=camera_position, look_at=look_at)
        render_product = rep.create.render_product(camera, (width, height))
        annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
        annotator.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)

        traj_path = log_dir / "traj.npy"
        traj = _load_fabrica_traj(traj_path)
        render_indices = list(range(0, len(traj), max(stride, 1)))
        if render_indices[-1] != len(traj) - 1:
            render_indices.append(len(traj) - 1)
        if max_frames is not None:
            render_indices = render_indices[: max(max_frames, 1)]

        for _ in range(max(warmup_steps, 0)):
            _flush_world_for_capture(env)
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)

        captured_indices: list[int] = []
        for output_index, source_index in enumerate(render_indices):
            frame = traj[source_index]
            for prim in replay_prims:
                if prim.body_key not in frame:
                    continue
                _set_replay_prim(prim, frame[prim.body_key], world_offset_m=world_offset_m)

            _flush_world_for_capture(env)
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)
            imageio.imwrite(frames_dir / f"rgb_{output_index:05d}.png", _to_uint8_rgba(annotator.get_data()))
            captured_indices.append(source_index)

        print("[render_fabrica_task_env] waiting for replicator completion...", flush=True)
        rep.orchestrator.wait_until_complete()
        print("[render_fabrica_task_env] encoding mp4...", flush=True)
        png_paths = _encode_mp4(frames_dir=frames_dir, output_path=output_path, fps=fps)
        print("[render_fabrica_task_env] mp4 encoding complete.", flush=True)

        summary = {
            "mode": "official_fabrica_traj_replay_inside_roboassemblybench_task_env",
            "recipe": recipe,
            "scene_profile": scene_profile,
            "seed": seed,
            "assembly_name": assembly_dir.name,
            "arm": "ur5e",
            "gripper": "robotiq-85",
            "log_dir": str(log_dir),
            "traj_path": str(traj_path),
            "assembly_dir": str(assembly_dir),
            "asset_dir": str(asset_dir),
            "output_path": str(output_path),
            "frames_dir": str(frames_dir),
            "source_frame_count": int(len(traj)),
            "captured_frame_count": len(captured_indices),
            "written_png_count": len(png_paths),
            "captured_source_indices": captured_indices,
            "stride": stride,
            "fps": fps,
            "camera_width": width,
            "camera_height": height,
            "camera_option": camera_option,
            "camera_position": camera_position,
            "camera_look_at": look_at,
            "headless": bool(headless),
            "webrtc": bool(webrtc),
            "scene_asset_path": getattr(task_cfg, "scene_asset_path", ""),
            "scene_asset_fallback_path": getattr(task_cfg, "scene_asset_fallback_path", ""),
            "workspace_offset": getattr(task_cfg, "workspace_offset", []),
            "world_offset_m": world_offset_m.tolist(),
            "hidden_task_overlap_prim_paths": hidden_paths,
            "unit_scale_m_per_fabrica_unit": UNIT_SCALE,
            "limitations": [
                "This loads a RoboAssemblyBench task environment, then overlays Fabrica's official traj.npy body matrices.",
                "It is still a kinematic visual replay, not an Isaac PhysX contact simulation.",
                "It does not retarget the official motion through the RoboAssemblyBench UR5e articulation controller.",
            ],
        }
        output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    except BaseException as exc:
        print(f"[render_fabrica_task_env] aborting during render setup: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        print("[render_fabrica_task_env] closing task environment...", flush=True)
        try:
            env.close()
            print("[render_fabrica_task_env] task environment closed.", flush=True)
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render official Fabrica UR5e traj.npy replay inside a RoboAssemblyBench task environment."
    )
    parser.add_argument("--recipe", default="fabrica_plumbers_block_ur5e")
    parser.add_argument("--scene-profile", default="taoyuan_grscenes_tabletop")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--assembly-dir", type=Path, default=DEFAULT_ASSEMBLY_DIR)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_TASK_ENV_OUTPUT)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=DEFAULT_TASK_ENV_OUTPUT.with_name(DEFAULT_TASK_ENV_OUTPUT.stem + "_frames"),
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera-option", choices=("close", "front", "far"), default="close")
    parser.add_argument("--world-offset", default="0.47,0,1.012")
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--keep-task-replay-overlaps", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", action="store_false", dest="headless")
    parser.add_argument("--webrtc", action="store_true", help="Enable Isaac Sim WebRTC remote visualization.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        summary = render_fabrica_traj_replay_in_task_env(
            recipe=args.recipe,
            scene_profile=args.scene_profile,
            seed=args.seed,
            log_dir=args.log_dir,
            assembly_dir=args.assembly_dir,
            asset_dir=args.asset_dir,
            output_path=args.output,
            frames_dir=args.frames_dir,
            width=args.width,
            height=args.height,
            fps=args.fps,
            stride=args.stride,
            max_frames=args.max_frames,
            camera_option=args.camera_option,
            world_offset=args.world_offset,
            warmup_steps=args.warmup_steps,
            hide_task_replay_overlaps=not bool(args.keep_task_replay_overlaps),
            headless=bool(args.headless),
            webrtc=bool(args.webrtc),
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
