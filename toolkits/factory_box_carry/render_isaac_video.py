from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio

from toolkits.factory_box_carry.demo_policy import HumanoidBenchCarryDemoPolicy
from toolkits.factory_box_carry.generate_demos import _build_env
from toolkits.factory_box_carry.scene_builder import build_factory_box_carry_episode


def _default_paths(seed: int) -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent / "outputs" / "factory_box_carry"
    return (
        base_dir / f"episode_seed{seed:04d}_isaac.mp4",
        base_dir / f"episode_seed{seed:04d}_isaac_frames",
    )


def _camera_pose(task_cfg) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    box_x, box_y, _ = task_cfg.objects[0].position
    goal_x, goal_y, _ = task_cfg.goal_position
    mid_x = (box_x + goal_x) * 0.5
    mid_y = (box_y + goal_y) * 0.5
    span_x = abs(goal_x - box_x)
    span_y = abs(goal_y - box_y)
    span = max(span_x, span_y, 1.0)

    camera_position = (
        mid_x - 0.4,
        mid_y - max(4.2, 2.6 + span * 0.8),
        max(2.8, 2.2 + span * 0.45),
    )
    look_at = (mid_x, mid_y, 0.95)
    return camera_position, look_at


def _encode_mp4(frames_dir: Path, output_path: Path, fps: int) -> list[str]:
    png_paths = sorted(str(path) for path in frames_dir.rglob("*.png"))
    if not png_paths:
        raise RuntimeError(f"No PNG frames were written to {frames_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    try:
        for png_path in png_paths:
            writer.append_data(imageio.imread(png_path))
    finally:
        writer.close()
    return png_paths


def render_episode(seed: int, output_path: Path, frames_dir: Path, headless: bool, width: int, height: int, fps: int, stride: int):
    task_cfg = build_factory_box_carry_episode(seed=seed, episode_idx=0)
    frames_dir.parent.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    env = _build_env([task_cfg], headless=headless)
    env.runner.render_interval = 0
    policy = HumanoidBenchCarryDemoPolicy()

    task_name = None
    frame_count = 0
    capture_steps = []
    metrics = None
    writer = None
    summary = None

    try:
        env.reset()
        import omni.replicator.core as rep

        camera_position, look_at = _camera_pose(task_cfg)
        camera = rep.create.camera(position=camera_position, look_at=look_at)
        render_product = rep.create.render_product(camera, (width, height))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(frames_dir), rgb=True, frame_padding=5)
        writer.attach([render_product])

        rep.orchestrator.set_capture_on_play(False)
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)

        task_name = next(iter(env.runner.current_tasks.keys()))
        task = env.runner.current_tasks[task_name]

        for step_idx in range(task.config.max_steps):
            actions = policy.act(task)
            _, _, terminated, _, _ = env.step([actions])

            if step_idx % max(stride, 1) == 0:
                rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)
                frame_count += 1
                capture_steps.append(step_idx)

            if terminated[0]:
                metrics = task.calculate_metrics()
                rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)
                frame_count += 1
                capture_steps.append(step_idx)
                break

        rep.orchestrator.wait_until_complete()
        png_paths = _encode_mp4(frames_dir=frames_dir, output_path=output_path, fps=fps)

        summary = {
            "seed": seed,
            "task_name": task_name,
            "output_path": str(output_path),
            "frames_dir": str(frames_dir),
            "captured_frame_count": frame_count,
            "written_png_count": len(png_paths),
            "capture_steps": capture_steps,
            "metrics": metrics,
            "camera_width": width,
            "camera_height": height,
            "fps": fps,
            "stride": stride,
        }
        summary_path = output_path.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        if writer is not None:
            try:
                writer.detach()
            except Exception:
                pass
        env.close()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Render a true Isaac Sim 3D video for the factory_box_carry demo.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--frames-dir", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    default_output, default_frames_dir = _default_paths(args.seed)
    output_path = Path(args.output).resolve() if args.output else default_output
    frames_dir = Path(args.frames_dir).resolve() if args.frames_dir else default_frames_dir

    summary = render_episode(
        seed=args.seed,
        output_path=output_path,
        frames_dir=frames_dir,
        headless=args.headless,
        width=args.width,
        height=args.height,
        fps=args.fps,
        stride=args.stride,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
