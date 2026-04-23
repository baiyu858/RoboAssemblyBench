from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from toolkits.factory_box_carry.demo_policy import HumanoidBenchCarryDemoPolicy
from toolkits.factory_box_carry.generate_demos import _build_env
from toolkits.factory_box_carry.render_isaac_video import _encode_mp4
from toolkits.factory_box_carry.scene_builder import build_factory_box_carry_episode


def _json_dump(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _replay_camera_pose(task_cfg) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    box_x, box_y, _ = task_cfg.objects[0].position
    goal_x, goal_y, _ = task_cfg.goal_position
    mid_x = (box_x + goal_x) * 0.5
    mid_y = (box_y + goal_y) * 0.5
    span_x = abs(goal_x - box_x)
    span_y = abs(goal_y - box_y)
    span = max(span_x, span_y, 1.0)

    # Replay videos should read clearly as demos, so use a closer 3/4 view than
    # the wider online-render camera.
    camera_position = (
        mid_x - 0.15,
        mid_y - max(2.55, 1.95 + span * 0.35),
        max(1.85, 1.55 + span * 0.18),
    )
    look_at = (mid_x + 0.2, mid_y, 1.0)
    return camera_position, look_at


def _capture_state(task, step_idx: int) -> dict:
    task._resolve_box()  # noqa: SLF001 - replay helper needs direct access to tracked object
    box_position, box_orientation = task._get_box_pose()  # noqa: SLF001

    robots = {}
    for robot_name in task.config.robot_names:
        robot = task.robots[robot_name]
        position, orientation = robot.get_pose()
        robots[robot_name] = {
            "position": np.asarray(position).tolist(),
            "orientation": np.asarray(orientation).tolist(),
            "joint_positions": np.asarray(robot.articulation.get_joint_positions()).tolist(),
        }

    return {
        "step": step_idx,
        "phase": task.phase,
        "robots": robots,
        "box_position": np.asarray(box_position).tolist(),
        "box_orientation": np.asarray(box_orientation).tolist(),
    }


def record_rollout(seed: int, stride: int, output_path: Path):
    env = _build_env([build_factory_box_carry_episode(seed=seed, episode_idx=0)], headless=True)
    policy = HumanoidBenchCarryDemoPolicy()

    frames = []
    metrics = None
    task_name = None
    payload = None

    try:
        env.reset()
        task_name = next(iter(env.runner.current_tasks.keys()))
        task = env.runner.current_tasks[task_name]

        frames.append(_capture_state(task, step_idx=0))

        for step_idx in range(task.config.max_steps):
            actions = policy.act(task)
            _, _, terminated, _, _ = env.step([actions])

            if (step_idx + 1) % max(stride, 1) == 0 or terminated[0]:
                frames.append(_capture_state(task, step_idx=step_idx + 1))

            if terminated[0]:
                metrics = task.calculate_metrics()
                break

        payload = {
            "seed": seed,
            "stride": stride,
            "task_name": task_name,
            "metrics": metrics,
            "frames": frames,
        }
        _json_dump(output_path, payload)
        print(json.dumps(payload, indent=2))
    finally:
        env.close()


def _maybe_zero_world_velocity(articulation):
    if hasattr(articulation, "set_world_velocity"):
        articulation.set_world_velocity(np.zeros(6))
    if hasattr(articulation, "set_linear_velocity"):
        articulation.set_linear_velocity(np.zeros(3))


def render_replay(
    replay_path: Path,
    output_path: Path,
    frames_dir: Path,
    headless: bool,
    width: int,
    height: int,
    fps: int,
):
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    seed = int(replay["seed"])
    task_cfg = build_factory_box_carry_episode(seed=seed, episode_idx=0)

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    env = _build_env([task_cfg], headless=headless)
    env.runner.render_interval = 0
    summary = None

    try:
        env.reset()

        import omni.replicator.core as rep

        camera_position, look_at = _replay_camera_pose(task_cfg)
        camera = rep.create.camera(position=camera_position, look_at=look_at)
        render_product = rep.create.render_product(camera, (width, height))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(frames_dir), rgb=True, frame_padding=5)
        writer.attach([render_product])

        try:
            rep.orchestrator.set_capture_on_play(False)
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)

            task_name = next(iter(env.runner.current_tasks.keys()))
            task = env.runner.current_tasks[task_name]
            task._resolve_box()  # noqa: SLF001

            for frame in replay["frames"]:
                for robot_name, robot_state in frame["robots"].items():
                    robot = task.robots[robot_name]
                    robot.articulation.set_pose(
                        np.asarray(robot_state["position"], dtype=float),
                        np.asarray(robot_state["orientation"], dtype=float),
                    )
                    _maybe_zero_world_velocity(robot.articulation)
                    robot.articulation.set_joint_velocities(np.zeros(len(robot.articulation.dof_names), dtype=float))
                    robot.articulation.set_joint_positions(np.asarray(robot_state["joint_positions"], dtype=float))

                task._box.set_linear_velocity(np.zeros(3))  # noqa: SLF001
                task._box.set_pose(  # noqa: SLF001
                    np.asarray(frame["box_position"], dtype=float),
                    np.asarray(frame["box_orientation"], dtype=float),
                )
                rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)

            rep.orchestrator.wait_until_complete()
            png_paths = _encode_mp4(frames_dir=frames_dir, output_path=output_path, fps=fps)

            summary = {
                "seed": seed,
                "replay_path": str(replay_path),
                "output_path": str(output_path),
                "frames_dir": str(frames_dir),
                "written_png_count": len(png_paths),
                "camera_width": width,
                "camera_height": height,
                "fps": fps,
                "metrics": replay.get("metrics"),
            }
            _json_dump(output_path.with_suffix(".json"), summary)
            print(json.dumps(summary, indent=2))
        finally:
            writer.detach()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Record a successful rollout and replay it as a 3D Isaac video.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--seed", type=int, default=0)
    record_parser.add_argument("--stride", type=int, default=10)
    record_parser.add_argument("--output", type=str, required=True)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--replay", type=str, required=True)
    render_parser.add_argument("--output", type=str, required=True)
    render_parser.add_argument("--frames-dir", type=str, required=True)
    render_parser.add_argument("--headless", action="store_true")
    render_parser.add_argument("--width", type=int, default=320)
    render_parser.add_argument("--height", type=int, default=180)
    render_parser.add_argument("--fps", type=int, default=10)

    args = parser.parse_args()

    if args.mode == "record":
        record_rollout(seed=args.seed, stride=max(args.stride, 1), output_path=Path(args.output).resolve())
        return

    render_replay(
        replay_path=Path(args.replay).resolve(),
        output_path=Path(args.output).resolve(),
        frames_dir=Path(args.frames_dir).resolve(),
        headless=args.headless,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
