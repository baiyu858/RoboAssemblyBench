from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _collect_bounds(episode: dict):
    xs = []
    ys = []
    for step in episode["steps"]:
        xs.append(step["box_position"][0])
        ys.append(step["box_position"][1])
        for robot_name in ("carrier_left", "carrier_right"):
            position = step["observations"][robot_name]["position"]
            xs.append(position[0])
            ys.append(position[1])

    goal = episode["metrics"]["goal_position"]
    xs.append(goal[0])
    ys.append(goal[1])

    margin = 1.0
    return (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)


def _world_to_px(x: float, y: float, bounds, width: int, height: int, pad: int = 40):
    min_x, max_x, min_y, max_y = bounds
    draw_w = width - 2 * pad
    draw_h = height - 2 * pad
    px = pad + (x - min_x) / max(max_x - min_x, 1e-6) * draw_w
    py = height - pad - (y - min_y) / max(max_y - min_y, 1e-6) * draw_h
    return px, py


def _draw_zone(draw: ImageDraw.ImageDraw, center, half_extent: float, bounds, width: int, height: int, color):
    cx, cy = _world_to_px(center[0], center[1], bounds, width, height)
    ex = (half_extent / max(bounds[1] - bounds[0], 1e-6)) * (width - 80)
    ey = (half_extent / max(bounds[3] - bounds[2], 1e-6)) * (height - 80)
    draw.rounded_rectangle((cx - ex, cy - ey, cx + ex, cy + ey), radius=14, outline=color, width=4)


def _draw_robot(draw: ImageDraw.ImageDraw, position, label: str, color, bounds, width: int, height: int):
    px, py = _world_to_px(position[0], position[1], bounds, width, height)
    r = 14
    draw.ellipse((px - r, py - r, px + r, py + r), fill=color, outline=(20, 20, 20), width=2)
    draw.text((px + 18, py - 10), label, fill=(30, 30, 30))


def _draw_box(draw: ImageDraw.ImageDraw, box_position, bounds, width: int, height: int):
    px, py = _world_to_px(box_position[0], box_position[1], bounds, width, height)
    draw.rounded_rectangle((px - 18, py - 18, px + 18, py + 18), radius=6, fill=(217, 127, 66), outline=(60, 40, 20), width=2)


def _draw_path(draw: ImageDraw.ImageDraw, positions, bounds, width: int, height: int, color):
    if len(positions) < 2:
        return
    points = [_world_to_px(x, y, bounds, width, height) for x, y in positions]
    draw.line(points, fill=color, width=3)


def render_episode_video(episode_path: Path, output_path: Path, fps: int, stride: int):
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    steps = episode["steps"]
    bounds = _collect_bounds(episode)
    width, height = 1280, 720
    font = ImageFont.load_default()

    pickup_center = steps[0]["box_position"]
    goal_center = episode["metrics"]["goal_position"]

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    left_trace = []
    right_trace = []
    box_trace = []

    try:
        for idx in range(0, len(steps), stride):
            step = steps[idx]
            left_pos = step["observations"]["carrier_left"]["position"]
            right_pos = step["observations"]["carrier_right"]["position"]
            box_pos = step["box_position"]

            left_trace.append((left_pos[0], left_pos[1]))
            right_trace.append((right_pos[0], right_pos[1]))
            box_trace.append((box_pos[0], box_pos[1]))

            image = Image.new("RGB", (width, height), (245, 242, 235))
            draw = ImageDraw.Draw(image)

            draw.rounded_rectangle((20, 20, width - 20, height - 20), radius=24, outline=(200, 190, 175), width=2)
            _draw_zone(draw, pickup_center, half_extent=0.45, bounds=bounds, width=width, height=height, color=(64, 140, 78))
            _draw_zone(draw, goal_center, half_extent=0.45, bounds=bounds, width=width, height=height, color=(66, 111, 196))

            _draw_path(draw, box_trace, bounds, width, height, color=(227, 154, 102))
            _draw_path(draw, left_trace, bounds, width, height, color=(219, 89, 89))
            _draw_path(draw, right_trace, bounds, width, height, color=(66, 127, 219))

            _draw_box(draw, box_pos, bounds, width, height)
            _draw_robot(draw, left_pos, "left", (219, 89, 89), bounds, width, height)
            _draw_robot(draw, right_pos, "right", (66, 127, 219), bounds, width, height)

            draw.text((36, 32), f"seed={episode['seed']}  step={idx}  phase={step['phase']}", fill=(30, 30, 30), font=font)
            draw.text((36, 58), "orange=box  red/blue=robots  green=pickup  blue=goal", fill=(60, 60, 60), font=font)

            writer.append_data(np.asarray(image))
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="Render a top-down MP4 preview for a factory_box_carry episode.")
    parser.add_argument("episode", type=str, help="Path to episode_XXXX.json")
    parser.add_argument("--output", type=str, default=None, help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    episode_path = Path(args.episode).resolve()
    output_path = Path(args.output).resolve() if args.output else episode_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    render_episode_video(episode_path=episode_path, output_path=output_path, fps=args.fps, stride=max(args.stride, 1))
    print(output_path)


if __name__ == "__main__":
    main()
