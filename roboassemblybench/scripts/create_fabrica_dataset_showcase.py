from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_DATASET_DIR = Path('outputs/fabrica_plumbers_block_ur5e_right_base_prepare_2k_lerobot_v3')
DEFAULT_LAYOUT_SEEDS = [12, 94419, 44288, 25621]
DEFAULT_CAMERA_KEY = 'observation.images.front'


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'{path.suffix}.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def _quantile_indices(size: int, count: int) -> list[int]:
    if size <= 0 or count <= 0:
        return []
    if count >= size:
        return list(range(size))
    if count == 1:
        return [(size - 1) // 2]
    return [round(index * (size - 1) / (count - 1)) for index in range(count)]


def select_diverse_episodes(
    episodes: list[dict[str, Any]],
    *,
    layout_seeds: list[int],
    episodes_per_layout: int,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {int(seed): [] for seed in layout_seeds}
    for episode in episodes:
        layout_seed = int(episode['layout_seed'])
        if layout_seed in grouped:
            grouped[layout_seed].append(episode)

    selected: list[dict[str, Any]] = []
    for layout_seed in layout_seeds:
        candidates = sorted(
            grouped[int(layout_seed)],
            key=lambda item: (int(item['frame_count']), int(item['episode_index'])),
        )
        if len(candidates) < episodes_per_layout:
            raise ValueError(
                f'Layout seed {layout_seed} has {len(candidates)} episodes; ' f'{episodes_per_layout} are required.'
            )
        for quantile_rank, candidate_index in enumerate(_quantile_indices(len(candidates), episodes_per_layout)):
            candidate = dict(candidates[candidate_index])
            candidate['selection'] = {
                'layout_seed': int(layout_seed),
                'frame_count_quantile_rank': quantile_rank,
                'frame_count_quantile_count': episodes_per_layout,
            }
            selected.append(candidate)
    return selected


def _translation_mm(episode: dict[str, Any], group: str) -> tuple[float, float]:
    groups = (episode.get('domain_randomization') or {}).get('groups') or {}
    translation = (groups.get(group) or {}).get('translation') or [0.0, 0.0, 0.0]
    return float(translation[0]) * 1000.0, float(translation[1]) * 1000.0


def _resolve_video(episode: dict[str, Any], camera_key: str) -> tuple[Path, dict[str, Any]]:
    metadata_path = Path(str(episode['source_metadata'])).expanduser()
    metadata = _load_json(metadata_path)
    video_value = (metadata.get('videos') or {}).get(camera_key)
    if not video_value:
        raise KeyError(f'{camera_key!r} is missing from {metadata_path}.')
    video_path = Path(str(video_value)).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    return video_path, metadata


def _tile_filter(
    *,
    input_index: int,
    episode: dict[str, Any],
    metadata: dict[str, Any],
    tile_width: int,
    tile_height: int,
    target_duration: float,
    output_fps: int,
    font_path: Path,
) -> str:
    source_fps = float(metadata.get('fps', 30))
    frame_count = int(episode['frame_count'])
    source_duration = frame_count / source_fps
    timestamp_scale = target_duration / source_duration
    parts_x, parts_y = _translation_mm(episode, 'start_parts')
    base_x, base_y = _translation_mm(episode, 'assembly_base')
    line_one = (
        f"ep {int(episode['episode_index']):04d}  seed {int(episode['seed'])}  " f"layout {int(episode['layout_seed'])}"
    )
    line_two = f'parts {parts_x:+.1f} {parts_y:+.1f} mm  ' f'base {base_x:+.1f} {base_y:+.1f} mm'
    return (
        f'[{input_index}:v]'
        f'setpts={timestamp_scale:.12f}*(PTS-STARTPTS),'
        f'fps={output_fps},'
        f'trim=duration={target_duration:.3f},'
        f'scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,'
        f'pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2:color=black,'
        'drawbox=x=0:y=0:w=iw:h=54:color=black@0.62:t=fill,'
        f"drawtext=fontfile={font_path}:text='{line_one}':x=9:y=7:fontsize=17:fontcolor=white,"
        f"drawtext=fontfile={font_path}:text='{line_two}':x=9:y=29:fontsize=15:fontcolor=white"
        f'[tile{input_index}]'
    )


def build_ffmpeg_command(
    *,
    selected: list[dict[str, Any]],
    camera_key: str,
    output_path: Path,
    columns: int,
    output_width: int,
    output_height: int,
    duration_seconds: float,
    output_fps: int,
    font_path: Path,
    overwrite: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    if not selected:
        raise ValueError('At least one episode is required.')
    rows = math.ceil(len(selected) / columns)
    if len(selected) != rows * columns:
        raise ValueError('The selected episode count must fill the requested grid.')
    if output_width % columns or output_height % rows:
        raise ValueError('Output dimensions must be divisible by the grid dimensions.')

    tile_width = output_width // columns
    tile_height = output_height // rows
    command = ['ffmpeg', '-y' if overwrite else '-n', '-hide_banner', '-loglevel', 'warning']
    filters: list[str] = []
    resolved: list[dict[str, Any]] = []
    for input_index, episode in enumerate(selected):
        video_path, metadata = _resolve_video(episode, camera_key)
        command.extend(['-i', str(video_path)])
        filters.append(
            _tile_filter(
                input_index=input_index,
                episode=episode,
                metadata=metadata,
                tile_width=tile_width,
                tile_height=tile_height,
                target_duration=duration_seconds,
                output_fps=output_fps,
                font_path=font_path,
            )
        )
        resolved.append(
            {
                **episode,
                'video_path': str(video_path),
                'source_fps': int(metadata.get('fps', 30)),
            }
        )

    layout = '|'.join(
        f'{(index % columns) * tile_width}_{(index // columns) * tile_height}' for index in range(len(selected))
    )
    inputs = ''.join(f'[tile{index}]' for index in range(len(selected)))
    filters.append(f'{inputs}xstack=inputs={len(selected)}:layout={layout}:fill=black,format=yuv420p[outv]')
    command.extend(
        [
            '-filter_complex_threads',
            str(min(16, len(selected))),
            '-filter_complex',
            ';'.join(filters),
            '-map',
            '[outv]',
            '-an',
            '-c:v',
            'libx264',
            '-preset',
            'medium',
            '-crf',
            '22',
            '-movflags',
            '+faststart',
            '-t',
            f'{duration_seconds:.3f}',
            str(output_path),
        ]
    )
    return command, resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create a stratified MP4 mosaic from the Fabrica LeRobot v3 dataset.')
    parser.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--layout-seeds', type=int, nargs='+', default=DEFAULT_LAYOUT_SEEDS)
    parser.add_argument('--episodes-per-layout', type=int, default=4)
    parser.add_argument('--columns', type=int, default=4)
    parser.add_argument('--camera-key', default=DEFAULT_CAMERA_KEY)
    parser.add_argument('--duration-seconds', type=float, default=45.0)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--font-path', type=Path, default=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg was not found.')
    if args.episodes_per_layout <= 0 or args.columns <= 0 or args.fps <= 0:
        raise ValueError('Grid counts and fps must be positive.')
    if args.duration_seconds <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError('Duration and output dimensions must be positive.')
    if not args.font_path.is_file():
        raise FileNotFoundError(args.font_path)

    dataset_dir = args.dataset_dir.expanduser().resolve()
    conversion_path = dataset_dir / 'roboassemblybench_conversion_manifest.json'
    conversion = _load_json(conversion_path)
    episodes = conversion.get('episodes') or []
    if int(conversion.get('total_episodes', -1)) != len(episodes):
        raise ValueError('Conversion manifest episode count is inconsistent.')
    selected = select_diverse_episodes(
        episodes,
        layout_seeds=[int(seed) for seed in args.layout_seeds],
        episodes_per_layout=int(args.episodes_per_layout),
    )

    output_path = args.output
    if output_path is None:
        output_path = dataset_dir / 'showcase' / 'fabrica_2k_front_diversity_4x4.mp4'
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command, resolved = build_ffmpeg_command(
        selected=selected,
        camera_key=str(args.camera_key),
        output_path=output_path,
        columns=int(args.columns),
        output_width=int(args.width),
        output_height=int(args.height),
        duration_seconds=float(args.duration_seconds),
        output_fps=int(args.fps),
        font_path=args.font_path.expanduser().resolve(),
        overwrite=bool(args.overwrite),
    )

    selection_path = output_path.with_suffix('.selection.json')
    _write_json(
        selection_path,
        {
            'schema_version': 'roboassemblybench_dataset_showcase_v1',
            'dataset_dir': str(dataset_dir),
            'conversion_manifest': str(conversion_path),
            'source_total_episodes': len(episodes),
            'camera_key': str(args.camera_key),
            'grid': {
                'columns': int(args.columns),
                'rows': len(selected) // int(args.columns),
                'width': int(args.width),
                'height': int(args.height),
            },
            'duration_seconds': float(args.duration_seconds),
            'fps': int(args.fps),
            'episodes': resolved,
            'output': str(output_path),
        },
    )
    subprocess.run(command, check=True)
    print(f'Created {output_path}')
    print(f'Selection manifest: {selection_path}')


if __name__ == '__main__':
    main()
