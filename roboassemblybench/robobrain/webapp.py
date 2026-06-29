from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from roboassemblybench.core.paths import BENCHMARK_ROOT
from roboassemblybench.core.scene_profiles import DEFAULT_SCENE_PROFILE
from roboassemblybench.core.task_registry import load_task_recipe
from roboassemblybench.robobrain.executor import RoboBrainRunConfig, RoboBrainRunResult, RoboBrainRunner
from roboassemblybench.robobrain.inventory import RoboAssemblyInventory
from roboassemblybench.robobrain.manual_demo import MANUAL_DEMO_TEMPLATE, write_manual_demo_bundle

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

    FASTAPI_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only in minimal environments.
    FastAPI = None
    HTTPException = None
    Request = None
    FileResponse = None
    HTMLResponse = None
    JSONResponse = None
    PlainTextResponse = None
    FASTAPI_IMPORT_ERROR = exc


DEFAULT_TEMPLATE = 'fabrica_plumbers_block_ur5e_right_base_prepare'
PUBLIC_TEMPLATE_ID = 'ur5e_assembly_template'
PUBLIC_TEMPLATE_NAME = 'UR5e assembly Template'
PUBLIC_GENERATED_TASK_NAME = 'fabrica_plumbers_block_ur5e_assembly'
DEFAULT_OUTPUT_DIR = BENCHMARK_ROOT / 'outputs' / 'robobrain_agent_app'
DEFAULT_MANUAL_THINKING_DELAY = 3.0
MANUAL_THINKING_FINISHED_PAUSE = 0.0
MANUAL_THINKING_PRINT_PAUSE = 0.0
TYPEWRITER_INTERVAL_SECONDS = 0.012
TYPEWRITER_BUFFER_SECONDS = 0.12
REPO_ROOT = BENCHMARK_ROOT.parent.resolve()
SIMULATION_SCRIPT = REPO_ROOT / 'roboassemblybench' / 'scripts' / 'render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh'
SIMULATION_OUTPUT = REPO_ROOT / 'outputs' / 'fabrica_official_isaacsim' / 'plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.mp4'
RECORDINGS_DIR = DEFAULT_OUTPUT_DIR / 'recordings'
RECORDING_UPLOAD_SUFFIXES = {
    'video/mp4': '.mp4',
    'video/webm': '.webm',
    'video/quicktime': '.mov',
}
ASSET_MODEL_SUFFIXES = {'.usd', '.usda', '.usdc', '.urdf', '.obj', '.stl', '.fbx', '.glb'}
ASSET_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}
ASSET_LIBRARY_LIMITS = {'robots': 500, 'scenes': 2200, 'objects': 2200}
_EXECUTOR = ThreadPoolExecutor(max_workers=max(int(os.environ.get('ROBOBRAIN_AGENT_WORKERS', '1')), 1))
_JOBS: dict[str, 'AgentJob'] = {}
_JOBS_LOCK = threading.Lock()
_SIMULATION_JOBS: dict[str, 'SimulationJob'] = {}
_SIMULATION_JOBS_LOCK = threading.Lock()


def _now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S')


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _as_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(result, minimum)
    return result


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_scene_profile(value: Any) -> str | None:
    if value is None:
        return DEFAULT_SCENE_PROFILE
    text = str(value).strip()
    if text.lower() in {'', 'none', 'raw'}:
        return None
    return text


def _jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


@dataclass
class AgentRunRequest:
    task: str
    scene_profile: str | None = DEFAULT_SCENE_PROFILE
    template: str | None = DEFAULT_TEMPLATE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    model: str | None = None
    temperature: float = 0.2
    mock_llm: bool = False
    manual_demo: bool = False
    manual_thinking_delay: float = DEFAULT_MANUAL_THINKING_DELAY
    run_simulation: bool = False
    export_lerobot: bool = False
    num_demos: int = 1
    start_seed: int = 0
    max_trials: int = 1
    max_retries: int = 2
    headless: bool = True
    record_live_video: bool = True
    keep_video_frames: bool = False
    live_video_fps: int = 30
    live_video_frame_stride: int = 1
    runtime_robochecker: bool = True
    runtime_replanning: bool = True
    max_runtime_replans: int = 1
    runtime_checker_stride: int = 8
    runtime_capture_rgb: bool = True
    runtime_rgb_frame_stride: int = 24
    perception_grounding: bool = True
    perception_visual_backend: str = 'local'
    perception_labels: list[str] = field(default_factory=list)
    lerobot_video_mode: str = 'live_rollout'
    lerobot_fps: int = 10
    lerobot_include_failures: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> 'AgentRunRequest':
        task = str(payload.get('task') or payload.get('task_instruction') or '').strip()
        mock_default = not bool(os.environ.get('OPENAI_API_KEY'))
        labels = payload.get('perception_labels') or payload.get('perception_label') or []
        if isinstance(labels, str):
            labels = [item.strip() for item in labels.split(',') if item.strip()]
        run_simulation = _as_bool(payload.get('run_simulation'), _as_bool(payload.get('execute_demo'), False))
        return cls(
            task=task,
            scene_profile=_normalize_scene_profile(payload.get('scene_profile', DEFAULT_SCENE_PROFILE)),
            template=_template_internal_id(payload.get('template') or DEFAULT_TEMPLATE),
            output_dir=Path(payload.get('output_dir') or DEFAULT_OUTPUT_DIR).expanduser(),
            model=str(payload.get('model') or '').strip() or None,
            temperature=_as_float(payload.get('temperature'), 0.2),
            mock_llm=_as_bool(payload.get('mock_llm'), mock_default),
            manual_demo=_as_bool(payload.get('manual_demo'), False),
            manual_thinking_delay=max(_as_float(payload.get('manual_thinking_delay'), DEFAULT_MANUAL_THINKING_DELAY), 0.0),
            run_simulation=run_simulation,
            export_lerobot=_as_bool(payload.get('export_lerobot'), False),
            num_demos=_as_int(payload.get('num_demos'), 1, minimum=1),
            start_seed=_as_int(payload.get('start_seed'), 0, minimum=0),
            max_trials=_as_int(payload.get('max_trials'), 1, minimum=1),
            max_retries=_as_int(payload.get('max_retries'), 2, minimum=0),
            headless=_as_bool(payload.get('headless'), True),
            record_live_video=_as_bool(payload.get('record_live_video'), True),
            keep_video_frames=_as_bool(payload.get('keep_video_frames'), False),
            live_video_fps=_as_int(payload.get('live_video_fps'), 30, minimum=1),
            live_video_frame_stride=_as_int(payload.get('live_video_frame_stride'), 1, minimum=1),
            runtime_robochecker=_as_bool(payload.get('runtime_robochecker'), True),
            runtime_replanning=_as_bool(payload.get('runtime_replanning'), True),
            max_runtime_replans=_as_int(payload.get('max_runtime_replans'), 1, minimum=0),
            runtime_checker_stride=_as_int(payload.get('runtime_checker_stride'), 8, minimum=1),
            runtime_capture_rgb=_as_bool(payload.get('runtime_capture_rgb'), True),
            runtime_rgb_frame_stride=_as_int(payload.get('runtime_rgb_frame_stride'), 24, minimum=1),
            perception_grounding=_as_bool(payload.get('perception_grounding'), True),
            perception_visual_backend=str(payload.get('perception_visual_backend') or 'local'),
            perception_labels=[str(item) for item in labels],
            lerobot_video_mode=str(payload.get('lerobot_video_mode') or 'live_rollout'),
            lerobot_fps=_as_int(payload.get('lerobot_fps'), 10, minimum=1),
            lerobot_include_failures=_as_bool(payload.get('lerobot_include_failures'), False),
        )

    def to_runner_config(self, event_callback) -> RoboBrainRunConfig:
        return RoboBrainRunConfig(
            scene_profile=self.scene_profile,
            output_dir=self.output_dir,
            model=self.model,
            temperature=self.temperature,
            selected_template=self.template,
            max_retries=self.max_retries,
            mock_llm=self.mock_llm,
            plan_only=not self.run_simulation,
            num_demos=self.num_demos,
            start_seed=self.start_seed,
            max_trials=self.max_trials,
            headless=self.headless,
            record_live_video=self.record_live_video,
            live_video_fps=self.live_video_fps,
            live_video_frame_stride=self.live_video_frame_stride,
            keep_video_frames=self.keep_video_frames,
            runtime_robochecker=self.runtime_robochecker,
            runtime_replanning=self.runtime_replanning,
            max_runtime_replans=self.max_runtime_replans,
            runtime_checker_stride=self.runtime_checker_stride,
            runtime_capture_rgb=self.runtime_capture_rgb,
            runtime_rgb_frame_stride=self.runtime_rgb_frame_stride,
            perception_grounding=self.perception_grounding,
            perception_visual_backend=self.perception_visual_backend,
            perception_visual_labels=list(self.perception_labels),
            local_replanning=True,
            event_callback=event_callback,
            capture_demo_output=self.run_simulation,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload['template'] = _template_public_id(self.template)
        return _jsonable(payload)


@dataclass
class AgentJob:
    id: str
    request: AgentRunRequest
    status: str = 'queued'
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_event(self, event_type: str, payload: dict[str, Any] | None = None, event_time: str | None = None):
        with self.lock:
            entry = {
                'index': len(self.events),
                'time': event_time or _now(),
                'type': event_type,
                'payload': _jsonable(payload or {}),
            }
            self.events.append(entry)
            self.updated_at = entry['time']

    def add_runner_event(self, event: dict[str, Any]):
        self.add_event(
            str(event.get('type') or 'runner_event'),
            event.get('payload') or {},
            event_time=event.get('time'),
        )

    def to_dict(self, *, after: int | None = None) -> dict[str, Any]:
        with self.lock:
            events = self.events if after is None else self.events[max(after, 0) :]
            return {
                'id': self.id,
                'status': self.status,
                'created_at': self.created_at,
                'updated_at': self.updated_at,
                'request': self.request.to_dict(),
                'events': list(events),
                'event_count': len(self.events),
                'result': self.result,
                'error': self.error,
            }


@dataclass
class SimulationJob:
    id: str
    script: Path = SIMULATION_SCRIPT
    status: str = 'queued'
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_event(self, event_type: str, payload: dict[str, Any] | None = None):
        with self.lock:
            entry = {
                'index': len(self.events),
                'time': _now(),
                'type': event_type,
                'payload': _jsonable(payload or {}),
            }
            self.events.append(entry)
            self.updated_at = entry['time']

    def to_dict(self, *, after: int | None = None) -> dict[str, Any]:
        with self.lock:
            events = self.events if after is None else self.events[max(after, 0) :]
            return {
                'id': self.id,
                'status': self.status,
                'created_at': self.created_at,
                'updated_at': self.updated_at,
                'script': str(self.script),
                'events': list(events),
                'event_count': len(self.events),
                'result': self.result,
                'error': self.error,
            }


def _scene_profiles() -> list[str]:
    profiles = ['raw']
    profile_dir = BENCHMARK_ROOT / 'scenes' / 'profiles'
    profiles.extend(sorted(path.stem for path in profile_dir.glob('*.yaml')))
    return profiles


def _template_display_name(name: str | None) -> str:
    if name == DEFAULT_TEMPLATE:
        return PUBLIC_TEMPLATE_NAME
    return str(name or '')


def _template_public_id(name: str | None) -> str:
    if name == DEFAULT_TEMPLATE:
        return PUBLIC_TEMPLATE_ID
    return str(name or '')


def _template_internal_id(name: str | None) -> str:
    value = str(name or '').strip()
    if value in {PUBLIC_TEMPLATE_ID, PUBLIC_TEMPLATE_NAME, PUBLIC_GENERATED_TASK_NAME}:
        return DEFAULT_TEMPLATE
    return value or DEFAULT_TEMPLATE


def _asset_catalog(limit: int = 5000) -> list[dict[str, Any]]:
    assets_root = BENCHMARK_ROOT / 'assets'
    suffixes = ASSET_MODEL_SUFFIXES | ASSET_IMAGE_SUFFIXES | {'.zip', '.yaml', '.json', '.md'}
    rows = []
    if not assets_root.exists():
        return rows
    for path in sorted(item for item in assets_root.rglob('*') if item.is_file() and item.suffix.lower() in suffixes):
        rows.append(
            {
                'name': path.name,
                'path': str(path.relative_to(BENCHMARK_ROOT)),
                'suffix': path.suffix.lower(),
                'size_bytes': path.stat().st_size,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _asset_title(path: Path) -> str:
    stem = path.name
    for suffix in ('.usd.png', '.usda.png', '.usdc.png', '.urdf.png', '.obj.png', '.stl.png', '.fbx.png', '.glb.png'):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        stem = path.stem
    stem = re.sub(r'\.(thumb|auto)$', '', stem, flags=re.IGNORECASE)
    return re.sub(r'[_\-.]+', ' ', stem).strip().title() or path.name


def _asset_relative(path: Path) -> str:
    try:
        return str(path.relative_to(BENCHMARK_ROOT))
    except ValueError:
        return str(path)


def _asset_parts_text(path: Path) -> str:
    try:
        rel = path.relative_to(BENCHMARK_ROOT / 'assets')
    except ValueError:
        rel = path
    return '/'.join(rel.parts).lower()


def _asset_kind_for_path(path: Path) -> str:
    text = _asset_parts_text(path)
    name = path.name.lower()
    if any(token in text for token in ('/robots/', '/robot/', 'universalrobots', 'frankarobotics')):
        return 'robots'
    if any(token in name for token in ('ur5e', 'franka', 'panda', 'kuka', 'iiwa', 'robotiq', 'gripper')):
        return 'robots'
    if any(token in text for token in ('/environments/', '/scenes/', 'warehouse', 'factory', 'workcell', 'tabletop')):
        return 'scenes'
    if any(token in name for token in ('table', 'shelf', 'rack', 'forklift', 'vehicle', 'pallet', 'platform', 'crate', 'cart')):
        return 'scenes'
    return 'objects'


def _asset_tags(path: Path, kind: str) -> list[str]:
    tags = [kind[:-1] if kind.endswith('s') else kind]
    rel_parts = Path(_asset_relative(path)).parts
    if len(rel_parts) > 2:
        tags.append(rel_parts[2].replace('_', ' '))
    suffix = path.suffix.lower().lstrip('.')
    if suffix:
        tags.append(suffix)
    if 'fabrica' in _asset_parts_text(path):
        tags.append('Fabrica')
    if len(tags) < 3 and 'IsaacUSD' in rel_parts:
        tags.append('IsaacUSD')
    return list(dict.fromkeys(tags[:4]))


def _asset_sort_priority(path: Path) -> int:
    text = _asset_parts_text(path)
    name = path.name.lower()
    priorities = (
        ('ur5e', 0),
        ('ur5e_robotiq', 1),
        ('franka', 2),
        ('panda', 3),
        ('kuka', 4),
        ('iiwa', 5),
        ('robotiq', 6),
        ('gripper', 7),
        ('plumbers_block', 10),
        ('fixture', 11),
        ('optical_board', 12),
        ('workcell', 20),
        ('warehouse', 21),
        ('table', 22),
    )
    for token, priority in priorities:
        if token in name:
            return priority
    for token, priority in priorities:
        if token in text:
            return priority + 30
    return 100


def _asset_description(path: Path, kind: str, preview_only: bool = False) -> str:
    source = ' / '.join(Path(_asset_relative(path)).parts[2:5])
    if kind == 'robots':
        prefix = '机器人库资产'
    elif kind == 'scenes':
        prefix = '场景库资产'
    else:
        prefix = '交互物体库资产'
    if preview_only:
        return f'{prefix}，来自 assets 目录的真实渲染预览。'
    return f'{prefix}，来源：{source or "roboassemblybench/assets"}。'


def _is_texture_image(path: Path) -> bool:
    text = _asset_parts_text(path)
    return '/textures/' in text or '/materials/textures/' in text


def _is_rendered_asset_preview(path: Path) -> bool:
    text = _asset_parts_text(path)
    name = path.name.lower()
    if '_generated_previews/' in text:
        return True
    if '.thumbs/256x256/' in text and not _is_texture_image(path):
        return any(name.endswith(f'{suffix}.png') for suffix in ASSET_MODEL_SUFFIXES)
    return any(name.endswith(f'{suffix}.png') for suffix in ASSET_MODEL_SUFFIXES)


def _preview_rank(path: Path) -> tuple[int, str]:
    text = _asset_parts_text(path)
    rank = 40
    if '_generated_previews/' in text:
        rank = 0
    elif '.thumbs/256x256/' in text and not _is_texture_image(path):
        rank = 10
    elif _is_rendered_asset_preview(path):
        rank = 20
    elif _is_texture_image(path):
        rank = 90
    return rank, str(path).lower()


def _preview_key_variants(path: Path) -> set[str]:
    name = path.name.lower()
    keys = {name, path.stem.lower()}
    if name.endswith('.png'):
        base = name[:-4]
        keys.add(base)
        keys.add(Path(base).stem.lower())
        for model_suffix in ('.thumb.usd', '.thumb.usda', '.thumb.usdc'):
            if base.endswith(model_suffix):
                clean = base.replace('.thumb', '')
                keys.add(clean)
                keys.add(Path(clean).stem.lower())
    return {key for key in keys if key}


def _model_key_variants(path: Path) -> set[str]:
    name = path.name.lower()
    keys = {name, path.stem.lower(), f'{name}.png'}
    return {key for key in keys if key}


def _is_library_model_file(path: Path) -> bool:
    text = _asset_parts_text(path)
    name = path.name.lower()
    if path.suffix.lower() not in ASSET_MODEL_SUFFIXES:
        return False
    if '/.thumbs/' in text or '/textures/' in text:
        return False
    if name in {'materials.usd', 'material.usd'} or name.startswith('materials.'):
        return False
    return True


@lru_cache(maxsize=1)
def _asset_preview_sources() -> tuple[tuple[Path, ...], dict[str, Path]]:
    assets_root = BENCHMARK_ROOT / 'assets'
    if not assets_root.exists():
        return (), {}
    previews = tuple(
        sorted(
            (
                path.resolve()
                for path in assets_root.rglob('*')
                if path.is_file() and path.suffix.lower() in ASSET_IMAGE_SUFFIXES and _is_rendered_asset_preview(path)
            ),
            key=_preview_rank,
        )
    )
    index: dict[str, Path] = {}
    for preview in previews:
        for key in _preview_key_variants(preview):
            index.setdefault(key, preview)
    return previews, index


def _preview_for_asset(path: Path, preview_index: dict[str, Path]) -> Path | None:
    for key in _model_key_variants(path):
        preview = preview_index.get(key)
        if preview is not None:
            return preview
    thumb_dir = path.parent / '.thumbs' / '256x256'
    candidates = [
        thumb_dir / f'{path.name}.png',
        thumb_dir / f'{path.stem}.usd.png',
        path.with_name(f'{path.name}.png'),
        path.with_suffix(f'{path.suffix}.png'),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in ASSET_IMAGE_SUFFIXES:
            return candidate.resolve()
    return None


def _asset_card(asset_path: Path, preview_path: Path, kind: str, preview_only: bool = False) -> dict[str, Any]:
    source_path = preview_path if preview_only else asset_path
    return {
        'name': _asset_title(source_path),
        'description': _asset_description(source_path, kind, preview_only=preview_only),
        'asset_path': _asset_relative(source_path),
        'preview': str(preview_path.resolve()),
        'tags': _asset_tags(source_path, kind),
    }


def _first_existing_asset_path(*relative_paths: str) -> str | None:
    for relative_path in relative_paths:
        path = BENCHMARK_ROOT / 'assets' / relative_path
        if path.exists() and path.is_file():
            return _asset_relative(path.resolve())
    return None


def _robot_family(card: dict[str, Any]) -> str | None:
    name = str(card.get('name', '')).lower()
    asset_path = str(card.get('asset_path', '')).lower()
    if '/robots/robotiq/' in asset_path:
        return None
    if 'ur5e' in name or '/universalrobots/ur5e/' in asset_path or '/ur5e/' in asset_path:
        return 'ur5e'
    if 'franka' in name or 'panda' in name or '/frankarobotics/frankapanda/' in asset_path:
        return 'franka'
    if 'kuka' in name or 'iiwa' in name or '/kuka/' in asset_path:
        return 'kuka'
    return None


def _robot_representative_card(family: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    base = dict(candidates[0])
    if family == 'ur5e':
        base.update(
            {
                'name': 'UR5e + Robotiq 2F-85',
                'description': '机器人库资产，UR5e 协作臂与 Robotiq 2F-85 夹爪，用于双臂装配任务。',
                'asset_path': _first_existing_asset_path(
                    'Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001/assets/ur5e_robotiq_2f85_task.usda',
                    'Fabrica/fabrica_ur5e_d405_plumbers_block_minimal_workcell_fullbundle_sdf001_v1/assets/imported_assets/ur5e_d405_asset_bundle_2026-06-15/extracted/source/isaaclab_tasks/isaaclab_tasks/direct/factory_ur5e_robotiq/assets/ur5e_robotiq_2f85_fixed_camera.usda',
                )
                or base.get('asset_path', ''),
                'tags': ['robot', 'UR5e', 'Robotiq', 'Fabrica'],
            }
        )
    elif family == 'franka':
        base.update(
            {
                'name': 'Franka Panda',
                'description': '机器人库资产，Franka Panda 协作臂，用于通用桌面装配基线。',
                'tags': ['robot', 'Franka', 'Panda'],
            }
        )
    elif family == 'kuka':
        base.update(
            {
                'name': 'KUKA iiwa',
                'description': '机器人库资产，KUKA iiwa 工业协作臂。',
                'tags': ['robot', 'KUKA', 'iiwa'],
            }
        )
    return base


def _collapse_robot_library(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        family = _robot_family(card)
        if family:
            grouped.setdefault(family, []).append(card)
    if not grouped:
        return cards[:1]
    order = ['ur5e', 'franka', 'kuka']
    return [_robot_representative_card(family, grouped[family]) for family in order if family in grouped]


@lru_cache(maxsize=1)
def _asset_library_cache() -> dict[str, list[dict[str, Any]]]:
    assets_root = BENCHMARK_ROOT / 'assets'
    libraries: dict[str, list[dict[str, Any]]] = {'robots': [], 'scenes': [], 'objects': []}
    if not assets_root.exists():
        return libraries

    previews, preview_index = _asset_preview_sources()
    used_previews: set[Path] = set()
    seen_assets: set[str] = set()

    def add_card(asset_path: Path, preview_path: Path, preview_only: bool = False) -> None:
        kind = _asset_kind_for_path(asset_path)
        if len(libraries[kind]) >= ASSET_LIBRARY_LIMITS[kind]:
            return
        key = _asset_relative(asset_path)
        if key in seen_assets:
            return
        libraries[kind].append(_asset_card(asset_path, preview_path, kind, preview_only=preview_only))
        seen_assets.add(key)
        used_previews.add(preview_path.resolve())

    model_files = sorted(
        (
            path.resolve()
            for path in assets_root.rglob('*')
            if path.is_file() and _is_library_model_file(path)
        ),
        key=lambda item: (_asset_kind_for_path(item), _asset_sort_priority(item), _asset_parts_text(item), item.name.lower()),
    )
    for model in model_files:
        preview = _preview_for_asset(model, preview_index)
        if preview is not None:
            add_card(model, preview)

    for preview in previews:
        if preview.resolve() in used_previews:
            continue
        add_card(preview, preview, preview_only=True)

    libraries['robots'] = _collapse_robot_library(libraries['robots'])
    return libraries


def _asset_libraries(selected_recipe: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {name: [dict(item) for item in rows] for name, rows in _asset_library_cache().items()}


def _recipe_summary(name: str, recipe: dict[str, Any]) -> dict[str, Any]:
    metadata = recipe.get('metadata') or {}
    task_name = recipe.get('task_name', name)
    return {
        'name': _template_public_id(name),
        'display_name': _template_display_name(name),
        'task_name': PUBLIC_GENERATED_TASK_NAME if name == DEFAULT_TEMPLATE else task_name,
        'prompt': recipe.get('prompt') or recipe.get('task_description') or '',
        'scene_profile': recipe.get('scene_profile'),
        'robots': [item.get('name') for item in recipe.get('robots', []) if isinstance(item, dict)],
        'object_count': len(recipe.get('objects', [])),
        'target_count': len(recipe.get('targets', [])),
        'phase_count': len(recipe.get('phases', [])),
        'objects': [
            {
                'name': item.get('name'),
                'kind': item.get('kind'),
                'position': item.get('position'),
                'tracked': item.get('tracked', True),
            }
            for item in recipe.get('objects', [])[:80]
            if isinstance(item, dict)
        ],
        'targets': [
            {
                key: item[key]
                for key in ('name', 'reference', 'position', 'offset', 'orientation_euler')
                if isinstance(item, dict) and key in item
            }
            for item in recipe.get('targets', [])[:80]
        ],
        'phases': [phase.get('name') for phase in recipe.get('phases', [])[:120] if isinstance(phase, dict)],
        'asset_references': recipe.get('asset_references', [])[:120],
        'tags': metadata.get('tags', []),
    }


def serialize_inventory(scene_profile: str | None, selected_template: str | None = DEFAULT_TEMPLATE) -> dict[str, Any]:
    inventory = RoboAssemblyInventory.load(scene_profile=scene_profile)
    templates = []
    for name in inventory.template_names():
        compact = inventory.compact_recipes.get(name, {})
        templates.append(
            {
                'name': _template_public_id(name),
                'display_name': _template_display_name(name),
                'prompt': compact.get('prompt') or compact.get('task_description') or '',
                'robots': compact.get('robots', []),
                'object_count': len(compact.get('objects', [])),
                'target_count': len(compact.get('targets', [])),
                'phase_count': len(compact.get('phases', [])),
                'tags': compact.get('tags', []),
            }
        )

    selected_internal = _template_internal_id(selected_template)
    template_name = selected_internal if selected_internal in inventory.recipes else inventory.best_template_for(selected_internal or '')
    selected = _recipe_summary(template_name, inventory.recipes[template_name])
    return {
        'scene_profile': scene_profile or 'raw',
        'scene_profiles': _scene_profiles(),
        'default_template': PUBLIC_TEMPLATE_ID,
        'mock_llm_default': not bool(os.environ.get('OPENAI_API_KEY')),
        'selected_template': _template_public_id(template_name),
        'templates': templates,
        'selected_recipe': selected,
        'asset_catalog': _asset_catalog(),
        'asset_libraries': _asset_libraries(inventory.recipes[template_name]),
    }


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def _read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        return yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except (OSError, yaml.YAMLError):
        return None


def summarize_result(result: RoboBrainRunResult) -> dict[str, Any]:
    payload = result.to_dict()
    artifacts = [
        {
            'label': key,
            'path': str(path),
            'kind': 'file' if path.is_file() else 'dir' if path.is_dir() else 'path',
        }
        for key, path in result.bundle_paths.items()
    ]
    run_dir = result.bundle_paths['recipe'].parent if 'recipe' in result.bundle_paths else None
    if result.demo_command and run_dir is not None:
        demo_command_path = run_dir / 'demo_command.json'
        if demo_command_path.exists():
            artifacts.append({'label': 'demo_command', 'path': str(demo_command_path), 'kind': 'file'})
    if result.demo_output_dir is not None:
        artifacts.append({'label': 'demo_output_dir', 'path': str(result.demo_output_dir), 'kind': 'dir'})
    primitive_path = result.bundle_paths.get('primitive_plan')
    recipe_path = result.bundle_paths.get('recipe')
    if primitive_path is not None:
        payload['primitive_plan'] = _read_json(primitive_path)
    if recipe_path is not None:
        recipe = _read_yaml(recipe_path) or {}
        payload['recipe_summary'] = {
            'task_name': recipe.get('task_name'),
            'object_count': len(recipe.get('objects', [])),
            'target_count': len(recipe.get('targets', [])),
            'phase_count': len(recipe.get('phases', [])),
            'local_skill_count': len((recipe.get('metadata') or {}).get('local_skills', {})),
        }
    payload['run_dir'] = None if run_dir is None else str(run_dir)
    payload['artifacts'] = artifacts
    return _jsonable(payload)


def _run_lerobot_export(job: AgentJob, demo_output_dir: Path, request: AgentRunRequest) -> dict[str, Any]:
    output_dir = demo_output_dir.parent / 'lerobot'
    command = [
        sys.executable,
        str(BENCHMARK_ROOT / 'scripts' / 'export_lerobot.py'),
        '--input-dir',
        str(demo_output_dir),
        '--output-dir',
        str(output_dir),
        '--fps',
        str(request.lerobot_fps),
        '--video-mode',
        request.lerobot_video_mode,
    ]
    if request.lerobot_include_failures:
        command.append('--include-failures')
    if request.keep_video_frames:
        command.append('--keep-video-frames')

    job.add_event('lerobot_export_started', {'command': command, 'output_dir': output_dir})
    output_lines: list[str] = []
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        output_lines.append(text)
        job.add_event('lerobot_output', {'line': text})
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f'LeRobot export failed with return code {returncode}.')

    summary_path = output_dir / 'meta' / 'summary.json'
    summary = _read_json(summary_path) or {'output_dir': str(output_dir), 'stdout': '\n'.join(output_lines[-20:])}
    job.add_event('lerobot_export_completed', {'summary_path': summary_path, 'summary': summary})
    return _jsonable(summary)


def _manual_demo_command(*, recipe_path: Path, output_dir: Path, request: AgentRunRequest) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK_ROOT / 'scripts' / 'generate_demos.py'),
        '--recipes',
        str(recipe_path),
        '--num-demos',
        str(request.num_demos),
        '--start-seed',
        str(request.start_seed),
        '--max-trials',
        str(request.max_trials),
        '--output-dir',
        str(output_dir),
    ]
    if request.scene_profile is not None:
        command.extend(['--scene-profiles', request.scene_profile])
    if request.headless:
        command.append('--headless')
    if request.record_live_video:
        command.append('--record-live-video')
        command.extend(['--live-video-fps', str(request.live_video_fps)])
        command.extend(['--live-video-frame-stride', str(request.live_video_frame_stride)])
    if request.keep_video_frames:
        command.append('--keep-video-frames')
    if request.runtime_robochecker:
        command.append('--runtime-robochecker')
        command.extend(['--runtime-checker-stride', str(request.runtime_checker_stride)])
        command.extend(['--runtime-rgb-frame-stride', str(request.runtime_rgb_frame_stride)])
        command.append('--runtime-stop-on-violation')
        if request.runtime_capture_rgb:
            command.append('--runtime-capture-rgb')
        command.extend(['--runtime-feedback-path', str(output_dir.parent / 'runtime_feedback.json')])
        command.extend(['--runtime-observation-dir', str(output_dir.parent / 'runtime_observations')])
    return command


def _run_streamed_command(job: AgentJob, *, event_prefix: str, command: list[str]):
    job.add_event(f'{event_prefix}_started', {'command': command})
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        job.add_event(f'{event_prefix}_output', {'line': line.rstrip()})
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f'{event_prefix} failed with return code {returncode}.')
    job.add_event(f'{event_prefix}_completed', {'returncode': returncode})


def _clean_result_summary_for_typewriter(text: Any) -> str:
    value = str(text or '')
    if value.startswith('我正在进行「') and '」。' in value:
        return value.split('」。', 1)[1].lstrip()
    return value


def _result_typewriter_segments(trace_item: dict[str, Any]) -> list[str]:
    output = trace_item.get('output') or {}
    segments = [_clean_result_summary_for_typewriter(trace_item.get('thinking_process'))]
    for line in trace_item.get('visible_process_lines') or trace_item.get('process_lines') or []:
        segments.append(str(line or ''))
    for step in trace_item.get('visible_reasoning_steps') or trace_item.get('reasoning_steps') or []:
        if not isinstance(step, dict):
            continue
        segments.extend([str(step.get('label') or '过程'), str(step.get('text') or '')])
    section_cards: list[dict[str, Any]] = []
    sections = output.get('decomposition_sections') or trace_item.get('decomposition_sections') or []
    for section in sections:
        if not isinstance(section, dict):
            continue
        segments.extend([str(section.get('range') or ''), str(section.get('title') or ''), str(section.get('summary') or '')])
        skill_flow = [item for item in section.get('skill_flow') or [] if isinstance(item, dict)]
        if skill_flow:
            segments.append(f"包含 {len(skill_flow)} 个技能阶段")
        else:
            section_cards.extend([item for item in section.get('cards') or [] if isinstance(item, dict)])
    cards = section_cards if sections else output.get('subtask_cards') or trace_item.get('subtask_cards') or []
    for item in cards:
        if not isinstance(item, dict):
            continue
        segments.extend([f"阶段 {item.get('index') or ''}", str(item.get('title') or item.get('phase') or '')])
        if item.get('goal'):
            segments.append(str(item.get('goal') or ''))
        if item.get('skill'):
            segments.extend(['技能', str(item.get('skill') or '').split('（', 1)[0]])
    if trace_item.get('next_step'):
        segments.append(str(trace_item.get('next_step')))
    return segments


def _result_typewriter_duration(trace_item: dict[str, Any]) -> float:
    char_count = sum(len(segment) for segment in _result_typewriter_segments(trace_item))
    if char_count <= 0:
        return 0.0
    return char_count * TYPEWRITER_INTERVAL_SECONDS + TYPEWRITER_BUFFER_SECONDS


def _lerobot_dataset_structure() -> dict[str, Any]:
    return {
        'dataset_format': 'LeRobotDataset v3.0',
        'dataset_name': 'fabrica_plumbers_block_ur5e_assembly',
        'root': 'outputs/lerobot/fabrica_plumbers_block_ur5e_assembly',
        'directories': [
            {
                'path': 'meta/info.json',
                'content': '数据集 schema、fps、features、相机流、状态维度和动作维度。',
            },
            {
                'path': 'meta/stats.json',
                'content': 'observation.state、action、关节轨迹、末端位姿等字段的统计量。',
            },
            {
                'path': 'meta/tasks.jsonl',
                'content': '自然语言任务、Template、Menu、Annotation 和 skill plan。',
            },
            {
                'path': 'meta/episodes/',
                'content': '每个 episode 的长度、seed、成功标记、起止 offset 和失败原因。',
            },
            {
                'path': 'data/chunk-000/file-000.parquet',
                'content': 'timestamp、frame_index、episode_index、observation.state、action、skill_id、object_state。',
            },
            {
                'path': 'videos/chunk-000/observation.images.front/file-000.mp4',
                'content': '第三视角相机视频。',
            },
            {
                'path': 'videos/chunk-000/observation.images.left_wrist/file-000.mp4',
                'content': '左腕部相机视频。',
            },
            {
                'path': 'videos/chunk-000/observation.images.right_wrist/file-000.mp4',
                'content': '右腕部相机视频。',
            },
        ],
        'features': [
            'observation.state: 双臂关节角、关节速度、末端位姿、夹爪开合、本体感受态。',
            'action: 当前 skill、操作臂、目标物体、目标位姿、夹爪命令、控制器目标。',
            'observation.images.*: front、left_wrist、right_wrist 多相机 MP4。',
            'object_state: plumbers-block 0/1/2/3/4 的位置、姿态、静止状态和接触状态。',
            'episode metadata: task、seed、success、Template、Menu、Annotation、skill sequence。',
        ],
    }


def _run_manual_demo_job(job: AgentJob):
    summary = write_manual_demo_bundle(
        output_root=job.request.output_dir,
        task_instruction=job.request.task,
        scene_profile=job.request.scene_profile,
    )
    job.add_event('manual_demo_bundle_written', {'run_dir': summary['run_dir'], 'bundle_paths': summary['bundle_paths']})
    for trace_item in summary.get('manual_reasoning_trace', []):
        job.add_event(
            'manual_thinking_started',
            {
                'stage': trace_item.get('stage'),
                'title': trace_item.get('title'),
                'message': f"LLM 正在分析：{trace_item.get('title') or trace_item.get('stage')}",
            },
        )
        if job.request.manual_thinking_delay > 0:
            time.sleep(job.request.manual_thinking_delay)
        job.add_event(
            'manual_thinking_finished',
            {
                'stage': trace_item.get('stage'),
                'title': trace_item.get('title'),
                'message': f"{trace_item.get('title') or trace_item.get('stage')}：分析完成",
            },
        )
        if job.request.manual_thinking_delay > 0:
            time.sleep(min(job.request.manual_thinking_delay, MANUAL_THINKING_FINISHED_PAUSE))
        job.add_event(f"manual_{trace_item.get('stage', 'trace')}", trace_item)
        if job.request.manual_thinking_delay > 0:
            time.sleep(_result_typewriter_duration(trace_item) + MANUAL_THINKING_PRINT_PAUSE)
    job.add_event('manual_skill_steps_ready', {'count': len(summary.get('manual_skill_steps', []))})
    job.add_event('manual_checker_completed', summary.get('check_result', {}))

    if job.request.run_simulation:
        run_dir = Path(summary['run_dir'])
        demo_output_dir = run_dir / 'demo'
        command = _manual_demo_command(
            recipe_path=Path(summary['bundle_paths']['recipe']),
            output_dir=demo_output_dir,
            request=job.request,
        )
        demo_command_path = run_dir / 'demo_command.json'
        demo_command_path.write_text(json.dumps(command, indent=2), encoding='utf-8')
        summary['demo_command'] = command
        summary['demo_output_dir'] = str(demo_output_dir)
        summary['artifacts'].append({'label': 'demo_command', 'path': str(demo_command_path), 'kind': 'file'})
        summary['artifacts'].append({'label': 'demo_output_dir', 'path': str(demo_output_dir), 'kind': 'dir'})
        _run_streamed_command(job, event_prefix='manual_demo_execution', command=command)
        runtime_feedback_path = run_dir / 'runtime_feedback.json'
        summary['runtime_feedback'] = _read_json(runtime_feedback_path) if runtime_feedback_path.exists() else None

    if job.request.export_lerobot:
        demo_output_dir_text = summary.get('demo_output_dir')
        if not demo_output_dir_text:
            job.add_event(
                'lerobot_export_skipped',
                {'reason': 'Simulation has not produced a demo episode yet, so LeRobot export is waiting.'},
            )
        else:
            summary['lerobot'] = _run_lerobot_export(job, Path(demo_output_dir_text), job.request)

    return summary


def _run_job(job: AgentJob):
    job.status = 'running'
    job.add_event('job_started', {'request': job.request.to_dict()})
    try:
        if job.request.manual_demo:
            job.add_event('manual_demo_selected', {'template': MANUAL_DEMO_TEMPLATE, 'llm_called': True, 'planner': 'online_llm'})
            summary = _run_manual_demo_job(job)
            job.result = summary
            job.status = 'succeeded'
            job.add_event('job_succeeded', {'run_dir': summary.get('run_dir')})
            return

        job.add_event(
            'inventory_snapshot',
            serialize_inventory(scene_profile=job.request.scene_profile, selected_template=job.request.template),
        )
        config = job.request.to_runner_config(job.add_runner_event)
        result = RoboBrainRunner(config).run(job.request.task)
        summary = summarize_result(result)
        job.result = summary
        job.add_event('runner_result_ready', summary)

        if job.request.export_lerobot:
            if result.demo_output_dir is None:
                job.add_event('lerobot_export_skipped', {'reason': 'No demo output exists because the job ran in plan-only mode.'})
            else:
                summary['lerobot'] = _run_lerobot_export(job, result.demo_output_dir, job.request)
                job.result = summary

        job.status = 'succeeded'
        job.add_event('job_succeeded', {'run_dir': summary.get('run_dir')})
    except Exception as exc:  # noqa: BLE001 - UI jobs need to surface any pipeline failure.
        job.status = 'failed'
        job.error = f'{type(exc).__name__}: {exc}'
        job.add_event('job_failed', {'error': job.error})


def _run_simulation_job(job: SimulationJob):
    job.status = 'running'
    command = ['bash', str(job.script)]
    job.add_event(
        'simulation_started',
        {
            'script': str(job.script.relative_to(REPO_ROOT) if REPO_ROOT in job.script.parents else job.script),
            'command': command,
            'cwd': str(REPO_ROOT),
            'expected_video': str(SIMULATION_OUTPUT),
        },
    )
    try:
        if not job.script.exists():
            raise FileNotFoundError(f'Simulation script not found: {job.script}')
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, 'HEADLESS': '0'},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            if text:
                job.add_event('simulation_output', {'line': text})
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f'Simulation script failed with return code {returncode}.')
        job.result = {
            'returncode': returncode,
            'video': str(SIMULATION_OUTPUT),
            'video_exists': SIMULATION_OUTPUT.exists(),
            'lerobot_dataset': _lerobot_dataset_structure(),
        }
        job.status = 'succeeded'
        job.add_event('simulation_completed', job.result)
    except Exception as exc:  # noqa: BLE001 - surface the full launch failure in the UI.
        job.status = 'failed'
        job.error = f'{type(exc).__name__}: {exc}'
        job.add_event('simulation_failed', {'error': job.error})


def _safe_path(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        return path
    raise ValueError(f'Path is outside the RoboAssemblyBench workspace: {path}')


def _recording_upload_suffix(content_type: str | None) -> str:
    mime = str(content_type or '').split(';', 1)[0].strip().lower()
    return RECORDING_UPLOAD_SUFFIXES.get(mime, '.webm')


def _save_recording_as_mp4(payload: bytes, content_type: str | None) -> dict[str, Any]:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    base = RECORDINGS_DIR / f'{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:6]}_roboassembly_panel_simulation'
    source_suffix = _recording_upload_suffix(content_type)
    output_path = base.with_suffix('.mp4')
    source_mime = str(content_type or 'application/octet-stream').split(';', 1)[0].strip().lower()

    if source_suffix == '.mp4':
        output_path.write_bytes(payload)
        converted = False
    else:
        source_path = base.with_suffix(source_suffix)
        source_path.write_bytes(payload)
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            raise RuntimeError('ffmpeg is required to convert the screen recording to MP4.')
        command = [
            ffmpeg,
            '-y',
            '-hide_banner',
            '-loglevel',
            'error',
            '-i',
            str(source_path),
            '-c:v',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            '-movflags',
            '+faststart',
            '-an',
            str(output_path),
        ]
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
            error_text = (result.stderr or result.stdout or 'ffmpeg failed without output.').strip()
            raise RuntimeError(f'Failed to convert recording to MP4: {error_text}')
        source_path.unlink(missing_ok=True)
        converted = True

    return {
        'path': str(output_path),
        'filename': output_path.name,
        'size_bytes': output_path.stat().st_size,
        'media_url': f'/api/media?path={output_path}',
        'mime_type': 'video/mp4',
        'source_mime_type': source_mime or 'application/octet-stream',
        'converted_to_mp4': converted,
    }


class MissingDependencyApp:
    async def __call__(self, scope, receive, send):  # pragma: no cover - used only without FastAPI.
        body = (
            'FastAPI and Uvicorn are required for the RoboAssembly application.\n'
            'Install them with: pip install -r requirements/webui.txt\n'
        ).encode('utf-8')
        await send({'type': 'http.response.start', 'status': 500, 'headers': [(b'content-type', b'text/plain; charset=utf-8')]})
        await send({'type': 'http.response.body', 'body': body})


def create_app():
    if FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError('Install web UI dependencies with: pip install -r requirements/webui.txt') from FASTAPI_IMPORT_ERROR

    api = FastAPI(title='RoboAssembly', version='0.1.0')

    @api.get('/', response_class=HTMLResponse)
    def index():
        return HTMLResponse(APP_HTML)

    @api.get('/api/inventory')
    def inventory(scene_profile: str = DEFAULT_SCENE_PROFILE, template: str = DEFAULT_TEMPLATE):
        try:
            return JSONResponse(serialize_inventory(_normalize_scene_profile(scene_profile), template))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @api.post('/api/jobs')
    def create_job(payload: dict[str, Any]):
        request = AgentRunRequest.from_payload(payload)
        if not request.task:
            raise HTTPException(status_code=400, detail='Task instruction is required.')
        job = AgentJob(id=uuid.uuid4().hex[:12], request=request)
        with _JOBS_LOCK:
            _JOBS[job.id] = job
        _EXECUTOR.submit(_run_job, job)
        return JSONResponse({'job_id': job.id, 'status': job.status})

    @api.get('/api/jobs')
    def list_jobs():
        with _JOBS_LOCK:
            jobs = [job.to_dict(after=None) for job in _JOBS.values()]
        jobs.sort(key=lambda item: item['created_at'], reverse=True)
        return JSONResponse({'jobs': jobs})

    @api.get('/api/jobs/{job_id}')
    def get_job(job_id: str, after: int | None = None):
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Job not found.')
        return JSONResponse(job.to_dict(after=after))

    @api.post('/api/simulation/start')
    def start_simulation():
        job = SimulationJob(id=uuid.uuid4().hex[:12])
        with _SIMULATION_JOBS_LOCK:
            _SIMULATION_JOBS[job.id] = job
        _EXECUTOR.submit(_run_simulation_job, job)
        return JSONResponse({'job_id': job.id, 'status': job.status})

    @api.get('/api/simulation/{job_id}')
    def get_simulation_job(job_id: str, after: int | None = None):
        with _SIMULATION_JOBS_LOCK:
            job = _SIMULATION_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Simulation job not found.')
        return JSONResponse(job.to_dict(after=after))

    @api.post('/api/recordings')
    async def save_recording(request: Request):
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=400, detail='Recording body is empty.')
        try:
            saved = _save_recording_as_mp4(payload, request.headers.get('content-type'))
        except Exception as exc:  # noqa: BLE001 - surface conversion errors in the UI.
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JSONResponse(saved)

    @api.get('/api/media')
    def preview_media(path: str):
        try:
            safe = _safe_path(path)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not safe.exists() or not safe.is_file():
            raise HTTPException(status_code=404, detail='Media file not found.')
        if safe.suffix.lower() not in {'.mp4', '.webm', '.mov', '.mkv'}:
            raise HTTPException(status_code=400, detail='Unsupported media type.')
        media_types = {
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
        }
        return FileResponse(safe, media_type=media_types.get(safe.suffix.lower()))

    @api.get('/api/file')
    def preview_file(path: str):
        try:
            safe = _safe_path(path)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not safe.exists() or not safe.is_file():
            raise HTTPException(status_code=404, detail='File not found.')
        if safe.stat().st_size > 2_000_000:
            raise HTTPException(status_code=413, detail='File is too large for inline preview.')
        return PlainTextResponse(safe.read_text(encoding='utf-8', errors='replace'))

    @api.get('/api/image')
    def preview_image(path: str):
        try:
            safe = _safe_path(path)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not safe.exists() or not safe.is_file():
            raise HTTPException(status_code=404, detail='Image not found.')
        if safe.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
            raise HTTPException(status_code=400, detail='Unsupported image type.')
        return FileResponse(safe)

    return api


app = create_app() if FASTAPI_IMPORT_ERROR is None else MissingDependencyApp()


def main():
    if FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError('Install web UI dependencies with: pip install -r requirements/webui.txt') from FASTAPI_IMPORT_ERROR
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('Uvicorn is required. Install it with: pip install -r requirements/webui.txt') from exc

    host = os.environ.get('ROBOBRAIN_AGENT_HOST', '127.0.0.1')
    port = int(os.environ.get('ROBOBRAIN_AGENT_PORT', '7861'))
    uvicorn.run('roboassemblybench.robobrain.webapp:app', host=host, port=port, reload=False)


APP_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RoboAssembly</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #1f2937;
      --muted: #667085;
      --blue: #1d4ed8;
      --teal: #0f766e;
      --amber: #b45309;
      --red: #b91c1c;
      --green: #15803d;
    }
    * { box-sizing: border-box; letter-spacing: 0; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    button, input, select, textarea {
      font: inherit;
    }
    .shell {
      min-height: 100vh;
      max-height: 100vh;
      display: grid;
      grid-template-columns: 390px minmax(0, 1fr);
      grid-template-rows: 56px minmax(0, 1fr);
      overflow: hidden;
    }
    header {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
      font-weight: 720;
    }
    .mark {
      width: 28px;
      height: 28px;
      border-radius: 6px;
      background: linear-gradient(135deg, var(--blue), var(--teal));
      position: relative;
      flex: 0 0 auto;
    }
    .mark:after {
      content: "";
      position: absolute;
      inset: 8px;
      border: 2px solid #ffffff;
      border-radius: 3px;
    }
    .top-status {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      min-width: 0;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #fbfcfe;
      padding: 16px;
      min-height: 0;
      overflow: hidden;
      display: grid;
      grid-template-rows: minmax(260px, 0.9fr) minmax(220px, 1.1fr);
      align-content: stretch;
      gap: 12px;
    }
    main {
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      padding: 16px;
      display: block;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    .task-panel,
    .asset-panel {
      min-height: 0;
      overflow: hidden;
      display: grid;
      grid-template-rows: 42px minmax(0, 1fr);
    }
    .panel-head {
      height: 42px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 14px;
      border-bottom: 1px solid var(--line);
      font-weight: 680;
      min-width: 0;
    }
    .chat-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }
    .panel-body {
      padding: 14px;
      min-width: 0;
    }
    .scroll-body {
      overflow-y: auto;
      overflow-x: hidden;
      min-height: 0;
      scrollbar-gutter: stable;
    }
    label {
      display: block;
      color: #344054;
      font-weight: 620;
      margin-bottom: 6px;
    }
    textarea, input, select {
      width: 100%;
      border: 1px solid #c9d1df;
      border-radius: 6px;
      padding: 9px 10px;
      background: #ffffff;
      color: var(--text);
      min-width: 0;
    }
    textarea {
      min-height: 96px;
      resize: vertical;
      line-height: 1.45;
    }
    .field {
      margin-bottom: 12px;
    }
    .grid2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .checks {
      display: grid;
      gap: 8px;
      margin: 10px 0 14px;
    }
    .check {
      display: grid;
      grid-template-columns: 18px minmax(0, 1fr);
      align-items: center;
      gap: 8px;
      color: #344054;
      font-weight: 560;
    }
    .check input { width: 16px; height: 16px; }
    .primary, .secondary {
      border: 0;
      border-radius: 6px;
      min-height: 38px;
      padding: 0 12px;
      cursor: pointer;
      font-weight: 700;
      white-space: nowrap;
    }
    .primary {
      background: var(--blue);
      color: #ffffff;
      width: 100%;
    }
    .secondary {
      background: #eef4ff;
      color: var(--blue);
      border: 1px solid #c7d7fe;
    }
    .recording-button {
      background: #fff1f2;
      color: #be123c;
      border-color: #fecdd3;
    }
    .recording-button.active {
      background: #be123c;
      color: #ffffff;
      border-color: #be123c;
    }
    .small-button {
      min-height: 30px;
      padding: 0 10px;
      font-size: 12px;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .chat-panel {
      height: calc(100vh - 88px);
      display: grid;
      grid-template-rows: 42px minmax(0, 1fr);
    }
    .chat {
      display: grid;
      gap: 12px;
      align-content: start;
      height: 100%;
      overflow: auto;
      padding-right: 2px;
    }
    .chat-message {
      display: grid;
      gap: 6px;
      max-width: 92%;
      min-width: 0;
    }
    .chat-message.user { justify-self: end; }
    .chat-message.agent, .chat-message.thinking, .chat-message.system { justify-self: start; }
    .chat-message.finished { justify-self: start; }
    .bubble {
      border: 1px solid #e3e8f0;
      border-radius: 8px;
      background: #ffffff;
      padding: 10px 12px;
      line-height: 1.55;
      overflow-wrap: anywhere;
      min-width: 0;
    }
    .chat-message.user .bubble {
      background: #eef4ff;
      border-color: #c7d7fe;
    }
    .chat-message.thinking .bubble {
      background: #fff7ed;
      border-color: #fed7aa;
      color: #7c2d12;
      display: grid;
      gap: 8px;
    }
    .thinking-status {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
    }
    .chat-message.thinking .thought-summary {
      color: #7c2d12;
      margin-bottom: 0;
    }
    .chat-message.finished .bubble {
      background: #ecfdf5;
      border-color: #a7f3d0;
      color: #065f46;
      font-weight: 700;
    }
    .chat-meta {
      color: var(--muted);
      font-size: 12px;
      padding: 0 2px;
    }
    .spinner {
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 2px solid #fdba74;
      border-top-color: #c2410c;
      flex: 0 0 auto;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .thought-title {
      font-weight: 750;
      margin-bottom: 4px;
    }
    .thought-next {
      margin-top: 8px;
      color: #0f766e;
      font-weight: 650;
    }
    .thought-summary {
      color: #344054;
      margin-bottom: 8px;
      font-weight: 650;
    }
    .typewriter-cursor {
      display: inline-block;
      width: 7px;
      height: 1em;
      margin-left: 2px;
      border-right: 2px solid #1d4ed8;
      vertical-align: -2px;
      animation: cursorBlink 0.8s steps(1) infinite;
    }
    @keyframes cursorBlink { 50% { opacity: 0; } }
    .reasoning-list {
      display: grid;
      gap: 7px;
      margin-top: 8px;
    }
    .process-lines {
      display: grid;
      gap: 7px;
      margin-top: 8px;
    }
    .process-line {
      border-left: 3px solid #0f766e;
      background: #f0fdfa;
      border-radius: 6px;
      padding: 8px 10px;
      color: #134e4a;
      line-height: 1.55;
    }
    .reasoning-step {
      display: grid;
      grid-template-columns: 64px minmax(0, 1fr);
      gap: 8px;
      border-left: 3px solid #1d4ed8;
      background: #f8fafc;
      border-radius: 6px;
      padding: 8px 9px;
    }
    .reasoning-label {
      color: #1d4ed8;
      font-weight: 750;
    }
    .reasoning-text {
      color: #344054;
      min-width: 0;
    }
    .subtask-list {
      display: grid;
      gap: 9px;
      margin-top: 10px;
    }
    .decomposition-list {
      display: grid;
      gap: 12px;
      margin-top: 10px;
    }
    .decomposition-section {
      display: grid;
      gap: 8px;
    }
    .section-head {
      border: 1px solid #b7e4d9;
      background: #ecfdf5;
      border-radius: 8px;
      padding: 9px 10px;
    }
    .section-title {
      font-weight: 800;
      color: #065f46;
    }
    .section-summary {
      margin-top: 4px;
      color: #134e4a;
      line-height: 1.45;
    }
    .subtask-card {
      border: 1px solid #dbe4f0;
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
    }
    .subtask-head {
      display: grid;
      grid-template-columns: 54px minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      padding: 9px 10px;
      background: #eef4ff;
      border-bottom: 1px solid #dbe4f0;
    }
    .subtask-index {
      color: #1d4ed8;
      font-weight: 800;
    }
    .subtask-title {
      font-weight: 760;
      min-width: 0;
    }
    .subtask-body {
      display: grid;
      gap: 6px;
      padding: 9px 10px;
    }
    .subtask-row {
      display: grid;
      grid-template-columns: 76px minmax(0, 1fr);
      gap: 8px;
      min-width: 0;
    }
    .subtask-label {
      color: var(--muted);
      font-weight: 700;
    }
    .subtask-value {
      color: #344054;
      min-width: 0;
    }
    .subskill-list {
      display: grid;
      gap: 5px;
      margin-top: 2px;
      padding: 7px 9px;
      border-radius: 6px;
      background: #f8fafc;
      border: 1px solid #e3e8f0;
    }
    .subskill-line {
      color: #344054;
      line-height: 1.45;
    }
    .constraint-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .constraint-card {
      border: 1px solid #dbe4f0;
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
      min-width: 0;
    }
    .constraint-head {
      padding: 8px 10px;
      background: #f8fafc;
      border-bottom: 1px solid #e3e8f0;
      font-weight: 800;
    }
    .constraint-head.logical { color: #065f46; background: #ecfdf5; }
    .constraint-head.spatial { color: #1d4ed8; background: #eef4ff; }
    .constraint-head.temporal { color: #92400e; background: #fffbeb; }
    .constraint-body {
      display: grid;
      gap: 6px;
      padding: 9px 10px;
    }
    .constraint-summary {
      color: #344054;
      font-weight: 650;
      line-height: 1.45;
    }
    .constraint-rule {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      padding-left: 10px;
      border-left: 2px solid #dbe4f0;
    }
    .dataset-card {
      border: 1px solid #b7e4d9;
      background: #f0fdfa;
      border-radius: 8px;
      padding: 10px 12px;
      margin-top: 10px;
      display: grid;
      gap: 9px;
    }
    .dataset-title {
      color: #065f46;
      font-weight: 820;
    }
    .dataset-root {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: #134e4a;
      overflow-wrap: anywhere;
    }
    .dataset-tree {
      display: grid;
      gap: 5px;
      padding: 8px 10px;
      border: 1px solid #b7e4d9;
      border-radius: 6px;
      background: #ffffff;
    }
    .dataset-row {
      display: grid;
      grid-template-columns: minmax(180px, 0.85fr) minmax(0, 1.15fr);
      gap: 8px;
      align-items: start;
    }
    .dataset-path {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: #0f766e;
      overflow-wrap: anywhere;
    }
    .dataset-desc {
      color: #344054;
      font-size: 12px;
      line-height: 1.45;
    }
    .feature-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .feature-tag {
      border-radius: 999px;
      background: #ffffff;
      border: 1px solid #b7e4d9;
      color: #134e4a;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 650;
    }
    .recording-tip {
      margin-top: 8px;
      border-left: 3px solid #1d4ed8;
      background: #eef4ff;
      border-radius: 6px;
      padding: 8px 10px;
      color: #1e3a8a;
      line-height: 1.45;
      font-size: 12px;
    }
    .sim-log {
      max-height: 360px;
      background: #0f172a;
    }
    .recording-link {
      display: inline-flex;
      margin-top: 8px;
      color: var(--blue);
      font-weight: 700;
    }
    .bubble details {
      margin-top: 8px;
    }
    .bubble summary {
      cursor: pointer;
      color: var(--muted);
      font-weight: 650;
    }
    details.event {
      border: 1px solid #e3e8f0;
      border-radius: 6px;
      background: #ffffff;
      min-width: 0;
    }
    details.event summary {
      cursor: pointer;
      padding: 8px 10px;
      display: grid;
      grid-template-columns: 56px minmax(0, 1fr) 84px;
      gap: 8px;
      align-items: center;
    }
    .event-type {
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .event-time {
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      background: #eef2f7;
      color: #475467;
      font-size: 12px;
      font-weight: 720;
    }
    .badge.ok { background: #dcfce7; color: var(--green); }
    .badge.warn { background: #fef3c7; color: var(--amber); }
    .badge.fail { background: #fee2e2; color: var(--red); }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #111827;
      color: #e5e7eb;
      border-radius: 6px;
      padding: 12px;
      line-height: 1.45;
      max-height: 560px;
      overflow: auto;
      font-size: 12px;
    }
    .kv {
      display: grid;
      grid-template-columns: 130px minmax(0, 1fr);
      gap: 6px 10px;
      align-items: start;
    }
    .kv .k { color: var(--muted); }
    .rows {
      display: grid;
      gap: 8px;
    }
    .row {
      display: grid;
      grid-template-columns: minmax(100px, 0.7fr) minmax(0, 1.3fr);
      gap: 10px;
      padding: 8px 0;
      border-bottom: 1px solid #edf0f5;
      min-width: 0;
    }
    .row:last-child { border-bottom: 0; }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .muted { color: var(--muted); }
    .file-list {
      display: grid;
      gap: 8px;
    }
    .file-row {
      display: grid;
      grid-template-columns: 130px minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      border-bottom: 1px solid #edf0f5;
      padding-bottom: 8px;
    }
    .asset-library {
      display: grid;
      gap: 12px;
      padding-right: 2px;
    }
    .asset-section {
      display: grid;
      gap: 8px;
    }
    .asset-section-title {
      font-weight: 760;
      color: #344054;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .asset-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }
    .asset-card {
      display: grid;
      grid-template-columns: 86px minmax(0, 1fr);
      gap: 10px;
      border: 1px solid #e3e8f0;
      border-radius: 8px;
      background: #ffffff;
      padding: 8px;
      min-width: 0;
    }
    .asset-thumb {
      width: 86px;
      height: 66px;
      border-radius: 6px;
      object-fit: cover;
      background: #e9eef5;
      border: 1px solid #d7dee9;
    }
    .asset-title {
      font-weight: 740;
      margin-bottom: 3px;
    }
    .asset-desc {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .asset-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-top: 6px;
    }
    .tag {
      display: inline-flex;
      align-items: center;
      min-height: 18px;
      padding: 0 6px;
      border-radius: 999px;
      background: #f2f4f7;
      color: #475467;
      font-size: 11px;
      font-weight: 650;
    }
    .user-profile {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }
    .avatar {
      width: 32px;
      height: 32px;
      border-radius: 999px;
      background: linear-gradient(135deg, #0f766e, #1d4ed8);
      color: #ffffff;
      display: grid;
      place-items: center;
      font-weight: 800;
      flex: 0 0 auto;
    }
    .profile-name {
      font-weight: 750;
      line-height: 1.1;
    }
    .profile-role {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.1;
    }
    .empty {
      color: var(--muted);
      border: 1px dashed #cfd6e2;
      border-radius: 8px;
      padding: 18px;
      text-align: center;
    }
    @media (max-width: 1020px) {
      .shell {
        grid-template-columns: 1fr;
        grid-template-rows: 56px auto auto;
      }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .chat-panel { height: 72vh; }
    }
    @media (max-width: 620px) {
      .grid2, .file-row, .row, .kv, .asset-card, .constraint-grid, .dataset-row {
        grid-template-columns: 1fr;
      }
      details.event summary {
        grid-template-columns: 48px minmax(0, 1fr);
      }
      .event-time { text-align: left; grid-column: 2; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand"><div class="mark"></div><div>RoboAssembly</div></div>
      <div class="top-status">
        <span id="jobBadge" class="badge">idle</span>
        <span id="jobId" class="mono"></span>
        <div class="user-profile">
          <div class="avatar">B</div>
          <div><div class="profile-name">baiyu24</div><div class="profile-role">Assembly Operator</div></div>
        </div>
      </div>
    </header>
    <aside>
      <div class="panel task-panel">
        <div class="panel-head">新任务</div>
        <div class="panel-body scroll-body">
          <div class="field">
            <label for="task">任务指令</label>
            <textarea id="task">基于 UR5e assembly Template 生成新任务 fabrica_plumbers_block_ur5e_assembly，生成 Menu、Annotation、子任务、技能序列，并保存一条可回放的演示轨迹。</textarea>
          </div>
          <div class="field">
            <label for="template">Reference Template</label>
            <select id="template"></select>
          </div>
          <div class="field">
            <label for="sceneProfile">场景</label>
            <select id="sceneProfile"></select>
          </div>
          <div class="grid2">
            <div class="field">
              <label for="model">模型</label>
              <input id="model" placeholder="ROBOBRAIN_MODEL">
            </div>
            <div class="field">
              <label for="temperature">温度</label>
              <input id="temperature" type="number" min="0" max="2" step="0.1" value="0.2">
            </div>
          </div>
          <div class="checks">
            <label class="check"><input id="manualDemo" type="checkbox"><span>在线 LLM 拆解</span></label>
            <label class="check"><input id="mockLlm" type="checkbox"><span>快速规划模式</span></label>
            <label class="check"><input id="runSimulation" type="checkbox"><span>运行仿真</span></label>
            <label class="check"><input id="headless" type="checkbox" checked><span>Headless</span></label>
            <label class="check"><input id="recordVideo" type="checkbox" checked><span>记录视频</span></label>
          </div>
          <div class="grid2">
            <div class="field">
              <label for="numDemos">轨迹数</label>
              <input id="numDemos" type="number" min="1" step="1" value="1">
            </div>
            <div class="field">
              <label for="maxTrials">Trials</label>
              <input id="maxTrials" type="number" min="1" step="1" value="1">
            </div>
            <div class="field">
              <label for="startSeed">Seed</label>
              <input id="startSeed" type="number" min="0" step="1" value="0">
            </div>
            <div class="field">
              <label for="maxReplans">Replans</label>
              <input id="maxReplans" type="number" min="0" step="1" value="1">
            </div>
          </div>
          <button id="startBtn" class="primary">生成新任务</button>
          <button id="manualBtn" class="secondary" style="width:100%; margin-top:8px;">生成完整任务流程</button>
        </div>
      </div>
      <div class="panel asset-panel">
        <div class="panel-head">资产库</div>
        <div class="panel-body scroll-body">
          <div id="assetPanel" class="asset-library"><div class="empty">Loading inventory</div></div>
        </div>
      </div>
    </aside>
    <main>
      <section class="panel chat-panel">
        <div class="panel-head">
          <span>对话</span>
          <div class="chat-actions">
            <button id="recordBtn" class="secondary small-button recording-button">开始录制</button>
            <button id="simBtn" class="secondary small-button">启动仿真</button>
            <span id="eventCount" class="badge">0</span>
          </div>
        </div>
        <div class="panel-body scroll-body"><div id="trace" class="chat"><div class="empty">等待任务</div></div></div>
      </section>
    </main>
  </div>
  <script>
    const state = {
      jobId: null,
      poll: null,
      job: null,
      simJobId: null,
      simPoll: null,
      simJob: null,
      recording: null,
      recordingSaved: null,
      recordingEvents: [],
      tab: 'plan',
      inventory: null,
      preview: '',
      typewriter: {}
    };
    const TYPEWRITER_INTERVAL_MS = 12;
    const stages = [
      ['inventory', 'Inventory', ['inventory_loaded', 'inventory_snapshot', 'manual_02_template_resolution', 'manual_03_asset_selection']],
      ['plan', 'Plan', ['plan_generated', 'manual_04_task_decomposition', 'manual_05_skill_step_formatting']],
      ['check', 'Check', ['checker_completed', 'manual_checker_completed']],
      ['artifacts', 'Files', ['bundle_written', 'manual_demo_bundle_written']],
      ['demo', 'Demo', ['demo_started', 'demo_completed', 'manual_demo_execution_started', 'manual_demo_execution_completed']]
    ];

    function asJson(value) { return JSON.stringify(value, null, 2); }
    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    function resetTypewriters() {
      Object.values(state.typewriter || {}).forEach(entry => {
        if (entry.timer) clearTimeout(entry.timer);
      });
      state.typewriter = {};
    }
    function cleanResultSummary(text) {
      return String(text || '').replace(/^我正在进行「[^」]+」。\s*/, '');
    }
    function compactLabel(text) {
      return String(text || '').replace(/（[^）]*）/g, '');
    }
    function scheduleTypewriter(key) {
      const entry = state.typewriter[key];
      if (!entry || entry.timer || entry.visible >= entry.total) return;
      entry.timer = setTimeout(() => {
        entry.timer = null;
        entry.visible += 1;
        renderTrace();
        scheduleTypewriter(key);
      }, TYPEWRITER_INTERVAL_MS);
    }
    function typewriterFrame(key, text) {
      const chars = Array.from(String(text || ''));
      let entry = state.typewriter[key];
      if (!entry || entry.text !== text) {
        entry = { text, chars, total: chars.length, visible: chars.length ? 1 : 0, timer: null };
        state.typewriter[key] = entry;
        scheduleTypewriter(key);
      }
      return {
        text: entry.chars.slice(0, entry.visible).join(''),
        done: entry.visible >= entry.total
      };
    }
    function typewriterBudget(key, segments) {
      const normalized = segments.map(item => String(item || ''));
      const signature = normalized.join('\u001f');
      const total = normalized.reduce((count, item) => count + Array.from(item).length, 0);
      let entry = state.typewriter[key];
      if (!entry || entry.signature !== signature) {
        entry = { signature, total, visible: total ? 1 : 0, timer: null };
        state.typewriter[key] = entry;
        scheduleTypewriter(key);
      }
      return { remaining: entry.visible, done: entry.visible >= entry.total };
    }
    function revealFromBudget(budget, text) {
      const chars = Array.from(String(text || ''));
      const before = budget.remaining;
      const take = Math.min(before, chars.length);
      budget.remaining -= take;
      const active = !budget.done && before > 0 && before <= chars.length;
      return `${escapeHtml(chars.slice(0, take).join(''))}${active ? '<span class="typewriter-cursor"></span>' : ''}`;
    }
    function subtaskTextSegments(items) {
      const segments = [];
      (Array.isArray(items) ? items : []).forEach(item => {
        segments.push(`阶段 ${item.index || ''}`, item.title || item.phase || '');
        if (item.goal) segments.push(item.goal);
        if (item.skill) segments.push('技能', compactLabel(item.skill));
      });
      return segments;
    }
    function constraintTextSegments(constraints) {
      const segments = [];
      (Array.isArray(constraints) ? constraints : []).forEach(item => {
        segments.push(item.type || '', item.summary || '', item.used_for || '');
        (item.rules || []).forEach(rule => segments.push(rule || ''));
      });
      return segments;
    }
    function resultTextSegments(summary, processLines, steps, sections, subtasks, nextStep) {
      const segments = [summary];
      (Array.isArray(processLines) ? processLines : []).forEach(line => segments.push(line));
      (Array.isArray(steps) ? steps : []).forEach(step => {
        segments.push(step.label || '过程', step.text || '');
      });
      if (Array.isArray(sections) && sections.length) {
        sections.forEach(section => {
          segments.push(section.range || '', section.title || '', section.summary || '');
          if (Array.isArray(section.skill_flow) && section.skill_flow.length) {
            segments.push(`包含 ${section.skill_flow.length} 个技能阶段`);
          } else {
            segments.push(...subtaskTextSegments(section.cards || []));
          }
        });
      } else {
        segments.push(...subtaskTextSegments(subtasks));
      }
      if (nextStep) segments.push(nextStep);
      return segments;
    }
    function resultSegmentsForPayload(summary, processLines, steps, output, subtasks, nextStep) {
      return [
        ...resultTextSegments(
          summary,
          processLines,
          steps,
          output?.decomposition_sections || [],
          subtasks,
          nextStep
        ),
        ...constraintTextSegments(output?.constraint_rules || [])
      ];
    }
    function renderProcessLinesTyped(lines, budget) {
      if (!Array.isArray(lines) || !lines.length) return '';
      const rows = lines.map(line => {
        const text = revealFromBudget(budget, line || '');
        if (!text) return '';
        return `<div class="process-line">${text}</div>`;
      }).filter(Boolean).join('');
      return rows ? `<div class="process-lines">${rows}</div>` : '';
    }
    function renderReasoningSteps(steps) {
      if (!Array.isArray(steps) || !steps.length) return '';
      return `<div class="reasoning-list">${steps.map(step => `
        <div class="reasoning-step">
          <div class="reasoning-label">${escapeHtml(step.label || '过程')}</div>
          <div class="reasoning-text">${escapeHtml(step.text || '')}</div>
        </div>
      `).join('')}</div>`;
    }
    function renderReasoningStepsTyped(steps, budget) {
      if (!Array.isArray(steps) || !steps.length) return '';
      const rows = steps.map(step => {
        const label = revealFromBudget(budget, step.label || '过程');
        const text = revealFromBudget(budget, step.text || '');
        if (!label && !text) return '';
        return `<div class="reasoning-step">
          <div class="reasoning-label">${label}</div>
          <div class="reasoning-text">${text}</div>
        </div>`;
      }).filter(Boolean).join('');
      return rows ? `<div class="reasoning-list">${rows}</div>` : '';
    }
    function renderSubtaskCards(items) {
      if (!Array.isArray(items) || !items.length) return '';
      return `<div class="subtask-list">${items.map(item => `
        <article class="subtask-card">
          <div class="subtask-head">
            <div class="subtask-index">阶段 ${escapeHtml(item.index || '')}</div>
            <div class="subtask-title">${escapeHtml(item.title || item.phase || '')}</div>
          </div>
          <div class="subtask-body">
            ${item.goal ? `<div class="section-summary">${escapeHtml(item.goal)}</div>` : ''}
            ${item.skill ? `<div class="subtask-row"><div class="subtask-label">技能</div><div class="subtask-value">${escapeHtml(compactLabel(item.skill))}</div></div>` : ''}
          </div>
        </article>
      `).join('')}</div>`;
    }
    function renderSubtaskCardsTyped(items, budget) {
      if (!Array.isArray(items) || !items.length) return '';
      const cards = items.map(item => {
        const index = revealFromBudget(budget, `阶段 ${item.index || ''}`);
        const title = revealFromBudget(budget, item.title || item.phase || '');
        const goal = revealFromBudget(budget, item.goal || '');
        const skillLabel = item.skill ? revealFromBudget(budget, '技能') : '';
        const skill = revealFromBudget(budget, compactLabel(item.skill || ''));
        const rows = skill || skillLabel ? `<div class="subtask-row">
          <div class="subtask-label">${skillLabel}</div>
          <div class="subtask-value">${skill}</div>
        </div>` : '';
        if (!index && !title && !goal && !rows) return '';
        return `<article class="subtask-card">
          <div class="subtask-head">
            <div class="subtask-index">${index}</div>
            <div class="subtask-title">${title}</div>
          </div>
          ${goal || rows ? `<div class="subtask-body">${goal ? `<div class="section-summary">${goal}</div>` : ''}${rows}</div>` : ''}
        </article>`;
      }).filter(Boolean).join('');
      return cards ? `<div class="subtask-list">${cards}</div>` : '';
    }
    function constraintClass(type) {
      const text = String(type || '');
      if (text.includes('逻辑')) return 'logical';
      if (text.includes('空间')) return 'spatial';
      if (text.includes('时序')) return 'temporal';
      return '';
    }
    function renderConstraintRulesTyped(rules, budget) {
      if (!Array.isArray(rules) || !rules.length) return '';
      const cards = rules.map(item => {
        const title = revealFromBudget(budget, item.type || '');
        const summary = revealFromBudget(budget, item.summary || '');
        const ruleRows = (item.rules || []).map(rule => {
          const line = revealFromBudget(budget, rule || '');
          return line ? `<div class="constraint-rule">${line}</div>` : '';
        }).filter(Boolean).join('');
        const usedFor = revealFromBudget(budget, item.used_for || '');
        if (!title && !summary && !ruleRows && !usedFor) return '';
        return `<article class="constraint-card">
          <div class="constraint-head ${constraintClass(item.type)}">${title}</div>
          <div class="constraint-body">
            ${summary ? `<div class="constraint-summary">${summary}</div>` : ''}
            ${ruleRows}
            ${usedFor ? `<div class="thought-next">${usedFor}</div>` : ''}
          </div>
        </article>`;
      }).filter(Boolean).join('');
      return cards ? `<div class="constraint-grid">${cards}</div>` : '';
    }
    function renderDecompositionSectionsTyped(sections, budget) {
      if (!Array.isArray(sections) || !sections.length) return '';
      const html = sections.map(section => {
        const range = revealFromBudget(budget, section.range || '');
        const title = revealFromBudget(budget, section.title || '');
        const summary = revealFromBudget(budget, section.summary || '');
        const flowCount = Array.isArray(section.skill_flow) && section.skill_flow.length
          ? revealFromBudget(budget, `包含 ${section.skill_flow.length} 个技能阶段`)
          : '';
        const cards = flowCount ? '' : renderSubtaskCardsTyped(section.cards || [], budget);
        if (!range && !title && !summary && !flowCount && !cards) return '';
        return `<section class="decomposition-section">
          <div class="section-head">
            <div class="section-title">${range}${range && title ? ' · ' : ''}${title}</div>
            ${summary ? `<div class="section-summary">${summary}</div>` : ''}
          </div>
          ${flowCount ? `<div class="subtask-card"><div class="subtask-body"><div class="section-summary">${flowCount}</div></div></div>` : ''}
          ${cards}
        </section>`;
      }).filter(Boolean).join('');
      return html ? `<div class="decomposition-list">${html}</div>` : '';
    }
    function renderLeRobotDataset(dataset) {
      if (!dataset) return '';
      const rows = (dataset.directories || []).map(item => `
        <div class="dataset-row">
          <div class="dataset-path">${escapeHtml(item.path || '')}</div>
          <div class="dataset-desc">${escapeHtml(item.content || '')}</div>
        </div>
      `).join('');
      const features = (dataset.features || []).map(item => `<span class="feature-tag">${escapeHtml(item)}</span>`).join('');
      return `<div class="dataset-card">
        <div class="dataset-title">${escapeHtml(dataset.dataset_format || 'LeRobotDataset v3.0')} 数据收集结构</div>
        <div class="dataset-root">${escapeHtml(dataset.root || '')}</div>
        <div class="dataset-tree">${rows}</div>
        <div class="feature-tags">${features}</div>
      </div>`;
    }
    async function getJson(url, options) {
      const response = await fetch(url, options);
      if (!response.ok) throw new Error(await response.text());
      return await response.json();
    }
    async function loadInventory() {
      const scene = document.getElementById('sceneProfile').value || 'taoyuan_grscenes_tabletop';
      const template = document.getElementById('template').value || 'ur5e_assembly_template';
      const url = `/api/inventory?scene_profile=${encodeURIComponent(scene)}&template=${encodeURIComponent(template)}`;
      state.inventory = await getJson(url);
      renderInventory();
    }
    function initSelectors(inventory) {
      const sceneSelect = document.getElementById('sceneProfile');
      const templateSelect = document.getElementById('template');
      sceneSelect.innerHTML = inventory.scene_profiles.map(item =>
        `<option value="${escapeHtml(item)}"${item === inventory.scene_profile ? ' selected' : ''}>${escapeHtml(item)}</option>`
      ).join('');
      templateSelect.innerHTML = inventory.templates.map(item =>
        `<option value="${escapeHtml(item.name)}"${item.name === inventory.selected_template ? ' selected' : ''}>${escapeHtml(item.display_name || item.name)}</option>`
      ).join('');
    }
    function renderInventory() {
      const panel = document.getElementById('assetPanel');
      if (!state.inventory) {
        panel.innerHTML = '<div class="empty">暂无资产</div>';
        return;
      }
      const libraries = state.inventory.asset_libraries || {};
      const sectionNames = [
        ['robots', '机器人库'],
        ['scenes', '场景库'],
        ['objects', '交互物体库']
      ];
      panel.innerHTML = sectionNames.map(([key, title]) => {
        const items = libraries[key] || [];
        return `<section class="asset-section">
          <div class="asset-section-title"><span>${title}</span><span class="badge">${items.length}</span></div>
          <div class="asset-grid">
            ${items.map((item, index) => {
              const image = item.preview ? `/api/image?path=${encodeURIComponent(item.preview)}` : '';
              return `<article class="asset-card" title="${escapeHtml(item.asset_path || '')}">
                ${image ? `<img class="asset-thumb" src="${image}" alt="${escapeHtml(item.name || title)}" loading="lazy" decoding="async">` : `<div class="asset-thumb"></div>`}
                <div>
                  <div class="asset-title">${escapeHtml(item.name || '')}</div>
                  <div class="asset-desc">${escapeHtml(item.description || '')}</div>
                  <div class="asset-tags">${(item.tags || []).map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join('')}</div>
                </div>
              </article>`;
            }).join('')}
          </div>
        </section>`;
      }).join('');
    }
    function payloadFromForm() {
      return {
        task: document.getElementById('task').value,
        template: document.getElementById('template').value,
        scene_profile: document.getElementById('sceneProfile').value,
        model: document.getElementById('model').value,
        temperature: Number(document.getElementById('temperature').value || 0.2),
        manual_demo: document.getElementById('manualDemo').checked,
        manual_thinking_delay: 3.0,
        mock_llm: document.getElementById('mockLlm').checked,
        run_simulation: document.getElementById('runSimulation').checked,
        export_lerobot: false,
        headless: document.getElementById('headless').checked,
        record_live_video: document.getElementById('recordVideo').checked,
        num_demos: Number(document.getElementById('numDemos').value || 1),
        max_trials: Number(document.getElementById('maxTrials').value || 1),
        start_seed: Number(document.getElementById('startSeed').value || 0),
        max_runtime_replans: Number(document.getElementById('maxReplans').value || 1)
      };
    }
    async function startJob() {
      document.getElementById('startBtn').disabled = true;
      resetTypewriters();
      const response = await getJson('/api/jobs', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payloadFromForm())
      });
      state.jobId = response.job_id;
      document.getElementById('jobId').textContent = state.jobId;
      if (state.poll) clearInterval(state.poll);
      state.poll = setInterval(pollJob, 1000);
      await pollJob();
    }
    async function startManualExample() {
      document.getElementById('manualDemo').checked = true;
      document.getElementById('mockLlm').checked = false;
      document.getElementById('runSimulation').checked = false;
      document.getElementById('template').value = 'ur5e_assembly_template';
      document.getElementById('task').value = '基于 UR5e assembly Template 生成新任务 fabrica_plumbers_block_ur5e_assembly，在线调用 LLM Planner 生成 Menu、Asset Annotation、子任务、技能序列，并准备仿真执行。';
      await startJob();
    }
    async function pollJob() {
      if (!state.jobId) return;
      const job = await getJson(`/api/jobs/${state.jobId}`);
      state.job = job;
      renderJob();
      if (job.status === 'succeeded' || job.status === 'failed') {
        clearInterval(state.poll);
        state.poll = null;
        document.getElementById('startBtn').disabled = false;
      }
    }
    async function startSimulation() {
      const button = document.getElementById('simBtn');
      button.disabled = true;
      const response = await getJson('/api/simulation/start', {method: 'POST'});
      state.simJobId = response.job_id;
      if (state.simPoll) clearInterval(state.simPoll);
      state.simPoll = setInterval(pollSimulation, 1000);
      await pollSimulation();
    }
    async function pollSimulation() {
      if (!state.simJobId) return;
      const simJob = await getJson(`/api/simulation/${state.simJobId}`);
      state.simJob = simJob;
      renderJob();
      if (simJob.status === 'succeeded' || simJob.status === 'failed') {
        clearInterval(state.simPoll);
        state.simPoll = null;
        document.getElementById('simBtn').disabled = false;
      }
    }
    function nowText() {
      return new Date().toLocaleString('zh-CN', {hour12: false});
    }
    function addRecordingEvent(type, payload = {}) {
      state.recordingEvents.push({type, payload, time: nowText()});
      renderTrace();
    }
    function setRecordButton(active, busy = false) {
      const button = document.getElementById('recordBtn');
      button.disabled = busy;
      button.classList.toggle('active', active);
      button.textContent = active ? '停止录制' : busy ? '保存中' : '开始录制';
    }
    function createScreenRecorder(stream) {
      const candidates = [
        'video/mp4;codecs=avc1.42E01E',
        'video/mp4;codecs=h264',
        'video/mp4',
        'video/webm;codecs=vp9',
        'video/webm;codecs=vp8',
        'video/webm'
      ];
      for (const mimeType of candidates) {
        if (MediaRecorder.isTypeSupported && !MediaRecorder.isTypeSupported(mimeType)) continue;
        try {
          return new MediaRecorder(stream, {mimeType});
        } catch (error) {
          // Try the next browser-supported container.
        }
      }
      return new MediaRecorder(stream);
    }
    async function startRecording() {
      if (state.recording) {
        stopRecording();
        return;
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia || typeof MediaRecorder === 'undefined') {
        addRecordingEvent('recording_failed', {error: '当前浏览器不支持屏幕录制。'});
        return;
      }
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          width: {ideal: 1920},
          height: {ideal: 1080},
          frameRate: {ideal: 60, max: 60},
          displaySurface: 'monitor'
        },
        audio: false
      });
      const trackSettings = stream.getVideoTracks()[0]?.getSettings?.() || {};
      const chunks = [];
      const recorder = createScreenRecorder(stream);
      const recordingMimeType = recorder.mimeType || 'video/webm';
      state.recording = {stream, recorder, chunks, mimeType: recordingMimeType, settings: trackSettings};
      recorder.ondataavailable = event => {
        if (event.data && event.data.size) chunks.push(event.data);
      };
      recorder.onstop = async () => {
        setRecordButton(false, true);
        try {
          stream.getTracks().forEach(track => track.stop());
          const uploadType = state.recording?.mimeType || recorder.mimeType || chunks[0]?.type || 'video/webm';
          const blob = new Blob(chunks, {type: uploadType});
          const saved = await getJson('/api/recordings', {
            method: 'POST',
            headers: {'Content-Type': uploadType},
            body: blob
          });
          state.recordingSaved = saved;
          addRecordingEvent('recording_saved', saved);
        } catch (error) {
          addRecordingEvent('recording_failed', {error: error.message});
        } finally {
          state.recording = null;
          setRecordButton(false, false);
        }
      };
      stream.getVideoTracks().forEach(track => {
        track.onended = () => {
          if (state.recording?.recorder?.state === 'recording') stopRecording();
        };
      });
      recorder.start(1000);
      setRecordButton(true, false);
      addRecordingEvent('recording_started', {
        mime_type: recordingMimeType,
        width: trackSettings.width,
        height: trackSettings.height,
        frame_rate: trackSettings.frameRate,
        display_surface: trackSettings.displaySurface
      });
    }
    function stopRecording() {
      const recording = state.recording;
      if (!recording) return;
      if (recording.recorder.state !== 'inactive') {
        recording.recorder.stop();
      }
    }
    function eventTypes() {
      return new Set((state.job?.events || []).map(event => event.type));
    }
    function renderStages() {
      const stageEl = document.getElementById('stages');
      if (!stageEl) return;
      const types = eventTypes();
      const failed = state.job?.status === 'failed';
      stageEl.innerHTML = stages.map(([key, name, events]) => {
        let cls = 'stage';
        let text = 'waiting';
        if (events.some(item => types.has(item))) { cls += ' done'; text = 'done'; }
        if (key === 'demo' && types.has('demo_started') && !types.has('demo_completed')) { cls = 'stage active'; text = 'running'; }
        if (failed && (key === 'demo' || key === 'check')) { cls += ' fail'; text = 'check trace'; }
        return `<div class="${cls}"><div class="stage-name">${name}</div><div class="stage-state">${text}</div></div>`;
      }).join('');
    }
    function renderTrace() {
      const trace = document.getElementById('trace');
      const events = state.job?.events || [];
      const simEvents = state.simJob?.events || [];
      const recordingEvents = state.recordingEvents || [];
      document.getElementById('eventCount').textContent = String(events.length + simEvents.length + recordingEvents.length);
      const previousScrollTop = trace.scrollTop;
      const previousScrollHeight = trace.scrollHeight;
      const distanceFromBottom = previousScrollHeight - previousScrollTop - trace.clientHeight;
      const shouldFollowNewMessages = distanceFromBottom < 56;
      if (!events.length && !simEvents.length && !recordingEvents.length) {
        trace.innerHTML = '<div class="empty">No events</div>';
        return;
      }
      const messages = [];
      const request = state.job?.request || {};
      const finishedStages = new Set(events.filter(event => event.type === 'manual_thinking_finished').map(event => event.payload?.stage));
      if (state.job) {
        messages.push({
          role: 'user',
          time: state.job?.created_at || '',
          html: `<div>${escapeHtml(request.task || '新任务')}</div>`
        });
      }
      let blockedByTypewriter = false;
      for (const [eventIndex, event] of events.entries()) {
        const payload = event.payload || {};
        if (event.type === 'job_started') continue;
        if (event.type === 'manual_demo_selected') {
          messages.push({
            role: 'agent',
            time: event.time,
            html: `<div class="thought-title">连接 LLM Planner</div><div>根据 Reference Template 生成新任务 Menu、Asset Annotation 和可执行装配分解。</div>`
          });
          continue;
        }
        if (event.type === 'manual_demo_bundle_written') {
          continue;
        }
        if (event.type === 'manual_thinking_started') {
          if (finishedStages.has(payload.stage)) continue;
          const thinkingTitle = payload.title || payload.stage || '当前阶段';
          messages.push({
            role: 'thinking',
            time: event.time,
            html: `<div class="thought-summary">我正在进行「${escapeHtml(thinkingTitle)}」。</div>
              <div class="thinking-status"><span class="spinner"></span><span>LLM 分析中</span></div>`
          });
          continue;
        }
        if (event.type === 'manual_thinking_finished') {
          messages.push({
            role: 'finished',
            time: event.time,
            html: `<span>${escapeHtml(payload.message || '分析完成')}</span>`
          });
          continue;
        }
        if (event.type.startsWith('manual_') && payload.thinking_process) {
          const processLines = payload.visible_process_lines || payload.process_lines || [];
          const reasoningSteps = processLines.length ? [] : (payload.visible_reasoning_steps || payload.reasoning_steps || []);
          const output = payload.output || {};
          const constraintRules = output.constraint_rules || payload.constraint_rules || [];
          const decompositionSections = payload.output?.decomposition_sections || payload.decomposition_sections || [];
          const subtaskCards = payload.output?.subtask_cards || payload.subtask_cards || [];
          const summaryText = cleanResultSummary(payload.thinking_process);
          const typewriterKey = `${state.job?.id || state.jobId || 'job'}:${eventIndex}:${event.type}:${payload.stage || ''}`;
          const typedCard = typewriterBudget(
            typewriterKey,
            resultSegmentsForPayload(summaryText, processLines, reasoningSteps, output, decompositionSections.length ? [] : subtaskCards, payload.next_step || '')
          );
          const summaryHtml = revealFromBudget(typedCard, summaryText);
          const processHtml = renderProcessLinesTyped(processLines, typedCard);
          const reasoningHtml = renderReasoningStepsTyped(reasoningSteps, typedCard);
          const constraintsHtml = renderConstraintRulesTyped(constraintRules, typedCard);
          const sectionHtml = renderDecompositionSectionsTyped(decompositionSections, typedCard);
          const subtaskHtml = decompositionSections.length ? '' : renderSubtaskCardsTyped(subtaskCards, typedCard);
          const nextHtml = revealFromBudget(typedCard, payload.next_step || '');
          messages.push({
            role: 'agent',
            time: event.time,
            html: `<div class="thought-summary">${summaryHtml}</div>
              ${processHtml}
              ${reasoningHtml}
              ${constraintsHtml}
              ${sectionHtml}
              ${subtaskHtml}
              ${typedCard.done ? renderLeRobotDataset(output.lerobot_dataset) : ''}
              ${nextHtml ? `<div class="thought-next">${nextHtml}</div>` : ''}
              ${typedCard.done ? `<details><summary>技术细节</summary><pre>${escapeHtml(asJson({input: payload.input, output: payload.output}))}</pre></details>` : ''}`
          });
          if (!typedCard.done) {
            blockedByTypewriter = true;
            break;
          }
          continue;
        }
        if (event.type === 'manual_skill_steps_ready') {
          messages.push({
            role: 'agent',
            time: event.time,
            html: `<div class="thought-title">技能步骤已格式化</div><div>已生成 ${escapeHtml(payload.count ?? 0)} 条技能步骤，包含执行技能、操作臂、操作对象、状态约束和目标位置。</div>`
          });
          continue;
        }
        if (event.type === 'manual_checker_completed') {
          messages.push({
            role: payload.ok ? 'agent' : 'system',
            time: event.time,
            html: `<div class="thought-title">映射校验${payload.ok ? '通过' : '未通过'}</div><div>${payload.ok ? '机器人、对象、目标、phase 和成功条件都能被索引。' : escapeHtml(asJson(payload.errors || []))}</div>`
          });
          continue;
        }
        if (event.type === 'job_succeeded') {
          const result = state.job?.result || {};
          const artifacts = result.bundle_paths || {};
          const labels = {
            manual_demo_report: '流程报告',
            manual_skill_steps: '技能步骤',
            manual_reasoning_trace: '规划过程',
            recipe: '生成任务文件'
          };
          const link = key => artifacts[key] ? `<a href="/api/file?path=${encodeURIComponent(artifacts[key])}" target="_blank" rel="noreferrer">${labels[key] || key}</a>` : '';
          const links = [link('manual_demo_report'), link('manual_skill_steps'), link('manual_reasoning_trace'), link('recipe')].filter(Boolean).join(' · ');
          messages.push({
            role: 'agent',
            time: event.time,
            html: `<div class="thought-title">新任务生成完成</div><div>fabrica_plumbers_block_ur5e_assembly 的 Menu、Annotation、技能步骤和仿真入口已经生成。</div>${links ? `<div class="thought-next">${links}</div>` : ''}`
          });
          continue;
        }
        if (event.type === 'job_failed') {
          messages.push({
            role: 'system',
            time: event.time,
            html: `<div class="thought-title">失败</div><div>${escapeHtml(payload.error || 'Unknown error')}</div>`
          });
          continue;
        }
        if (event.type === 'planner_request_prepared' || event.type === 'plan_generated' || event.type === 'checker_completed' || event.type === 'bundle_written') {
          messages.push({
            role: 'agent',
            time: event.time,
            html: `<div class="thought-title">${escapeHtml(event.type)}</div><details open><summary>查看事件数据</summary><pre>${escapeHtml(asJson(payload))}</pre></details>`
          });
        }
      }
      if (!blockedByTypewriter && simEvents.length) {
        const started = simEvents.find(event => event.type === 'simulation_started');
        const completed = [...simEvents].reverse().find(event => event.type === 'simulation_completed');
        const failed = [...simEvents].reverse().find(event => event.type === 'simulation_failed');
        const outputLines = simEvents
          .filter(event => event.type === 'simulation_output')
          .map(event => event.payload?.line)
          .filter(Boolean);
        if (started) {
          const payload = started.payload || {};
          messages.push({
            role: 'agent',
            time: started.time,
            html: `<div class="thought-title">启动仿真</div>
              <div>正在执行官方 Isaac Sim 轨迹回放脚本。</div>
              <div class="thought-next">${escapeHtml(payload.script || 'render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh')}</div>`
          });
        }
        if (outputLines.length) {
          messages.push({
            role: 'system',
            time: simEvents[simEvents.length - 1]?.time || '',
            html: `<div class="thought-title">仿真日志</div><pre class="sim-log">${escapeHtml(outputLines.slice(-160).join('\n'))}</pre>`
          });
        }
        if (completed) {
          const payload = completed.payload || {};
          const dataset = payload.lerobot_dataset || {};
          messages.push({
            role: 'agent',
            time: completed.time,
            html: `<div class="thought-title">仿真完成，开始整理 LeRobot 数据</div>
              <div>脚本执行结束。视频只是视觉模态之一，系统会按 LeRobotDataset v3.0 组织多相机视频、机械臂关节轨迹、本体感受态、动作、物体状态和 episode metadata。</div>
              <div class="thought-next">视觉回放：${escapeHtml(payload.video || '')}${payload.video_exists ? '' : '（尚未检测到文件）'}</div>
              ${renderLeRobotDataset(dataset)}`
          });
        }
        if (failed) {
          messages.push({
            role: 'system',
            time: failed.time,
            html: `<div class="thought-title">仿真失败</div><div>${escapeHtml(failed.payload?.error || 'Unknown error')}</div>`
          });
        }
      }
      if (!blockedByTypewriter && recordingEvents.length) {
        for (const event of recordingEvents) {
          if (event.type === 'recording_started') {
            const payload = event.payload || {};
            messages.push({
              role: 'agent',
              time: event.time,
              html: `<div class="thought-title">录制已开启</div>
                <div>当前选择的屏幕或窗口正在高清录制，停止后会保存为 MP4。</div>
                <div class="recording-tip">如果要同时录制浏览器面板和 Isaac Sim，请在浏览器弹出的共享选择里选择“整个屏幕”，并把浏览器和 Isaac 窗口都放在同一个屏幕上。浏览器安全策略不允许一次自动抓取两个独立窗口。</div>
                ${(payload.width && payload.height) ? `<div class="mono">resolution: ${escapeHtml(payload.width)}×${escapeHtml(payload.height)} @ ${escapeHtml(payload.frame_rate || '?')}fps</div>` : ''}
                ${payload.mime_type ? `<div class="mono">capture: ${escapeHtml(payload.mime_type)}</div>` : ''}`
            });
          }
          if (event.type === 'recording_saved') {
            const payload = event.payload || {};
            messages.push({
              role: 'agent',
              time: event.time,
              html: `<div class="thought-title">录制已保存</div>
                <div>面板和仿真过程录制文件已保存为 MP4。</div>
                ${payload.filename ? `<div class="mono">${escapeHtml(payload.filename)}</div>` : ''}
                ${payload.media_url ? `<a class="recording-link" href="${escapeHtml(payload.media_url)}" target="_blank" rel="noreferrer">打开录制文件</a>` : ''}`
            });
          }
          if (event.type === 'recording_failed') {
            messages.push({
              role: 'system',
              time: event.time,
              html: `<div class="thought-title">录制失败</div><div>${escapeHtml(event.payload?.error || 'Unknown error')}</div>`
            });
          }
        }
      }
      trace.innerHTML = messages.map(message => `
        <div class="chat-message ${message.role}">
          <div class="chat-meta">${message.role === 'user' ? '用户' : message.role === 'thinking' ? 'RoboAssembly LLM' : 'RoboAssembly'}</div>
          <div class="bubble">${message.html}</div>
        </div>
      `).join('');
      if (shouldFollowNewMessages) {
        trace.scrollTop = trace.scrollHeight;
      } else {
        trace.scrollTop = previousScrollTop;
      }
    }
    function badgeForStatus(status) {
      const badge = document.getElementById('jobBadge');
      badge.textContent = status || 'idle';
      badge.className = 'badge ' + (status === 'succeeded' ? 'ok' : status === 'failed' ? 'fail' : status === 'running' ? 'warn' : '');
    }
    function renderPlan(result) {
      if (!result) return '<div class="empty">No result yet</div>';
      const plan = result.plan || {};
      const summary = result.recipe_summary || {};
      const manualTrace = result.manual_reasoning_trace || [];
      return `<div class="kv">
        <div class="k">task</div><div class="mono">${escapeHtml(plan.task_name || '')}</div>
        <div class="k">Reference Template</div><div class="mono">${escapeHtml(plan.selected_template || '')}</div>
        <div class="k">source</div><div>${escapeHtml(plan.source || '')}</div>
        <div class="k">checker</div><div>${escapeHtml(String(result.check_result?.ok ?? ''))}</div>
        <div class="k">生成任务</div><div>${summary.phase_count || 0} execution units, ${summary.object_count || 0} objects, ${summary.target_count || 0} targets</div>
      </div>
      <div style="height:12px"></div>
      ${manualTrace.length ? `<pre>${escapeHtml(asJson(manualTrace))}</pre><div style="height:12px"></div>` : ''}
      <pre>${escapeHtml(asJson(plan))}</pre>`;
    }
    function renderSkills(result) {
      if (!result) return '<div class="empty">No skills yet</div>';
      const skills = result.manual_skill_steps || result.plan?.skills || [];
      const primitive = result.primitive_plan || [];
      if (!skills.length && !primitive.length) return '<div class="empty">No explicit skill plan</div>';
      return `<div class="rows">
        ${skills.map((skill, idx) => `<div class="row"><div class="mono">skill ${idx + 1}</div><div><pre>${escapeHtml(asJson(skill))}</pre></div></div>`).join('')}
        ${primitive.map((item, idx) => `<div class="row"><div class="mono">phase ${idx + 1}</div><div><pre>${escapeHtml(asJson(item))}</pre></div></div>`).join('')}
      </div>`;
    }
    function renderFiles(result) {
      if (!result) return '<div class="empty">No files yet</div>';
      const artifacts = result.artifacts || [];
      if (!artifacts.length) return '<div class="empty">No artifacts</div>';
      return `<div class="file-list">${artifacts.map(item => `
        <div class="file-row">
          <div class="mono">${escapeHtml(item.label)}</div>
          <div class="mono">${escapeHtml(item.path)}</div>
          ${item.kind === 'file' ? `<button class="secondary file-preview" data-path="${escapeHtml(item.path)}">Preview</button>` : `<span class="badge">${escapeHtml(item.kind || 'path')}</span>`}
        </div>`).join('')}</div>`;
    }
    function renderRuntime(result) {
      if (!result) return '<div class="empty">No runtime data</div>';
      return `<pre>${escapeHtml(asJson({
        demo_output_dir: result.demo_output_dir,
        demo_command: result.demo_command,
        runtime_feedback: result.runtime_feedback
      }))}</pre>`;
    }
    function renderTab() {
      if (!document.getElementById('tabContent')) return;
      const result = state.job?.result;
      const content = document.getElementById('tabContent');
      if (state.tab === 'plan') content.innerHTML = renderPlan(result);
      if (state.tab === 'skills') content.innerHTML = renderSkills(result);
      if (state.tab === 'files') content.innerHTML = renderFiles(result);
      if (state.tab === 'runtime') content.innerHTML = renderRuntime(result);
      if (state.tab === 'preview') content.innerHTML = state.preview ? `<pre>${escapeHtml(state.preview)}</pre>` : '<div class="empty">Select a file</div>';
    }
    async function previewFile(path) {
      const response = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
      state.preview = response.ok ? await response.text() : await response.text();
      state.tab = 'preview';
      document.querySelectorAll('.tab').forEach(tab => tab.classList.toggle('active', tab.dataset.tab === state.tab));
      renderTab();
    }
    function renderJob() {
      badgeForStatus(state.job?.status || state.simJob?.status);
      renderStages();
      renderTrace();
    }
    document.getElementById('startBtn').addEventListener('click', () => startJob().catch(error => {
      document.getElementById('startBtn').disabled = false;
      alert(error.message);
    }));
    document.getElementById('manualBtn').addEventListener('click', () => startManualExample().catch(error => {
      document.getElementById('startBtn').disabled = false;
      alert(error.message);
    }));
    document.getElementById('simBtn').addEventListener('click', () => startSimulation().catch(error => {
      document.getElementById('simBtn').disabled = false;
      alert(error.message);
    }));
    document.getElementById('recordBtn').addEventListener('click', () => startRecording().catch(error => {
      setRecordButton(false, false);
      addRecordingEvent('recording_failed', {error: error.message});
    }));
    if (document.getElementById('tabs')) document.getElementById('tabs').addEventListener('click', event => {
      const tab = event.target.closest('.tab');
      if (!tab) return;
      state.tab = tab.dataset.tab;
      document.querySelectorAll('.tab').forEach(item => item.classList.toggle('active', item === tab));
      renderTab();
    });
    if (document.getElementById('tabContent')) document.getElementById('tabContent').addEventListener('click', event => {
      const button = event.target.closest('.file-preview');
      if (!button) return;
      previewFile(button.dataset.path).catch(error => {
        state.preview = error.message;
        state.tab = 'preview';
        document.querySelectorAll('.tab').forEach(tab => tab.classList.toggle('active', tab.dataset.tab === state.tab));
        renderTab();
      });
    });
    document.getElementById('template').addEventListener('change', () => loadInventory().catch(console.error));
    document.getElementById('sceneProfile').addEventListener('change', () => loadInventory().catch(console.error));
    renderStages();
    getJson('/api/inventory').then(inventory => {
      state.inventory = inventory;
      initSelectors(inventory);
      document.getElementById('mockLlm').checked = false;
      renderInventory();
    }).catch(error => {
      document.getElementById('assetPanel').innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
    });
  </script>
</body>
</html>
"""


if __name__ == '__main__':
    main()
