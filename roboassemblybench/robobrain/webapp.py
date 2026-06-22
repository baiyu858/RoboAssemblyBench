from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
            template=str(payload.get('template') or DEFAULT_TEMPLATE).strip() or None,
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
        return _jsonable(self.__dict__)


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


def _asset_catalog(limit: int = 250) -> list[dict[str, Any]]:
    assets_root = BENCHMARK_ROOT / 'assets'
    suffixes = {'.usd', '.usda', '.usdc', '.zip', '.yaml', '.json', '.md'}
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


def _existing_image(*candidates: str | Path | None) -> str | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists() and path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
            return str(path.resolve())
    return None


def _preview_image(index: int = 0) -> str | None:
    candidates = [
        REPO_ROOT / 'outputs' / 'fabrica_plumbers_block_ur5e' / 'plumbers_block_ur5e_scene_preview_check_frames' / 'rgb_00000.png',
        REPO_ROOT / 'outputs' / 'fabrica_plumbers_block_ur5e' / 'plumbers_block_ur5e_scene_preview_frames' / 'rgb_00074.png',
        REPO_ROOT / 'outputs' / 'fabrica_plumbers_block_ur5e' / 'plumbers_block_ur5e_scene_preview_frames' / 'rgb_00066.png',
        REPO_ROOT / 'outputs' / 'fabrica_plumbers_block_ur5e' / 'plumbers_block_ur5e_scene_preview_frames' / 'rgb_00091.png',
        REPO_ROOT / 'outputs' / 'fabrica_plumbers_block_ur5e' / 'plumbers_block_ur5e_scene_preview_frames' / 'rgb_00102.png',
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return str(existing[index % len(existing)])


def _asset_libraries(selected_recipe: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    asset_refs = selected_recipe.get('asset_references', [])
    previews = {
        'ur5e': _existing_image(
            'roboassemblybench/assets/Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001/assets/isaac_official/Isaac/Robots/UniversalRobots/ur5e/configuration/.thumbs/256x256/ur5e_physics.usd.png',
            _preview_image(0),
        ),
        'franka': _existing_image(
            'IsaacLab/docs/source/_static/tasks/manipulation/franka_reach.jpg',
            'third_part/Fabrica/isaacgym/isaacgym/docs/_images/example_franka_attractor.png',
            _preview_image(1),
        ),
        'kuka': _existing_image(
            'IsaacLab/docs/source/_static/tasks/manipulation/kuka_allegro_reorient.jpg',
            'third_part/Fabrica/isaacgym/isaacgym/docs/_images/example_kuka_bin.png',
            _preview_image(2),
        ),
        'factory_cell': _existing_image(
            'outputs/fabrica_plumbers_block_ur5e/plumbers_block_ur5e_scene_preview_check_frames/rgb_00000.png',
            _preview_image(0),
        ),
        'warehouse': _existing_image(
            'outputs/fabrica_plumbers_block_ur5e/plumbers_block_ur5e_scene_preview_frames/rgb_00074.png',
            'static/benchmark.png',
            _preview_image(1),
        ),
        'tabletop': _existing_image(
            'roboassemblybench/assets/Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001/assets/isaac_official/Isaac/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/.thumbs/256x256/SM_HeavyDutyPackingTable_C02_01.usd.png',
            'outputs/fabrica_plumbers_block_ur5e/plumbers_block_ur5e_scene_preview_frames/rgb_00066.png',
            _preview_image(2),
        ),
        'factory_props': _existing_image(
            'roboassemblybench/assets/Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001/assets/isaac_official/Isaac/Props/PackingTable/props/SM_Crate_A08_Blue_01/.thumbs/256x256/SM_Crate_A08_Blue_01.usd.png',
            'outputs/fabrica_plumbers_block_ur5e/plumbers_block_ur5e_scene_preview_frames/rgb_00091.png',
            _preview_image(3),
        ),
        'plumbers_block': _existing_image(
            'outputs/fabrica_plumbers_block_ur5e/plumbers_block_ur5e_scene_preview_frames/rgb_00102.png',
            'third_part/Fabrica/logs/codex_plumbers_block_ur5e_official/plumbers_block/precedence.png',
            _preview_image(4),
        ),
        'fixture': _existing_image(
            'third_part/Fabrica/logs/codex_plumbers_block_ur5e_official/plumbers_block/fixture/fixture.png',
            'outputs/fabrica_plumbers_block_ur5e/plumbers_block_ur5e_scene_preview_frames/rgb_00066.png',
            _preview_image(2),
        ),
        'optical_board': _existing_image(
            'outputs/fabrica_plumbers_block_ur5e/plumbers_block_ur5e_scene_preview_frames/rgb_00012.png',
            'roboassemblybench/assets/Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001/assets/isaac_official/Isaac/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/.thumbs/256x256/SM_HeavyDutyPackingTable_C02_01.usd.png',
            _preview_image(3),
        ),
        'assembled': _existing_image(
            'third_part/Fabrica/logs/codex_plumbers_block_ur5e_official/plumbers_block/precedence.png',
            'outputs/fabrica_plumbers_block_ur5e/plumbers_block_ur5e_scene_preview_frames/rgb_00074.png',
            _preview_image(1),
        ),
    }

    def ref_path(name: str, fallback: str = '') -> str:
        lowered = name.lower()
        for item in asset_refs:
            haystack = f"{item.get('name', '')} {item.get('path', '')}".lower()
            if lowered in haystack:
                return str(item.get('path') or fallback)
        return fallback

    return {
        'robots': [
            {
                'name': 'UR5e + Robotiq 2F-85',
                'description': '六轴协作臂，适合抓取、搬运、对准和插装准备。',
                'asset_path': ref_path('ur5e', 'roboassemblybench/assets/Fabrica/.../ur5e_robotiq_2f85_task.usda'),
                'preview': previews['ur5e'],
                'tags': ['6-DoF', 'gripper', 'assembly'],
            },
            {
                'name': 'Franka Panda',
                'description': '通用双臂装配基线，适合精细抓取和桌面操作。',
                'asset_path': ref_path('franka', 'roboassemblybench/assets/planning/franka_mplib.urdf'),
                'preview': previews['franka'],
                'tags': ['7-DoF', 'dual-arm', 'baseline'],
            },
            {
                'name': 'KUKA iiwa',
                'description': '工业协作臂候选，可用于后续扩展工厂装配任务。',
                'asset_path': 'Isaac Sim / Omniverse industrial robot asset candidate',
                'preview': previews['kuka'],
                'tags': ['industrial', 'candidate'],
            },
        ],
        'scenes': [
            {
                'name': 'Factory Cell',
                'description': '本地工厂单元场景，适合快速预览和调试。',
                'asset_path': ref_path('factory_cell', str(BENCHMARK_ROOT / 'scenes' / 'usd' / 'factory_cell.usda')),
                'preview': previews['factory_cell'],
                'tags': ['factory', 'offline'],
            },
            {
                'name': 'Warehouse With Forklifts',
                'description': '带叉车和仓储背景的工厂环境。',
                'asset_path': ref_path('warehouse', '${ISAAC_ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd'),
                'preview': previews['warehouse'],
                'tags': ['warehouse', 'forklift'],
            },
            {
                'name': 'Tabletop Workcell',
                'description': '桌面装配工作区，适合夹具、零件和机械臂协作。',
                'asset_path': 'roboassemblybench/scenes/profiles/taoyuan_grscenes_tabletop.yaml',
                'preview': previews['tabletop'],
                'tags': ['table', 'workcell'],
            },
            {
                'name': 'Factory Props',
                'description': '背景物件集合，可包含车辆、货架、箱体和工厂辅助物。',
                'asset_path': 'roboassemblybench/assets/IsaacUSD',
                'preview': previews['factory_props'],
                'tags': ['props', 'vehicle', 'background'],
            },
        ],
        'objects': [
            {
                'name': 'Plumbers Block 2',
                'description': '当前任务的主要可操作零件，右臂需要抓取、重定向并移动到 staging 位。',
                'asset_path': ref_path('fabrica_plumbers_block_2'),
                'preview': previews['plumbers_block'],
                'tags': ['tracked', 'rigid', 'target'],
            },
            {
                'name': 'Fixture Tray',
                'description': '用于支撑和定位 plumbers-block 零件的夹具。',
                'asset_path': ref_path('fixture'),
                'preview': previews['fixture'],
                'tags': ['fixture', 'support'],
            },
            {
                'name': 'Optical Board',
                'description': 'Fabrica 工作台基底，可作为装配支撑面。',
                'asset_path': ref_path('optical_board'),
                'preview': previews['optical_board'],
                'tags': ['board', 'tabletop'],
            },
            {
                'name': 'Assembled Preview',
                'description': '目标装配状态参考模型，用于后续任务规划和成功状态说明。',
                'asset_path': ref_path('assembled'),
                'preview': previews['assembled'],
                'tags': ['reference', 'assembly'],
            },
        ],
    }


def _recipe_summary(name: str, recipe: dict[str, Any]) -> dict[str, Any]:
    metadata = recipe.get('metadata') or {}
    return {
        'name': name,
        'task_name': recipe.get('task_name', name),
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
                'name': name,
                'prompt': compact.get('prompt') or compact.get('task_description') or '',
                'robots': compact.get('robots', []),
                'object_count': len(compact.get('objects', [])),
                'target_count': len(compact.get('targets', [])),
                'phase_count': len(compact.get('phases', [])),
                'tags': compact.get('tags', []),
            }
        )

    template_name = selected_template if selected_template in inventory.recipes else inventory.best_template_for(selected_template or '')
    selected = _recipe_summary(template_name, inventory.recipes[template_name])
    return {
        'scene_profile': scene_profile or 'raw',
        'scene_profiles': _scene_profiles(),
        'default_template': DEFAULT_TEMPLATE,
        'mock_llm_default': not bool(os.environ.get('OPENAI_API_KEY')),
        'selected_template': template_name,
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
            for item in skill_flow:
                segments.extend(
                    [
                        f"阶段 {item.get('index') or ''}",
                        str(item.get('title') or ''),
                        str(item.get('skill') or ''),
                    ]
                )
        else:
            section_cards.extend([item for item in section.get('cards') or [] if isinstance(item, dict)])
    cards = section_cards if sections else output.get('subtask_cards') or trace_item.get('subtask_cards') or []
    for item in cards:
        if not isinstance(item, dict):
            continue
        segments.extend([f"阶段 {item.get('index') or ''}", str(item.get('title') or item.get('phase') or '')])
        for label, key in [
            ('原始 phase', 'phase'),
            ('目标', 'goal'),
            ('技能', 'skill'),
            ('操作臂', 'arm'),
            ('对象', 'object'),
            ('状态', 'arm_state'),
            ('位置', 'target'),
            ('条件', 'completion'),
            ('结果', 'expected_result'),
        ]:
            value = item.get(key)
            if value:
                segments.extend([label, str(value)])
        for line in item.get('subskills') or []:
            segments.extend(['子技能', str(line)])
    if trace_item.get('next_step'):
        segments.append(str(trace_item.get('next_step')))
    return segments


def _result_typewriter_duration(trace_item: dict[str, Any]) -> float:
    char_count = sum(len(segment) for segment in _result_typewriter_segments(trace_item))
    if char_count <= 0:
        return 0.0
    return char_count * TYPEWRITER_INTERVAL_SECONDS + TYPEWRITER_BUFFER_SECONDS


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
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        path = RECORDINGS_DIR / f'{time.strftime("%Y%m%d_%H%M%S")}_roboassembly_panel_simulation.webm'
        path.write_bytes(payload)
        return JSONResponse(
            {
                'path': str(path),
                'size_bytes': path.stat().st_size,
                'media_url': f'/api/media?path={path}',
            }
        )

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
        return FileResponse(safe)

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
      .grid2, .file-row, .row, .kv, .asset-card {
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
            <textarea id="task">基于 plumbers-block UR5e assembly Template 生成新任务 fabrica_plumbers_block_ur5e_right_base_prepare，生成 Menu、Annotation、子任务、技能序列，并保存一条可导出的演示轨迹。</textarea>
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
            <label class="check"><input id="exportLerobot" type="checkbox"><span>导出 LeRobot</span></label>
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
      ['demo', 'Demo', ['demo_started', 'demo_completed', 'manual_demo_execution_started', 'manual_demo_execution_completed']],
      ['lerobot', 'LeRobot', ['lerobot_export_started', 'lerobot_export_completed']]
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
        [
          ['原始 phase', item.phase],
          ['目标', item.goal],
          ['技能', item.skill],
          ['操作臂', item.arm],
          ['对象', item.object],
          ['状态', item.arm_state],
          ['位置', item.target],
          ['条件', item.completion],
          ['结果', item.expected_result]
        ].forEach(([label, value]) => {
          if (value) segments.push(label, value);
        });
        (item.subskills || []).forEach(line => segments.push('子技能', line));
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
            section.skill_flow.forEach(item => {
              segments.push(`阶段 ${item.index || ''}`, item.title || '', compactLabel(item.skill || ''));
            });
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
      const row = (label, value) => value ? `
        <div class="subtask-row">
          <div class="subtask-label">${label}</div>
          <div class="subtask-value">${escapeHtml(value)}</div>
        </div>` : '';
      return `<div class="subtask-list">${items.map(item => `
        <article class="subtask-card">
          <div class="subtask-head">
            <div class="subtask-index">阶段 ${escapeHtml(item.index || '')}</div>
            <div class="subtask-title">${escapeHtml(item.title || item.phase || '')}</div>
          </div>
          <div class="subtask-body">
            ${row('原始 phase', item.phase)}
            ${row('目标', item.goal)}
            ${row('技能', item.skill)}
            ${row('操作臂', item.arm)}
            ${row('对象', item.object)}
            ${row('状态', item.arm_state)}
            ${row('位置', item.target)}
            ${row('条件', item.completion)}
            ${row('结果', item.expected_result)}
            ${(item.subskills || []).length ? `<div class="subskill-list">${item.subskills.map(line => `<div class="subskill-line">${escapeHtml(line)}</div>`).join('')}</div>` : ''}
          </div>
        </article>
      `).join('')}</div>`;
    }
    function renderSubtaskCardsTyped(items, budget) {
      if (!Array.isArray(items) || !items.length) return '';
      const row = (label, value) => {
        if (!value) return '';
        const labelText = revealFromBudget(budget, label);
        const valueText = revealFromBudget(budget, value);
        if (!labelText && !valueText) return '';
        return `<div class="subtask-row">
          <div class="subtask-label">${labelText}</div>
          <div class="subtask-value">${valueText}</div>
        </div>`;
      };
      const cards = items.map(item => {
        const index = revealFromBudget(budget, `阶段 ${item.index || ''}`);
        const title = revealFromBudget(budget, item.title || item.phase || '');
        const rows = [
          row('原始 phase', item.phase),
          row('目标', item.goal),
          row('技能', item.skill),
          row('操作臂', item.arm),
          row('对象', item.object),
          row('状态', item.arm_state),
          row('位置', item.target),
          row('条件', item.completion),
          row('结果', item.expected_result)
        ].filter(Boolean).join('');
        const subskills = (item.subskills || []).map(line => {
          const labelText = revealFromBudget(budget, '子技能');
          const lineText = revealFromBudget(budget, line || '');
          if (!labelText && !lineText) return '';
          return `<div class="subskill-line">${labelText ? `<strong>${labelText}</strong> ` : ''}${lineText}</div>`;
        }).filter(Boolean).join('');
        if (!index && !title && !rows && !subskills) return '';
        return `<article class="subtask-card">
          <div class="subtask-head">
            <div class="subtask-index">${index}</div>
            <div class="subtask-title">${title}</div>
          </div>
          ${rows || subskills ? `<div class="subtask-body">${rows}${subskills ? `<div class="subskill-list">${subskills}</div>` : ''}</div>` : ''}
        </article>`;
      }).filter(Boolean).join('');
      return cards ? `<div class="subtask-list">${cards}</div>` : '';
    }
    function renderDecompositionSectionsTyped(sections, budget) {
      if (!Array.isArray(sections) || !sections.length) return '';
      const html = sections.map(section => {
        const range = revealFromBudget(budget, section.range || '');
        const title = revealFromBudget(budget, section.title || '');
        const summary = revealFromBudget(budget, section.summary || '');
        const flow = Array.isArray(section.skill_flow) ? section.skill_flow.map(item => {
          const index = revealFromBudget(budget, `阶段 ${item.index || ''}`);
          const titleText = revealFromBudget(budget, item.title || '');
          const skill = revealFromBudget(budget, compactLabel(item.skill || ''));
          if (!index && !titleText && !skill) return '';
          return `<div class="subtask-row">
            <div class="subtask-label">${index}</div>
            <div class="subtask-value">${titleText}${skill ? ` · ${skill}` : ''}</div>
          </div>`;
        }).filter(Boolean).join('') : '';
        const cards = flow ? '' : renderSubtaskCardsTyped(section.cards || [], budget);
        if (!range && !title && !summary && !flow && !cards) return '';
        return `<section class="decomposition-section">
          <div class="section-head">
            <div class="section-title">${range}${range && title ? ' · ' : ''}${title}</div>
            ${summary ? `<div class="section-summary">${summary}</div>` : ''}
          </div>
          ${flow ? `<div class="subtask-card"><div class="subtask-body">${flow}</div></div>` : ''}
          ${cards}
        </section>`;
      }).filter(Boolean).join('');
      return html ? `<div class="decomposition-list">${html}</div>` : '';
    }
    async function getJson(url, options) {
      const response = await fetch(url, options);
      if (!response.ok) throw new Error(await response.text());
      return await response.json();
    }
    async function loadInventory() {
      const scene = document.getElementById('sceneProfile').value || 'taoyuan_grscenes_tabletop';
      const template = document.getElementById('template').value || 'fabrica_plumbers_block_ur5e_right_base_prepare';
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
        `<option value="${escapeHtml(item.name)}"${item.name === inventory.selected_template ? ' selected' : ''}>${escapeHtml(item.name)}</option>`
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
                ${image ? `<img class="asset-thumb" src="${image}" alt="${escapeHtml(item.name || title)}">` : `<div class="asset-thumb"></div>`}
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
        export_lerobot: document.getElementById('exportLerobot').checked,
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
      document.getElementById('exportLerobot').checked = false;
      document.getElementById('template').value = 'fabrica_plumbers_block_ur5e_right_base_prepare';
      document.getElementById('task').value = '基于 plumbers-block UR5e assembly Template 生成新任务 fabrica_plumbers_block_ur5e_right_base_prepare，在线调用 LLM Planner 生成 Menu、Asset Annotation、子任务、技能序列，并准备仿真执行。';
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
        video: {frameRate: 30},
        audio: false
      });
      const chunks = [];
      let recorder;
      try {
        recorder = new MediaRecorder(stream, {mimeType: 'video/webm;codecs=vp9'});
      } catch (error) {
        recorder = new MediaRecorder(stream, {mimeType: 'video/webm'});
      }
      state.recording = {stream, recorder, chunks};
      recorder.ondataavailable = event => {
        if (event.data && event.data.size) chunks.push(event.data);
      };
      recorder.onstop = async () => {
        setRecordButton(false, true);
        try {
          stream.getTracks().forEach(track => track.stop());
          const blob = new Blob(chunks, {type: 'video/webm'});
          const saved = await getJson('/api/recordings', {
            method: 'POST',
            headers: {'Content-Type': 'video/webm'},
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
      addRecordingEvent('recording_started', {});
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
        if (key === 'lerobot' && types.has('lerobot_export_started') && !types.has('lerobot_export_completed')) { cls = 'stage active'; text = 'running'; }
        if (failed && (key === 'demo' || key === 'lerobot' || key === 'check')) { cls += ' fail'; text = 'check trace'; }
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
          const decompositionSections = payload.output?.decomposition_sections || payload.decomposition_sections || [];
          const subtaskCards = payload.output?.subtask_cards || payload.subtask_cards || [];
          const summaryText = cleanResultSummary(payload.thinking_process);
          const typewriterKey = `${state.job?.id || state.jobId || 'job'}:${eventIndex}:${event.type}:${payload.stage || ''}`;
          const typedCard = typewriterBudget(
            typewriterKey,
            resultTextSegments(summaryText, processLines, reasoningSteps, decompositionSections, decompositionSections.length ? [] : subtaskCards, payload.next_step || '')
          );
          const summaryHtml = revealFromBudget(typedCard, summaryText);
          const processHtml = renderProcessLinesTyped(processLines, typedCard);
          const reasoningHtml = renderReasoningStepsTyped(reasoningSteps, typedCard);
          const sectionHtml = renderDecompositionSectionsTyped(decompositionSections, typedCard);
          const subtaskHtml = decompositionSections.length ? '' : renderSubtaskCardsTyped(subtaskCards, typedCard);
          const nextHtml = revealFromBudget(typedCard, payload.next_step || '');
          messages.push({
            role: 'agent',
            time: event.time,
            html: `<div class="thought-summary">${summaryHtml}</div>
              ${processHtml}
              ${reasoningHtml}
              ${sectionHtml}
              ${subtaskHtml}
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
            html: `<div class="thought-title">新任务生成完成</div><div>fabrica_plumbers_block_ur5e_right_base_prepare 的 Menu、Annotation、技能步骤和仿真入口已经生成。</div>${links ? `<div class="thought-next">${links}</div>` : ''}`
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
          messages.push({
            role: 'agent',
            time: completed.time,
            html: `<div class="thought-title">仿真完成</div>
              <div>脚本执行结束，视频文件${payload.video_exists ? '已经生成' : '尚未检测到'}。</div>
              <div class="thought-next">${escapeHtml(payload.video || '')}</div>`
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
            messages.push({
              role: 'agent',
              time: event.time,
              html: `<div class="thought-title">录制已开启</div><div>当前选择的屏幕或窗口正在录制。</div>`
            });
          }
          if (event.type === 'recording_saved') {
            const payload = event.payload || {};
            messages.push({
              role: 'agent',
              time: event.time,
              html: `<div class="thought-title">录制已保存</div>
                <div>面板和仿真过程录制文件已写入本地输出目录。</div>
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
        runtime_feedback: result.runtime_feedback,
        lerobot: result.lerobot
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
