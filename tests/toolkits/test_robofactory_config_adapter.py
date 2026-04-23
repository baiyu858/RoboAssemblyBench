from __future__ import annotations

import json
from pathlib import Path

import yaml

from toolkits.robofactory_compat.config_adapter import (
    DEFAULT_ASSET_DIR,
    iter_robofactory_config_paths,
    load_robofactory_config,
    normalize_robofactory_config,
    normalize_robofactory_config_file,
    write_normalized_config,
)
from toolkits.robofactory_compat.cli import main as robofactory_cli_main


TABLE_CONFIG_DIR = Path('third_part/RoboFactory/robofactory/configs/table')


def _contains_placeholder(value, placeholder='${ASSET_DIR}'):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'placeholder_map':
                continue
            if _contains_placeholder(item, placeholder):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_placeholder(item, placeholder) for item in value)
    return placeholder in str(value)


def test_iter_robofactory_config_paths_discovers_table_configs():
    paths = iter_robofactory_config_paths(TABLE_CONFIG_DIR)
    assert paths
    assert any(path.name == 'take_photo.yaml' for path in paths)
    assert any(path.name == 'stack_cube.yaml' for path in paths)


def test_load_robofactory_config_reads_yaml():
    config = load_robofactory_config(TABLE_CONFIG_DIR / 'take_photo.yaml')

    assert config['task_name'] == 'TakePhoto'
    assert config['scene']['name'] == 'Table'


def test_normalize_take_photo_preserves_camera_metadata_and_replaces_assets():
    normalized = normalize_robofactory_config_file(TABLE_CONFIG_DIR / 'take_photo.yaml')

    assert normalized['task_name'] == 'TakePhoto'
    assert normalized['task_slug'] == 'take_photo'
    assert normalized['scene']['name'] == 'Table'
    assert normalized['scene_assets'][0]['uid'] == 'table'
    assert normalized['scene_assets'][0]['section'] == 'scene.assets'
    assert normalized['scene_primitives'][0]['index'] == 0
    assert normalized['objects'][0]['uid'] == 'camera'
    assert normalized['agents'][0]['uid'] == 'panda-0'
    assert normalized['metadata']['camera_groups'] == ['sensor', 'human_render']
    assert normalized['metadata']['sections']['scene.assets'] == 1
    assert normalized['metadata']['counts']['cameras'] == 6
    assert normalized['scene_assets'][0]['file_path'] == str(DEFAULT_ASSET_DIR / 'scenes/table/table.glb')

    camera_groups = normalized['cameras']['groups']
    assert set(camera_groups) == {'sensor', 'human_render'}
    assert camera_groups['sensor'][0]['group'] == 'sensor'
    assert camera_groups['sensor'][0]['uid'] == 'head_camera_agent0'
    assert camera_groups['sensor'][0]['group_index'] == 0
    assert camera_groups['sensor'][0]['pose']['type'] == 'pose'
    assert normalized['cameras']['all'][0]['width'] == 320
    assert normalized['cameras']['all'][0]['height'] == 240
    assert normalized['cameras']['all'][0]['near'] == 0.1
    assert normalized['cameras']['all'][0]['far'] == 100

    assert not _contains_placeholder(normalized)


def test_normalize_stack_cube_handles_missing_sections():
    normalized = normalize_robofactory_config_file(TABLE_CONFIG_DIR / 'stack_cube.yaml')

    assert normalized['scene_assets'][0]['name'] == 'table'
    assert normalized['objects'] == []
    assert normalized['scene_primitives'][0]['name'] == 'cubeA'
    assert normalized['scene_primitives'][1]['builder'].endswith('build_cube')
    assert normalized['metadata']['counts']['objects'] == 0
    assert normalized['metadata']['counts']['scene_primitives'] == 2
    assert normalized['metadata']['sections']['objects'] == 0


def test_normalize_in_memory_config_preserves_raw_camera_metadata():
    raw_config = {
        'task_name': 'DemoTask',
        'scene': {
            'name': 'CustomScene',
            'assets': [
                {
                    'file_path': '${ASSET_DIR}/scenes/demo/demo.glb',
                    'collision': {'type': 'box'},
                },
            ],
        },
        'objects': [
            {
                'file_path': '${ASSET_DIR}/objects/demo/demo.glb',
            },
        ],
        'agents': [
            {
                'robot_uid': 'panda-special',
            },
        ],
        'cameras': {
            'sensor': [
                {
                    'pose': {'type': 'look_at', 'params': [[1, 2, 3], [0, 0, 0]]},
                    'width': 64,
                    'height': 48,
                    'custom_field': {'exposure': 0.5},
                },
            ],
        },
    }

    normalized = normalize_robofactory_config(
        raw_config,
        asset_dir='/tmp/robofactory-assets',
        placeholder_map={'${EXTRA_ROOT}': '/tmp/unused'},
    )

    assert normalized['scene_assets'][0]['uid'] == 'scene_asset_0'
    assert normalized['scene_assets'][0]['section'] == 'scene.assets'
    assert normalized['objects'][0]['uid'] == 'object_0'
    assert normalized['agents'][0]['uid'] == 'panda-special'
    assert normalized['cameras']['all'][0]['custom_field']['exposure'] == 0.5
    assert normalized['cameras']['all'][0]['group_index'] == 0
    assert normalized['cameras']['all'][0]['section'] == 'cameras.sensor'
    assert normalized['scene_assets'][0]['file_path'] == '/tmp/robofactory-assets/scenes/demo/demo.glb'
    assert normalized['metadata']['sections']['cameras'] == 1
    assert not _contains_placeholder(normalized)


def test_write_normalized_config_supports_json_and_yaml(tmp_path):
    normalized = normalize_robofactory_config_file(TABLE_CONFIG_DIR / 'take_photo.yaml')

    json_path = tmp_path / 'normalized.json'
    yaml_path = tmp_path / 'normalized.yaml'
    write_normalized_config(json_path, normalized, output_format='json')
    write_normalized_config(yaml_path, normalized, output_format='yaml')

    assert json.loads(json_path.read_text(encoding='utf-8'))['task_slug'] == 'take_photo'
    assert yaml.safe_load(yaml_path.read_text(encoding='utf-8'))['task_name'] == 'TakePhoto'


def test_cli_normalizes_single_config_to_yaml(tmp_path):
    output_dir = tmp_path / 'normalized'
    exit_code = robofactory_cli_main(
        [
            '--input',
            str(TABLE_CONFIG_DIR / 'take_photo.yaml'),
            '--output-dir',
            str(output_dir),
            '--format',
            'yaml',
        ]
    )

    assert exit_code == 0
    output_path = output_dir / 'take_photo.yaml'
    assert output_path.exists()
    payload = yaml.safe_load(output_path.read_text(encoding='utf-8'))
    assert payload['metadata']['counts']['objects'] == 2
    assert payload['cameras']['groups']['human_render'][0]['uid'] == 'render_camera'


def test_cli_normalizes_directory_to_json(tmp_path):
    output_dir = tmp_path / 'normalized'
    exit_code = robofactory_cli_main(
        [
            '--input',
            str(TABLE_CONFIG_DIR),
            '--output-dir',
            str(output_dir),
            '--format',
            'json',
        ]
    )

    assert exit_code == 0
    output_path = output_dir / 'take_photo.json'
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding='utf-8'))
    assert payload['task_slug'] == 'take_photo'
    assert payload['metadata']['sections']['scene.primitives'] == 1
