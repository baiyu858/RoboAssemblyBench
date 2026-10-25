"""The seven tasks use physical mesh masses, not 1 kg defaults or 0.25 kg caps."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from roboassemblybench.core.fabrica_mass_properties import compute_part_mass_properties, part_mass_kg
from roboassemblybench.scripts.recompute_fabrica_part_masses import DEFAULT_METADATA, mass_records_match

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA = json.loads(DEFAULT_METADATA.read_text(encoding='utf-8'))
PARTS = [(assembly, part) for assembly, task in METADATA['tasks'].items() for part in task['parts']]


def test_all_seven_tasks_have_individual_mass_records():
    assert set(METADATA['tasks']) == {
        'beam',
        'car',
        'cooling_manifold',
        'duct',
        'gamepad',
        'plumbers_block',
        'stool_circular',
    }
    assert len(PARTS) == 46
    for _, part in PARTS:
        assert part_mass_kg(part) > 0
    plumbers = {part['part_id']: part for part in METADATA['tasks']['plumbers_block']['parts']}
    assert part_mass_kg(plumbers['2']) == pytest.approx(0.29495437381431616)
    assert part_mass_kg(plumbers['0']) == pytest.approx(0.14119460807987402)


@pytest.mark.parametrize('assembly,part', PARTS, ids=[f'{a}/{p["part_id"]}' for a, p in PARTS])
def test_mass_record_matches_closed_official_mesh_and_units(assembly, part):
    source_root = REPO_ROOT / 'third_part/Fabrica/assets/fabrica'
    scene_path = REPO_ROOT / METADATA['tasks'][assembly]['scene_spec_path']
    if not (source_root / assembly / f'{part["part_id"]}.obj').exists() or not scene_path.exists():
        pytest.skip('Official OBJ and canonical bundle required for geometry audit.')
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    expected = compute_part_mass_properties(
        part,
        assembly=assembly,
        source_meshes_root=source_root,
        scale_to_m=scene['official_units']['scale_cm_to_m'],
        density_kg_per_m3=scene['scene']['parts'][part['part_id']]['density_kg_per_m3'],
    )
    assert mass_records_match(part['mass_properties'], expected)


@pytest.mark.parametrize('field', ['mass_kg', 'volume_m3', 'density_kg_per_m3'])
@pytest.mark.parametrize('value', [None, True, '1', 0.0, -1.0, float('nan'), float('inf')])
def test_invalid_mass_record_is_not_replaced_by_a_default(field, value):
    part = copy.deepcopy(PARTS[0][1])
    part['mass_properties'][field] = value
    with pytest.raises(ValueError, match='must be finite and positive'):
        part_mass_kg(part)


def test_missing_or_inconsistent_mass_record_is_rejected():
    with pytest.raises(ValueError, match='missing mass_properties'):
        part_mass_kg({'name': 'missing'})
    part = copy.deepcopy(PARTS[0][1])
    part['mass_properties']['mass_kg'] = 1.0
    with pytest.raises(ValueError, match='does not equal'):
        part_mass_kg(part)


def test_small_mass_is_not_rounded_up_to_a_solver_or_grasp_floor():
    part = METADATA['tasks']['gamepad']['parts'][0]
    assert part_mass_kg(part) == pytest.approx(0.002518982149525137)


@pytest.mark.parametrize('platform', ['ur5e', 'franka'])
@pytest.mark.parametrize('assembly', list(METADATA['tasks']))
def test_mass_survives_both_compilers_and_scene_object_config(assembly, platform):
    from roboassemblybench.core.task_registry import load_task_recipe
    from toolkits.factory_dual_franka_assembly.scene_builder import _build_object_cfg

    recipe = load_task_recipe(f'fabrica_{assembly}_{platform}_staged')
    expected = {part['name']: part_mass_kg(part) for part in METADATA['tasks'][assembly]['parts']}
    objects = {obj['name']: obj for obj in recipe['objects']}
    for name, mass in expected.items():
        obj = objects[name]
        assert obj['mass'] == pytest.approx(mass, rel=1e-12)
        config = _build_object_cfg(obj, position=np.array(obj['position']), orientation=np.array(obj['orientation']))
        assert config.mass == pytest.approx(mass, rel=1e-12)


def test_beam_compiler_does_not_cap_a_declared_mass_at_025kg(monkeypatch):
    from roboassemblybench.core import beam_physical_compiler as compiler
    from roboassemblybench.core.task_registry import load_task_recipe

    payload = copy.deepcopy(METADATA)
    part = payload['tasks']['beam']['parts'][0]
    # A synthetic denser material tests the old cap, even though the current
    # official Beam parts all happen to weigh less than 0.25 kg.
    part['mass_properties']['density_kg_per_m3'] = 0.6 / part['mass_properties']['volume_m3']
    part['mass_properties']['mass_kg'] = 0.6
    monkeypatch.setattr(compiler, 'load_fabrica_canonical_metadata', lambda *_: copy.deepcopy(payload))
    recipe = load_task_recipe('fabrica_beam_ur5e_staged')
    actual = next(obj for obj in recipe['objects'] if obj['name'] == part['name'])
    assert actual['mass'] == pytest.approx(0.6)


def test_geometry_integrator_rejects_open_mesh_and_wrong_scale(tmp_path):
    import trimesh

    directory = tmp_path / 'toy'
    directory.mkdir()
    path = directory / '0.obj'
    mesh = trimesh.creation.box(extents=[10.0, 2.0, 3.0])  # centimeters
    part = {'part_id': '0', 'bbox_min': [-0.05, -0.01, -0.015], 'bbox_max': [0.05, 0.01, 0.015]}
    mesh.export(path)
    kwargs = dict(assembly='toy', source_meshes_root=tmp_path, scale_to_m=0.01, density_kg_per_m3=1250.0)
    result = compute_part_mass_properties(part, **kwargs)
    assert result['volume_m3'] == pytest.approx(60e-6)
    assert result['mass_kg'] == pytest.approx(0.075)
    with pytest.raises(ValueError, match='bounds do not match'):
        compute_part_mass_properties(part, **{**kwargs, 'scale_to_m': 1.0})
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    mesh.export(path)
    with pytest.raises(ValueError, match='closed, consistently oriented'):
        compute_part_mass_properties(part, **kwargs)


def test_geometry_integrator_preserves_holes_instead_of_using_convex_hull(tmp_path):
    import trimesh

    directory = tmp_path / 'toy'
    directory.mkdir()
    mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=1.0)
    mesh.export(directory / '0.obj')
    part = {'part_id': '0', 'bbox_min': mesh.bounds[0].tolist(), 'bbox_max': mesh.bounds[1].tolist()}
    result = compute_part_mass_properties(
        part, assembly='toy', source_meshes_root=tmp_path, scale_to_m=1.0, density_kg_per_m3=1250.0
    )
    assert result['volume_m3'] == pytest.approx(mesh.volume)
    assert result['volume_m3'] < mesh.convex_hull.volume * 0.8
