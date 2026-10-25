"""Build-time mesh integration and runtime validation of Fabrica part masses.

Runtime compilation only reads the precomputed JSON; it does not need trimesh
or the official OBJ checkout. Never substitute a default mass for missing data.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

VOLUME_METHOD = 'watertight_triangle_mesh_weld_1e-8m'


def part_mass_kg(part: dict[str, Any]) -> float:
    """Require a positive, internally consistent density/volume/mass record."""
    label = part.get('name', part.get('part_id', 'unknown'))
    properties = part.get('mass_properties')
    if not isinstance(properties, dict):
        raise ValueError(
            f'{label}: missing mass_properties; recompute canonical metadata with '
            'roboassemblybench/scripts/recompute_fabrica_part_masses.py.'
        )
    values = {}
    for key in ('mass_kg', 'volume_m3', 'density_kg_per_m3'):
        value = properties.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{label}: mass_properties.{key} must be finite and positive.')
        values[key] = float(value)
    if not math.isclose(
        values['mass_kg'], values['volume_m3'] * values['density_kg_per_m3'], rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError(f'{label}: mass_kg does not equal density_kg_per_m3 * volume_m3.')
    return values['mass_kg']


def compute_part_mass_properties(
    part: dict[str, Any],
    *,
    assembly: str,
    source_meshes_root: Path,
    scale_to_m: float,
    density_kg_per_m3: float,
) -> dict[str, Any]:
    """Integrate a closed official OBJ at the same scale as its canonical USD.

    Some official OBJs split vertices at normal/texture seams. Geometry-only
    welding at 1e-8 m closes these seams without filling holes, using a convex
    hull, or changing the simulator's mesh. Reject genuinely open meshes.
    """
    import numpy as np
    import trimesh

    if not math.isfinite(scale_to_m) or scale_to_m <= 0:
        raise ValueError('Mesh scale_to_m must be finite and positive.')
    if not math.isfinite(density_kg_per_m3) or density_kg_per_m3 <= 0:
        raise ValueError('Part density must be finite and positive.')
    relative_mesh = Path(assembly) / f'{part["part_id"]}.obj'
    mesh_path = source_meshes_root / relative_mesh
    mesh = trimesh.load(mesh_path, force='mesh', process=True)
    mesh.apply_scale(scale_to_m)
    mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=8)
    if not mesh.is_watertight or not mesh.is_winding_consistent or not mesh.is_volume:
        raise ValueError(f'{mesh_path}: mass calculation requires a closed, consistently oriented solid mesh.')
    expected_bounds = np.asarray([part['bbox_min'], part['bbox_max']], dtype=float)
    if expected_bounds.shape != (2, 3) or not np.allclose(mesh.bounds, expected_bounds, rtol=1e-6, atol=1e-8):
        raise ValueError(f'{mesh_path}: scaled OBJ bounds do not match canonical USD metadata; check mesh/units.')
    volume = float(mesh.volume)
    properties = {
        'density_kg_per_m3': float(density_kg_per_m3),
        'mass_kg': volume * density_kg_per_m3,
        'source_mesh': relative_mesh.as_posix(),
        'source_mesh_scale_to_m': float(scale_to_m),
        'source_mesh_sha256': hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        'volume_m3': volume,
        'volume_method': VOLUME_METHOD,
    }
    part_mass_kg({**part, 'mass_properties': properties})
    return properties
