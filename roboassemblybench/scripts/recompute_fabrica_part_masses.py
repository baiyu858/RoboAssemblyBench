#!/usr/bin/env python3
"""Recompute all seven tasks' masses without loading Fabrica planning pickles.

No files are changed unless --output is supplied. --check verifies that the
checked-in metadata matches the official OBJ geometry and scene densities.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboassemblybench.core.fabrica_mass_properties import compute_part_mass_properties

DEFAULT_METADATA = REPO_ROOT / 'roboassemblybench/assets/Fabrica/canonical_7_bundles/canonical_tasks.json'
DEFAULT_MESHES = REPO_ROOT / 'third_part/Fabrica/assets/fabrica'


def recompute_metadata(
    payload: dict[str, Any], *, repo_root: Path = REPO_ROOT, source_meshes_root: Path = DEFAULT_MESHES
) -> dict[str, Any]:
    updated = copy.deepcopy(payload)
    for assembly, task in updated['tasks'].items():
        scene_spec = json.loads((repo_root / task['scene_spec_path']).read_text(encoding='utf-8'))
        scene_parts = scene_spec['scene']['parts']
        if set(scene_parts) != {str(part['part_id']) for part in task['parts']}:
            raise ValueError(f'{assembly}: scene/canonical part IDs differ.')
        for part in task['parts']:
            part['mass_properties'] = compute_part_mass_properties(
                part,
                assembly=assembly,
                source_meshes_root=source_meshes_root,
                scale_to_m=float(scene_spec['official_units']['scale_cm_to_m']),
                density_kg_per_m3=float(scene_parts[str(part['part_id'])]['density_kg_per_m3']),
            )
    return updated


def mass_records_match(left: dict | None, right: dict) -> bool:
    if not isinstance(left, dict) or left.keys() != right.keys():
        return False
    for key, expected in right.items():
        actual = left[key]
        if isinstance(expected, float):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (float, int))
                or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-14)
            ):
                return False
        elif actual != expected:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--source-meshes-root', type=Path, default=DEFAULT_MESHES)
    parser.add_argument('--output', type=Path, help='Write updated metadata; may be the input path.')
    parser.add_argument('--check', action='store_true', help='Fail if any stored mass record is missing or stale.')
    args = parser.parse_args()
    if args.check and args.output:
        parser.error('--check and --output are mutually exclusive.')
    payload = json.loads(args.metadata.read_text(encoding='utf-8'))
    updated = recompute_metadata(payload, source_meshes_root=args.source_meshes_root)
    mismatches = []
    count = 0
    for assembly, task in updated['tasks'].items():
        originals = {str(p['part_id']): p for p in payload['tasks'][assembly]['parts']}
        for part in task['parts']:
            properties = part['mass_properties']
            print(
                f'{assembly}/{part["part_id"]}: {properties["mass_kg"]:.12g} kg '
                f'({properties["volume_m3"] * 1e6:.6f} cm^3, '
                f'{properties["density_kg_per_m3"]:g} kg/m^3)'
            )
            if not mass_records_match(originals[str(part['part_id'])].get('mass_properties'), properties):
                mismatches.append(f'{assembly}/{part["part_id"]}')
            count += 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(updated, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
        print(f'Wrote {count} part masses to {args.output}.')
    elif args.check and mismatches:
        raise SystemExit(f'Missing/stale mass records: {", ".join(mismatches)}')
    else:
        print(f'Computed {count} parts; {len(mismatches)} stored records differ. No files changed.')


if __name__ == '__main__':
    main()
