# fabrica_ur5e_d405_plumbers_block_minimal_workcell_fullbundle_sdf001_v1

Minimal self-contained Isaac Sim package for the UR5E+D405 Fabrica plumbers_block workcell.

This package is built from a fresh minimal `scene/scene.usda`, not by retaining the full warehouse scene and hiding/deleting objects.

It includes only:

- one clean long packing worktable,
- two UR5E + Robotiq 2F85 + D405 wrist-camera robots,
- official-aligned Fabrica plumbers_block loose parts,
- official optical board and pickup fixture,
- lights and a third-person camera.

It intentionally does not include warehouse background, Go2, G2_omnipicker, transport tray, far workbench, or Franka references.

## Entry Point

- `scene/scene.usda`

## Validate

```bash
cd fabrica_ur5e_d405_plumbers_block_minimal_workcell_fullbundle_sdf001_v1
/path/to/isaac-sim/python.sh scripts/validate_bundle.py
```
