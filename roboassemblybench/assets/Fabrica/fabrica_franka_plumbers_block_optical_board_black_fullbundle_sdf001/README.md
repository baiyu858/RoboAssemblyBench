# Fabrica Franka Panda Plumbers Block IsaacLab Handoff

This full-bundle package contains a scene-level Isaac Sim / IsaacLab handoff for a dual-arm Fabrica `plumbers_block` assembly workcell.

## Contents

- `scene/scene.usda`: Isaac Sim USD stage with one clean packing table, two Franka Panda robots, a black Fabrica optical board, five loose `plumbers_block` parts, and an assembled reference composed from the same five parts.
- Cameras: official-style IsaacLab CameraCfg entries are included for `table_cam`, `table_high_cam`, `/RobotLeft/panda_hand/wrist_cam`, and `/RobotRight/panda_hand/wrist_cam`; the USD stage contains matching Camera prims without visible body/lens proxy geometry.
- `scene/scene_spec.json`: structured scene description, object transforms, support surface metadata, and official assembled target poses.
- `isaaclab_cfg/fabrica_dual_franka_scene_cfg.py`: IsaacLab `InteractiveSceneCfg` bridge with controllable robot articulations and Fabrica part `RigidObjectCfg` entries.
- `assets/fabrica_original_usd_sdf_margin_001/aligned/plumbers_block`: regenerated Fabrica USD assets for `plumbers_block`.
- `assets/fabrica_fixture/plumbers_block/fixture_pickup_tray.usda`: task-specific Fabrica pickup fixture generated from official `planning/run_fixture_gen.py`; loose parts in `scene.usda` are placed from the generated `pickup.json` poses.
- `assets/fabrica_support/optical_board.obj`: Fabrica optical board source mesh used as the black locating/support board.
- `assets/isaac_official/Isaac/...`: bundled robot and table assets needed by this scene.
- `scripts/json_to_usd.py`: rebuild/update `scene.usda` from `scene/scene_spec.json`.
- `scripts/open_in_isaacsim.py` and `.sh`: convenience Isaac Sim preview launcher when available.

## Physics Notes

- Fabrica parts use PhysX SDF mesh collision.
- `physxSDFMeshCollision:sdfMargin` is `0.001 m` on every `plumbers_block` part.
- Fabrica material settings are preserved in intent: density `1250 kg/m^3`, static/dynamic friction `0.5`, restitution `0.0`.
- Loose parts and assembled-reference parts are dynamic rigid objects with gravity enabled. If the assembled reference should be a pure target/display, downstream code can mark those bodies kinematic.
- The black optical board is a static support surface with collision enabled and friction metadata recorded in `scene_spec.json`.

## Isaac Sim Preview

Open the stage directly:

```bash
scene/scene.usda
```

Or use the helper from the package root:

```bash
ISAAC_SIM_ROOT=/path/to/isaac-sim ./scripts/open_in_isaacsim.sh --scene scene/scene.usda
```

## IsaacLab Use

```python
from pathlib import Path
import importlib.util

pkg = Path("/path/to/fabrica_franka_plumbers_block_optical_board_black_fullbundle_sdf001")
cfg_path = pkg / "isaaclab_cfg" / "fabrica_dual_franka_scene_cfg.py"
spec = importlib.util.spec_from_file_location("fabrica_scene_cfg", cfg_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

scene_cfg = mod.FabricaDualFrankaWorkcellSceneCfg(num_envs=1, env_spacing=2.0)
```

This is not a complete RL/imitation-learning task. It intentionally leaves actions, observations, rewards, resets, demonstrations, and controllers to downstream task code.

- Franka cameras use official-style mount points, but their USD/IsaacLab offsets are scene-specific OpenGL look-at poses so `table_cam`, `table_high_cam`, and both wrist cameras see the pickup fixture work area.
