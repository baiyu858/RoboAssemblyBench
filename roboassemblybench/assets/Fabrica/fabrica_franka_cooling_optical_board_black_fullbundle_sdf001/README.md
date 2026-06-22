# Fabrica Franka Cooling Manifold IsaacLab Handoff

This package is a scene-level handoff for an IsaacLab-ready Fabrica assembly
workcell. It is intended for downstream imitation learning / RL / controller
development, not as a complete reward/task implementation.

## Contents

- `scene/scene.usda`: Isaac Sim USD stage with one clean packing table, two Franka/Panda
  robots, a loose Fabrica cooling manifold set, and an assembled reference set.
- `scene/clean_packing_table.usda`: wrapper around the Isaac official packing
  table with the default tray/container disabled.
- `isaaclab_cfg/fabrica_dual_franka_scene_cfg.py`: `InteractiveSceneCfg` style
  IsaacLab scene config.
- `assets/fabrica_original_usd_sdf_margin_001/aligned/cooling_manifold`: bundled
  Fabrica cooling manifold USD assets.
- `docs/COOLING_MANIFOLD_INSERTABILITY_VALIDATION.md`: geometry/SDF validation
  notes.

## Bundled Assets

This is a full-bundle handoff. The package includes the Fabrica cooling manifold parts plus the Isaac official robot/table assets used by this scene:

- `assets/isaac_official/Isaac/Props/PackingTable/packing_table.usd`
- `assets/isaac_official/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd`

The receiver still needs Isaac Sim 5.1 and IsaacLab installed, but should not need to separately download Isaac official assets for this scene. `ISAAC_ASSET_ROOT` can still override the bundled root if desired.

## Isaac Sim Preview

After patching paths if needed, open:

```bash
scene/scene.usda
```

This is the fastest way to visually confirm the delivered static stage.

## IsaacLab Use

Import the scene config from `isaaclab_cfg/fabrica_dual_franka_scene_cfg.py`.
Example skeleton:

```python
from pathlib import Path
import importlib.util

pkg = Path("/path/to/fabrica_franka_cooling_isaaclab_fullbundle_sdf001")
cfg_path = pkg / "isaaclab_cfg" / "fabrica_dual_franka_scene_cfg.py"
spec = importlib.util.spec_from_file_location("fabrica_dual_franka_scene_cfg", cfg_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

scene_cfg = mod.FabricaDualFrankaWorkcellSceneCfg(num_envs=1, env_spacing=2.0)
```

Downstream users can wrap this scene in an IsaacLab task/env and add actions,
observations, rewards, reset logic, demonstrations, or controllers.

## Physics Notes

- Fabrica parts use PhysX SDF mesh collision.
- SDF resolution is `512`.
- SDF margin was reduced to `0.001 m` for easier tight insertion than the earlier
  `0.005 m` conversion.
- Contact settings in the IsaacLab cfg use `contact_offset=0.005` and
  `rest_offset=0.0`, matching the official Fabrica/Factory-style settings.
- Loose parts and assembled-reference parts are currently dynamic rigid objects
  with gravity enabled. If the assembled reference should be used only as a
  kinematic target/display, downstream code can mark those `assembled_display_*`
  bodies kinematic or remove them from the physics task.

## Known Scope

This package does not include a complete IsaacLab `DirectRLEnv` or manager-based
task definition. It is a clean scene bridge: assets, layout, collision settings,
and a scene config that another team can plug into training code.

## Relation to IsaacLab Factory

This handoff is structurally close to IsaacLab Factory scene configuration: it uses `InteractiveSceneCfg`, `ArticulationCfg`, `UsdFileCfg`, and `RigidObjectCfg` with Factory/Fabrica-style contact settings. It is not a complete `DirectRLEnvCfg`; downstream users still need to add observations, actions, rewards, resets, and controller logic.
## Black Fabrica Optical Board Scene

This package contains the `franka` dual-arm Fabrica cooling-manifold scene with a black Fabrica `optical_board.obj` placed on the shared work area. All loose parts and the assembled reference are positioned on top of this board.

Authoritative runtime scene:

- `scene/scene.usda`

Editable scene description:

- `scene/scene_spec.json`

JSON-to-USD updater:

```bash
/isaac-sim/python.sh scripts/json_to_usd.py --spec scene/scene_spec.json --usd scene/scene.usda
```

Typical editing flow:

1. Edit object transforms in `scene/scene_spec.json`, especially `scene_graph.objects[].transform`.
2. Run `scripts/json_to_usd.py`.
3. Load `scene/scene.usda` in Isaac Sim or reference it from Isaac Lab.

Included assets:

- Isaac official clean packing table.
- Isaac official `franka` robot assets and dependencies.
- Fabrica cooling-manifold part USD assets with SDF collision margin 0.001.
- Fabrica black optical board source OBJ at `assets/fabrica_support/optical_board.obj`.

Fabrica-style physics values recorded in JSON:

- Fabrica parts: density `1250 kg/m^3`, static/dynamic friction `0.5`.
- Optical board/support surface: density `1000 kg/m^3`, static/dynamic friction `0.3`, restitution `0.0`.
