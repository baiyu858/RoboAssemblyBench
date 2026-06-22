# fabrica_official_plumbers_block_fixture_pickup_sdf512_v1

Official-aligned Fabrica IsaacSim/IsaacLab-ready package for `plumbers_block`.

## Entrypoints

- Scene: `scene/scene.usda`
- Scene spec: `scene/scene_spec.json`
- Validation: `validation/validation.json`
- Plan info: `metadata/plan_info.pkl`
- Pickup poses: `assets/fabrica_official_fixture/plumbers_block/pickup.json`

## Semantics

- Source meshes are official Fabrica OBJ files scaled from centimeters to meters with `scale=0.01`.
- Pickup parts are dynamic rigid bodies with gravity enabled.
- The fixture and optical board are static PhysX SDF colliders generated from official Fabrica outputs.
- The assembled reference is composed from the same raw-frame part assets in the official final OBJ frame, and every assembled child part is a dynamic rigid body with gravity enabled.
- SDF collision resolution is 512; dynamic pickup part SDF margin is 0.001 m.

## Validation

Status: `pass`.
