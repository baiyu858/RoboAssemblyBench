# Cooling Manifold Insertability Validation

Validation target:

```text
/data/hjj/src/fabrica_isaaclab/outputs/scene_only_dual_ur5e_workcell/cooling_manifold_factory_core_v2/scene.usda
```

## Summary

The current `cooling_manifold` assets are suitable as a handoff baseline for
Fabrica-style insertion task development:

- The parts are regenerated from the original Fabrica OBJ geometry.
- The converted USD part assets use PhysX SDF mesh collision, not convex hull.
- The assembled target pose is sourced from the original Fabrica assembled part
  coordinates.
- Static geometry checks did not find obvious target-pose interpenetration.

This validates the scene as a geometry/collision-ready task starting point.  It
does not replace a downstream closed-loop insertion policy test.

## SDF Collision Check

Each converted part asset under:

```text
/data/hjj/src/fabrica_isaaclab/assets/fabrica_original_usd/aligned/cooling_manifold/parts
```

has:

- `physics:approximation = sdf`
- `PhysicsCollisionAPI`
- `PhysicsMeshCollisionAPI`
- `physxSDFMeshCollision:sdfResolution = 512`
- `physxSDFMeshCollision:sdfSubgridResolution = 6`
- `physxSDFMeshCollision:sdfBitsPerSubgridPixel = BitsPerPixel16`

The composed v2 scene preserves these properties:

```text
sdf_meshes: 14
collision_meshes: 14
rigid_roots: 14
bad: []
```

There are 14 SDF collision meshes because the scene contains 7 loose parts and
7 assembled-reference parts.

## Static Geometry Fit Check

The original Fabrica OBJ meshes were loaded in meters using the same `0.01`
scale used by the conversion pipeline.

All seven OBJ meshes are watertight:

```text
part 0 watertight True
part 1 watertight True
part 2 watertight True
part 3 watertight True
part 4 watertight True
part 5 watertight True
part 6 watertight True
```

Signed-distance checks against the main socket/body part (`part 1`) showed no
plug/display part vertices inside the solid body by more than the tolerance:

```text
part 0 positive_count 0
part 2 positive_count 0
part 3 positive_count 0
part 4 positive_count 0
part 5 positive_count 0
part 6 positive_count 0
```

Pairwise vertex signed-distance checks with a `0.1 mm` threshold also reported
no pairwise interpenetration at the original assembled target pose.

## Important Handoff Notes

- The geometry and SDF collision settings preserve holes better than convex
  hull collision. Convex hull or coarse convex decomposition should not be used
  for the insertion-critical cooling manifold parts.
- `scene_spec.json` contains the authoritative target poses under
  `assembled_display.assembly_target_poses`.
- For learning, use loose parts as dynamic objects. Use the assembled reference
  as kinematic display/target metadata, or add fixed joints/compound rigid-body
  modeling if a physically stable completed assembly is required.
- The current validation is a static geometry and USD collision validation.
  Downstream task owners should still run a PhysX insertion smoke test once the
  controller/action interface is implemented.
