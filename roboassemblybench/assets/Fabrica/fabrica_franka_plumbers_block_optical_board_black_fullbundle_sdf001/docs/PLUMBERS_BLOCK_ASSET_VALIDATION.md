# Plumbers Block Asset Validation

This package uses the Fabrica `plumbers_block` assembly.

## Verified Properties

- Part count: 5 loose parts.
- Assembled reference: composed from the same five part USDs using Fabrica final assembly translations from the regenerated manifest.
- Collision approximation: PhysX SDF mesh collision.
- SDF margin: `0.001 m` on every part mesh.
- Density: `1250 kg/m^3`.
- Static/dynamic friction: `0.5`.
- Restitution: `0.0`.

## Scene Layout

- The black Fabrica optical board is placed on the clean packing table.
- The assembled reference and all loose parts are placed on top of the black board.
- Footprints were checked during generation so the loose parts and assembled reference stay within the board bounds and do not overlap in XY.
