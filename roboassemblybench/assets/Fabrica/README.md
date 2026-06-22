# Fabrica Assets

This directory stores Fabrica full-bundle handoff packages used by RoboAssemblyBench tasks.

The `fabrica_cooling_manifold` task reuses the benchmark's existing factory scene profile and
references the Franka cooling-manifold bundle:

```bash
cd roboassemblybench/assets/Fabrica
unzip fabrica_franka_cooling_optical_board_black_fullbundle_sdf001.zip
```

After extraction, the task can load per-part USD assets from the unpacked bundle while still using
the standard RoboAssemblyBench factory scene/fallback scene.
