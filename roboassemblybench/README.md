# RoboAssemblyBench

`RoboAssemblyBench` is the standalone dual-arm assembly benchmark package in this repository.

The benchmark is organized as a framework instead of a Taoyuan-only demo:

- `core/`: shared loaders, scene profile resolution, task registry, and runtime-facing APIs
- `tasks/<task_name>/`: one folder per benchmark task with `recipe.yaml` and `annotation.yaml`
- `scenes/profiles/`: scene profile definitions, including lightweight procedural and asset-backed variants
- `scripts/`: benchmark entrypoints for rollout generation, dataset conversion, LeRobot export, and camera preview

Current task folders:

- `tasks/peg_insertion/`
- `tasks/screw_fastening/`
- `tasks/panel_alignment/`
- `tasks/bracket_latching/`
- `tasks/connector_docking/`
- `tasks/gear_pair_mesh/`
- `tasks/nut_thread_after_hold/`
- `tasks/handover_fastener_then_insert/`

The legacy toolkit under `toolkits/factory_dual_franka_assembly/` is kept as a compatibility layer, but new task authoring should target `roboassemblybench/`.
