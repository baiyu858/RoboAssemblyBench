# RoboFactory Compat Toolkit

This toolkit adds a practical migration bridge between the checked-in `third_part/RoboFactory` assets/configs and the
current `InternUtopia` benchmark/data pipeline.

It currently covers two layers:

- `normalize`: parse RoboFactory YAML configs into a stable, InternUtopia-friendly normalized spec with preserved
  scene/assets/object/agent/camera metadata.
- `export`: convert current InternUtopia `episode_*.json` outputs into RoboFactory-style per-agent dataset artifacts.

## Normalize RoboFactory Configs

Normalize one table config:

```bash
python -m toolkits.robofactory_compat normalize \
  --input third_part/RoboFactory/robofactory/configs/table/take_photo.yaml \
  --output-dir /tmp/robofactory_compat \
  --format json
```

Normalize the whole checked-in table config directory:

```bash
python -m toolkits.robofactory_compat normalize \
  --input third_part/RoboFactory/robofactory/configs/table \
  --output-dir /tmp/robofactory_compat \
  --format yaml
```

## Export InternUtopia Episodes

Convert existing InternUtopia episode JSONs into RoboFactory-style pickle trees:

```bash
python -m toolkits.robofactory_compat export \
  --input-dir toolkits/factory_dual_franka_assembly/outputs/factory_dual_franka_assembly \
  --output-dir toolkits/robofactory_compat/outputs
```

If `zarr` is installed, you can additionally request zarr export:

```bash
python -m toolkits.robofactory_compat export \
  --input-dir toolkits/factory_dual_franka_assembly/outputs/factory_dual_franka_assembly \
  --output-dir toolkits/robofactory_compat/outputs \
  --zarr
```

## Notes

- The config adapter preserves RoboFactory structure and metadata, but it does not automatically translate every
  normalized config into a live InternUtopia runtime task.
- The dataset exporter is designed around the current InternUtopia episode JSON structure where each step stores
  per-agent observations/actions keyed by agent name.
