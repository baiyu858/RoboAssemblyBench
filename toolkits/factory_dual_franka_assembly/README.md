# Factory Dual Franka Assembly Toolkit

This toolkit now provides a multi-task dual-arm Franka assembly benchmark on top of `InternUtopia`:

- `screw_fastening`: left arm aligns the part, right arm places and fastens the screw
- `peg_insertion`: left arm holds the housing, right arm inserts the peg into the jig
- `panel_alignment`: left arm aligns the panel, right arm inserts the locating pin
- `bracket_latching`: left arm seats the bracket, right arm closes the latch
- `connector_docking`: left arm stabilizes the socket while the right arm docks the connector body
- `gear_pair_mesh`: both arms align a gear pair and settle the mesh in the shared fixture
- `nut_thread_after_hold`: one arm holds the bracket while the other threads the nut
- `handover_fastener_then_insert`: the fastener is handed off between arms before the final insertion

The implementation mirrors the useful parts of `RoboFactory`:

- YAML task specs for scene/object/robot/phase constraints
- declarative annotation YAMLs for object roles, target roles, phase notes, and export tags
- YAML scene profiles for asset-backed workcell variants
- stage-based scripted solutions
- recipe x scene-profile benchmark matrix generation
- success-seed search before export with resume support
- separate dataset conversion step with phase segments, instruction text, and richer annotation metadata

The first-batch migration now includes task families for:

- connector docking
- gear meshing
- threaded fastener placement
- explicit dual-arm handoff before insertion

The scene profiles currently include:

- `taoyuan_tabletop`: uses the bundled `InternUtopia` white-table USD asset and lifts the assembly workspace onto the tabletop
- `taoyuan_grscenes_tabletop`: uses a real `GRScenes-100` table USD anchor from the downloaded Taoyuan assets
- `proxy_factory_cell`: keeps a lightweight procedural fallback for quick iteration

The first two assembly recipes now default to the Taoyuan tabletop profile:

- `screw_fastening`
- `peg_insertion`

Run the default asset-backed benchmark pair:

```bash
cd /home/baiyu24/model/InternUtopia
python toolkits/factory_dual_franka_assembly/generate_demos.py --recipes screw_fastening peg_insertion --num-demos 2 --max-trials 20 --resume --headless
```

Run multiple scene profiles in one pass:

```bash
cd /home/baiyu24/model/InternUtopia
python toolkits/factory_dual_franka_assembly/generate_demos.py --recipes all --scene-profiles taoyuan_grscenes_tabletop proxy_factory_cell --num-demos 2 --max-trials 20 --resume --headless
```

Run a single recipe on the procedural fallback profile:

```bash
cd /home/baiyu24/model/InternUtopia
python toolkits/factory_dual_franka_assembly/generate_demos.py --recipes screw_fastening --scene-profiles proxy_factory_cell --num-demos 2 --headless
```

Run a single recipe on the real GRScenes-backed Taoyuan tabletop profile:

```bash
cd /home/baiyu24/model/InternUtopia
python toolkits/factory_dual_franka_assembly/generate_demos.py --recipes screw_fastening peg_insertion --scene-profiles taoyuan_grscenes_tabletop --num-demos 2 --max-trials 20 --resume --headless
```

Convert exported episodes into train/val JSONL:

```bash
cd /home/baiyu24/model/InternUtopia
python toolkits/factory_dual_franka_assembly/convert_dataset.py
```

Export the successful episodes to a LeRobot v2.1-style dataset with `meta/`, `data/`, and `videos/`:

```bash
cd /home/baiyu24/model/InternUtopia
python toolkits/factory_dual_franka_assembly/export_lerobot.py
```

Export Isaac replay videos with three camera streams (front, left wrist, right wrist):

```bash
cd /home/baiyu24/model/InternUtopia
python toolkits/factory_dual_franka_assembly/export_lerobot.py --video-mode isaac_replay
```

The LeRobot export writes:

- `meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`, and `meta/episodes_stats.jsonl`
- `data/chunk-000/episode_XXXXXX.parquet`
- `videos/chunk-000/observation.images.topdown/episode_XXXXXX.mp4`

When `--video-mode isaac_replay` is used, the exporter writes three synchronized streams by default:

- `videos/chunk-000/observation.images.front/episode_XXXXXX.mp4`
- `videos/chunk-000/observation.images.left_wrist/episode_XXXXXX.mp4`
- `videos/chunk-000/observation.images.right_wrist/episode_XXXXXX.mp4`

Outputs are written under `toolkits/factory_dual_franka_assembly/outputs/factory_dual_franka_assembly/<scene_profile>/<recipe>/`
with per-run manifests, successful seed logs, and raw episode JSONs.

Task specs live in `toolkits/factory_dual_franka_assembly/task_specs/` and scene profiles live in
`toolkits/factory_dual_franka_assembly/scene_profiles/`. Annotation assets live in
`toolkits/factory_dual_franka_assembly/annotations/`. All three layers support `${ASSET_PATH}` placeholders, so you can
keep porting more RoboFactory-style task decompositions onto real InternUtopia USD assets without rewriting the pipeline.

The exported episode JSON now carries annotation-derived fields such as `task_description`, `annotation_target_roles`,
`target_annotations`, and `phase_annotations`, and `convert_dataset.py` forwards them into the JSONL dataset entries as
`instruction` plus per-task metadata.
