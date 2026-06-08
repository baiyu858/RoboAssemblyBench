# RoboAssemblyBench

`RoboAssemblyBench` is the standalone dual-arm assembly benchmark package in this repository.

The benchmark is organized as a framework instead of a Taoyuan-only demo:

- `core/`: shared loaders, scene profile resolution, task registry, and runtime-facing APIs
- `tasks/<task_name>/`: one folder per benchmark task with `recipe.yaml` and `annotation.yaml`
- `scenes/profiles/`: scene profile definitions, including lightweight procedural and asset-backed variants
- `scripts/`: benchmark entrypoints for rollout generation, dataset conversion, LeRobot export, and camera preview
- `robobrain/`: online RoboFactory-style RoboBrain planner that turns a natural-language task into
  subgoals, compositional constraints, a generated recipe, and optional demo rollout.

Current task folders:

- `tasks/peg_insertion/`
- `tasks/screw_fastening/`
- `tasks/panel_alignment/`
- `tasks/bracket_latching/`
- `tasks/connector_docking/`
- `tasks/gear_pair_mesh/`
- `tasks/nut_thread_after_hold/`
- `tasks/handover_fastener_then_insert/`
- `tasks/box_packing/`

The legacy toolkit under `toolkits/factory_dual_franka_assembly/` is kept as a compatibility layer, but new task authoring should target `roboassemblybench/`.

## Online RoboBrain

RoboBrain can generate a task plan from a new natural-language instruction, validate it with
RoboChecker, compile it into a temporary RoboAssemblyBench `recipe.yaml`, and optionally call the
existing demo generator. For new task compositions it can ask the LLM for a compact skill plan
(`pick`, `place`, `insert`, `lift`, `press`, `handover`) and compile that into generated targets,
phases, attachments, and success checks instead of only replaying an existing template.

During demo generation it can also run a runtime RoboChecker that captures RGB/state observations,
validates interaction/scheduling/spatial constraints online, grounds the latest images and state
snapshots into planner hints, runs optional visual foundation-model grounding, tries a deterministic
local recipe repair, and then sends failure feedback plus recent image paths back into RoboBrain for
automatic re-planning if local repair does not recover the rollout.

Plan-only smoke run without network access:

```bash
python -m roboassemblybench.robobrain \
  "right Franka inserts a peg into the socket" \
  --mock-llm \
  --plan-only
```

Online planning with OpenAI:

```bash
export OPENAI_API_KEY=...
python -m roboassemblybench.robobrain \
  "two arms lift the steel barrier together" \
  --num-demos 1 \
  --max-trials 10 \
  --max-runtime-replans 1 \
  --record-live-video \
  --headless
```

Each run writes `plan.json`, `primitive_plan.json`, `checker_report.json`, `annotation.yaml`, and
the generated `recipe.yaml` under `roboassemblybench/outputs/robobrain/<timestamp>_<task>/`. When
`--plan-only` is omitted, demo outputs are written to the run's `demo/` subdirectory. Runtime
RoboChecker artifacts are written to `runtime_feedback.json` and `runtime_observations/`.

Useful runtime flags:

```bash
--no-runtime-robochecker        # disable online checking
--no-runtime-replanning         # keep online checking, but do not re-plan after failure
--no-local-replanning           # skip deterministic local recipe repair before LLM re-plan
--no-perception-grounding       # do not summarize runtime RGB/state into the next planner prompt
--perception-visual-backend local
                                # local, owlvit, groundingdino, grounded-sam, or none
--perception-label "peg"        # add open-vocabulary detector labels
--perception-detector-model IDEA-Research/grounding-dino-tiny
                                # override the HF zero-shot detector model
--perception-sam-checkpoint /path/to/sam_vit_h.pth
                                # enable SAM masks for grounded-sam
--perception-sam2-config /path/to/sam2.yaml
--perception-sam2-checkpoint /path/to/sam2.pt
                                # enable SAM2 masks for grounded-sam
--perception-vlm-grounding      # ask the configured OpenAI vision model for semantic grounding
--runtime-checker-stride 8      # state-check interval in rollout steps
--runtime-rgb-frame-stride 24   # RGB capture interval in rollout steps
--max-runtime-replans 2         # number of feedback-driven re-plans after rollout failure
```

Visual grounding backends are optional. The default `local` backend has no model download and
produces color-region proposals. `owlvit` and `groundingdino` use the Hugging Face
`zero-shot-object-detection` pipeline. `grounded-sam` runs the detector first and then uses SAM or
SAM2 masks when `ROBOBRAIN_SAM_CHECKPOINT` or `ROBOBRAIN_SAM2_CHECKPOINT`/`ROBOBRAIN_SAM2_CONFIG`
are configured.

The local repair pass currently handles common runtime failures conservatively:

- scheduling failures: increase timeout budgets and slightly relax target-reaching tolerance;
- spatial failures: widen left/right target lanes by generating safe-lane targets;
- interaction failures: repair grasp/attach tolerances and enforce contact-gated attachment.
