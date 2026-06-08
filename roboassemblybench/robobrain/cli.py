from __future__ import annotations

import argparse
import json
from pathlib import Path

from roboassemblybench.core.paths import BENCHMARK_ROOT
from roboassemblybench.core.scene_profiles import DEFAULT_SCENE_PROFILE
from roboassemblybench.robobrain.executor import RoboBrainRunConfig, RoboBrainRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Online RoboBrain planner and demo generator for RoboAssemblyBench.')
    parser.add_argument('task', help='Natural-language task instruction.')
    parser.add_argument('--scene-profile', default=DEFAULT_SCENE_PROFILE)
    parser.add_argument('--output-dir', default=str(BENCHMARK_ROOT / 'outputs' / 'robobrain'))
    parser.add_argument('--template', default=None, help='Force a specific existing task recipe as the executable template.')
    parser.add_argument('--model', default=None, help='OpenAI model name. Defaults to ROBOBRAIN_MODEL or gpt-4o.')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--mock-llm', action='store_true', help='Use deterministic local planning without calling OpenAI.')
    parser.add_argument('--plan-only', action='store_true', help='Generate plan artifacts and recipe without running Isaac demo generation.')
    parser.add_argument('--observation-image', action='append', default=[], help='Optional RGB observation image path for VLM planning.')
    parser.add_argument('--num-demos', type=int, default=1)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--max-trials', type=int, default=10)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--record-live-video', action='store_true')
    parser.add_argument('--live-video-fps', type=int, default=30)
    parser.add_argument('--live-video-frame-stride', type=int, default=4)
    parser.add_argument('--keep-video-frames', action='store_true')
    parser.add_argument('--no-runtime-robochecker', action='store_true')
    parser.add_argument('--no-runtime-replanning', action='store_true')
    parser.add_argument('--max-runtime-replans', type=int, default=1)
    parser.add_argument('--runtime-checker-stride', type=int, default=8)
    parser.add_argument('--runtime-stop-on-violation', action='store_true', default=True)
    parser.add_argument('--no-runtime-stop-on-violation', dest='runtime_stop_on_violation', action='store_false')
    parser.add_argument('--runtime-capture-rgb', action='store_true', default=True)
    parser.add_argument('--no-runtime-capture-rgb', dest='runtime_capture_rgb', action='store_false')
    parser.add_argument('--runtime-rgb-frame-stride', type=int, default=24)
    parser.add_argument('--no-perception-grounding', action='store_true')
    parser.add_argument('--perception-max-images', type=int, default=8)
    parser.add_argument('--perception-max-state-files', type=int, default=8)
    parser.add_argument(
        '--perception-visual-backend',
        default='local',
        help='Visual grounding backend: local, owlvit, groundingdino, grounded-sam, or none.',
    )
    parser.add_argument('--perception-label', action='append', default=[], help='Extra open-vocabulary visual label.')
    parser.add_argument('--perception-score-threshold', type=float, default=0.20)
    parser.add_argument('--perception-max-detections', type=int, default=16)
    parser.add_argument('--perception-detector-model', default=None)
    parser.add_argument('--perception-visual-device', default=None, help='cpu, cuda, cuda:0, or pipeline device index.')
    parser.add_argument('--perception-sam-checkpoint', default=None)
    parser.add_argument('--perception-sam-model-type', default='vit_h')
    parser.add_argument('--perception-sam2-checkpoint', default=None)
    parser.add_argument('--perception-sam2-config', default=None)
    parser.add_argument('--perception-vlm-grounding', action='store_true')
    parser.add_argument('--perception-vlm-model', default=None)
    parser.add_argument('--no-local-replanning', action='store_true')
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    config = RoboBrainRunConfig(
        scene_profile=None if args.scene_profile in {'none', 'raw', ''} else args.scene_profile,
        output_dir=Path(args.output_dir),
        model=args.model,
        temperature=args.temperature,
        selected_template=args.template,
        max_retries=args.max_retries,
        mock_llm=bool(args.mock_llm),
        plan_only=bool(args.plan_only),
        num_demos=args.num_demos,
        start_seed=args.start_seed,
        max_trials=args.max_trials,
        headless=bool(args.headless),
        record_live_video=bool(args.record_live_video),
        live_video_fps=args.live_video_fps,
        live_video_frame_stride=args.live_video_frame_stride,
        keep_video_frames=bool(args.keep_video_frames),
        observation_images=list(args.observation_image),
        runtime_robochecker=not bool(args.no_runtime_robochecker),
        runtime_replanning=not bool(args.no_runtime_replanning),
        max_runtime_replans=args.max_runtime_replans,
        runtime_checker_stride=args.runtime_checker_stride,
        runtime_stop_on_violation=bool(args.runtime_stop_on_violation),
        runtime_capture_rgb=bool(args.runtime_capture_rgb),
        runtime_rgb_frame_stride=args.runtime_rgb_frame_stride,
        perception_grounding=not bool(args.no_perception_grounding),
        perception_max_images=args.perception_max_images,
        perception_max_state_files=args.perception_max_state_files,
        perception_visual_backend=args.perception_visual_backend,
        perception_visual_labels=list(args.perception_label),
        perception_visual_score_threshold=args.perception_score_threshold,
        perception_visual_max_detections=args.perception_max_detections,
        perception_detector_model=args.perception_detector_model,
        perception_sam_checkpoint=args.perception_sam_checkpoint,
        perception_sam_model_type=args.perception_sam_model_type,
        perception_sam2_checkpoint=args.perception_sam2_checkpoint,
        perception_sam2_config=args.perception_sam2_config,
        perception_visual_device=args.perception_visual_device,
        perception_vlm_grounding=bool(args.perception_vlm_grounding),
        perception_vlm_model=args.perception_vlm_model,
        local_replanning=not bool(args.no_local_replanning),
    )
    result = RoboBrainRunner(config).run(args.task)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
