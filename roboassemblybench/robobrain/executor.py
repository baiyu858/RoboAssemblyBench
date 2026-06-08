from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roboassemblybench.core.paths import BENCHMARK_ROOT
from roboassemblybench.core.scene_profiles import DEFAULT_SCENE_PROFILE
from roboassemblybench.core.task_registry import load_task_recipe
from roboassemblybench.robobrain.checker import RoboChecker
from roboassemblybench.robobrain.compiler import compile_plan_to_recipe, write_recipe_bundle
from roboassemblybench.robobrain.inventory import RoboAssemblyInventory
from roboassemblybench.robobrain.llm import MockRoboBrainClient, OpenAIRoboBrainClient
from roboassemblybench.robobrain.models import CheckResult, RoboBrainPlan, slugify_task_name
from roboassemblybench.robobrain.perception import ObservationGrounder
from roboassemblybench.robobrain.prompts import SYSTEM_PROMPT, build_user_prompt
from roboassemblybench.robobrain.replanner import LocalReplanner


@dataclass
class RoboBrainRunConfig:
    scene_profile: str | None = DEFAULT_SCENE_PROFILE
    output_dir: Path = BENCHMARK_ROOT / 'outputs' / 'robobrain'
    model: str | None = None
    temperature: float = 0.2
    selected_template: str | None = None
    max_retries: int = 2
    mock_llm: bool = False
    plan_only: bool = False
    num_demos: int = 1
    start_seed: int = 0
    max_trials: int = 10
    headless: bool = False
    record_live_video: bool = False
    live_video_fps: int = 30
    live_video_frame_stride: int = 4
    keep_video_frames: bool = False
    observation_images: list[str] = field(default_factory=list)
    runtime_robochecker: bool = True
    runtime_replanning: bool = True
    max_runtime_replans: int = 1
    runtime_checker_stride: int = 8
    runtime_stop_on_violation: bool = True
    runtime_capture_rgb: bool = True
    runtime_rgb_frame_stride: int = 24
    perception_grounding: bool = True
    perception_max_images: int = 8
    perception_max_state_files: int = 8
    perception_visual_backend: str = 'local'
    perception_visual_labels: list[str] = field(default_factory=list)
    perception_visual_score_threshold: float = 0.20
    perception_visual_max_detections: int = 16
    perception_detector_model: str | None = None
    perception_sam_checkpoint: str | None = None
    perception_sam_model_type: str = 'vit_h'
    perception_sam2_checkpoint: str | None = None
    perception_sam2_config: str | None = None
    perception_visual_device: str | None = None
    perception_vlm_grounding: bool = False
    perception_vlm_model: str | None = None
    local_replanning: bool = True


@dataclass
class RoboBrainRunResult:
    plan: RoboBrainPlan
    recipe: dict[str, Any]
    check_result: CheckResult
    bundle_paths: dict[str, Path]
    demo_output_dir: Path | None = None
    demo_command: list[str] | None = None
    runtime_feedback: dict[str, Any] | None = None
    replan_attempt: int = 0
    local_replan_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'plan': self.plan.to_dict(),
            'check_result': self.check_result.to_dict(),
            'bundle_paths': {key: str(value) for key, value in self.bundle_paths.items()},
            'demo_output_dir': None if self.demo_output_dir is None else str(self.demo_output_dir),
            'demo_command': self.demo_command,
            'runtime_feedback': self.runtime_feedback,
            'replan_attempt': self.replan_attempt,
            'local_replan_report': self.local_replan_report,
        }


class RoboBrainRunner:
    def __init__(self, config: RoboBrainRunConfig | None = None):
        self.config = config or RoboBrainRunConfig()
        self.checker = RoboChecker()

    def run(self, task_instruction: str) -> RoboBrainRunResult:
        inventory = RoboAssemblyInventory.load(scene_profile=self.config.scene_profile)
        selected_template = self.config.selected_template or inventory.best_template_for(task_instruction)
        feedback: list[str] = []
        feedback_observation_images: list[str] = []
        feedback_runtime_payload: dict[str, Any] | None = None
        max_runtime_replans = max(int(self.config.max_runtime_replans), 0) if self.config.runtime_replanning else 0
        last_error: subprocess.CalledProcessError | None = None

        for replan_attempt in range(max_runtime_replans + 1):
            plan, recipe, check_result = self._plan_until_static_valid(
                task_instruction=task_instruction,
                inventory=inventory,
                selected_template=selected_template,
                feedback=feedback,
                observation_images=feedback_observation_images,
                runtime_feedback=feedback_runtime_payload,
            )
            run_dir = self._run_dir_for(task_instruction=task_instruction, plan=plan, attempt=replan_attempt)
            bundle_paths = write_recipe_bundle(output_dir=run_dir, plan=plan, recipe=recipe, check_result=check_result)

            if self.config.plan_only:
                return RoboBrainRunResult(
                    plan=plan,
                    recipe=recipe,
                    check_result=check_result,
                    bundle_paths=bundle_paths,
                    replan_attempt=replan_attempt,
                )

            demo_output_dir = run_dir / 'demo'
            runtime_feedback_path = run_dir / 'runtime_feedback.json'
            runtime_observation_dir = run_dir / 'runtime_observations'
            demo_command = self._demo_command(
                recipe_path=bundle_paths['recipe'],
                output_dir=demo_output_dir,
                runtime_feedback_path=runtime_feedback_path,
                runtime_observation_dir=runtime_observation_dir,
            )
            (run_dir / 'demo_command.json').write_text(json.dumps(demo_command, indent=2), encoding='utf-8')

            try:
                subprocess.run(demo_command, check=True)
                runtime_feedback = self._read_runtime_feedback(runtime_feedback_path)
                return RoboBrainRunResult(
                    plan=plan,
                    recipe=recipe,
                    check_result=check_result,
                    bundle_paths=bundle_paths,
                    demo_output_dir=demo_output_dir,
                    demo_command=demo_command,
                    runtime_feedback=runtime_feedback,
                    replan_attempt=replan_attempt,
                )
            except subprocess.CalledProcessError as exc:
                last_error = exc
                runtime_feedback = self._read_runtime_feedback(runtime_feedback_path)
                error_path = run_dir / 'execution_error.json'
                error_path.write_text(
                    json.dumps(
                        {
                            'returncode': exc.returncode,
                            'command': demo_command,
                            'message': 'Demo generation failed after RoboBrain planning.',
                            'runtime_feedback': runtime_feedback,
                            'replan_attempt': replan_attempt,
                        },
                        indent=2,
                    ),
                    encoding='utf-8',
                )
                if replan_attempt >= max_runtime_replans:
                    raise
                if self.config.local_replanning:
                    local_result, local_feedback = self._try_local_replan(
                        task_instruction=task_instruction,
                        plan=plan,
                        recipe=recipe,
                        runtime_feedback=runtime_feedback,
                        replan_attempt=replan_attempt + 1,
                    )
                    if local_result is not None:
                        return local_result
                    if local_feedback is not None:
                        runtime_feedback = local_feedback
                feedback.extend(self._feedback_strings_from_runtime(runtime_feedback))
                feedback_observation_images = self._feedback_images_from_runtime(runtime_feedback)
                feedback_runtime_payload = runtime_feedback
                if not feedback:
                    feedback.append(f'Demo generation failed with return code {exc.returncode}. Re-plan more conservatively.')

        if last_error is not None:
            raise last_error
        raise RuntimeError('RoboBrain failed without producing a runnable plan.')

    def _plan_until_static_valid(
        self,
        *,
        task_instruction: str,
        inventory: RoboAssemblyInventory,
        selected_template: str,
        feedback: list[str],
        observation_images: list[str] | None = None,
        runtime_feedback: dict[str, Any] | None = None,
    ) -> tuple[RoboBrainPlan, dict[str, Any], CheckResult]:
        plan = None
        recipe = None
        check_result = None
        for attempt in range(max(int(self.config.max_retries), 0) + 1):
            plan = self._generate_plan(
                task_instruction=task_instruction,
                inventory=inventory,
                selected_template=selected_template,
                feedback=feedback,
                observation_images=observation_images or [],
                runtime_feedback=runtime_feedback,
            )
            if plan.selected_template not in inventory.recipes:
                feedback.append(
                    f"selected_template {plan.selected_template!r} is unavailable; use one of {inventory.template_names()}."
                )
                plan.selected_template = selected_template

            base_recipe = load_task_recipe(plan.selected_template, scene_profile=self.config.scene_profile)
            recipe = compile_plan_to_recipe(plan=plan, base_recipe=base_recipe, scene_profile=self.config.scene_profile)
            check_result = self.checker.check(plan=plan, recipe=recipe)
            if check_result.ok:
                return plan, recipe, check_result
            feedback.extend(check_result.errors)
            if attempt >= int(self.config.max_retries):
                break

        assert plan is not None
        assert recipe is not None
        assert check_result is not None
        raise RuntimeError(f'RoboChecker rejected the generated plan: {check_result.errors}')

    def _generate_plan(
        self,
        *,
        task_instruction: str,
        inventory: RoboAssemblyInventory,
        selected_template: str,
        feedback: list[str],
        observation_images: list[str] | None = None,
        runtime_feedback: dict[str, Any] | None = None,
    ) -> RoboBrainPlan:
        grounding = self._ground_observations(
            task_instruction=task_instruction,
            observation_images=observation_images or [],
            runtime_feedback=runtime_feedback,
        )
        if self.config.mock_llm:
            payload = MockRoboBrainClient().generate(
                task_instruction=task_instruction,
                inventory=inventory,
                selected_template=selected_template,
                feedback=feedback,
            )
            plan = RoboBrainPlan.from_dict(payload, task_instruction=task_instruction, source='mock')
            plan.grounding = grounding
            return plan

        client = OpenAIRoboBrainClient(model=self.config.model, temperature=self.config.temperature)
        user_prompt = build_user_prompt(
            task_instruction=task_instruction,
            inventory=inventory,
            selected_template_hint=selected_template,
            feedback=feedback,
            grounding=grounding,
        )
        payload = client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            observation_images=[*self.config.observation_images, *(observation_images or [])],
        )
        plan = RoboBrainPlan.from_dict(payload, task_instruction=task_instruction, source='openai')
        if grounding and not plan.grounding:
            plan.grounding = grounding
        return plan

    def _ground_observations(
        self,
        *,
        task_instruction: str | None = None,
        observation_images: list[str],
        runtime_feedback: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not self.config.perception_grounding:
            return {}
        return ObservationGrounder(
            max_images=self.config.perception_max_images,
            max_state_files=self.config.perception_max_state_files,
            visual_backend=self.config.perception_visual_backend,
            visual_labels=self.config.perception_visual_labels,
            visual_score_threshold=self.config.perception_visual_score_threshold,
            visual_max_detections=self.config.perception_visual_max_detections,
            detector_model=self.config.perception_detector_model,
            sam_checkpoint=self.config.perception_sam_checkpoint,
            sam_model_type=self.config.perception_sam_model_type,
            sam2_checkpoint=self.config.perception_sam2_checkpoint,
            sam2_config=self.config.perception_sam2_config,
            visual_device=self.config.perception_visual_device,
            vlm_grounding=self.config.perception_vlm_grounding,
            vlm_model=self.config.perception_vlm_model,
        ).ground(
            observation_images=[*self.config.observation_images, *observation_images],
            runtime_feedback=runtime_feedback,
            task_instruction=task_instruction,
        )

    def _run_dir_for(self, *, task_instruction: str, plan: RoboBrainPlan, attempt: int | str = 0) -> Path:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        slug = plan.task_name or slugify_task_name(task_instruction)
        attempt_text = str(attempt)
        suffix = '' if attempt_text in {'', '0'} else f'_replan{slugify_task_name(attempt_text)}'
        return Path(self.config.output_dir).resolve() / f'{timestamp}_{slug}{suffix}'

    def _demo_command(
        self,
        *,
        recipe_path: Path,
        output_dir: Path,
        runtime_feedback_path: Path | None = None,
        runtime_observation_dir: Path | None = None,
    ) -> list[str]:
        command = [
            sys.executable,
            str(BENCHMARK_ROOT / 'scripts' / 'generate_demos.py'),
            '--recipes',
            str(recipe_path),
            '--num-demos',
            str(int(self.config.num_demos)),
            '--start-seed',
            str(int(self.config.start_seed)),
            '--max-trials',
            str(int(self.config.max_trials)),
            '--output-dir',
            str(output_dir),
        ]
        if self.config.scene_profile is not None:
            command.extend(['--scene-profiles', self.config.scene_profile])
        if self.config.headless:
            command.append('--headless')
        if self.config.record_live_video:
            command.append('--record-live-video')
            command.extend(['--live-video-fps', str(int(self.config.live_video_fps))])
            command.extend(['--live-video-frame-stride', str(int(self.config.live_video_frame_stride))])
        if self.config.keep_video_frames:
            command.append('--keep-video-frames')
        if self.config.runtime_robochecker:
            command.append('--runtime-robochecker')
            command.extend(['--runtime-checker-stride', str(int(self.config.runtime_checker_stride))])
            command.extend(['--runtime-rgb-frame-stride', str(int(self.config.runtime_rgb_frame_stride))])
            if self.config.runtime_stop_on_violation:
                command.append('--runtime-stop-on-violation')
            if self.config.runtime_capture_rgb:
                command.append('--runtime-capture-rgb')
            if runtime_feedback_path is not None:
                command.extend(['--runtime-feedback-path', str(runtime_feedback_path)])
            if runtime_observation_dir is not None:
                command.extend(['--runtime-observation-dir', str(runtime_observation_dir)])
        return command

    def _try_local_replan(
        self,
        *,
        task_instruction: str,
        plan: RoboBrainPlan,
        recipe: dict[str, Any],
        runtime_feedback: dict[str, Any] | None,
        replan_attempt: int,
    ) -> tuple[RoboBrainRunResult | None, dict[str, Any] | None]:
        local_replan = LocalReplanner().replan(
            plan=plan,
            recipe=recipe,
            runtime_feedback=runtime_feedback,
            attempt=replan_attempt,
        )
        if local_replan is None:
            return None, None

        check_result = self.checker.check(plan=local_replan.plan, recipe=local_replan.recipe)
        run_dir = self._run_dir_for(
            task_instruction=task_instruction,
            plan=local_replan.plan,
            attempt=f'local{replan_attempt}',
        )
        bundle_paths = write_recipe_bundle(
            output_dir=run_dir,
            plan=local_replan.plan,
            recipe=local_replan.recipe,
            check_result=check_result,
        )
        (run_dir / 'local_replan_report.json').write_text(
            json.dumps(local_replan.report, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        if not check_result.ok:
            (run_dir / 'local_replan_rejected.json').write_text(
                json.dumps(check_result.to_dict(), indent=2, ensure_ascii=False),
                encoding='utf-8',
            )
            return None, None

        demo_output_dir = run_dir / 'demo'
        runtime_feedback_path = run_dir / 'runtime_feedback.json'
        runtime_observation_dir = run_dir / 'runtime_observations'
        demo_command = self._demo_command(
            recipe_path=bundle_paths['recipe'],
            output_dir=demo_output_dir,
            runtime_feedback_path=runtime_feedback_path,
            runtime_observation_dir=runtime_observation_dir,
        )
        (run_dir / 'demo_command.json').write_text(json.dumps(demo_command, indent=2), encoding='utf-8')

        try:
            subprocess.run(demo_command, check=True)
            local_feedback = self._read_runtime_feedback(runtime_feedback_path)
            return (
                RoboBrainRunResult(
                    plan=local_replan.plan,
                    recipe=local_replan.recipe,
                    check_result=check_result,
                    bundle_paths=bundle_paths,
                    demo_output_dir=demo_output_dir,
                    demo_command=demo_command,
                    runtime_feedback=local_feedback,
                    replan_attempt=replan_attempt,
                    local_replan_report=local_replan.report,
                ),
                local_feedback,
            )
        except subprocess.CalledProcessError as exc:
            local_feedback = self._read_runtime_feedback(runtime_feedback_path)
            (run_dir / 'execution_error.json').write_text(
                json.dumps(
                    {
                        'returncode': exc.returncode,
                        'command': demo_command,
                        'message': 'Local runtime replan failed; falling back to LLM re-planning.',
                        'runtime_feedback': local_feedback,
                        'replan_attempt': replan_attempt,
                        'local_replan_report': local_replan.report,
                    },
                    indent=2,
                ),
                encoding='utf-8',
            )
            return None, local_feedback

    @staticmethod
    def _read_runtime_feedback(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _feedback_strings_from_runtime(runtime_feedback: dict[str, Any] | None) -> list[str]:
        if not runtime_feedback:
            return []
        messages = []
        for item in runtime_feedback.get('feedback', []):
            if not isinstance(item, dict):
                continue
            message = item.get('message')
            validation = item.get('validation')
            phase = item.get('phase')
            step_index = item.get('step_index')
            evidence = item.get('evidence')
            messages.append(
                json.dumps(
                    {
                        'validation': validation,
                        'phase': phase,
                        'step_index': step_index,
                        'message': message,
                        'evidence': evidence,
                        'image_paths': item.get('image_paths', []),
                    },
                    ensure_ascii=False,
                )
            )
        return messages

    @staticmethod
    def _feedback_images_from_runtime(runtime_feedback: dict[str, Any] | None) -> list[str]:
        if not runtime_feedback:
            return []
        images = []
        for image_path in runtime_feedback.get('latest_images', []):
            if isinstance(image_path, str) and Path(image_path).exists():
                images.append(image_path)
        for item in runtime_feedback.get('feedback', []):
            if not isinstance(item, dict):
                continue
            for image_path in item.get('image_paths', []):
                if isinstance(image_path, str) and Path(image_path).exists() and image_path not in images:
                    images.append(image_path)
        return images[:8]
