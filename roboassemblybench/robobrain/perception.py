from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from roboassemblybench.robobrain.vision_models import (
    FoundationVisionGrounder,
    VisionBackendConfig,
    labels_from_grounding,
)


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


class ObservationGrounder:
    """Turn runtime RGB/state observations into compact planner grounding."""

    def __init__(
        self,
        *,
        max_images: int = 8,
        max_state_files: int = 8,
        visual_backend: str = 'local',
        visual_labels: list[str] | None = None,
        visual_score_threshold: float = 0.20,
        visual_max_detections: int = 16,
        detector_model: str | None = None,
        sam_checkpoint: str | None = None,
        sam_model_type: str = 'vit_h',
        sam2_checkpoint: str | None = None,
        sam2_config: str | None = None,
        visual_device: str | None = None,
        vlm_grounding: bool = False,
        vlm_model: str | None = None,
    ):
        self.max_images = max(int(max_images), 0)
        self.max_state_files = max(int(max_state_files), 0)
        self.visual_backend = str(visual_backend or 'local')
        self.visual_labels = list(visual_labels or [])
        self.visual_score_threshold = float(visual_score_threshold)
        self.visual_max_detections = max(int(visual_max_detections), 1)
        self.detector_model = detector_model
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.visual_device = visual_device
        self.vlm_grounding = bool(vlm_grounding)
        self.vlm_model = vlm_model

    def ground(
        self,
        *,
        observation_images: list[str] | None = None,
        runtime_feedback: dict[str, Any] | None = None,
        task_instruction: str | None = None,
    ) -> dict[str, Any]:
        image_paths = self._collect_image_paths(observation_images=observation_images, runtime_feedback=runtime_feedback)
        state_payloads = self._collect_state_payloads(runtime_feedback=runtime_feedback)
        feedback_items = [
            item for item in (runtime_feedback or {}).get('feedback', []) if isinstance(item, dict)
        ]
        state_observations = [self._summarize_state(payload) for payload in state_payloads[: self.max_state_files]]
        image_observations = [self._summarize_image(path) for path in image_paths[: self.max_images]]
        visual_labels = labels_from_grounding(
            task_instruction=task_instruction,
            state_observations=state_observations,
            extra_labels=self.visual_labels,
        )
        visual_grounding = self._ground_visual_models(image_paths=image_paths[: self.max_images], labels=visual_labels)
        grounding = {
            'image_observations': image_observations,
            'visual_grounding': visual_grounding,
            'state_observations': state_observations,
            'runtime_feedback': [self._summarize_feedback_item(item) for item in feedback_items[-16:]],
        }
        if self.vlm_grounding:
            grounding['vlm_grounding'] = self._ground_vlm(
                image_paths=image_paths[: self.max_images],
                state_grounding={key: grounding[key] for key in ('state_observations', 'runtime_feedback')},
                task_instruction=task_instruction,
            )
        grounding['inferred_relations'] = self._infer_relations(grounding)
        grounding['planner_hints'] = self._planner_hints(grounding)
        return _jsonable(grounding)

    def _collect_image_paths(
        self,
        *,
        observation_images: list[str] | None,
        runtime_feedback: dict[str, Any] | None,
    ) -> list[str]:
        paths = []
        for image_path in observation_images or []:
            if isinstance(image_path, str) and Path(image_path).exists() and image_path not in paths:
                paths.append(image_path)
        if runtime_feedback:
            for image_path in runtime_feedback.get('latest_images', []):
                if isinstance(image_path, str) and Path(image_path).exists() and image_path not in paths:
                    paths.append(image_path)
            for item in runtime_feedback.get('feedback', []):
                if not isinstance(item, dict):
                    continue
                for image_path in item.get('image_paths', []):
                    if isinstance(image_path, str) and Path(image_path).exists() and image_path not in paths:
                        paths.append(image_path)
        return paths

    def _collect_state_payloads(self, *, runtime_feedback: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not runtime_feedback:
            return []
        paths = []
        for state_path in runtime_feedback.get('latest_state_paths', []):
            if isinstance(state_path, str) and Path(state_path).exists() and state_path not in paths:
                paths.append(state_path)
        observation_dir = runtime_feedback.get('observation_dir')
        if observation_dir and Path(observation_dir).exists():
            state_files = sorted(Path(observation_dir).glob('episode_*/state/step_*.json'))
            for state_path in state_files[-self.max_state_files :]:
                text_path = str(state_path)
                if text_path not in paths:
                    paths.append(text_path)

        payloads = []
        for state_path in paths[-self.max_state_files :]:
            try:
                payloads.append(json.loads(Path(state_path).read_text(encoding='utf-8')))
            except (OSError, json.JSONDecodeError):
                continue
        return payloads

    def _summarize_image(self, path: str) -> dict[str, Any]:
        summary = {'path': path, 'exists': Path(path).exists()}
        try:
            import cv2
            import numpy as np
        except ImportError:
            summary['note'] = 'cv2/numpy unavailable; RGB file is passed directly to the VLM planner.'
            return summary

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            summary['readable'] = False
            return summary
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean_rgb = np.mean(rgb.reshape(-1, 3), axis=0)
        summary.update(
            {
                'readable': True,
                'width': int(rgb.shape[1]),
                'height': int(rgb.shape[0]),
                'mean_rgb': [round(float(item), 2) for item in mean_rgb],
                'brightness': round(float(np.mean(mean_rgb)), 2),
                'dominant_rgb': [int(item) for item in self._dominant_rgb(rgb)],
            }
        )
        return summary

    def _ground_visual_models(self, *, image_paths: list[str], labels: list[str]) -> dict[str, Any]:
        if not image_paths:
            return {'backend': self.visual_backend, 'enabled': False, 'labels': labels, 'detections': [], 'relations': []}
        config = VisionBackendConfig(
            backend=self.visual_backend,
            labels=labels,
            score_threshold=self.visual_score_threshold,
            max_detections=self.visual_max_detections,
            detector_model=self.detector_model or os.environ.get('ROBOBRAIN_DETECTOR_MODEL'),
            sam_checkpoint=self.sam_checkpoint or os.environ.get('ROBOBRAIN_SAM_CHECKPOINT'),
            sam_model_type=self.sam_model_type,
            sam2_checkpoint=self.sam2_checkpoint or os.environ.get('ROBOBRAIN_SAM2_CHECKPOINT'),
            sam2_config=self.sam2_config or os.environ.get('ROBOBRAIN_SAM2_CONFIG'),
            device=self.visual_device or os.environ.get('ROBOBRAIN_VISION_DEVICE'),
        )
        return FoundationVisionGrounder(config).ground_images(image_paths=image_paths, labels=labels)

    def _ground_vlm(
        self,
        *,
        image_paths: list[str],
        state_grounding: dict[str, Any],
        task_instruction: str | None,
    ) -> dict[str, Any]:
        if not image_paths:
            return {'enabled': False, 'reason': 'No images available for VLM grounding.'}
        try:
            from roboassemblybench.robobrain.llm import OpenAIVisualGroundingClient

            return {
                'enabled': True,
                'result': OpenAIVisualGroundingClient(model=self.vlm_model).ground(
                    image_paths=image_paths,
                    state_grounding=state_grounding,
                    task_instruction=task_instruction,
                ),
            }
        except Exception as exc:
            return {'enabled': False, 'error': str(exc)}

    @staticmethod
    def _dominant_rgb(rgb):
        import numpy as np

        if rgb.size == 0:
            return [0, 0, 0]
        sample = rgb.reshape(-1, 3)
        stride = max(int(len(sample) / 4096), 1)
        sample = sample[::stride]
        bins = (sample // 32).astype(int)
        keys, counts = np.unique(bins, axis=0, return_counts=True)
        dominant = keys[int(np.argmax(counts))] * 32 + 16
        return np.clip(dominant, 0, 255).astype(int).tolist()

    def _summarize_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        tracked_objects = payload.get('tracked_objects') or {}
        tracked_robots = payload.get('tracked_robots') or {}
        objects = {}
        for name, state in tracked_objects.items():
            if not isinstance(state, dict):
                continue
            objects[name] = {
                'position': state.get('position'),
                'orientation': state.get('orientation'),
                'status': state.get('status'),
                'attached_to': state.get('attached_to'),
                'grasped_by': state.get('grasped_by'),
                'locked_target': state.get('locked_target'),
                'target_reached': state.get('target_reached'),
                'position_error': state.get('position_error'),
                'linear_speed': state.get('linear_speed'),
                'angular_speed': state.get('angular_speed'),
                'physical_hold_valid': (state.get('attachment') or {}).get('physical_hold_valid')
                if isinstance(state.get('attachment'), dict)
                else None,
            }
        robots = {}
        for name, state in tracked_robots.items():
            if not isinstance(state, dict):
                continue
            robots[name] = {
                'position': state.get('position'),
                'target_name': state.get('target_name'),
                'target_reached': state.get('target_reached'),
                'position_error': state.get('position_error'),
                'gripper_opening': state.get('gripper_opening'),
            }
        return {
            'path': payload.get('path'),
            'episode_idx': payload.get('episode_idx'),
            'step_index': payload.get('step_index'),
            'phase': payload.get('phase'),
            'runtime_state': payload.get('runtime_state', {}),
            'objects': objects,
            'robots': robots,
            'feedback': payload.get('feedback', []),
        }

    @staticmethod
    def _summarize_feedback_item(item: dict[str, Any]) -> dict[str, Any]:
        return {
            'severity': item.get('severity'),
            'validation': item.get('validation'),
            'phase': item.get('phase'),
            'step_index': item.get('step_index'),
            'message': item.get('message'),
            'evidence': item.get('evidence'),
            'image_paths': item.get('image_paths', []),
        }

    def _infer_relations(self, grounding: dict[str, Any]) -> list[dict[str, Any]]:
        relations = []
        for state in grounding.get('state_observations', []):
            phase = state.get('phase')
            for object_name, object_state in (state.get('objects') or {}).items():
                attached_to = object_state.get('attached_to')
                if attached_to:
                    relations.append({'relation': 'attached', 'object': object_name, 'robot': attached_to, 'phase': phase})
                if object_state.get('physical_hold_valid') is False:
                    relations.append({'relation': 'invalid_physical_hold', 'object': object_name, 'phase': phase})
                if object_state.get('target_reached') is False and object_state.get('position_error') is not None:
                    relations.append(
                        {
                            'relation': 'object_not_at_target',
                            'object': object_name,
                            'position_error': object_state.get('position_error'),
                            'phase': phase,
                        }
                    )
            for robot_name, robot_state in (state.get('robots') or {}).items():
                if robot_state.get('target_reached') is False and robot_state.get('position_error') is not None:
                    relations.append(
                        {
                            'relation': 'robot_not_at_target',
                            'robot': robot_name,
                            'target': robot_state.get('target_name'),
                            'position_error': robot_state.get('position_error'),
                            'phase': phase,
                        }
                    )
        for relation in (grounding.get('visual_grounding') or {}).get('relations', []):
            if isinstance(relation, dict):
                relations.append({'relation': f"visual_{relation.get('relation')}", **relation})
        for item in (grounding.get('visual_grounding') or {}).get('detections', []):
            if isinstance(item, dict):
                relations.append(
                    {
                        'relation': 'visual_detection',
                        'label': item.get('label'),
                        'image_path': item.get('image_path'),
                        'bbox_2d': item.get('bbox_2d'),
                        'center_2d': item.get('center_2d'),
                        'score': item.get('score'),
                        'source': item.get('source'),
                    }
                )
        return relations[-32:]

    @staticmethod
    def _planner_hints(grounding: dict[str, Any]) -> list[str]:
        hints = []
        validations = {
            item.get('validation')
            for item in grounding.get('runtime_feedback', [])
            if isinstance(item, dict) and item.get('validation')
        }
        if 'Validate_Spatial_Occupancy' in validations:
            hints.append('Widen approach lanes or stagger dual-arm motion before retrying.')
        if 'Validate_Interaction' in validations:
            hints.append('Repair grasp/contact phases before transport; add a safer approach and stronger attach checks.')
        if 'Validate_Scheduling' in validations:
            hints.append('Increase timeout budgets or add intermediate waypoints for the failing phase.')
        visual_grounding = grounding.get('visual_grounding') or {}
        if visual_grounding.get('detections'):
            hints.append('Use visual detections/masks to adjust object-relative approach targets and verify occlusion/contact.')
        if (grounding.get('vlm_grounding') or {}).get('enabled'):
            hints.append('Use VLM grounding result to refine object identity, scene relations, and recovery strategy.')
        if not hints and grounding.get('image_observations'):
            hints.append('Use supplied RGB observations to ground object placement and visible failure context.')
        return hints
