from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_VISUAL_LABELS = (
    'franka left gripper',
    'franka right gripper',
    'peg',
    'socket',
    'hole',
    'barrier',
    'fixture',
    'workbench',
    'button',
    'box',
)


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def _dedupe_labels(labels: list[str]) -> list[str]:
    deduped = []
    for label in labels:
        text = str(label).strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def labels_from_grounding(
    *,
    task_instruction: str | None = None,
    state_observations: list[dict[str, Any]] | None = None,
    extra_labels: list[str] | None = None,
) -> list[str]:
    labels = list(extra_labels or [])
    labels.extend(DEFAULT_VISUAL_LABELS)
    for state in state_observations or []:
        labels.extend((state.get('objects') or {}).keys())
        labels.extend((state.get('robots') or {}).keys())
        for robot_state in (state.get('robots') or {}).values():
            if isinstance(robot_state, dict) and robot_state.get('target_name'):
                labels.append(str(robot_state['target_name']).replace('_', ' '))
    for token in str(task_instruction or '').replace('_', ' ').split():
        token = token.strip(',.!?;:()[]{}"\'').lower()
        if len(token) >= 4 and token not in {'with', 'from', 'into', 'then', 'task'}:
            labels.append(token)
    return _dedupe_labels(labels)[:48]


@dataclass
class VisionBackendConfig:
    backend: str = 'local'
    labels: list[str] = field(default_factory=list)
    score_threshold: float = 0.20
    max_detections: int = 16
    detector_model: str | None = None
    sam_checkpoint: str | None = None
    sam_model_type: str = 'vit_h'
    sam2_checkpoint: str | None = None
    sam2_config: str | None = None
    device: str | None = None


class FoundationVisionGrounder:
    """Optional visual foundation-model backend for RoboBrain perception.

    Backends:
    - local: no model dependency; color/connected-component proposals.
    - owlvit: Hugging Face zero-shot object detection pipeline.
    - groundingdino: Hugging Face zero-shot object detection pipeline with a GroundingDINO model.
    - grounded-sam: zero-shot detector plus SAM/SAM2 masks when configured.

    Heavy dependencies are imported lazily so RoboBrain remains usable without installing them.
    """

    def __init__(self, config: VisionBackendConfig | None = None):
        self.config = config or VisionBackendConfig()

    def ground_images(
        self,
        *,
        image_paths: list[str],
        labels: list[str],
    ) -> dict[str, Any]:
        backend = str(self.config.backend or 'local').lower().replace('_', '-')
        labels = _dedupe_labels([*self.config.labels, *labels])
        if backend in {'none', 'off', 'disabled'}:
            return {'backend': backend, 'enabled': False, 'labels': labels, 'detections': [], 'relations': []}

        if backend == 'local':
            result = self._ground_local(image_paths=image_paths, labels=labels)
        elif backend in {'owlvit', 'owl-vit', 'groundingdino', 'grounding-dino'}:
            model_kind = 'groundingdino' if backend in {'groundingdino', 'grounding-dino'} else 'owlvit'
            result = self._ground_hf_detector(image_paths=image_paths, labels=labels, model_kind=model_kind)
        elif backend in {'grounded-sam', 'groundedsam', 'sam', 'sam2'}:
            detector_kind = os.environ.get('ROBOBRAIN_DETECTOR_KIND', 'groundingdino')
            result = self._ground_hf_detector(image_paths=image_paths, labels=labels, model_kind=detector_kind)
            result['segmentations'] = self._segment_detections(image_paths=image_paths, detections=result.get('detections', []))
            result['backend'] = 'grounded-sam'
        else:
            result = {
                'backend': backend,
                'enabled': False,
                'labels': labels,
                'detections': [],
                'relations': [],
                'errors': [f'Unknown visual backend {backend!r}.'],
            }

        result.setdefault('labels', labels)
        result['relations'] = self._infer_visual_relations(result.get('detections', []), result.get('segmentations', []))
        return _jsonable(result)

    def _ground_local(self, *, image_paths: list[str], labels: list[str]) -> dict[str, Any]:
        detections = []
        errors = []
        for path in image_paths:
            try:
                detections.extend(self._local_color_regions(path))
            except Exception as exc:  # pragma: no cover - defensive for OpenCV edge cases.
                errors.append(f'{path}: {exc}')
        return {
            'backend': 'local',
            'enabled': True,
            'labels': labels,
            'detections': detections[: self.config.max_detections],
            'errors': errors,
            'notes': [
                'Local backend returns color-region proposals without semantic object identity.',
                'Use owlvit, groundingdino, or grounded-sam for open-vocabulary semantic grounding.',
            ],
        }

    def _local_color_regions(self, path: str) -> list[dict[str, Any]]:
        import cv2
        import numpy as np

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            return []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[..., 1]
        value = hsv[..., 2]
        mask = ((saturation > 45) & (value > 35)).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections = []
        image_area = max(width * height, 1)
        for index in range(1, num_labels):
            x, y, w, h, area = stats[index]
            if area < max(48, image_area * 0.0005):
                continue
            component_mask = labels == index
            mean_rgb = np.mean(rgb[component_mask], axis=0)
            detections.append(
                {
                    'label': 'colored_region',
                    'source': 'local_color_region',
                    'score': round(min(float(area) / float(image_area) * 8.0, 1.0), 3),
                    'image_path': path,
                    'bbox_2d': [int(x), int(y), int(x + w), int(y + h)],
                    'center_2d': [round(float(centroids[index][0]), 2), round(float(centroids[index][1]), 2)],
                    'area_px': int(area),
                    'area_ratio': round(float(area) / float(image_area), 5),
                    'mean_rgb': [round(float(item), 2) for item in mean_rgb],
                    'spatial_hint': self._spatial_hint(
                        center_x=float(centroids[index][0]),
                        center_y=float(centroids[index][1]),
                        width=width,
                        height=height,
                    ),
                }
            )
        return sorted(detections, key=lambda item: item.get('area_px', 0), reverse=True)

    def _ground_hf_detector(self, *, image_paths: list[str], labels: list[str], model_kind: str) -> dict[str, Any]:
        try:
            from PIL import Image
            from transformers import pipeline
        except ImportError as exc:
            return {
                'backend': model_kind,
                'enabled': False,
                'labels': labels,
                'detections': [],
                'errors': [f'transformers/Pillow dependency unavailable: {exc}'],
            }

        model_kind = str(model_kind).lower().replace('_', '-')
        default_model = (
            'IDEA-Research/grounding-dino-tiny'
            if model_kind in {'groundingdino', 'grounding-dino'}
            else 'google/owlvit-base-patch32'
        )
        model_id = self.config.detector_model or os.environ.get('ROBOBRAIN_DETECTOR_MODEL') or default_model
        device = self._pipeline_device()
        try:
            detector = pipeline('zero-shot-object-detection', model=model_id, device=device)
        except Exception as exc:
            return {
                'backend': model_kind,
                'enabled': False,
                'model': model_id,
                'labels': labels,
                'detections': [],
                'errors': [f'Could not load zero-shot detector {model_id!r}: {exc}'],
            }

        detections = []
        errors = []
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                result = detector(image, candidate_labels=labels)
                width, height = image.size
            except Exception as exc:
                errors.append(f'{path}: {exc}')
                continue
            for item in result:
                score = float(item.get('score', 0.0))
                if score < float(self.config.score_threshold):
                    continue
                box = item.get('box') or {}
                bbox = [
                    int(round(float(box.get('xmin', 0)))),
                    int(round(float(box.get('ymin', 0)))),
                    int(round(float(box.get('xmax', 0)))),
                    int(round(float(box.get('ymax', 0)))),
                ]
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0
                detections.append(
                    {
                        'label': item.get('label'),
                        'source': model_kind,
                        'model': model_id,
                        'score': round(score, 4),
                        'image_path': path,
                        'bbox_2d': bbox,
                        'center_2d': [round(center_x, 2), round(center_y, 2)],
                        'area_px': max(bbox[2] - bbox[0], 0) * max(bbox[3] - bbox[1], 0),
                        'spatial_hint': self._spatial_hint(
                            center_x=center_x,
                            center_y=center_y,
                            width=width,
                            height=height,
                        ),
                    }
                )
        detections = sorted(detections, key=lambda item: item.get('score', 0.0), reverse=True)[: self.config.max_detections]
        return {
            'backend': model_kind,
            'enabled': True,
            'model': model_id,
            'labels': labels,
            'detections': detections,
            'errors': errors,
        }

    def _segment_detections(self, *, image_paths: list[str], detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        predictor_info = self._load_sam_predictor()
        if not predictor_info.get('enabled'):
            return [{'enabled': False, **predictor_info}]

        import cv2
        import numpy as np

        predictor = predictor_info['predictor']
        predictor_kind = predictor_info['kind']
        segmentations = []
        detections_by_image: dict[str, list[dict[str, Any]]] = {}
        for detection in detections:
            detections_by_image.setdefault(str(detection.get('image_path')), []).append(detection)

        for image_path in image_paths:
            image_detections = detections_by_image.get(str(image_path), [])
            if not image_detections:
                continue
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            try:
                predictor.set_image(image_rgb)
            except Exception as exc:
                segmentations.append({'enabled': False, 'image_path': image_path, 'error': str(exc), 'kind': predictor_kind})
                continue
            for detection in image_detections:
                bbox = np.asarray(detection.get('bbox_2d'), dtype=float)
                try:
                    masks, scores, _ = predictor.predict(box=bbox, multimask_output=True)
                except TypeError:
                    masks, scores, _ = predictor.predict(box=bbox[None, :], multimask_output=True)
                except Exception as exc:
                    segmentations.append(
                        {
                            'enabled': False,
                            'image_path': image_path,
                            'label': detection.get('label'),
                            'error': str(exc),
                            'kind': predictor_kind,
                        }
                    )
                    continue
                if len(masks) == 0:
                    continue
                best_index = int(np.argmax(scores))
                mask = masks[best_index].astype(bool)
                ys, xs = np.where(mask)
                if xs.size == 0 or ys.size == 0:
                    continue
                segmentations.append(
                    {
                        'enabled': True,
                        'kind': predictor_kind,
                        'image_path': image_path,
                        'label': detection.get('label'),
                        'score': detection.get('score'),
                        'mask_score': round(float(scores[best_index]), 4),
                        'bbox_2d': [
                            int(xs.min()),
                            int(ys.min()),
                            int(xs.max()) + 1,
                            int(ys.max()) + 1,
                        ],
                        'center_2d': [round(float(xs.mean()), 2), round(float(ys.mean()), 2)],
                        'mask_area_px': int(mask.sum()),
                    }
                )
        return segmentations

    def _load_sam_predictor(self) -> dict[str, Any]:
        sam2_checkpoint = self.config.sam2_checkpoint or os.environ.get('ROBOBRAIN_SAM2_CHECKPOINT')
        sam2_config = self.config.sam2_config or os.environ.get('ROBOBRAIN_SAM2_CONFIG')
        if sam2_checkpoint and sam2_config:
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                model = build_sam2(sam2_config, sam2_checkpoint, device=self.config.device)
                return {'enabled': True, 'kind': 'sam2', 'predictor': SAM2ImagePredictor(model)}
            except Exception as exc:
                return {'enabled': False, 'kind': 'sam2', 'error': f'Could not load SAM2: {exc}'}

        sam_checkpoint = self.config.sam_checkpoint or os.environ.get('ROBOBRAIN_SAM_CHECKPOINT')
        if sam_checkpoint:
            try:
                from segment_anything import SamPredictor, sam_model_registry

                model_type = self.config.sam_model_type or os.environ.get('ROBOBRAIN_SAM_MODEL_TYPE') or 'vit_h'
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                if self.config.device:
                    sam.to(device=self.config.device)
                return {'enabled': True, 'kind': 'sam', 'predictor': SamPredictor(sam)}
            except Exception as exc:
                return {'enabled': False, 'kind': 'sam', 'error': f'Could not load SAM: {exc}'}

        return {
            'enabled': False,
            'kind': 'sam',
            'error': 'No SAM checkpoint configured. Set ROBOBRAIN_SAM_CHECKPOINT or ROBOBRAIN_SAM2_CHECKPOINT.',
        }

    def _pipeline_device(self):
        device = self.config.device or os.environ.get('ROBOBRAIN_VISION_DEVICE')
        if device is None or str(device).lower() in {'cpu', '-1'}:
            return -1
        if str(device).startswith('cuda'):
            if ':' in str(device):
                try:
                    return int(str(device).split(':', 1)[1])
                except ValueError:
                    return 0
            return 0
        try:
            return int(device)
        except ValueError:
            return -1

    @staticmethod
    def _spatial_hint(*, center_x: float, center_y: float, width: int, height: int) -> str:
        horizontal = 'left' if center_x < width / 3 else 'right' if center_x > 2 * width / 3 else 'center'
        vertical = 'top' if center_y < height / 3 else 'bottom' if center_y > 2 * height / 3 else 'middle'
        return f'{vertical}-{horizontal}'

    @staticmethod
    def _infer_visual_relations(detections: list[dict[str, Any]], segmentations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = [item for item in segmentations if item.get('enabled') and item.get('bbox_2d')]
        if not items:
            items = [item for item in detections if item.get('bbox_2d')]
        relations = []
        for index, lhs in enumerate(items):
            for rhs in items[index + 1 :]:
                if lhs.get('image_path') != rhs.get('image_path'):
                    continue
                relation = FoundationVisionGrounder._bbox_relation(lhs.get('bbox_2d'), rhs.get('bbox_2d'))
                if relation:
                    relations.append(
                        {
                            'relation': relation,
                            'subject': lhs.get('label'),
                            'object': rhs.get('label'),
                            'image_path': lhs.get('image_path'),
                            'confidence': round(float(lhs.get('score', 0.5)) * float(rhs.get('score', 0.5)), 3),
                        }
                    )
        return relations[:32]

    @staticmethod
    def _bbox_relation(lhs_bbox, rhs_bbox) -> str | None:
        if not lhs_bbox or not rhs_bbox:
            return None
        lx1, ly1, lx2, ly2 = [float(item) for item in lhs_bbox]
        rx1, ry1, rx2, ry2 = [float(item) for item in rhs_bbox]
        inter_x1 = max(lx1, rx1)
        inter_y1 = max(ly1, ry1)
        inter_x2 = min(lx2, rx2)
        inter_y2 = min(ly2, ry2)
        inter_area = max(inter_x2 - inter_x1, 0.0) * max(inter_y2 - inter_y1, 0.0)
        lhs_area = max(lx2 - lx1, 0.0) * max(ly2 - ly1, 0.0)
        rhs_area = max(rx2 - rx1, 0.0) * max(ry2 - ry1, 0.0)
        min_area = max(min(lhs_area, rhs_area), 1.0)
        if inter_area / min_area > 0.2:
            return 'overlapping'
        lhs_center = ((lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0)
        rhs_center = ((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0)
        center_distance = ((lhs_center[0] - rhs_center[0]) ** 2 + (lhs_center[1] - rhs_center[1]) ** 2) ** 0.5
        average_size = max(((lhs_area ** 0.5) + (rhs_area ** 0.5)) / 2.0, 1.0)
        if center_distance / average_size < 1.4:
            return 'near'
        return None
