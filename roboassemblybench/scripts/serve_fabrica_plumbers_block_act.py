from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboassemblybench.datasets.cartesian_episode import CAMERA_KEYS, STATE_NAMES
from roboassemblybench.policies.act_rpc import receive_message, send_message


def _load_policy(checkpoint: Path, *, device_name: str, use_amp: bool):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device_name
    config.use_amp = bool(use_amp)
    policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config)
    policy.to(device_name)
    policy.eval()
    device_override = {'device': device_name}
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor': device_override},
        postprocessor_overrides={'device_processor': device_override},
    )
    return torch, config, policy, preprocessor, postprocessor


def _expected_image_shape(config) -> tuple[int, int, int]:
    shapes = set()
    for key in CAMERA_KEYS:
        feature = (config.input_features or {}).get(key)
        shape = tuple(int(value) for value in getattr(feature, 'shape', ()))
        if len(shape) != 3:
            raise ValueError(f'Checkpoint is missing a 3D visual feature for {key}.')
        shapes.add((shape[1], shape[2], shape[0]))
    if len(shapes) != 1:
        raise ValueError(f'Checkpoint camera features do not share one shape: {sorted(shapes)}.')
    return shapes.pop()


def _validate_observation(observation: dict, *, expected_image_shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    if not isinstance(observation, dict):
        raise TypeError('Observation must be a dictionary.')
    validated = {}
    state = np.asarray(observation.get('observation.state'), dtype=np.float32)
    if state.shape != (len(STATE_NAMES),) or not np.all(np.isfinite(state)):
        raise ValueError(f'Invalid policy state shape or values: {state.shape}.')
    validated['observation.state'] = state
    for key in CAMERA_KEYS:
        image = np.asarray(observation.get(key))
        if image.shape != expected_image_shape or image.dtype != np.uint8:
            raise ValueError(f'Invalid {key} image: shape={image.shape}, dtype={image.dtype}.')
        validated[key] = np.ascontiguousarray(image)
    return validated


def _portable_action_payload(action) -> list[float]:
    """Keep the RPC payload independent of the NumPy version in the Isaac process."""

    action = np.asarray(action, dtype=np.float32)
    if action.shape != (16,) or not np.all(np.isfinite(action)):
        raise ValueError(f'ACT produced invalid action: {action}.')
    return [float(value) for value in action]


def main() -> None:
    parser = argparse.ArgumentParser(description='Serve a trained ACT policy to the Isaac evaluator.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--use-amp', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not (checkpoint / 'config.json').is_file():
        raise FileNotFoundError(f'ACT checkpoint is missing config.json: {checkpoint}')
    torch, config, policy, preprocessor, postprocessor = _load_policy(
        checkpoint,
        device_name=str(args.device),
        use_amp=bool(args.use_amp),
    )
    expected_image_shape = _expected_image_shape(config)
    from lerobot.policies.utils import prepare_observation_for_inference

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((str(args.host), int(args.port)))
    server.listen(1)
    print(
        json.dumps(
            {
                'ready': True,
                'host': args.host,
                'port': args.port,
                'checkpoint': str(checkpoint),
                'policy_type': config.type,
                'device': config.device,
                'use_amp': config.use_amp,
            }
        ),
        flush=True,
    )

    shutdown = False
    try:
        while not shutdown:
            connection, _ = server.accept()
            with connection:
                while not shutdown:
                    try:
                        request = receive_message(connection)
                    except ConnectionError:
                        break
                    try:
                        command = str(request.get('command', ''))
                        if command == 'ping':
                            response = {'ok': True, 'policy_type': config.type, 'device': config.device}
                        elif command == 'reset':
                            policy.reset()
                            response = {'ok': True}
                        elif command == 'shutdown':
                            response = {'ok': True}
                            shutdown = True
                        elif command == 'predict':
                            started = time.perf_counter()
                            observation = _validate_observation(
                                request.get('observation'),
                                expected_image_shape=expected_image_shape,
                            )
                            observation = prepare_observation_for_inference(
                                observation,
                                torch.device(config.device),
                                task=str(request.get('task') or ''),
                                robot_type='dual_ur5e_robotiq_2f85',
                            )
                            observation = preprocessor(observation)
                            autocast = (
                                torch.autocast(device_type='cuda')
                                if config.use_amp and str(config.device).startswith('cuda')
                                else nullcontext()
                            )
                            with torch.inference_mode(), autocast:
                                action = postprocessor(policy.select_action(observation))
                            action = _portable_action_payload(action.squeeze(0).detach().cpu())
                            response = {
                                'ok': True,
                                'action': action,
                                'inference_seconds': time.perf_counter() - started,
                            }
                        else:
                            raise ValueError(f'Unknown policy RPC command: {command!r}.')
                    except Exception as exc:
                        response = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
                    send_message(connection, response)
    finally:
        server.close()


if __name__ == '__main__':
    main()
