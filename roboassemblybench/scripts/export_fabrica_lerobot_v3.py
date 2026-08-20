from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboassemblybench.scripts.export_fabrica_plumbers_block_lerobot_v3 import export_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Export one Fabrica task collection to LeRobot v3.')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--max-episodes', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-streaming-encoding', action='store_true')
    parser.add_argument('--encoder-threads', type=int, default=2)
    parser.add_argument('--vcodec', default='h264')
    args = parser.parse_args()

    summary = export_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        repo_id=str(args.repo_id),
        max_episodes=args.max_episodes,
        resume=bool(args.resume),
        streaming_encoding=not bool(args.no_streaming_encoding),
        encoder_threads=max(int(args.encoder_threads), 1),
        vcodec=str(args.vcodec),
        require_extended_observations=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
