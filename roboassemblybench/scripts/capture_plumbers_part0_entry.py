"""Compatibility example for the generic insertion capture command."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from roboassemblybench.core.rl.cli import main

if __name__ == '__main__':
    sys.argv.extend(['--mode', 'capture', '--recipe', 'fabrica_plumbers_block_ur5e_staged@rl', '--stage', '0'])
    raise SystemExit(main())
