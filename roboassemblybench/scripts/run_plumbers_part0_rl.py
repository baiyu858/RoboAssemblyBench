"""Compatibility example; all training/control lives in core.rl."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from roboassemblybench.core.rl.cli import main, full_flow_passed

if __name__ == '__main__':
    sys.argv = [arg.replace('resume-part0', 'resume') for arg in sys.argv]
    if '--recipe' not in sys.argv:
        sys.argv.extend(['--recipe', 'fabrica_plumbers_block_ur5e_staged@rl'])
    if '--stage' not in sys.argv:
        sys.argv.extend(['--stage', '0'])
    raise SystemExit(main())
