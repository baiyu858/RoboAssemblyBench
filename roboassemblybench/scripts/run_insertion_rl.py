"""CLI entry for the reusable contact-insertion RL library."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from roboassemblybench.core.rl.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
