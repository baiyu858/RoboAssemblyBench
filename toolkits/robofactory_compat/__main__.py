from __future__ import annotations

import sys

from toolkits.robofactory_compat.config_adapter import main as normalize_main
from toolkits.robofactory_compat.export_dataset import main as export_main


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {'-h', '--help'}:
        print(
            'Usage:\n'
            '  python -m toolkits.robofactory_compat normalize [args...]\n'
            '  python -m toolkits.robofactory_compat export [args...]\n'
        )
        return 0

    command = argv.pop(0)
    if command == 'normalize':
        return normalize_main(argv)
    if command == 'export':
        return export_main(argv)

    raise SystemExit(f'Unknown subcommand {command!r}. Expected one of: normalize, export')


if __name__ == '__main__':
    raise SystemExit(main())
