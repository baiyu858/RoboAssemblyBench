from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkits.factory_dual_franka_assembly import export_lerobot as legacy_export


if __name__ == '__main__':
    if '--render-single-episode-isaac' in sys.argv:
        legacy_export._render_single_episode_isaac_cli()
    else:
        legacy_export.main()
