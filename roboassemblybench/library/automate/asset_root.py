"""Resolve AutoMate assets from the local mirror or Isaac Lab Nucleus.

The upstream AutoMate configs use ``ISAACLAB_NUCLEUS_DIR`` directly.  Keeping
that as the fallback preserves the original behavior, while preferring the
downloaded mirror makes the vendored task runnable without a Nucleus server.
"""

from __future__ import annotations

import os
from pathlib import Path


LOCAL_ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "AutoMate"


def _nucleus_asset_root() -> str:
    try:
        from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    except Exception:
        return str(LOCAL_ASSET_ROOT)
    return f"{ISAACLAB_NUCLEUS_DIR}/AutoMate"


def get_asset_root() -> str:
    """Return the configured AutoMate root, preferring a complete local mirror."""

    override = os.environ.get("ROBOASSEMBLYBENCH_AUTOMATE_ASSET_ROOT")
    if override:
        return str(Path(override).expanduser().resolve())
    if (LOCAL_ASSET_ROOT / "plug_grasps.json").is_file():
        return str(LOCAL_ASSET_ROOT)
    return _nucleus_asset_root()


ASSET_DIR = get_asset_root()
LOCAL_TABLE_USD = LOCAL_ASSET_ROOT / "Table" / "table.usd"
TABLE_USD_PATH = str(LOCAL_TABLE_USD) if LOCAL_TABLE_USD.is_file() else None


def resolve_asset_file(path: str, *, download_dir: str | None = None) -> str:
    """Return a local path for an AutoMate file.

    Local mirrors are returned directly.  Nucleus URLs retain Isaac Lab's
    original lazy-download behavior through ``retrieve_file_path``.
    """

    if os.path.isfile(path):
        return os.path.abspath(path)
    from isaaclab.utils.assets import retrieve_file_path

    return retrieve_file_path(path, download_dir=download_dir)
