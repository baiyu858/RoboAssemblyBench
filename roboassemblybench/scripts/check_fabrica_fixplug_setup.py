#!/usr/bin/env python3
"""Check whether the official Fabrica FixPlug cooling-manifold baseline is runnable."""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path


COOLING_MANIFOLD_PAIRS = (("0", "1"), ("2", "1"), ("3", "1"), ("4", "1"), ("5", "1"), ("6", "1"))


def _install_numpy_pickle_compat() -> None:
    """Allow NumPy 2 pickles that reference numpy._core to load on NumPy 1.x."""
    try:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric
    except Exception:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy_numeric)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _module_status(name: str) -> str:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:  # pragma: no cover - defensive import check
        return f"ERR {exc!r}"
    if spec is None:
        return "MISSING"
    return f"OK {spec.origin or '<namespace>'}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="", help="Optional rl-games checkpoint path to validate.")
    args = parser.parse_args()

    root = _repo_root()
    fabrica_root = root / "third_part" / "Fabrica"
    learning_root = fabrica_root / "learning"
    isaacgymenvs_root = learning_root / "isaacgymenvs"
    plan_info_path = isaacgymenvs_root / "tasks" / "fabrica" / "data" / "plan_info" / "cooling_manifold.pkl"
    urdf_dir = learning_root / "assets" / "fabrica" / "urdf" / "fabrica_franka"

    print(f"python: {sys.executable}")
    print(f"repo: {root}")
    print(f"fabrica: {fabrica_root} {'OK' if fabrica_root.exists() else 'MISSING'}")
    print(f"isaacgymenvs root: {isaacgymenvs_root} {'OK' if isaacgymenvs_root.exists() else 'MISSING'}")

    missing = False
    for module in ("isaacgym", "isaacgymenvs", "torch", "hydra", "omegaconf", "rl_games"):
        status = _module_status(module)
        print(f"module {module}: {status}")
        if status == "MISSING" and module in {"isaacgym", "isaacgymenvs"}:
            missing = True

    if not plan_info_path.exists():
        print(f"plan_info: MISSING {plan_info_path}")
        missing = True
    else:
        _install_numpy_pickle_compat()
        with plan_info_path.open("rb") as f:
            plan_info = pickle.load(f)
        keys = set(plan_info)
        print(f"plan_info: OK {plan_info_path}")
        print(f"plan_info pairs: {sorted(keys)}")
        for pair in COOLING_MANIFOLD_PAIRS:
            if pair not in keys:
                print(f"missing pair in plan_info: {pair}")
                missing = True

    for plug, socket in COOLING_MANIFOLD_PAIRS:
        urdf_path = urdf_dir / f"cooling_manifold_{plug}_{socket}.urdf"
        print(f"fixplug urdf {plug}->{socket}: {'OK' if urdf_path.exists() else 'MISSING'} {urdf_path}")
        if not urdf_path.exists():
            missing = True

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        print(f"checkpoint: {'OK' if checkpoint_path.exists() else 'MISSING'} {checkpoint_path}")
        if not checkpoint_path.exists():
            missing = True

    if missing:
        print("status: NOT_READY")
        return 1

    print("status: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
