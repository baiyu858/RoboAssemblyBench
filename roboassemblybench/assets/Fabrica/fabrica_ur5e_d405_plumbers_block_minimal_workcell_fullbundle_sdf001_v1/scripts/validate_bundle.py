#!/usr/bin/env python3
from pathlib import Path
from pxr import Usd, UsdGeom

root = Path(__file__).resolve().parents[1]
scene = root / "scene" / "scene.usda"
stage = Usd.Stage.Open(str(scene))
if not stage:
    raise SystemExit(f"Failed to open {scene}")
print(f"Opened: {scene}")
print(f"Prim count: {sum(1 for _ in stage.Traverse())}")

cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
)

required_prims = [
    "/World/Objects/Robots/UR5ELeft",
    "/World/Objects/Robots/UR5ERight",
    "/World/Objects/Fabrica/PlumbersBlockRecipe",
    "/World/Objects/Fabrica/PlumbersBlockRecipe/FabricaOfficialFrame/PickupParts",
    "/World/Objects/Fabrica/PlumbersBlockRecipe/FabricaOfficialFrame/OpticalBoard",
    "/World/Objects/Fabrica/PlumbersBlockRecipe/FabricaOfficialFrame/PickupFixture",
    "/World/Objects/Workbenches/MainWorkbench",
]
removed_prims = [
    "/World/Background",
    "/World/Objects/Robots/Go2ByWorkbench",
    "/World/Objects/Robots/G2Omnipicker",
    "/World/Objects/Trays",
    "/World/Objects/Workbenches/DistantWorkbench",
    "/World/Objects/Robots/FrankaLeft",
    "/World/Objects/Robots/FrankaRight",
]

for prim_path in required_prims:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        print(f"MISSING {prim_path}")
        continue
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    mn, mx = box.GetMin(), box.GetMax()
    print(
        prim_path,
        "min=", [round(float(x), 4) for x in mn],
        "max=", [round(float(x), 4) for x in mx],
    )

for prim_path in removed_prims:
    if stage.GetPrimAtPath(prim_path):
        raise SystemExit(f"Unexpected remaining prim: {prim_path}")
print("Minimal bundle validation passed.")
