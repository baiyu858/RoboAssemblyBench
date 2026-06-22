#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

try:
    from pxr import PhysxSchema
except Exception:  # noqa: BLE001
    PhysxSchema = None


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _ops(prim):
    return {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}


def set_transform(stage: Usd.Stage, prim_path: str, transform: dict) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    ops = _ops(prim)
    translate = transform.get("translation")
    if translate is not None:
        value = Gf.Vec3d(*[float(v) for v in translate])
        if "xformOp:translate" in ops:
            ops["xformOp:translate"].Set(value)
        else:
            xf.AddTranslateOp().Set(value)
    rotate = transform.get("rotation_xyz_degrees")
    if rotate is not None:
        value = Gf.Vec3d(*[float(v) for v in rotate])
        if "xformOp:rotateXYZ" in ops:
            ops["xformOp:rotateXYZ"].Set(value)
        else:
            xf.AddRotateXYZOp().Set(value)
    scale = transform.get("scale")
    if scale is not None:
        value = Gf.Vec3d(*[float(v) for v in scale])
        if "xformOp:scale" in ops:
            ops["xformOp:scale"].Set(value)
        else:
            xf.AddScaleOp().Set(value)


def define_black_board(stage: Usd.Stage, root: Path, board: dict) -> None:
    prim_path = board["prim_path"]
    obj_path = _resolve(root, board["asset_path"])
    obj_to_m = float(board.get("obj_to_m", 0.01))
    board_scale = float(board.get("board_scale", 1.0))
    translation = board["translation"]

    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    materials = stage.GetPrimAtPath("/World/Materials")
    if not materials or not materials.IsValid():
        stage.DefinePrim("/World/Materials", "Xform")
    mat = UsdShade.Material.Define(stage, "/World/Materials/BlackOpticalBoard_Material")
    shader = UsdShade.Shader.Define(stage, "/World/Materials/BlackOpticalBoard_Material/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    visual = board.get("visual_material", {})
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*[float(v) for v in visual.get("diffuse_color", [0.005, 0.005, 0.004])])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(visual.get("roughness", 0.72)))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(visual.get("metallic", 0.0)))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = board.get("physics", {})
    phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys.CreateStaticFrictionAttr(float(physics.get("static_friction", 0.3)))
    phys.CreateDynamicFrictionAttr(float(physics.get("dynamic_friction", 0.3)))
    phys.CreateRestitutionAttr(float(physics.get("restitution", 0.0)))
    phys.CreateDensityAttr(float(physics.get("density_kg_per_m3", 1000.0)))

    mesh = trimesh.load(str(obj_path), force="mesh")
    vertices = mesh.vertices * obj_to_m * board_scale
    faces = mesh.faces.astype(int)
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    xform.AddRotateXYZOp().Set(Gf.Vec3d(*[float(v) for v in board.get("rotation_xyz_degrees", [0, 0, 0])]))
    xform.AddScaleOp().Set(Gf.Vec3d(*[float(v) for v in board.get("scale", [1, 1, 1])]))
    mesh_prim = UsdGeom.Mesh.Define(stage, f"{prim_path}/mesh")
    mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh_prim.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in vertices])
    mesh_prim.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh_prim.CreateFaceVertexIndicesAttr([int(i) for tri in faces for i in tri])
    mesh_prim.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in visual.get("diffuse_color", [0.005, 0.005, 0.004])])])
    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim()).Bind(mat)
    col = UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
    col.CreateCollisionEnabledAttr(bool(physics.get("collision_enabled", True)))
    try:
        mesh_col = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim.GetPrim())
        if hasattr(mesh_col, "CreateApproximationAttr"):
            mesh_col.CreateApproximationAttr().Set("none")
    except Exception:  # noqa: BLE001
        pass
    if PhysxSchema is not None:
        try:
            physx_col = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim.GetPrim())
            if hasattr(physx_col, "CreateContactOffsetAttr"):
                physx_col.CreateContactOffsetAttr(0.002)
            if hasattr(physx_col, "CreateRestOffsetAttr"):
                physx_col.CreateRestOffsetAttr(0.0)
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply scene_spec.json transforms/support-surface edits to scene.usda.")
    parser.add_argument("--spec", default="scene/scene_spec.json")
    parser.add_argument("--usd", default="scene/scene.usda")
    parser.add_argument("--output", default="", help="Optional output USDA/USD path. Defaults to in-place update.")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    spec_path = _resolve(package_root, args.spec)
    usd_path = _resolve(package_root, args.usd)
    out_path = _resolve(package_root, args.output) if args.output else usd_path
    spec = json.loads(spec_path.read_text())

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    for board in spec.get("support_surfaces", []):
        if board.get("id") == "black_fabrica_optical_board":
            define_black_board(stage, package_root, board)

    graph = spec.get("scene_graph", {}).get("objects", [])
    for obj in graph:
        prim_path = obj.get("prim_path")
        transform = obj.get("transform")
        if prim_path and transform and stage.GetPrimAtPath(prim_path).IsValid():
            set_transform(stage, prim_path, transform)

    if str(out_path) != str(usd_path):
        stage.GetRootLayer().Export(str(out_path))
    else:
        stage.GetRootLayer().Save()
    print(f"Updated USD from JSON spec: {out_path}")


if __name__ == "__main__":
    main()
