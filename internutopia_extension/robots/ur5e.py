from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, List, Optional

import numpy as np

from internutopia.core.robot.isaacsim.articulation import IsaacsimArticulation
from internutopia.core.robot.rigid_body import IRigidBody
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia.core.util import log
from internutopia_extension.configs.robots.ur5e import (
    DEFAULT_UR5E_READY_JOINTS,
    UR5eRobotCfg,
)

_UR5E_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class UR5e(IsaacsimArticulation):
    def __init__(
        self,
        prim_path: str,
        name: str = "ur5e_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_name: Optional[str] = None,
        gripper_open_position: Optional[float] = None,
        gripper_closed_position: Optional[float] = None,
        gripper_xform_orient: Optional[List[float]] = None,
        gripper_mount_local_pos0: Optional[List[float]] = None,
        gripper_mount_local_pos1: Optional[List[float]] = None,
        gripper_mount_local_rot0: Optional[List[float]] = None,
        gripper_mount_local_rot1: Optional[List[float]] = None,
        configure_gripper_mount_joint: bool = True,
        gripper_base_link_path: Optional[str] = None,
        gripper_container_path: Optional[str] = None,
        gripper_container_orient: Optional[List[float]] = None,
        author_gripper_collision_pads: bool = True,
        deltas: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
    ) -> None:
        from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper

        self._end_effector = None
        self._gripper = None
        self._root_prim_path = prim_path
        self._start_position = None if position is None else np.asarray(position, dtype=float)
        self._start_orientation = None if orientation is None else np.asarray(orientation, dtype=float)
        self._end_effector_prim_name = end_effector_prim_name or "tool0"
        self._end_effector_prim_path = prim_path + "/" + self._end_effector_prim_name
        self._gripper_dof_name = gripper_dof_name or "finger_joint"
        if gripper_open_position is None:
            gripper_open_position = 0.0
        if gripper_closed_position is None:
            gripper_closed_position = 0.80
        if deltas is None:
            delta_sign = 1.0 if float(gripper_open_position) > float(gripper_closed_position) else -1.0
            deltas = np.array([0.04 * delta_sign], dtype=float)
        self._gripper_open_position = float(gripper_open_position)
        self._gripper_closed_position = float(gripper_closed_position)
        self._gripper_deltas = np.asarray(deltas, dtype=float)
        self._gripper_xform_orient = gripper_xform_orient
        self._gripper_mount_local_pos0 = gripper_mount_local_pos0
        self._gripper_mount_local_pos1 = gripper_mount_local_pos1
        self._gripper_mount_local_rot0 = gripper_mount_local_rot0
        self._gripper_mount_local_rot1 = gripper_mount_local_rot1
        self._configure_gripper_mount_joint = bool(configure_gripper_mount_joint)
        self._gripper_base_link_path_override = gripper_base_link_path
        self._gripper_container_path_override = gripper_container_path
        self._gripper_container_orient = gripper_container_orient
        self._author_gripper_collision_pads_enabled = bool(author_gripper_collision_pads)

        super().__init__(
            usd_path=usd_path,
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
            scale=scale,
        )
        self._author_root_xform_pose()
        self._author_gripper_mount_pose()
        self._author_gripper_visuals_visible()
        self._author_gripper_collision_pads()
        self._resolve_end_effector_prim_path()
        self._gripper = self._make_gripper(ParallelGripper)

    def _author_root_xform_pose(self) -> None:
        if self._start_position is None and self._start_orientation is None:
            return
        try:
            from isaacsim.core.utils.prims import get_prim_at_path
            from pxr import Gf, UsdGeom
        except Exception:
            try:
                from omni.isaac.core.utils.prims import get_prim_at_path
                from pxr import Gf, UsdGeom
            except Exception:
                return

        try:
            prim = get_prim_at_path(self._root_prim_path)
        except Exception:
            return
        if prim is None or not prim.IsValid():
            return

        xformable = UsdGeom.Xformable(prim)
        if self._start_position is not None:
            value = Gf.Vec3d(*(float(component) for component in self._start_position))
            for op in xformable.GetOrderedXformOps():
                if op.GetOpName() == "xformOp:translate":
                    op.Set(value)
                    break
            else:
                xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(value)

        if self._start_orientation is not None:
            quat = [float(component) for component in self._start_orientation]
            value_d = Gf.Quatd(quat[0], quat[1], quat[2], quat[3])
            for op in xformable.GetOrderedXformOps():
                if op.GetOpName() == "xformOp:orient":
                    if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                        op.Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))
                    else:
                        op.Set(value_d)
                    break
            else:
                xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(value_d)

    def _author_gripper_mount_pose(self) -> None:
        if not any(
            value is not None
            for value in (
                self._gripper_xform_orient,
                self._gripper_container_orient,
                self._gripper_mount_local_pos0,
                self._gripper_mount_local_pos1,
                self._gripper_mount_local_rot0,
                self._gripper_mount_local_rot1,
            )
        ):
            return
        try:
            from isaacsim.core.utils.prims import get_prim_at_path
            from pxr import Gf, Sdf, UsdGeom
        except Exception:
            try:
                from omni.isaac.core.utils.prims import get_prim_at_path
                from pxr import Gf, Sdf, UsdGeom
            except Exception:
                return

        def _set_or_add_orient(prim_path: str, quat_wxyz):
            prim = get_prim_at_path(prim_path)
            if prim is None or not prim.IsValid():
                return
            quat = [float(component) for component in quat_wxyz]
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpName() == "xformOp:orient":
                    if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                        op.Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))
                    else:
                        op.Set(Gf.Quatd(quat[0], quat[1], quat[2], quat[3]))
                    return
            xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(quat[0], quat[1], quat[2], quat[3])
            )

        def _set_or_add_translate(prim_path: str, xyz):
            prim = get_prim_at_path(prim_path)
            if prim is None or not prim.IsValid():
                return
            value = Gf.Vec3d(*(float(component) for component in xyz))
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpName() == "xformOp:translate":
                    op.Set(value)
                    return
            xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(value)

        def _quat_from_attr(prim, name: str, fallback):
            attr = prim.GetAttribute(name)
            if not attr or not attr.IsValid():
                return fallback
            value = attr.Get()
            if value is None:
                return fallback
            return Gf.Quatd(float(value.GetReal()), *(float(component) for component in value.GetImaginary()))

        def _vec3_from_attr(prim, name: str, fallback):
            attr = prim.GetAttribute(name)
            if not attr or not attr.IsValid():
                return fallback
            value = attr.Get()
            if value is None:
                return fallback
            return Gf.Vec3d(*(float(component) for component in value))

        def _rotate_vec(quat, vec):
            return quat.GetNormalized().Transform(vec)

        def _align_reset_gripper_to_mount(gripper_xform_path: str, joint_prim) -> None:
            gripper_prim = get_prim_at_path(gripper_xform_path)
            if gripper_prim is None or not gripper_prim.IsValid():
                return
            xformable = UsdGeom.Xformable(gripper_prim)
            try:
                reset_stack = bool(xformable.GetResetXformStack())
            except Exception:
                reset_stack = False
            if not reset_stack:
                return

            body0_rel = joint_prim.GetRelationship("physics:body0")
            if not body0_rel:
                return
            body0_targets = body0_rel.GetTargets()
            if not body0_targets:
                return
            body0_prim = get_prim_at_path(str(body0_targets[0]))
            if body0_prim is None or not body0_prim.IsValid():
                return

            cache = UsdGeom.XformCache()
            body0_world = cache.GetLocalToWorldTransform(body0_prim)
            body0_pos = body0_world.ExtractTranslation()
            body0_rot = body0_world.ExtractRotationQuat()
            local_pos0 = _vec3_from_attr(joint_prim, "physics:localPos0", Gf.Vec3d(0.0, 0.0, 0.0))
            local_pos1 = _vec3_from_attr(joint_prim, "physics:localPos1", Gf.Vec3d(0.0, 0.0, 0.0))
            local_rot0 = _quat_from_attr(joint_prim, "physics:localRot0", Gf.Quatd(1.0, 0.0, 0.0, 0.0))
            local_rot1 = _quat_from_attr(joint_prim, "physics:localRot1", Gf.Quatd(1.0, 0.0, 0.0, 0.0))

            target_rot = (body0_rot * local_rot0 * local_rot1.GetInverse()).GetNormalized()
            target_pos = body0_pos + _rotate_vec(body0_rot, local_pos0) - _rotate_vec(target_rot, local_pos1)
            _set_or_add_translate(gripper_xform_path, target_pos)
            _set_or_add_orient(
                gripper_xform_path,
                [
                    target_rot.GetReal(),
                    target_rot.GetImaginary()[0],
                    target_rot.GetImaginary()[1],
                    target_rot.GetImaginary()[2],
                ],
            )

        def _set_vec3_attr(prim, name: str, value):
            if value is None:
                return
            attr = prim.GetAttribute(name)
            if not attr or not attr.IsValid():
                attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float3)
            attr.Set(Gf.Vec3f(*(float(component) for component in value)))

        def _set_quat_attr(prim, name: str, value):
            if value is None:
                return
            quat = [float(component) for component in value]
            attr = prim.GetAttribute(name)
            if not attr or not attr.IsValid():
                attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Quatf)
            attr.Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))

        gripper_xform_path = self._resolve_gripper_xform_path()
        gripper_base_path = self._resolve_gripper_base_path()

        if self._gripper_xform_orient is not None and gripper_xform_path is not None:
            _set_or_add_orient(gripper_xform_path, self._gripper_xform_orient)
        if self._gripper_container_orient is not None and gripper_xform_path is not None:
            _set_or_add_orient(gripper_xform_path, self._gripper_container_orient)

        if not self._configure_gripper_mount_joint:
            return
        joint_prim = get_prim_at_path(f"{self._root_prim_path}/joints/robot_gripper_joint")
        if joint_prim is None or not joint_prim.IsValid():
            return
        if gripper_base_path is not None:
            try:
                joint_prim.CreateRelationship("physics:body0", False).SetTargets(
                    [Sdf.Path(f"{self._root_prim_path}/wrist_3_link")]
                )
                joint_prim.CreateRelationship("physics:body1", False).SetTargets([Sdf.Path(gripper_base_path)])
            except Exception:
                pass
        _set_vec3_attr(joint_prim, "physics:localPos0", self._gripper_mount_local_pos0)
        _set_vec3_attr(joint_prim, "physics:localPos1", self._gripper_mount_local_pos1)
        _set_quat_attr(joint_prim, "physics:localRot0", self._gripper_mount_local_rot0)
        _set_quat_attr(joint_prim, "physics:localRot1", self._gripper_mount_local_rot1)
        if gripper_xform_path is not None:
            _align_reset_gripper_to_mount(gripper_xform_path, joint_prim)

    def _author_gripper_visuals_visible(self) -> None:
        try:
            from isaacsim.core.utils.prims import get_prim_at_path
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
        except Exception:
            try:
                from omni.isaac.core.utils.prims import get_prim_at_path
                from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
            except Exception:
                return

        gripper_roots = [
            f"{self._root_prim_path}/wrist_3_link/Gripper/Robotiq_2F_85",
            f"{self._root_prim_path}/Gripper/Robotiq_2F_85",
            f"{self._root_prim_path}/wrist_3_link/Gripper",
            f"{self._root_prim_path}/Gripper",
        ]

        root_prim = None
        for root_path in gripper_roots:
            try:
                candidate = get_prim_at_path(root_path)
            except Exception:
                candidate = None
            if candidate is not None and candidate.IsValid():
                root_prim = candidate
                break
        if root_prim is None:
            return

        stage = root_prim.GetStage()
        material = self._visible_fingertip_material(stage, Gf=Gf, Sdf=Sdf, UsdGeom=UsdGeom, UsdShade=UsdShade)
        keywords = ("finger", "fingertip", "tip", "pad")
        for prim in Usd.PrimRange(root_prim):
            path_text = str(prim.GetPath()).lower()
            name_text = prim.GetName().lower()
            if not any(keyword in path_text or keyword in name_text for keyword in keywords):
                continue
            try:
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)
            except Exception:
                pass
            try:
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                    imageable.CreatePurposeAttr().Set("default")
            except Exception:
                pass
            try:
                gprim = UsdGeom.Gprim(prim)
                if gprim:
                    gprim.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.72, 0.68)])
                    gprim.CreateDisplayOpacityAttr([1.0])
                    if material is not None:
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                            material,
                            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                        )
            except Exception:
                pass

    def _visible_fingertip_material(self, stage, *, Gf, Sdf, UsdGeom, UsdShade):
        try:
            safe_name = self._root_prim_path.strip("/").replace("/", "_") or "ur5e"
            looks_path = Sdf.Path(f"/World/Looks/{safe_name}_robotiq_visible_fingertips")
            UsdGeom.Scope.Define(stage, looks_path.GetParentPath())
            material = UsdShade.Material.Define(stage, looks_path)
            shader = UsdShade.Shader.Define(stage, looks_path.AppendPath("Shader"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.72, 0.72, 0.68))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return material
        except Exception:
            return None

    def _author_gripper_collision_pads(self) -> None:
        if not self._author_gripper_collision_pads_enabled:
            return
        try:
            from isaacsim.core.utils.prims import get_prim_at_path
            from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
        except Exception:
            try:
                from omni.isaac.core.utils.prims import get_prim_at_path
                from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
            except Exception:
                return

        finger_specs = {
            "inner_pad_collision": {
                "links": ("left_inner_finger", "right_inner_finger"),
                "center": (-0.0535, 0.1288, 0.0),
                "size": (0.024, 0.044, 0.024),
            },
            "inner_finger_collision": {
                "links": ("left_inner_finger", "right_inner_finger"),
                "center": (-0.0590, 0.1035, 0.0),
                "size": (0.034, 0.032, 0.030),
            },
            "outer_finger_collision": {
                "links": ("left_outer_finger", "right_outer_finger"),
                "center": (-0.0625, 0.0737, 0.0),
                "size": (0.034, 0.064, 0.030),
            },
        }
        link_roots = [
            f"{self._root_prim_path}/wrist_3_link/Gripper/Robotiq_2F_85",
            f"{self._root_prim_path}/Gripper/Robotiq_2F_85",
        ]

        stage = None
        material = None
        for root_path in link_roots:
            root_prim = get_prim_at_path(root_path)
            if root_prim is None or not root_prim.IsValid():
                continue
            stage = root_prim.GetStage()
            material = self._visible_fingertip_material(stage, Gf=Gf, Sdf=Sdf, UsdGeom=UsdGeom, UsdShade=UsdShade)
            for spec_name, spec in finger_specs.items():
                for link_name in spec["links"]:
                    link_path = f"{root_path}/{link_name}"
                    link_prim = get_prim_at_path(link_path)
                    if link_prim is None or not link_prim.IsValid():
                        continue
                    self._define_gripper_collision_cube(
                        stage=stage,
                        path=f"{link_path}/{spec_name}",
                        center=spec["center"],
                        size=spec["size"],
                        material=material,
                        Gf=Gf,
                        Sdf=Sdf,
                        UsdGeom=UsdGeom,
                        UsdPhysics=UsdPhysics,
                        UsdShade=UsdShade,
                    )
            break

    @staticmethod
    def _define_gripper_collision_cube(
        *,
        stage,
        path: str,
        center,
        size,
        material,
        Gf,
        Sdf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
    ) -> None:
        cube = UsdGeom.Cube.Define(stage, path)
        prim = cube.GetPrim()
        if prim is None or not prim.IsValid():
            return

        cube.CreateSizeAttr(1.0)
        try:
            imageable = UsdGeom.Imageable(prim)
            imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            imageable.CreatePurposeAttr().Set("default")
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.72, 0.68)])
            cube.CreateDisplayOpacityAttr([1.0])
            if material is not None:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                )
        except Exception:
            pass

        xformable = UsdGeom.Xformable(prim)
        translate = Gf.Vec3d(*(float(component) for component in center))
        scale = Gf.Vec3f(*(float(component) for component in size))
        found_translate = False
        found_scale = False
        for op in xformable.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                op.Set(translate)
                found_translate = True
            elif op.GetOpName() == "xformOp:scale":
                op.Set(scale)
                found_scale = True
        if not found_translate:
            xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(translate)
        if not found_scale:
            xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(scale)

        try:
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            collision_api.CreateCollisionEnabledAttr(True)
        except Exception:
            try:
                prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
            except Exception:
                pass
        try:
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.004)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
        except Exception:
            pass
        del material, UsdShade

    def _resolve_existing_prim_path(self, relative_paths: list[str]) -> Optional[str]:
        try:
            from isaacsim.core.utils.prims import get_prim_at_path
        except Exception:
            try:
                from omni.isaac.core.utils.prims import get_prim_at_path
            except Exception:
                return None

        for relative_path in relative_paths:
            path = f"{self._root_prim_path}/{relative_path}"
            try:
                prim = get_prim_at_path(path)
            except Exception:
                continue
            if prim is not None and prim.IsValid() and prim.IsActive():
                return path
        return None

    def _resolve_gripper_xform_path(self) -> Optional[str]:
        relative_paths = []
        if self._gripper_container_path_override:
            relative_paths.append(str(self._gripper_container_path_override))
        relative_paths.extend(
            [
                "wrist_3_link/Gripper",
                "Gripper",
            ]
        )
        return self._resolve_existing_prim_path(relative_paths)

    def _resolve_gripper_base_path(self) -> Optional[str]:
        relative_paths = []
        if self._gripper_base_link_path_override:
            relative_paths.append(str(self._gripper_base_link_path_override))
        relative_paths.extend(
            [
                "wrist_3_link/Gripper/Robotiq_2F_85/base_link",
                "wrist_3_link/Gripper/base_link",
                "Gripper/Robotiq_2F_85/base_link",
                "Gripper/base_link",
            ]
        )
        return self._resolve_existing_prim_path(relative_paths)

    def _make_gripper(self, gripper_cls=None):
        if gripper_cls is None:
            from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper as gripper_cls

        gripper = gripper_cls(
            end_effector_prim_path=self._end_effector_prim_path,
            joint_prim_names=[self._gripper_dof_name],
            joint_opened_positions=np.array([self._gripper_open_position], dtype=float),
            joint_closed_positions=np.array([self._gripper_closed_position], dtype=float),
            action_deltas=self._gripper_deltas,
            use_mimic_joints=True,
        )
        try:
            gripper.set_default_state(np.array([self._gripper_open_position], dtype=float))
        except Exception:
            pass
        return gripper

    @staticmethod
    def _normalized_dof_name(dof_name: str) -> str:
        return str(dof_name).replace("\\", "/").strip("/")

    @classmethod
    def _dof_name_matches(cls, candidate: str, requested: str) -> bool:
        candidate_name = cls._normalized_dof_name(candidate)
        requested_name = cls._normalized_dof_name(requested)
        if candidate_name == requested_name:
            return True
        if not candidate_name or not requested_name:
            return False
        return candidate_name.split("/")[-1] == requested_name.split("/")[-1]

    @classmethod
    def _resolve_dof_name_from_dofs(cls, requested: str, dof_names: List[str]) -> Optional[str]:
        requested_name = str(requested or "finger_joint")
        available_names = [str(name) for name in (dof_names or [])]
        if requested_name in available_names:
            return requested_name

        suffix_matches = [name for name in available_names if cls._dof_name_matches(name, requested_name)]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            requested_norm = cls._normalized_dof_name(requested_name)
            for name in suffix_matches:
                if cls._normalized_dof_name(name).endswith("/" + requested_norm):
                    return name
            return suffix_matches[0]
        return None

    def resolve_dof_name(self, requested: str) -> Optional[str]:
        try:
            dof_names = self.dof_names
        except Exception:
            return None
        return self._resolve_dof_name_from_dofs(requested, dof_names)

    @property
    def gripper_dof_name(self) -> str:
        return self._gripper_dof_name

    def _set_gripper_dof_name(self, dof_name: str) -> None:
        self._gripper_dof_name = str(dof_name)
        if self._gripper is None:
            return
        if hasattr(self._gripper, "_joint_prim_names"):
            self._gripper._joint_prim_names = [self._gripper_dof_name]
        if hasattr(self._gripper, "_joint_dof_indicies"):
            self._gripper._joint_dof_indicies = np.array([None, None])

    def _resolve_gripper_dof_name_for_initialized_articulation(self) -> str:
        requested_name = self._gripper_dof_name or "finger_joint"
        try:
            dof_names = list(self.dof_names or [])
        except Exception:
            return requested_name
        resolved_name = self._resolve_dof_name_from_dofs(requested_name, dof_names)
        if resolved_name is None:
            log.warn(
                f"ur5e {self.name}: failed to resolve gripper DOF {requested_name!r}; "
                f"available DOFs: {dof_names!r}"
            )
            return requested_name
        if resolved_name != requested_name:
            log.info(f"ur5e {self.name}: resolved gripper DOF {requested_name!r} -> {resolved_name!r}.")
        self._set_gripper_dof_name(resolved_name)
        return resolved_name

    def _resolve_end_effector_prim_path(self) -> str:
        try:
            from isaacsim.core.utils.prims import is_prim_path_valid
        except Exception:
            try:
                from omni.isaac.core.utils.prims import is_prim_path_valid
            except Exception:
                return self._end_effector_prim_path

        candidates = [
            self._end_effector_prim_name,
            "tool0",
            "flange",
            "wrist_3_link",
            "wrist_3_link/ft_frame",
            "wrist_3_link/Gripper/Robotiq_2F_85/base_link",
            "wrist_3_link/Gripper/base_link",
            "Gripper/Robotiq_2F_85/base_link",
            "Gripper/base_link",
        ]
        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            candidate_path = f"{self._root_prim_path}/{candidate}"
            try:
                if is_prim_path_valid(candidate_path):
                    self._end_effector_prim_name = candidate
                    self._end_effector_prim_path = candidate_path
                    return candidate_path
            except Exception:
                continue
        return self._end_effector_prim_path

    @property
    def end_effector(self) -> IRigidBody:
        return self._end_effector

    @property
    def gripper(self):
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        self.unwrap().initialize(physics_sim_view)
        if self._start_position is not None or self._start_orientation is not None:
            try:
                self.set_world_pose(position=self._start_position, orientation=self._start_orientation)
            except Exception:
                pass
            self._author_root_xform_pose()
        self._author_gripper_mount_pose()
        self._author_gripper_visuals_visible()
        self._author_gripper_collision_pads()
        previous_end_effector_path = self._end_effector_prim_path
        self._resolve_end_effector_prim_path()
        if self._end_effector_prim_path != previous_end_effector_path:
            self._gripper = self._make_gripper()
        self._resolve_gripper_dof_name_for_initialized_articulation()
        self._end_effector = IRigidBody.create(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.unwrap().initialize(physics_sim_view)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )

    def post_reset(self) -> None:
        self.unwrap().post_reset()
        if self._start_position is not None or self._start_orientation is not None:
            try:
                self.set_world_pose(position=self._start_position, orientation=self._start_orientation)
            except Exception:
                pass
            self._author_root_xform_pose()
        self._author_gripper_mount_pose()
        self._author_gripper_visuals_visible()
        self._author_gripper_collision_pads()
        self._gripper.post_reset()
        for dof_index in self.gripper.active_joint_indices:
            self._articulation_controller.switch_dof_control_mode(dof_index=dof_index, mode="position")


class _ArticulationPoseProxy:
    def __init__(self, articulation: UR5e):
        self._articulation = articulation

    def get_pose(self):
        return self._articulation.get_pose()

    def get_local_pose(self):
        return self._articulation.get_local_pose()


class _UsdPrimPoseProxy:
    def __init__(self, prim_path: str):
        self._prim_path = prim_path

    def _pose(self):
        try:
            from isaacsim.core.utils.prims import get_prim_at_path
            from pxr import Usd, UsdGeom
        except Exception:
            try:
                from omni.isaac.core.utils.prims import get_prim_at_path
                from pxr import Usd, UsdGeom
            except Exception:
                return np.array([0.0, 0.0, 0.0], dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        try:
            prim = get_prim_at_path(self._prim_path)
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = transform.ExtractTranslation()
            rotation = transform.ExtractRotationQuat()
        except Exception:
            return np.array([0.0, 0.0, 0.0], dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        imaginary = rotation.GetImaginary()
        position = np.array([translation[0], translation[1], translation[2]], dtype=float)
        orientation = np.array([rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]], dtype=float)
        return position, orientation

    def get_pose(self):
        return self._pose()

    def get_local_pose(self):
        return self._pose()


@BaseRobot.register("UR5eRobot")
class UR5eRobot(BaseRobot):
    def __init__(self, config: UR5eRobotCfg, scene: IScene):
        super().__init__(config, scene)
        self._robot_ik_base = None
        self._start_position = np.array(config.position) if config.position is not None else None
        self._start_orientation = np.array(config.orientation) if config.orientation is not None else None
        self._robot_scale = np.array([1.0, 1.0, 1.0])
        if config.scale is not None:
            self._robot_scale = np.array(config.scale)

        if config.usd_path is None:
            raise ValueError("UR5eRobotCfg.usd_path must be set to a UR5e USD asset.")

        log.debug(f"ur5e {config.name}: position    : {self._start_position}")
        log.debug(f"ur5e {config.name}: orientation : {self._start_orientation}")
        log.debug(f"ur5e {config.name}: usd_path    : {config.usd_path}")

        self.articulation = UR5e(
            prim_path=config.prim_path,
            name=config.name,
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=os.path.abspath(config.usd_path),
            end_effector_prim_name=config.end_effector_prim_name,
            gripper_dof_name=config.gripper_dof_name,
            gripper_open_position=config.gripper_open_position,
            gripper_closed_position=config.gripper_closed_position,
            gripper_xform_orient=config.gripper_xform_orient,
            gripper_mount_local_pos0=config.gripper_mount_local_pos0,
            gripper_mount_local_pos1=config.gripper_mount_local_pos1,
            gripper_mount_local_rot0=config.gripper_mount_local_rot0,
            gripper_mount_local_rot1=config.gripper_mount_local_rot1,
            configure_gripper_mount_joint=config.configure_gripper_mount_joint,
            gripper_base_link_path=config.gripper_base_link_path,
            gripper_container_path=config.gripper_container_path,
            gripper_container_orient=config.gripper_container_orient,
            author_gripper_collision_pads=config.author_gripper_collision_pads,
            scale=self._robot_scale,
        )
        self.last_action = []

    def get_robot_scale(self):
        return self._robot_scale

    def get_robot_ik_base(self):
        return self._robot_ik_base

    def post_reset(self):
        super().post_reset()
        self._robot_ik_base = self._resolve_ik_base_rigid_body()
        self._sync_resolved_gripper_dof_name_to_config()
        self._apply_initial_joint_positions()
        self._configure_drive_gains()
        self._apply_gripper_contact_material()

    def _robot_rigid_body_by_suffix(self, suffix: str):
        suffix = f"/{suffix}"
        for prim_path, rigid_body in self._rigid_body_map.items():
            if prim_path.endswith(suffix):
                return rigid_body
        return None

    def _resolve_ik_base_rigid_body(self):
        candidate_names = [
            self.config.ik_base_prim_name,
            "base_link",
            "base",
            "base_link_inertia",
            "shoulder_link",
        ]
        seen = set()
        for candidate_name in candidate_names:
            if not candidate_name or candidate_name in seen:
                continue
            seen.add(candidate_name)
            rigid_body = self._robot_rigid_body_by_suffix(str(candidate_name))
            if rigid_body is not None:
                return rigid_body

        if self._rigid_body_map:
            first_path, first_body = next(iter(self._rigid_body_map.items()))
            log.warn(
                f"ur5e {self.config.name}: failed to resolve IK base {self.config.ik_base_prim_name!r}; "
                f"falling back to first rigid body {first_path!r}."
            )
            return first_body

        usd_pose_proxy = self._resolve_ik_base_usd_prim(candidate_names)
        if usd_pose_proxy is not None:
            return usd_pose_proxy

        log.warn(
            f"ur5e {self.config.name}: no rigid bodies were found for IK base resolution; "
            "using the articulation root pose as IK base."
        )
        return _ArticulationPoseProxy(self.articulation)

    def _resolve_ik_base_usd_prim(self, candidate_names):
        try:
            from isaacsim.core.utils.prims import get_prim_at_path
        except Exception:
            try:
                from omni.isaac.core.utils.prims import get_prim_at_path
            except Exception:
                return None

        seen = set()
        for candidate_name in candidate_names:
            if not candidate_name or candidate_name in seen:
                continue
            seen.add(candidate_name)
            prim_path = f"{self.config.prim_path}/{candidate_name}"
            try:
                prim = get_prim_at_path(prim_path)
            except Exception:
                prim = None
            if prim is not None and prim.IsValid():
                log.info(
                    f"ur5e {self.config.name}: using USD prim {prim_path!r} as IK base pose source."
                )
                return _UsdPrimPoseProxy(prim_path)
        return None

    def _resolved_articulation_dof_name(self, dof_name: str) -> Optional[str]:
        resolver = getattr(self.articulation, "resolve_dof_name", None)
        if callable(resolver):
            resolved_name = resolver(dof_name)
            if resolved_name is not None:
                return resolved_name
        try:
            self.articulation.get_dof_index(dof_name)
            return dof_name
        except Exception:
            return None

    def _sync_resolved_gripper_dof_name_to_config(self) -> None:
        resolved_name = getattr(self.articulation, "gripper_dof_name", None)
        if not resolved_name:
            resolved_name = self._resolved_articulation_dof_name(self.config.gripper_dof_name or "finger_joint")
        if resolved_name and resolved_name != self.config.gripper_dof_name:
            self.config.gripper_dof_name = str(resolved_name)

    def _apply_initial_joint_positions(self) -> None:
        joint_targets = dict(DEFAULT_UR5E_READY_JOINTS)
        if self.config.initial_joint_positions:
            joint_targets.update({str(k): float(v) for k, v in self.config.initial_joint_positions.items()})
        for joint_name, joint_pos in joint_targets.items():
            resolved_joint_name = self._resolved_articulation_dof_name(joint_name)
            if resolved_joint_name is None:
                continue
            try:
                joint_index = self.articulation.get_dof_index(resolved_joint_name)
                self.articulation.set_joint_positions(
                    np.array([float(joint_pos)], dtype=float),
                    joint_indices=np.array([joint_index], dtype=np.int64),
                )
            except Exception:
                continue

    def _configure_drive_gains(self) -> None:
        try:
            arm_indices = np.asarray([self.articulation.get_dof_index(name) for name in _UR5E_ARM_JOINT_NAMES], dtype=np.int64)
            self.articulation.set_gains(
                kps=np.full(arm_indices.shape, 8.0e4, dtype=float),
                kds=np.full(arm_indices.shape, 4.0e3, dtype=float),
                joint_indices=arm_indices,
            )
        except Exception:
            pass
        try:
            gripper_dof_name = self._resolved_articulation_dof_name(self.config.gripper_dof_name or "finger_joint")
            if gripper_dof_name is None:
                return
            gripper_index = np.asarray([self.articulation.get_dof_index(gripper_dof_name)], dtype=np.int64)
            self.articulation.set_gains(
                kps=np.asarray([7.5e3], dtype=float),
                kds=np.asarray([1.73e2], dtype=float),
                joint_indices=gripper_index,
            )
        except Exception:
            pass

    def _apply_gripper_contact_material(self):
        try:
            from isaacsim.core.api.materials import PhysicsMaterial
        except Exception:
            try:
                from omni.isaac.core.materials import PhysicsMaterial
            except Exception:
                return

        try:
            material_name = f"{self.config.name}_robotiq_high_friction"
            physics_material = PhysicsMaterial(
                prim_path=f"/World/Physics_Materials/{material_name}",
                name=material_name,
                static_friction=3.0,
                dynamic_friction=2.5,
                restitution=0.0,
            )
        except Exception:
            return

        for link_name in (self.config.left_finger_link_name, self.config.right_finger_link_name):
            if not link_name:
                continue
            rigid_body = self._robot_rigid_body_by_suffix(str(link_name))
            if rigid_body is None:
                continue
            try:
                rigid_body.unwrap().apply_physics_material(physics_material)
            except Exception:
                pass

    @staticmethod
    def action_to_dict(action):
        def numpy_to_list(array):
            return array.tolist() if isinstance(array, np.ndarray) else array

        return {
            "joint_efforts": numpy_to_list(action.joint_efforts),
            "joint_indices": numpy_to_list(action.joint_indices),
            "joint_positions": numpy_to_list(action.joint_positions),
            "joint_velocities": numpy_to_list(action.joint_velocities),
        }

    @staticmethod
    def _bounded_revolute_joint_values(values):
        values = np.asarray(values, dtype=float).copy()
        wrapped = (values + np.pi) % (2.0 * np.pi) - np.pi
        return np.where(np.abs(values) > np.pi + 0.25, wrapped, values)

    def _current_arm_joint_positions_for_normalization(self, expected_size: int):
        try:
            indices = np.asarray(
                [self.articulation.get_dof_index(name) for name in _UR5E_ARM_JOINT_NAMES],
                dtype=np.int64,
            )
            current = np.asarray(self.articulation.get_joint_positions(joint_indices=indices), dtype=float).reshape(-1)
        except Exception:
            return None
        if current.shape[0] != expected_size or not np.all(np.isfinite(current)):
            return None
        return current

    def _renormalize_arm_joint_state_if_needed(self) -> None:
        try:
            indices = np.asarray(
                [self.articulation.get_dof_index(name) for name in _UR5E_ARM_JOINT_NAMES],
                dtype=np.int64,
            )
            current = np.asarray(self.articulation.get_joint_positions(joint_indices=indices), dtype=float).reshape(-1)
        except Exception:
            return
        if current.shape[0] != len(_UR5E_ARM_JOINT_NAMES) or not np.all(np.isfinite(current)):
            return
        bounded = self._bounded_revolute_joint_values(current)
        if float(np.max(np.abs(current - bounded))) <= 1e-6:
            return
        try:
            self.articulation.set_joint_positions(bounded, joint_indices=indices)
        except Exception:
            return
        try:
            self.articulation.set_joint_velocities(np.zeros_like(bounded), joint_indices=indices)
        except Exception:
            pass

    def _normalize_arm_joint_controller_action(self, controller_name: str, controller_action):
        if controller_name != "arm_joint_controller" or not isinstance(controller_action, (list, tuple)):
            return controller_action
        if not controller_action:
            return controller_action
        joint_positions = controller_action[0]
        if joint_positions is None:
            return controller_action
        try:
            values = np.asarray(joint_positions, dtype=float).copy()
        except Exception:
            return controller_action
        if values.size == 0 or not np.all(np.isfinite(values)):
            return controller_action
        values = self._bounded_revolute_joint_values(values)
        current = self._current_arm_joint_positions_for_normalization(values.shape[0])
        if current is not None:
            current = self._bounded_revolute_joint_values(current)
            period = 2.0 * np.pi
            for index, value in enumerate(values):
                center = int(round((float(current[index]) - float(value)) / period))
                candidates = value + (center + np.arange(-2, 3, dtype=float)) * period
                bounded_candidates = candidates[np.abs(candidates) <= 2.0 * np.pi]
                if bounded_candidates.size:
                    candidates = bounded_candidates
                costs = np.abs(candidates - current[index])
                values[index] = candidates[int(np.argmin(costs))]
        normalized_action = list(controller_action)
        normalized_action[0] = values.tolist()
        return normalized_action

    def _normalize_arm_control(self, controller_name: str, control):
        if controller_name not in {"arm_joint_controller", "arm_ik_controller"}:
            return control
        joint_positions = getattr(control, "joint_positions", None)
        if joint_positions is None:
            return control
        try:
            values = np.asarray(joint_positions, dtype=float).copy()
        except Exception:
            return control
        if values.size == 0 or not np.all(np.isfinite(values)):
            return control
        values = self._bounded_revolute_joint_values(values)
        current = self._current_arm_joint_positions_for_normalization(values.shape[0])
        if current is not None:
            current = self._bounded_revolute_joint_values(current)
            period = 2.0 * np.pi
            for index, value in enumerate(values):
                candidates = value + np.arange(-1, 2, dtype=float) * period
                bounded_candidates = candidates[np.abs(candidates) <= 2.0 * np.pi]
                if bounded_candidates.size:
                    candidates = bounded_candidates
                costs = np.abs(candidates - current[index])
                values[index] = candidates[int(np.argmin(costs))]
        control.joint_positions = values
        return control

    def apply_action(self, action: dict):
        self.last_action = []
        deferred_controls = []
        has_joint_override = "arm_joint_controller" in action and "arm_ik_controller" in action
        self._renormalize_arm_joint_state_if_needed()
        for controller_name, controller_action in action.items():
            if controller_name not in self.controllers:
                log.warn(f"unknown controller {controller_name} in action")
                continue
            controller = self.controllers[controller_name]
            controller_action = self._normalize_arm_joint_controller_action(controller_name, controller_action)
            control = controller.action_to_control(controller_action)
            if control is None:
                if os.environ.get("UR5E_DEBUG_GRASP", "0").lower() in {"1", "true", "yes"}:
                    log.warn(
                        f"ur5e {self.config.name}: controller {controller_name} returned None "
                        f"for action {controller_action!r}; skipping this control."
                    )
                continue
            control = self._normalize_arm_control(controller_name, control)
            if has_joint_override and controller_name == "arm_ik_controller":
                self.last_action.append(self.action_to_dict(control))
                continue
            deferred_controls.append(control)
            self.last_action.append(self.action_to_dict(control))
        for control in deferred_controls:
            self.articulation.apply_action(control)

    def get_last_action(self):
        return self.last_action

    def get_obs(self) -> OrderedDict[str, Any]:
        position, orientation = self.articulation.get_pose()
        obs = {
            "position": position,
            "orientation": orientation,
            "joint_action": self.get_last_action(),
            "controllers": {},
            "sensors": {},
        }
        eef_pose = self.articulation.end_effector.get_pose()
        obs["eef_body_position"] = eef_pose[0]
        obs["eef_body_orientation"] = eef_pose[1]
        if "arm_ik_controller" in self.controllers:
            ik_obs = self.controllers["arm_ik_controller"].get_obs()
            obs["eef_position"] = ik_obs.get("eef_position", eef_pose[0])
            obs["eef_orientation"] = ik_obs.get("eef_orientation", eef_pose[1])
        else:
            obs["eef_position"] = eef_pose[0]
            obs["eef_orientation"] = eef_pose[1]

        for c_obs_name, controller_obs in self.controllers.items():
            obs["controllers"][c_obs_name] = controller_obs.get_obs()
        for sensor_name, sensor_obs in self.sensors.items():
            obs["sensors"][sensor_name] = sensor_obs.get_data()
        return self._make_ordered(obs)
