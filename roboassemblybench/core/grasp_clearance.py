"""Read-only native USD gripper closure checks against real fixture meshes.

USD/PhysX state is never edited. Joint frames and mimic ratios define FK;
python-fcl checks the authored collider geometry at bounded angular samples.
This is a geometric preflight, not proof of a force-qualified physical grasp.
"""
from __future__ import annotations

import math
import numpy as np


def pose_matrix(position, orientation):
    from scipy.spatial.transform import Rotation
    q = np.asarray(orientation, dtype=float)
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = position
    return result


def _joint_frame(joint, side):
    quat = getattr(joint, f'GetLocalRot{side}Attr')().Get()
    return pose_matrix(getattr(joint, f'GetLocalPos{side}Attr')().Get(),
                       [quat.GetReal(), *quat.GetImaginary()])


def collider_meshes(root):
    """Return native collider meshes in their containing rigid body's frame."""
    from pxr import Usd, UsdGeom, UsdPhysics
    import trimesh

    cache = UsdGeom.XformCache()
    meshes = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if (not prim.HasAPI(UsdPhysics.CollisionAPI)
                or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            if prim.IsA(UsdGeom.Gprim):
                raise ValueError(f'Unsupported non-mesh collider in native closure preflight: {prim.GetPath()}')
            continue
        owner = prim
        while owner != root and not (
            owner.HasAPI(UsdPhysics.RigidBodyAPI)
            and UsdPhysics.RigidBodyAPI(owner).GetRigidBodyEnabledAttr().Get() is not False
        ):
            owner = owner.GetParent()
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        faces, cursor = [], 0
        for count in counts:
            polygon = indices[cursor:cursor + count]
            faces.extend([polygon[0], polygon[i], polygon[i + 1]] for i in range(1, count - 1))
            cursor += count
        local = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(owner)).T) @ np.asarray(
            cache.GetLocalToWorldTransform(prim)).T
        vertices = (local @ np.column_stack([points, np.ones(len(points))]).T).T[:, :3]
        shape = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        # Native Robotiq colliders use convex hulls. A conservative convex hull
        # also rejects unsafe candidates for other convex approximations.
        if approximation in ('convexHull', 'convexDecomposition'):
            shape = shape.convex_hull
        meshes.append(dict(path=str(prim.GetPath()), body=str(owner.GetPath()), mesh=shape,
                           approximation=approximation))
    if not meshes:
        raise ValueError(f'No enabled mesh colliders beneath {root.GetPath()}')
    return meshes


class NativeGripper:
    def __init__(self, robot_prim, base_path, master_joint):
        from pxr import Usd, UsdPhysics

        self.base_path = base_path
        gripper_root = robot_prim.GetStage().GetPrimAtPath(base_path).GetParent()
        self.meshes = collider_meshes(gripper_root)
        self.joints = []
        prefix = str(gripper_root.GetPath()) + '/'
        for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            b0, b1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not b0 or not b1 or not all(str(b).startswith(prefix) for b in (b0[0], b1[0])):
                continue
            if joint.GetExcludeFromArticulationAttr().Get():
                continue
            fixed = prim.IsA(UsdPhysics.FixedJoint)
            if not fixed and not prim.IsA(UsdPhysics.RevoluteJoint):
                raise ValueError(f'Unsupported gripper joint: {prim.GetPath()}')
            reference, multiplier = None, 0.
            if not fixed:
                if prim.GetName() == master_joint:
                    multiplier = 1.
                else:
                    attrs = [a for a in prim.GetAttributes()
                             if a.GetName().startswith('physxMimicJoint:') and a.GetName().endswith(':gearing')]
                    if len(attrs) != 1:
                        raise ValueError(f'Missing unique mimic relation on {prim.GetPath()}')
                    name = attrs[0].GetName().rsplit(':', 1)[0]
                    reference = str(prim.GetRelationship(name + ':referenceJoint').GetTargets()[0]).split('/')[-1]
                    multiplier = -float(attrs[0].Get())
                    offset = prim.GetAttribute(name + ':offset')
                    if offset and abs(float(offset.Get() or 0.)) > 1e-8:
                        raise ValueError('Nonzero mimic offset is not supported by this closure preflight')
            self.joints.append(dict(
                name=prim.GetName(), parent=str(b0[0]), child=str(b1[0]),
                frame0=_joint_frame(joint, 0), inverse_frame1=np.linalg.inv(_joint_frame(joint, 1)),
                axis=None if fixed else str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()),
                reference=reference, multiplier=multiplier))
        self.master_joint = master_joint
        coefficients = {master_joint: 1.}
        spans = {base_path: (0., 0.)}
        pending = list(self.joints)
        while pending:
            progressed = False
            for joint in pending[:]:
                if joint['parent'] not in spans or (joint['reference'] and joint['reference'] not in coefficients):
                    continue
                coefficient = joint['multiplier'] * coefficients[joint['reference']] if joint['reference'] else joint['multiplier']
                span, angular_sum = spans[joint['parent']]
                span += np.linalg.norm(joint['frame0'][:3, 3]) + np.linalg.norm(joint['inverse_frame1'][:3, 3])
                spans.setdefault(joint['child'], (span, angular_sum + abs(coefficient)))
                coefficients[joint['name']] = coefficient
                pending.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError('Native gripper joint graph is disconnected or cyclic')
        # Triangle inequality bounds each vertex's distance to every ancestor
        # joint for the whole interval, including rotations between samples.
        self.motion_radius = max(
            (float(np.linalg.norm(entry['mesh'].vertices, axis=1).max()) + spans[entry['body']][0])
            * spans[entry['body']][1] for entry in self.meshes)

    def motion_bound(self, joint_travel):
        return self.motion_radius * abs(joint_travel)

    def link_transforms(self, q):
        angles = {self.master_joint: float(q)}
        poses = {self.base_path: np.eye(4)}
        pending = list(self.joints)
        while pending:
            progressed = False
            for joint in pending[:]:
                if joint['parent'] not in poses or (joint['reference'] and joint['reference'] not in angles):
                    continue
                angle = (joint['multiplier'] * angles[joint['reference']] if joint['reference'] else
                         joint['multiplier'] * q)
                rotation = np.eye(4)
                if joint['axis']:
                    from scipy.spatial.transform import Rotation
                    axis = np.eye(3)['XYZ'.index(joint['axis'])]
                    rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
                value = poses[joint['parent']] @ joint['frame0'] @ rotation @ joint['inverse_frame1']
                if joint['child'] not in poses:
                    poses[joint['child']] = value
                angles[joint['name']] = angle
                pending.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError('Native gripper joint graph is disconnected or cyclic')
        missing = {entry['body'] for entry in self.meshes} - poses.keys()
        if missing:
            raise ValueError(f'Native gripper collider bodies missing from FK: {sorted(missing)}')
        return poses


def check_closure(model, fixture, target, *, tcp_pose, fixture_pose, target_pose,
                  start_q, end_q, finger_names, margin=.0015, joint_step=.005, preload=.01):
    """Check every gripper mesh until bilateral geometric capture + preload.

    The sampling uncertainty (maximum vertex displacement between adjacent
    samples) is added to the clearance margin. This conservatively covers the
    short swept intervals as well as the sampled poses.
    """
    import trimesh

    if not (0 < joint_step <= .01 and margin > 0 and preload >= 0):
        raise ValueError('Invalid native closure clearance settings')
    gripper = trimesh.collision.CollisionManager()
    obstacles = trimesh.collision.CollisionManager()
    moving = trimesh.collision.CollisionManager()
    tcp = pose_matrix(**tcp_pose)
    for entry in fixture:
        obstacles.add_object(entry['path'], entry['mesh'], transform=pose_matrix(**fixture_pose))
    for entry in target:
        moving.add_object(entry['path'], entry['mesh'], transform=pose_matrix(**target_pose))
    previous_vertices, previous_q = None, None
    capture_q = None
    minimum = math.inf
    samples = np.linspace(start_q, end_q, max(2, math.ceil(abs(end_q - start_q) / joint_step) + 1))
    for sample_index, q in enumerate(samples):
        links = model.link_transforms(q)
        world_vertices = []
        for entry in model.meshes:
            transform = tcp @ links[entry['body']]
            if sample_index == 0:
                gripper.add_object(entry['path'], entry['mesh'], transform=transform)
            else:
                gripper.set_transform(entry['path'], transform)
            v = entry['mesh'].vertices
            world_vertices.append((transform[:3, :3] @ v.T).T + transform[:3, 3])
        vertices = np.concatenate(world_vertices)
        displacement = (0. if previous_vertices is None else
                        float(np.linalg.norm(vertices - previous_vertices, axis=1).max()))
        if previous_q is not None and hasattr(model, 'motion_bound'):
            displacement = max(displacement, model.motion_bound(float(q - previous_q)))
        previous_q = float(q)
        previous_vertices = vertices
        clearance = float(gripper.min_distance_other(obstacles))
        minimum = min(minimum, clearance)
        if clearance <= margin + displacement:
            _, pairs = gripper.in_collision_other(obstacles, return_names=True)
            return dict(passed=False, reason='fixture_blocks_finger_sweep', joint_q=float(q),
                        minimum_clearance_m=minimum, required_margin_m=margin,
                        sampling_motion_bound_m=displacement, collision_pairs=sorted(pairs),
                        samples_checked=sample_index + 1)
        _, pairs = gripper.in_collision_other(moving, return_names=True)
        contact_paths = {path for path, _ in pairs}
        contact_bodies = {entry['body'].rsplit('/', 1)[-1] for entry in model.meshes
                          if entry['path'] in contact_paths}
        sides = [name in contact_bodies for name in finger_names]
        if not all(sides):
            capture_q = None
        if all(sides) and capture_q is None:
            capture_q = float(q)
        if capture_q is not None and abs(q - capture_q) + 1e-12 >= preload:
            return dict(passed=True, reason='fixture_clear_through_bilateral_capture',
                        minimum_clearance_m=minimum, capture_joint_q=capture_q,
                        final_joint_q=float(q), samples_checked=sample_index + 1,
                        required_margin_m=margin)
    return dict(passed=False, reason='no_bilateral_capture_in_sweep', minimum_clearance_m=minimum,
                capture_joint_q=capture_q, samples_checked=len(samples))


def _native_closure_report(task, robot_name, spec, tcp_pose):
    """Load/cache read-only native geometry, then check the current scene poses."""
    from isaacsim.core.utils.prims import get_prim_at_path

    settings = spec['fixture_closure_clearance']
    robot = task.robots[robot_name]
    cache = getattr(task, '_native_closure_geometry', None)
    if cache is None:
        cache = task._native_closure_geometry = {}
    key = (robot_name, settings['fixture'], spec['object'])
    if key not in cache:
        root = get_prim_at_path(robot.config.prim_path)
        base = str(root.GetPath()) + '/' + robot.config.end_effector_prim_name
        model = NativeGripper(root, base, robot.config.gripper_dof_name)
        cache[key] = (model, collider_meshes(task._object_prims[settings['fixture']]),
                      collider_meshes(task._object_prims[spec['object']]))
    fixture_pose = task._resolve_object(settings['fixture']).get_pose()
    object_pose = task._resolve_object(spec['object']).get_pose()
    # Skill targets use Lula's control frame, while USD joint frames are
    # relative to the physical gripper body. This asset mounts those frames
    # with a fixed pi rotation. Measure the rigid frame transform explicitly.
    from scipy.spatial.transform import Rotation
    control = task._get_robot_kinematics_pose(robot_name)
    physical = task._get_robot_eef_pose(robot_name)
    if control is None:
        raise ValueError('No control/physical frame transform for native closure preflight')
    relative = np.linalg.inv(pose_matrix(*control)) @ pose_matrix(*physical)
    native = pose_matrix(**tcp_pose) @ relative
    quat = Rotation.from_matrix(native[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return check_closure(
        *cache[key], tcp_pose=dict(position=native[:3, 3], orientation=quat),
        fixture_pose=dict(position=fixture_pose[0], orientation=fixture_pose[1]),
        target_pose=dict(position=object_pose[0], orientation=object_pose[1]),
        start_q=float(robot.config.gripper_open_position), end_q=float(robot.config.gripper_closed_position),
        finger_names=(robot.config.left_finger_link_name, robot.config.right_finger_link_name),
        margin=float(settings.get('margin_m', .0015)) + float(spec.get('close_gate_recenter_max_offset', 0.)),
        joint_step=float(settings.get('joint_step', .005)),
        preload=float(settings.get('preload_joint_travel', .01)))


def native_closure_report(task, robot_name, spec, tcp_pose):
    """Fail closed with a reviewable report if native geometry is unsupported."""
    try:
        report = _native_closure_report(task, robot_name, spec, tcp_pose)
    except (ImportError, ValueError, RuntimeError) as exc:
        report = dict(passed=False, reason='native_closure_geometry_unavailable',
                      error=f'{type(exc).__name__}: {exc}')
    reports = getattr(task, '_native_closure_clearance_reports', None)
    if reports is None:
        reports = task._native_closure_clearance_reports = {}
    reports[f'{task.phase}:{robot_name}'] = report
    return report
