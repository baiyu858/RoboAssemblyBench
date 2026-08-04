"""Collision detection — agent-agent and agent-environment.

Extends the inter-arm link distance check (from demos/two_arm_collide.py)
with environment obstacle collision detection, supporting box-shaped
obstacles (table, blocks) and ground planes.

Usage:
    detector = CollisionDetector(threshold=0.12)

    # register agents
    detector.add_agent("franka_a", link_xf_a)
    detector.add_agent("franka_b", link_xf_b)

    # register environment
    detector.add_box("table", center=(0.3, 0.0, 0.4),
                     half_extents=(1.0, 0.7, 0.2))
    detector.add_ground(z=0.0)

    # per-frame check
    events = detector.check_all()
    for e in events:
        print(e)

Does NOT modify existing files — works alongside two_arm_collide.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CollisionEvent:
    """A single collision or proximity violation."""

    step: int
    kind: str  # "inter_agent" | "agent_env"
    entity_a: str  # agent name or link name
    entity_b: str  # agent/link name or obstacle name
    distance: float  # meters
    threshold: float  # meters
    pos_a: Optional[np.ndarray] = None  # world position of entity_a
    pos_b: Optional[np.ndarray] = None  # world position of entity_b

    def __str__(self):
        return (
            f'[step {self.step:04d}] {self.kind}: '
            f'{self.entity_a} <-> {self.entity_b}  '
            f'({self.distance * 100:.1f}cm, threshold={self.threshold * 100:.1f}cm)'
        )


# ═══════════════════════════════════════════════════════════════════════════
# geometric helpers
# ═══════════════════════════════════════════════════════════════════════════


def _orientation_matrix(orient: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion or pass through a cached 3x3 matrix."""

    value = np.asarray(orient, dtype=np.float64)
    if value.shape == (3, 3):
        return value
    if value.shape != (4,):
        raise ValueError(f'Expected wxyz quaternion or 3x3 matrix, got shape {value.shape}.')
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = value / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _closest_point_box(
    point: np.ndarray, center: np.ndarray, half_extents: np.ndarray, orient: Optional[np.ndarray] = None
) -> np.ndarray:
    """Closest point on (or inside) a box to a given world-space *point*.

    Args:
        point:       3D world position of the query point.
        center:      3D world position of the box center.
        half_extents: (hx, hy, hz) of the box.
        orient:      Orientation quaternion [w, x, y, z] or cached 3x3 matrix.
                     If None, the box is axis-aligned.

    Returns:
        closest: The world-space point on the box surface closest to *point*.
    """
    point = np.asarray(point, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    he = np.asarray(half_extents, dtype=np.float64)

    # Transform point into box local frame
    if orient is not None:
        rotation = _orientation_matrix(orient)
        local = rotation.T @ (point - center)
    else:
        local = point - center

    # Clamp to box surface in local frame
    closest_local = np.clip(local, -he, he)

    # Transform back to world
    if orient is not None:
        closest_world = center + rotation @ closest_local
    else:
        closest_world = center + closest_local

    return closest_world


def _point_box_distance(
    point: np.ndarray, center: np.ndarray, half_extents: np.ndarray, orient: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Minimum distance from a point to a box surface, and the closest point."""
    cp = _closest_point_box(point, center, half_extents, orient)
    d = np.linalg.norm(point - cp)
    return float(d), cp


def _point_ground_distance(point: np.ndarray, ground_z: float) -> float:
    """Signed distance from a point to a ground plane at *ground_z*.
    Positive = above ground, negative = penetration.
    """
    return float(point[2] - ground_z)


def _closest_segment_segment(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Closest points between two line segments [p1,q1] and [p2,q2].

    Exact clamped solution (Ericson, Real-Time Collision Detection).
    Returns (c1, c2): the closest point on each segment.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    d1 = q1 - p1  # direction of segment 1
    d2 = q2 - p2  # direction of segment 2
    r = p1 - p2
    a = float(d1 @ d1)  # squared length of segment 1
    e = float(d2 @ d2)  # squared length of segment 2
    f = float(d2 @ r)
    eps = 1e-12

    if a <= eps and e <= eps:  # both segments are points
        return p1, p2
    if a <= eps:  # segment 1 is a point
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(d1 @ r)
        if e <= eps:  # segment 2 is a point
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:  # general non-degenerate case
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    return c1, c2


def _segment_box_distance(
    p: np.ndarray, q: np.ndarray, center: np.ndarray, half_extents: np.ndarray, orient: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Minimum distance from a segment [p,q] to a (possibly oriented) box.

    The squared distance from a point on the segment to a convex box is convex
    along the segment parameter t, so a 1D golden-section search converges to
    the global minimum. Returns (distance, pt_on_segment, pt_on_box).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d = q - p

    def closest(t: float) -> Tuple[float, np.ndarray, np.ndarray]:
        pt = p + t * d
        cp = _closest_point_box(pt, center, half_extents, orient)
        diff = pt - cp
        return float(diff @ diff), pt, cp

    gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate
    a, b = 0.0, 1.0
    c = b - gr * (b - a)
    dd = a + gr * (b - a)
    fc = closest(c)[0]
    fd = closest(dd)[0]
    # Eighteen iterations leave less than 0.2 mm parameter uncertainty for a
    # 1 m segment while avoiding the heavy 60-iteration cost in live checks.
    for _ in range(18):
        if fc < fd:
            b, dd, fd = dd, c, fc
            c = b - gr * (b - a)
            fc = closest(c)[0]
        else:
            a, c, fc = c, dd, fd
            dd = a + gr * (b - a)
            fd = closest(dd)[0]
    t = 0.5 * (a + b)
    dist_sq, pt, cp = closest(t)
    return float(np.sqrt(dist_sq)), pt, cp


# ═══════════════════════════════════════════════════════════════════════════
# CollisionDetector
# ═══════════════════════════════════════════════════════════════════════════


class CollisionDetector:
    """Unified collision detector for all agents and environment objects.

    Parameters:
        threshold:  World-space distance threshold (m).  Distances below this
                    are reported as proximity/collision events.
    """

    def __init__(self, threshold: float = 0.12):
        self.threshold = threshold

        # Registered agents: name -> dict of {link_name: SingleXFormPrim}
        self._agents: Dict[str, dict] = {}

        # Environment obstacles
        self._boxes: Dict[str, dict] = {}  # name -> {center, half_extents, orient}
        self._ground_z: Optional[float] = None  # ground plane Z height

        # FRANKA_LINKS — link names shared across all Franka arms
        self.franka_links = [
            'panda_link0',
            'panda_link1',
            'panda_link2',
            'panda_link3',
            'panda_link4',
            'panda_link5',
            'panda_link6',
            'panda_link7',
            'panda_hand',
            'panda_leftfinger',
            'panda_rightfinger',
        ]

        # FRANKA_CAPSULES — each arm link is modelled as a capsule: the line
        # segment between two consecutive link frames, swept by a radius. This
        # replaces the old point-per-link approximation so that the *whole*
        # length of every link (not just its origin) is tested for collision.
        # Tuple: (parent_link, child_link, radius_m). Radii are conservative
        # bounding-cylinder estimates of the Franka Panda geometry.
        # NOTE: panda_link0->link1 (base column) is excluded since it's fixed
        # to the mounting surface and doesn't move during execution.
        self.franka_capsules = [
            # ("panda_link0", "panda_link1", 0.060),  # excluded: fixed base
            ('panda_link1', 'panda_link2', 0.060),
            ('panda_link2', 'panda_link3', 0.060),
            ('panda_link3', 'panda_link4', 0.055),
            ('panda_link4', 'panda_link5', 0.050),
            ('panda_link5', 'panda_link6', 0.050),
            ('panda_link6', 'panda_link7', 0.045),
            ('panda_link7', 'panda_hand', 0.045),
            ('panda_hand', 'panda_leftfinger', 0.022),
            ('panda_hand', 'panda_rightfinger', 0.022),
        ]

    # ── registration ────────────────────────────────────────────────────

    def add_agent(self, name: str, link_xforms: dict) -> None:
        """Register an agent with its link SingleXFormPrim readers.

        Args:
            name:        Agent identifier (e.g. "franka_a").
            link_xforms: Dict of {link_name: SingleXFormPrim}, as returned
                         by setup_link_xforms() in two_arm_collide.py.
        """
        self._agents[name] = link_xforms

    def add_box(
        self,
        name: str,
        center: Union[list, tuple, np.ndarray],
        half_extents: Union[list, tuple, np.ndarray],
        orient: Optional[Union[list, tuple, np.ndarray]] = None,
    ) -> None:
        """Register a box-shaped obstacle.

        Args:
            name:         Obstacle identifier (e.g. "table").
            center:       3D world center of the box.
            half_extents: (hx, hy, hz) half-dimensions.
            orient:       Orientation quaternion [w, x, y, z] (optional).
        """
        self._boxes[name] = {
            'center': np.asarray(center, dtype=np.float64),
            'half_extents': np.asarray(half_extents, dtype=np.float64),
            'orient': _orientation_matrix(orient) if orient is not None else None,
        }

    def add_ground(self, z: float = 0.0) -> None:
        """Register a ground plane at Z = *z*."""
        self._ground_z = float(z)

    # ── internal helpers ────────────────────────────────────────────────

    def _read_link_positions(self, name: str) -> Tuple[List[str], np.ndarray]:
        """Read world positions of all links for agent *name*.

        Returns (link_names, positions) where positions is (N, 3).
        Silently skips links whose XForm reads fail.
        """
        link_xforms = self._agents.get(name, {})
        names = []
        pos_list = []
        for ln in self.franka_links:
            xf = link_xforms.get(ln)
            if xf is None:
                continue
            try:
                pos, _ = xf.get_world_pose()
                names.append(ln)
                pos_list.append(np.asarray(pos, dtype=np.float64))
            except Exception:
                pass
        if not pos_list:
            return [], np.empty((0, 3))
        return names, np.stack(pos_list)

    def _read_link_position_map(self, name: str) -> Dict[str, np.ndarray]:
        """Read world positions of all links as a {link_name: pos} dict."""
        link_xforms = self._agents.get(name, {})
        out: Dict[str, np.ndarray] = {}
        for ln in self.franka_links:
            xf = link_xforms.get(ln)
            if xf is None:
                continue
            try:
                pos, _ = xf.get_world_pose()
                out[ln] = np.asarray(pos, dtype=np.float64)
            except Exception:
                pass
        return out

    def _build_capsules(self, name: str) -> List[Tuple[str, np.ndarray, np.ndarray, float]]:
        """Build the capsule chain for agent *name* from current link poses.

        Returns a list of (capsule_label, p, q, radius) where [p, q] is the
        world-space segment and *radius* its sweep radius. Capsules whose
        endpoints are unavailable are skipped.
        """
        pos = self._read_link_position_map(name)
        return self._build_capsules_from_positions(pos)

    def _build_capsules_from_positions(
        self, pos: Dict[str, np.ndarray]
    ) -> List[Tuple[str, np.ndarray, np.ndarray, float]]:
        """Build capsule chain from an externally-supplied {link_name: position} dict.

        Used by check_config() for offline (non-physics) collision checks.
        """
        capsules: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
        for parent, child, radius in self.franka_capsules:
            if parent in pos and child in pos:
                label = f'{parent}->{child}'
                capsules.append((label, pos[parent], pos[child], float(radius)))
        return capsules

    def check_config(
        self, link_positions: Dict[str, np.ndarray], step: int = 0, agent_name: str = 'agent'
    ) -> List[CollisionEvent]:
        """Check a *hypothetical* set of link world positions against obstacles.

        Unlike check_agent_env, this does NOT read live XForms — positions are
        supplied directly (e.g. from FK during execution-time preflight), so it
        never touches the physics simulation.

        Each arm link is modelled as a capsule (segment + radius) for accurate
        body-vs-box and body-vs-ground clearance.
        """
        capsules = self._build_capsules_from_positions(link_positions)
        if not capsules:
            return []

        events: List[CollisionEvent] = []

        for bname, box in self._boxes.items():
            for label, p, q, radius in capsules:
                d_seg, pt, cp = _segment_box_distance(
                    p,
                    q,
                    box['center'],
                    box['half_extents'],
                    box['orient'],
                )
                clearance = d_seg - radius
                if clearance < self.threshold:
                    events.append(
                        CollisionEvent(
                            step=step,
                            kind='agent_env',
                            entity_a=f'{agent_name}/{label}',
                            entity_b=bname,
                            distance=clearance,
                            threshold=self.threshold,
                            pos_a=pt,
                            pos_b=cp,
                        )
                    )

        if self._ground_z is not None:
            for label, p, q, radius in capsules:
                lo_z = min(p[2], q[2])
                pt = p if p[2] <= q[2] else q
                clearance = (lo_z - self._ground_z) - radius
                if clearance < self.threshold:
                    events.append(
                        CollisionEvent(
                            step=step,
                            kind='agent_env',
                            entity_a=f'{agent_name}/{label}',
                            entity_b='ground',
                            distance=clearance,
                            threshold=self.threshold,
                            pos_a=pt,
                            pos_b=np.array([pt[0], pt[1], self._ground_z]),
                        )
                    )

        return events

    # ── collision checks ────────────────────────────────────────────────

    def check_inter_agent(self, step: int = 0) -> List[CollisionEvent]:
        """Check capsule-vs-capsule clearance between all registered agent pairs.

        Each arm link is a capsule (segment + radius); the reported distance is
        the true surface-to-surface clearance (centerline distance minus both
        radii), so the *whole* length and thickness of each link is considered.
        Returns a list of CollisionEvent for capsule pairs below threshold.
        """
        events: List[CollisionEvent] = []
        agent_names = list(self._agents.keys())
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                na, nb = agent_names[i], agent_names[j]
                caps_a = self._build_capsules(na)
                caps_b = self._build_capsules(nb)
                if not caps_a or not caps_b:
                    continue

                for la, pa, qa, ra in caps_a:
                    for lb, pb, qb, rb in caps_b:
                        c1, c2 = _closest_segment_segment(pa, qa, pb, qb)
                        centerline = float(np.linalg.norm(c1 - c2))
                        clearance = centerline - ra - rb
                        if clearance < self.threshold:
                            events.append(
                                CollisionEvent(
                                    step=step,
                                    kind='inter_agent',
                                    entity_a=f'{na}/{la}',
                                    entity_b=f'{nb}/{lb}',
                                    distance=clearance,
                                    threshold=self.threshold,
                                    pos_a=c1,
                                    pos_b=c2,
                                )
                            )
        return events

    def check_agent_env(self, step: int = 0) -> List[CollisionEvent]:
        """Check capsule-vs-environment clearance for every agent.

        Each arm link is a capsule (segment + radius); clearance to a box is the
        true segment-to-box surface distance minus the radius, and clearance to
        the ground is the minimum link-segment height minus the radius. This
        tests the entire link body, not just its origin point.
        Returns a list of CollisionEvent for capsules below threshold.
        """
        events: List[CollisionEvent] = []

        for aname in self._agents:
            capsules = self._build_capsules(aname)
            if not capsules:
                continue

            # ── box obstacles ──
            for bname, box in self._boxes.items():
                for label, p, q, radius in capsules:
                    d_seg, pt, cp = _segment_box_distance(
                        p,
                        q,
                        box['center'],
                        box['half_extents'],
                        box['orient'],
                    )
                    clearance = d_seg - radius
                    if clearance < self.threshold:
                        events.append(
                            CollisionEvent(
                                step=step,
                                kind='agent_env',
                                entity_a=f'{aname}/{label}',
                                entity_b=bname,
                                distance=clearance,
                                threshold=self.threshold,
                                pos_a=pt,
                                pos_b=cp,
                            )
                        )

            # ── ground plane ──
            if self._ground_z is not None:
                for label, p, q, radius in capsules:
                    # lowest point of the capsule body
                    lo_z = min(p[2], q[2])
                    pt = p if p[2] <= q[2] else q
                    clearance = (lo_z - self._ground_z) - radius
                    if clearance < self.threshold:
                        events.append(
                            CollisionEvent(
                                step=step,
                                kind='agent_env',
                                entity_a=f'{aname}/{label}',
                                entity_b='ground',
                                distance=clearance,
                                threshold=self.threshold,
                                pos_a=pt,
                                pos_b=np.array([pt[0], pt[1], self._ground_z]),
                            )
                        )

        return events

    def check_all(self, step: int = 0) -> List[CollisionEvent]:
        """Run both inter-agent and agent-environment collision checks.

        Returns a combined list of all CollisionEvent instances.
        """
        return self.check_inter_agent(step) + self.check_agent_env(step)

    def has_collision(self, step: int = 0) -> bool:
        """Quick check: True if ANY collision event exists this step."""
        return len(self.check_all(step)) > 0

    # ── convenience summary ─────────────────────────────────────────────

    def summary(self, events: List[CollisionEvent]) -> str:
        """Return a human-readable summary string for a list of events."""
        if not events:
            return 'no collision'
        by_kind: Dict[str, int] = {}
        min_d = float('inf')
        for e in events:
            by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
            if e.distance < min_d:
                min_d = e.distance
        parts = [f'{v} {k}' for k, v in by_kind.items()]
        return f"{', '.join(parts)}  (min={min_d*100:.1f}cm)"


# ═══════════════════════════════════════════════════════════════════════════
# convenience: mirror two_arm_collide.py setup function
# ═══════════════════════════════════════════════════════════════════════════


def setup_link_xforms(prefix: str) -> dict:
    """Create SingleXFormPrim readers for all Franka links under *prefix*.

    Mirrors the function in demos/two_arm_collide.py for standalone usage.
    """
    try:
        from isaacsim.core.prims import SingleXFormPrim
    except ImportError:
        from omni.isaac.core.prims import SingleXFormPrim

    FRANKA_LINKS = [
        'panda_link0',
        'panda_link1',
        'panda_link2',
        'panda_link3',
        'panda_link4',
        'panda_link5',
        'panda_link6',
        'panda_link7',
        'panda_hand',
        'panda_leftfinger',
        'panda_rightfinger',
    ]
    xforms = {}
    for name in FRANKA_LINKS:
        try:
            xf = SingleXFormPrim(prim_path=f'{prefix}/{name}')
            xf.get_world_pose()
            xforms[name] = xf
        except Exception:
            pass
    return xforms
