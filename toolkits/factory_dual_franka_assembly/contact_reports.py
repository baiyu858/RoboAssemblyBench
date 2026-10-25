"""Prepare USD contact reporting before PhysX starts; read it without reauthoring.

Adding PhysxContactReportAPI to a running body can make its instanced visuals
disappear. Keeping preparation separate also avoids RigidPrim's implicit
rigid-body/sleep-policy authoring when all we need is a filtered force view.
"""
from __future__ import annotations


def prepare_contact_reports(roots) -> set[str]:
    """Prepare task-owned bodies, static colliders and their object roots.

    Call after asset loading, before creating the episode's physics view.
    Instance proxies are inspected, never edited or deinstanced; their enclosing
    object/body receives the report API. Purely visual subtrees are untouched.
    No mass, sleep, collision, material or transform properties are changed.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    selected = {}
    for root in roots:
        if root is None or not root.IsValid():
            raise RuntimeError('Cannot prepare contact reporting on an invalid task prim')
        physical = [
            prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not physical:
            continue
        # Probe filters may name an enclosing USD object rather than its body.
        for prim in [root, *physical]:
            if not prim.IsInstanceProxy():
                selected[str(prim.GetPath())] = prim

    for path, prim in selected.items():
        report = (PhysxSchema.PhysxContactReportAPI(prim)
                  if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                  else PhysxSchema.PhysxContactReportAPI.Apply(prim))
        threshold = report.GetThresholdAttr()
        if not threshold or threshold.Get() != 0.0:
            if not report.CreateThresholdAttr().Set(0.0):
                raise RuntimeError(f'Failed to prepare contact-report threshold on {path}')
        require_contact_report(prim)
    return set(selected)


def require_contact_report(prim) -> None:
    """Read-only readiness check; never repair a live USD stage implicitly."""
    from pxr import PhysxSchema

    if prim is None or not prim.IsValid():
        raise RuntimeError('Contact reporting requested for an invalid prim')
    if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
        threshold = PhysxSchema.PhysxContactReportAPI(prim).GetThresholdAttr()
        if threshold and threshold.Get() == 0.0:
            return
    raise RuntimeError(
        f'Contact reporting was not prepared before physics initialization: {prim.GetPath()}; '
        'load this body/filter through the task scene setup before starting physics'
    )


def create_prepared_contact_view(prim, filter_prim, *, name, disable_stablization=False):
    """Create a force-only view without authoring contact/rigid-body schemas."""
    require_contact_report(prim)
    require_contact_report(filter_prim)
    try:
        from isaacsim.core.api.sensors.rigid_contact_view import RigidContactView
        from isaacsim.core.simulation_manager import SimulationManager
    except ImportError:
        from omni.isaac.core.sensors import RigidContactView
        from omni.isaac.core.simulation_manager import SimulationManager

    simulation_view = SimulationManager.get_physics_sim_view()
    if simulation_view is None:
        raise RuntimeError('Contact force view requested before physics initialization')
    probe = RigidContactView(
        prim_paths_expr=str(prim.GetPath()),
        filter_paths_expr=[str(filter_prim.GetPath())],
        name=name,
        prepare_contact_sensors=False,
        # Preserve the task's existing stabilization choice, not the native
        # contact view's default (which disables stabilization).
        disable_stablization=disable_stablization,
        max_contact_count=64,
    )
    probe.initialize(physics_sim_view=simulation_view)
    if not probe.is_physics_handle_valid():
        raise RuntimeError(f'Invalid initialized contact view for {prim.GetPath()}')
    return probe
