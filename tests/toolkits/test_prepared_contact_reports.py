"""Report setup belongs before PhysX starts, never in a live force getter."""
import copy
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from internutopia.core.task import BaseTask
from internutopia_extension.tasks.factory_dual_franka_assembly_task import FactoryDualFrankaAssemblyTask
from toolkits.factory_dual_franka_assembly.contact_reports import (
    create_prepared_contact_view, prepare_contact_reports, require_contact_report,
)


@pytest.fixture
def usd(monkeypatch):
    writes = []
    rigid, collision = object(), object()

    class Prim:
        def __init__(self, path, schemas=(), children=(), proxy=False):
            self.path, self.schemas, self.children = path, set(schemas), list(children)
            self.proxy = proxy
            self.threshold = None
            self.other = {'sleep': .01, 'mass': .25, 'visibility': 'inherited',
                          'instanceable': True, 'material': 'original_black'}

        def IsValid(self):
            return True

        def GetPath(self):
            return self.path

        def HasAPI(self, schema):
            return schema in self.schemas

        def IsInstanceProxy(self):
            return self.proxy

    class Attribute:
        def __init__(self, prim):
            self.prim = prim

        def Get(self):
            return self.prim.threshold

        def Set(self, value):
            writes.append(('threshold', self.prim.path, value))
            self.prim.threshold = value
            return True

    class Report:
        def __init__(self, prim):
            self.prim = prim

        @staticmethod
        def Apply(prim):
            assert not prim.proxy, 'Must never edit an instance proxy'
            writes.append(('apply', prim.path))
            prim.schemas.add(Report)
            return Report(prim)

        def GetThresholdAttr(self):
            return Attribute(self.prim)

        def CreateThresholdAttr(self):
            return Attribute(self.prim)

    def walk(root, *_):
        yield root
        for child in root.children:
            yield from walk(child)

    module = ModuleType('pxr')
    module.Usd = NS(PrimRange=walk, TraverseInstanceProxies=lambda: None)
    module.UsdPhysics = NS(RigidBodyAPI=rigid, CollisionAPI=collision)
    module.PhysxSchema = NS(PhysxContactReportAPI=Report)
    monkeypatch.setitem(sys.modules, 'pxr', module)
    return NS(Prim=Prim, rigid=rigid, collision=collision, report=Report, writes=writes)


def test_setup_covers_instanced_palm_nested_bodies_static_filters_but_not_visuals(usd):
    mesh = usd.Prim('/robot/palm/visuals/mesh', [usd.collision], proxy=True)
    palm = usd.Prim('/robot/palm', [usd.rigid], [mesh])
    robot = usd.Prim('/robot', children=[palm])
    part_body = usd.Prim('/part/body', [usd.rigid, usd.collision])
    part = usd.Prim('/part', children=[part_body])
    table = usd.Prim('/table', [usd.collision])
    visual = usd.Prim('/visual_only')
    prims = [robot, palm, mesh, part, part_body, table, visual]
    before = [copy.deepcopy(p.other) for p in prims]

    paths = prepare_contact_reports([robot, part, table, visual, robot])

    assert paths == {'/robot', '/robot/palm', '/part', '/part/body', '/table'}
    assert [p.other for p in prims] == before
    assert not mesh.HasAPI(usd.report) and not visual.HasAPI(usd.report)
    assert not table.HasAPI(usd.rigid)  # A static support must remain static.
    assert all(p.threshold == 0. for p in (robot, palm, part, part_body, table))
    count = len(usd.writes)
    assert prepare_contact_reports([robot, part, table, visual]) == paths
    assert len(usd.writes) == count  # Repeated setup is idempotent.


def test_invalid_setup_root_fails_instead_of_silently_skipping(usd):
    with pytest.raises(RuntimeError, match='invalid task prim'):
        prepare_contact_reports([None])


@pytest.mark.parametrize('threshold', [None, 1., float('nan')])
def test_readiness_rejects_bad_threshold_without_repairing_it(usd, threshold):
    body = usd.Prim('/body', [usd.rigid, usd.report])
    body.threshold = threshold
    with pytest.raises(RuntimeError, match='not prepared before physics'):
        require_contact_report(body)
    assert usd.writes == []


def test_readiness_rejects_missing_api_without_adding_one(usd):
    body = usd.Prim('/body', [usd.rigid])
    with pytest.raises(RuntimeError, match='/body'):
        require_contact_report(body)
    assert not body.HasAPI(usd.report) and usd.writes == []


@pytest.fixture
def native(monkeypatch):
    calls = []
    simulation_view = object()
    manager = NS(get_physics_sim_view=lambda: simulation_view)

    class View:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.valid = False

        def initialize(self, *, physics_sim_view):
            assert physics_sim_view is simulation_view
            self.valid = True

        def is_physics_handle_valid(self):
            return self.valid

    for name, values in {
        'isaacsim.core.api.sensors.rigid_contact_view': {'RigidContactView': View},
        'isaacsim.core.simulation_manager': {'SimulationManager': manager},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    return NS(calls=calls, manager=manager)


def test_runtime_view_reads_prepared_reports_without_any_usd_writes(usd, native):
    body, table = usd.Prim('/body', [usd.rigid]), usd.Prim('/table', [usd.collision])
    prepare_contact_reports([body, table])
    usd.writes.clear()
    probe = create_prepared_contact_view(body, table, name='probe')
    assert probe.is_physics_handle_valid()
    assert native.calls == [dict(prim_paths_expr='/body', filter_paths_expr=['/table'],
                                 name='probe', prepare_contact_sensors=False,
                                 disable_stablization=False, max_contact_count=64)]
    assert usd.writes == []


@pytest.mark.parametrize('missing', ['source', 'filter'])
def test_runtime_missing_report_never_constructs_or_repairs_view(usd, native, missing):
    body, table = usd.Prim('/body', [usd.rigid]), usd.Prim('/table', [usd.collision])
    prepare_contact_reports([body if missing == 'filter' else table])
    usd.writes.clear()
    with pytest.raises(RuntimeError, match='not prepared before physics'):
        create_prepared_contact_view(body, table, name='probe')
    assert usd.writes == [] and native.calls == []


def test_runtime_requires_live_physics_view(usd, native):
    body, table = usd.Prim('/body', [usd.rigid]), usd.Prim('/table', [usd.collision])
    prepare_contact_reports([body, table])
    native.manager.get_physics_sim_view = lambda: None
    with pytest.raises(RuntimeError, match='before physics initialization'):
        create_prepared_contact_view(body, table, name='probe')
    assert native.calls == []


def test_task_prepares_after_asset_loading(usd, monkeypatch):
    task = object.__new__(FactoryDualFrankaAssemblyTask)
    palm, part = usd.Prim('/robot/palm', [usd.rigid]), usd.Prim('/part', [usd.rigid])
    nodes = {p.path: p for p in [palm, part]}
    module = ModuleType('isaacsim.core.utils.prims')
    module.get_prim_at_path = nodes.get
    monkeypatch.setitem(sys.modules, 'isaacsim.core.utils.prims', module)
    order = []

    def load(self, scene):
        order.append('assets_loaded')
        self.robots = {'right': NS(config=NS(prim_path='/robot/palm'))}
        self.objects = {'part': NS(config=NS(prim_path='/part'))}

    monkeypatch.setattr(BaseTask, 'set_up_scene', load)
    task._preconfigure_attachment_gripper_collision_filters = lambda: order.append('filters')
    task.set_up_scene(object())
    assert order == ['assets_loaded', 'filters']
    assert task._prepared_contact_report_paths == set(nodes)
    assert all(p.HasAPI(usd.report) for p in nodes.values())


def test_task_reuses_valid_probe_and_rebuilds_invalid_handles(usd, monkeypatch):
    task = object.__new__(FactoryDualFrankaAssemblyTask)
    task.name, task._contact_probes = 'test', {}
    task.get_current_phase_spec = lambda: {'beam_coordinated': True}
    nodes = {p.path: p for p in [usd.Prim('/body'), usd.Prim('/table')]}
    module = ModuleType('isaacsim.core.utils.prims')
    module.get_prim_at_path = nodes.get
    monkeypatch.setitem(sys.modules, 'isaacsim.core.utils.prims', module)
    views, calls = [], []

    def create(*args, **kw):
        calls.append((args, kw))
        view = NS(valid=True)
        view.is_physics_handle_valid = lambda: view.valid
        views.append(view)
        return view

    monkeypatch.setattr('internutopia_extension.tasks.factory_dual_franka_assembly_task.create_prepared_contact_view', create)
    first = task._get_contact_probe('/body', '/table')
    assert task._get_contact_probe('/body', '/table') is first
    assert len(calls) == 1 and not calls[0][1]['disable_stablization']
    first.valid = False
    assert task._get_contact_probe('/body', '/table') is not first
    assert len(calls) == 2
    assert task._get_contact_probe('/missing', '/table') is None
    assert len(calls) == 2


def test_unprepared_contact_is_explicit_invalid_force_not_a_fake_zero():
    task = object.__new__(FactoryDualFrankaAssemblyTask)
    task._contact_physics_dt = lambda: 1 / 240

    def missing(*args):
        raise RuntimeError('Contact reporting was not prepared before physics initialization')

    task._get_contact_probe = missing
    result = task._pair_contact_observation('/finger', '/part')
    assert not result['valid']
    assert 'not prepared before physics' in result['error']
