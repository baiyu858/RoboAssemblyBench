"""A state accessor must not disable sleeping as a contact-report side effect."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from internutopia.core.robot.isaacsim.rigid_body import IsaacsimRigidBody
from internutopia.core.util.pose_mixin import PoseMixin


@pytest.fixture
def native(monkeypatch):
    class Attribute:
        def __init__(self):
            self.authored = True
            self.value = .01
            self.default = 5e-5

        def HasAuthoredValueOpinion(self):
            return self.authored

        def Get(self):
            return self.value if self.authored else self.default

        def Clear(self):
            self.authored = False

    attr = Attribute()
    prim = SimpleNamespace(IsValid=lambda: True, GetAttribute=lambda name: attr)
    creations = []

    class NativeBody:
        def __init__(self, **params):
            self.prim_path, self.name = params['prim_path'], params['name']
            self._rigid_prim_view = object()
            # Reproduce the installed SingleRigidPrim's unwanted side effect.
            attr.authored, attr.value = True, 0.
            creations.append(params)

        def set_sleep_threshold(self, value):
            attr.authored, attr.value = True, value

        def get_sleep_threshold(self):
            return attr.Get()

    modules = {
        'isaacsim.core.prims': {'SingleRigidPrim': NativeBody},
        'isaacsim.core.utils.prims': {'get_prim_at_path': lambda path: prim},
        'isaacsim.core.utils.stage': {'add_reference_to_stage': lambda *a, **kw: None},
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(PoseMixin.env_offset_map, '0', [0., 0., 0.])
    return SimpleNamespace(attr=attr, creations=creations)


@pytest.mark.parametrize('threshold', [.01, 0., 1e-5])
def test_repeated_state_wrapping_preserves_authored_sleep_threshold(native, threshold):
    native.attr.value = threshold
    for _ in range(3):
        body = IsaacsimRigidBody(prim_path='/World/env_0/part', name='part')
        assert body.unwrap().get_sleep_threshold() == threshold
        assert native.attr.HasAuthoredValueOpinion()
    assert len(native.creations) == 3


def test_unconfigured_sleep_uses_schema_default_instead_of_forced_zero(native):
    native.attr.authored = False
    body = IsaacsimRigidBody(prim_path='/World/env_0/part', name='part')
    assert body.unwrap().get_sleep_threshold() == native.attr.default
    assert not native.attr.HasAuthoredValueOpinion()


@pytest.mark.parametrize('has_status', [False, True])
def test_restore_status_preserves_sleep_and_refreshes_native_view(native, monkeypatch, has_status):
    body = IsaacsimRigidBody(prim_path='/World/env_0/part', name='part')
    previous_view = body._rigid_prim_view
    body.status = {'saved': True} if has_status else {}
    restored = []
    monkeypatch.setattr('internutopia.core.robot.isaacsim.rigid_body.set_rigidbody_status',
                        lambda prim, status: restored.append((prim, status)))
    monkeypatch.setattr(body, 'save_status', lambda: None)
    body.restore_status()
    assert body.unwrap().get_sleep_threshold() == .01
    assert body._rigid_prim_view is body.unwrap()._rigid_prim_view
    assert body._rigid_prim_view is not previous_view
    assert restored == ([(body.unwrap(), body.status)] if has_status else [])
