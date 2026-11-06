"""Constructor census must inspect real types without calling unrelated H code."""
import gc
import ctypes
import weakref

import numpy as np
import pytest

from anygeometry._planning_constructor import audit
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_executor import freeze_model, thaw_model, unpack
from anygeometry._planning_qualification import QualificationDecline
from anygeometry.surfaces import Plane


@pytest.mark.parametrize('hostile_metaclass', [False, True])
def test_constructor_census_does_not_dispatch_unrelated_class_hooks(hostile_metaclass):
    events = []

    class HostileMeta(type):
        def __getattribute__(cls, name):
            if name in ('__class__', '__mro__', '__bases__'):
                events.append(('metaclass', name))
                raise AssertionError('unrelated metaclass hook executed')
            return super().__getattribute__(name)

        def __subclasscheck__(cls, subclass):
            events.append(('subclass', subclass))
            raise AssertionError('unrelated subclass hook executed')

    class Foreign(metaclass=HostileMeta if hostile_metaclass else type):
        @property
        def __class__(self):
            events.append(('instance', '__class__'))
            raise AssertionError('unrelated instance hook executed')

    foreign = Foreign()
    assert gc.is_tracked(foreign)
    assert audit({}) == []
    assert events == []


def _plane():
    return Plane((0., 0., 0.), (5., 0., 0.), (0., 5., 0.))


def test_plane_normal_warm_freeze_is_cold_packet_and_preserves_host():
    model, _domains = _fixture()
    cold = freeze_model(model)
    for face in model.faces.values():
        face.surface.normal
    before = tuple((id(face.surface.normal), face.surface.normal.tobytes(),
                    bool(face.surface.normal.flags.writeable)) for face in model.faces.values())
    warm = freeze_model(model)
    assert warm == cold
    assert 'constructor_state' not in unpack(warm, 'model')
    rebuilt = thaw_model(warm)
    assert all('normal' not in vars(face.surface) for face in rebuilt.faces.values())
    # from_dict installs an empty extension mapping; WorkerModel deliberately
    # normalizes only absent -> empty metadata, never changed/nonempty extensions.
    from anygeometry._planning_stages import WorkerModel
    from anygeometry.serialization import to_dict
    assert WorkerModel(warm).verify() == warm['sha256']
    assert to_dict(rebuilt) == to_dict(model)
    assert tuple((id(face.surface.normal), face.surface.normal.tobytes(),
                  bool(face.surface.normal.flags.writeable)) for face in model.faces.values()) == before


def test_plane_normal_is_revalidated_after_an_earlier_success():
    value = _plane()
    value.normal
    assert audit({'plane': value}) == []
    value.normal.flags.writeable = True
    value.normal[2] = .5
    value.normal.flags.writeable = False
    with pytest.raises(QualificationDecline):
        audit({'plane': value})


@pytest.mark.parametrize('kind', ['signed-zero', 'unknown-extra', 'writable', 'dtype',
                                 'shape', 'external-reference', 'weak-reference', 'tracked-view'])
def test_plane_normal_noncanonical_state_refuses(kind):
    value = _plane()
    value.normal
    outside = None
    if kind == 'signed-zero':
        value.normal.flags.writeable = True
        value.normal[0] = -0.
        value.normal.flags.writeable = False
    elif kind == 'unknown-extra':
        object.__setattr__(value, 'foreign_cache', 1)
    elif kind == 'writable':
        value.normal.flags.writeable = True
    elif kind == 'dtype':
        object.__setattr__(value, 'normal', value.normal.astype(np.float32))
    elif kind == 'shape':
        object.__setattr__(value, 'normal', value.normal.reshape(1, 3))
    elif kind == 'external-reference':
        outside = value.normal
    elif kind == 'weak-reference':
        outside = weakref.ref(value.normal)
    elif kind == 'tracked-view':
        class Tracked(np.ndarray):
            pass
        raw = (ctypes.c_double * 3).from_address(value.normal.ctypes.data)
        outside = np.ctypeslib.as_array(raw).view(Tracked)
        assert gc.is_tracked(outside)
    with pytest.raises(QualificationDecline):
        audit({'plane': value})
    if kind in ('external-reference', 'tracked-view'):
        assert outside.shape == (3,)
