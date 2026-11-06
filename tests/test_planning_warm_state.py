"""Declarative transport keeps registered host caches intact, not authoritative."""
import json
from pathlib import Path

import pytest

from anygeometry._planning_executor import freeze_model, thaw_model
from anygeometry._planning_qualification import QualificationDecline, check_types, input_type_names
from anygeometry.serialization import from_dict, to_dict
from anygeometry.material_arrangement import MaterialDomain, ArrangementPath
from anygeometry.arrangement_geometry import LinePath
from anygeometry.surfaces import Plane


def model():
    return from_dict(json.loads((Path(__file__).parent / 'data/pp5c-cylinder-input.json').read_text()))


@pytest.mark.parametrize('warm', ['lengths', 'trim', 'spatial'])
def test_real_warm_model_transport_preserves_host_and_cold_packet(warm):
    value = model()
    cold = freeze_model(value)
    if warm == 'lengths':
        for edge in value.edges:
            value.edge_length(edge)
        cache = value._edge_length_cache
    elif warm == 'trim':
        for face in value.faces:
            value.face_trim_loops_uv(face)
        cache = value._trim_loops_cache
    else:
        cache = value._spatial()
    assert cache is not None
    cache_ids = {k: id(v) for k, v in vars(value).items()}
    packet = freeze_model(value)
    assert packet == cold
    assert {k: id(v) for k, v in vars(value).items()} == cache_ids
    if warm != 'spatial':
        assert cache
    rebuilt = thaw_model(packet)
    assert to_dict(rebuilt) == to_dict(value)
    assert rebuilt._edge_length_cache == {}
    assert rebuilt._trim_loops_cache == {}
    assert rebuilt._spatial_index is None


def domain():
    support = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    for array in (support.origin, support.u_vector, support.v_vector):
        array.flags.writeable = False
    points = ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.))
    loop = tuple(ArrangementPath(LinePath(a, b)) for a, b in zip(points, points[1:] + points[:1]))
    return MaterialDomain(1, support, (loop,))


def test_registered_material_domain_caches_are_ignored_without_mutation():
    value = domain()
    value._exact_chart()
    value._prepared_exact_boundary(value.boundaries[0])
    value._float_ray_scale(value.boundaries[0])
    names = ('_exact_chart_cache', '_exact_boundary_cache', '_float_scale_cache')
    before = tuple(getattr(value, name) for name in names)
    assert all(part is not None for part in before)
    check_types(value, {'types': input_type_names()})
    from anygeometry._planning_constructor import audit
    assert audit({'domain': value}) == []
    assert all(getattr(value, name) is part for name, part in zip(names, before))


@pytest.mark.parametrize('kind', ['hook', 'transaction', 'unknown', 'preparation', 'cache-type'])
def test_authority_and_unregistered_state_remain_refused(kind):
    value = model()
    if kind == 'hook':
        value._change_hooks.append(lambda *args: None)
    elif kind == 'transaction':
        value._transaction_journal = object()
    elif kind == 'unknown':
        value._unknown_cache = {}
    elif kind == 'preparation':
        value._anyfem_transport_requires_preparation = True
    else:
        value._edge_length_cache = []
    with pytest.raises(QualificationDecline):
        freeze_model(value)


def test_warm_cache_cannot_hide_constructor_mutation():
    value = model()
    value._edge_length_cache[123] = (0, 1.)
    surface = next(face.surface for face in value.faces.values())
    object.__setattr__(surface, 'unregistered_extra', {})
    with pytest.raises(QualificationDecline):
        freeze_model(value)


def test_warm_cache_cannot_hide_extension_alias():
    value = model()
    extension = {'unknown': []}
    value._serialization_extensions = extension
    value._edge_length_cache[123] = extension['unknown']
    with pytest.raises(QualificationDecline, match='extension alias'):
        freeze_model(value)


def test_non_definition_metadata_does_not_register_policy_authority_as_cache():
    from anygeometry.batch_intersections import IntersectionBatchPolicy
    from anygeometry._planning_constructor import audit
    with pytest.raises(QualificationDecline):
        audit({'policy': IntersectionBatchPolicy(planning_mode='exact')})


def branch_curve():
    from anygeometry.branch_algebra import BezierRuledSupport
    from anygeometry.branch_curves import BezierQuadricCurve
    from anygeometry.quadric_algebra import QuadricSupport
    ruled = BezierRuledSupport(((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.)), (0., 0., 1.))
    quadric = QuadricSupport('general', matrix=(0., 0., 0., 0., 0., 0., 0., 0., 1.),
                             linear=(0., 0., 0.), constant=-1.)
    return BezierQuadricCurve(ruled, quadric, 0., .5)


def test_registered_inversion_cache_preserves_exact_container_requirement():
    from anygeometry._planning_constructor import audit
    curve = branch_curve()
    object.__setattr__(curve, '_inversions', None)
    with pytest.raises(QualificationDecline, match='cache type'):
        audit({'curve': curve})
