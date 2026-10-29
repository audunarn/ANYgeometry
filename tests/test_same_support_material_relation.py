"""Fresh material winding proofs, independently of history/clone/load."""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from anygeometry import (GeometryError, GeometryModel, Plane, Straight, apply_plane_split, plan_plane_split,
    query_same_support_material_disjoint_interiors as query,
    query_prepared_partition_disjoint_interiors as live_query, to_dict, from_dict)
from anygeometry.nurbs import NurbsSurface, NurbsCurve, CurveOnSurface
from anygeometry.plane_nurbs import CURRENT_WORK, charge_box
from anygeometry.section_audit import Work
from anygeometry.same_support_material_relation import _atomic_intervals
from test_plane_split import bowl


def flat(reverse_u=False, reverse_v=False):
    return NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))),
        u_range=(1., 0.) if reverse_u else (0., 1.), v_range=(1., 0.) if reverse_v else (0., 1.))


def polygon(model, support, points, holes=()):
    loops = []
    for values in (points, *holes):
        vertices = model.add_points(tuple(tuple(float(x) for x in support.exact_point(*p))
                                         if type(support) is NurbsSurface else tuple(support.evaluate(*p)) for p in values))
        edges = []
        for i, point in enumerate(values):
            if type(support) is NurbsSurface:
                curve = CurveOnSurface(support, NurbsCurve(1, (0., 0., 1., 1.), (point, values[(i+1)%len(values)])))
                edges.append(model.add_curve(vertices[i], vertices[(i+1)%len(values)], curve))
            else:
                edges.append(model.add_line(vertices[i], vertices[(i+1)%len(values)]))
        loops.append(tuple(edges))
    face = model.add_face(loops[0], surface=support)
    if len(loops) > 1:
        # add_face has no holes keyword. Use the owner's existing committed
        # hole-fixture construction, then fresh validation (no proof receipt).
        from anygeometry.entities import OrientedEdge
        with model.transaction():
            model._put_entity('face', replace(model.faces[face], holes=tuple(
                tuple(OrientedEdge(edge, True) for edge in loop) for loop in loops[1:])))
    return face


def rectangle(x0, x1, y0=0., y1=1.):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def pair(model, ids):
    return tuple(model.handle('face', identifier) for identifier in ids)


@pytest.fixture(scope='module')
def flat_cut():
    model = GeometryModel(); support = flat()
    face = polygon(model, support, rectangle(0., 1.))
    result = apply_plane_split(model, plan_plane_split(model, [face],
        Plane((.6, 0., 0.), (0., 1., 0.), (0., 0., 1.))), policy='imprint')
    return model, tuple(dict(result.parent_children)[face])


@pytest.fixture(scope='module')
def closed_cut():
    model, face, plane = bowl()
    result = apply_plane_split(model, plan_plane_split(model, [face], plane), policy='imprint')
    return model, tuple(dict(result.parent_children)[face])


@pytest.mark.parametrize('copy', ('direct', 'clone', 'load'))
def test_flat_public_split_regenerates_after_plain_clone_and_load(flat_cut, copy):
    original, ids = flat_cut
    model = original if copy == 'direct' else original.clone() if copy == 'clone' else from_dict(to_dict(original))
    operands = pair(model, ids)
    assert live_query(model, *operands) is False
    before = to_dict(model)
    assert query(model, *operands) is True
    assert to_dict(model) == before


@pytest.mark.parametrize('copy', ('direct', 'clone', 'load'))
def test_closed_public_disk_complement_cancel_only_fresh_complete_cycle(closed_cut, copy):
    original, ids = closed_cut
    model = original if copy == 'direct' else original.clone() if copy == 'clone' else from_dict(to_dict(original))
    operands = pair(model, ids)
    assert live_query(model, *operands) is False
    assert query(model, *operands) is True


@pytest.mark.parametrize('reverse_u,reverse_v', ((True, False), (False, True), (True, True)))
def test_exact_axis_split_with_reversed_support_ranges(reverse_u, reverse_v):
    model = GeometryModel(); support = flat(reverse_u, reverse_v)
    face = polygon(model, support, rectangle(0., 1.))
    apply_plane_split(model, plan_plane_split(model, [face],
        Plane((.5, 0., 0.), (0., 1., 0.), (0., 0., 1.))), policy='imprint')
    assert query(model, *pair(model, tuple(model.faces))) is True


@pytest.mark.parametrize('first,second', (
    ((0., .5, 0., 1.), (.5, 1., 0., 1.)),
    ((0., .5, 0., 1.), (.5, 1., .25, .75)),
    ((0., .25, 0., 1.), (.75, 1., 0., 1.)),
    ((0., .75, 0., 1.), (.25, 1., 0., 1.)),
    ((0., 1., 0., 1.), (.25, .75, .25, .75)),
    ((0., 1., 0., 1.), (0., 1., 0., 1.)),
))
def test_exact_polygon_relation_matches_independent_rectangle_area(first, second):
    model = GeometryModel(); support = flat()
    ids = polygon(model, support, rectangle(*first)), polygon(model, support, rectangle(*second))
    overlap = max(Q(0), min(Q(first[1]), Q(second[1]))-max(Q(first[0]), Q(second[0])))
    overlap *= max(Q(0), min(Q(first[3]), Q(second[3]))-max(Q(first[2]), Q(second[2])))
    assert query(model, *pair(model, ids)) is (overlap == 0)


def test_polygon_hole_fill_and_distinct_hole_material_are_qualified():
    model = GeometryModel(); support = flat()
    inner = rectangle(.25, .75, .25, .75)
    ring = polygon(model, support, rectangle(0., 1.), (inner[::-1],))
    disk = polygon(model, support, inner)
    assert query(model, *pair(model, (ring, disk))) is True
    inside_hole = polygon(model, support, rectangle(.375, .625, .375, .625))
    assert query(model, *pair(model, (ring, inside_hole))) is True


def test_repeated_public_axis_cuts_and_saved_partitions():
    model = GeometryModel(); face = polygon(model, flat(), rectangle(0., 1.))
    for x in (.25, .75):
        apply_plane_split(model, plan_plane_split(model, tuple(model.faces),
            Plane((x, 0., 0.), (0., 1., 0.), (0., 0., 1.))), policy='imprint')
    loaded = from_dict(to_dict(model))
    ids = tuple(loaded.faces)
    assert len(ids) == 3
    for i, first in enumerate(ids):
        for second in ids[i+1:]: assert query(loaded, *pair(loaded, (first, second))) is True


def test_boundary_orientation_is_independent_of_material_side():
    model = GeometryModel(); support = flat()
    ids = polygon(model, support, rectangle(0., .5)[::-1]), polygon(model, support, rectangle(.5, 1.))
    assert query(model, *pair(model, ids)) is True


def test_unrelated_parameterization_and_different_chart_are_unknown():
    model = GeometryModel(); support = flat()
    first = polygon(model, support, rectangle(0., .5))
    second = polygon(model, replace(support, u_range=(0., .5)), rectangle(.5, 1.))
    assert query(model, *pair(model, (first, second))) is False
    model = GeometryModel(); support = flat()
    first = polygon(model, support, rectangle(0., .5))
    second = polygon(model, support, rectangle(.5, 1.))
    assert query(model, *pair(model, (first, second))) is True
    face = model.faces[first]
    original = face.parameterization
    object.__setattr__(face, 'parameterization', replace(support))
    try: assert query(model, *pair(model, (first, second))) is False
    finally: object.__setattr__(face, 'parameterization', original)


def test_equal_support_swap_and_detached_original_poison_refuses():
    model = GeometryModel()
    a = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    b = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    ids = polygon(model, a, rectangle(0., .5)), polygon(model, b, rectangle(.5, 1.))
    face = model.faces[ids[0]]; origin = a.origin
    clean = Plane(tuple(a.origin), tuple(a.u_vector), tuple(a.v_vector))
    def attack(phase):
        if phase == 'fresh material relation':
            object.__setattr__(face, 'surface', clean)
            changed = origin.copy(); changed[0] += .1
            object.__setattr__(a, 'origin', changed)
    try:
        with pytest.raises(GeometryError): query(model, *pair(model, ids), cancellation_check=attack)
    finally:
        object.__setattr__(face, 'surface', a); object.__setattr__(a, 'origin', origin)


def test_cancellation_and_callback_exception_identity(flat_cut):
    model, ids = flat_cut
    with pytest.raises(GeometryError, match='cancelled'):
        query(model, *pair(model, ids), cancellation_check=lambda phase: True)
    original = GeometryError('NURBS support immersion and injective projection are not certified')
    def callback(phase):
        if phase == 'fresh material relation injectivity': raise original
    with pytest.raises(GeometryError) as caught:
        query(model, *pair(model, ids), cancellation_check=callback)
    assert caught.value is original


def test_callback_private_unsupported_exception_identity_is_preserved(flat_cut):
    from anygeometry.same_support_material_relation import _Unsupported
    model, ids = flat_cut
    original = _Unsupported('caller-owned private-type exception')
    calls = 0
    def callback(phase):
        nonlocal calls
        calls += 1
        if calls == 2: raise original
        return False
    with pytest.raises(_Unsupported) as caught:
        query(model, *pair(model, ids), cancellation_check=callback)
    assert caught.value is original


def test_native_kernel_poison_and_wrong_handles_are_hard_refusals(flat_cut):
    original, ids = flat_cut
    model = original.clone()
    curve = next(edge.curve.pcurve for edge in model.edges.values() if type(edge.curve) is CurveOnSurface)
    old = curve._prepared
    bad = replace(curve, weights=(1., 2.))
    object.__setattr__(curve, '_prepared', bad._prepared)
    try:
        with pytest.raises(GeometryError): query(model, *pair(model, ids))
    finally: object.__setattr__(curve, '_prepared', old)
    with pytest.raises(GeometryError): query(model, *pair(original, ids))


def test_zero_work_and_many_collinear_event_work_are_bounded(flat_cut):
    model, ids = flat_cut
    token = CURRENT_WORK.set(Work(0, 12))
    try:
        with pytest.raises(GeometryError): query(model, *pair(model, ids))
    finally: CURRENT_WORK.reset(token)
    token = CURRENT_WORK.set(Work(20, 12))
    try:
        with pytest.raises(GeometryError):
            tuple(_atomic_intervals(((Q(i), Q(i+2)) for i in range(1000)), charge_box))
    finally: CURRENT_WORK.reset(token)


def test_event_sweep_opposite_partial_ranges_cancel_exactly():
    rows = ((Q(0), Q(1)), (Q(3, 4), Q(1, 4)))
    assert tuple(_atomic_intervals(rows, lambda: None)) == (
        (Q(0), Q(1, 4), 1), (Q(1, 4), Q(3, 4), 0), (Q(3, 4), Q(1), 1))
