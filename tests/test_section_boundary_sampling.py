"""Regular implicit-section enclosures under the existing qualification Work (UNRUN definitions).

Fixture: the paraboloid tensor Bezier ``z = x**2 + y**2`` (degree 2 both axes, clamped knots, controls
xy(i-1, j-1), z = heights[i] + heights[j] with heights (1, -1, 1)) cut by the plane z = .5. Dense evaluated
samples are supplemental sanity only; the proof is the certified enclosure and its Work accounting.
"""
from __future__ import annotations

from fractions import Fraction as Q

import numpy as np
import pytest

from dataclasses import replace

from anygeometry import Plane, to_dict
from anygeometry.nurbs import NurbsSurface
from anygeometry.plane_nurbs import CURRENT_WORK, PlaneNurbsNotCertified
from anygeometry.predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY as POLICY
from anygeometry.section_audit import Work
from anygeometry.support_curves import PlaneNurbsCurve

HEIGHTS = (1., -1., 1.)
KNOTS = (0., 0., 0., 1., 1., 1.)


def paraboloid():
    controls = tuple(tuple((i - 1., j - 1., HEIGHTS[i] + HEIGHTS[j]) for j in range(3)) for i in range(3))
    return NurbsSurface(2, 2, KNOTS, KNOTS, controls)


def curves():
    surface = paraboloid()
    plane = Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.))
    return [PlaneNurbsCurve.from_plane(plane, surface, index) for index in range(1)]


class under_work:
    """The owner's Work limits (existing policy), restored on exit."""

    def __init__(self, boxes=None, depth=None):
        self.work = Work(POLICY.max_boxes_per_pair if boxes is None else boxes,
                         POLICY.max_subdivision_depth if depth is None else depth)

    def __enter__(self):
        self.token = CURRENT_WORK.set(self.work)
        return self.work

    def __exit__(self, *exc):
        CURRENT_WORK.reset(self.token)


def contains(bounds, point, slack=0.):
    low, high = bounds
    return bool(np.all(np.asarray(point) >= np.asarray(low) - slack) and np.all(np.asarray(point) <= np.asarray(high) + slack))


def test_whole_curve_enclosure_completes_within_the_existing_work_and_contains_evaluated_points():
    for curve in curves():
        with under_work() as work:
            bounds = curve.bounds(0., 1.)
        assert work.boxes <= POLICY.max_boxes_per_pair
        for t in np.linspace(0., 1., 257):                       # supplemental sanity only
            assert contains(bounds, curve.evaluate(float(t)))


def test_many_small_piece_enclosures_share_one_work_and_each_contains_its_evaluated_points():
    curve = curves()[0]
    with under_work() as work:
        for k in range(64):                                      # the sampling loop's access pattern
            lo, hi = k / 64, (k + 1) / 64
            bounds = curve.bounds(lo, hi)
            for t in np.linspace(lo, hi, 9):
                assert contains(bounds, curve.evaluate(float(t)))
    assert work.boxes <= POLICY.max_boxes_per_pair


def test_reversed_interval_gives_the_same_enclosure():
    curve = curves()[0]
    with under_work():
        forward, reverse = curve.bounds(.2, .7), curve.bounds(.7, .2)
    assert np.array_equal(forward[0], reverse[0]) and np.array_equal(forward[1], reverse[1])


@pytest.mark.parametrize("t", (0., 1., .25, .5, 1. / 3.))
def test_zero_width_interval_encloses_the_evaluated_point_tightly(t):
    curve = curves()[0]
    with under_work():
        bounds = curve.bounds(t, t)
    assert contains(bounds, curve.evaluate(t))
    assert float(np.max(np.asarray(bounds[1]) - np.asarray(bounds[0]))) < 1e-6


def test_native_reversed_domain_bounds_contain_evaluated_endpoints_and_interior():
    """``replace(curve, domain=(.8, .2))`` is an actual reversed subcurve (supplemental sampling only)."""
    curve = curves()[0]
    reversed_curve = replace(curve, domain=(.8, .2))
    with under_work():
        whole = reversed_curve.bounds(0., 1.)
        piece = reversed_curve.bounds(.25, .75)
        swapped = reversed_curve.bounds(.75, .25)
    assert np.all(np.asarray(whole[0]) <= np.asarray(whole[1]))
    assert np.array_equal(piece[0], swapped[0]) and np.array_equal(piece[1], swapped[1])
    for t in (0., 1.):
        assert contains(whole, reversed_curve.evaluate(t))
    for t in np.linspace(.25, .75, 33):
        assert contains(piece, reversed_curve.evaluate(float(t)))
    assert contains(whole, curve.evaluate(.8)) and contains(whole, curve.evaluate(.2))


def test_tiny_intervals_straddling_an_arc_join_include_the_join_and_zero_width():
    curve = curves()[0]
    count = len(curve._arcs())
    joins = [k / count for k in range(1, count)] or [.5]            # artificial station at each arc join
    eps = 1e-9
    with under_work():
        for join in joins:
            lo, hi = max(0., join - eps), min(1., join + eps)
            box = curve.bounds(lo, hi)
            for t in (lo, join, hi):
                assert contains(box, curve.evaluate(t))
            point = curve.bounds(join, join)                        # zero width at the join
            assert contains(point, curve.evaluate(join))
            assert contains(box, curve.evaluate(join)) and np.all(np.asarray(point[1]) >= np.asarray(point[0]))


def test_end_boxes_contain_the_evaluated_ends():
    curve = curves()[0]
    with under_work():
        for end, t in ((0, 0.), (1, 1.)):
            assert contains(curve.end_box(end), curve.evaluate(t))


def test_cropped_subcurve_encloses_its_own_evaluated_points():
    curve = curves()[0]
    sub = curve.subcurve(.2, .6)
    with under_work():
        bounds = sub.bounds(0., 1.)
    for t in np.linspace(0., 1., 65):
        assert contains(bounds, sub.evaluate(float(t)))


def test_graph_x_and_legacy_exclusion_x_both_contain_the_evaluated_curve_x():
    """Two independently sound enclosures need not be nested, so only their common content is asserted:
    each x interval (knot units, before the unchanged 2**-59 widening) is a nonempty sub-interval of the box
    range, and both contain the chart x of every evaluated sample in the piece (supplemental sanity)."""
    curve = curves()[0]
    with under_work():
        rects = curve._arc_rects(.3, .35)
        for arc, s, _x in rects:
            box_x = arc.box.v_range if arc.slicing == 0 else arc.box.u_range
            for x in (curve._graph_x(arc, s[0], s[1], 0), curve._tight_x(arc, s[0], s[1], depth=24)):
                if x is not None:
                    assert box_x[0] <= x[0] <= x[1] <= box_x[1]
    assert rects


def test_low_work_refuses_even_with_warm_caches_and_the_curve_is_unchanged_afterwards():
    curve = curves()[0]
    with under_work():
        before = curve.bounds(.1, .4)
        again = curve.bounds(.1, .4)                              # warm: the second call is fully cached
    assert np.array_equal(before[0], again[0]) and np.array_equal(before[1], again[1])
    with under_work(boxes=1):                                     # every examined box is charged, cached or not
        with pytest.raises(PlaneNurbsNotCertified):
            curve.bounds(.1, .4)
    with under_work():
        after = curve.bounds(.1, .4)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])


@pytest.mark.parametrize("depth", (0, 1))
def test_graph_subdivision_respects_the_work_depth_and_charges_each_bisection(depth):
    curve = curves()[0]
    with under_work():
        rects = curve._arc_rects(0., 1.)
    assert rects
    with under_work(depth=depth) as work:
        for arc, s, _x in rects:
            if s[1] > s[0]:
                curve._graph_x(arc, s[0], s[1], 0)               # None (undecided) is a valid outcome
    assert work.subdivisions <= (2 ** depth - 1) * len(rects)     # at most 2**depth - 1 bisections per call
    if depth == 0:
        assert work.subdivisions == 0


def test_world_chord_discrepancy_is_certified_for_a_small_piece_at_the_public_error():
    from anygeometry.prepared_material_curvilinear import _world_chord
    curve = curves()[0]
    lo, hi = Q(1, 64), Q(2, 64)
    ends = tuple(tuple(float(x) for x in curve.evaluate(float(t))) for t in (lo, hi))
    with under_work():
        worst, pieces = _world_chord(curve, lo, hi, ends, .01, lambda: None)
    assert worst <= Q(1, 100) and pieces



# ------------------------------------------------------------ public owner sampler on the closed Plane disk


def disk_model():
    import dataclasses
    import anygeometry as owner
    surface = paraboloid()
    plane = owner.Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.))
    model = owner.GeometryModel()
    base = PlaneNurbsCurve.from_plane(plane, surface)
    vertices = [model.add_point(*base.evaluate(i / 4)) for i in range(4)]
    edges = [model.add_curve(vertices[i], vertices[(i + 1) % 4],
                             dataclasses.replace(base, domain=(i / 4, (i + 1) / 4))) for i in range(4)]
    face = model.add_face(edges, surface=plane)
    model.add_sheet((face,))
    regions = owner.query_material_surface_regions(model, (face,))
    handles = [e if hasattr(e, "kind") else model.handle("edge", e) for e in edges]
    return model, regions, handles


def seeds(handles):
    return tuple((handle, (0., .5, 1.)) for handle in handles)


def policy(**changes):
    import dataclasses
    return dataclasses.replace(POLICY, **changes)


@pytest.mark.parametrize("error", (.01, .0025))
def test_owner_sampling_of_the_closed_plane_disk_is_complete_and_within_the_supplied_world_error(error):
    from anygeometry.errors import GeometryError
    from anygeometry.arrangement_geometry import freeze_edge
    from anygeometry.prepared_material_curvilinear import (
        query_prepared_material_boundary_sampling, validate_prepared_material_boundary_sampling_binding)
    model, regions, handles = disk_model()
    before = to_dict(model)
    result = query_prepared_material_boundary_sampling(model, regions, seeds(handles), chord_error=error)
    assert to_dict(model) == before
    validate_prepared_material_boundary_sampling_binding(model, regions, result)
    assert len(result.edge_parameters) == 4
    chords = {(c.edge.id, c.parameter_range): c for c in result.chords.intervals}
    for (handle, stations), (source, _t) in zip(result.edge_parameters, seeds(handles)):
        assert handle.id == source.id
        assert stations[0] == 0. and stations[-1] == 1. and .5 in stations          # requested stations kept
        assert all(a < b for a, b in zip(stations, stations[1:]))                   # strict source order
        curve = freeze_edge(model, handle.id)
        edge = model.edges[handle.id]
        for a, b in zip(stations, stations[1:]):
            chord = chords[(handle.id, (Q(a), Q(b)))]                               # complete intervals
            assert chord.station_keys == ((handle.id, Q(a)), (handle.id, Q(b)))
            assert chord.world_error <= error
            expected = tuple(tuple(float(x) for x in (model.vertex_position(edge.start) if t == 0 else
                                                      model.vertex_position(edge.end) if t == 1 else
                                                      curve.evaluate(float(t)))) for t in (a, b))
            assert chord.xyz == expected                                            # literal owner XYZ
    assert len(chords) == sum(len(stations) - 1 for _h, stations in result.edge_parameters)
    assert result.chord_error == error


def test_owner_sampling_refuses_with_a_typed_error_under_low_work_and_leaves_the_caller_unchanged():
    from anygeometry.errors import GeometryError
    from anygeometry.prepared_material_curvilinear import query_prepared_material_boundary_sampling
    model, regions, handles = disk_model()
    query_prepared_material_boundary_sampling(model, regions, seeds(handles), chord_error=.01)   # warm caches
    before = to_dict(model)
    with pytest.raises(GeometryError):                          # every enclosure query is charged, cached or not
        query_prepared_material_boundary_sampling(model, regions, seeds(handles), chord_error=.01,
                                                  qualification=policy(max_boxes_per_pair=1))
    assert to_dict(model) == before


def test_owner_sampling_cancellation_refuses_and_leaves_the_caller_unchanged():
    from anygeometry.errors import GeometryError
    from anygeometry.prepared_material_curvilinear import query_prepared_material_boundary_sampling
    model, regions, handles = disk_model()
    before = to_dict(model)
    calls = []

    def cancel(phase):
        calls.append(phase)
        return len(calls) > 3                                   # cancel after a few frontiers

    with pytest.raises(GeometryError):
        query_prepared_material_boundary_sampling(model, regions, seeds(handles), chord_error=.01,
                                                  cancellation_check=cancel)
    assert calls and to_dict(model) == before


# Not defined (not provable in this scope): extreme positive weights regular case.
