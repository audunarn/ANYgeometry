"""Conservative bounds-index pruning evidence for the planar owner receipt.

The optimized material membership path is compared against an independently
retained exhaustive oracle on small deterministic fixtures: outside carriers,
material holes, touching faces, far coplanar faces, parallel distinct
planes, concavity, arbitrarily oriented planes, rescaled plane frames and
endpoint transverse boundaries. Pruning is measured deterministically
through the census helper; no test in this file times execution.
"""
import importlib.util
from pathlib import Path
import sys
from fractions import Fraction

import pytest

from anygeometry import (GeometryError, plan_intersections, apply_intersections,
    to_dict, query_prepared_planar_member_sheet_network as query,
    validate_prepared_planar_member_sheet_network_binding as validate)
import anygeometry.prepared_planar_member_sheet_network as module
from anygeometry.prepared_planar_member_sheet_network import (
    _boundary_events, _chart_uv, _face_bounds_index, _material_candidate_census,
    _material_membership, _on_boundary_xyz, _on_plane, _point_in_face)

spec = importlib.util.spec_from_file_location('planar_bounds_fixtures',
    Path(__file__).parents[1]/'tools/general_intersections/large_connected_fixtures.py')
builders = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builders
spec.loader.exec_module(builders)


def build(family='connected_hub', count=4):
    fixture = getattr(builders, family)(count)
    model = fixture.model
    plan = plan_intersections(model, fixture.operands, policy='connect')
    apply_intersections(model, plan, policy='connect')
    joint = next(a.target_id for a in model.attachments.values() if a.kind == 'sheet_on_joint')
    return model, joint


def snapshots(model, joint):
    from anygeometry.prepared_sheet_joint_component import _index
    scope = query(model, joint).scope
    return scope, _index(scope.authored_document), _index(scope.current_document)


def carrier(*points):
    """Carrier controls as exact (numerator, denominator) coordinate pairs."""
    return tuple(tuple((Fraction(x).numerator, Fraction(x).denominator)
                       for x in point) for point in points)


def make_frame(origin, u, v):
    """Exact planar chart frame with its dual-basis inverse rows."""
    origin, u, v = (tuple(Fraction(x) for x in row) for row in (origin, u, v))
    uu, uv, vv = (sum(x*y for x, y in zip(u, u)), sum(x*y for x, y in zip(u, v)),
                  sum(x*y for x, y in zip(v, v)))
    determinant = uu*vv-uv*uv
    assert determinant > 0
    inverse = (tuple((vv*x-uv*y)/determinant for x, y in zip(u, v)),
               tuple((uu*y-uv*x)/determinant for x, y in zip(u, v)))
    return {'kind': 'plane', 'origin': origin, 'u': u, 'v': v, 'inverse': inverse}


def make_chart(frame, outer, holes=()):
    """Consistent exact uv/xyz loop pairs for one certified-style face."""
    uv_loops, xyz_loops = [], []
    for polygon in (outer, *holes):
        xyz = [tuple(Fraction(x) for x in point) for point in polygon]
        uv = [_chart_uv(frame, point) for point in xyz]
        xyz_loops.append(tuple(zip(xyz, xyz[1:]+xyz[:1])))
        uv_loops.append(tuple(zip(uv, uv[1:]+uv[:1])))
    return (frame, tuple(uv_loops), tuple(xyz_loops))


def diverse_charts():
    """Ten exact faces: touching, holed, concave, far, tilted and transverse."""
    flat = make_frame((0, 0, 0), (1, 0, 0), (0, 1, 0))
    charts = {
        1: make_chart(flat, ((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0))),
        2: make_chart(flat, ((4, 0, 0), (8, 0, 0), (8, 4, 0), (4, 4, 0))),
        3: make_chart(flat, ((100, 0, 0), (104, 0, 0), (104, 4, 0), (100, 4, 0))),
        4: make_chart(flat, ((10, 0, 0), (14, 0, 0), (14, 4, 0),
                             (12, 4, 0), (12, 2, 0), (10, 2, 0))),
        5: make_chart(flat, ((20, 0, 0), (24, 0, 0), (24, 4, 0), (20, 4, 0)),
                      holes=(((21, 1, 0), (22, 1, 0), (22, 2, 0), (21, 2, 0)),)),
        6: make_chart(make_frame((3, 0, 0), (1, -1, 0), (1, 1, -2)),
                      ((3, 0, 0), (0, 3, 0), (0, 0, 3))),
        7: make_chart(make_frame((30, 0, 0), (1, 0, 0), (0, 0, 1)),
                      ((30, 0, -1), (32, 0, -1), (32, 0, 1), (30, 0, 1))),
        8: make_chart(make_frame((0, 0, 0), (2, 0, 0), (0, 3, 0)),
                      ((40, 0, 0), (44, 0, 0), (44, 4, 0), (40, 4, 0))),
        9: make_chart(make_frame((0, 0, 5), (1, 0, 0), (0, 1, 0)),
                      ((0, 0, 5), (4, 0, 5), (4, 4, 5), (0, 4, 5))),
        10: make_chart(make_frame((2, -1, -1), (0, 1, 0), (0, 0, 1)),
                       ((2, -1, -1), (2, 3, -1), (2, 3, 1), (2, -1, 1))),
    }
    return {key: charts[key] for key in sorted(charts)}


BATTERY = (
    ('inside_square', (1, 1, 0), (3, 3, 0)),
    ('across_touching_squares', (1, 2, 0), (7, 2, 0)),
    ('along_shared_edge', (4, 1, 0), (4, 3, 0)),
    ('along_top_edges', (3, 4, 0), (5, 4, 0)),
    ('outside_all_material', (50, 50, 0), (51, 50, 0)),
    ('no_support_plane', (10, 10, 7), (11, 10, 7)),
    ('far_parallel_plane_outside', (10, 10, 5), (11, 10, 5)),
    ('inside_parallel_plane_face', (1, 1, 5), (2, 2, 5)),
    ('through_hole', (Fraction(41, 2), Fraction(3, 2), 0),
     (Fraction(47, 2), Fraction(3, 2), 0)),
    ('beside_hole', (Fraction(41, 2), Fraction(1, 2), 0),
     (Fraction(47, 2), Fraction(1, 2), 0)),
    ('ends_on_hole_boundary', (Fraction(41, 2), Fraction(3, 2), 0),
     (21, Fraction(3, 2), 0)),
    ('concave_bar', (Fraction(21, 2), 1, 0), (Fraction(27, 2), 1, 0)),
    ('concave_notch', (11, 3, 0), (11, 1, 0)),
    ('tilted_triangle_interior', (2, Fraction(1, 2), Fraction(1, 2)),
     (Fraction(1, 2), 2, Fraction(1, 2))),
    ('tilted_and_planar_together', (1, 2, 0), (2, 1, 0)),
    ('transverse_endpoint_on_far_boundary', (1, 1, 0), (30, 0, 0)),
    ('transverse_boundary_corner_in_material', (1, 1, 0), (2, 3, 0)),
    ('transverse_face_partial', (2, -2, 0), (2, Fraction(-1, 2), 0)),
    ('inside_rescaled_frame_face', (41, 1, 0), (43, 2, 0)),
    ('far_coplanar_inside', (101, 1, 0), (103, 2, 0)),
    ('touching_bbox_only', (8, 1, 0), (9, 1, 0)),
    ('hole_graze', (Fraction(43, 2), Fraction(3, 2), 0),
     (Fraction(45, 2), Fraction(3, 2), 0)),
)


def exhaustive_membership(controls, charts, check=None):
    """Independently retained exhaustive oracle: the original algorithm.

    Every face is enumerated for coplanarity, every coplanar face contributes
    boundary events, every interval midpoint and endpoint is classified
    against every coplanar face, and the endpoint fallback scans every face.
    Nothing is ever pruned.
    """
    prefix = 'prepared planar member Sheet network '
    segment = tuple(tuple(Fraction(*value) for value in point) for point in controls)
    coplanar = [(face_id, frame, _chart_uv(frame, segment[0]),
                 _chart_uv(frame, segment[1]), uv_loops)
                for face_id, (frame, uv_loops, _) in sorted(charts.items())
                if _on_plane(frame, segment[0]) and _on_plane(frame, segment[1])]
    if not coplanar:
        raise GeometryError(prefix+'original Member carrier lies in no original support plane')
    events = {Fraction(0), Fraction(1)}
    for face_id, frame, a, b, uv_loops in coplanar:
        if check is not None:
            check()
        events |= _boundary_events(a, b, uv_loops)
    ordered = sorted(events)
    intervals = []
    for low, high in zip(ordered, ordered[1:]):
        if check is not None:
            check()
        middle = (low+high)/2
        faces = sorted(face_id for face_id, frame, a, b, uv_loops in coplanar
                       if _point_in_face(((1-middle)*a[0]+middle*b[0],
                                          (1-middle)*a[1]+middle*b[1]), uv_loops))
        if not faces:
            raise GeometryError(prefix+'original Member carrier leaves the original face material')
        intervals.append({'interval': [[low.numerator, low.denominator],
                                       [high.numerator, high.denominator]],
                          'material_faces': faces})
    endpoints = []
    for point in segment:
        faces = sorted(face_id for face_id, frame, a, b, uv_loops in coplanar
                       if _point_in_face(_chart_uv(frame, point), uv_loops))
        if not faces:
            faces = sorted(face_id for face_id, (_, _, xyz_loops) in charts.items()
                           if _on_boundary_xyz(point, xyz_loops))
        if not faces:
            raise GeometryError(prefix+'original Member carrier endpoint leaves the original face material')
        endpoints.append(faces)
    return {'endpoint_material_faces': endpoints,
            'interior_material_intervals': intervals}


def assert_membership_agrees(controls, charts):
    """Optimized membership and the exhaustive oracle must never differ."""
    try:
        optimized = _material_membership(controls, charts, lambda: None)
        optimized_error = None
    except GeometryError as error:
        optimized, optimized_error = None, error
    try:
        oracle = exhaustive_membership(controls, charts)
        oracle_error = None
    except GeometryError as error:
        oracle, oracle_error = None, error
    if optimized_error is not None or oracle_error is not None:
        assert optimized_error is not None and oracle_error is not None, (
            str(optimized_error), str(oracle_error))
        assert str(optimized_error) == str(oracle_error)
    else:
        assert optimized == oracle


@pytest.mark.parametrize('name,start,end', BATTERY)
def test_optimized_membership_matches_exhaustive_oracle(name, start, end):
    assert_membership_agrees(carrier(start, end), diverse_charts())


@pytest.mark.parametrize('name,start,end', BATTERY)
def test_grouped_coplanarity_matches_per_face_scan(name, start, end):
    charts = diverse_charts()
    index = _face_bounds_index(charts)
    segment = tuple(tuple(Fraction(x) for x in point) for point in (start, end))
    per_face = sorted(face_id for face_id, (frame, _, _) in charts.items()
                      if _on_plane(frame, segment[0]) and _on_plane(frame, segment[1]))
    assert index.coplanar_faces(segment) == per_face


@pytest.mark.parametrize('name,start,end', BATTERY)
def test_pruned_faces_carry_no_events_or_material(name, start, end):
    """Every face the index removes is provably irrelevant, exactly."""
    charts = diverse_charts()
    index = _face_bounds_index(charts)
    segment = tuple(tuple(Fraction(x) for x in point) for point in (start, end))
    coplanar = index.coplanar_faces(segment)
    pruned = set(coplanar)-set(index.material_candidates(segment, coplanar))
    for face_id in pruned:
        frame, uv_loops, _ = charts[face_id]
        a, b = (_chart_uv(frame, point) for point in segment)
        assert _boundary_events(a, b, uv_loops) == set()
        assert not _point_in_face(a, uv_loops) and not _point_in_face(b, uv_loops)


def test_bound_pruning_never_hides_material_or_boundary_points():
    """Closed-bound containment is implied by material and edge membership."""
    charts = diverse_charts()
    index = _face_bounds_index(charts)
    steps = [Fraction(n, 2) for n in range(-2, 9)]
    for face_id, (frame, uv_loops, xyz_loops) in charts.items():
        origin, u, v = frame['origin'], frame['u'], frame['v']
        for first in steps:
            for second in steps:
                point = tuple(o+first*x+second*y
                              for o, x, y in zip(origin, u, v))
                if _point_in_face(_chart_uv(frame, point), uv_loops):
                    assert index.bound_contains(face_id, point)
                if _on_boundary_xyz(point, xyz_loops):
                    assert index.bound_contains(face_id, point)


@pytest.mark.parametrize('name,start,end,expected', [
    ('inside_square', (1, 1, 0), (3, 3, 0),
     {'certified_faces': 10, 'coplanar_faces': 6, 'material_candidates': 1,
      'conservatively_pruned_faces': 5}),
    ('along_shared_edge', (4, 1, 0), (4, 3, 0),
     {'certified_faces': 10, 'coplanar_faces': 6, 'material_candidates': 2,
      'conservatively_pruned_faces': 4}),
    ('far_parallel_plane_outside', (10, 10, 5), (11, 10, 5),
     {'certified_faces': 10, 'coplanar_faces': 1, 'material_candidates': 0,
      'conservatively_pruned_faces': 1}),
    ('inside_rescaled_frame_face', (41, 1, 0), (43, 2, 0),
     {'certified_faces': 10, 'coplanar_faces': 6, 'material_candidates': 1,
      'conservatively_pruned_faces': 5}),
    ('far_coplanar_inside', (101, 1, 0), (103, 2, 0),
     {'certified_faces': 10, 'coplanar_faces': 6, 'material_candidates': 1,
      'conservatively_pruned_faces': 5}),
])
def test_deterministic_pruning_census_on_diverse_charts(name, start, end, expected):
    assert _material_candidate_census(carrier(start, end), diverse_charts()) == expected


def _source_carrier_censuses(source):
    from anygeometry.prepared_planar_member_sheet_network import _planar_face_charts
    charts = _planar_face_charts(source, lambda: None)
    censuses = []
    for member in sorted(source['members']):
        for key in source['members'][member]['edge_use_ids']:
            edge = source['edges'][source['member_edge_uses'][key]['edge_id']]
            controls = tuple(tuple((Fraction(float(x)).numerator,
                                    Fraction(float(x)).denominator)
                                  for x in source['vertices'][vertex]['position'])
                             for vertex in (edge['start'], edge['end']))
            censuses.append(_material_candidate_census(controls, charts))
    return censuses


@pytest.mark.parametrize('family,count,expected', [
    ('connected_strip', 10, {'certified_faces': 7, 'coplanar_faces': 5,
                             'material_candidates': 1,
                             'conservatively_pruned_faces': 4}),
    ('connected_hub', 4, {'certified_faces': 3, 'coplanar_faces': 1,
                          'material_candidates': 1,
                          'conservatively_pruned_faces': 0}),
])
def test_deterministic_pruning_census_on_real_fixtures(family, count, expected):
    model, joint = build(family, count)
    _, source, _ = snapshots(model, joint)
    assert _source_carrier_censuses(source) == [expected]*len(source['members'])


@pytest.mark.parametrize('family,count,points', [('connected_strip', 10, 0),
                                                 ('connected_hub', 4, 4)])
def test_public_receipts_remain_equivalent(family, count, points):
    model, joint = build(family, count)
    before = to_dict(model)
    receipt = query(model, joint)
    assert len(receipt.relations['point_contacts']) == points
    validate(model, receipt)
    assert to_dict(model) == before


@pytest.mark.parametrize('family,count', [('connected_strip', 10),
                                         ('connected_hub', 4)])
def test_real_fixture_payload_matches_independent_oracle(monkeypatch, family, count):
    """The whole qualification payload is invariant under the exhaustive oracle."""
    model, joint = build(family, count)
    scope, source, current = snapshots(model, joint)
    optimized = module._qualify(model, scope, source, current, None).payload

    def oracle_membership(controls, charts, check, bounds_index=None):
        return exhaustive_membership(controls, charts, check)

    monkeypatch.setattr(module, '_material_membership', oracle_membership)
    assert module._qualify(model, scope, source, current, None).payload == optimized


@pytest.mark.parametrize('family,count', [('connected_strip', 10),
                                         ('connected_hub', 4)])
def test_real_fixture_membership_matches_unpruned_index(monkeypatch, family, count):
    """A never-pruning index reproduces the optimized payload exactly."""
    model, joint = build(family, count)
    scope, source, current = snapshots(model, joint)
    optimized = module._qualify(model, scope, source, current, None).payload
    original = module._material_membership

    def unpruned(controls, charts, check, bounds_index=None):
        index = _face_bounds_index(charts)
        for face_id in index.entries:
            index.entries[face_id] = index.entries[face_id]._replace(bounds=None)
        return original(controls, charts, check, index)

    monkeypatch.setattr(module, '_material_membership', unpruned)
    assert module._qualify(model, scope, source, current, None).payload == optimized


def test_real_fixture_point_contacts_match_unpruned_index(monkeypatch):
    """The transverse contact scan is invariant under never-pruning bounds."""
    model, joint = build('connected_hub', 4)
    scope, source, current = snapshots(model, joint)
    captured = {}
    original = module._qualify_member_point_contacts

    def capture(*args):
        captured['args'] = args
        return original(*args)

    monkeypatch.setattr(module, '_qualify_member_point_contacts', capture)
    module._qualify(model, scope, source, current, None)
    args = captured['args']
    pruned = original(*args)
    index = _face_bounds_index(args[6])
    for face_id in index.entries:
        index.entries[face_id] = index.entries[face_id]._replace(bounds=None)
    unpruned = original(*args[:11], index)
    assert pruned == unpruned


def test_membership_cancellation_checks_are_retained():
    charts = diverse_charts()

    def cancelled():
        raise GeometryError('prepared planar member Sheet network cancelled')

    with pytest.raises(GeometryError, match='cancelled'):
        _material_membership(carrier((1, 1, 0), (3, 3, 0)), charts, cancelled)


def test_membership_check_granularity_is_per_candidate_and_interval():
    charts = diverse_charts()
    counter = {'calls': 0}

    def check():
        counter['calls'] += 1

    membership = _material_membership(carrier((1, 1, 0), (3, 3, 0)), charts, check)
    assert len(membership['interior_material_intervals']) == 1
    # One retained candidate contributes events, one interval is classified.
    assert counter['calls'] == 2


def test_refusal_types_are_preserved_under_full_pruning():
    charts = diverse_charts()
    with pytest.raises(GeometryError, match='no original support plane'):
        _material_membership(carrier((10, 10, 7), (11, 10, 7)), charts, lambda: None)
    # The carrier lies in one support plane but every candidate is pruned;
    # the typed material refusal is unchanged from the exhaustive scan.
    with pytest.raises(GeometryError, match='leaves the original face material'):
        _material_membership(carrier((10, 10, 5), (11, 10, 5)), charts, lambda: None)
    with pytest.raises(GeometryError, match='leaves the original face material'):
        exhaustive_membership(carrier((10, 10, 5), (11, 10, 5)), charts)
