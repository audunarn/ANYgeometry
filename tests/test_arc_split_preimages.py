"""Independent circle/winding counterexamples and actual split recording."""
from fractions import Fraction as F
from types import SimpleNamespace

import pytest

from anygeometry import GeometryError, plan_intersections, to_dict
from anygeometry.cylinder_boundary_correspondence import _arc_geometric_tiling
from anygeometry.edge_subcurve_preimages import (
    ArcEdgeDefinition, _coplanar_circle_bound, _edge_ancestry_definition,
    _pack, _point_circle_distance, _sqrt_interval,
)
import test_edge_subcurve_preimages as owner


def point(x, y):
    return (F(x), F(y), F(0))


def record(edge, start, via, end):
    positions = tuple(tuple(_pack(v) for v in p) for p in (start, via, end))
    return SimpleNamespace(edge_id=edge,
        current_definition=ArcEdgeDefinition(edge, edge * 3, edge * 3 + 1,
                                             edge * 3 + 2, positions, 'fixture'))


O = point(1, 0)
N = point(0, 1)
W = point(-1, 0)
S = point(0, -1)
A = point(F(3, 5), F(4, 5))
B = point(F(-3, 5), F(4, 5))
C = point(F(-4, 5), F(-3, 5))
D = point(F(-4, 5), F(3, 5))
E = point(F(-3, 5), F(-4, 5))


def test_quarter_tiling_rejects_extra_revolution():
    rows = (record(1, O, N, W), record(2, W, E, S), record(3, S, O, N))
    assert _arc_geometric_tiling((O, A, N), rows) is not None


def test_major_tiling_rejects_backtracking_with_every_anchor_inside():
    # All anchors lie in the original 270-degree span, but this chain winds
    # 630 degrees. Pairwise local ordering cannot distinguish the two.
    rows = (record(1, O, N, B), record(2, B, W, C),
            record(3, C, E, A), record(4, A, N, D), record(5, D, W, S))
    assert _arc_geometric_tiling((O, W, S), rows) is not None


@pytest.mark.parametrize('reverse', (False, True))
def test_complete_major_tiling_accepts_both_stored_edge_directions(reverse):
    triples = ((O, N, B), (B, W, C), (C, E, S))
    rows = tuple(record(i + 1, *(tuple(reversed(t)) if reverse else t))
                 for i, t in enumerate(triples))
    assert _arc_geometric_tiling((O, W, S), rows) is None


def test_circle_bound_covers_overlapping_square_root_intervals():
    # Rational center/radius data realizing the independently found bug.
    # True max distance is 4 - (sqrt(2)-sqrt(17/10)) > 3.88.
    bound = _coplanar_circle_bound((F(0),) * 3, F(16),
                                  (F(1), F(1), F(0)), F(17, 10))
    assert bound >= F(97, 25)


def test_coplanar_small_refit_has_small_whole_circle_bound():
    eps = F(1, 2**40)
    bound = _coplanar_circle_bound((F(0),) * 3, F(1),
                                  (eps, F(0), F(0)), F(1))
    assert eps <= bound <= 2 * eps


def test_point_circle_distance_includes_plane_displacement():
    low, high = _point_circle_distance((F(0),) * 3, F(1),
                                     (F(0), F(0), F(1)),
                                     (F(1, 10), F(0), F(1)))
    assert low * low <= F(181, 100) <= high * high
    assert high > F(13, 10)


def test_square_root_enclosure_is_sound():
    for value in (F(2), F(17, 10), F(1, 2**80), F(16)):
        low, high = _sqrt_interval(value)
        assert 0 <= low <= high and low * low <= value <= high * high


def test_actual_repeated_arc_splits_preserve_ancestry_not_parameter_identity():
    source, _, _ = owner.fixture()
    vertices = source.add_points(((8, 0, 0), (9, 1, 0), (10, 0, 0)))
    arc = source.add_arc(*vertices)
    plan = plan_intersections(source, tuple(source.faces), policy='connect')
    draft = owner.capture(source, allow_seed=True)
    candidate = source.clone(preserve_identity=True)

    def split(edge, parameter):
        parent = _edge_ancestry_definition(candidate, edge)
        _, children = candidate.split_edge(edge, parameter)
        owner.record(draft, edge, parameter, children, 1e-9,
                     model=candidate, parent_definition=parent)
        return children

    left, right = split(arc, .31)
    children = (*split(left, .47), right)
    binding = owner.commit(source, candidate, draft, plan)
    rows = [r for r in binding.arc_records if r.ancestor.definition.edge_id == arc]
    assert {r.edge_id for r in rows} == set(children)
    assert all(r.parameter_mapping_qualified is False for r in rows)
    assert all(r.ancestor.definition.positions == rows[0].ancestor.definition.positions
               for r in rows)
    spans = sorted((F(*r.interval[0]), F(*r.interval[1])) for r in rows)
    assert spans[0][0] == 0 and spans[-1][1] == 1
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    before = to_dict(source)
    marker = RuntimeError('caller cancellation')

    def cancel(_):
        raise marker

    with pytest.raises(RuntimeError) as caught:
        owner.query(source, cancellation_check=cancel)
    assert caught.value is marker and to_dict(source) == before
    source.add_point(20, 20, 20)
    with pytest.raises(GeometryError, match='stale|binding|preparation'):
        owner.validate(source, binding)
