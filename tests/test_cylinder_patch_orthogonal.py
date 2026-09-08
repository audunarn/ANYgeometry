"""Genuine concave cylinder trims and independent bounded rejection fixtures."""

from fractions import Fraction
import math
from types import SimpleNamespace

import pytest
from anygeometry import Cylinder, GeometryModel, OrientedEdge
from anygeometry.cylinder_patch import (
    CylinderPatchPolicy, CylinderPatchStatus, _PatchProof,
    query_cylinder_patch, validate_cylinder_patch_binding,
)
from anygeometry._cylinder_patch_orthogonal import certify_loop, certify_material
from anygeometry.cylinder_charts import _Refusal


def _notched(*, reverse=False, start=0., sweep=math.pi / 4, radius=1., height=2., origin=(0., 0., 0.)):
    model = GeometryModel()
    surface = Cylinder(origin, (0., 0., 1.), (1., 0., 0.), radius, height, start, sweep)
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.), (0., .625),
          (.25, .625), (.25, .375), (0., .375))
    vertices = {}
    def vertex(point):
        if point not in vertices:
            vertices[point] = model.add_point(*surface.evaluate(*point))
        return vertices[point]
    with model.transaction():
        uses = []
        for a, b in zip(uv, uv[1:] + uv[:1]):
            if a[1] == b[1]:
                midpoint = ((a[0] + b[0]) / 2, a[1])
                edge = model.add_arc(vertex(a), vertex(midpoint), vertex(b))
            else:
                edge = model.add_line(vertex(a), vertex(b))
            uses.append(OrientedEdge(edge, True))
        if reverse:
            uses = [OrientedEdge(use.edge, False) for use in reversed(uses)]
        face = model.add_face_from_loop(tuple(uses), surface=surface)
        part = model.add_part(name="orthogonal patch proof")
        sheet = model.add_sheet((face,), part_id=part)
    selected = (model.handle('face_use', model.sheets[sheet].face_use_ids[0]),)
    return model, selected


@pytest.mark.parametrize('settings', (
    {}, {'reverse': True}, {'start': 7 * math.pi / 4}, {'sweep': -math.pi / 4},
    {'height': -2.}, {'radius': 2., 'height': 4., 'origin': (2., -3., 5.)},
))
def test_notched_patch_certificate_is_pure_repeatable_and_fresh(settings):
    model, selected = _notched(**settings)
    before = repr(model.__dict__)
    result = query_cylinder_patch(model, selected, expected_revision=model.revision)
    assert result.status is CylinderPatchStatus.QUALIFIED, result.diagnostics
    assert result.certificate.complete and len(result.occurrences) == 8
    assert result.certificate.algorithm == 'cylinder-partial-orthogonal-v1'
    u, v = result.loops[0].seed_uv
    assert 0 < u < 1 and 0 < v < 1
    assert not (u < .25 and .375 < v < .625)
    assert query_cylinder_patch(model, selected, expected_revision=model.revision) == result
    validate_cylinder_patch_binding(model, result, selected, expected_revision=model.revision)
    assert repr(model.__dict__) == before


def _exact_rows(points, proof):
    rows = []
    for index, (a, b) in enumerate(zip(points, points[1:] + points[:1])):
        axis = 0 if a[1] == b[1] else 1
        assert a[1 - axis] == b[1 - axis]
        rows.append(dict(key=index, axis=axis, sign=1 if b[axis] > a[axis] else -1,
                         band=proof.i(a[1 - axis]),
                         first=tuple(proof.i(value) for value in a),
                         last=tuple(proof.i(value) for value in b)))
    return rows


@pytest.mark.parametrize('points', (
    ((0, 0), (3, 0), (3, 3), (1, 3), (1, -1), (2, -1), (2, 2), (0, 2)),
    ((0, 0), (3, 0), (3, 3), (1, 3), (1, 0), (2, 0), (2, 2), (0, 2)),
    ((0, 0), (3, 0), (3, 3), (1, 3), (1, Fraction(1, 10**9)),
     (2, Fraction(1, 10**9)), (2, 2), (0, 2)),
))
def test_crossing_touching_and_tolerance_ambiguous_tubes_reject(points):
    proof = _PatchProof(CylinderPatchPolicy(), None)
    geometry = SimpleNamespace(radius=Fraction(1), height=Fraction(1),
        surface=SimpleNamespace(sweep_angle=1.), parameter=Fraction(1, 10**6), tau=Fraction(1, 10**7))
    rows = _exact_rows(points, proof)
    with pytest.raises(_Refusal, match='patch_nonadjacent_carriers_touch'):
        certify_loop(rows, tuple(range(len(rows))), 0, geometry, proof)


def test_notched_budget_exhaustion_never_returns_consumable_evidence():
    model, selected = _notched()
    before = repr(model.__dict__)
    result = query_cylinder_patch(model, selected, expected_revision=model.revision,
                                 policy=CylinderPatchPolicy(max_pair_tests=1))
    assert result.status is not CylinderPatchStatus.QUALIFIED
    assert not result.certificate.complete
    assert result.loops == result.occurrences == ()
    assert repr(model.__dict__) == before


def test_orthogonal_proof_cancellation_keeps_the_primary_exception():
    error = RuntimeError('cancel orthogonal proof')
    def cancel(phase):
        raise error
    proof = _PatchProof(CylinderPatchPolicy(), cancel)
    points = ((0, 0), (2, 0), (2, 2), (1, 2), (1, 1), (0, 1))
    # Build the interval fixture separately so cancellation targets the proof.
    rows = _exact_rows(points, _PatchProof(CylinderPatchPolicy(), None))
    geometry = SimpleNamespace(radius=Fraction(1), height=Fraction(1),
        surface=SimpleNamespace(sweep_angle=1.), parameter=Fraction(1, 10**6), tau=Fraction(1, 10**7))
    with pytest.raises(RuntimeError) as caught:
        certify_loop(rows, tuple(range(len(rows))), 0, geometry, proof)
    assert caught.value is error


_MULTI_OUTER = ((0, 0), (12, 0), (12, 12), (0, 12),
                (0, 8), (2, 8), (2, 4), (0, 4))
_MULTI_HOLES = (
    ((3, 2), (3, 3), (4, 3), (4, 2)),
    ((7, 7), (7, 9), (9, 9), (9, 7)),
)


def _material_fixture(holes=_MULTI_HOLES, *, reverse=False):
    proof = _PatchProof(CylinderPatchPolicy(), None)
    geometry = SimpleNamespace(
        radius=Fraction(1), height=Fraction(1),
        surface=SimpleNamespace(sweep_angle=1.),
        parameter=Fraction(1, 10**9), tau=Fraction(1, 10**10))
    loops = []
    for index, raw in enumerate((_MULTI_OUTER, *holes)):
        points = tuple((Fraction(x, 12), Fraction(y, 12)) for x, y in raw)
        if reverse:
            points = tuple(reversed(points))
        rows = _exact_rows(points, proof)
        loops.append(certify_loop(rows, tuple(range(len(rows))), index, geometry, proof))
    return tuple(loops), geometry, proof


@pytest.mark.parametrize('reverse', (False, True))
@pytest.mark.parametrize('reverse_hole_order', (False, True))
def test_disjoint_holes_share_winding_and_have_distinct_material_witnesses(
        reverse, reverse_hole_order):
    holes = tuple(reversed(_MULTI_HOLES)) if reverse_hole_order else _MULTI_HOLES
    loops, geometry, proof = _material_fixture(holes, reverse=reverse)
    before_rows = tuple(loop['rows'] for loop in loops)
    result = certify_material(loops, geometry, proof)
    assert result is loops
    assert loops[1]['winding'] == loops[2]['winding'] == -loops[0]['winding']
    assert tuple(loop['rows'] for loop in loops) == before_rows
    x, y = loops[0]['seed']
    assert 0 < x < 1 and 0 < y < 1
    assert not (x < Fraction(2, 12) and Fraction(4, 12) < y < Fraction(8, 12))
    for loop, raw in zip(loops[1:], holes):
        xmin, xmax = min(p[0] for p in raw), max(p[0] for p in raw)
        ymin, ymax = min(p[1] for p in raw), max(p[1] for p in raw)
        u, v = loop['seed']
        assert Fraction(xmin, 12) < u < Fraction(xmax, 12)
        assert Fraction(ymin, 12) < v < Fraction(ymax, 12)
        assert not (Fraction(xmin, 12) < x < Fraction(xmax, 12)
                    and Fraction(ymin, 12) < y < Fraction(ymax, 12))
    repeated, other_geometry, other_proof = _material_fixture(holes, reverse=reverse)
    certify_material(repeated, other_geometry, other_proof)
    assert tuple(loop['seed'] for loop in repeated) == tuple(loop['seed'] for loop in loops)


@pytest.mark.parametrize('holes,diagnostic', (
    ((tuple(reversed(_MULTI_HOLES[0])), _MULTI_HOLES[1]), 'patch_hole_winding'),
    ((((3, 2), (3, 7), (8, 7), (8, 2)),
      ((4, 3), (4, 4), (5, 4), (5, 3))), 'patch_nested_holes'),
    ((_MULTI_HOLES[0], ((4, 2), (4, 3), (5, 3), (5, 2))), 'patch_hole_clearance'),
    ((_MULTI_HOLES[0], ((13, 1), (13, 2), (14, 2), (14, 1))), 'patch_hole_outside_material'),
))
def test_invalid_multi_hole_material_is_not_published(holes, diagnostic):
    loops, geometry, proof = _material_fixture(holes)
    before_rows = tuple(loop['rows'] for loop in loops)
    with pytest.raises(_Refusal, match=diagnostic):
        certify_material(loops, geometry, proof)
    assert not any('seed' in loop for loop in loops)
    assert tuple(loop['rows'] for loop in loops) == before_rows
