"""Exact cubic source-minus-literal-FE bounds, including whole-cell owner gates."""
from dataclasses import replace
from fractions import Fraction as Q
from itertools import product
import pytest

from anygeometry import GeometryError, GeometryModel, CurvilinearMaterialCell
from anygeometry import query_material_surface_regions, query_prepared_material_boundary_chords
from anygeometry import validate_prepared_material_curvilinear_cells, validate_prepared_material_curvilinear_cells_binding
from anygeometry.nurbs import NurbsSurface, NurbsCurve, CurveOnSurface
from anygeometry.polynomial_cell_bounds import coefficients, triangle_residual_bound, quad_residual_bound, _evaluate
from anygeometry.regular_source_reference import SourceNode, SourceSide
from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle, _polynomial_fast_path
from anygeometry.serialization import to_dict


def cubic(active=(0., 1.), shifted=False):
    low, high = (2., 6.) if shifted else (0., 1.)
    urange = tuple(low+(high-low)*x for x in active)
    return NurbsSurface(3, 1, (low,)*4+(high,)*4, (1., 1., 5., 5.),
        tuple(((float(i), 0., float(i == 3)), (float(i), 2., float(i == 3))) for i in range(4)),
        u_range=urange, v_range=(1., 5.))


def exact(support, point):
    return tuple(support.exact_point(*point))


def node(point):
    return SourceNode(('exact', *point), tuple((x, x) for x in point))


def test_cubic_coefficients_are_exact_original_control_conversion_with_active_range_composition():
    support = replace(cubic((.25, .75), shifted=True), v_range=(2., 4.))
    table = coefficients(support)
    assert len(table[0]) == 4
    assert (table[2][0][0], table[2][1][0], table[2][2][0], table[2][3][0]) == (Q(1, 64), Q(3, 32), Q(3, 16), Q(1, 8))
    assert (table[0][0][0], table[0][1][0], table[1][0][0], table[1][0][1]) == (Q(3, 4), Q(3, 2), Q(1, 2), Q(1))
    for u, v in product((Q(0), Q(1, 3), Q(1)), repeat=2):
        value = _evaluate(table, (u, u), (v, v))
        assert all(a == b == actual for (a, b), actual in zip(value, exact(support, (u, v))))


def test_mixed_total_cubic_is_supported_but_every_higher_total_term_refuses():
    knots = (0., 0., 0., 1., 1., 1.)
    support = NurbsSurface(2, 2, knots, knots,
        tuple(tuple((float(i), float(j), float((Q(j, 2) if i == 2 else 0)+(Q(i, 2) if j == 2 else 0)))
                    for j in range(3)) for i in range(3)))
    table = coefficients(support)
    assert table[2][2][1] == table[2][1][2] == 1 and table[2][2][2] == 0
    higher = replace(cubic(), controls=tuple(((float(i), 0., 0.), (float(i), 2., float(i == 3))) for i in range(4)))
    assert coefficients(higher) is None  # u^3*v is TOTAL4, not a cubic residual
    weighted = replace(cubic(), weights=((1., 1.), (2., 2.), (1., 1.), (1., 1.)))
    assert coefficients(weighted) is None
    multispan = NurbsSurface(3, 1, (0.,)*4+(.5,)+(1.,)*4, (0., 0., 1., 1.),
        tuple(((float(i), 0., 0.), (float(i), 1., 0.)) for i in range(5)))
    assert coefficients(multispan) is None


@pytest.mark.parametrize('quad', (False, True))
def test_cubic_triangle_and_quad_control_bounds_cover_independent_exact_residuals(quad):
    support = cubic()
    uv = ((Q(0), Q(0)), (Q(1), Q(0)), (Q(3, 4), Q(1)), (Q(0), Q(3, 4))) if quad else (
        (Q(0), Q(0)), (Q(1), Q(0)), (Q(1, 4), Q(3, 4)))
    xyz = tuple(tuple(x+(Q(1, 1000) if axis == 2 and i == 0 else 0) for axis, x in enumerate(exact(support, p)))
                for i, p in enumerate(uv))
    bound = (quad_residual_bound if quad else triangle_residual_bound)(support, uv, xyz, orientation=1)
    assert bound is not None
    for i in range(7):
        for j in range(7 if quad else 7-i):
            r, s = Q(i, 6), Q(j, 6)
            weights = ((1-r)*(1-s), r*(1-s), r*s, (1-r)*s) if quad else (1-r-s, r, s)
            point = tuple(sum(w*p[k] for w, p in zip(weights, uv)) for k in range(2))
            target = tuple(sum(w*p[k] for w, p in zip(weights, xyz)) for k in range(3))
            assert sum((a-b)**2 for a, b in zip(exact(support, point), target)) <= bound**2


@pytest.mark.parametrize('quad', (False, True))
def test_cubic_interval_corner_controls_cover_every_true_corner_choice(quad):
    support = cubic((.25, .75))
    nominal = ((Q(1, 4), Q(1, 4)), (Q(3, 4), Q(1, 4)), (Q(3, 4), Q(3, 4)), (Q(1, 4), Q(3, 4)))
    nominal = nominal if quad else nominal[:3]
    eps = Q(1, 1024)
    boxes = tuple(tuple((x-eps, x+eps) for x in point) for point in nominal)
    xyz = tuple(exact(support, p) for p in nominal)  # literal targets NEVER move with interval choices
    bound = (quad_residual_bound if quad else triangle_residual_bound)(support, boxes, xyz, orientation=1)
    assert bound is not None
    for flags in product((0, 1), repeat=2*len(boxes)):
        chosen = tuple(tuple(interval[flags[2*i+k]] for k, interval in enumerate(point)) for i, point in enumerate(boxes))
        for r, s in ((Q(1, 4), Q(1, 4)), (Q(1, 2), Q(1, 4))):
            weights = ((1-r)*(1-s), r*(1-s), r*s, (1-r)*s) if quad else (1-r-s, r, s)
            point = tuple(sum(w*p[k] for w, p in zip(weights, chosen)) for k in range(2))
            target = tuple(sum(w*p[k] for w, p in zip(weights, xyz)) for k in range(3))
            assert sum((a-b)**2 for a, b in zip(exact(support, point), target)) <= bound**2


def test_cubic_bounds_preserve_inside_orientation_and_degree_two_method_semantics():
    from test_source_cell_certification import cropped_support
    support = cubic()
    uv = ((Q(0), Q(0)), (Q(1), Q(0)), (Q(1), Q(1)), (Q(0), Q(1)))
    xyz = tuple(exact(support, point) for point in uv)
    assert quad_residual_bound(support, uv[::-1], xyz[::-1], orientation=1) is None
    assert quad_residual_bound(support, uv[::-1], xyz[::-1], orientation=-1) is not None
    assert triangle_residual_bound(support, (uv[0], uv[1], (Q(-1, 100), Q(1))), xyz[:3], orientation=1) is None
    assert quad_residual_bound(support, uv, xyz, method='hierarchical') is None  # no quadratic formula on cubics
    quadratic = cropped_support(.25, .75)
    targets = tuple(exact(quadratic, p) for p in uv)
    assert len(coefficients(quadratic)[0]) == 3
    assert quad_residual_bound(quadratic, uv, targets) == min(
        quad_residual_bound(quadratic, uv, targets, method='bernstein'),
        quad_residual_bound(quadratic, uv, targets, method='hierarchical'))


def test_actual_cubic_fan_dispatch_proves_fine_cells_and_keeps_coarse_general_refusal(monkeypatch):
    from anygeometry import curvilinear_reference_maps as maps
    support = cubic((31/64, 33/64))
    uv = ((Q(0), Q(0)), (Q(1), Q(0)), (Q(1, 2), Q(1, 2)))
    xyz = tuple(exact(support, p) for p in uv)
    side = SourceSide(node(uv[0]), node(uv[1]))
    real = maps._source_box
    calls = []
    def observed(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(maps, '_source_box', observed)
    error, boxes = certify_fan_subtriangle(support, side, uv[2], xyz[:2], xyz[2], orientation=1, surface_error=Q(25, 10000))
    assert boxes == 1 and 0 < error <= Q(25, 10000) and not calls
    coarse = cubic()
    targets = tuple(exact(coarse, p) for p in uv)
    with pytest.raises(GeometryError):
        certify_fan_subtriangle(coarse, side, uv[2], targets[:2], targets[2], orientation=1, surface_error=Q(25, 10000))
    assert calls


def test_cubic_fast_path_still_excludes_actual_curved_source_sides():
    from anygeometry import Plane
    from anygeometry.support_curves import PlaneNurbsCurve
    support = cubic()
    curve = PlaneNurbsCurve.from_plane(Plane((0., .5, 0.), (1., 0., 0.), (0., 0., 1.)), support)
    first, last, centre = (Q(0), Q(1, 4)), (Q(1), Q(1, 4)), (Q(1, 2), Q(1, 2))
    side = SourceSide(node(first), node(last), curve=curve)
    xyz = tuple(exact(support, p) for p in (first, last, centre))
    assert _polynomial_fast_path(support, side, centre, xyz[:2], xyz[2], 1, None, None) is None


def test_cubic_proof_honours_work_and_entry_cancellation():
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    support = cubic((31/64, 33/64))
    uv = ((Q(0), Q(0)), (Q(1), Q(0)), (Q(1, 2), Q(1, 2)))
    xyz = tuple(exact(support, point) for point in uv)
    token = CURRENT_WORK.set(Work(0, 0))
    try:
        with pytest.raises(GeometryError): triangle_residual_bound(support, uv, xyz, orientation=1)
    finally:
        CURRENT_WORK.reset(token)
    side = SourceSide(node(uv[0]), node(uv[1]))
    with pytest.raises(GeometryError, match='cancelled'):
        certify_fan_subtriangle(support, side, uv[2], xyz[:2], xyz[2], orientation=1,
            surface_error=Q(25, 10000), cancellation_check=lambda: True)


def _owner(support):
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    model = GeometryModel()
    xyz = tuple(tuple(float(x) for x in exact(support, p)) for p in uv)
    vertices = model.add_points(xyz)
    edges = [model.add_curve(vertices[i], vertices[(i+1)%4], CurveOnSurface(support,
        NurbsCurve(1, (0., 0., 1., 1.), (p, uv[(i+1)%4])))) for i, p in enumerate(uv)]
    face = model.add_face(edges, surface=support)
    regions = query_material_surface_regions(model, [face])
    return model, face, edges, regions


@pytest.mark.parametrize('quad', (False, True))
def test_genuine_public_cubic_whole_cells_keep_all_source_gates_and_literal_targets(monkeypatch, quad):
    from anygeometry import _cubic_cell_bounds as cubic_bounds
    support = cubic((31/64, 33/64))
    model, face, edges, regions = _owner(support)
    stations = (0., .5, 1.) if quad else (0., 1.)
    intervals = tuple((model.handle('edge', edge), a, b) for edge in edges for a, b in zip(stations, stations[1:]))
    chords = query_prepared_material_boundary_chords(model, regions, intervals, chord_error=.01)
    handle = model.handle('face', face)
    uv = tuple((float(i)/2, float(j)/2) for j in range(3) for i in range(3)) if quad else ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    xyz = tuple(tuple(float(x) for x in exact(support, point)) for point in uv)
    rows = (((0, 1, 4, 3), ((0, 0, True), (3, 7, True))),
            ((1, 2, 5, 4), ((0, 1, True), (1, 2, True))),
            ((3, 4, 7, 6), ((2, 5, True), (3, 6, True))),
            ((4, 5, 8, 7), ((1, 3, True), (2, 4, True)))) if quad else (
            ((0, 1, 2), ((0, 0, True), (1, 1, True))),
            ((0, 2, 3), ((1, 2, True), (2, 3, True))))
    cells = tuple(CurvilinearMaterialCell(handle, ids, tuple(uv[i] for i in ids), tuple(xyz[i] for i in ids), boundary)
                  for ids, boundary in rows)
    calls, original = [], cubic_bounds.residual_bound
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(cubic_bounds, 'residual_bound', counted)
    before = to_dict(model)
    receipt = validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.0025)
    validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)
    assert calls and receipt.cells == cells and all(row[-1] <= .0025 for row in receipt.source_map_bounds)
    original_face = model.faces[face]
    def mutate(phase):
        model._faces[face] = replace(original_face, metadata={'callback': 'changed'})
    try:
        with pytest.raises(GeometryError):
            validate_prepared_material_curvilinear_cells(model, regions, chords, cells,
                surface_error=.0025, cancellation_check=mutate)
    finally:
        model._faces[face] = original_face
    assert to_dict(model) == before


def test_public_folded_cubic_support_still_refuses_separate_owner_immersion_gate():
    support = NurbsSurface(3, 1, (0.,)*4+(1.,)*4, (0., 0., 1., 1.),
        tuple(((x, 0., 0.), (x, 1., 0.)) for x in (0., -1/6, -1/3, .5)))
    assert coefficients(support) is not None  # residual algebra itself is not immersion authority
    with pytest.raises(GeometryError):
        model, face, edges, regions = _owner(support)
        query_prepared_material_boundary_chords(model, regions,
            tuple((model.handle('edge', edge), 0., 1.) for edge in edges), chord_error=1.)


def captured_principal_support():
    """Literal face6 parameterization INPUT from the saved actual principal.

    root-run27-principal-measure-operands-20261025-003810/principal-cut-model.json;
    only its nurbs record is reconstructed, never the mesh/model/history.
    """
    import json
    from pathlib import Path
    from anygeometry.nurbs import nurbs_from_record
    return nurbs_from_record(json.loads((Path(__file__).parent/'data'/'principal_cubic_support.json').read_text()))


@pytest.mark.parametrize('active', ((0., .25), (.25, .5), (.5, .25)))
@pytest.mark.parametrize('quad', (False, True))
def test_actual_principal_record_cells_within_one_original_span_certify(active, quad):
    support = replace(captured_principal_support(), u_range=active)
    a, b, h = Q(1, 2), Q(1, 4), Q(1, 1024)
    uv = ((a, b), (a+h, b), (a+h, b+h), (a, b+h)) if quad else ((a, b), (a+h, b), (a, b+h))
    xyz = tuple(tuple(Q(float(x)) for x in exact(support, point)) for point in uv)
    bound = (quad_residual_bound if quad else triangle_residual_bound)(support, uv, xyz, orientation=1)
    assert bound is not None and bound <= Q(25, 10000)
    # New service is a whole-cell proof, not a vertex interpolation claim.
    weights = (Q(1, 4),)*4 if quad else (Q(1, 3),)*3
    point = tuple(sum(w*p[k] for w, p in zip(weights, uv)) for k in range(2))
    target = tuple(sum(w*p[k] for w, p in zip(weights, xyz)) for k in range(3))
    assert sum((a-b)**2 for a, b in zip(exact(support, point), target)) <= bound**2


def test_actual_principal_knot_crossing_hulls_fallback_and_closed_span_endpoint_is_sound():
    support = replace(captured_principal_support(), u_range=(0., 1.))
    knot, h = Q(support.knots_u[4]), Q(1, 1024)
    uv = ((knot-h, Q(1, 4)), (knot+h, Q(1, 4)), (knot+h, Q(1, 2)), (knot-h, Q(1, 2)))
    xyz = tuple(exact(support, p) for p in uv)
    assert quad_residual_bound(support, uv, xyz, orientation=1) is None
    assert triangle_residual_bound(support, uv[:3], xyz[:3], orientation=1) is None
    left = ((knot-h, Q(1, 4)), (knot, Q(1, 4)), (knot, Q(1, 2)), (knot-h, Q(1, 2)))
    targets = tuple(exact(support, p) for p in left)
    bound = quad_residual_bound(support, left, targets, orientation=1)
    assert bound is not None  # C0 position matches at the shared original knot
    table = coefficients(support, tuple(tuple((x, x) for x in p) for p in left))
    value = _evaluate(table, (knot, knot), (Q(1, 4), Q(1, 4)))
    assert all(a == b == actual for (a, b), actual in zip(value, targets[1]))
    assert all(sum((a-b)**2 for a, b in zip(exact(support, p), t)) <= bound**2 for p, t in zip(left, targets))


def test_actual_principal_uncertain_true_corners_keep_fixed_literal_targets():
    support = captured_principal_support()
    a, b, h, eps = Q(1, 2), Q(1, 4), Q(1, 1024), Q(1, 2**40)
    uv = ((a, b), (a+h, b), (a+h, b+h), (a, b+h))
    boxes = tuple(tuple((x-eps, x+eps) for x in point) for point in uv)
    xyz = tuple(tuple(Q(float(x)) for x in exact(support, point)) for point in uv)
    bound = quad_residual_bound(support, boxes, xyz, orientation=1)
    assert bound is not None and bound <= Q(25, 10000)
    for flags in ((0,)*8, (1,)*8, (0, 1)*4):
        chosen = tuple(tuple(interval[flags[2*i+k]] for k, interval in enumerate(point)) for i, point in enumerate(boxes))
        point = tuple(sum(p[k] for p in chosen)/4 for k in range(2))
        target = tuple(sum(p[k] for p in xyz)/4 for k in range(3))
        assert sum((a-b)**2 for a, b in zip(exact(support, point), target)) <= bound**2
