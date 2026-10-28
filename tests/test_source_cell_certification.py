"""Source-cell certification: proof-local side-enclosure reuse and the Euclidean fan bound (UNRUN).

Real owner objects only: a true SourceSide of the closed paraboloid z=.5 section, and the real
``square_cells`` owner cells. Counters delegate to the actual ``SourceSide`` box/derivative functions.
"""
from fractions import Fraction as Q

import pytest

from anygeometry import (GeometryError, validate_prepared_material_curvilinear_cells,
    validate_prepared_material_curvilinear_cells_binding)
from anygeometry.curvilinear_reference_maps import _norm_upper
from anygeometry.regular_source_reference import NurbsReference, SourceSide
from anygeometry.source_cell_complex import _SideEnclosures
from test_prepared_material_curvilinear import square_cells
from test_regular_source_reference import closed_quarters


@pytest.mark.parametrize('values', ((Q(3), Q(4)), (Q(1, 3),)*3, (Q(0),)*3, (Q(5, 7), Q(0), Q(2, 11)), (Q(10**9), Q(1))))
def test_euclidean_norm_bound_is_sound_and_never_above_the_coordinate_sum(values):
    bound = _norm_upper(values)
    exact_square = sum(v*v for v in values)
    assert bound*bound >= exact_square                              # never below the true norm
    assert bound <= sum(values) + Q(1, 2**62)                       # never above the old coordinate sum
    assert (bound - Q(1, 2**62))**2 <= exact_square + Q(1, 2**20) * (1 + exact_square)   # and nearly tight


def test_zero_vector_norm_bound_is_exactly_zero():
    assert _norm_upper((Q(0), Q(0), Q(0))) == 0 and _norm_upper(()) == 0


def cropped_paraboloid_triangles(low, high, chord_error):
    """Real public regions/chords for the paraboloid cropped to [low, high]^2 in its chart, cut as two
    genuine linear triangles whose corners are exact surface points."""
    from dataclasses import replace
    from anygeometry import (GeometryModel, CurvilinearMaterialCell, query_material_surface_regions,
        query_prepared_material_boundary_chords)
    from anygeometry.nurbs import NurbsCurve, CurveOnSurface
    from test_section_boundary_sampling import paraboloid
    support = replace(paraboloid(), u_range=(low, high), v_range=(low, high))
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    xyz = tuple(tuple(float(x) for x in support.exact_point(*p)) for p in uv)
    model = GeometryModel()
    vertices = model.add_points(xyz)
    edges = [model.add_curve(vertices[i], vertices[(i + 1) % 4], CurveOnSurface(support,
        NurbsCurve(1, (0., 0., 1., 1.), (p, uv[(i + 1) % 4])))) for i, p in enumerate(uv)]
    face = model.add_face(edges, surface=support)
    regions = query_material_surface_regions(model, [face])
    chords = query_prepared_material_boundary_chords(model, regions,
        tuple((model.handle('edge', edge), 0., 1.) for edge in edges), chord_error=chord_error)
    handle = model.handle('face', face)
    cells = (CurvilinearMaterialCell(handle, (0, 1, 2), uv[:3], xyz[:3], ((0, 0, True), (1, 1, True))),
             CurvilinearMaterialCell(handle, (0, 2, 3), (uv[0], uv[2], uv[3]), (xyz[0], xyz[2], xyz[3]),
                                     ((1, 2, True), (2, 3, True))))
    return model, regions, chords, cells


def test_tiny_curved_patch_with_linear_triangles_certifies_within_the_unchanged_public_error():
    model, regions, chords, cells = cropped_paraboloid_triangles(63 / 128, 65 / 128, .01)
    receipt = validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.0025)
    validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)
    assert receipt.cells == cells and all(0 <= row[-1] <= .0025 for row in receipt.source_map_bounds)


def test_coarse_curved_patch_with_linear_triangles_is_refused_by_the_true_source_minus_fe_error():
    # chord_error 1.0 only lets the (coarse) boundary chords be issued; the interior source-vs-FE
    # centroid error (~.25 m) is what must refuse at the unchanged final surface_error .0025
    model, regions, chords, cells = cropped_paraboloid_triangles(.25, .75, 1.0)
    with pytest.raises(GeometryError):
        validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.0025)


def real_side():
    model, face, vertices, edges = closed_quarters()
    reference = NurbsReference(model, model.faces[face])
    interior = reference.node(edges[0], Q(1, 8))
    seam = reference.node(edges[0], Q(0), vertices[0])
    side = reference.side(edges[0], (Q(1, 8), Q(0)), interior, seam)
    assert isinstance(side, SourceSide)
    return side


def test_side_enclosures_reuse_identical_exact_intervals_and_return_the_sides_own_values(monkeypatch):
    side = real_side()
    calls = {'box': [], 'derivative': []}
    real_box, real_derivative = SourceSide.box, SourceSide.derivative

    def counting_box(self, lo, hi):
        calls['box'].append((lo, hi))
        return real_box(self, lo, hi)

    def counting_derivative(self, lo, hi):
        calls['derivative'].append((lo, hi))
        return real_derivative(self, lo, hi)

    monkeypatch.setattr(SourceSide, 'box', counting_box)
    monkeypatch.setattr(SourceSide, 'derivative', counting_derivative)
    memo = _SideEnclosures(side)
    intervals = [(Q(0), Q(1)), (Q(0), Q(1)), (Q(1, 2), Q(1)), (Q(0), Q(1)), (Q(1, 2), Q(1)), (Q(1, 4), Q(1, 2))]
    first = [memo.box(lo, hi) for lo, hi in intervals]
    assert calls['box'] == [(Q(0), Q(1)), (Q(1, 2), Q(1)), (Q(1, 4), Q(1, 2))]      # one actual call per interval
    for (lo, hi), value in zip(intervals, first):
        assert value == real_box(side, lo, hi)                                       # the side's own enclosure
    again = [memo.derivative(lo, hi) for lo, hi in intervals]
    assert calls['derivative'] == [(Q(0), Q(1)), (Q(1, 2), Q(1)), (Q(1, 4), Q(1, 2))]
    for (lo, hi), value in zip(intervals, again):
        assert value == real_derivative(side, lo, hi)


def skeleton_operands():
    """Two far-apart unit squares, each cut into two triangles by one internal diagonal, with real
    SourceNode/SourceSide operands: 2 internal edges x 8 boundary sides, many pairs broad-phase disjoint."""
    from anygeometry import CurvilinearMaterialCell
    from anygeometry.regular_source_reference import SourceNode
    points = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1), 4: (10, 10), 5: (11, 10), 6: (11, 11), 7: (10, 11)}
    nodes = {k: SourceNode(('node', k), tuple((Q(x), Q(x)) for x in p)) for k, p in points.items()}
    triangles = ((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7))
    cells = tuple(CurvilinearMaterialCell(None, ids, (), (), ()) for ids in triangles)
    internal = {(0, 2), (4, 6)}
    boundary_sides, sides = {}, {}
    for i, ids in enumerate(triangles):
        for side, (a, b) in enumerate(zip(ids, ids[1:] + ids[:1])):
            if tuple(sorted((a, b))) not in internal:
                boundary_sides[(i, side)] = len(boundary_sides)
                sides[(i, side)] = SourceSide(nodes[a], nodes[b])
    assert len(boundary_sides) == 8
    return cells, boundary_sides, sides, nodes


def test_every_internal_boundary_examination_is_checked_and_charged_even_when_disjoint():
    import anygeometry.source_cell_complex as module
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    cells, boundary_sides, sides, nodes = skeleton_operands()
    checks = []
    work = Work(10**6, 40)
    token = CURRENT_WORK.set(work)
    try:
        module._skeleton(cells, boundary_sides, sides, nodes, lambda: checks.append(1))
    finally:
        CURRENT_WORK.reset(token)
    assert len(checks) >= 2 * 8 and work.boxes >= 2 * 8              # one logical step per pair, cache hits too


def test_exhausted_work_budget_refuses_in_the_broad_phase_and_cancellation_refuses():
    import anygeometry.source_cell_complex as module
    from anygeometry.plane_nurbs import CURRENT_WORK, PlaneNurbsNotCertified
    from anygeometry.section_audit import Work
    cells, boundary_sides, sides, nodes = skeleton_operands()
    token = CURRENT_WORK.set(Work(12, 40))        # fewer than the 16 broad-phase pairs alone
    try:
        with pytest.raises((GeometryError, PlaneNurbsNotCertified)):
            module._skeleton(cells, boundary_sides, sides, nodes, lambda: None)
    finally:
        CURRENT_WORK.reset(token)
    seen = []

    def cancel():
        seen.append(1)
        if len(seen) > 5:
            raise GeometryError('cancelled')

    with pytest.raises(GeometryError, match='cancelled'):
        module._skeleton(cells, boundary_sides, sides, nodes, cancel)
    assert len(seen) == 6


@pytest.mark.parametrize('local', (Q(1, 3), Q(1, 2), Q(7, 10)))
def test_zero_width_non_root_slice_is_a_tiny_enclosure_that_contains_the_actual_section_point(local):
    """Before: a point slice fell back to the whole transverse box (~.004 wide at the first fan).
    Now it is an exact 2**-100 bracket of the unique graph zero. The actual point is the section's chart at
    the matching edge parameter (supplemental containment only; the width is the discriminating check)."""
    model, face, vertices, edges = closed_quarters()
    reference = NurbsReference(model, model.faces[face])
    interior = reference.node(edges[0], Q(1, 8))
    seam = reference.node(edges[0], Q(0), vertices[0])
    side = reference.side(edges[0], (Q(1, 8), Q(0)), interior, seam)
    box = side.box(local, local)
    assert all(hi - lo < Q(1, 2**40) for lo, hi in box)                 # not the whole transverse box
    edge_parameter = float(Q(1, 8) * (1 - local))
    chart = model.edges[edges[0]].curve.chart(edge_parameter)
    for (lo, hi), value in zip(box, chart):
        assert lo - Q(1, 10**9) <= Q(float(value)) <= hi + Q(1, 10**9)


def test_point_slice_enclosures_shrink_with_the_interval_and_endpoints_are_unchanged():
    model, face, vertices, edges = closed_quarters()
    reference = NurbsReference(model, model.faces[face])
    interior = reference.node(edges[0], Q(1, 8))
    seam = reference.node(edges[0], Q(0), vertices[0])
    side = reference.side(edges[0], (Q(1, 8), Q(0)), interior, seam)
    wide = side.box(Q(1, 4), Q(3, 4))
    point = side.box(Q(1, 2), Q(1, 2))
    assert all(p[0] >= w[0] and p[1] <= w[1] for p, w in zip(point, wide))   # the point lies in the wider piece
    assert side.box(Q(0), Q(0)) == side.start.box and side.box(Q(1), Q(1)) == side.end.box


def test_point_slice_work_is_charged():
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    side = real_side()
    work = Work(10**6, 40)
    token = CURRENT_WORK.set(work)
    try:
        side.box(Q(1, 2), Q(1, 2))
    finally:
        CURRENT_WORK.reset(token)
    assert work.boxes >= 1


def test_fan_side_memo_runs_the_exact_ranges_once_per_interval_and_serves_box_and_derivative(monkeypatch):
    from anygeometry.curvilinear_reference_maps import _SideMemo
    side = real_side()
    calls = []
    real = SourceSide._ranges
    monkeypatch.setattr(SourceSide, '_ranges', lambda self, lo, hi: calls.append((lo, hi)) or real(self, lo, hi))
    expected_box, expected_derivative = side.box(Q(1, 4), Q(1, 2)), side.derivative(Q(1, 4), Q(1, 2))
    assert len(calls) == 2                                           # the side itself computes ranges twice
    calls.clear()
    memo = _SideMemo(side)
    for _ in range(3):
        assert memo.box(Q(1, 4), Q(1, 2)) == expected_box
        assert memo.derivative(Q(1, 4), Q(1, 2)) == expected_derivative
    assert calls == [(Q(1, 4), Q(1, 2))]                             # once for both values and all repeats
    memo.box(Q(1, 2), Q(1, 2))
    memo.derivative(Q(1, 2), Q(1, 2))                                # the midpoint query shares its ranges too
    assert calls == [(Q(1, 4), Q(1, 2)), (Q(1, 2), Q(1, 2))]
    assert memo.box(Q(0), Q(0)) == side.start.box and memo.box(Q(1), Q(1)) == side.end.box
    assert len(calls) == 2                                           # exact endpoint cases never compute ranges
    assert memo.curve is side.curve and memo.start is side.start and memo.end is side.end


def curved_fan_fixture():
    """A small fan on the real closed-section side of the paraboloid face: exact endpoint targets, the
    centre offset on the positive side of the chord (so orientation +1 is strict by construction)."""
    model, face, vertices, edges = closed_quarters()
    reference = NurbsReference(model, model.faces[face])
    interior = reference.node(edges[0], Q(1, 32))
    seam = reference.node(edges[0], Q(0), vertices[0])
    side = reference.side(edges[0], (Q(1, 32), Q(0)), interior, seam)
    support = model.faces[face].surface
    mid = lambda node: tuple((lo + hi) / 2 for lo, hi in node.box)
    first, last = mid(interior), mid(seam)
    chord = (last[0] - first[0], last[1] - first[1])
    centre = tuple((first[k] + last[k]) / 2 + Q(1, 2) * n for k, n in enumerate((-chord[1], chord[0])))
    return support, side, centre, (exact(support, first), exact(support, last)), exact(support, centre)


def _certify_outcome(fixture, check=None, side=None, surface_error=Q(25, 10000)):
    from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle
    support, own_side, centre, xyz_side, centre_xyz = fixture
    try:
        return ('ok', certify_fan_subtriangle(support, side or own_side, centre, xyz_side, centre_xyz,
                                              orientation=1, surface_error=surface_error,
                                              cancellation_check=check))
    except GeometryError as error:
        return ('error', str(error))


def test_certifier_result_is_unchanged_by_the_memo_and_repeated_enclosures_are_reduced(monkeypatch):
    """Limitation: if the fixture is refused, this shows identical refusal on both paths, not parity of an
    accepted bound; neither outcome is a scientific acceptance."""
    import anygeometry.curvilinear_reference_maps as module
    fixture = curved_fan_fixture()
    calls = []
    real = SourceSide._ranges
    monkeypatch.setattr(SourceSide, '_ranges', lambda self, lo, hi: calls.append((lo, hi)) or real(self, lo, hi))
    memoized = _certify_outcome(fixture)
    memo_calls = len(calls)
    calls.clear()
    monkeypatch.setattr(module, '_SideMemo', lambda side: side)         # test-only identity wrapper
    plain = _certify_outcome(fixture)
    plain_calls = len(calls)
    assert memoized == plain                                           # same bound and pieces, or same refusal
    assert 0 < memo_calls < plain_calls                                # fewer exact enclosure evaluations
    assert len(set(calls)) <= memo_calls                               # each interval at most once when memoized


def test_memo_active_general_loop_honours_cancellation_and_propagates_callback_exceptions():
    fixture = curved_fan_fixture()
    seen = []

    def cancel_in_loop():
        seen.append(1)
        return len(seen) >= 2                       # call 1 is the entry check, call 2 the first loop check

    assert _certify_outcome(fixture, cancel_in_loop) == ('error', 'source-curved reference map proof cancelled')

    class Boom(BaseException):
        pass

    marker, count = Boom(), []

    def failing():
        count.append(1)
        if len(count) >= 2:
            raise marker
        return False

    with pytest.raises(Boom) as raised:
        _certify_outcome(fixture, failing)
    assert raised.value is marker


def test_only_the_exact_sourceside_is_memoized_subclasses_delegate_every_query(monkeypatch):
    """Synthetic eligibility test (not scientific coverage): the general loop necessarily queries ``box`` at
    least twice (the top-of-loop query and then the midpoint query or the next iteration), so a boundary that
    refuses its second query MUST surface that refusal when every query is delegated."""
    import anygeometry.curvilinear_reference_maps as module
    fixture = curved_fan_fixture()
    support, side, centre, xyz_side, centre_xyz = fixture
    log = []

    class Hostile(SourceSide):
        def box(self, lo, hi):
            log.append((lo, hi))
            if len(log) >= 2:
                raise GeometryError('hostile second query')
            return super().box(lo, hi)

    hostile = Hostile(side.start, side.end, side.curve, side.pieces, side.parameters, side.endpoint_transport)
    with pytest.raises(TypeError):
        module._SideMemo(hostile)                                        # not eligible for the memo
    built = []
    real_init = module._SideMemo.__init__
    monkeypatch.setattr(module._SideMemo, '__init__', lambda self, s: built.append(type(s)) or real_init(self, s))
    assert _certify_outcome(fixture, side=hostile) == ('error', 'hostile second query')
    assert built == [] and len(log) == 2                                 # never wrapped; refusal surfaced at once
    _certify_outcome(fixture)
    assert built == [SourceSide]                                         # the exact side is wrapped once


def _recording_memo(monkeypatch, hits):
    """Delegating instrumentation of the REAL memo: records, per ``_ranges`` call, whether it was a reuse."""
    import anygeometry.curvilinear_reference_maps as module
    real = module._SideMemo

    class Recording(real):
        def _ranges(self, lo, hi):
            hits.append((lo, hi) in self._both)
            return super()._ranges(lo, hi)

    monkeypatch.setattr(module, '_SideMemo', Recording)


def test_cancellation_after_memo_reuse_in_a_later_iteration(monkeypatch):
    """A strictly positive but tiny surface_error cannot be met by the real chord/sagitta difference, so the
    first loop box subdivides and a second iteration (a third callback call) exists; by then the memo has
    already served ``derivative`` from the ``box`` computation of the same interval (a reuse hit)."""
    hits, calls = [], []
    _recording_memo(monkeypatch, hits)

    def cancel():
        calls.append(len(hits))
        return any(hits)

    outcome = _certify_outcome(curved_fan_fixture(), cancel, surface_error=Q(1, 10**12))
    assert outcome == ('error', 'source-curved reference map proof cancelled')
    assert True in hits and calls[0] == 0 and calls[-1] > 0 and len(calls) >= 3     # cancelled AFTER reuse


def test_identical_base_exception_after_memo_reuse_in_a_later_iteration(monkeypatch):
    hits = []
    _recording_memo(monkeypatch, hits)

    class Boom(BaseException):
        pass

    marker = Boom()

    def failing():
        if any(hits):
            raise marker
        return False

    with pytest.raises(Boom) as raised:
        _certify_outcome(curved_fan_fixture(), failing, surface_error=Q(1, 10**12))
    assert raised.value is marker and True in hits


def test_fan_side_memo_does_not_cache_failures_and_is_not_shared_between_instances(monkeypatch):
    from anygeometry.curvilinear_reference_maps import _SideMemo
    side = real_side()
    attempts = []
    real = SourceSide._ranges

    def flaky(self, lo, hi):
        attempts.append(1)
        if len(attempts) == 1:
            raise GeometryError('transient proof refusal')
        return real(self, lo, hi)

    monkeypatch.setattr(SourceSide, '_ranges', flaky)
    memo = _SideMemo(side)
    with pytest.raises(GeometryError):
        memo.box(Q(1, 4), Q(1, 2))
    assert memo.box(Q(1, 4), Q(1, 2)) == real(side, Q(1, 4), Q(1, 2))[0]      # retried, not a cached failure
    other = _SideMemo(side)
    other.derivative(Q(1, 4), Q(1, 2))
    assert len(attempts) == 3                                                  # a second instance recomputes


def test_two_memos_never_share_state():
    side = real_side()
    first, second = _SideEnclosures(side), _SideEnclosures(side)
    first.box(Q(0), Q(1))
    assert not second._boxes and not second._derivatives            # proof-local, no global trust cache


@pytest.mark.parametrize('quad', (False, True))
def test_valid_owner_cells_still_certify_and_bind(quad):
    model, regions, chords, cells = square_cells(quad)
    receipt = validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.001)
    validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)
    assert all(0 <= row[-1] <= .001 for row in receipt.source_map_bounds)


def test_cancellation_and_changed_targets_still_refuse():
    model, regions, chords, cells = square_cells(False)
    with pytest.raises(GeometryError, match='cancelled'):
        validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.001,
                                                     cancellation_check=lambda phase: True)
    from dataclasses import replace
    changed = replace(cells[0], xyz=((.01, 0., 0.), *cells[0].xyz[1:]))
    with pytest.raises(GeometryError):
        validate_prepared_material_curvilinear_cells(model, regions, chords, (changed,), surface_error=.1)

def cropped_support(low, high):
    from dataclasses import replace
    from test_section_boundary_sampling import paraboloid
    return replace(paraboloid(), u_range=(low, high), v_range=(low, high))


def exact(support, point):
    return tuple(Q(x) for x in support.exact_point(*point))


def test_admissibility_requires_proven_total_degree_two_uniform_single_span():
    from anygeometry import Plane
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.polynomial_cell_bounds import admissible
    from test_section_boundary_sampling import KNOTS, paraboloid
    assert admissible(paraboloid()) and admissible(cropped_support(.25, .75))
    heights = (1., -1., 1.)
    product = NurbsSurface(2, 2, KNOTS, KNOTS, tuple(tuple((i - 1., j - 1., heights[i]*heights[j])
                                                          for j in range(3)) for i in range(3)))
    assert not admissible(product)                                    # bidegree (2,2) but has u^2 v^2
    weighted = NurbsSurface(2, 2, KNOTS, KNOTS, tuple(tuple((i - 1., j - 1., heights[i] + heights[j])
                                                           for j in range(3)) for i in range(3)),
                            ((1., 1., 1.), (1., 2., 1.), (1., 1., 1.)))
    assert not admissible(weighted)
    assert not admissible(Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))


@pytest.mark.parametrize('low, high, fine', ((63 / 128, 65 / 128, True), (.25, .75, False)))
def test_triangle_polynomial_bound_is_sound_for_the_whole_cell_and_decides_fine_versus_coarse(low, high, fine):
    from anygeometry.polynomial_cell_bounds import triangle_residual_bound
    support = cropped_support(low, high)
    uv = ((Q(0), Q(0)), (Q(1), Q(0)), (Q(1, 2), Q(1, 2)))
    xyz = tuple(exact(support, p) for p in uv)
    bound = triangle_residual_bound(support, uv, xyz)
    assert bound is not None and (bound <= Q(25, 10000)) == fine
    for i in range(9):                                                 # exact samples inside the triangle
        for j in range(9 - i):
            l0, l1, l2 = Q(i, 8), Q(j, 8), Q(8 - i - j, 8)
            point = tuple(l0*a + l1*b + l2*c for a, b, c in zip(*uv))
            target = tuple(l0*a + l1*b + l2*c for a, b, c in zip(*xyz))
            residual = sum((s - t)**2 for s, t in zip(exact(support, point), target))
            assert residual <= bound*bound


@pytest.mark.parametrize('low, high, fine', ((63 / 128, 65 / 128, True), (.25, .75, False)))
def test_quad_polynomial_bound_is_sound_for_the_whole_cell_and_decides_fine_versus_coarse(low, high, fine):
    from anygeometry.polynomial_cell_bounds import quad_residual_bound
    support = cropped_support(low, high)
    uv = ((Q(0), Q(0)), (Q(1), Q(0)), (Q(1), Q(1)), (Q(0), Q(1)))
    xyz = tuple(exact(support, p) for p in uv)
    bound = quad_residual_bound(support, uv, xyz)
    assert bound is not None and (bound <= Q(25, 10000)) == fine
    for i in range(9):
        for j in range(9):
            xi, eta = Q(i, 8), Q(j, 8)
            weights = ((1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta)
            point = tuple(sum(w*p[k] for w, p in zip(weights, uv)) for k in range(2))
            target = tuple(sum(w*p[k] for w, p in zip(weights, xyz)) for k in range(3))
            residual = sum((s - t)**2 for s, t in zip(exact(support, point), target))
            assert residual <= bound*bound


def test_fan_certifier_uses_the_exact_polynomial_route_for_a_fine_cell_without_the_general_proof(monkeypatch):
    from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle
    from anygeometry.regular_source_reference import SourceNode
    support = cropped_support(63 / 128, 65 / 128)
    node = lambda p: SourceNode(('point', *p), tuple((x, x) for x in p))
    uv = ((Q(0), Q(0)), (Q(1), Q(0)), (Q(1, 2), Q(1, 2)))
    xyz = tuple(exact(support, p) for p in uv)
    calls = []
    real = SourceSide.box
    monkeypatch.setattr(SourceSide, 'box', lambda self, lo, hi: calls.append(1) or real(self, lo, hi))
    error, boxes = certify_fan_subtriangle(support, SourceSide(node(uv[0]), node(uv[1])), uv[2],
        (xyz[0], xyz[1]), xyz[2], orientation=1, surface_error=Q(25, 10000))
    assert boxes == 1 and 0 < error <= Q(25, 10000) and calls == []     # the general interval proof never ran


def test_fan_certifier_quad_route_and_nonpolynomial_fallback_helper():
    from anygeometry.curvilinear_reference_maps import _polynomial_fast_path
    from anygeometry.regular_source_reference import SourceNode
    from anygeometry.nurbs import NurbsSurface
    from test_section_boundary_sampling import KNOTS
    support = cropped_support(63 / 128, 65 / 128)
    node = lambda p: SourceNode(('point', *p), tuple((x, x) for x in p))
    uv = ((Q(0), Q(0)), (Q(1), Q(0)), (Q(1), Q(1)), (Q(0), Q(1)))
    xyz = tuple(exact(support, p) for p in uv)
    boxes = tuple(tuple((x, x) for x in p) for p in uv)
    side = SourceSide(node(uv[0]), node(uv[1]))
    bound = _polynomial_fast_path(support, side, (Q(1, 2), Q(1, 2)), (xyz[0], xyz[1]), (Q(0),)*3, 1, xyz, boxes, 0)
    assert bound is not None and 0 < bound <= Q(25, 10000)
    heights = (1., -1., 1.)
    product = NurbsSurface(2, 2, KNOTS, KNOTS, tuple(tuple((i - 1., j - 1., heights[i]*heights[j])
                                                          for j in range(3)) for i in range(3)))
    assert _polynomial_fast_path(product, side, (Q(1, 2), Q(1, 2)), (xyz[0], xyz[1]), (Q(0),)*3, 1, xyz,
                                 boxes, 0) is None
    reversed_orientation = _polynomial_fast_path(support, side, (Q(1, 2), Q(1, 2)), (xyz[0], xyz[1]),
                                                 (Q(0),)*3, -1, xyz, boxes, 0)
    assert reversed_orientation is None                                # strict orientation is still required


EPS = Q(1, 2**20)


def _exact_residual_squared(support, points, targets, weights):
    point = tuple(sum(w*p[k] for w, p in zip(weights, points)) for k in range(2))
    target = tuple(sum(w*t[k] for w, t in zip(weights, targets)) for k in range(3))
    return sum((s - t)**2 for s, t in zip(exact(support, point), target))


@pytest.mark.parametrize('low, high', ((63 / 128, 65 / 128), (.25, .75)))
def test_triangle_with_certified_interval_vertices_encloses_every_choice_of_true_vertices(low, high):
    from itertools import product as choices
    from anygeometry.polynomial_cell_bounds import triangle_residual_bound
    support = cropped_support(low, high)
    centre = (Q(1, 2), Q(1, 2))
    nominal = ((Q(1, 8), Q(1, 8)), (Q(7, 8), Q(1, 8)), centre)
    xyz = tuple(exact(support, p) for p in nominal)
    boxes = tuple(tuple((x, x + EPS) for x in p) for p in nominal[:2]) + (centre,)
    interval_bound = triangle_residual_bound(support, boxes, xyz, orientation=1)
    assert interval_bound is not None
    point_bound = triangle_residual_bound(support, nominal, xyz, orientation=1)
    assert interval_bound >= point_bound                              # the box includes the nominal choice
    # every independent low/high choice of each coordinate of each uncertain vertex (4 x 4 = 16 choices)
    for first, second in choices(choices((0, 1), repeat=2), repeat=2):
        chosen = tuple(tuple(box[k][c] for k, c in enumerate(corner)) for box, corner in
                       zip(boxes[:2], (first, second))) + (centre,)
        for i in range(5):                                             # coarse barycentric grid
            for j in range(5 - i):
                weights = (Q(i, 4), Q(j, 4), Q(4 - i - j, 4))
                assert _exact_residual_squared(support, chosen, xyz, weights) <= interval_bound**2


def test_quad_with_certified_interval_corners_encloses_every_extreme_corner_choice():
    from itertools import product as choices
    from anygeometry.polynomial_cell_bounds import quad_residual_bound
    support = cropped_support(63 / 128, 65 / 128)
    nominal = ((Q(1, 8), Q(1, 8)), (Q(7, 8), Q(1, 8)), (Q(7, 8), Q(7, 8)), (Q(1, 8), Q(7, 8)))
    xyz = tuple(exact(support, p) for p in nominal)
    boxes = tuple(tuple((x, x + EPS) for x in p) for p in nominal)
    bound = quad_residual_bound(support, boxes, xyz, orientation=1)
    assert bound is not None and bound >= quad_residual_bound(support, nominal, xyz, orientation=1)
    # all 2**8 = 256 independent low/high choices of every coordinate of every corner (no correlation)
    for selection in choices((0, 1), repeat=8):
        chosen = tuple(tuple(box[k][selection[2*n + k]] for k in range(2)) for n, box in enumerate(boxes))
        for i in range(3):                                             # small 3x3 reference grid
            for j in range(3):
                xi, eta = Q(i, 2), Q(j, 2)
                weights = ((1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta)
                assert _exact_residual_squared(support, chosen, xyz, weights) <= bound**2


SKEW_QUAD = ((Q(1, 8), Q(1, 8)), (Q(7, 8), Q(1, 8)), (Q(3, 4), Q(7, 8)), (Q(1, 8), Q(7, 8)))     # not a parallelogram


def _exact_quad_residual(support, uv, xyz, x, y):
    weights = ((1-x)*(1-y), x*(1-y), x*y, (1-x)*y)
    point = tuple(sum(w*p[k] for w, p in zip(weights, uv)) for k in range(2))
    target = tuple(sum(w*p[k] for w, p in zip(weights, xyz)) for k in range(3))
    return tuple(s - t for s, t in zip(exact(support, point), target))


@pytest.mark.parametrize('uv', (((Q(0), Q(0)), (Q(1), Q(0)), (Q(1), Q(1)), (Q(0), Q(1))), SKEW_QUAD))
def test_hierarchical_identity_is_exact_for_known_quadratic_and_quartic_cross_residuals(uv):
    """R = B + x(1-x)[a0(1-y)+a1 y] + y(1-y)[b0(1-x)+b1 x] + x(1-x)y(1-y) c, exactly, at rational points;
    the skew corner map composes the quadratic support into a nonzero quartic cross term."""
    from anygeometry.polynomial_cell_bounds import (coefficients, quad_hierarchical_parts, _residual,
                                                    _bilinear, _point)
    support = cropped_support(63 / 128, 65 / 128)
    xyz = tuple(exact(support, p) for p in uv)
    table = coefficients(support)
    nodes = (Q(0), Q(1, 2), Q(1))

    def target(x, y):
        weights = ((1-x)*(1-y), x*(1-y), x*y, (1-x)*y)
        return tuple(sum(w*p[k] for w, p in zip(weights, xyz)) for k in range(3))

    boxes = tuple(tuple(_point(c) for c in p) for p in uv)
    values = [[_residual(table, _bilinear(boxes, x, y), target(x, y)) for y in nodes] for x in nodes]
    low = lambda vec: tuple(t[0] for t in vec)              # singleton intervals: exact values
    c00, c10, c01, c11 = [low(v) for v in quad_hierarchical_parts(values)[0]]       # parts order (0,0),(1,0),(0,1),(1,1)
    corner_values = [c00, c10, c11, c01]                                             # weights order of B
    _c, (_a0, _a1), (_b0, _b1), _cc = quad_hierarchical_parts(values)
    a0, a1, b0, b1, c = low(_a0), low(_a1), low(_b0), low(_b1), low(_cc)
    for x, y in ((Q(1, 3), Q(2, 7)), (Q(1, 2), Q(1, 2)), (Q(5, 8), Q(1, 8)), (Q(0), Q(3, 4)), (Q(1), Q(1))):
        weights = ((1-x)*(1-y), x*(1-y), x*y, (1-x)*y)
        rebuilt = tuple(
            sum(w*corner_values[k][axis] for k, w in enumerate(weights))
            + x*(1-x)*(a0[axis]*(1-y) + a1[axis]*y) + y*(1-y)*(b0[axis]*(1-x) + b1[axis]*x)
            + x*(1-x)*y*(1-y)*c[axis] for axis in range(3))
        assert rebuilt == _exact_quad_residual(support, uv, xyz, x, y)


@pytest.mark.parametrize('uv', (((Q(0), Q(0)), (Q(1), Q(0)), (Q(1), Q(1)), (Q(0), Q(1))), SKEW_QUAD))
def test_quad_bound_is_the_minimum_of_two_sound_bounds_and_covers_exact_residuals(uv):
    from anygeometry.polynomial_cell_bounds import quad_residual_bound
    support = cropped_support(63 / 128, 65 / 128)
    xyz = tuple(exact(support, p) for p in uv)
    both = [quad_residual_bound(support, uv, xyz, method=m) for m in ('bernstein', 'hierarchical')]
    best = quad_residual_bound(support, uv, xyz)
    assert best == min(both) and best <= both[0]                     # never worse than the Bernstein bound
    for bound in both:
        for i in range(9):
            for j in range(9):
                residual = _exact_quad_residual(support, uv, xyz, Q(i, 8), Q(j, 8))
                assert sum(r*r for r in residual) <= bound*bound


def test_hierarchical_bound_with_interval_corners_encloses_every_independent_extreme_choice():
    from itertools import product as choices
    from anygeometry.polynomial_cell_bounds import quad_residual_bound
    support = cropped_support(63 / 128, 65 / 128)
    xyz = tuple(exact(support, p) for p in SKEW_QUAD)
    boxes = tuple(tuple((x, x + EPS) for x in p) for p in SKEW_QUAD)
    bound = quad_residual_bound(support, boxes, xyz, orientation=1, method='hierarchical')
    assert bound is not None
    for selection in choices((0, 1), repeat=8):
        chosen = tuple(tuple(box[k][selection[2*n + k]] for k in range(2)) for n, box in enumerate(boxes))
        for i in range(3):
            for j in range(3):
                residual = _exact_quad_residual(support, chosen, xyz, Q(i, 2), Q(j, 2))
                assert sum(r*r for r in residual) <= bound*bound


def test_coarse_genuine_error_is_still_over_the_limit_under_both_bounds():
    from anygeometry.polynomial_cell_bounds import quad_residual_bound
    support = cropped_support(.25, .75)
    uv = ((Q(0), Q(0)), (Q(1), Q(0)), (Q(1), Q(1)), (Q(0), Q(1)))
    xyz = tuple(exact(support, p) for p in uv)
    for method in ('bernstein', 'hierarchical', None):
        assert quad_residual_bound(support, uv, xyz, method=method) > Q(25, 10000)


def test_uncertain_orientation_domain_escape_and_nonquadratic_support_all_fall_back():
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.polynomial_cell_bounds import triangle_residual_bound, quad_residual_bound
    from test_section_boundary_sampling import KNOTS
    support = cropped_support(63 / 128, 65 / 128)
    centre = (Q(1, 2), Q(1, 20))
    ambiguous = (((Q(0), Q(0)), (Q(0), Q(1, 10))), (Q(1), Q(0)), centre)       # wedge interval straddles zero
    xyz = tuple(exact(support, (min(max(p[0] if not isinstance(p[0], tuple) else Q(0), Q(0)), Q(1)),
                                min(max(p[1] if not isinstance(p[1], tuple) else Q(0), Q(0)), Q(1))))
                for p in ambiguous)
    assert triangle_residual_bound(support, ambiguous, xyz, orientation=1) is None
    outside = (((-Q(1, 100), Q(0)), (Q(0), Q(0))), (Q(1), Q(0)), (Q(1, 2), Q(1, 2)))
    assert triangle_residual_bound(support, outside, xyz, orientation=1) is None             # leaves the chart
    heights = (1., -1., 1.)
    product = NurbsSurface(2, 2, KNOTS, KNOTS, tuple(tuple((i - 1., j - 1., heights[i]*heights[j])
                                                          for j in range(3)) for i in range(3)))
    nominal = ((Q(1, 8), Q(1, 8)), (Q(7, 8), Q(1, 8)), (Q(7, 8), Q(7, 8)), (Q(1, 8), Q(7, 8)))
    boxes = tuple(tuple((x, x + EPS) for x in p) for p in nominal)
    quad_xyz = tuple(exact(support, p) for p in nominal)
    assert quad_residual_bound(product, boxes, quad_xyz, orientation=1) is None
    flipped = (nominal[0], nominal[3], nominal[2], nominal[1])
    assert quad_residual_bound(support, flipped, quad_xyz, orientation=1) is None           # clockwise corners
    reversed_box = (((Q(1, 4), Q(1, 8)), (Q(0), Q(0))), (Q(1), Q(0)), (Q(1, 2), Q(1, 2)))   # lo > hi: malformed
    assert triangle_residual_bound(support, reversed_box, xyz, orientation=1) is None


def test_fan_certifier_cancels_at_entry_before_any_fast_path_and_propagates_callback_errors():
    from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle
    from anygeometry.regular_source_reference import SourceNode
    support = cropped_support(63 / 128, 65 / 128)
    uv = ((Q(1, 8), Q(1, 8)), (Q(7, 8), Q(1, 8)), (Q(1, 2), Q(1, 2)))
    xyz = tuple(exact(support, p) for p in uv)
    node = lambda p: SourceNode(('true', *p), tuple((x, x + EPS) for x in p))
    side = SourceSide(node(uv[0]), node(uv[1]))

    def call(check):
        return certify_fan_subtriangle(support, side, uv[2], (xyz[0], xyz[1]), xyz[2], orientation=1,
                                       surface_error=Q(25, 10000), cancellation_check=check)

    assert call(lambda: False)[1] == 1                                 # uncancelled interval fast path accepts
    with pytest.raises(GeometryError, match='cancelled'):
        call(lambda: True)

    class Boom(BaseException):
        pass

    marker = Boom()

    def failing():
        raise marker

    with pytest.raises(Boom) as raised:
        call(failing)
    assert raised.value is marker


def test_fan_certifier_accepts_uncertain_true_endpoint_boxes_with_one_charged_step_and_no_general_proof(monkeypatch):
    from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.regular_source_reference import SourceNode
    from anygeometry.section_audit import Work
    support = cropped_support(63 / 128, 65 / 128)
    uv = ((Q(1, 8), Q(1, 8)), (Q(7, 8), Q(1, 8)), (Q(1, 2), Q(1, 2)))
    xyz = tuple(exact(support, p) for p in uv)
    node = lambda p: SourceNode(('true', *p), tuple((x, x + EPS) for x in p))
    calls = []
    real = SourceSide.box
    monkeypatch.setattr(SourceSide, 'box', lambda self, lo, hi: calls.append(1) or real(self, lo, hi))
    work = Work(10**6, 40)
    token = CURRENT_WORK.set(work)
    try:
        error, boxes = certify_fan_subtriangle(support, SourceSide(node(uv[0]), node(uv[1])), uv[2],
            (xyz[0], xyz[1]), xyz[2], orientation=1, surface_error=Q(25, 10000))
    finally:
        CURRENT_WORK.reset(token)
    assert boxes == 1 and 0 < error <= Q(25, 10000) and calls == [] and work.boxes == 1


# Not covered: an undeclared-contact skeleton refusal and the consumer's saved 350-cell candidate (which stays
# invalid by its true centre error); both are left to root's saved-cell replay.
