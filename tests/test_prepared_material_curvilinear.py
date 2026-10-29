from dataclasses import replace
import pytest
from anygeometry import (GeometryError, query_material_surface_regions,
    query_prepared_material_boundary_chords, validate_prepared_material_boundary_chords_binding,
    validate_prepared_material_curvilinear_cells, validate_prepared_material_curvilinear_cells_binding,
    CurvilinearMaterialCell, GeometryModel, plan_plane_split, apply_plane_split)
from anygeometry import (query_prepared_material_boundary_sampling,
    validate_prepared_material_boundary_sampling_binding, IntersectionQualificationPolicy)
from anygeometry.nurbs import NurbsSurface, NurbsCurve, CurveOnSurface
from anygeometry.nurbs_metric import certify_projected_chart
from test_plane_split import bowl
from anygeometry.serialization import to_dict
import math
import numpy as np


def test_chord_receipt_binds_literal_station_source_and_regular_chart():
    model, face, plane = bowl()
    regions = query_material_surface_regions(model, [face])
    edge = model.faces[face].loop[0].edge
    result = query_prepared_material_boundary_chords(model, regions,
        [(model.handle('edge', edge), 0., .25)], chord_error=.1)
    assert result.intervals[0].world_error <= .1
    assert result.reference_uv_by_face
    validate_prepared_material_boundary_chords_binding(model, regions, result)
    object.__setattr__(result, 'chord_error', 1.)
    with pytest.raises(GeometryError, match='changed'):
        validate_prepared_material_boundary_chords_binding(model, regions, result)


def test_surface_chart_projection_rejects_rank_loss_at_boundary():
    surface = NurbsSurface(1, 2, (0., 0., 1., 1.), (0., 0., 0., 1., 1., 1.),
        tuple(tuple((x, y, 0.) for y in (0., 0., 1.)) for x in (0., 1.)))
    with pytest.raises(GeometryError, match='immersion'):
        certify_projected_chart(surface)


def square_cells(quad=True):
    model = GeometryModel()
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    vertices = model.add_points(tuple((*p, 0.) for p in uv))
    edges = [model.add_curve(vertices[i], vertices[(i+1)%4], CurveOnSurface(support,
        NurbsCurve(1, (0., 0., 1., 1.), (p, uv[(i+1)%4])))) for i, p in enumerate(uv)]
    face = model.add_face(edges, surface=support)
    regions = query_material_surface_regions(model, [face])
    chords = query_prepared_material_boundary_chords(model, regions,
        tuple((model.handle('edge', edge), 0., 1.) for edge in edges), chord_error=.01)
    xyz = tuple((*p, 0.) for p in uv)
    if quad:
        cells = (CurvilinearMaterialCell(model.handle('face', face), (0, 1, 2, 3), uv, xyz,
                    tuple((i, i, True) for i in range(4))),)
    else:
        cells = (CurvilinearMaterialCell(model.handle('face', face), (0, 1, 2), uv[:3], xyz[:3],
                    ((0, 0, True), (1, 1, True))),
                 CurvilinearMaterialCell(model.handle('face', face), (0, 2, 3), (uv[0], uv[2], uv[3]),
                    (xyz[0], xyz[2], xyz[3]), ((1, 2, True), (2, 3, True))))
    return model, regions, chords, cells


@pytest.mark.parametrize('quad', (False, True))
def test_public_source_cell_triangle_and_bilinear_quad_identity(quad):
    model, regions, chords, cells = square_cells(quad)
    receipt = validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.001)
    validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)
    assert receipt.reference_uv_by_cell
    assert all(math.isfinite(row[-1]) and 0 <= row[-1] <= .001 for row in receipt.source_map_bounds)
    assert receipt.cells == cells
    assert len(receipt.boundary_correspondence) == 4


def test_cell_receipt_refuses_copy_tamper_and_changed_literal_station():
    model, regions, chords, cells = square_cells()
    receipt = validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.001)
    with pytest.raises(GeometryError, match='not issued'):
        validate_prepared_material_curvilinear_cells_binding(model, regions, replace(receipt))
    object.__setattr__(receipt, 'surface_error', .5)
    with pytest.raises(GeometryError, match='changed'):
        validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)
    changed = replace(cells[0], xyz=((.01, 0., 0.), *cells[0].xyz[1:]))
    with pytest.raises(GeometryError, match='literal target'):
        validate_prepared_material_curvilinear_cells(model, regions, chords, (changed,), surface_error=.1)


def test_cell_final_cancellation_keeps_source_and_issues_no_receipt():
    model, regions, chords, cells = square_cells()
    before = to_dict(model)
    with pytest.raises(GeometryError, match='cancelled'):
        validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.001,
            cancellation_check=lambda phase: phase == 'curvilinear source-cell qualification complete')
    assert to_dict(model) == before


def test_native_evaluation_kernel_and_point_cache_are_not_geometry_authority():
    from anygeometry.native_support_snapshots import capture_native_supports
    model, regions, chords, cells = square_cells()
    curve = model.edges[next(iter(model.edges))].curve.pcurve
    original = capture_native_supports(model)
    curve.evaluate(.25)
    assert capture_native_supports(model) == original
    other = replace(curve, weights=(1., 2.))
    prepared = curve._prepared
    object.__setattr__(curve, '_prepared', other._prepared)
    with pytest.raises(GeometryError, match='evaluated kernel'):
        capture_native_supports(model)
    object.__setattr__(curve, '_prepared', prepared)
    object.__setattr__(curve, '_point_values', {np.float64(.5).tobytes(): np.ones((1, 1, 2))})
    with pytest.raises(GeometryError, match='point cache'):
        capture_native_supports(model)


def test_closed_section_disk_has_true_source_boundary_coverage():
    model, face, plane = bowl()
    apply_plane_split(model, plan_plane_split(model, [face], plane), policy='imprint')
    disk = next(f for f in model.faces.values() if not f.holes)
    regions = query_material_surface_regions(model, [disk.id])
    chords = query_prepared_material_boundary_chords(model, regions,
        tuple((model.handle('edge', use.edge), 0., 1.) for use in disk.loop), chord_error=1.)
    support = disk.surface
    centre_uv = (.5, .5)
    centre_xyz = tuple(map(float, support.evaluate(*centre_uv)))
    cells = []
    for index, use in enumerate(disk.loop):
        chord = chords.intervals[index]
        occurrence = next(o for o in chord.occurrences if o.face.id == disk.id)
        edge = model.edges[use.edge]
        indices = (0, 1) if use.forward else (1, 0)
        vertices = (edge.start, edge.end) if use.forward else (edge.end, edge.start)
        cells.append(CurvilinearMaterialCell(model.handle('face', disk.id),
            (*vertices, 0), (*tuple(occurrence.uv[i] for i in indices), centre_uv),
            (*tuple(chord.xyz[i] for i in indices), centre_xyz), ((0, index, use.forward),)))
    receipt = validate_prepared_material_curvilinear_cells(model, regions, chords, tuple(cells), surface_error=2.)
    validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)
    assert len(receipt.boundary_correspondence) == len(disk.loop)
    assert all(row[-1] <= 2 for row in receipt.source_map_bounds)


def test_annulus_quad_complex_covers_hole_once_without_filling_it():
    # Exact planar tensor chart isolates hole/degree/incidence authority from
    # curved-map approximation. The separate disk fixture exercises true
    # algebraic boundary vertices and non-affine source geometry.
    from anygeometry.entities import OrientedEdge
    model, _regions, _chords, _cells = square_cells()
    face = next(iter(model.faces.values()))
    support = face.surface
    inner = ((.25, .25), (.75, .25), (.75, .75), (.25, .75))
    vertices = model.add_points(tuple((*p, 0.) for p in inner))
    edges = [model.add_curve(vertices[i], vertices[(i+1)%4], CurveOnSurface(support,
        NurbsCurve(1, (0., 0., 1., 1.), (p, inner[(i+1)%4])))) for i, p in enumerate(inner)]
    with model.transaction():
        model._put_entity('face', replace(face, holes=(tuple(OrientedEdge(edge, True) for edge in edges),)))
    regions = query_material_surface_regions(model, [face.id])
    outer_edges = tuple(use.edge for use in face.loop)
    chords = query_prepared_material_boundary_chords(model, regions,
        tuple((model.handle('edge', edge), 0., 1.) for edge in (*outer_edges, *edges)), chord_error=.01)
    outer = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    cells = []
    for i in range(4):
        j = (i+1)%4
        uv = (inner[i], outer[i], outer[j], inner[j])
        cells.append(CurvilinearMaterialCell(model.handle('face', face.id), (4+i, i, j, 4+j), uv,
            tuple((*p, 0.) for p in uv), ((1, i, True), (3, 4+i, False))))
    receipt = validate_prepared_material_curvilinear_cells(model, regions, chords, tuple(cells), surface_error=1.)
    assert len(receipt.cells) == 4
    # Removing a cell must not acquire authority for an incomplete annulus.
    with pytest.raises(GeometryError):
        validate_prepared_material_curvilinear_cells(model, regions, chords, tuple(cells[:-1]), surface_error=1.)


def test_cell_binding_rejects_stale_source_and_callback_input_tampering():
    model, regions, chords, cells = square_cells()
    receipt = validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.001)
    model.add_point(2., 2., 0.)
    with pytest.raises(GeometryError):
        validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)
    model, regions, chords, cells = square_cells()
    def tamper(phase):
        if phase == 'curvilinear source-cell qualification complete':
            object.__setattr__(cells[0], 'node_ids', (3, 2, 1, 0))
        return False
    with pytest.raises(GeometryError, match='changed'):
        validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.001,
            cancellation_check=tamper)


def test_boundary_final_gate_refuses_callback_poisoned_private_cache():
    model, regions, _chords, _cells = square_cells()
    curve = model.edges[next(iter(model.edges))].curve.pcurve
    def poison(phase):
        if phase == 'curvilinear boundary qualification complete':
            object.__setattr__(curve, '_point_values', {np.float64(.5).tobytes(): np.ones((1, 1, 2))})
        return False
    with pytest.raises(GeometryError, match='point cache'):
        query_prepared_material_boundary_chords(model, regions,
            tuple((model.handle('edge', edge), 0., 1.) for edge in model.edges),
            chord_error=.01, cancellation_check=poison)


def test_malformed_private_definition_binding_refuses_with_geometry_error():
    from anygeometry.native_support_snapshots import capture_native_supports
    model, _regions, _chords, _cells = square_cells()
    curve = model.edges[next(iter(model.edges))].curve.pcurve
    object.__setattr__(curve, '_prepared_binding', ())
    with pytest.raises(GeometryError, match='definition binding'):
        capture_native_supports(model)


@pytest.mark.parametrize('attack_phase', ('curvilinear source binding', 'curvilinear source-cell qualification'))
def test_temporary_repair_then_restore_cannot_certify_invalid_cells(attack_phase):
    model, regions, chords, _cells = square_cells()
    face = next(iter(model.faces))
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.), (.5, .5))
    cells = tuple(CurvilinearMaterialCell(model.handle('face', face), (i, (i+1)%4, 4),
        (uv[i], uv[(i+1)%4], uv[4]),
        ((*uv[i], 0.), (*uv[(i+1)%4], 0.), (.5, .5, .1)), ((0, i, True),)) for i in range(4))
    bad = tuple(cell.xyz for cell in cells)
    repaired, completed = False, False
    def attack(phase):
        nonlocal repaired, completed
        if phase == attack_phase and not repaired:
            repaired = True
            for cell in cells:
                object.__setattr__(cell, 'xyz', (*cell.xyz[:2], (.5, .5, 0.)))
        if phase == 'curvilinear source-cell qualification complete':
            completed = True
            for cell, original in zip(cells, bad):
                object.__setattr__(cell, 'xyz', original)
        return False
    try:
        with pytest.raises(GeometryError, match='changed during callback'):
            validate_prepared_material_curvilinear_cells(model, regions, chords, cells,
                surface_error=.001, cancellation_check=attack)
        assert repaired and not completed
    finally:
        for cell, original in zip(cells, bad):
            object.__setattr__(cell, 'xyz', original)


@pytest.mark.parametrize('operand', ('chords', 'source', 'regions'))
@pytest.mark.parametrize('attack_phase', ('curvilinear source binding', 'curvilinear source-cell qualification'))
def test_temporary_operand_changes_are_rejected_before_proof_reads(operand, attack_phase):
    model, regions, chords, cells = square_cells()
    if operand == 'chords':
        target, field = chords.intervals[0], 'xyz'
        original = target.xyz
        replacement = ((.01, 0., 0.), original[1])
    elif operand == 'source':
        target, field = model.vertices[next(iter(model.vertices))], 'position'
        original = target.position
        replacement = np.array((.01, 0., 0.))
    else:
        target, field = regions.regions[0].sources[0], 'world_tolerance'
        original = target.world_tolerance
        replacement = original*2
    changed, completed = False, False
    def attack(phase):
        nonlocal changed, completed
        if phase == attack_phase and not changed:
            changed = True
            object.__setattr__(target, field, replacement)
        if phase == 'curvilinear source-cell qualification complete':
            completed = True
            object.__setattr__(target, field, original)
        return False
    try:
        with pytest.raises(GeometryError, match='changed during callback'):
            validate_prepared_material_curvilinear_cells(model, regions, chords, cells,
                surface_error=.001, cancellation_check=attack)
        assert changed and not completed
    finally:
        object.__setattr__(target, field, original)


def test_issued_cell_literals_do_not_alias_caller_dataclasses():
    model, regions, chords, cells = square_cells()
    receipt = validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.001)
    assert receipt.cells == cells and receipt.cells[0] is not cells[0]
    assert receipt.cells[0].face is not cells[0].face
    object.__setattr__(cells[0], 'xyz', ((99., 0., 0.), *cells[0].xyz[1:]))
    validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)


def s_bend_boundary():
    model = GeometryModel()
    support = NurbsSurface(3, 1, (0., 0., 0., 0., 1., 1., 1., 1.), (0., 0., 1., 1.),
        tuple(tuple((x, y, z) for y in (0., 1.))
              for x, z in zip((0., 1/3, 2/3, 1.), (0., 1., -1., 0.))))
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    vertices = model.add_points(tuple(tuple(support.evaluate(*p)) for p in uv))
    edges = [model.add_curve(vertices[i], vertices[(i+1)%4], CurveOnSurface(support,
        NurbsCurve(1, (0., 0., 1., 1.), (p, uv[(i+1)%4])))) for i, p in enumerate(uv)]
    face = model.add_face(edges, surface=support)
    return model, query_material_surface_regions(model, [face]), model.handle('edge', edges[0])


def test_owner_sampling_refines_s_bend_with_midpoint_on_original_chord():
    model, regions, edge = s_bend_boundary()
    before = to_dict(model)
    # The central point is on the endpoint chord; quarter points deviate.
    assert abs(model.edges[edge.id].curve.evaluate(.5)[2]) < 1e-14
    result = query_prepared_material_boundary_sampling(model, regions, ((edge, (0., 1.)),), chord_error=.02)
    validate_prepared_material_boundary_sampling_binding(model, regions, result)
    stations = result.edge_parameters[0][1]
    assert len(stations) > 2 and stations[0] == 0 and stations[-1] == 1
    assert all(float(t).as_integer_ratio() == (t.numerator, t.denominator) for t in stations)
    assert len(result.chords.intervals) == len(stations)-1
    assert all(chord.world_error <= .02 for chord in result.chords.intervals)
    assert to_dict(model) == before


def test_owner_sampling_retains_reversed_seeds_and_binds_literal_receipt():
    model, regions, edge = s_bend_boundary()
    seeds = (1., .75, 0.)
    result = query_prepared_material_boundary_sampling(model, regions, ((edge, seeds),), chord_error=.05)
    stations = result.edge_parameters[0][1]
    assert all(a > b for a, b in zip(stations, stations[1:])) and set(seeds).issubset(stations)
    with pytest.raises(GeometryError, match='not issued'):
        validate_prepared_material_boundary_sampling_binding(model, regions, replace(result))
    object.__setattr__(result, 'chord_error', .5)
    with pytest.raises(GeometryError, match='changed'):
        validate_prepared_material_boundary_sampling_binding(model, regions, result)


def test_owner_sampling_exhaustion_and_cancellation_refuse_without_mutation():
    model, regions, edge = s_bend_boundary()
    before = to_dict(model)
    with pytest.raises(GeometryError, match='budget'):
        query_prepared_material_boundary_sampling(model, regions, ((edge, (0., 1.)),), chord_error=.001,
            qualification=IntersectionQualificationPolicy(max_boxes_per_pair=1))
    with pytest.raises(GeometryError, match='cancelled'):
        query_prepared_material_boundary_sampling(model, regions, ((edge, (0., 1.)),), chord_error=.05,
            cancellation_check=lambda phase: phase == 'curvilinear boundary sampling complete')
    assert to_dict(model) == before


def test_degenerate_plane_chart_enclosures_are_each_resource_charged():
    from anygeometry.prepared_material_curvilinear import _uv_box, _uv_interval
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    from anygeometry import Plane
    class Bounds:
        calls = 0
        def bounds(self, a, b):
            self.calls += 1
            return ((a, 0., 0.), (b, 0., 0.))
    curve, work = Bounds(), Work()
    token = CURRENT_WORK.set(work)
    try:
        plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
        _uv_box(plane, curve, .5)
        _uv_interval(plane, curve, .25, .75)
        assert curve.calls == work.boxes == 2
    finally:
        CURRENT_WORK.reset(token)


def closed_disk_chords():
    model, face, plane = bowl()
    apply_plane_split(model, plan_plane_split(model, [face], plane), policy='imprint')
    disk = next(f for f in model.faces.values() if not f.holes)
    regions = query_material_surface_regions(model, [disk.id])
    intervals = tuple((model.handle('edge', use.edge), 0., 1.) for use in disk.loop)
    chords = query_prepared_material_boundary_chords(model, regions, intervals, chord_error=1.)
    curve = model.edges[disk.loop[0].edge].curve
    return model, regions, intervals, chords, curve


def test_section_runtime_changed_arc_parameter_refuses_existing_binding():
    model, regions, _intervals, chords, curve = closed_disk_chords()
    arc = curve._arcs()[0]
    original = arc.s0
    try:
        object.__setattr__(arc, 's0', (arc.s0+arc.s1)/2)
        with pytest.raises(GeometryError, match='runtime cache'):
            validate_prepared_material_boundary_chords_binding(model, regions, chords)
    finally:
        object.__setattr__(arc, 's0', original)


def test_section_runtime_final_callback_cannot_poison_arc_parameter():
    model, regions, intervals, _chords, curve = closed_disk_chords()
    arc, changed = curve._arcs()[0], False
    original = arc.s0
    def poison(phase):
        nonlocal changed
        if phase == 'curvilinear boundary qualification complete':
            changed = True
            object.__setattr__(arc, 's0', (arc.s0+arc.s1)/2)
        return False
    try:
        with pytest.raises(GeometryError, match='changed during callback.*runtime cache'):
            query_prepared_material_boundary_chords(model, regions, intervals, chord_error=1.,
                cancellation_check=poison)
        assert changed
    finally:
        object.__setattr__(arc, 's0', original)


def test_fresh_section_oracle_does_not_trust_already_poisoned_cache():
    from anygeometry.nurbs_runtime_binding import _SECTION_CANONICAL, validate_section_runtime
    model, _regions, _intervals, _chords, curve = closed_disk_chords()
    key = (curve.plane_record, curve.surface, curve.path)
    _SECTION_CANONICAL.pop(key, None)
    arc = curve._arcs()[0]
    original = arc.s0
    try:
        object.__setattr__(arc, 's0', (arc.s0+arc.s1)/2)
        with pytest.raises(GeometryError, match='runtime cache'):
            validate_section_runtime(curve)
    finally:
        object.__setattr__(arc, 's0', original)


@pytest.mark.parametrize('warp', (0., .1))
def test_true_bilinear_source_quads_cover_skew_grid_with_small_error(warp):
    model = GeometryModel()
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., warp))))
    corners = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    vertices = model.add_points(tuple(tuple(support.evaluate(*p)) for p in corners))
    edges = [model.add_curve(vertices[i], vertices[(i+1)%4], CurveOnSurface(support,
        NurbsCurve(1, (0., 0., 1., 1.), (p, corners[(i+1)%4])))) for i, p in enumerate(corners)]
    face = model.add_face(edges, surface=support)
    regions = query_material_surface_regions(model, [face])
    chords = query_prepared_material_boundary_chords(model, regions,
        tuple((model.handle('edge', edge), a, b) for edge in edges for a, b in ((0., .5), (.5, 1.))),
        chord_error=.01)
    uv = ((0., 0.), (.5, 0.), (1., 0.), (0., .5), (.3, .4), (1., .5), (0., 1.), (.5, 1.), (1., 1.))
    connectivity = ((0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7))
    boundaries = (((0, 0, True), (3, 7, True)), ((0, 1, True), (1, 2, True)),
                  ((2, 5, True), (3, 6, True)), ((1, 3, True), (2, 4, True)))
    xyz = tuple(tuple(map(float, support.evaluate(*p))) for p in uv)
    cells = tuple(CurvilinearMaterialCell(model.handle('face', face), ids, tuple(uv[i] for i in ids),
        tuple(xyz[i] for i in ids), boundary) for ids, boundary in zip(connectivity, boundaries))
    receipt = validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.01)
    assert receipt.cells == cells and len(receipt.cells) == 4
    assert all(0 <= bound[-1] <= .01 for bound in receipt.source_map_bounds)
    validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)


def _captured_flat_implicit_quad_fan():
    """Literal cell from root-run25-flat-fan-failure-capture-20261024-234911.

    Fresh owner kernels regenerate its implicit side; no successful receipt or
    material authority is fabricated by this standalone certificate fixture.
    """
    from fractions import Fraction as Q
    from anygeometry import Plane
    from anygeometry.support_curves import PlaneNurbsCurve
    from anygeometry.regular_source_reference import SourceNode, source_node, source_side
    support = NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,0.))))
    curve = PlaneNurbsCurve.from_plane(Plane((.6,0.,0.),(0.,1.,0.),(0.,0.,1.)),support,0)
    first,last = source_node(curve,Q(0)),source_node(curve,Q(1,4))
    boundary = source_side(curve,(Q(0),Q(1,4)),first,last)
    points = ((Q(.6),Q(0)),(Q(.6),Q(1,4)),
              (Q(7860590359242521,18014398509481984),Q(6600890115006585,18014398509481984)),
              (Q(7057864605421405,18014398509481984),Q(1645884317696385,9007199254740992)))
    assert first.box == tuple((x,x) for x in points[0])
    assert last.box == tuple((x,x) for x in points[1])
    nodes = (first,last,*(SourceNode(('captured_interior',i),tuple((x,x) for x in point))
                           for i,point in enumerate(points[2:],2)))
    uv = tuple(node.box for node in nodes)
    xyz = tuple(tuple(float(x) for x in point)+(0.,) for point in points)
    centre = (Q(570870830875661,1125899906842624),Q(3599064594442463,18014398509481984))
    centre_xyz = tuple(sum(Q(row[i]) for row in xyz)/4 for i in range(3))
    return support,boundary,centre,xyz,centre_xyz,uv,nodes


def test_actual_captured_flat_implicit_quad_fan_certifies_with_canonical_corner_hull():
    from fractions import Fraction as Q
    from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle
    support,boundary,centre,xyz,centre_xyz,uv,nodes = _captured_flat_implicit_quad_fan()
    bound,pieces = certify_fan_subtriangle(support,boundary,centre,xyz[:2],centre_xyz,
        orientation=1,surface_error=Q(.0025),quad_xyz=xyz,quad_side=0,quad_uv=uv,quad_nodes=nodes)
    assert 0 <= bound <= Q(.0025) and pieces > 0


def test_captured_fan_domain_tightening_preserves_map_derivatives_and_old_unproved_route():
    from fractions import Fraction as Q
    from anygeometry.curvilinear_reference_maps import _source_quad_fan
    _,boundary,_,_,_,uv,_ = _captured_flat_implicit_quad_fan()
    s,t = (Q(1,64),Q(1,32)),(Q(0),Q(1,64))
    old = _source_quad_fan(uv,boundary,0,s,t)
    tight = _source_quad_fan(uv,boundary,0,s,t,corner_hull=True)
    assert old[0][1][0] < 0  # actual original conservative refusal discriminator
    assert all(0 <= a <= b <= 1 for a,b in tight[0])
    assert tight[1:] == old[1:]
    assert all(a <= c <= d <= b for (a,b),(c,d) in zip(old[0],tight[0]))


@pytest.mark.parametrize('side',range(4))
def test_corrected_quad_fan_convex_weights_and_adversarial_point_map_coverage(side):
    from fractions import Fraction as Q
    from anygeometry.regular_source_reference import SourceNode
    from anygeometry.curvilinear_reference_maps import _source_quad_fan
    corners = ((Q(0),Q(0)),(Q(1),Q(0)),(Q(4,5),Q(1)),(Q(1,10),Q(3,4)))
    uv = tuple(tuple((x,x) for x in p) for p in corners)
    class Boundary:
        curve = object()
        start = SourceNode(('first',),uv[side])
        end = SourceNode(('last',),uv[(side+1)%4])
        def box(self,a,b):
            assert a == b
            p,q = corners[side],corners[(side+1)%4]
            point = (p[0]+a*(q[0]-p[0]),p[1]+a*(q[1]-p[1])-a*(1-a)/8)
            return tuple((x,x) for x in point)
        def derivative(self,a,b):
            p,q = corners[side],corners[(side+1)%4]
            return ((q[0]-p[0],)*2,(q[1]-p[1]-(1-2*a)/8,)*2)
    boundary = Boundary()
    for s in (Q(0),Q(1,7),Q(1,2),Q(1)):
        for t in (Q(0),Q(2,9),Q(1)):
            u = s/2+(1-s)*t
            residual = (s*s/4+s*(1-s)*t/2,s*s/4+s*(1-s)*(1-t)/2,s*u/2,s*(1-u)/2)
            assert all(w >= 0 for w in residual) and sum(residual) == s
            weights = ((1-s),*residual)
            point_q = tuple(a for a,b in boundary.box(t,t))
            point = tuple(weights[0]*point_q[i]+sum(residual[k]*corners[(side+k)%4][i] for k in range(4)) for i in (0,1))
            old = _source_quad_fan(uv,boundary,side,(s,s),(t,t))
            tight = _source_quad_fan(uv,boundary,side,(s,s),(t,t),corner_hull=True)
            assert all(a <= x <= b for (a,b),x in zip(tight[0],point))
            assert tight[1:] == old[1:]
            if side == 0 and s == 0 and t == Q(2,9):
                assert tight[0][1][0] < 0  # true outside curve is never global-chart clamped


def test_quad_corner_hull_requires_same_exact_canonical_endpoint_identity_and_boxes():
    from fractions import Fraction as Q
    from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle
    support,boundary,centre,xyz,centre_xyz,uv,nodes = _captured_flat_implicit_quad_fan()
    wrong = (replace(nodes[0],identity=('unrelated_equal_box',)),*nodes[1:])
    with pytest.raises(GeometryError,match='canonical corner identities and boxes'):
        certify_fan_subtriangle(support,boundary,centre,xyz[:2],centre_xyz,
            orientation=1,surface_error=Q(.0025),quad_xyz=xyz,quad_side=0,quad_uv=uv,quad_nodes=wrong)


def test_captured_quad_hull_preserves_entry_cancel_and_zero_box_work_refusal():
    from fractions import Fraction as Q
    from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    support,boundary,centre,xyz,centre_xyz,uv,nodes = _captured_flat_implicit_quad_fan()
    kwargs = dict(orientation=1,surface_error=Q(.0025),quad_xyz=xyz,quad_side=0,quad_uv=uv,quad_nodes=nodes)
    with pytest.raises(GeometryError,match='cancelled'):
        certify_fan_subtriangle(support,boundary,centre,xyz[:2],centre_xyz,cancellation_check=lambda: True,**kwargs)
    token = CURRENT_WORK.set(Work(0,12))
    try:
        with pytest.raises(GeometryError,match='budget exhausted'):
            certify_fan_subtriangle(support,boundary,centre,xyz[:2],centre_xyz,**kwargs)
    finally:
        CURRENT_WORK.reset(token)
