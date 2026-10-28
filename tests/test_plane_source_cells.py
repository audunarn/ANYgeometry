"""Plane implicit-loop source-cell authority, separate from literal facets."""
from fractions import Fraction as Q
import pytest
from anygeometry import (GeometryModel, Plane, CurvilinearMaterialCell,
    query_material_surface_regions, query_prepared_material_boundary_chords,
    validate_prepared_material_curvilinear_cells,
    validate_prepared_material_curvilinear_cells_binding,
    plan_intersections, apply_intersections)
from test_bounded_nurbs import fixture


def direct_plane_disk():
    from dataclasses import replace
    from test_plane_split import bowl
    from anygeometry.support_curves import PlaneNurbsCurve
    model, wall, plane = bowl()
    curve = PlaneNurbsCurve.from_plane(plane, model.faces[wall].surface)
    points = [model.add_point(*curve.evaluate(i*.25)) for i in range(4)]
    edges = [model.add_curve(points[i], points[(i+1)%4], replace(curve, domain=(i*.25, (i+1)*.25))) for i in range(4)]
    face = model.add_face(edges, surface=plane)
    return model, model.faces[face]


def test_plane_implicit_chart_area_uses_same_estimator_as_binding():
    from anygeometry import query_trimmed_surface_charts, validate_trimmed_surface_charts_binding
    model, disk = direct_plane_disk()
    charts = query_trimmed_surface_charts(model, [disk.id])
    validate_trimmed_surface_charts_binding(model, charts)
    regions = query_material_surface_regions(model, [disk.id])
    assert regions.regions


def test_affine_plane_quad_source_cell_uses_published_literal_targets():
    model = GeometryModel()
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    uv = ((-2., -2.), (2., -2.), (2., 2.), (-2., 2.))
    vertices = model.add_points(tuple((*p, 0.) for p in uv))
    edges = [model.add_line(vertices[i], vertices[(i+1)%4]) for i in range(4)]
    face = model.add_face(edges, surface=plane)
    regions = query_material_surface_regions(model, [face])
    chords = query_prepared_material_boundary_chords(model, regions,
        [(model.handle('edge', edge), 0., 1.) for edge in edges], chord_error=.001)
    cell = CurvilinearMaterialCell(model.handle('face', face), (1, 2, 3, 4), uv,
        tuple((*p, 0.) for p in uv), tuple((i, i, True) for i in range(4)))
    receipt = validate_prepared_material_curvilinear_cells(model, regions, chords, [cell], surface_error=.001)
    validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)
    assert receipt.cells == (cell,)


def test_planar_closed_section_disk_has_true_curved_cell_partition():
    # Direct owner-defined four-quarter section avoids spending this focused
    # source-cell gate's time in the separately tested bounded imprint path.
    model, disk = direct_plane_disk()
    regions = query_material_surface_regions(model, [disk.id])
    intervals = [(model.handle('edge', use.edge), a, b) for use in disk.loop for a, b in ((Q(0), Q(1, 2)), (Q(1, 2), Q(1)))]
    chords = query_prepared_material_boundary_chords(model, regions, intervals, chord_error=1.)
    ids, cells = {}, []
    def station(edge, t):
        key = ('v', edge.start if t == 0 else edge.end) if t in (0, 1) else ('e', edge.id, t)
        if key not in ids:
            ids[key] = len(ids)+1
        return ids[key]
    centre = 1000000
    for index, chord in enumerate(chords.intervals):
        edge = model.edges[chord.edge.id]
        use = next(use for use in disk.loop if use.edge == edge.id)
        occurrence = next(o for o in chord.occurrences if o.face.id == disk.id)
        a, b = chord.parameter_range
        node_ids = (station(edge, a), station(edge, b))
        uv, xyz = occurrence.uv, chord.xyz
        if not use.forward:
            node_ids, uv, xyz = node_ids[::-1], uv[::-1], xyz[::-1]
        cells.append(CurvilinearMaterialCell(model.handle('face', disk.id), (*node_ids, centre),
            (*uv, (0., 0.)), (*xyz, (0., 0., .5)), ((0, index, use.forward),)))
    receipt = validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=1.)
    validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)
    assert len(receipt.boundary_correspondence) == len(intervals)
