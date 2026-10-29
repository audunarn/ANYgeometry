"""Axis COS/implicit true endpoint reference-cell integration (unrun)."""
from anygeometry import (Plane, plan_plane_split, apply_plane_split, query_material_surface_regions,
    query_prepared_material_boundary_chords, validate_prepared_material_curvilinear_cells,
    validate_prepared_material_curvilinear_cells_binding, CurvilinearMaterialCell)
from test_prepared_material_curvilinear import square_cells


def fan_receipt(model, face):
    regions = query_material_surface_regions(model, [face.id])
    chords = query_prepared_material_boundary_chords(model, regions,
        tuple((model.handle('edge', use.edge), 0., 1.) for use in face.loop), chord_error=.01)
    uv, xyz, nodes = [], [], []
    for index, use in enumerate(face.loop):
        chord = chords.intervals[index]
        occurrence = next(item for item in chord.occurrences if item.face.id == face.id)
        end = 0 if use.forward else 1
        uv.append(occurrence.uv[end]); xyz.append(chord.xyz[end])
        edge = model.edges[use.edge]
        nodes.append(edge.start if use.forward else edge.end)
    centre = tuple(sum(p[i] for p in uv)/len(uv) for i in (0, 1))
    target = tuple(float(x) for x in face.surface.evaluate(*centre))
    centre_id = max(model.vertices)+face.id+1
    cells = tuple(CurvilinearMaterialCell(model.handle('face', face.id),
        (nodes[i], nodes[(i+1)%len(nodes)], centre_id),
        (uv[i], uv[(i+1)%len(uv)], centre),
        (xyz[i], xyz[(i+1)%len(xyz)], target), ((0, i, use.forward),))
        for i, use in enumerate(face.loop))
    receipt = validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.001)
    validate_prepared_material_curvilinear_cells_binding(model, regions, receipt)
    return receipt


def test_public_axis_cut_mixed_boundaries_have_true_source_fan_coverage():
    model, _, _, _ = square_cells()
    plan = plan_plane_split(model, list(model.faces), Plane((.6, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    apply_plane_split(model, plan, policy='imprint')
    for face in model.faces.values():
        assert fan_receipt(model, face).boundary_correspondence