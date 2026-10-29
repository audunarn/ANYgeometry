"""World section adjacency must not be interpreted in one incident chart."""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from anygeometry import (
    GeometryError, GeometryModel, Plane, apply_plane_split, plan_plane_split,
    query_material_surface_regions, query_prepared_material_boundary_sampling,
    validate_prepared_material_boundary_sampling_binding,
)
from anygeometry.entities import OrientedEdge
from anygeometry.nurbs import NurbsCurve
from anygeometry.surfaces import CoonsSurface
from anygeometry.serialization import from_dict, to_dict
from anygeometry.definition_binding import definition_checksum
from anygeometry.nurbs_sweep_chart import chart_model
from anygeometry.regular_source_reference import NurbsReference
from anygeometry._canonical_plate_joint import (
    JointRefused, _clear_incident_fragments, plate_joint,
)


def _cut_adjacent_sweeps():
    """Two orthogonal genuine swept faces share their world connector."""
    model = GeometryModel()
    a, b, c, d, e, f = model.add_points((
        (0., 0., 0.), (0., 1., 0.), (5., 1., 0.),
        (5., 0., 0.), (0., 1., 1.), (5., 1., 1.)))
    profile = NurbsCurve(1, (0., 0., 1., 1.), ((0., 0., 0.), (0., 1., 0.)))
    other = NurbsCurve(1, (0., 0., 1., 1.), ((0., 1., 0.), (0., 1., 1.)))
    def translated(curve):
        return replace(curve, controls=tuple((x+5., y, z) for x, y, z in curve.controls))
    bottom = model.add_curve(a, b, profile)
    shared = model.add_line(b, c)
    top = model.add_curve(d, c, translated(profile))
    left = model.add_line(d, a)
    second_bottom = model.add_curve(b, e, other)
    second_right = model.add_line(e, f)
    second_top = model.add_curve(c, f, translated(other))
    def face(rows):
        return model.add_face_from_loop(tuple(OrientedEdge(edge, forward) for edge, forward in rows),
            (0, 1, 2, 3), surface=CoonsSurface())
    first = face(((bottom, True), (shared, True), (top, False), (left, True)))
    second = face(((second_bottom, True), (second_right, True), (second_top, False), (shared, False)))
    plane = Plane((2.5, 0., 0.), (0., 1., 0.), (0., 0., 1.))
    result = apply_plane_split(model, plan_plane_split(model, (first, second), plane), policy='imprint')
    assert len(result.trace_edges) == 2
    return model, tuple(result.trace_edges)


@pytest.mark.parametrize('reload', (False, True))
def test_public_sampling_preserves_foreign_world_section_adjacency(reload):
    model, traces = _cut_adjacent_sweeps()
    if reload:
        model = from_dict(to_dict(model))
    first, second = (model.edges[edge] for edge in traces)
    assert first.curve.surface != second.curve.surface
    shared = {first.start, first.end} & {second.start, second.end}
    assert len(shared) == 1
    vertex = shared.pop()
    for edge_id, foreign_id in (traces, traces[::-1]):
        face_id = min(model.faces_using_edge(edge_id))
        view = chart_model(model, face_id)
        assert view.edges[edge_id] is model.edges[edge_id]
        with pytest.raises(GeometryError, match='another support'):
            view.edges[foreign_id]
        with pytest.raises(GeometryError, match='another support'):
            plate_joint(view, vertex, foreign_id)
        assert plate_joint(model, vertex, edge_id) is None
        assert plate_joint(view, vertex, edge_id) is None
        reference = NurbsReference(view, view.faces[face_id])
        assert vertex in reference.vertices
        regions = query_material_surface_regions(model, [face_id])
        parameters = tuple((model.handle('edge', use.edge), (0., .5, 1.))
                           for use in model.faces[face_id].loop)
        sampled = query_prepared_material_boundary_sampling(model, regions, parameters, chord_error=.0025)
        validate_prepared_material_boundary_sampling_binding(model, regions, sampled)
        assert sampled.chords.intervals


def test_owned_sweep_nonincident_trims_still_refuse_world_joint_box():
    model, traces = _cut_adjacent_sweeps()
    for edge_id in traces:
        curve = model.edges[edge_id].curve
        own = (curve.plane_record, curve.path, definition_checksum(curve.surface))
        # This deliberately oversized world box contains original NON-own
        # boundary trims. Raw Coons markers must not hide those owned trims.
        with pytest.raises(JointRefused):
            _clear_incident_fragments(model, (edge_id,), own,
                (Q(-1), Q(-1), Q(-1)), (Q(6), Q(2), Q(2)), lambda: None)
