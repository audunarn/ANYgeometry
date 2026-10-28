"""Occurrence-bound endpoint path parameters of canonical shared source vertices (UNRUN definitions).

Real owner fixture: the paraboloid tensor Bezier cut by z = .5 gives one closed PlaneNurbsCurve; four cropped
quarter edges on the NURBS face share the seam vertex with path parameters 0 and 1.
"""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from anygeometry import GeometryModel, Plane
from anygeometry.regular_source_reference import NurbsReference
from anygeometry.support_curves import PlaneNurbsCurve
from test_section_boundary_sampling import paraboloid


def closed_quarters():
    surface = paraboloid()
    plane = Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.))
    model = GeometryModel()
    base = PlaneNurbsCurve.from_plane(plane, surface)
    vertices = [model.add_point(*base.evaluate(i / 4)) for i in range(4)]
    edges = [model.add_curve(vertices[i], vertices[(i + 1) % 4], replace(base, domain=(i / 4, (i + 1) / 4)))
             for i in range(4)]
    face = model.add_face(edges, surface=surface)
    return model, face, vertices, edges


def test_each_incident_edge_keeps_its_own_endpoint_path_parameter_on_the_closed_seam():
    model, face, vertices, edges = closed_quarters()
    reference = NurbsReference(model, model.faces[face])
    for i, edge in enumerate(edges):
        start = reference.node(edge, Q(0), vertices[i])
        end = reference.node(edge, Q(1), vertices[(i + 1) % 4])
        assert start.path_parameter == Q(i, 4) and end.path_parameter == Q(i + 1, 4)
    seam_start = reference.node(edges[0], Q(0), vertices[0])
    seam_end = reference.node(edges[3], Q(1), vertices[0])
    assert seam_start.path_parameter == 0 and seam_end.path_parameter == 1
    # one topological vertex: identical canonical identity and enclosure
    assert seam_start.identity == seam_end.identity and seam_start.box == seam_end.box
    assert seam_start.identity == reference.vertices[vertices[0]].identity


def test_reversed_interval_at_the_seam_is_a_short_oriented_interval_not_a_wrap_of_the_closed_path():
    model, face, vertices, edges = closed_quarters()
    reference = NurbsReference(model, model.faces[face])
    first_edge = edges[0]
    interior = reference.node(first_edge, Q(1, 8))                       # a true interior station
    seam = reference.node(first_edge, Q(0), vertices[0])
    side = reference.side(first_edge, (Q(1, 8), Q(0)), interior, seam)
    assert side.start.path_parameter - side.end.path_parameter == Q(1, 32)       # not 1 - 1/32
    assert side.pieces[0][0] == 0 and side.pieces[-1][1] == 1
    assert len(side.pieces) <= len(model.edges[first_edge].curve._arcs())


def test_end_of_a_cropped_edge_does_not_inherit_the_other_occurrence():
    model, face, vertices, edges = closed_quarters()
    reference = NurbsReference(model, model.faces[face])
    last = reference.node(edges[3], Q(1), vertices[0])
    first = reference.node(edges[0], Q(0), vertices[0])
    assert last.path_parameter != first.path_parameter
    assert reference.vertices[vertices[0]].path_parameter in (first.path_parameter, last.path_parameter)


def test_a_vertex_that_is_not_that_edge_end_returns_the_canonical_node_unchanged():
    model, face, vertices, edges = closed_quarters()
    reference = NurbsReference(model, model.faces[face])
    wrong = reference.node(edges[0], Q(0), vertices[2])                  # not edges[0].start: no inference
    assert wrong is reference.vertices[vertices[2]]
