"""Exact sweep audit fast path must preserve unsupported-pair fallbacks."""
from dataclasses import replace

import numpy as np
import pytest

from anygeometry import GeometryModel, query_intersection
from anygeometry.nurbs import NurbsCurve
from anygeometry.entities import OrientedEdge
from anygeometry._translational_pairs import (
    _sweep, separated_sweep_connectors, topology_boundary_uv,
)


def pair(*, weights=None, vector=(3., 0., 0.), reverse=False, crossing=False,
         independent=False):
    model = GeometryModel()
    controls = [((0., 0., 0.), (0., 1., 1.), (0., 2., 0.)),
                ((0., 2., 0.), (0., 3., -1.), (0., 4., 0.))]
    if crossing:
        controls = [((0., -1., -1.), (0., 0., 0.), (0., 1., 1.)),
                    ((0., -1., 1.), (0., 0., 0.), (0., 1., -1.))]
    vertices = {}
    edges = []
    for points in controls:
        curve = NurbsCurve(2, (0., 0., 0., 1., 1., 1.), points, weights)
        if reverse:
            curve = replace(curve, domain=(1., 0.))
        ends = []
        for point in curve.evaluate([0., 1.]):
            key = tuple(point)
            if independent or key not in vertices:
                vertex = model.add_point(*point)
                if not independent:
                    vertices[key] = vertex
            else:
                vertex = vertices[key]
            ends.append(vertex)
        edges.append(model.add_curve(*ends, curve))
    faces = model.extrude(edges, vector)
    return model, tuple(faces)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('vector', [(3., 0., 0.), (-3., 0., 0.)])
@pytest.mark.parametrize('weights', [None, (1., 2., 1.)])
def test_certified_adjacent_sweeps_preserve_exact_boundary_paths(reverse, vector, weights):
    model, faces = pair(reverse=reverse, vector=vector, weights=weights)
    connectors = separated_sweep_connectors(model, *faces)
    assert connectors is not None and len(connectors) == 1
    result = query_intersection(model, *(model.handle('face', face) for face in faces))
    assert result.kind.name == 'OVERLAP_CURVE'
    assert result.certificate.complete
    for component in result.components:
        for face, path in zip(faces, (component.first_parameter_path, component.second_parameter_path)):
            np.testing.assert_allclose(model.evaluate_face_many(face, path), component.witnesses, atol=1e-12)


def test_crossing_and_independent_contact_are_not_certified_disjoint():
    for options in ({'crossing': True}, {'independent': True}):
        model, faces = pair(**options)
        assert separated_sweep_connectors(model, *faces) is None


def test_rounded_control_translation_is_not_an_exact_certificate():
    model, faces = pair(vector=(3., .1, 0.))
    assert any(_sweep(model, face) is None for face in faces)
    assert separated_sweep_connectors(model, *faces) is None


def test_arbitrary_parameterization_and_nonsweep_definitions_fall_back():
    model, faces = pair()
    face = model.faces[faces[0]]
    # The private boundary map must never override a separately declared chart.
    model._faces[face.id] = replace(face, parameterization=face.surface)
    assert separated_sweep_connectors(model, *faces) is None
    edge = face.loop[0].edge
    assert topology_boundary_uv(model, face.id, edge, .5) is None


def test_changed_top_control_refuses_translation_proof():
    model, faces = pair()
    face = model.faces[faces[0]]
    top = model.edges[face.loop[2].edge]
    controls = list(top.curve.controls)
    controls[1] = (controls[1][0], controls[1][1], controls[1][2] + .25)
    model._edges[top.id] = replace(top, curve=replace(top.curve, controls=tuple(controls)))
    assert _sweep(model, face.id) is None
    assert separated_sweep_connectors(model, *faces) is None


def test_reversing_face_loop_preserves_recognition():
    model, faces = pair()
    face = model.faces[faces[0]]
    model._faces[face.id] = replace(face,
        loop=tuple(OrientedEdge(use.edge, not use.forward) for use in reversed(face.loop)),
        corners=(0, 1, 2, 3))
    assert separated_sweep_connectors(model, *faces) is not None


def test_public_boundary_map_validates_and_does_not_mutate():
    model, faces = pair()
    revision = model.revision
    for face in faces:
        for use in model.faces[face].loop:
            parameters = np.linspace(0., 1., 9)
            uv = model.face_boundary_uv(face, use.edge, parameters)
            np.testing.assert_allclose(model.evaluate_face_many(face, uv),
                                       model.sample_edge(use.edge, parameters), atol=1e-12)
    assert model.revision == revision
    with pytest.raises(ValueError):
        model.face_boundary_uv(faces[0], model.faces[faces[0]].loop[0].edge, [1.1])
