"""Two-sector owner proof: source identity, open boundary and fresh binding."""

from dataclasses import replace
import math

import numpy as np
import pytest

from anygeometry import (
    Cylinder, GeometryModel, OrientedEdge,
    CylinderOpenComponentError, CylinderOpenComponentErrorCode,
    query_cylinder_open_component, validate_cylinder_open_component_binding,
    evaluate_cylinder_open_component_occurrences,
)
from anygeometry.cylinder_patch import CylinderPatchOccurrenceRequest, CylinderPatchStatus
from anygeometry.structural import SheetTopologyPolicy, ConnectivityPolicy


def sectors(*, shared=True, tangent=True):
    model = GeometryModel()
    vertices, edges, faces = {}, {}, []

    def face(index):
        surface = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.),
                           2., 3., index * math.pi / 4, math.pi / 4)
        keys = (f"b{index}", f"b{index+1}", f"t{index+1}", f"t{index}")
        if not shared and index:
            keys = tuple(f"copy-{key}" for key in keys)
        uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
        for key, point in zip(keys, uv):
            if key not in vertices:
                vertices[key] = model.add_point(*surface.evaluate(*point))
        uses = []
        for side, (a, b) in enumerate(zip(keys, keys[1:] + keys[:1])):
            if (b, a) in edges:
                edge, forward = edges[b, a], False
            else:
                if side in (0, 2):
                    midpoint = ((uv[side][0] + uv[(side+1)%4][0]) / 2,
                                (uv[side][1] + uv[(side+1)%4][1]) / 2)
                    middle = model.add_point(*surface.evaluate(*midpoint))
                    edge = model.add_arc(vertices[a], middle, vertices[b])
                else:
                    edge = model.add_line(vertices[a], vertices[b])
                edges[a, b] = edge
                forward = True
            uses.append(OrientedEdge(edge, forward))
        faces.append(model.add_face_from_loop(tuple(uses), corners=(0, 1, 2, 3), surface=surface))

    face(0)
    face(1)
    if tangent:
        from anygeometry import RuledSurface
        bottom = np.array(((2., 0., 0.), (2., -1., 0.)))
        top = np.array(((2., 0., 3.), (2., -1., 3.)))
        surface = RuledSurface(bottom, top)
        keys = ("b0", "t0", "pt", "pb")
        for key, point in zip(keys, ((0, 0), (0, 1), (1, 1), (1, 0))):
            if key not in vertices:
                vertices[key] = model.add_point(*surface.evaluate(*point))
        uses = []
        for a, b in zip(keys, keys[1:] + keys[:1]):
            if (b, a) in edges:
                uses.append(OrientedEdge(edges[b, a], False))
            else:
                edge = model.add_line(vertices[a], vertices[b]); edges[a, b] = edge
                uses.append(OrientedEdge(edge, True))
        faces.append(model.add_face_from_loop(tuple(uses), corners=(0, 1, 2, 3), surface=surface))
    part = model.add_part(name="open cylinders")
    sheet = model.add_sheet(tuple(faces), part_id=part,
        policy=SheetTopologyPolicy(connectivity=ConnectivityPolicy.ALLOW_DISCONNECTED)
        if not shared else SheetTopologyPolicy())
    selected = tuple(model.handle("face_use", use.id) for use in model.face_uses.values()
                     if use.face_id in faces[:2])
    return model, selected


def test_open_component_qualifies_two_sectors_and_tangent_incidence():
    model, selected = sectors()
    before = model.revision
    result = query_cylinder_open_component(model, selected, expected_revision=before)
    assert result.status is CylinderPatchStatus.QUALIFIED
    assert result.complete and len(result.exterior_cycle) == 6
    assert len(result.external_coedges) == 1
    assert model.revision == before
    validate_cylinder_open_component_binding(model, result, selected, expected_revision=before)
    requests = tuple(CylinderPatchOccurrenceRequest(use, .5) for use in result.interface_coedges)
    samples = evaluate_cylinder_open_component_occurrences(
        model, result, requests, face_uses=selected, expected_revision=before).samples
    assert samples[0].identity_key == samples[1].identity_key
    assert np.linalg.norm(np.asarray(samples[0].point) - samples[1].point) < 1e-10


def test_open_component_rejects_coordinate_alias_and_tampered_record():
    model, selected = sectors(shared=False, tangent=False)
    result = query_cylinder_open_component(model, selected, expected_revision=model.revision)
    assert not result.complete and result.diagnostics == ("one_shared_generator_required",)
    model, selected = sectors()
    result = query_cylinder_open_component(model, selected, expected_revision=model.revision)
    with pytest.raises(CylinderOpenComponentError) as caught:
        validate_cylinder_open_component_binding(model, replace(result, shared_edge=None), selected,
                                                 expected_revision=model.revision)
    assert caught.value.code is CylinderOpenComponentErrorCode.INVALID_RESULT


def test_open_component_rejects_stale_and_preserves_cancel_identity():
    model, selected = sectors()
    result = query_cylinder_open_component(model, selected, expected_revision=model.revision)
    with model.transaction():
        model.add_point(10., 10., 10.)
    with pytest.raises(CylinderOpenComponentError) as caught:
        validate_cylinder_open_component_binding(model, result, selected, expected_revision=result.revision)
    assert caught.value.code is CylinderOpenComponentErrorCode.STALE_RESULT
    error = RuntimeError("cancel open sector")
    def cancel(_phase):
        raise error
    with pytest.raises(RuntimeError) as caught:
        query_cylinder_open_component(model, selected, expected_revision=model.revision,
                                      cancellation_check=cancel)
    assert caught.value is error
