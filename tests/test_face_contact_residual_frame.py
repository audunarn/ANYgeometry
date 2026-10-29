"""Face-point contacts are checked in the face's own (u, v) frame.

``apply_intersections`` records a VERTEX_ON_FACE attachment for every planned
face-point contact. Its uv comes from ``GeometryModel.face_local_uv``, which
inverts ``face.parameterization`` when the face has one, so the residual must
be measured with the matching evaluator ``GeometryModel.face_point`` and not
with ``face.surface.evaluate`` (the support's own frame). A topology-backed
Coons parameterization on a 360-degree cylinder support (as on the tripod
skins) made every such contact refuse with a metre-sized residual.
"""
from dataclasses import replace

import numpy as np
import pytest

from anygeometry import to_dict
from anygeometry.batch_intersections import apply_intersections, plan_intersections
from anygeometry.generators import cylinder
from anygeometry.structural import AttachmentKind, ConnectionIntent
from anygeometry.surfaces import CoonsSurface, Cylinder


def _touching_quarter_cylinders(*, parameterize):
    """Two 4-segment cylinders touching along x at (1, 0, 1).

    Every quarter face gets a full 360-degree Cylinder support whose angular
    origin sits 45 degrees before the face, so the support's (u, v) frame
    differs from the face's four-corner patch frame. With ``parameterize`` the
    faces also carry a topology-backed ``CoonsSurface`` parameterization.
    """
    model = cylinder(1., 1., circumferential_segments=4)
    model.insert_model(cylinder(1., 1., origin=(2., 0., 1.), circumferential_segments=4))
    for face_id, face in list(model.faces.items()):
        assert isinstance(face.surface, Cylinder) and len(face.corners) == 4
        model.set_face_surface(face_id, replace(
            face.surface, start_angle=face.surface.start_angle - np.pi / 4, sweep_angle=2 * np.pi))
        if parameterize:
            model.set_face_parameterization(face_id, CoonsSurface())
    return model


def _face_contact_attachments(model):
    return [item for item in model.attachments.values()
            if item.kind is AttachmentKind.VERTEX_ON_FACE]


def test_parameterized_face_contact_residual_uses_face_frame():
    model = _touching_quarter_cylinders(parameterize=True)
    plan = plan_intersections(model, tuple(model.faces), policy=ConnectionIntent.CONNECT)
    assert plan.face_contacts
    # The defect: the support frame disagrees with the face frame at the contact.
    contact = plan.face_contacts[0]
    face_id = contact.face_ids[0]
    uv = model.face_local_uv(face_id, contact.position)
    support_gap = float(np.linalg.norm(model.faces[face_id].surface.evaluate(*uv) - contact.position))
    assert support_gap > 0.1
    assert float(np.linalg.norm(model.face_point(face_id, *uv) - contact.position)) <= contact.world_tolerance

    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)

    attachments = _face_contact_attachments(model)
    assert attachments
    for attachment in attachments:
        assert attachment.max_residual <= attachment.tolerance_used
        assert model.faces[attachment.target_id].parameterization is not None
        u, v = (value.start for value in attachment.target_parameters)
        station = model.vertex_position(attachment.source_id)
        assert float(np.linalg.norm(model.face_point(attachment.target_id, u, v) - station)) \
            == attachment.max_residual
    assert model.validate_topology() == ()
    after = to_dict(model)
    assert apply_intersections(model, plan, policy=ConnectionIntent.CONNECT).reused
    assert to_dict(model) == after


def test_unparameterized_face_contact_residual_is_the_support_residual():
    """Faces without a parameterization keep the old residual bit for bit."""
    model = _touching_quarter_cylinders(parameterize=False)
    plan = plan_intersections(model, tuple(model.faces), policy=ConnectionIntent.CONNECT)
    assert plan.face_contacts
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    attachments = _face_contact_attachments(model)
    assert attachments
    for attachment in attachments:
        face = model.faces[attachment.target_id]
        assert face.parameterization is None
        u, v = (value.start for value in attachment.target_parameters)
        station = model.vertex_position(attachment.source_id)
        # The pre-fix formula, evaluated exactly as the old code did.
        assert float(np.linalg.norm(face.surface.evaluate(u, v) - station)) == attachment.max_residual
        assert attachment.max_residual <= attachment.tolerance_used
    assert model.validate_topology() == ()
