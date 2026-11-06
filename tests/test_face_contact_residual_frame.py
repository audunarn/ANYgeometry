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

from anygeometry import GeometryModel, to_dict
from anygeometry.batch_intersections import apply_intersections, plan_intersections
from anygeometry.errors import GeometryError
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


def _support_residual(model, face_id, point):
    support = model.faces[face_id].surface
    uv = np.clip(support.local_uv(point), 0., 1.)
    return float(np.linalg.norm(support.evaluate(float(uv[0]), float(uv[1])) - point))


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
        # The stored residual is the larger of the face-frame residual and the
        # authoritative support's own residual (old formula on the support).
        assert max(float(np.linalg.norm(model.face_point(attachment.target_id, u, v) - station)),
                   _support_residual(model, attachment.target_id, station)) == attachment.max_residual
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


def test_parameterized_face_contact_off_its_support_is_refused():
    """The face-frame residual alone is self-confirming; the support still decides.

    The first cylinder's boundary sits 1.2e-9 off its support (origin moved),
    the second cylinder's 0.8e-9 the other way, so the merged station lies
    about 1.2e-9 from the first support, beyond the 1e-9 contact tolerance.
    A topology Coons parameterization interpolates the boundary and so contains
    the station by construction; the contact must still be refused.
    """
    model = cylinder(1., 1., origin=(1.2e-9, 0., 0.), circumferential_segments=4)
    first = set(model.faces)
    model.insert_model(cylinder(1., 1., origin=(2. + 0.8e-9, 0., 1.), circumferential_segments=4))
    for face_id, face in list(model.faces.items()):
        support = replace(face.surface, start_angle=face.surface.start_angle - np.pi / 4,
                          sweep_angle=2 * np.pi)
        if face_id in first:
            support = replace(support, origin=np.asarray(support.origin) - np.array([1.2e-9, 0., 0.]))
        model.set_face_surface(face_id, support)
        model.set_face_parameterization(face_id, CoonsSurface())
    assert model.validate_topology() == ()
    plan = plan_intersections(model, tuple(model.faces), policy=ConnectionIntent.CONNECT)
    assert plan.face_contacts
    assert all(contact.world_tolerance == 1e-9 for contact in plan.face_contacts)
    with pytest.raises(GeometryError, match='canonical face contact exceeds its qualified tolerance'):
        apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    assert not _face_contact_attachments(model)


def test_unparameterized_topology_coons_face_contact_keeps_old_refusal():
    """A topology-backed Coons support without a parameterization refuses as before."""
    model = GeometryModel()
    first = model.add_plate(model.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
    second = model.add_plate(model.add_points(((1, 1, 0), (2, 1, 0), (2, 1, 1), (1, 1, 1))))
    for face_id in (first, second):
        model.set_face_surface(face_id, CoonsSurface())
        assert model.faces[face_id].parameterization is None
    plan = plan_intersections(model, tuple(model.faces), policy=ConnectionIntent.CONNECT)
    assert plan.face_contacts
    before = to_dict(model)
    with pytest.raises(GeometryError, match='topology-backed Coons evaluation requires GeometryModel.face_point'):
        apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    assert not _face_contact_attachments(model)
    assert to_dict(model) == before
