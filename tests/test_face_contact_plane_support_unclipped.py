"""S0f: the authoritative-support term of a face-point contact measures a Plane unclipped.

S0c fix1 checks a parameterized face's contact station twice: with ``face_point`` at the
stored (parameterization-frame) uv, and against the authoritative support with the
support's own inverse and evaluator. Fix1 clipped the support uv to the unit square,
which is only valid for a bounded chart. A ``Plane`` is unbounded and its stored frame
need not cover the face (all 540 tripod web planes): clipping moved the evaluation point
up to 24 m away and every web/skin contact refused. The Plane support term is now the
unclipped distance; every other support keeps the clipped formula bit for bit, the stored
uv stay the parameterization uv in [0, 1], and the tolerance is unchanged.

Fixtures ``data/s0f_web_skin_*.json`` are the working models ANYfem hands to
``plan_intersections`` for the ACFD minimal pairs (web face 2 + skin face 1090 of the
MODEL-NS exercise model 66f91041, web face 2 + skin face 121 of model A ce559cd6),
serialized with ``to_dict`` (redesign/s0f_fixture.py). Face 1 is the web (Plane support
+ topology Coons), face 2 the skin (Cylinder support + topology Coons).
"""
import json
import pathlib
from dataclasses import replace

import numpy as np
import pytest

from anygeometry import GeometryModel, from_dict, to_dict
from anygeometry.batch_intersections import apply_intersections, plan_intersections
from anygeometry.errors import GeometryError
from anygeometry.generators import cylinder
from anygeometry.structural import AttachmentKind, ConnectionIntent
from anygeometry.surfaces import CoonsSurface, Cylinder, Plane

DATA = pathlib.Path(__file__).resolve().parent / 'data'
FIXTURES = ('s0f_web_skin_ns_faces_2_1090.json', 's0f_web_skin_a_faces_2_121.json')


def _face_contact_attachments(model):
    return [item for item in model.attachments.values()
            if item.kind is AttachmentKind.VERTEX_ON_FACE]


def _clipped_support_residual(model, face_id, point):
    """Fix1's support term, kept verbatim for every support other than Plane."""
    support = model.faces[face_id].surface
    uv = np.clip(support.local_uv(point), 0., 1.)
    return float(np.linalg.norm(support.evaluate(float(uv[0]), float(uv[1])) - point))


def _unclipped_support_residual(model, face_id, point):
    support = model.faces[face_id].surface
    u, v = support.local_uv(point)
    return float(np.linalg.norm(support.evaluate(float(u), float(v)) - point))


def _expected_residual(model, face_id, uv, point):
    face = model.faces[face_id]
    frame = float(np.linalg.norm(model.face_point(face_id, *uv) - point))
    if type(face.surface) is Plane:
        return max(frame, _unclipped_support_residual(model, face_id, point))
    return max(frame, _clipped_support_residual(model, face_id, point))


def _check_attachments(model):
    attachments = _face_contact_attachments(model)
    assert attachments
    for attachment in attachments:
        face = model.faces[attachment.target_id]
        assert face.parameterization is not None
        uv = tuple(value.start for value in attachment.target_parameters)
        assert all(0. <= value <= 1. for value in uv)
        station = model.vertex_position(attachment.source_id)
        assert attachment.max_residual <= attachment.tolerance_used
        assert attachment.max_residual == _expected_residual(model, attachment.target_id, uv, station)
    return attachments


@pytest.mark.parametrize('name', FIXTURES)
def test_web_skin_contact_at_the_ruling_applies(name):
    model = from_dict(json.loads((DATA / name).read_text()))
    assert type(model.faces[1].surface) is Plane and type(model.faces[2].surface) is Cylinder
    assert all(isinstance(face.parameterization, CoonsSurface) for face in model.faces.values())
    plan = plan_intersections(model, tuple(model.faces), policy=ConnectionIntent.CONNECT)
    assert len(plan.face_contacts) == 1
    contact = plan.face_contacts[0]
    assert contact.world_tolerance == 1e-9
    point = np.asarray(contact.position, dtype=float)
    # The defect: the web plane's stored frame does not cover the contact, so the clipped
    # support term is metres off while the point lies on the infinite plane.
    raw_u, _raw_v = model.faces[1].surface.local_uv(point)
    assert raw_u < -20.
    assert _clipped_support_residual(model, 1, point) > 18.
    assert _unclipped_support_residual(model, 1, point) <= 1e-13

    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)

    attachments = _check_attachments(model)
    assert {model.faces[item.target_id].surface.__class__ for item in attachments} <= {Plane, Cylinder}
    assert model.validate_topology() == ()
    after = to_dict(model)
    assert apply_intersections(model, plan, policy=ConnectionIntent.CONNECT).reused
    assert to_dict(model) == after


def _corner_touching_plates(*, partial_frames, parameterize, z_first=0.):
    """Two perpendicular plates sharing the corner (1, 1, 0).

    With ``partial_frames`` each Plane's unit square covers only part of its face
    (first: a quarter, second: a band that misses the corner), as the tripod webs.
    """
    model = GeometryModel()
    first = model.add_plate(model.add_points(
        ((0, 0, z_first), (1, 0, z_first), (1, 1, z_first), (0, 1, z_first))))
    second = model.add_plate(model.add_points(((1, 1, 0), (2, 1, 0), (2, 1, 1), (1, 1, 1))))
    if partial_frames:
        model.set_face_surface(first, Plane((0, 0, 0), (.5, 0, 0), (0, .5, 0)))
        model.set_face_surface(second, Plane((1.5, 1, 0), (1, 0, 0), (0, 0, 1)))
    else:
        model.set_face_surface(first, Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
        model.set_face_surface(second, Plane((1, 1, 0), (1, 0, 0), (0, 0, 1)))
    if parameterize:
        for face_id in (first, second):
            model.set_face_parameterization(face_id, CoonsSurface())
    return model, first, second


def test_plane_frame_covering_part_of_the_face_applies():
    model, first, second = _corner_touching_plates(partial_frames=True, parameterize=True)
    assert model.validate_topology() == ()
    plan = plan_intersections(model, tuple(model.faces), policy=ConnectionIntent.CONNECT)
    assert len(plan.face_contacts) == 1
    point = np.asarray(plan.face_contacts[0].position, dtype=float)
    for face_id in (first, second):
        raw = model.faces[face_id].surface.local_uv(point)
        assert not all(0. <= value <= 1. for value in raw)
        assert _clipped_support_residual(model, face_id, point) > .4
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    assert {item.target_id for item in _check_attachments(model)} == {first, second}
    assert model.validate_topology() == ()


def test_plane_frame_covering_the_face_is_unchanged():
    """Inside the unit square the clip is a no-op: covering frames keep the fix1 value."""
    model, first, second = _corner_touching_plates(partial_frames=False, parameterize=True)
    plan = plan_intersections(model, tuple(model.faces), policy=ConnectionIntent.CONNECT)
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    for attachment in _check_attachments(model):
        uv = tuple(value.start for value in attachment.target_parameters)
        station = model.vertex_position(attachment.source_id)
        frame = float(np.linalg.norm(model.face_point(attachment.target_id, *uv) - station))
        assert attachment.max_residual == max(
            frame, _clipped_support_residual(model, attachment.target_id, station))


def test_unparameterized_plane_with_partial_frame_keeps_the_old_refusal():
    """Plane faces without a parameterization are out of scope (S0c-DECISION option A):
    their stored uv is the clipped support uv, so the old residual and refusal stand."""
    model, _first, _second = _corner_touching_plates(partial_frames=True, parameterize=False)
    plan = plan_intersections(model, tuple(model.faces), policy=ConnectionIntent.CONNECT)
    assert plan.face_contacts
    before = to_dict(model)
    with pytest.raises(GeometryError, match='canonical face contact exceeds its qualified tolerance'):
        apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    assert to_dict(model) == before


def test_plane_support_term_catches_a_self_confirming_coons_map():
    """A topology Coons map over a boundary 1.2e-9 off its Plane support contains every
    point of the boundary's own plane, so face_point confirms itself; the unclipped
    Plane support term still measures the 1.2e-9 and exceeds the 1e-9 tolerance, also
    outside the part of the face that the stored frame covers.

    The planner itself drops such contacts (measured: no contact for offsets of
    1.2e-9 and more on these plates and on the web fixture), so the residual is checked
    on the residual function that apply uses.
    """
    from anygeometry.batch_intersections import _face_contact_residual

    model, first, _second = _corner_touching_plates(
        partial_frames=True, parameterize=True, z_first=1.2e-9)
    assert model.validate_topology() == ()
    for point in ((1., 1., 1.2e-9), (.8, .9, 1.2e-9), (.2, .3, 1.2e-9)):
        point = np.asarray(point)
        uv = model.face_local_uv(first, point)
        assert float(np.linalg.norm(model.face_point(first, *uv) - point)) <= 1e-15
        residual = _face_contact_residual(model, first, point, uv)
        assert residual == pytest.approx(1.2e-9, rel=1e-6)
        assert residual > 1e-9
    # The same function keeps the fix1 formula for a Cylinder support, bit for bit.
    skin = from_dict(json.loads((DATA / FIXTURES[0]).read_text()))
    point = skin.vertex_position(3)
    uv = skin.face_local_uv(2, point)
    assert _face_contact_residual(skin, 2, point, uv) == max(
        float(np.linalg.norm(skin.face_point(2, *uv) - point)),
        _clipped_support_residual(skin, 2, point))


def test_codex_original_p1_repro_still_refuses():
    """Codex's S0c review P1 case as reported (REVIEW-S0c-REPORT.md): two four-segment
    cylinders with their generator supports and topology Coons parameterizations; the
    first cylinder's boundary origin is (1.2e-9, 0, 0) while its support origin is
    (0, 0, 0); the second cylinder's origin is (2 + 0.8e-9, 0, 1). Planning gives four
    contacts; the canonical station (1.0000000012, 0, 1) lies 1.2e-9 from the first
    support and must be refused, leaving the model unchanged."""
    model = cylinder(1., 1., origin=(1.2e-9, 0., 0.), circumferential_segments=4)
    first = set(model.faces)
    model.insert_model(cylinder(1., 1., origin=(2. + 0.8e-9, 0., 1.), circumferential_segments=4))
    for face_id, face in list(model.faces.items()):
        assert type(face.surface) is Cylinder
        if face_id in first:
            model.set_face_surface(face_id, replace(
                face.surface, origin=np.asarray(face.surface.origin) - np.array([1.2e-9, 0., 0.])))
        model.set_face_parameterization(face_id, CoonsSurface())
    assert model.validate_topology() == ()
    plan = plan_intersections(model, tuple(model.faces), policy=ConnectionIntent.CONNECT)
    assert len(plan.face_contacts) == 4
    assert all(contact.world_tolerance == 1e-9 for contact in plan.face_contacts)
    station = np.array([1. + 1.2e-9, 0., 1.])
    assert any(_clipped_support_residual(model, face_id, station) > 1e-9 for face_id in first)
    before = to_dict(model)
    with pytest.raises(GeometryError, match='canonical face contact exceeds its qualified tolerance'):
        apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    assert not _face_contact_attachments(model)
    assert to_dict(model) == before
