"""The B1 driving case: a closed curved-bow half-hull between two frames (generality slice B1-P5).

``build_b1_half_hull`` makes six faces (the bow piece bounded by four plane x NURBS section curves,
deck, bottom, two frames, centreline) sharing twelve edges. The model must audit clean; a plane face
cutting through it without shared topology is reported.
"""
from __future__ import annotations

import pytest

from anygeometry import Plane
from anygeometry.b1_example import build_b1_half_hull


@pytest.fixture(scope="module")
def b1():
    return build_b1_half_hull()


def test_every_edge_bounds_exactly_two_faces(b1):
    model, faces = b1
    assert len(faces) == 6
    uses: dict = {}
    for face in model.faces.values():
        for loop in (face.loop,) + tuple(face.holes):
            for item in loop:
                uses.setdefault(item.edge, []).append(face.id)
    assert len(uses) == 12 and all(len(owners) == 2 for owners in uses.values())
    assert len(model.vertices) == 8                              # V - E + F = 8 - 12 + 6 = 2


def test_the_b1_half_hull_audits_clean(b1):
    model, _faces = b1
    report = model.strict_audit()
    assert report.clean, [(issue.code, issue.details) for issue in report.issues]


def test_a_plane_cutting_through_the_hull_is_reported():
    model, _faces = build_b1_half_hull()
    level = Plane((0., 0., 2.), (1., 0., 0.), (0., 1., 0.))
    corners = [model.add_point(x, y, 2.) for x, y in ((2., 1.), (3., 1.), (3., 9.), (2., 9.))]
    model.add_face([model.add_line(corners[i], corners[(i + 1) % 4]) for i in range(4)], surface=level)
    assert not model.strict_audit().clean
