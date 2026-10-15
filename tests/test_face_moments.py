"""Face moments in a frame, clipped below a plane (generality slice H1a)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry import GeometryModel, Plane
from anygeometry.entities import OrientedEdge

QUADS = (((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
         ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
         ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)))  # outward


def rotation(yaw, pitch, roll):
    cz, sz, cy, sy, cx, sx = (f(a) for a in (yaw, pitch, roll) for f in (math.cos, math.sin))
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz @ ry @ rx


def box(dims=(2., 3., 4.), turn=np.eye(3), shift=(0., 0., 0.)):
    """A closed outward box ``[0, a] x [0, b] x [0, c]`` turned and shifted: ``(model, faces, corner)``."""
    model = GeometryModel()
    dims, shift = np.asarray(dims), np.asarray(shift)

    def place(ijk):
        return shift + turn @ (dims * np.asarray(ijk, dtype=float))
    p = {(i, j, k): model.add_point(*place((i, j, k))) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    lines, faces = {}, []
    for q in QUADS:
        loop = []
        for i in range(4):
            u, v = q[i], q[(i + 1) % 4]
            key = frozenset((u, v))
            if key not in lines:
                lines[key] = model.add_line(p[u], p[v])
            loop.append(OrientedEdge(lines[key], model.edges[lines[key]].start == p[u]))
        corner = place(q[0])
        faces.append(model.add_face_from_loop(loop, surface=Plane(corner, place(q[1]) - corner, place(q[3]) - corner)))
    return model, faces, place


def test_a_turned_box_has_its_exact_volume_and_centroid():
    from anygeometry.shell_volume import enclosed_volume
    turn = rotation(.3, -.7, 1.1)
    model, faces, place = box(turn=turn, shift=(10., -4., 7.))
    result = enclosed_volume(model, faces)
    assert result["volume"] == pytest.approx(24., rel=1e-12)
    assert result["centroid"] == pytest.approx(place((.5, .5, .5)), abs=1e-11)


def test_the_box_below_a_plane_has_its_exact_volume():
    from anygeometry.face_moments import Frame, shell_moments
    # an upright 2 x 3 x 4 box cut by the plane z = 1.5: volume 9, centroid (1, 1.5, 0.75)
    model, faces, _place = box()
    frame = Frame((0., 0., 1.5), (1., 0., 0.), (0., 1., 0.))
    moments = shell_moments(model, faces, frame, clip=True)
    assert moments[0] == pytest.approx(9., rel=1e-13)
    assert moments[1:] / moments[0] == pytest.approx([1., 1.5, -.75], abs=1e-13)   # frame coordinates


def test_a_tilted_cut_of_a_box_matches_the_closed_form():
    from anygeometry.face_moments import Frame, shell_moments
    # the unit cube below the plane z = 0.5 + 0.25 (x - 0.5) (it cuts x = 0 at 0.375, x = 1 at 0.625):
    # volume 0.5 (symmetric about the cube centre)
    model, faces, _place = box(dims=(1., 1., 1.))
    e1 = np.array([1., 0., .25])
    frame = Frame((.5, .5, .5), e1, (0., 1., 0.))
    moments = shell_moments(model, faces, frame, clip=True)
    assert moments[0] == pytest.approx(.5, rel=1e-13)


def test_a_plane_missing_the_shell_gives_none_or_all_of_it():
    from anygeometry.face_moments import Frame, shell_moments
    model, faces, _place = box()
    above = shell_moments(model, faces, Frame((0., 0., 10.), (1., 0., 0.), (0., 1., 0.)), clip=True)
    below = shell_moments(model, faces, Frame((0., 0., -1.), (1., 0., 0.), (0., 1., 0.)), clip=True)
    assert above[0] == pytest.approx(24., rel=1e-13)
    assert below[0] == pytest.approx(0., abs=1e-13)
