"""Hydrostatics of a closed shell at a waterplane (generality slice H1b)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry.hydrostatics import hydrostatics

from test_face_moments import box, rotation


# ------------------------------------------------------------------ boxes against closed forms


@pytest.mark.parametrize("phi", [0.0, 0.2, -0.35])
def test_a_heeled_box_matches_the_closed_forms(phi):
    # a 2 x 3 x 4 box, waterplane z = T + tan(phi) (y - B/2) through (L/2, B/2, T), not reaching the
    # bottom or the deck: a trapezoidal prism
    length, beam, depth, draft = 2., 3., 4., 1.5
    model, faces, _place = box((length, beam, depth))
    result = hydrostatics(model, faces, (length / 2, beam / 2, draft), (0., -math.sin(phi), math.cos(phi)))
    t = math.tan(phi)
    # h(y) = draft + t (y - B/2): int h = B T, int y h, int h^2 / 2 over [0, B]
    area = beam * draft
    y_moment = draft * beam ** 2 / 2 + t * beam ** 3 / 12
    z_moment = (draft ** 2 * beam + t ** 2 * beam ** 3 / 12) / 2
    assert result["volume"] == pytest.approx(length * area, rel=1e-12)
    assert result["buoyancy_centre"] == pytest.approx([length / 2, y_moment / area, z_moment / area], abs=1e-12)
    width = beam / math.cos(phi)
    assert result["waterplane_area"] == pytest.approx(length * width, rel=1e-12)
    assert result["flotation_centre"] == pytest.approx([length / 2, beam / 2, draft], abs=1e-12)
    assert result["i_transverse"] == pytest.approx(length * width ** 3 / 12, rel=1e-10)
    assert result["i_longitudinal"] == pytest.approx(width * length ** 3 / 12, rel=1e-10)
    assert result["bm_transverse"] == pytest.approx(length * width ** 3 / 12 / (length * area), rel=1e-10)


def test_a_turned_box_in_a_level_waterplane_matches_a_polynomial_reference():
    # the box turned (heel, trim and yaw) and floating in z = 1: in box coordinates the waterplane is a
    # general plane over the bottom; depth h(x, y) is linear, so Gauss rules of low order are exact
    dims = np.array([2., 3., 4.])
    turn = rotation(.4, .12, -.18)
    model, faces, place = box(dims, turn, (5., -2., 0.))
    level = 1.2
    result = hydrostatics(model, faces, (0., 0., level))
    # in box coordinates: world z = shift_z + (turn @ p)_z; below the waterplane: (turn @ p)_z < level
    row = turn[2]                                       # world z of a box point p is row @ p
    nodes, weights = np.polynomial.legendre.leggauss(6)
    x = .5 * dims[0] * (nodes + 1)
    y = .5 * dims[1] * (nodes + 1)
    wx, wy = .5 * dims[0] * weights, .5 * dims[1] * weights
    total, moments = 0., np.zeros(3)
    for xi, wxi in zip(x, wx):
        for yj, wyj in zip(y, wy):
            h = (level - row[0] * xi - row[1] * yj) / row[2]       # box-local z of the waterplane
            assert 0 < h < dims[2]
            total += wxi * wyj * h
            moments += wxi * wyj * np.array([xi * h, yj * h, h * h / 2])
    centre = place(moments / total / dims)
    assert result["volume"] == pytest.approx(total, rel=1e-12)
    assert result["buoyancy_centre"] == pytest.approx(centre, abs=1e-11)
    assert result["waterplane_area"] == pytest.approx(dims[0] * dims[1] / abs(row[2]), rel=1e-12)


def test_waterplanes_missing_the_box_give_nothing_or_everything():
    model, faces, _place = box()
    dry = hydrostatics(model, faces, (0., 0., -1.))
    assert dry["volume"] == pytest.approx(0., abs=1e-13) and dry["buoyancy_centre"] is None
    assert dry["waterplane_area"] == pytest.approx(0., abs=1e-13)
    sunk = hydrostatics(model, faces, (0., 0., 10.))
    assert sunk["volume"] == pytest.approx(24., rel=1e-13)
    assert sunk["buoyancy_centre"] == pytest.approx([1., 1.5, 2.], abs=1e-12)
    assert sunk["waterplane_area"] == pytest.approx(0., abs=1e-12)


# ------------------------------------------------------------------ the B1 half-hull


@pytest.fixture(scope="module")
def b1():
    from anygeometry.b1_example import build_b1_half_hull
    model, faces = build_b1_half_hull()
    return model, list(faces.values())


def _solve_u(surface, target, v, u=.4):
    for _ in range(60):
        point = surface.evaluate_many(np.array([[u, v]]))[0]
        s_u, _s_v = surface.derivatives_many(np.array([[u, v]]))
        step = (point[0] - target) / s_u[0][0]
        u -= step
        if abs(step) < 1e-15:
            break
    return u


def b1_reference(draft, trim=0.0, x0=2.5, frames=(1.5, 3.5), bottom=.5, deck=3., order=60):
    """Independent: below the plane z = draft + trim (x - x0), by z-slices (v = z / 4): for each slice
    the x-range inside the frames and below the plane (Newton in u for its ends), int y dx by Gauss in
    u; the slices split where the x-range meets a frame. Volume, centroid and the waterplane area."""
    from anygeometry.b1_example import b1_bow_surface
    surface = b1_bow_surface()
    nodes, weights = np.polynomial.legendre.leggauss(order)
    a, b = frames
    top = min(deck, draft + trim * (b - x0)) if trim >= 0 else min(deck, draft + trim * (a - x0))
    breaks = [bottom, top]
    if trim:
        for x in frames:
            z = draft + trim * (x - x0)
            if bottom < z < top:
                breaks.append(z)
    breaks = sorted(breaks)
    totals = np.zeros(4)
    for z0, z1 in zip(breaks, breaks[1:]):
        for z, wz in zip(.5 * (z1 - z0) * (nodes + 1) + z0, .5 * (z1 - z0) * weights):
            v = z / 4
            lo, hi = a, b
            if trim > 0:
                lo = max(a, x0 + (z - draft) / trim)
            elif trim < 0:
                hi = min(b, x0 + (z - draft) / trim)
            ua, ub = _solve_u(surface, lo, v, .3), _solve_u(surface, hi, v, .5)
            u = .5 * (ub - ua) * (nodes + 1) + ua
            uv = np.stack((u, np.full_like(u, v)), axis=1)
            points = surface.evaluate_many(uv)
            s_u, _s_v = surface.derivatives_many(uv)
            w = .5 * (ub - ua) * weights * s_u[:, 0] * wz
            x, y = points[:, 0], points[:, 1]
            totals += np.array([w @ y, w @ (x * y), w @ (.5 * y * y), w @ (z * y)])
    # waterplane: x in the frames, y up to the hull at z(x) = draft + trim (x - x0); area in the plane
    x = .5 * (b - a) * (nodes + 1) + a
    wx = .5 * (b - a) * weights
    y = np.array([surface.evaluate_many(np.array([[_solve_u(surface, xi, (draft + trim * (xi - x0)) / 4),
                                                    (draft + trim * (xi - x0)) / 4]]))[0][1] for xi in x])
    stretch = math.sqrt(1 + trim * trim)
    area = float(wx @ y) * stretch
    return totals[0], totals[1:] / totals[0], area, (wx, x, y, stretch)


@pytest.mark.parametrize("draft", [1.0, 1.75, 2.5])
def test_the_upright_b1_hull_matches_the_independent_reference(b1, draft):
    from anygeometry.b1_example import build_b1_half_hull
    from anygeometry.shell_volume import enclosed_volume
    model, faces = b1
    result = hydrostatics(model, faces, (2.5, 0., draft))
    volume, centre, area, (wx, x, y, _s) = b1_reference(draft)
    assert result["volume"] == pytest.approx(volume, rel=1e-9)
    assert result["buoyancy_centre"] == pytest.approx(centre, abs=1e-9)
    assert result["waterplane_area"] == pytest.approx(area, rel=1e-9)
    eta_f = float(wx @ (y * y / 2)) / area
    assert result["flotation_centre"][1] == pytest.approx(eta_f, abs=1e-9)
    assert result["i_transverse"] == pytest.approx(float(wx @ (y ** 3 / 3)) - area * eta_f ** 2, rel=1e-8)
    xi_f = float(wx @ (x * y)) / area
    assert result["i_longitudinal"] == pytest.approx(float(wx @ (x * x * y)) - area * xi_f ** 2, rel=1e-8)
    # the same volume as the closed shell cut at the draft (P5c)
    cut_model, cut_faces = build_b1_half_hull(deck=draft)
    cut = enclosed_volume(cut_model, cut_faces.values())
    assert result["volume"] == pytest.approx(cut["volume"], rel=1e-9)
    assert result["buoyancy_centre"] == pytest.approx(cut["centroid"], abs=1e-9)


@pytest.mark.parametrize("trim", [0.15, -0.2])
def test_the_trimmed_b1_hull_matches_the_extended_reference(b1, trim):
    model, faces = b1
    draft = 1.75
    normal = np.array([-trim, 0., 1.]) / math.sqrt(1 + trim * trim)
    result = hydrostatics(model, faces, (2.5, 0., draft), normal)
    volume, centre, area, _w = b1_reference(draft, trim)
    assert result["volume"] == pytest.approx(volume, rel=1e-8)
    assert result["buoyancy_centre"] == pytest.approx(centre, abs=1e-8)
    assert result["waterplane_area"] == pytest.approx(area, rel=1e-8)


def test_the_volume_grows_with_the_draft_at_the_waterplane_area(b1):
    model, faces = b1
    draft, h = 1.75, 1e-2

    def volume(t):
        return hydrostatics(model, faces, (2.5, 0., t))["volume"]
    # Richardson on central differences: O(h^4)
    d1 = (volume(draft + h) - volume(draft - h)) / (2 * h)
    d2 = (volume(draft + h / 2) - volume(draft - h / 2)) / h
    slope = (4 * d2 - d1) / 3
    assert slope == pytest.approx(hydrostatics(model, faces, (2.5, 0., draft))["waterplane_area"], rel=1e-7)
