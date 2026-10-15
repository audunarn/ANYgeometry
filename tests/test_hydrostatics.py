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


# ------------------------------------------------------------------ sweeps (H1c)


def test_a_box_floats_at_its_draft_for_its_displacement():
    from anygeometry.hydrostatics import float_at
    model, faces, _place = box((10., 4., 4.))
    result = float_at(model, faces, 10. * 4. * 1.5)
    assert result["zeta"] == pytest.approx(1.5, abs=1e-11)
    assert result["volume"] == pytest.approx(60., rel=1e-12)


def test_a_wall_sided_box_has_the_wall_sided_gz():
    from anygeometry.hydrostatics import gz_curve
    # a 10 x 4 x 4 box at draft 1.5, KG 1.8: wall-sided while tan(heel) B / 2 < T (up to about 36 deg)
    length, beam, depth, draft, kg = 10., 4., 4., 1.5, 1.8
    model, faces, _place = box((length, beam, depth))
    heels = np.radians([0., 5., 10., 20., 30.])
    rows = gz_curve(model, faces, length * beam * draft, (length / 2, beam / 2, kg), heels,
                    reference=(length / 2, beam / 2, 0.))
    kb, bm = draft / 2, beam ** 2 / (12 * draft)
    gm = kb + bm - kg
    for row, heel in zip(rows, heels):
        expected = math.sin(heel) * (gm + .5 * bm * math.tan(heel) ** 2)
        assert row["gz"] == pytest.approx(expected, abs=1e-10)
        # the box heels about its centreline: the waterplane passes through the centreline at the draft
        assert row["result"]["volume"] == pytest.approx(length * beam * draft, rel=1e-12)


def test_the_b1_hydrostatic_table_is_consistent(b1):
    import time
    from anygeometry.hydrostatics import float_at, hydrostatic_table
    model, faces = b1
    drafts = [1.0, 1.5, 2.0, 2.5]
    start = time.perf_counter()
    rows = hydrostatic_table(model, faces, drafts)
    per_waterplane = (time.perf_counter() - start) / len(drafts)
    volumes = [row["volume"] for row in rows]
    assert all(b > a for a, b in zip(volumes, volumes[1:]))
    assert all(row["waterplane_area"] > 0 and row["bm_transverse"] > 0 for row in rows)
    # floating at a tabulated displacement returns its draft
    found = float_at(model, faces, volumes[2], start=1.8)
    assert found["zeta"] == pytest.approx(2.0, abs=1e-9)
    print(f"B1 hydrostatics per waterplane: {per_waterplane:.2f} s")


# ------------------------------------------------------------------ review H1 round 1


def basin(a, b, footprint=(.1, .9, .1, .9), top=1.):
    """A closed outward shell: a quadratic NURBS bottom z = A(x) + B(y) over [0, 1]^2 (Bernstein
    coefficients ``a`` of A and ``b`` of B), vertical sides at the footprint planes (section edges on
    the bottom) and a flat top at ``top``."""
    from anygeometry import GeometryModel
    from anygeometry.b1_example import _parameter
    from anygeometry.entities import OrientedEdge
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.support_curves import PlaneNurbsCurve
    from anygeometry import Plane
    grid = (0., .5, 1.)
    surface = NurbsSurface(2, 2, (0., 0., 0., 1., 1., 1.), (0., 0., 0., 1., 1., 1.),
                           tuple(tuple((x, y, ai + bj) for y, bj in zip(grid, b)) for x, ai in zip(grid, a)))
    return basin_on(surface, footprint, top)


def basin_on(surface, footprint=(.1, .9, .1, .9), top=1.):
    """``basin`` with any bottom surface over [0, 1]^2 in x, y (planes x, y = footprint cut it once)."""
    from anygeometry import GeometryModel
    from anygeometry.b1_example import _parameter
    from anygeometry.entities import OrientedEdge
    from anygeometry.support_curves import PlaneNurbsCurve
    from anygeometry import Plane
    x0, x1, y0, y1 = footprint
    planes = {"w": (Plane((x0, 0., 0.), (0., 1., 0.), (0., 0., 1.)), (1, 0, 0), x0),
              "e": (Plane((x1, 0., 0.), (0., 1., 0.), (0., 0., 1.)), (1, 0, 0), x1),
              "s": (Plane((0., y0, 0.), (1., 0., 0.), (0., 0., 1.)), (0, 1, 0), y0),
              "n": (Plane((0., y1, 0.), (1., 0., 0.), (0., 0., 1.)), (0, 1, 0), y1)}
    curves = {k: PlaneNurbsCurve.from_plane(plane, surface, 0) for k, (plane, _n, _d) in planes.items()}
    corners = {"sw": ("s", "w"), "se": ("s", "e"), "ne": ("n", "e"), "nw": ("n", "w")}
    t = {}
    for name, (first, second) in corners.items():
        t[(first, name)] = _parameter(curves[first], planes[second][1], planes[second][2])
        t[(second, name)] = _parameter(curves[second], planes[first][1], planes[first][2])
    model = GeometryModel()
    low = {name: model.add_point(*curves[first].evaluate(t[(first, name)])) for name, (first, _s) in corners.items()}
    xy = {"sw": (x0, y0), "se": (x1, y0), "ne": (x1, y1), "nw": (x0, y1)}
    high = {name: model.add_point(*xy[name], top) for name in corners}

    def section(key, a_, b_):
        return model.add_curve(low[a_], low[b_], curves[key].subcurve(t[(key, a_)], t[(key, b_)]))
    sec = {"s": section("s", "sw", "se"), "e": section("e", "se", "ne"), "n": section("n", "ne", "nw"),
           "w": section("w", "nw", "sw")}
    tops = {"s": model.add_line(high["sw"], high["se"]), "e": model.add_line(high["se"], high["ne"]),
            "n": model.add_line(high["ne"], high["nw"]), "w": model.add_line(high["nw"], high["sw"])}
    up = {name: model.add_line(low[name], high[name]) for name in corners}

    def o(edge, forward=True):
        return OrientedEdge(edge, forward)
    faces = [
        model.add_face_from_loop([o(sec["w"], False), o(sec["n"], False), o(sec["e"], False), o(sec["s"], False)],
                                 surface=surface),
        model.add_face_from_loop([o(tops[k]) for k in "senw"], surface=Plane((0., 0., top), (1., 0., 0.), (0., 1., 0.))),
        model.add_face_from_loop([o(sec["s"]), o(up["se"]), o(tops["s"], False), o(up["sw"], False)], surface=planes["s"][0]),
        model.add_face_from_loop([o(sec["e"]), o(up["ne"]), o(tops["e"], False), o(up["se"], False)], surface=planes["e"][0]),
        model.add_face_from_loop([o(sec["n"]), o(up["nw"]), o(tops["n"], False), o(up["ne"], False)], surface=planes["n"][0]),
        model.add_face_from_loop([o(sec["w"]), o(up["sw"]), o(tops["w"], False), o(up["nw"], False)], surface=planes["w"][0]),
    ]
    return model, faces


def test_a_wet_patch_inside_a_nurbs_face_is_not_missed():
    from anygeometry.errors import GeometryError
    from anygeometry.shell_volume import closed_shell
    # bottom z = (x - 0.5)^2 + (y - 0.52)^2 - 1e-4: at z = 0 a wet disk of radius 0.01 inside the bottom
    # face (review H1 round 1: both orders returned 0); exact V = pi r^4 / 2, A = pi r^2
    model, faces = basin((.25, -.25, .25), (.2703, -.2497, .2303))
    assert closed_shell(model, faces) == []
    try:
        result = hydrostatics(model, faces, (.5, .5, 0.))
    except GeometryError as error:
        assert "closed waterline" in str(error)                 # refused for this reason only
        return
    assert result["volume"] == pytest.approx(math.pi * 1e-8 / 2, rel=1e-6)
    assert result["waterplane_area"] == pytest.approx(math.pi * 1e-4, rel=1e-6)


def test_a_curved_bottom_below_its_vertices_floats():
    from anygeometry.hydrostatics import float_at
    # bottom z = (x - 0.5)^2 - 0.1: the vertices at z = 0.06, the surface down to -0.1 (review H1 round 1)
    model, faces = basin((.15, -.35, .15), (0., 0., 0.))
    target = hydrostatics(model, faces, (.5, .5, 0.))["volume"]
    # exact: 0.8 * int_{|x - .5| < sqrt(.1)} (0.1 - (x - .5)^2) dx = 0.8 * (4 / 3) 0.1^(3/2)
    assert target == pytest.approx(.8 * 4 / 3 * .1 ** 1.5, rel=1e-9)
    found = float_at(model, faces, target)
    assert found["zeta"] == pytest.approx(0., abs=1e-9)


def test_waterplane_moments_do_not_depend_on_the_point_given_on_the_plane():
    model, faces, _place = box((1., 1., 1.))
    near = hydrostatics(model, faces, (.5, .5, .5))
    far = hydrostatics(model, faces, (1e8, 0., .5))
    for key in ("volume", "waterplane_area", "i_transverse", "i_longitudinal"):
        assert far[key] == pytest.approx(near[key], rel=1e-12)
    assert far["i_longitudinal"] == pytest.approx(1 / 12, rel=1e-12)
    assert far["flotation_centre"] == pytest.approx([.5, .5, .5], abs=1e-9)


def test_the_shell_cache_does_not_alias_clones(monkeypatch):
    from anygeometry import hydrostatics as module
    model, faces, _place = box((1., 1., 1.))
    twin = model.clone(preserve_identity=True)
    assert twin.model_id == model.model_id and twin.revision == model.revision
    calls = []
    original = module.shell_moments

    def counting(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "shell_moments", counting)
    module._shell(model, faces, 24)
    module._shell(twin, faces, 24)                              # a different object: recomputed
    module._shell(model, faces, 24)                             # cached
    assert [m is model for m in calls] == [True, False]
    assert calls[1] is twin


def test_section_points_on_shifted_knots_use_the_exact_definition():
    from anygeometry.face_moments import _section_float
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.support_curves import PlaneNurbsCurve
    from anygeometry import Plane
    # (u, u^2, v - 0.3) on knots shifted to [1e16, 1e16 + 256] (review H1 round 1)
    k0, k1 = 1e16, 1e16 + 256
    surface = NurbsSurface(2, 1, (k0, k0, k0, k1, k1, k1), (0., 0., 1., 1.),
                           (((0., 0., -.3), (0., 0., .7)), ((.5, 0., -.3), (.5, 0., .7)), ((1., 1., -.3), (1., 1., .7))))
    curve = PlaneNurbsCurve.from_plane(Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), surface, 0)
    t = np.array([.37])
    _chart, _rates, world, rates = _section_float(curve, t)
    assert world == pytest.approx(curve.evaluate(t), abs=1e-12)
    assert rates == pytest.approx(curve.derivative(t), abs=1e-9)


# ------------------------------------------------------------------ review H1 round 2


def test_a_narrow_wet_cap_at_a_face_boundary_is_found():
    # bottom z = (x - c)^2 - 0.1, waterplane through (c, 0.1, -0.099999) with normal (0, 1, 1): a narrow
    # wet cap against y = 0.1 whose boundary crossings fall between samples (review H1 round 2)
    c = .503
    model, faces = basin((c * c - .1, c * c - c - .1, (1 - c) ** 2 - .1), (0., 0., 0.))
    result = hydrostatics(model, faces, (c, .1, -.099999), (0., 1., 1.))
    # exact: in the plane frame the cap is {y > 0.1, z < -0.099999 - (y - 0.1), z > (x - c)^2 - 0.1}:
    # with h = 1e-6 and s = y - 0.1, depth h - s - (x - c)^2 over s, x: V = int int (h - s - X^2)_+ ds dX
    h = 1e-6
    volume = 8 / 15 * h ** 2.5                                  # int_X (h - X^2)^2 / 2 dX
    area = 4 / 3 * h ** 1.5 * math.sqrt(2)                      # int_X (h - X^2) dX, in the tilted plane
    assert result["volume"] == pytest.approx(volume, rel=1e-6)
    assert result["waterplane_area"] == pytest.approx(area, rel=1e-6)


def test_an_oblique_plane_given_by_a_far_point_is_the_same_plane():
    model, faces, _place = box((1., 1., 1.))
    normal = (1., -1., 1.)
    for a in (0., 1e8, 1e12):
        result = hydrostatics(model, faces, (a + .5, a + .5, .5), normal)
        assert result["volume"] == pytest.approx(.5, rel=1e-12)


@pytest.mark.parametrize("scale", [1.0, 1e-12])
def test_a_flat_section_root_is_converged(scale):
    from anygeometry.face_moments import Frame, _section_float, face_moments, waterplane_fields
    from anygeometry import GeometryModel, Plane
    from anygeometry.entities import OrientedEdge
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.support_curves import PlaneNurbsCurve
    # S(u, v) = (u, v, u^7 + 1e-48 u - 1e-42): its section with z = 0 is the line x = 1e-6 (review H1 round 2)
    z = [(1. if i == 7 else 0.) + 1e-48 * i / 7 - 1e-42 for i in range(8)]
    # scale 1e-12: the same geometry on a u-knot domain [0, 1e-12] (review H1 round 3)
    surface = NurbsSurface(7, 1, (0.,) * 8 + (scale,) * 8, (0., 0., 1., 1.),
                           tuple(((i / 7, 0., z[i]), (i / 7, 1., z[i])) for i in range(8)))
    flat = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    curve = PlaneNurbsCurve.from_plane(flat, surface, 0)
    t = np.linspace(.1, .9, 5)
    _chart, _rates, world, _w = _section_float(curve, t)
    assert world[:, 0] == pytest.approx(1e-6, rel=1e-9)
    model = GeometryModel()
    start, end = curve.evaluate(0.), curve.evaluate(1.)
    p0, p1 = model.add_point(*start), model.add_point(*end)
    q1, q0 = model.add_point(1., end[1], 0.), model.add_point(1., start[1], 0.)
    section = model.add_curve(p0, p1, curve)
    lines = [model.add_line(p1, q1), model.add_line(q1, q0), model.add_line(q0, p0)]
    face = model.add_face_from_loop([OrientedEdge(section, True)] + [OrientedEdge(e, True) for e in lines],
                                    surface=flat)
    area = abs(face_moments(model, face, Frame.world(), waterplane_fields)[0])
    assert area == pytest.approx((1 - 1e-6) * abs(end[1] - start[1]), rel=1e-12)


def test_a_waterline_loop_outside_the_trimmed_face_does_not_refuse():
    # the bottom's support has a closed waterline at z = 0 (around (0.5, 0.52)), outside this footprint
    model, faces = basin((.25, -.25, .25), (.2703, -.2497, .2303), footprint=(.1, .2, .1, .2))
    result = hydrostatics(model, faces, (.15, .15, 0.))
    assert result["volume"] == 0.0 or result["volume"] == pytest.approx(0., abs=1e-15)


# ------------------------------------------------------------------ review H1 round 3


@pytest.mark.parametrize("c, h", [(.503, 1e-6), (.50625, 1e-5)])
def test_wet_caps_between_equal_samples_are_found(c, h):
    # the cap of round 2, and one placed symmetrically between two boundary samples (review H1 round 3)
    model, faces = basin((c * c - .1, c * c - c - .1, (1 - c) ** 2 - .1), (0., 0., 0.))
    result = hydrostatics(model, faces, (c, .1, -.1 + h), (0., 1., 1.))
    assert result["volume"] == pytest.approx(8 / 15 * h ** 2.5, rel=1e-6)
    assert result["waterplane_area"] == pytest.approx(4 / 3 * h ** 1.5 * math.sqrt(2), rel=1e-6)


@pytest.mark.parametrize("normal, point", [((1., -1., 1.), (1e12 + .5, 1e12 + .5, .5)),
                                           ((1., 3., 7.), (7e12 + .5, .5, -1e12 + .5))])
def test_planes_given_far_away_with_any_normal_are_exact(normal, point):
    # the plane through the unit cube's centre given by a point 1e12 away (review H1 round 3: normal (1, 3, 7))
    model, faces, _place = box((1., 1., 1.))
    assert hydrostatics(model, faces, point, normal)["volume"] == pytest.approx(.5, rel=1e-12)


# ------------------------------------------------------------------ review H1 round 4


def test_an_ill_conditioned_waterline_root_is_exact():
    from anygeometry.nurbs import NurbsSurface
    # bottom z = (u - 0.5)^6 (degree 6, Bernstein controls (-1)^i / 64), waterplane z = 2**-42: the
    # waterline at |u - 0.5| = 2**-7 (review H1 round 4: float roots were off by 1e-7)
    surface = NurbsSurface(6, 1, (0.,) * 7 + (1.,) * 7, (0., 0., 1., 1.),
                           tuple(((i / 6, 0., (-1) ** i / 64), (i / 6, 1., (-1) ** i / 64)) for i in range(7)))
    model, faces = basin_on(surface, (.125, .875, .125, .875))
    level = 2.0 ** -42
    result = hydrostatics(model, faces, (.5, .5, level))
    half = 2.0 ** -7
    assert result["waterplane_area"] == pytest.approx(2 * half * .75, rel=1e-10)
    volume = .75 * (2 * half * level - 2 * half ** 7 / 7)
    assert result["volume"] == pytest.approx(volume, rel=1e-8)


def test_a_longitudinal_direction_near_the_normal_is_refused():
    from anygeometry.errors import GeometryError
    from anygeometry.hydrostatics import waterplane_frame
    model, faces, _place = box((1., 1., 1.))
    with pytest.raises(GeometryError, match="nearly"):
        hydrostatics(model, faces, (.5, .5, .5), (1., 3., 7.), (1.000000001, 3., 7.))
    # an oblique but well-conditioned direction: an orthonormal frame and the exact centre of buoyancy
    frame = waterplane_frame((.5, .5, .5), (1., 3., 7.), (1., 3., 6.))
    assert frame.axes @ frame.axes.T == pytest.approx(np.eye(3), abs=1e-15)
    result = hydrostatics(model, faces, (.5, .5, .5), (1., 3., 7.), (1., 3., 6.))
    assert result["buoyancy_centre"] == pytest.approx([10 / 21, 3 / 7, 157 / 588], abs=1e-12)


# ------------------------------------------------------------------ review H1 round 5


def test_a_nurbs_face_on_far_shifted_knots_is_refused():
    from anygeometry.errors import GeometryError
    from anygeometry.nurbs import NurbsSurface
    # bottom z = (u - .503)^2 - .1 on u-knots [1e16, 1e16 + 256] (review H1 round 5: a wet interval was
    # classified dry by float evaluation, area 0 for 0.0015)
    k0, k1, c = 1e16, 1e16 + 256, .503
    z = (c * c - .1, c * c - c - .1, (1 - c) ** 2 - .1)
    surface = NurbsSurface(2, 1, (k0, k0, k0, k1, k1, k1), (0., 0., 1., 1.),
                           tuple(((i / 2, 0., z[i]), (i / 2, 1., z[i])) for i in range(3)))
    model, faces = basin_on(surface, (.125, .875, .125, .875))
    with pytest.raises(GeometryError, match="knot domain"):
        hydrostatics(model, faces, (.5, .5, -.1 + 1e-6))


def test_a_far_reference_keeps_the_offset_exact():
    from anygeometry.hydrostatics import float_at, hydrostatic_table
    # the unit cube, normal (1, 0, 1), reference 1e12 away on the plane through the origin's diagonal
    # (review H1 round 5: reference + zeta n rounded the offset by 1e-5)
    model, faces, _place = box((1., 1., 1.))
    reference = (1e12, 0., -1e12)
    found = float_at(model, faces, .5, normal=(1., 0., 1.), reference=reference, start=1 / math.sqrt(2) + 1e-5)
    assert found["zeta"] == pytest.approx(1 / math.sqrt(2), abs=1e-12)
    row = hydrostatic_table(model, faces, [1 / math.sqrt(2)], reference=reference, normal=(1., 0., 1.))[0]
    assert row["volume"] == pytest.approx(.5, rel=1e-12)
