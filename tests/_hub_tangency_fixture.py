"""Tripod centre section: three equal-radius crossbeams meeting at a hub (pure ANYgeometry).

The crossbeam axes are horizontal at z = zc and meet at the origin at 120 degrees.  The
crossbeams are cut on three vertical central planes (at 60 + 120 p degrees); each plane
carries the elliptic hub seam shared by the two crossbeams meeting on it.  All three
cylinders are mutually tangent at the two hub axis points (0, 0, zc +- R), where the
three seams meet.  Each crossbeam is cut square ``overrun`` metres past the hub seams and
is two half-turn skins split on the arm plane, with radial longitudinal web planes.  The
webs at theta = 0 and pi lie on the top and bottom skin seams and end at the hub axis
points.

Arms 2 and 3 are an exact rational (Pythagorean) rotation of arm 1's doubles, as in the
tripod example builder; ``rotation="float"`` rotates with cos/sin doubles instead, so
every equal-radius contact is tangent only within rounding.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from anygeometry import GeometryModel
from anygeometry.exact_curves import EllipticArc
from anygeometry.surfaces import Cylinder, Plane


def _rotation(k, mode):
    if k == 0:
        return lambda p: np.asarray(p, dtype=float)
    if mode == "float":
        angle = 2 * math.pi / 3 * (1 if k == 1 else -1)
        c, s = math.cos(angle), math.sin(angle)
        return lambda p: np.array((c * p[0] - s * p[1], s * p[0] + c * p[1], float(p[2])))
    t = Fraction(math.sqrt(3.))
    c, s = (1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)
    if k == 2:
        s = -s

    def rotate(p):
        x, y, z = (Fraction(float(value)) for value in p)
        return np.array((float(c * x - s * y), float(s * x + c * y), float(z)))
    return rotate


def hub(radius=6.0, zc=14.0, *, webs_per_turn=48, web_depth=.3, overrun=1.0, webs="all",
        web_arms=(0, 1, 2), rotation="rational", start_angle=-.25):
    """Return the hub model.  ``webs``: "all", "bottom" (theta = pi only), "top" (theta = 0 only)
    or "none"; webs are added only on the arms in ``web_arms``."""
    R, N, h = float(radius), int(webs_per_turn), float(web_depth)
    g = GeometryModel()
    hub_centre = np.array((0., 0., float(zc)))
    stretch = R / math.sin(math.pi / 3.)
    top, bottom = g.add_points([hub_centre + (0., 0., R), hub_centre - (0., 0., R)])

    def hub_direction(p):
        return (np.array((.5, math.sqrt(3.) / 2., 0.)), np.array((-1., 0., 0.)),
                np.array((.5, -math.sqrt(3.) / 2., 0.)))[p]

    seams_hub = [g.add_curve(top, bottom, EllipticArc(tuple(hub_centre), (0., 0., R),
                                                      tuple(stretch * hub_direction(p)), 0., math.pi))
                 for p in range(3)]
    trig = []
    for j in range(N):
        m = min(j, N - j)
        c, s = (1., 0.) if m == 0 else (-1., 0.) if 2 * m == N else (
            (0., 1.) if 4 * m == N else (math.cos(2 * math.pi * m / N), math.sin(2 * math.pi * m / N)))
        trig.append((c, s if j <= N // 2 else -s))
    hub_points = {("axis", 0, False): top, ("axis", N // 2, False): bottom}

    def hub_point(p, j, inner):
        c, s = trig[j]
        scale = (R - h) / R if inner else 1.
        key = ("axis", j, inner) if s == 0. else (p, min(j, N - j), inner)
        if key not in hub_points:
            hub_points[key] = g.add_point(*(hub_centre + scale * (c * np.array((0., 0., R))
                                                                  + abs(s) * stretch * hub_direction(p))))
        return hub_points[key]

    hub_lines = {}

    def hub_line(a, b):
        key = (min(a, b), max(a, b))
        if key not in hub_lines:
            hub_lines[key] = g.add_line(a, b)
        return hub_lines[key]

    selected = {"all": range(N), "bottom": (N // 2,), "top": (0,), "none": ()}[webs]
    direction = np.array((-1., 0., 0.))
    tangent = np.array((0., 1., 0.))
    radial = np.cross(tangent, direction)
    centre = np.array((R / math.tan(math.pi / 3.) + float(overrun), 0., float(zc)))
    u, v = R * radial, R * tangent
    faces = {"skins": [], "webs": []}
    for k in range(3):
        at = _rotation(k, rotation)
        plus_plane, minus_plane = k, (k + 2) % 3
        a, b = g.add_points([at(centre + u), at(centre - u)])
        arcs = (g.add_curve(a, b, EllipticArc(tuple(at(centre)), tuple(at(u)), tuple(at(v)), 0., math.pi)),
                g.add_curve(b, a, EllipticArc(tuple(at(centre)), tuple(at(u)), tuple(at(v)), math.pi, math.pi)))
        ring = []
        for j in range(N):
            c, s = trig[j]
            outer = (a if c > 0 else b) if s == 0. else g.add_point(*at(centre + c * u + s * v))
            inner = g.add_point(*at(centre + (R - h) / R * (c * u + s * v)))
            ring.append((outer, inner, g.add_line(outer, inner)))
        hub_web = []
        for j in range(N):
            c, s = trig[j]
            p = plus_plane if s > 0. else minus_plane
            outer, inner = hub_point(p, j, False), hub_point(p, j, True)
            hub_web.append((outer, inner, hub_line(outer, inner)))
        seams = [g.add_line(ring[0][0], hub_web[0][0]), g.add_line(ring[N // 2][0], hub_web[N // 2][0])]
        support = dict(origin=tuple(at(centre - 7. * direction)), axis=tuple(at(direction)),
                       radial_direction=tuple(at(radial)), radius=R, height=float(centre[0]) + 14.,
                       sweep_angle=2. * math.pi)
        far = (seams_hub[plus_plane], seams_hub[minus_plane])
        for half in (0, 1):
            loop = (arcs[half], seams[1 - half] if half == 0 else seams[0],
                    far[half], seams[0] if half == 0 else seams[1])
            surface = Cylinder(**{**support, "start_angle": half * math.pi + start_angle})
            faces["skins"].append(g.add_face(loop, corners=(0, 1, 2, 3), surface=surface))
        if k not in web_arms:
            continue
        for j in selected:
            o0, i0, e0 = ring[j]
            o1, i1, e1 = hub_web[j]
            root = seams[0] if j == 0 else seams[1] if j == N // 2 else g.add_line(o0, o1)
            origin = g.vertex_position(o0)
            plane = Plane(tuple(origin), tuple(g.vertex_position(i0) - origin),
                          tuple(g.vertex_position(o1) - origin))
            faces["webs"].append(g.add_face((e0, g.add_line(i0, i1), e1, root), corners=(0, 1, 2, 3),
                                            surface=plane))
    return g, faces
