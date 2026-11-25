"""Acceptance corpus for display tessellation of trimmed faces (plan S0, 2026-11-25).

Every geometry here is a valid ANYgeometry face that a viewer once drew wrongly: the floating
tripod's column skins (exported from ANYfem's menu example) and the adversarial cases of three
Codex review rounds of ANYfem's first, viewer-local trim drawing (local 479c276, not merged).
Builders return a :class:`Case` with the model, the named face ids and the exact description
of each trim, so tests can check a drawing against the true trim rather than against itself.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from anygeometry import GeometryModel
from anygeometry.entities import OrientedEdge
from anygeometry.exact_curves import EllipticArc
from anygeometry.extrusions import EllipseDirectrix
from anygeometry.surfaces import Cone, Cylinder, ExtrudedSurface

DATA = Path(__file__).with_name("data")
FINE_STEP = math.radians(7.5)          # ANYfem Auto 'Fine' angular step
INTERACTION_STEP = math.radians(22.5)  # ANYfem interaction angular step
STEPS = (FINE_STEP, INTERACTION_STEP)


@dataclass
class Case:
    model: GeometryModel
    faces: dict
    info: dict = field(default_factory=dict)


def _arc(g, start, end, centre, first, second, a0, a1):
    return g.add_curve(start, end, EllipticArc(tuple(map(float, centre)), tuple(map(float, first)),
                                               tuple(map(float, second)), float(a0), float(a1 - a0)))


def _cyl_point(g, theta, z, radius=1.0):
    return g.add_point(radius * math.cos(theta), radius * math.sin(theta), z)


def _section(g, start, end, a, b, c, t0, t1, radius=1.0):
    """The circle-of-``radius`` section by the plane z = a x + b y + c, as an exact arc."""
    return _arc(g, start, end, (0., 0., c), (radius, 0., a * radius), (0., radius, b * radius), t0, t1)


def _section_point(g, theta, a, b, c, radius=1.0):
    x, y = radius * math.cos(theta), radius * math.sin(theta)
    return g.add_point(x, y, a * x + b * y + c)


def _set_holes(g, face, holes):
    with g.transaction():
        g._put_entity("face", replace(g.faces[face], holes=tuple(tuple(h) for h in holes)))


# ---------------------------------------------------------------------------------------------
# the floating tripod's skins (ANYfem menu example)
# ---------------------------------------------------------------------------------------------
def tripod_skins() -> Case:
    """18 cylinder skins on full-turn supports that overrun the faces (by up to 7 m)."""
    doc = json.loads((DATA / "display_tripod_skins.json").read_text())
    g = GeometryModel()
    vertex = {k: g.add_point(*v) for k, v in doc["vertices"].items()}
    edge = {}
    for k, e in doc["edges"].items():
        a, b = vertex[str(e["start"])], vertex[str(e["end"])]
        c = e["curve"]
        if c["kind"] == "line":
            edge[k] = g.add_line(a, b)
        else:
            edge[k] = g.add_curve(a, b, EllipticArc(tuple(c["center"]), tuple(c["u"]), tuple(c["v"]),
                                                     c["start"], c["sweep"]))
    faces, supports = {}, {}
    for f in doc["faces"]:
        s = f["cylinder"]
        support = Cylinder(s["origin"], s["axis"], s["radial"], s["radius"], s["height"], s["start"], s["sweep"])
        loop = tuple(OrientedEdge(edge[str(e)], forward) for e, forward in f["loop"])
        made = g.add_face_from_loop(loop, f["corners"], surface=support)
        faces[f"{f['name']} {f['id']}"] = made
        supports[made] = support
    return Case(g, faces, {"supports": supports})


# ---------------------------------------------------------------------------------------------
# a four-cornered face on a cylinder/cone support that runs past it in every direction
# ---------------------------------------------------------------------------------------------
def overrun_patch(kind="cylinder", *, angles=(0.3, 1.4), heights=(1.0, 2.5), start=0.0,
                  sweep=2 * math.pi, overrun=3.0, radii=(2.0, 1.0)) -> Case:
    g = GeometryModel()
    z_low, z_high = heights
    bottom = z_low - overrun
    height = (z_high - z_low) + 2 * overrun
    if kind == "cylinder":
        def radius(z):
            return radii[0]
        support = Cylinder((0., 0., bottom), (0., 0., 1.), (1., 0., 0.), radii[0], height, start, sweep)
    else:
        def radius(z):
            return radii[0] + (z - bottom) / height * (radii[1] - radii[0])
        support = Cone((0., 0., bottom), (0., 0., 1.), (1., 0., 0.), radii[0], radii[1], height, start, sweep)
    a0, a1 = angles

    def point(a, z):
        return g.add_point(radius(z) * math.cos(a), radius(z) * math.sin(a), z)

    p00, p10, p11, p01 = point(a0, z_low), point(a1, z_low), point(a1, z_high), point(a0, z_high)

    def arc(p, q, z):
        r = radius(z)
        return _arc(g, p, q, (0., 0., z), (r * math.cos(a0), r * math.sin(a0), 0.),
                    (-r * math.sin(a0), r * math.cos(a0), 0.), 0., a1 - a0)

    loop = (arc(p00, p10, z_low), g.add_line(p10, p11), arc(p01, p11, z_high), g.add_line(p01, p00))
    face = g.add_face(loop, corners=(0, 1, 2, 3), surface=support)
    return Case(g, {"patch": face}, {"support": support, "angles": angles, "heights": heights})


# ---------------------------------------------------------------------------------------------
# Codex round 1 (348d57a)
# ---------------------------------------------------------------------------------------------
def box_corner_cut() -> Case:
    """r1#1: corners on the box corners, bottom side the section z = .4 sin(theta) (not v = 0)."""
    g = GeometryModel()
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.0, 1.0, 0.0, math.pi)
    p00, p10 = _cyl_point(g, 0., 0.), _cyl_point(g, math.pi, 0.)
    p11, p01 = _cyl_point(g, math.pi, 1.), _cyl_point(g, 0., 1.)
    bottom = _arc(g, p00, p10, (0., 0., 0.), (1., 0., 0.), (0., 1., .4), 0., math.pi)
    top = _arc(g, p01, p11, (0., 0., 1.), (1., 0., 0.), (0., 1., 0.), 0., math.pi)
    face = g.add_face((bottom, g.add_line(p10, p11), top, g.add_line(p01, p00)), corners=(0, 1, 2, 3), surface=support)
    return Case(g, {"cut": face}, {"support": support, "low": lambda t: .4 * np.sin(t), "high": lambda t: np.ones_like(t)})


L_RING = [(.1, .1), (.8, .1), (.8, .4), (.4, .4), (.4, .8), (.1, .8)]


def l_shape() -> Case:
    """r1#2: an L-shaped four-cornered trim on a half cylinder (area .7*.3 + .3*.4)."""
    g = GeometryModel()
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.0, 1.0, 0.0, math.pi)
    ids = [_cyl_point(g, math.pi * u, v) for u, v in L_RING]
    edges = []
    n = len(L_RING)
    for k, (a, b) in enumerate(zip(L_RING, L_RING[1:] + L_RING[:1])):
        if a[1] == b[1]:
            first, second = (k, (k + 1) % n) if a[0] < b[0] else ((k + 1) % n, k)
            u0, u1 = sorted((a[0], b[0]))
            edges.append(_arc(g, ids[first], ids[second], (0., 0., a[1]), (1., 0., 0.), (0., 1., 0.),
                              math.pi * u0, math.pi * u1))
        else:
            edges.append(g.add_line(ids[k], ids[(k + 1) % n]))
    face = g.add_face(tuple(edges), corners=(0, 1, 3, 5), surface=support)
    return Case(g, {"l": face}, {"support": support, "uv_ring": L_RING, "uv_area": .7 * .3 + .3 * .4})


def flat_thin_sweep() -> Case:
    """r1#4: four straight edges at z = 1..2 on a 1e-5 rad, 1000 high cylinder."""
    g = GeometryModel()
    sweep = 1e-5
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.0, 1000.0, 0.0, sweep)
    a, b = _cyl_point(g, 0., 1.), _cyl_point(g, sweep, 1.)
    c, d = _cyl_point(g, sweep, 2.), _cyl_point(g, 0., 2.)
    face = g.add_face((g.add_line(a, b), g.add_line(b, c), g.add_line(c, d), g.add_line(d, a)), corners=(0, 1, 2, 3),
                      surface=support)
    return Case(g, {"thin": face}, {"support": support, "z": (1.0, 2.0)})


def cone_apex(a0_deg=90., a1_deg=120.) -> Case:
    """r1#5 / r2: a sector from a cone's apex to an arc at z = .8 (30 and 220 degrees)."""
    g = GeometryModel()
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 0.0, 1.0, 1.0)
    apex = g.add_point(0., 0., 0.)
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    p, q = _cyl_point(g, a0, .8, .8), _cyl_point(g, a1, .8, .8)
    arc = _arc(g, p, q, (0., 0., .8), (.8, 0., 0.), (0., .8, 0.), a0, a1)
    face = g.add_face((g.add_line(apex, p), arc, g.add_line(q, apex)), surface=support)
    return Case(g, {"sector": face}, {"support": support, "azimuth": (a0, a1), "z_max": .8})


def wavy_band(corners=(0, 1, 2, 3), split=None) -> Case:
    """r1#6/#7: a +-36 degree band across a full-turn cylinder's seam, z = .2 - .1 cos .. .8 + .1 cos."""
    g = GeometryModel()
    support = Cylinder((0., 0., -1.), (0., 0., 1.), (1., 0., 0.), 1.0, 3.0)
    a = math.radians(36.)

    def low(t):
        return .2 - .1 * math.cos(t)

    def high(t):
        return .8 + .1 * math.cos(t)

    p00, p10 = _cyl_point(g, -a, low(-a)), _cyl_point(g, a, low(a))
    p11, p01 = _cyl_point(g, a, high(a)), _cyl_point(g, -a, high(-a))
    lower = (0., 0., .2), (1., 0., -.1), (0., 1., 0.)
    upper = (0., 0., .8), (1., 0., .1), (0., 1., 0.)
    if split is None:
        bottom = (_arc(g, p00, p10, *lower, -a, a),)
    else:
        cuts = [-a, *split, a]
        ids = [p00, *[_cyl_point(g, t, low(t)) for t in split], p10]
        bottom = tuple(_arc(g, ids[k], ids[k + 1], *lower, cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1))
    loop = (*bottom, g.add_line(p10, p11), _arc(g, p01, p11, *upper, -a, a), g.add_line(p01, p00))
    if corners:
        n = len(bottom)
        face = g.add_face(loop, corners=(0, n, n + 1, n + 2), surface=support)
    else:
        face = g.add_face(loop, surface=support)
    return Case(g, {"band": face}, {"support": support, "half_angle": a,
                                    "low": lambda t: .2 - .1 * np.cos(t), "high": lambda t: .8 + .1 * np.cos(t)})


def multi_edge_half_cylinder() -> Case:
    """r1#7: a half cylinder whose lower side is split at normalized u = .1, .444.., .577.., .8."""
    g = GeometryModel()
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.0, 1.0, 0.0, math.pi)
    cuts = [0., .1, .44403494237574187, .5774513333656028, .8, 1.]
    ids = [_cyl_point(g, math.pi * u, 0.) for u in cuts]
    bottom = [_arc(g, ids[k], ids[k + 1], (0., 0., 0.), (1., 0., 0.), (0., 1., 0.), math.pi * cuts[k], math.pi * cuts[k + 1])
              for k in range(len(cuts) - 1)]
    t0, t1 = _cyl_point(g, 0., 1.), _cyl_point(g, math.pi, 1.)
    top = _arc(g, t0, t1, (0., 0., 1.), (1., 0., 0.), (0., 1., 0.), 0., math.pi)
    n = len(bottom)
    face = g.add_face((*bottom, g.add_line(ids[-1], t1), top, g.add_line(t0, ids[0])), corners=(0, n, n + 1, n + 2),
                      surface=support)
    return Case(g, {"split": face}, {"support": support})


def shared_circle() -> Case:
    """r1#8: two patches sharing a circular edge at z = .5; the lower one has a bowed bottom."""
    g = GeometryModel()
    support = Cylinder((0., 0., -2.), (0., 0., 1.), (1., 0., 0.), 1.0, 5.0, math.pi)
    a = math.radians(45.)
    q00, q10 = _cyl_point(g, -a, -1. + .4 * math.cos(a)), _cyl_point(g, a, -1. + .4 * math.cos(a))
    m0, m1 = _cyl_point(g, -a, .5), _cyl_point(g, a, .5)
    t0, t1 = _cyl_point(g, -a, 1.5), _cyl_point(g, a, 1.5)
    shared = _arc(g, m0, m1, (0., 0., .5), (1., 0., 0.), (0., 1., 0.), -a, a)
    bottom = _arc(g, q00, q10, (0., 0., -1.), (1., 0., .4), (0., 1., 0.), -a, a)
    top = _arc(g, t0, t1, (0., 0., 1.5), (1., 0., 0.), (0., 1., 0.), -a, a)
    below = g.add_face((bottom, g.add_line(q10, m1), shared, g.add_line(m0, q00)), corners=(0, 1, 2, 3), surface=support)
    above = g.add_face((shared, g.add_line(m1, t1), top, g.add_line(t0, m0)), corners=(0, 1, 2, 3), surface=support)
    return Case(g, {"below": below, "above": above}, {"support": support, "shared": shared, "half_angle": a})


def trimmed_and_whole_neighbours() -> Case:
    """r2#8: a whole-support face and a trimmed face sharing a 15 degree circular edge."""
    g = GeometryModel()
    span = math.radians(15.)
    whole = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.0, 1.0, 0.0, span)
    wide = Cylinder((0., 0., -2.), (0., 0., 1.), (1., 0., 0.), 1.0, 3.0, -1.0, 3.0)
    b0, b1 = _cyl_point(g, 0., 0.), _cyl_point(g, span, 0.)
    t0, t1 = _cyl_point(g, 0., 1.), _cyl_point(g, span, 1.)
    shared = _arc(g, b0, b1, (0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., span)
    top = _arc(g, t0, t1, (0., 0., 1.), (1., 0., 0.), (0., 1., 0.), 0., span)
    upper = g.add_face((shared, g.add_line(b1, t1), top, g.add_line(t0, b0)), corners=(0, 1, 2, 3), surface=whole)
    # below: bounded by z = -1 + .3 sin(theta)
    l0, l1 = _section_point(g, 0., 0., .3, -1.), _section_point(g, span, 0., .3, -1.)
    low = _section(g, l0, l1, 0., .3, -1., 0., span)
    lower = g.add_face((low, g.add_line(l1, b1), shared, g.add_line(b0, l0)), corners=(0, 1, 2, 3), surface=wide)
    return Case(g, {"whole": upper, "trimmed": lower}, {"shared": shared})


def _cell_outline(cells, origin, side):
    """Closed lattice loops (counter-clockwise outer, clockwise holes) of a union of unit cells."""
    directed = set()
    for i, j in cells:
        corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            if (b, a) in directed:
                directed.remove((b, a))
            else:
                directed.add((a, b))
    following = {}
    for a, b in directed:
        following.setdefault(a, []).append(b)
    loops = []
    while following:
        start = min(following)
        loop, at = [start], start
        while True:
            nxt = following[at].pop()
            if not following[at]:
                del following[at]
            if nxt == start:
                break
            loop.append(nxt)
            at = nxt
        # merge collinear lattice points
        def turns(k):
            (ax, ay), (bx, by), (cx, cy) = loop[k - 1], loop[k], loop[(k + 1) % len(loop)]
            return (bx - ax) * (cy - by) - (by - ay) * (cx - bx) != 0

        merged = [p for k, p in enumerate(loop) if turns(k)]
        loops.append([(origin[0] + side * i, origin[1] + side * j) for i, j in merged])
    return loops


def _cone_uv_face(g, support: Cone, rings):
    """A face whose rings (support uv, lattice-aligned) are iso-v circles and generator lines."""
    def point(u, v):
        return g.add_point(*support.evaluate(u, v))

    def ring_edges(ring):
        ids = [point(u, v) for u, v in ring]
        n = len(ring)
        edges = []
        for k in range(n):
            (ua, va), (ub, vb) = ring[k], ring[(k + 1) % n]
            if va == vb:
                first, second = (k, (k + 1) % n) if ua < ub else ((k + 1) % n, k)
                u0, u1 = sorted((ua, ub))
                r = (1 - va) * support.radius_start + va * support.radius_end
                centre = support.origin + va * support.height * support.axis
                e1, e2 = r * support.radial_direction, r * support.circumferential_direction
                a0 = support.start_angle + u0 * support.sweep_angle
                a1 = support.start_angle + u1 * support.sweep_angle
                made = _arc(g, ids[first], ids[second], centre, e1, e2, a0, a1)
                edges.append(OrientedEdge(made, first == k))
            else:
                edges.append(OrientedEdge(g.add_line(ids[k], ids[(k + 1) % n]), True))
        return tuple(edges)

    outer = ring_edges(rings[0])
    face = g.add_face_from_loop(outer, None, surface=support)
    if len(rings) > 1:
        _set_holes(g, face, [ring_edges(r) for r in rings[1:]])
    return face


def cone_comb_r1() -> Case:
    """r1#9: a cornerless cone trim, the union of ten .12 cells offset by (.5, .5) (UV area .144)."""
    g = GeometryModel()
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.0, 0.5, 1.0)
    rows = {-1: (0,), 0: (0, 1, 2), 1: (-1, 0, 1), 2: (-1, 0, 1)}
    cells = [(i, j) for j, xs in rows.items() for i in xs]
    rings = _cell_outline(cells, (.5, .5), .12)
    face = _cone_uv_face(g, support, rings)
    return Case(g, {"comb": face}, {"support": support, "uv_area": .144, "rings": rings})


# ---------------------------------------------------------------------------------------------
# Codex round 1 #3 and round 2/3 extrusions
# ---------------------------------------------------------------------------------------------
def elliptic_extrusion_band() -> Case:
    """r1#3: an elliptic extrusion (u_range .75..1.25) trimmed to +-36 degrees across the directrix's angle 0."""
    g = GeometryModel()
    support = ExtrudedSurface(EllipseDirectrix((0., 0., 0.), (2., 0., 0.), (0., 1., 0.)), (0., 0., 1.),
                              (.75, 1.25), (0., 1.))
    a = math.radians(36.)

    def point(t, z):
        return g.add_point(2 * math.cos(t), math.sin(t), z)

    p00, p10 = point(-a, .2 - .1 * math.cos(a)), point(a, .2 - .1 * math.cos(a))
    p11, p01 = point(a, .8 + .1 * math.cos(a)), point(-a, .8 + .1 * math.cos(a))
    lower = _arc(g, p00, p10, (0., 0., .2), (2., 0., -.1), (0., 1., 0.), -a, a)
    upper = _arc(g, p01, p11, (0., 0., .8), (2., 0., .1), (0., 1., 0.), -a, a)
    face = g.add_face((lower, g.add_line(p10, p11), upper, g.add_line(p01, p00)), corners=(0, 1, 2, 3), surface=support)
    return Case(g, {"band": face}, {"support": support, "half_angle": a})


def whole_box_extrusion() -> Case:
    """r1#3: a face that is its whole elliptic-extrusion box, u_range (1.05, 1.2)."""
    g = GeometryModel()
    support = ExtrudedSurface(EllipseDirectrix((0., 0., 0.), (2., 0., 0.), (0., 1., 0.)), (0., 0., 1.),
                              (1.05, 1.2), (0., 1.))
    t0, t1 = 2 * math.pi * 1.05, 2 * math.pi * 1.2

    def point(t, z):
        return g.add_point(2 * math.cos(t), math.sin(t), z)

    p00, p10, p11, p01 = point(t0, 0.), point(t1, 0.), point(t1, 1.), point(t0, 1.)
    lower = _arc(g, p00, p10, (0., 0., 0.), (2., 0., 0.), (0., 1., 0.), t0, t1)
    upper = _arc(g, p01, p11, (0., 0., 1.), (2., 0., 0.), (0., 1., 0.), t0, t1)
    face = g.add_face((lower, g.add_line(p10, p11), upper, g.add_line(p01, p00)), corners=(0, 1, 2, 3), surface=support)
    return Case(g, {"whole": face}, {"support": support})


def half_ellipse_extrusion(ratio=1000.0) -> Case:
    """r2: a half ellipse of semiaxes ``ratio`` and 1 extruded along z, trimmed to z = .2 .. .8."""
    g = GeometryModel()
    directrix = EllipseDirectrix((0., 0., 0.), (ratio, 0., 0.), (0., 1., 0.), 0.0, math.pi)
    support = ExtrudedSurface(directrix, (0., 0., 1.), (0., 1.), (0., 1.))
    p00, p10 = g.add_point(ratio, 0., .2), g.add_point(-ratio, 0., .2)
    p01, p11 = g.add_point(ratio, 0., .8), g.add_point(-ratio, 0., .8)
    lower = _arc(g, p00, p10, (0., 0., .2), (ratio, 0., 0.), (0., 1., 0.), 0., math.pi)
    upper = _arc(g, p01, p11, (0., 0., .8), (ratio, 0., 0.), (0., 1., 0.), 0., math.pi)
    face = g.add_face((lower, g.add_line(p10, p11), upper, g.add_line(p01, p00)), corners=(0, 1, 2, 3), surface=support)
    return Case(g, {"half_ellipse": face}, {"support": support, "lower": lower, "upper": upper, "ratio": ratio})


def ellipse_edge(ratio=20.0) -> Case:
    """r1#10: a lone 20:1 half ellipse edge (no face)."""
    g = GeometryModel()
    p, q = g.add_point(ratio, 0., 0.), g.add_point(-ratio, 0., 0.)
    edge = _arc(g, p, q, (0., 0., 0.), (ratio, 0., 0.), (0., 1., 0.), 0., math.pi)
    return Case(g, {}, {"edge": edge})


def mixed_route_extrusion() -> Case:
    """r3: on a unit circular extrusion over 0..pi, a whole-support face z = 0..1 above a trimmed
    neighbour bounded below by z = -1 + .2 sin(theta); they share the semicircle at z = 0."""
    g = GeometryModel()
    directrix = EllipseDirectrix((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0.0, math.pi)
    whole = ExtrudedSurface(directrix, (0., 0., 1.), (0., 1.), (0., 1.))
    wide = ExtrudedSurface(directrix, (0., 0., 1.), (0., 1.), (-2., 1.))
    b0, b1 = g.add_point(1., 0., 0.), g.add_point(-1., 0., 0.)
    t0, t1 = g.add_point(1., 0., 1.), g.add_point(-1., 0., 1.)
    l0, l1 = g.add_point(1., 0., -1.), g.add_point(-1., 0., -1.)
    shared = _arc(g, b0, b1, (0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., math.pi)
    top = _arc(g, t0, t1, (0., 0., 1.), (1., 0., 0.), (0., 1., 0.), 0., math.pi)
    low = _arc(g, l0, l1, (0., 0., -1.), (1., 0., 0.), (0., 1., .2), 0., math.pi)
    upper = g.add_face((shared, g.add_line(b1, t1), top, g.add_line(t0, b0)), corners=(0, 1, 2, 3), surface=whole)
    lower = g.add_face((low, g.add_line(l1, b1), shared, g.add_line(b0, l0)), corners=(0, 1, 2, 3), surface=wide)
    return Case(g, {"whole": upper, "trimmed": lower}, {"shared": shared})


# ---------------------------------------------------------------------------------------------
# Codex round 2 (40f5eaa) and round 3 (479c276)
# ---------------------------------------------------------------------------------------------
def lens(half_width=.05, *, with_neighbour=False) -> Case:
    """r2/r3: a thin lens on an equal-radius unit cone, theta = .5 -+ ``half_width``: lower arc at
    z = .4 split at .5, upper side z = .4 + .5 (cos(theta - .5) - cos(half_width)). With
    ``with_neighbour`` a second face above shares the upper arc (r3: face-local retry cracked it)."""
    g = GeometryModel()
    support = Cone((0., 0., -1.), (0., 0., 1.), (1., 0., 0.), 1.0, 1.0, 3.0)
    t0, tm, t1 = .5 - half_width, .5, .5 + half_width
    a, b, c = .5 * math.cos(.5), .5 * math.sin(.5), .4 - .5 * math.cos(half_width)
    p0, pm, p1 = _cyl_point(g, t0, .4), _cyl_point(g, tm, .4), _cyl_point(g, t1, .4)
    lower = (_arc(g, p0, pm, (0., 0., .4), (1., 0., 0.), (0., 1., 0.), t0, tm),
             _arc(g, pm, p1, (0., 0., .4), (1., 0., 0.), (0., 1., 0.), tm, t1))
    upper = _section(g, p0, p1, a, b, c, t0, t1)
    faces = {"lens": g.add_face((*lower, upper), surface=support)}
    if with_neighbour:
        q0, q1 = _cyl_point(g, t0, .8), _cyl_point(g, t1, .8)
        top = _arc(g, q0, q1, (0., 0., .8), (1., 0., 0.), (0., 1., 0.), t0, t1)
        faces["above"] = g.add_face((upper, g.add_line(p1, q1), top, g.add_line(q0, p0)), corners=(0, 1, 2, 3),
                                    surface=support)
    return Case(g, faces, {"support": support, "theta": (t0, t1), "upper": upper, "z_low": .4})


def steep_cuts() -> Case:
    """r2: a band between z = 100 sin(theta) and 200 + 100 sin(theta) over +-45 degrees."""
    g = GeometryModel()
    support = Cylinder((0., 0., -200.), (0., 0., 1.), (1., 0., 0.), 1.0, 600.0, math.pi)
    a = math.radians(45.)
    p00, p10 = _section_point(g, -a, 0., 100., 0.), _section_point(g, a, 0., 100., 0.)
    p01, p11 = _section_point(g, -a, 0., 100., 200.), _section_point(g, a, 0., 100., 200.)
    lower = _section(g, p00, p10, 0., 100., 0., -a, a)
    upper = _section(g, p01, p11, 0., 100., 200., -a, a)
    face = g.add_face((lower, g.add_line(p10, p11), upper, g.add_line(p01, p00)), corners=(0, 1, 2, 3), surface=support)
    return Case(g, {"steep": face}, {"support": support, "half_angle": a,
                                     "low": lambda t: 100 * np.sin(t), "high": lambda t: 200 + 100 * np.sin(t)})


def winding_fixture() -> Case:
    """r2: start -1.8, sweep 2.1, origin z -10, height 50; a sheared band over theta -1.6 .. .1."""
    g = GeometryModel()
    support = Cylinder((0., 0., -10.), (0., 0., 1.), (1., 0., 0.), 1.0, 50.0, -1.8, 2.1)
    t0, t1 = -1.6, .1
    low, high = (8., -6.5, 0.), (7., -3., math.sqrt(13.25) + .1)
    p00, p10 = _section_point(g, t0, *low), _section_point(g, t1, *low)
    p01, p11 = _section_point(g, t0, *high), _section_point(g, t1, *high)
    face = g.add_face((_section(g, p00, p10, *low, t0, t1), g.add_line(p10, p11),
                       _section(g, p01, p11, *high, t0, t1), g.add_line(p01, p00)), corners=(0, 1, 2, 3), surface=support)
    return Case(g, {"sheared": face}, {"support": support})


def half_cylinder_before_start() -> Case:
    """r2: a trim over -25 .. 5 degrees on a half cylinder starting at 0 (partly before its start)."""
    g = GeometryModel()
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.0, 1.0, 0.0, math.pi)
    a0, a1 = math.radians(-25.), math.radians(5.)
    p00, p10 = _cyl_point(g, a0, .2), _cyl_point(g, a1, .2)
    p11, p01 = _cyl_point(g, a1, .8), _cyl_point(g, a0, .8)
    lower = _arc(g, p00, p10, (0., 0., .2), (1., 0., 0.), (0., 1., 0.), a0, a1)
    upper = _arc(g, p01, p11, (0., 0., .8), (1., 0., 0.), (0., 1., 0.), a0, a1)
    face = g.add_face((lower, g.add_line(p10, p11), upper, g.add_line(p01, p00)), corners=(0, 1, 2, 3), surface=support)
    return Case(g, {"early": face}, {"support": support, "azimuth": (a0, a1)})


def seam_hole(rotation=0) -> Case:
    """r2: on a full-turn unit cylinder an outer ring -.002 .. .004 deg, z .1 .. .9, with a hole
    -.001 .. .002 deg, z .3 .. .7, both across the seam; ``rotation`` picks the outer loop's first edge."""
    g = GeometryModel()
    support = Cylinder((0., 0., -1.), (0., 0., 1.), (1., 0., 0.), 1.0, 3.0)

    def rect(a0, a1, z0, z1):
        a0, a1 = math.radians(a0), math.radians(a1)
        p00, p10, p11, p01 = (_cyl_point(g, a0, z0), _cyl_point(g, a1, z0), _cyl_point(g, a1, z1), _cyl_point(g, a0, z1))
        low = _arc(g, p00, p10, (0., 0., z0), (1., 0., 0.), (0., 1., 0.), a0, a1)
        high = _arc(g, p01, p11, (0., 0., z1), (1., 0., 0.), (0., 1., 0.), a0, a1)
        return [OrientedEdge(low, True), OrientedEdge(g.add_line(p10, p11), True), OrientedEdge(high, False),
                OrientedEdge(g.add_line(p01, p00), True)]

    outer = rect(-.002, .004, .1, .9)
    outer = outer[rotation:] + outer[:rotation]
    inner = rect(-.001, .002, .3, .7)
    hole = tuple(OrientedEdge(e.edge, not e.forward) for e in reversed(inner))
    face = g.add_face_from_loop(tuple(outer), None, surface=support)
    _set_holes(g, face, [hole])
    return Case(g, {"holed": face}, {"support": support,
                                     "uv_area_deg_z": (.006 * .8 - .003 * .4)})


def cone_comb_r3() -> Case:
    """r3: unit equal-radius cone, 1 rad sweep; the union of six 1/16 cells from (.45, .45)."""
    g = GeometryModel()
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.0, 1.0, 1.0, 0.0, 1.0)
    cells = [(-1, 1), (-1, 2), (0, 0), (0, 1), (1, 1), (2, 1)]
    rings = _cell_outline(cells, (.45, .45), 1 / 16)
    face = _cone_uv_face(g, support, rings)
    return Case(g, {"comb": face}, {"support": support, "uv_area": 6 / 256, "rings": rings})


def holed_cone_comb() -> Case:
    """A cone trim with a lattice hole (outer 4x4 cells, hole the middle 2x2) for hole coverage."""
    g = GeometryModel()
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.0, 0.5, 1.0)
    cells = [(i, j) for i in range(4) for j in range(4) if not (1 <= i <= 2 and 1 <= j <= 2)]
    rings = _cell_outline(cells, (.2, .2), .1)
    face = _cone_uv_face(g, support, rings)
    return Case(g, {"ring": face}, {"support": support, "uv_area": 12 * .01, "rings": rings})


ALL = {
    "tripod_skins": tripod_skins,
    "overrun_cylinder": lambda: overrun_patch("cylinder"),
    "overrun_cone": lambda: overrun_patch("cone"),
    "overrun_cylinder_start": lambda: overrun_patch("cylinder", start=-0.4),
    "overrun_cone_seam": lambda: overrun_patch("cone", angles=(0.0, 1.4)),
    "box_corner_cut": box_corner_cut,
    "l_shape": l_shape,
    "flat_thin_sweep": flat_thin_sweep,
    "cone_apex_30": cone_apex,
    "cone_apex_220": lambda: cone_apex(20., 240.),
    "wavy_band": wavy_band,
    "wavy_band_cornerless": lambda: wavy_band(corners=()),
    "wavy_band_split": lambda: wavy_band(split=(-.3, .05, .2)),
    "multi_edge_half_cylinder": multi_edge_half_cylinder,
    "shared_circle": shared_circle,
    "trimmed_and_whole_neighbours": trimmed_and_whole_neighbours,
    "cone_comb_r1": cone_comb_r1,
    "elliptic_extrusion_band": elliptic_extrusion_band,
    "whole_box_extrusion": whole_box_extrusion,
    "half_ellipse_1000": half_ellipse_extrusion,
    "half_ellipse_20": lambda: half_ellipse_extrusion(20.0),
    "ellipse_edge_20": ellipse_edge,
    "mixed_route_extrusion": mixed_route_extrusion,
    "lens_01": lens,
    "lens_0002": lambda: lens(.001),
    "lens_with_neighbour": lambda: lens(with_neighbour=True),
    "steep_cuts": steep_cuts,
    "winding_fixture": winding_fixture,
    "half_cylinder_before_start": half_cylinder_before_start,
    "seam_hole_0": seam_hole,
    "seam_hole_1": lambda: seam_hole(1),
    "seam_hole_2": lambda: seam_hole(2),
    "seam_hole_3": lambda: seam_hole(3),
    "cone_comb_r3": cone_comb_r3,
    "holed_cone_comb": holed_cone_comb,
}
