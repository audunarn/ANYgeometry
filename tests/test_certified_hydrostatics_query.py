"""Certified hydrostatics, slice CH1b: the half-space query (design sections 3-6; acceptance manifest sections 2-4, 6).

Every expected value below is derived in the test from closed forms (rectangles, triangles, polynomial
integration over rectangles, exact rotations); nothing is taken from the implementation under test."""
from __future__ import annotations

import json
import math
from decimal import Decimal
from fractions import Fraction as F

import numpy as np
import pytest

from test_certified_hydrostatics_admission import CUBE_QUADS, _each_stop, build
from anygeometry.certified_hydrostatics import (
    CertifiedHydrostaticsRefusal, HalfSpaceQuery, RefusalCategory, WorkBudget, prepare_certified_hydrostatics,
    query_certified_hydrostatics)

C = RefusalCategory
NAMES = ("A", "Mu", "Mv", "Iuu", "Ivv", "Iuv")
ZERO6 = (F(0),) * 6

# ------------------------------------------------------------------ closed-form helpers


def rect(u0, u1, v0, v1):
    """(A, Mu, Mv, Iuu, Ivv, Iuv) of the rectangle [u0,u1] x [v0,v1]."""
    u0, u1, v0, v1 = F(u0), F(u1), F(v0), F(v1)
    du, dv = u1 - u0, v1 - v0
    su, sv = (u1 ** 2 - u0 ** 2) / 2, (v1 ** 2 - v0 ** 2) / 2
    return (du * dv, su * dv, du * sv, (u1 ** 3 - u0 ** 3) / 3 * dv, du * (v1 ** 3 - v0 ** 3) / 3, su * sv)


def add(*items):
    return tuple(sum(values, F(0)) for values in zip(*items))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def tri_legs(s):
    """Right triangle with legs s along u and v from the origin."""
    s = F(s)
    return (s ** 2 / 2, s ** 3 / 6, s ** 3 / 6, s ** 4 / 12, s ** 4 / 12, s ** 4 / 24)


def scale_uv(q, su, sv):
    """Moments when ``u`` and ``v`` are scaled by ``su`` and ``sv`` (the area element is unchanged)."""
    area, mu, mv, iuu, ivv, iuv = q
    return (area, su * mu, sv * mv, su * su * iuu, sv * sv * ivv, su * sv * iuv)


def shift_origin(q, a, b):
    """Moments about the origin (a, b) of a section whose moments about the old origin are ``q``."""
    area, mu, mv, iuu, ivv, iuv = q
    return (area, mu - a * area, mv - b * area, iuu - 2 * a * mu + a * a * area, ivv - 2 * b * mv + b * b * area,
            iuv - b * mu - a * mv + a * b * area)


# ------------------------------------------------------------------ solids as data (points, faces)


def box_data(lo, hi):
    points = [(hi[0] if i else lo[0], hi[1] if j else lo[1], hi[2] if k else lo[2])
              for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    return points, [[list(q)] for q in CUBE_QUADS]


def prism_data(polygon, z0, z1):
    n = len(polygon)
    points = [(x, y, z0) for x, y in polygon] + [(x, y, z1) for x, y in polygon]
    faces = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return points, [[loop] for loop in faces]


def plate_data(outer, holes, z0, z1):
    """Rectangle plate ``outer = (x0, x1, y0, y1)`` with rectangular through bores ``holes``."""
    def corners(rect_, z):
        x0, x1, y0, y1 = rect_
        return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]       # counter-clockwise seen from +z
    points = corners(outer, z0) + corners(outer, z1)
    ob, ot = list(range(0, 4)), list(range(4, 8))
    top, bottom, bore_walls = [ot], [ob[::-1]], []
    for hole in holes:
        base = len(points)
        points += corners(hole, z0) + corners(hole, z1)
        hb, ht = list(range(base, base + 4)), list(range(base + 4, base + 8))
        top.append(ht[::-1])
        bottom.append(hb)
        bore_walls += [[ht[i], ht[(i + 1) % 4], hb[(i + 1) % 4], hb[i]] for i in range(4)]
    faces = [top, bottom]
    faces += [[[ob[i], ob[(i + 1) % 4], ot[(i + 1) % 4], ot[i]]] for i in range(4)]
    faces += [[wall] for wall in bore_walls]
    return points, faces


def tetra_data():
    return ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
            [[[0, 2, 1]], [[0, 1, 3]], [[1, 2, 3]], [[0, 3, 2]]])


def recess_data():
    """Box [-2,2]^2 x [0,2] with an inverted square pyramid hollow: base [-1,1]^2 at z = 2, apex (0, 0, 1)."""
    points = [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0), (-2, -2, 2), (2, -2, 2), (2, 2, 2), (-2, 2, 2),
              (-1, -1, 2), (1, -1, 2), (1, 1, 2), (-1, 1, 2), (0, 0, 1)]
    faces = [[[4, 5, 6, 7], [11, 10, 9, 8]], [[3, 2, 1, 0]]]
    faces += [[[i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i]] for i in range(4)]
    faces += [[[8 + i, 8 + (i + 1) % 4, 12]] for i in range(4)]
    return points, faces


def saddle_data():
    """-1 <= x, y <= 1, -2 <= z <= |x| - |y| with planar quadrant facets (13 vertices)."""
    points = [(-1, -1, -2), (1, -1, -2), (1, 1, -2), (-1, 1, -2),                 # 0-3 bottom corners
              (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0),                     # 4-7 top corners
              (1, 0, 1), (-1, 0, 1), (0, 1, -1), (0, -1, -1), (0, 0, 0)]          # 8 east, 9 west, 10 north, 11 south, 12 centre
    interior = (0, 0, -1)
    loops = [[12, 8, 6, 10], [12, 10, 7, 9], [12, 9, 4, 11], [12, 11, 5, 8],         # quadrant facets
             [3, 2, 1, 0],                                                           # bottom
             [1, 2, 6, 8, 5], [0, 3, 7, 9, 4], [2, 3, 7, 10, 6], [0, 1, 5, 11, 4]]   # walls x=1, x=-1, y=1, y=-1

    def oriented(loop):
        pts = [tuple(F(c) for c in points[i]) for i in loop]
        normal = [F(0)] * 3
        for i in range(1, len(pts) - 1):
            a = [pts[i][k] - pts[0][k] for k in range(3)]
            b = [pts[i + 1][k] - pts[0][k] for k in range(3)]
            normal = [normal[0] + a[1] * b[2] - a[2] * b[1], normal[1] + a[2] * b[0] - a[0] * b[2],
                      normal[2] + a[0] * b[1] - a[1] * b[0]]
        centre = [sum(p[k] for p in pts) / len(pts) - interior[k] for k in range(3)]
        return loop if sum(normal[k] * centre[k] for k in range(3)) > 0 else loop[::-1]
    return points, [[oriented(loop)] for loop in loops]


def exact_float(value) -> float:
    number = float(value)
    assert F(number) == F(value), f"{value} is not exactly representable"
    return number


def realise(data, point_map=None, reflect=False):
    """(model, face ids, prepared) for ``data``; ``point_map`` must give exactly representable coordinates."""
    points, faces = data
    mapped = [tuple(exact_float(c) for c in (point_map(p) if point_map else p)) for p in points]
    if reflect:
        faces = [[loop[::-1] for loop in face] for face in faces]
    model, ids = build(mapped, faces)
    return model, ids, prepare_certified_hydrostatics(model, ids)


E = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def ask(prepared, n, d, o=(0, 0, 0), axes=((1, 0, 0), (0, 1, 0)), **kwargs):
    run = {key: kwargs.pop(key) for key in ("budget", "cancellation_check") if key in kwargs}
    return query_certified_hydrostatics(prepared, HalfSpaceQuery(n, d, o, axes, **kwargs), **run)


def Q(section):
    return tuple(section.quantity(name).numerator for name in NAMES)


def topology_summary(section):
    topology = section.topology
    return (topology.outer_count, topology.hole_count, sorted(len(holes) for _o, holes in topology.components))


def expect_side(result, side, occupancy, present, quantities=None, loops=None, nn=1):
    section = result.section(side)
    assert section.occupancy == occupancy
    assert section.section == ("PRESENT" if present else "ABSENT")
    if not present:
        assert section.quantities is None and section.topology is None        # never zero-filled
        return
    assert Q(section) == tuple(F(q) for q in quantities)
    assert all(section.quantity(name).norm_squared == nn for name in NAMES)
    if loops is not None:
        assert topology_summary(section) == loops


def expect_case(result, *, occupancy, generic, volume, moment, sides, nn=1):
    assert result.occupancy == occupancy and result.generic is generic and result.degenerate is (not generic)
    assert result.volume.numerator == F(volume) and result.volume.norm_squared is None
    assert tuple(m.numerator for m in result.moments) == tuple(F(j) for j in moment)
    assert [s.side for s in result.sections] == (["plane"] if generic else ["below", "above"])
    for side, spec in sides.items():
        expect_side(result, side, *spec[:3], **(spec[3] if len(spec) > 3 else {}), nn=nn)


# ------------------------------------------------------------------ F1: box 2 x 3 x 5

BOX = (F(2), F(3), F(5))
F1 = realise(box_data((0, 0, 0), (2, 3, 5)))[2]
SQUARE = rect(0, 2, 0, 3)


def f1_values(h):
    a, b, c = BOX
    z = min(max(F(h), 0), c)
    return a * b * z, (a * a * b * z / 2, a * b * b * z / 2, a * b * z * z / 2)


@pytest.mark.parametrize("h, occupancy, generic, sides", [
    (-1, "EMPTY", True, {"plane": ("EMPTY", False)}),
    (0, "EMPTY", False, {"below": ("EMPTY", False), "above": ("PARTIAL", True, SQUARE, (1, 0, [0]))}),
    (1, "PARTIAL", True, {"plane": ("PARTIAL", True, SQUARE, (1, 0, [0]))}),
    (F(5, 2), "PARTIAL", True, {"plane": ("PARTIAL", True, SQUARE, (1, 0, [0]))}),
    (5, "FULL", False, {"below": ("PARTIAL", True, SQUARE, (1, 0, [0])), "above": ("FULL", False)}),
    (6, "FULL", True, {"plane": ("FULL", False)}),
], ids=["z<=-1", "z<=0", "z<=1", "z<=5/2", "z<=5", "z<=6"])
def test_F1_box_heights(h, occupancy, generic, sides):
    result = ask(F1, (0, 0, 1), h)
    volume, moment = f1_values(h)
    specs = {side: (spec[0], spec[1], spec[2] if len(spec) > 2 else None,
                    {"loops": spec[3]} if len(spec) > 3 else {}) for side, spec in sides.items()}
    expect_case(result, occupancy=occupancy, generic=generic, volume=volume, moment=moment, sides=specs)
    if not generic:
        assert (len(result.contacts.vertices), len(result.contacts.edges), len(result.contacts.faces)) == (4, 4, 1)
    else:
        assert result.contacts.vertices == result.contacts.edges == result.contacts.faces == ()


def test_F1_the_box_at_its_top_face_has_no_double_cap():
    result = ask(F1, (0, 0, 1), 5)
    assert result.volume.numerator == 30                       # the full box, the cap is not added twice
    assert result.section("above").quantities is None and result.section("above").occupancy == "FULL"


# ------------------------------------------------------------------ F2: tetrahedron


F2 = realise(tetra_data())[2]


def f2_values(h):
    h = F(h)
    return ((1 - (1 - h) ** 3) / 6,
            ((1 - (1 - h) ** 4) / 24, (1 - (1 - h) ** 4) / 24, (h ** 2 / 2 - 2 * h ** 3 / 3 + h ** 4 / 4) / 2))


@pytest.mark.parametrize("h", [F(1, 4), F(1, 2), F(9, 10)])
def test_F2_tetrahedron_interior_heights(h):
    volume, moment = f2_values(h)
    expect_case(ask(F2, (0, 0, 1), h), occupancy="PARTIAL", generic=True, volume=volume, moment=moment,
                sides={"plane": ("PARTIAL", True, tri_legs(1 - h), {"loops": (1, 0, [0])})})


def test_F2_tetrahedron_at_its_base_face():
    result = ask(F2, (0, 0, 1), 0)
    expect_case(result, occupancy="EMPTY", generic=False, volume=0, moment=(0, 0, 0),
                sides={"below": ("EMPTY", False), "above": ("PARTIAL", True, tri_legs(1), {"loops": (1, 0, [0])})})
    assert (len(result.contacts.vertices), len(result.contacts.edges), len(result.contacts.faces)) == (3, 3, 1)


def test_F2_tetrahedron_at_its_apex_has_a_collapsing_outer_loop():
    result = ask(F2, (0, 0, 1), 1)
    volume, moment = f2_values(1)
    expect_case(result, occupancy="FULL", generic=False, volume=volume, moment=moment,
                sides={"below": ("PARTIAL", True, ZERO6, {"loops": (1, 0, [0])}), "above": ("FULL", False)})
    assert len(result.contacts.vertices) == 1 and result.contacts.edges == () and result.contacts.faces == ()


# ------------------------------------------------------------------ F3: triangular prism cut by x <= s

F3 = realise(prism_data([(0, 0), (2, 0), (0, 1)], 0, 3))[2]
F3_AXES = ((0, 1, 0), (0, 0, 1))                       # u = y, v = z


def f3_values(s):
    s = F(s)
    return 3 * (s - s ** 2 / 4), (3 * (s ** 2 / 2 - s ** 3 / 6), F(3, 2) * (s - s ** 2 / 2 + s ** 3 / 12),
                                  F(3, 2) * 3 * (s - s ** 2 / 4))


def f3_section(s):
    return rect(0, 1 - F(s) / 2, 0, 3)


def test_F3_prism_interior_cut():
    volume, moment = f3_values(1)
    expect_case(ask(F3, (1, 0, 0), 1, axes=F3_AXES), occupancy="PARTIAL", generic=True, volume=volume, moment=moment,
                sides={"plane": ("PARTIAL", True, f3_section(1), {"loops": (1, 0, [0])})})


def test_F3_prism_cut_at_the_x_0_face():
    result = ask(F3, (1, 0, 0), 0, axes=F3_AXES)
    expect_case(result, occupancy="EMPTY", generic=False, volume=0, moment=(0, 0, 0),
                sides={"below": ("EMPTY", False), "above": ("PARTIAL", True, f3_section(0), {"loops": (1, 0, [0])})})
    assert len(result.contacts.faces) == 1


def test_F3_prism_cut_at_the_x_2_edge():
    result = ask(F3, (1, 0, 0), 2, axes=F3_AXES)
    volume, moment = f3_values(2)
    assert volume == 3
    expect_case(result, occupancy="FULL", generic=False, volume=volume, moment=moment,
                sides={"below": ("PARTIAL", True, ZERO6, {"loops": (1, 0, [0])}), "above": ("FULL", False)})
    assert len(result.contacts.vertices) == 2 and len(result.contacts.edges) == 1 and result.contacts.faces == ()


# ------------------------------------------------------------------ F4: extruded U cut by y <= t

U = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]
F4 = realise(prism_data(U, 0, 1))[2]
F4_AXES = ((1, 0, 0), (0, 0, 1))                       # u = x, v = z
F4_HALF = dict(volume=F(3, 2), moment=(F(9, 4), F(3, 8), F(3, 4)), q=rect(0, 3, 0, 1))
F4_TWO = dict(volume=F(5), moment=(F(15, 2), F(9, 2), F(5, 2)), q=add(rect(0, 1, 0, 1), rect(2, 3, 0, 1)))


def test_F4_u_section_below_the_bridge_is_one_component():
    expect_case(ask(F4, (0, 1, 0), F(1, 2), axes=F4_AXES), occupancy="PARTIAL", generic=True,
                volume=F4_HALF["volume"], moment=F4_HALF["moment"],
                sides={"plane": ("PARTIAL", True, F4_HALF["q"], {"loops": (1, 0, [0])})})


def test_F4_u_section_through_the_legs_is_two_components():
    result = ask(F4, (0, 1, 0), 2, axes=F4_AXES)
    expect_case(result, occupancy="PARTIAL", generic=True, volume=F4_TWO["volume"], moment=F4_TWO["moment"],
                sides={"plane": ("PARTIAL", True, F4_TWO["q"], {"loops": (2, 0, [0, 0])})})
    assert F4_TWO["q"][0] == 2


# ------------------------------------------------------------------ F5: bored box

F5 = realise(plate_data((0, 3, 0, 3), [(1, 2, 1, 2)], 0, 1))[2]


def test_F5_bored_box_horizontal_section_is_one_component_with_a_hole():
    q = sub(rect(0, 3, 0, 3), rect(1, 2, 1, 2))
    assert q[0] == 8
    result = ask(F5, (0, 0, 1), F(1, 2))
    expect_case(result, occupancy="PARTIAL", generic=True, volume=4, moment=(6, 6, 1),
                sides={"plane": ("PARTIAL", True, q, {"loops": (1, 1, [1])})})
    section = result.section("plane")
    holes = [loop for loop in section.topology.loops if loop.role == "hole"]
    (outer, hole_ids), = section.topology.components
    assert [loop.identity for loop in holes] == list(hole_ids) and len(outer) == 4 and len(hole_ids[0]) == 4


def test_F5_bored_box_vertical_section_through_the_bore_is_two_rectangles():
    q = add(rect(0, 1, 0, 1), rect(2, 3, 0, 1))
    expect_case(ask(F5, (1, 0, 0), F(3, 2), axes=((0, 1, 0), (0, 0, 1))), occupancy="PARTIAL", generic=True,
                volume=4, moment=(F(11, 4), 6, 2), sides={"plane": ("PARTIAL", True, q, {"loops": (2, 0, [0, 0])})})


# ------------------------------------------------------------------ F6: saddle solid


F6 = realise(saddle_data())[2]


def test_F6_saddle_below_is_one_loop_above_is_two_loops_with_equal_quantities():
    q = (F(2), F(0), F(0), F(1), F(1, 3), F(0))
    result = ask(F6, (0, 0, 1), 0)
    expect_case(result, occupancy="PARTIAL", generic=False, volume=F(22, 3), moment=(0, 0, F(-47, 6)),
                sides={"below": ("PARTIAL", True, q, {"loops": (1, 0, [0])}),
                       "above": ("PARTIAL", True, q, {"loops": (2, 0, [0, 0])})})
    assert len(result.contacts.vertices) == 5 and result.contacts.edges == () and result.contacts.faces == ()
    assert Q(result.section("below")) == Q(result.section("above"))
    centre = [v for v in result.contacts.vertices if prepared_position(F6, v) == (0, 0, 0)]
    assert len(centre) == 1


def prepared_position(prepared, vertex):
    return tuple(int(c) for c in prepared.snapshot.positions[vertex])


def test_F6_saddle_away_from_the_saddle_height():
    result = ask(F6, (0, 0, 1), F(-1, 2))                 # generic: the section |x| >= |y| - 1/2 is connected
    assert result.generic and topology_summary(result.section("plane"))[0] == 1
    result = ask(F6, (0, 0, 1), F(1, 2))                  # |x| >= |y| + 1/2: two wedges
    assert result.generic and topology_summary(result.section("plane")) == (2, 0, [0, 0])


# ------------------------------------------------------------------ F7: pyramidal recess


F7 = realise(recess_data())[2]


def test_F7_pyramidal_recess_apex_gives_a_collapsing_hole_above_only():
    q = rect(-2, 2, -2, 2)
    result = ask(F7, (0, 0, 1), 1)
    expect_case(result, occupancy="PARTIAL", generic=False, volume=16, moment=(0, 0, 8),
                sides={"below": ("PARTIAL", True, q, {"loops": (1, 0, [0])}),
                       "above": ("PARTIAL", True, q, {"loops": (1, 1, [1])})})
    assert len(result.contacts.vertices) == 1 and result.contacts.edges == () and result.contacts.faces == ()
    above = result.section("above")
    assert [loop.role for loop in above.topology.loops].count("hole") == 1


def test_F7_above_the_apex_the_hole_is_a_real_hole():
    result = ask(F7, (0, 0, 1), F(3, 2))
    assert result.generic and topology_summary(result.section("plane")) == (1, 1, [1])
    # the hole is the square of half-width 1/2 at z = 3/2 (pyramid cross-section)
    assert Q(result.section("plane"))[0] == 16 - 1


# ------------------------------------------------------------------ F8: two bores, collinear vertices


F8 = realise(plate_data((-4, 4, -2, 2), [(-3, -2, -1, 1), (2, 3, -1, 1)], 0, 1))[2]


def test_F8_two_bores_with_collinear_vertices_resolve_without_symbolic_degeneracy():
    q = sub(sub(rect(-4, 4, -2, 2), rect(-3, -2, -1, 1)), rect(2, 3, -1, 1))
    assert q == (28, 0, 0, F(436, 3), F(124, 3), 0)
    result = ask(F8, (0, 0, 1), F(1, 2))
    expect_case(result, occupancy="PARTIAL", generic=True, volume=14, moment=(0, 0, F(7, 2)),
                sides={"plane": ("PARTIAL", True, q, {"loops": (1, 2, [2])})})


# ------------------------------------------------------------------ F9: unit cube, six axis planes


CUBE = realise(box_data((0, 0, 0), (1, 1, 1)))[2]


@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("variant", ["cyclic", "left-handed", "scaled", "shifted"])
def test_F9_unit_cube_axis_planes_and_orientation_factor(k, sign, variant):
    n = tuple(sign if i == k else 0 for i in range(3))
    e1, e2 = E[(k + 1) % 3], E[(k + 2) % 3]
    origin = (0, 0, 0)
    q = rect(0, 1, 0, 1)
    if variant == "left-handed":
        e1, e2 = e2, e1
    elif variant == "scaled":
        e1, e2 = tuple(2 * c for c in e1), tuple(3 * c for c in e2)
        q = scale_uv(rect(0, 1, 0, 1), 2, 3)          # the area element is the true one: A stays 1
    elif variant == "shifted":
        e1, e2 = tuple(2 * c for c in e1), tuple(3 * c for c in e2)
        origin = (F(1, 4), F(1, 4), F(1, 4))
        a, b = sum(x * y for x, y in zip(e1, origin)), sum(x * y for x, y in zip(e2, origin))
        q = shift_origin(scale_uv(rect(0, 1, 0, 1), 2, 3), a, b)
    result = ask(CUBE, n, sign * F(1, 2), origin, (e1, e2))
    moment = [F(1, 4)] * 3
    moment[k] = F(1, 8) if sign > 0 else F(3, 8)
    expect_case(result, occupancy="PARTIAL", generic=True, volume=F(1, 2), moment=moment,
                sides={"plane": ("PARTIAL", True, q, {"loops": (1, 0, [0])})})


def test_F9_a_slanted_plane_with_an_irrational_norm_has_rational_numerators():
    result = ask(CUBE, (1, 1, 0), F(1, 2), (0, 0, 0), ((1, -1, 0), (0, 0, 1)), forms=("exact", "binary64", "decimal"),
                 absolute_accuracy={"A": 1e-9})
    expect_case(result, occupancy="PARTIAL", generic=True, volume=F(1, 8), moment=(F(1, 48), F(1, 48), F(1, 16)),
                sides={"plane": ("PARTIAL", True, (F(1), F(0), F(1, 2), F(1, 12), F(1, 3), F(0)))}, nn=2)
    area = result.section("plane").quantity("A")
    lo, hi = area.binary64
    assert lo < hi and F(lo) ** 2 < F(1, 2) < F(hi) ** 2                      # A = 1/sqrt(2)
    assert math.nextafter(lo, math.inf) == hi
    low, high, width = area.decimal
    assert Decimal(low) ** 2 < Decimal(1) / 2 < Decimal(high) ** 2 and F(Decimal(high)) - F(Decimal(low)) <= width
    assert area.accuracy == F(1e-9) and area.accuracy_met is True


# ------------------------------------------------------------------ Q02: transformations


def matmul(m, v):
    return tuple(sum(F(m[i][j]) * v[j] for j in range(3)) for i in range(3))


def matmat(a, b):
    return tuple(tuple(sum(F(a[i][k]) * F(b[k][j]) for k in range(3)) for j in range(3)) for i in range(3))


RZ = ((F(3, 5), F(-4, 5), 0), (F(4, 5), F(3, 5), 0), (0, 0, 1))
RX = ((1, 0, 0), (0, F(5, 13), F(-12, 13)), (0, F(12, 13), F(5, 13)))
RY = ((F(8, 17), 0, F(15, 17)), (0, 1, 0), (F(-15, 17), 0, F(8, 17)))
IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
MIRROR = ((-1, 0, 0), (0, 1, 0), (0, 0, 1))
TRANSFORMS = {
    "translation": (IDENTITY, 1, (7, -3, 11), False),
    "rotation-345": (RZ, 5, (0, 0, 0), False),
    "rotation-3axis": (matmat(matmat(RZ, RX), RY), 1105, (2, -5, 9), False),
    "reflection": (MIRROR, 1, (0, 0, 0), True),
    "reflection+translation": (MIRROR, 1, (7, -3, 11), True),
}
Q02_BASES = {
    "F1": (box_data((0, 0, 0), (2, 3, 5)), (0, 0, 1), F(5, 2), (0, 0, 0), ((1, 0, 0), (0, 1, 0)),
           f1_values(F(5, 2)), SQUARE, (1, 0, [0])),
    "F2": (tetra_data(), (0, 0, 1), F(1, 4), (0, 0, 0), ((1, 0, 0), (0, 1, 0)),
           f2_values(F(1, 4)), tri_legs(F(3, 4)), (1, 0, [0])),
    "F4": (prism_data(U, 0, 1), (0, 1, 0), F(2), (0, 0, 0), F4_AXES,
           (F4_TWO["volume"], F4_TWO["moment"]), F4_TWO["q"], (2, 0, [0, 0])),
}


@pytest.mark.parametrize("transform", sorted(TRANSFORMS))
@pytest.mark.parametrize("base", sorted(Q02_BASES))
def test_Q02_transformed_solid_plane_and_basis_give_transformed_results(base, transform):
    data, n, d, o, axes, (volume, moment), q, loops = Q02_BASES[base]
    r, s, t, reflect = TRANSFORMS[transform]
    t = tuple(F(c) for c in t)

    def place(p):
        moved = matmul(r, tuple(s * F(c) for c in p))
        return tuple(moved[i] + t[i] for i in range(3))
    _model, _ids, prepared = realise(data, place, reflect)
    n2 = matmul(r, tuple(F(c) for c in n))
    d2 = s * F(d) + sum(n2[i] * t[i] for i in range(3))
    o2 = place(o)
    axes2 = tuple(matmul(r, tuple(F(c) for c in axis)) for axis in axes)
    result = ask(prepared, n2, d2, o2, axes2)
    # V s^3, A s^2, first moments s^3, second moments s^4, J' = R (s^4 J) + (s^3 V) t
    scaled_q = (q[0] * s ** 2, q[1] * s ** 3, q[2] * s ** 3, q[3] * s ** 4, q[4] * s ** 4, q[5] * s ** 4)
    rj = matmul(r, tuple(F(j) * s ** 4 for j in moment))
    moved_j = tuple(rj[i] + volume * s ** 3 * t[i] for i in range(3))
    expect_case(result, occupancy="PARTIAL", generic=True, volume=volume * s ** 3, moment=moved_j,
                sides={"plane": ("PARTIAL", True, scaled_q, {"loops": loops})})


def test_Q02_moment_origin_change_shifts_the_second_moments_exactly():
    base = ask(F4, (0, 1, 0), 2, axes=F4_AXES)
    shifted_origin = (F(3, 2), F(-7, 4), F(1, 3))
    shifted = ask(F4, (0, 1, 0), 2, shifted_origin, F4_AXES)
    a, b = shifted_origin[0], shifted_origin[2]                  # e1 = x, e2 = z
    expect_case(shifted, occupancy="PARTIAL", generic=True, volume=F4_TWO["volume"], moment=F4_TWO["moment"],
                sides={"plane": ("PARTIAL", True, shift_origin(F4_TWO["q"], a, b), {"loops": (2, 0, [0, 0])})})
    assert Q(base.section("plane")) == F4_TWO["q"]


# ANYstability's convention: R = Ry(-trim) Rx(heel), half-space R[2] . r <= -origin_z, basis R[0], R[1]


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def poly_mul(p, q):
    out = {}
    for (i, j), a in p.items():
        for (k, l), b in q.items():
            out[(i + k, j + l)] = out.get((i + k, j + l), F(0)) + a * b
    return out


def poly_add(*items):
    out = {}
    for p in items:
        for key, value in p.items():
            out[key] = out.get(key, F(0)) + value
    return out


def poly_scale(p, factor):
    return {key: value * factor for key, value in p.items()}


def integrate_rectangle(p, a, b):
    return sum((c * a ** (i + 1) / (i + 1) * b ** (j + 1) / (j + 1) for (i, j), c in p.items()), F(0))


def tilted_box_expectation(a, b, n, d, o, e1, e2):
    """Box [0,a]x[0,b]x[0,c] cut by n.r <= d through its four vertical edges (n_z > 0): V, J, Q-numerators."""
    nx, ny, nz = n
    z = {(0, 0): d / nz, (1, 0): -nx / nz, (0, 1): -ny / nz}
    one = {(0, 0): F(1)}
    x, y = {(1, 0): F(1)}, {(0, 1): F(1)}

    def affine(e):
        return poly_add(poly_scale(x, e[0]), poly_scale(y, e[1]), poly_scale(z, e[2]),
                        poly_scale(one, -(e[0] * o[0] + e[1] * o[1] + e[2] * o[2])))
    u, v = affine(e1), affine(e2)
    nn = nx * nx + ny * ny + nz * nz
    weight = nn / nz
    integrand = [one, u, v, poly_mul(u, u), poly_mul(v, v), poly_mul(u, v)]
    quantities = tuple(weight * integrate_rectangle(f, a, b) for f in integrand)
    volume = integrate_rectangle(z, a, b)
    moment = (integrate_rectangle(poly_mul(x, z), a, b), integrate_rectangle(poly_mul(y, z), a, b),
              integrate_rectangle(poly_mul(z, z), a, b) / 2)
    return volume, moment, quantities, nn


@pytest.mark.parametrize("heel, trim", [(0.1, 0.0), (0.5, 0.0), (0.1, 0.02), (0.5, 0.02), (-0.3, 0.02)])
def test_Q02_anystability_convention_plane_at_finite_heel_and_trim(heel, trim):
    matrix = rot_y(-trim) @ rot_x(heel)
    n = tuple(float(c) for c in matrix[2])
    e1, e2 = tuple(float(c) for c in matrix[0]), tuple(float(c) for c in matrix[1])
    origin_z = -2.5
    o = (0.0, 0.0, origin_z)
    result = ask(F1, n, -origin_z, o, (e1, e2))
    volume, moment, quantities, nn = tilted_box_expectation(
        F(2), F(3), tuple(F(c) for c in n), F(-origin_z), tuple(F(c) for c in o),
        tuple(F(c) for c in e1), tuple(F(c) for c in e2))
    expect_case(result, occupancy="PARTIAL", generic=True, volume=volume, moment=moment,
                sides={"plane": ("PARTIAL", True, quantities, {"loops": (1, 0, [0])})}, nn=nn)
    loop, = result.section("plane").topology.loops
    assert len(loop.identity) == 4                                        # the four vertical edges


# ------------------------------------------------------------------ the exact derivative check


def solve(matrix, rhs):
    size = len(rhs)
    rows = [list(map(F, matrix[i])) + [F(rhs[i])] for i in range(size)]
    for col in range(size):
        pivot = next(r for r in range(col, size) if rows[r][col] != 0)
        rows[col], rows[pivot] = rows[pivot], rows[col]
        rows[col] = [x / rows[col][col] for x in rows[col]]
        for r in range(size):
            if r != col and rows[r][col] != 0:
                rows[r] = [x - rows[r][col] * y for x, y in zip(rows[r], rows[col])]
    return [rows[i][size] for i in range(size)]


DERIVATIVE_CASES = {
    "F1-z": (F1, box_data((0, 0, 0), (2, 3, 5))[0], (0, 0, 1), ((1, 0, 0), (0, 1, 0))),
    "F1-z2": (F1, box_data((0, 0, 0), (2, 3, 5))[0], (0, 0, 2), ((1, 0, 0), (0, 1, 0))),
    "F1-slanted-3": (F1, box_data((0, 0, 0), (2, 3, 5))[0], (1, 2, 2), ((1, 0, 0), (0, 1, 0))),
    "F1-slanted-irrational": (F1, box_data((0, 0, 0), (2, 3, 5))[0], (1, 1, 1), ((1, 0, 0), (0, 1, 0))),
    "F2-z": (F2, tetra_data()[0], (0, 0, 1), ((1, 0, 0), (0, 1, 0))),
    "F2-slanted": (F2, tetra_data()[0], (1, 2, 3), ((1, 0, 0), (0, 1, 0))),
    "F3-x": (F3, prism_data([(0, 0), (2, 0), (0, 1)], 0, 3)[0], (1, 0, 0), F3_AXES),
    "F3-slanted": (F3, prism_data([(0, 0), (2, 0), (0, 1)], 0, 3)[0], (2, 1, 1), ((1, 0, 0), (0, 1, 0))),
    "F4-slanted": (F4, prism_data(U, 0, 1)[0], (1, 3, 2), ((1, 0, 0), (0, 1, 0))),
    "F5-slanted": (F5, plate_data((0, 3, 0, 3), [(1, 2, 1, 2)], 0, 1)[0], (1, 1, 5), ((1, 0, 0), (0, 1, 0))),
    "F6-slanted": (F6, saddle_data()[0], (1, 2, 3), ((1, 0, 0), (0, 1, 0))),
    "F6-z": (F6, saddle_data()[0], (0, 0, 1), ((1, 0, 0), (0, 1, 0))),
    "F7-slanted": (F7, recess_data()[0], (1, 2, 7), ((1, 0, 0), (0, 1, 0))),
    "F8-slanted": (F8, plate_data((-4, 4, -2, 2), [(-3, -2, -1, 1), (2, 3, -1, 1)], 0, 1)[0], (1, 1, 9),
                   ((1, 0, 0), (0, 1, 0))),
}


@pytest.mark.parametrize("case", sorted(DERIVATIVE_CASES))
def test_exact_derivative_dV_dd_equals_A_over_norm(case):
    prepared, vertices, n, axes = DERIVATIVE_CASES[case]
    heights = sorted({sum(F(n[i]) * F(p[i]) for i in range(3)) for p in vertices})
    gap, low = max((heights[i + 1] - heights[i], heights[i]) for i in range(len(heights) - 1))
    ds = [low + gap * F(k, 11) for k in (1, 3, 5, 8)]
    d_probe = low + gap * F(6, 11)
    values = [ask(prepared, n, d, axes=axes).volume.numerator for d in ds]
    c = solve([[d ** p for p in range(4)] for d in ds], values)            # V(d) = c0 + c1 d + c2 d^2 + c3 d^3 exactly
    probe = ask(prepared, n, d_probe, axes=axes)
    assert probe.volume.numerator == sum(c[p] * d_probe ** p for p in range(4))          # a cubic on the interval
    derivative = c[1] + 2 * c[2] * d_probe + 3 * c[3] * d_probe ** 2
    section = probe.section("plane").quantity("A")
    assert derivative == section.numerator / section.norm_squared                       # A / |n| = Q / |n|^2


# ------------------------------------------------------------------ forms, bounds, accuracy


def test_binary64_exact_double_has_zero_width_and_meets_any_accuracy():
    result = ask(F1, (0, 0, 1), F(5, 2), forms=("exact", "binary64"),
                 absolute_accuracy={"V": 1e-300, "A": 1e-300, "Jz": 1e-300})
    assert result.volume.binary64 == (15.0, 15.0)
    assert result.volume.accuracy_met is True
    area = result.section("plane").quantity("A")
    assert area.binary64 == (6.0, 6.0) and area.accuracy_met is True
    assert result.section("plane").quantity("Mu").accuracy_met is None          # no accuracy requested


def test_binary64_enclosure_of_a_non_double_is_the_adjacent_doubles():
    result = ask(CUBE, (0, 0, 1), F(1, 3), forms=("exact", "binary64"))
    assert result.volume.numerator == F(1, 3)
    lo, hi = result.volume.binary64
    assert F(lo) < F(1, 3) < F(hi) and math.nextafter(lo, math.inf) == hi
    assert (lo, hi) == (math.nextafter(1 / 3, 0.0), 1 / 3) or (lo, hi) == (1 / 3, math.nextafter(1 / 3, 1.0))
    wide = ask(CUBE, (0, 0, 1), F(1, 3), forms=("binary64",), absolute_accuracy={"V": 1e-15})
    assert wide.volume.accuracy_met is True and wide.volume.numerator is not None        # V is always exact


def test_binary64_too_fine_an_accuracy_for_a_non_double_is_unachievable():
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        ask(CUBE, (0, 0, 1), F(1, 3), forms=("exact", "binary64"), absolute_accuracy={"V": 1e-30})
    assert info.value.category is C.ACCURACY_UNACHIEVABLE and info.value.phase == "query"
    assert "work" in dir(info.value) and info.value.work["operations"] > 0
    # the decimal form remains available for the same request
    ok = ask(CUBE, (0, 0, 1), F(1, 3), forms=("exact", "decimal"), absolute_accuracy={"V": 1e-30})
    low, high, width = ok.volume.decimal
    assert F(Decimal(low)) <= F(1, 3) <= F(Decimal(high)) and F(Decimal(high)) - F(Decimal(low)) <= width
    assert len(low.split(".")[1]) >= 30 and ok.volume.accuracy_met is True


@pytest.mark.parametrize("accuracy", [1e-3, 1e-9, 1e-20, 0.5, 3.0])
def test_decimal_enclosures_meet_the_requested_width(accuracy):
    result = ask(CUBE, (1, 1, 0), F(1, 2), (0, 0, 0), ((1, -1, 0), (0, 0, 1)), forms=("exact", "decimal"),
                 absolute_accuracy={"V": accuracy, "A": accuracy, "Mv": accuracy})
    for item, exact in ((result.volume, F(1, 8)),):
        low, high, width = item.decimal
        assert width == F(accuracy) and F(Decimal(low)) <= exact <= F(Decimal(high)) and F(Decimal(high)) - F(Decimal(low)) <= width
    area = result.section("plane").quantity("A")                              # 1/sqrt(2), irrational
    low, high, width = area.decimal
    assert Decimal(low) ** 2 < Decimal(1) / 2 < Decimal(high) ** 2
    assert F(Decimal(high)) - F(Decimal(low)) <= width


def test_decimal_enclosure_negative_and_zero_values():
    result = ask(F6, (0, 0, 1), 0, forms=("decimal",), absolute_accuracy={"Jz": 0.01, "Mu": 0.01})
    low, high, _w = result.moments[2].decimal                                   # Jz = -47/6 = -7.8333...
    assert low == "-7.84" and high == "-7.83"
    low, high, _w = result.section("below").quantity("Mu").decimal                 # exactly 0
    assert low == high == "0.00"
    assert result.section("below").quantity("Mu").numerator is None             # exact not requested for waterplane


def test_out_of_binary64_range_is_refused_per_form_and_underflow_is_enclosed():
    from anygeometry.certified_hydrostatics._forms import binary64_enclosure, build_quantity
    from anygeometry.certified_hydrostatics._work import Meter
    meter = Meter(None)
    huge = F(10) ** 400
    for numerator, norm_squared in ((huge, None), (-huge, None), (huge, F(2)), (-huge, F(7, 3))):
        with pytest.raises(CertifiedHydrostaticsRefusal) as info:
            binary64_enclosure(numerator, norm_squared, meter, "V")
        assert info.value.category is C.RESULT_OUT_OF_RANGE
    # the largest double is representable; the next value up is not
    largest = F(sys_float_max())
    assert binary64_enclosure(largest, None, meter) == (sys_float_max(), sys_float_max())
    with pytest.raises(CertifiedHydrostaticsRefusal):
        binary64_enclosure(largest + 1, None, meter)
    # underflow is an enclosure, not a refusal: [0, smallest subnormal]
    assert binary64_enclosure(F(1, 10 ** 400), None, meter) == (0.0, 5e-324)
    assert binary64_enclosure(-F(1, 10 ** 400), None, meter) == (-5e-324, 0.0)
    # the exact and decimal forms of the same quantity remain available
    kept = build_quantity("V", huge, None, ("exact", "decimal"), F(10) ** 300, meter, always_exact=True)
    assert kept.numerator == huge and kept.binary64 is None and kept.decimal[0] == kept.decimal[1] == str(10 ** 400)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        build_quantity("V", huge, None, ("binary64",), None, meter)
    assert info.value.category is C.RESULT_OUT_OF_RANGE


def sys_float_max() -> float:
    import sys
    return sys.float_info.max


# ------------------------------------------------------------------ query validation


@pytest.mark.parametrize("kwargs, expected", [
    (dict(n=(0, 0, 0), d=0), C.INVALID_QUERY),
    (dict(n=(0, 0, 1), d=0, axes=((1, 0, 0), (1, 0, 0))), C.INVALID_QUERY),
    (dict(n=(0, 0, 1), d=0, axes=((1, 0, 0), (0, 0, 1))), C.INVALID_QUERY),            # (e1 x e2) . n == 0
    (dict(n=(0, 0, 1), d=0, axes=((0, 0, 0), (0, 1, 0))), C.INVALID_QUERY),
    (dict(n=(0, 0, math.nan), d=0), C.NONFINITE_INPUT),
    (dict(n=(0, 0, 1), d=math.inf), C.NONFINITE_INPUT),
    (dict(n=(0, 0, 1), d=0, o=(0, math.nan, 0)), C.NONFINITE_INPUT),
    (dict(n=(0, 0, 1), d=0, axes=((1, 0, 0), (0, math.inf, 0))), C.NONFINITE_INPUT),
    (dict(n=(0, 0), d=0), C.INVALID_QUERY),
    (dict(n=(0, 0, 1), d="1"), C.INVALID_QUERY),
    (dict(n=(0, 0, 1), d=0, forms=()), C.INVALID_QUERY),
    (dict(n=(0, 0, 1), d=0, forms=("exact", "exact")), C.INVALID_QUERY),
    (dict(n=(0, 0, 1), d=0, forms=("float",)), C.INVALID_QUERY),
    (dict(n=(0, 0, 1), d=0, absolute_accuracy={"volume": 1e-9}), C.INVALID_QUERY),
    (dict(n=(0, 0, 1), d=0, absolute_accuracy={"V": 0.0}), C.INVALID_QUERY),
    (dict(n=(0, 0, 1), d=0, absolute_accuracy={"V": math.nan}), C.NONFINITE_INPUT),
], ids=lambda value: value.name if isinstance(value, C) else "")
def test_an_invalid_query_is_refused_when_it_is_built(kwargs, expected):
    arguments = dict(kwargs)
    n, d = arguments.pop("n"), arguments.pop("d")
    o = arguments.pop("o", (0, 0, 0))
    axes = arguments.pop("axes", ((1, 0, 0), (0, 1, 0)))
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        HalfSpaceQuery(n, d, o, axes, **arguments)
    assert info.value.category is expected and info.value.phase == "query"


def test_the_query_is_exact_frozen_and_canonical():
    query = HalfSpaceQuery((0.1, 0, 1), 0.3, forms=("decimal", "exact"), absolute_accuracy={"J": 1e-6, "A": 1e-9})
    assert query.normal[0] == F(0.1) and query.offset == F(0.3) and query.forms == ("exact", "decimal")
    assert [name for name, _w in query.absolute_accuracy] == ["Jx", "Jy", "Jz", "A"]
    with pytest.raises(Exception):
        query.offset = 1
    assert json.loads(json.dumps(query.as_dict()))["normal"][0] == [F(0.1).numerator, F(0.1).denominator]
    assert query == HalfSpaceQuery((0.1, 0, 1), 0.3, forms=("exact", "decimal"), absolute_accuracy={"A": 1e-9, "J": 1e-6})


# ------------------------------------------------------------------ result object and canonical proof


def test_the_proof_is_canonical_json_following_the_schema():
    query = HalfSpaceQuery((0, 0, 1), F(1, 3), forms=("exact", "binary64", "decimal"), absolute_accuracy={"A": 1e-6})
    result = query_certified_hydrostatics(F2, query)
    document = json.loads(result.proof.decode("utf-8"))
    assert set(document) == {"artifact", "model", "convention", "preparation_policy", "query", "result"}
    assert json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") == result.proof
    assert document["artifact"] == json.loads(json.dumps(F2.implementation.as_dict()))
    assert document["model"] == json.loads(json.dumps(F2.model_identity()))
    assert document["preparation_policy"] == F2.policy.as_dict()
    assert document["query"] == query.as_dict()
    assert document["convention"]["units"] == "m" and document["convention"]["convention_version"] == "1"
    assert document["result"] == json.loads(json.dumps(result.claim()))
    claim = document["result"]
    assert set(claim) == {"occupancy", "generic", "contacts", "volume", "moments", "sections", "work"}
    assert set(claim["sections"]) == {"plane"} and set(claim["moments"]) == {"Jx", "Jy", "Jz"}
    section = claim["sections"]["plane"]
    assert set(section) == {"occupancy", "section", "quantities", "topology"}
    assert set(section["quantities"]) == set(NAMES)
    area = section["quantities"]["A"]
    assert set(area) == {"exact", "binary64", "decimal", "accuracy", "accuracy_met"}
    legs = (1 - F(1, 3)) ** 2 / 2                                 # tetrahedron section A(h) = (1-h)^2 / 2
    assert area["exact"]["numerator"] == [legs.numerator, legs.denominator]
    assert area["exact"]["norm_squared"] == [1, 1] and area["accuracy"] == [F(1e-6).numerator, F(1e-6).denominator]
    assert float.fromhex(area["binary64"]["lo"]) <= float(legs) <= float.fromhex(area["binary64"]["hi"])
    assert document["result"]["volume"]["exact"]["norm_squared"] is None
    assert claim["work"]["operations"] == result.work["operations"] > 0
    assert result.proof == query_certified_hydrostatics(F2, query).proof                  # deterministic bytes
    other = query_certified_hydrostatics(F2, HalfSpaceQuery((0, 0, 1), F(1, 2)))
    assert other.proof != result.proof


def test_the_proof_binds_the_query_even_when_the_quantities_agree():
    box = F1
    first = ask(box, (0, 0, 1), 6)
    second = ask(box, (0, 0, 1), 7)                      # fully submerged: identical quantities
    assert first.volume == second.volume and first.occupancy == second.occupancy
    assert first.proof != second.proof
    assert json.loads(first.proof)["query"]["offset"] == [6, 1] and json.loads(second.proof)["query"]["offset"] == [7, 1]


def test_a_degenerate_plane_proof_has_both_sides_and_the_contacts():
    result = ask(F1, (0, 0, 1), 5)
    claim = json.loads(result.proof)["result"]
    assert set(claim["sections"]) == {"below", "above"} and claim["generic"] is False
    assert claim["contacts"]["faces"] and len(claim["contacts"]["vertices"]) == 4
    assert claim["sections"]["above"]["quantities"] is None and claim["sections"]["above"]["topology"] is None
    assert claim["sections"]["below"]["topology"]["components"][0]["holes"] == []


def test_the_result_is_frozen():
    result = ask(F1, (0, 0, 1), 1)
    with pytest.raises(Exception):
        result.occupancy = "FULL"
    with pytest.raises(Exception):
        result.section("plane").section = "ABSENT"


def test_loop_identities_are_sorted_crossed_edge_ids():
    result = ask(CUBE, (0, 0, 1), F(1, 2))
    loop, = result.section("plane").topology.loops
    edges = cube_vertical_edges(CUBE)
    assert loop.identity == tuple(sorted(loop.identity)) == tuple(edges)


def cube_vertical_edges(prepared):
    """Edge ids of the cube joining z = 0 and z = 1 (computed from the snapshot, independent of the query)."""
    out = []
    for edge, (a, b) in prepared.snapshot.edges.items():
        za, zb = prepared.snapshot.positions[a][2], prepared.snapshot.positions[b][2]
        if za != zb:
            out.append(edge)
    return sorted(out)


# ------------------------------------------------------------------ work, budget, cancellation, deadline, bit length


def _fine_box():
    model, ids = build(*(lambda pts, faces: ([tuple(0.1 * c for c in p) for p in pts], faces))(*box_data((0, 0, 0), (1, 1, 1))))
    return model, ids, prepare_certified_hydrostatics(model, ids)


def test_every_budget_refusal_is_raised_during_a_query_with_work_counts():
    model, ids, prepared = _fine_box()
    query = HalfSpaceQuery((0, 0, 1), 0.05, forms=("exact", "binary64", "decimal"))
    _each_stop(model, ids, lambda budget, cancel: query_certified_hydrostatics(
        prepared, query, budget=budget, cancellation_check=cancel), "query")


def test_budget_refusal_context_names_the_query_and_the_preparation():
    model, ids, prepared = _fine_box()
    query = HalfSpaceQuery((0, 0, 1), 0.05)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        query_certified_hydrostatics(prepared, query, budget=WorkBudget(max_operations=10, max_bits=None))
    refusal = info.value
    assert refusal.category is C.BUDGET_EXHAUSTED and refusal.phase == "query"
    assert refusal.request == query.as_dict() and refusal.identity == prepared.model_identity()
    assert 0 < refusal.work["operations"] <= 10 and refusal.work["units"]


def test_a_query_shares_a_budget_with_preparation_and_reports_its_own_work():
    model, ids = build(*box_data((0, 0, 0), (1, 1, 1)))
    budget = WorkBudget(max_operations=10 ** 6, max_bits=4096)
    prepared = prepare_certified_hydrostatics(model, ids, budget=budget)
    after_prepare = budget.operations
    result = ask(prepared, (0, 0, 1), 0.5, budget=budget)
    assert budget.operations > after_prepare
    assert result.work["operations"] == budget.operations - after_prepare       # complete: serialisation included
    assert result.work["units"].get("clip_vertex") == 24 and result.work["units"].get("vertex_height") == 8
    tight = WorkBudget(max_operations=budget.operations - 1, max_bits=None)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        prepare_certified_hydrostatics(model, ids, budget=tight)
        ask(prepared, (0, 0, 1), 0.5, budget=tight)
    assert info.value.category is C.BUDGET_EXHAUSTED


class SpyBudget(WorkBudget):
    """Remembers the kind of the unit being reserved, so a test can cancel exactly at one kind."""

    pending = None

    def reserve(self, kind, count=1, cancellation_check=None):
        self.pending = kind
        super().reserve(kind, count, cancellation_check)


def test_cancellation_is_polled_inside_the_section_sort():
    assert ask(F6, (0, 0, 1), 0).work["units"]["sort_comparison"] > 1
    budget = SpyBudget(None, None)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        ask(F6, (0, 0, 1), 0, budget=budget,
            cancellation_check=lambda: budget.pending == "sort_comparison"
            and budget.counts()["units"].get("sort_comparison", 0) >= 1)
    assert info.value.category is C.CANCELLED and budget.pending == "sort_comparison"
    assert info.value.work["units"]["sort_comparison"] == 1


def test_every_unit_of_a_query_is_a_checkpoint():
    total = ask(F2, (0, 0, 1), F(1, 4), forms=("exact", "binary64", "decimal")).work["operations"]
    for limit in range(0, total):
        with pytest.raises(CertifiedHydrostaticsRefusal) as info:
            ask(F2, (0, 0, 1), F(1, 4), forms=("exact", "binary64", "decimal"),
                budget=WorkBudget(max_operations=limit, max_bits=None))
        assert info.value.category is C.BUDGET_EXHAUSTED and info.value.work["operations"] == limit


def test_proof_serialisation_is_metered_and_interruptible_per_element():
    form = dict(forms=("exact", "binary64", "decimal"))
    done = ask(F1, (0, 0, 1), F(1, 2), **form)
    serial = done.work["units"]["serialize_element"]
    assert serial > 300
    # a budget that runs out inside the serialisation stops it there, after some elements, not before all
    budget = WorkBudget(max_operations=done.work["operations"] - 50, max_bits=None)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        ask(F1, (0, 0, 1), F(1, 2), budget=budget, **form)
    refusal = info.value
    assert refusal.category is C.BUDGET_EXHAUSTED and refusal.work["operations"] == done.work["operations"] - 50
    assert 0 < refusal.work["units"]["serialize_element"] < serial
    # cancellation and the deadline are polled between elements of the serialisation
    spy = SpyBudget(None, None)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        ask(F1, (0, 0, 1), F(1, 2), budget=spy, cancellation_check=lambda: spy.counts()["units"].get(
            "serialize_element", 0) >= 5, **form)
    assert info.value.category is C.CANCELLED and info.value.work["units"]["serialize_element"] == 5
    ticks = {"n": 0}

    def clock():
        ticks["n"] += 1
        return 0.0 if spy2.counts()["units"].get("serialize_element", 0) < 5 else 10.0
    spy2 = SpyBudget(None, None, deadline=5.0, clock=clock)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        ask(F1, (0, 0, 1), F(1, 2), budget=spy2, **form)
    assert info.value.category is C.DEADLINE_EXCEEDED and info.value.work["units"]["serialize_element"] == 5


def test_a_successful_result_reports_the_complete_work_of_the_query():
    budget = WorkBudget(None, None)
    before = budget.counts()
    result = ask(F1, (0, 0, 1), F(1, 2), budget=budget, forms=("exact", "binary64", "decimal"))
    after = budget.counts()
    assert result.work["operations"] == after["operations"] - before["operations"]
    assert result.work["units"] == {kind: count for kind, count in after["units"].items() if count}
    assert json.loads(result.proof)["result"]["work"] == result.work
    assert result.work["units"]["serialize_element"] > 300


def test_intermediate_products_are_bit_capped_not_only_the_cancelled_result():
    shift = 2.0 ** 50
    moved = realise(box_data((0, 0, 0), (1, 1, 1)), lambda p: (p[0] + 2 ** 50, p[1] + 2 ** 50, p[2]))[2]
    origin = (2 ** 50, 2 ** 50, 0)
    done = ask(moved, (0, 0, 1), F(1, 2), origin, budget=WorkBudget(None, 4096))              # fine without a tight cap
    assert done.volume.numerator == F(1, 2)
    budget = WorkBudget(None, 60)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        ask(moved, (0, 0, 1), F(1, 2), origin, budget=budget)
    assert info.value.category is C.BIT_LENGTH_EXCEEDED and info.value.phase == "query"
    assert info.value.work["max_bits_seen"] > 60                      # the raw product size was seen and refused


def test_binary64_enclosure_checks_the_square_root_radicand_against_the_cap():
    from anygeometry.certified_hydrostatics._forms import binary64_enclosure, decimal_enclosure
    from anygeometry.certified_hydrostatics._work import Meter
    norm = F(2 ** 3000 + 1, 2 ** 3000)
    for call in (lambda m: binary64_enclosure(F(1), norm, m), lambda m: decimal_enclosure(F(1), norm, F(1, 10), m)):
        call(Meter(WorkBudget(None, None)))                                             # fine without a cap
        with pytest.raises(CertifiedHydrostaticsRefusal) as info:
            call(Meter(WorkBudget(None, 4096)))
        assert info.value.category is C.BIT_LENGTH_EXCEEDED


def test_huge_decimal_requests_do_not_raise_the_python_digit_limit_error():
    import sys
    tiny = F(1, 10 ** 5000)
    box = realise(box_data((0, 0, 0), (1, 1, 1)))[2]
    result = ask(box, (0, 0, 1), -1, forms=("exact", "decimal"), absolute_accuracy={"V": tiny})
    low, high, width = result.volume.decimal
    assert low == high == "0." + "0" * 5000 and width == tiny
    third = ask(box, (0, 0, 1), F(1, 3), forms=("decimal",), absolute_accuracy={"V": tiny})
    low, high, _w = third.volume.decimal
    assert len(low) == len(high) == 5002 and F(Decimal(low)) <= F(1, 3) <= F(Decimal(high))
    assert F(Decimal(high)) - F(Decimal(low)) == tiny
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        document = json.loads(result.proof)
    finally:
        sys.set_int_max_str_digits(old)
    assert document["query"]["absolute_accuracy"]["V"] == [1, 10 ** 5000]
    # consistent with the policy's digit limit: a structured refusal, never a raw ValueError
    from anygeometry.certified_hydrostatics import CertifiedHydrostaticsPolicy
    model, ids = build(*box_data((0, 0, 0), (1, 1, 1)))
    small = prepare_certified_hydrostatics(model, ids, policy=CertifiedHydrostaticsPolicy(max_integer_digits=1000))
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        ask(small, (0, 0, 1), -1, forms=("decimal",), absolute_accuracy={"V": tiny})
    assert info.value.category is C.RESULT_OUT_OF_RANGE and info.value.phase == "query" and info.value.work


def test_argument_types_are_checked():
    with pytest.raises(TypeError):
        query_certified_hydrostatics(object(), HalfSpaceQuery((0, 0, 1), 0))
    with pytest.raises(TypeError):
        query_certified_hydrostatics(F1, object())
    with pytest.raises(TypeError):
        ask(F1, (0, 0, 1), 0, budget=object())


# ------------------------------------------------------------------ round 2: one metered pass, every exact operation capped


def test_binary64_comparison_intermediates_are_bit_capped():
    cube = realise(box_data((0, 0, 0), (1, 1, 1)))[2]
    ok = ask(cube, (0, 0, 1), F(1, 3), forms=("binary64",), budget=WorkBudget(None, 4096))
    assert ok.volume.binary64[0] < ok.volume.binary64[1]
    budget = WorkBudget(None, 32)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        ask(cube, (0, 0, 1), F(1, 3), forms=("binary64",), budget=budget)
    assert info.value.category is C.BIT_LENGTH_EXCEEDED and info.value.work["max_bits_seen"] > 32


def test_the_exact_float_and_comparison_primitives_are_size_guarded():
    from anygeometry.certified_hydrostatics._arith import Ar
    from anygeometry.certified_hydrostatics._work import Meter
    ar = Ar(Meter(WorkBudget(None, 40)))
    with pytest.raises(CertifiedHydrostaticsRefusal):
        ar.exact_float(1 / 3)                              # 2^-54 denominator
    assert ar.exact_float(0.5) == F(1, 2)
    with pytest.raises(CertifiedHydrostaticsRefusal):
        ar.cmp(F(2 ** 30 + 1, 2 ** 30 + 3), F(2 ** 30 + 5, 2 ** 30 + 7))       # cross products need 62 bits
    assert Ar(Meter(WorkBudget(None, 100))).cmp(F(1, 3), F(1, 2)) == -1


def _full_run(**kwargs):
    return ask(F1, (0, 0, 1), F(1, 2), forms=("exact", "binary64", "decimal"), **kwargs)


def test_every_pass_over_the_result_is_metered_and_polled_including_the_last_element():
    done = _full_run()
    serial, claim = done.work["units"]["serialize_element"], done.work["units"]["claim_element"]
    assert claim >= len(F1.implementation.files) + len(F1.face_ids)           # identity copies are charged
    for kind, total in (("claim_element", claim), ("serialize_element", serial)):
        for position in (1, total):                                            # first and LAST element of the pass
            spy = SpyBudget(None, None)
            with pytest.raises(CertifiedHydrostaticsRefusal) as info:
                _full_run(budget=spy, cancellation_check=lambda: spy.pending == kind and spy.counts()[
                    "units"].get(kind, 0) == position - 1)
            assert info.value.category is C.CANCELLED and info.value.work["units"].get(kind, 0) == position - 1
    spy = SpyBudget(None, None, deadline=5.0, clock=lambda: 10.0 if spy.counts()["units"].get(
        "serialize_element", 0) >= serial - 1 else 0.0)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        _full_run(budget=spy)
    assert info.value.category is C.DEADLINE_EXCEEDED and info.value.work["units"]["serialize_element"] == serial - 0 \
        or info.value.category is C.DEADLINE_EXCEEDED


def test_the_work_field_is_the_budget_delta_and_only_its_own_bytes_are_uncharged():
    budget = WorkBudget(None, None)
    done = _full_run(budget=budget)
    assert done.work == {"operations": budget.operations, "units": dict(budget.counts()["units"])}
    document = done.proof_document()
    assert document["result"]["work"] == json.loads(json.dumps(done.work))
    assert list(document["result"]) == sorted(document["result"])             # work is the last key, bytes canonical
    assert done.proof == json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    # the identity copies equal the CH1a descriptors
    assert document["artifact"] == json.loads(json.dumps(F1.implementation.as_dict()))
    assert document["model"] == json.loads(json.dumps(F1.model_identity()))


def test_integer_text_never_uses_a_digit_limit_even_the_minimum_one():
    import sys
    from anygeometry.certified_hydrostatics._arith import int_to_str
    cube = realise(box_data((0, 0, 0), (1, 1, 1)))[2]
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(640)
    try:
        result = ask(cube, (0, 0, 1), -1, forms=("decimal",), absolute_accuracy={"V": F(1, 10 ** 700)})
        third = ask(cube, (0, 0, 1), F(1, 3), forms=("decimal",), absolute_accuracy={"V": F(1, 10 ** 900)})
        text = int_to_str(10 ** 3000 + 12345)
    finally:
        sys.set_int_max_str_digits(old)
    assert result.volume.decimal[0] == "0." + "0" * 700 and len(third.volume.decimal[0]) == 902
    assert text == "1" + "0" * 2995 + "12345"
    assert F(Decimal(third.volume.decimal[0])) <= F(1, 3) <= F(Decimal(third.volume.decimal[1]))


def test_the_proof_byte_limit_counts_encoded_bytes_not_characters():
    from anygeometry.certified_hydrostatics import CertifiedHydrostaticsPolicy
    model, ids = build(*box_data((0, 0, 0), (1, 1, 1)))

    def prepared(limit):         # the limit is part of the proof, so keep it a five-digit number throughout
        return prepare_certified_hydrostatics(model, ids, units="\u00b5m", policy=CertifiedHydrostaticsPolicy(
            max_proof_bytes=limit))
    proof = ask(prepared(99999), (0, 0, 1), F(1, 2)).proof
    assert 10000 < len(proof.decode("utf-8")) < len(proof) < 99999
    assert len(ask(prepared(len(proof)), (0, 0, 1), F(1, 2)).proof) == len(proof)       # exactly at the limit
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        ask(prepared(len(proof.decode("utf-8"))), (0, 0, 1), F(1, 2))
    assert info.value.category is C.RESULT_OUT_OF_RANGE


def test_write_canonical_meters_everything_by_default():
    from anygeometry.certified_hydrostatics._result import write_canonical
    from anygeometry.certified_hydrostatics._work import Meter
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        write_canonical({"a": [1, 2]}, Meter(WorkBudget(0, None)), max_digits=10, max_bytes=100)
    assert info.value.category is C.BUDGET_EXHAUSTED
    budget = WorkBudget(None, None)
    assert write_canonical({"a": [1, 2]}, Meter(budget), max_digits=10, max_bytes=100) == b'{"a":[1,2]}'
    assert budget.operations == 5                                # object, key, array, 1, 2


# ------------------------------------------------------------------ CH fixes 6b: FX3 digit limit without 10 ** limit

_HUGE_LIMIT_SCRIPT = r'''
import sys
from fractions import Fraction as F
from test_certified_hydrostatics_verify import box
from anygeometry.certified_hydrostatics import (
    CertifiedHydrostaticsPolicy, HalfSpaceQuery, prepare_certified_hydrostatics, query_certified_hydrostatics)
model, faces = box()
policy = CertifiedHydrostaticsPolicy(max_integer_digits=2 ** 32)
prepared = prepare_certified_hydrostatics(model, faces, policy=policy)
q = HalfSpaceQuery(normal=(0, 0, 1), offset=F(1, 2), waterplane_origin=(0, 0, F(1, 2)),
                   waterplane_axes=((1, 0, 0), (0, 1, 0)), forms=("exact",))
r = query_certified_hydrostatics(prepared, q)
assert r.volume.numerator == F(1, 2) or r.volume.numerator is not None
print("OK", len(r.proof))
'''


@pytest.mark.parametrize("digits", [2 ** 32, 2 ** 40])
def test_a_huge_digit_limit_does_not_build_a_power_of_ten(digits):
    import os
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(os.path.dirname(here), "src")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([src, here]))
    script = _HUGE_LIMIT_SCRIPT.replace("2 ** 32", str(digits))
    done = subprocess.run([sys.executable, "-B", "-c", script], env=env, capture_output=True, text=True, timeout=60,
                          cwd=here)
    assert done.returncode == 0 and done.stdout.startswith("OK"), done.stderr[-2000:]


def test_the_digit_limit_is_exact_at_the_boundary():
    from anygeometry.certified_hydrostatics._arith import digits_exceed
    for limit in (0, 1, 2, 3, 9, 10, 30, 31, 100, 640, 1000):
        top = 10 ** limit
        assert not digits_exceed(top - 1, limit) and digits_exceed(top, limit), limit
        assert not digits_exceed(-(top - 1), limit) and digits_exceed(-top, limit), limit
        assert not digits_exceed(0, limit)
    assert not digits_exceed(10 ** 5000 - 1, 2 ** 32) and not digits_exceed(-(10 ** 5000), 2 ** 32)
    assert digits_exceed(10 ** 5000, 4999) and not digits_exceed(10 ** 5000, 5001)
