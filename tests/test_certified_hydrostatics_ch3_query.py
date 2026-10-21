"""CH3c: the curved query engine (design CH3 sections 3.3, 4, 5, 8; narrowed plan of 2026-10-21).

What is pinned here

* exact anchors: F-A (parabolic prism) ``V = 53/1536`` full, ``29/3072`` for ``z <= 1/2``, ``A = 49/768``, with hand
  integrals of ``J`` and the waterplane moments computed in the test with Fractions; F-C (diamond prism) equal to
  CH1 on the planar-equivalent solid, exactly;
* the 12 fixed pilot planes of ANYstability's Q08 answer on the full B1 hull (and the same planes on the half
  hull): working accuracy ``1e-9``, outward binary64 enclosures that CONTAIN the independent references;
* group M (two crossings on one deck edge), the refusal of planes through a corner or an edge, tangent planes
  and planes containing a ruling, dry / full planes (section absent), the half/full relation, the +- heel
  reflection pair, scale invariance, the claim protocol and the proof fields.

Independent references (``tests/ch3_reference.py``; no production code is imported there): (1) **analytic
stations** with ``mpmath`` for the upright planes ``z = h`` (the iso-``v`` section of the bow is a rational
quadratic; ``A`` and the moments are ``u``-integrals of ``y x'``; ``V`` and ``J`` integrate the station areas
over ``z``), (2) an **independently implemented slice integration** for every plane: horizontal slices of the
hull clipped by the half-plane ``n_x x + n_y y <= d - n_z z`` with its own 2-D section computation (closed-form
intersection parameters, midpoint tests, chords closing exit -> enter) and Green's theorem, integrated over ``z``
between the heights at which the line meets a frame corner; the cap quantities integrate the chord weights with
the exact area element ``|n| / sqrt(n_x^2 + n_y^2) dt dz``. The reference itself is checked against the H1
numbers of the design (``V(1.5) = 22.569943729716``, ``A(1.5) = 23.319938010999``, ``V(2.9)``, ``V(3)``).

The half/full relation (design 8.2). ``R`` is the reflection ``(x, y, z) -> (x, -y, z)``. The full hull is
``stb U R(stb)``, so its clipped solid is ``[stb & H(n, d)] U R[stb & H(Rn, d)]``; the second term is the half hull
queried with the genuinely mirrored plane ``(n~, d, o~, e1~, e2~) = (R n, d, R o, R e1, R e2)``. Every waterplane
quantity adds (both plane coordinates and the area element are preserved), ``V`` adds, ``Jx`` and ``Jz`` add, and
``Jy_full = Jy(n, d) - Jy(n~, d)`` (the only sign flip: it is the world coordinate that reflects). The centre face
contributes nothing and its cut cancels between the two caps. The full hull's claim must intersect the interval
sum of the two half-hull claims."""
from __future__ import annotations

import math
import time
from fractions import Fraction as F
from functools import lru_cache

import mpmath as mp
import numpy as np
import pytest

import ch3_reference as ref
import ch3_solids as solids
from anygeometry.certified_hydrostatics import (CertifiedHydrostaticsPolicy, CertifiedHydrostaticsRefusal,
                                                HalfSpaceQuery, RefusalCategory, WorkBudget,
                                                prepare_certified_hydrostatics, query_certified_hydrostatics)
from test_certified_hydrostatics_ch3_admission import build_face, planes_fc, planes_fa, surface_fa, surface_fc
from test_certified_hydrostatics_admission import extrusion

C = RefusalCategory
ACC = F(1e-9)                  # the exact value of the double 1e-9 (queries convert floats exactly)
NAMES = ("V", "Jx", "Jy", "Jz", "A", "Mu", "Mv", "Iuu", "Ivv", "Iuv")
ALL_ACCURACY = {"V": 1e-9, "J": 1e-9, "A": 1e-9, "Mu": 1e-9, "Mv": 1e-9, "Iuu": 1e-9, "Ivv": 1e-9, "Iuv": 1e-9}
FORMS = ("enclosure", "binary64", "decimal")
QUERY_CEILING = 30.0          # seconds per query (Q08 minimum answer)
PREPARE_CEILING = 120.0       # seconds, cold preparation
TIMES: dict = {}


# ------------------------------------------------------------------ planes, models, caches

def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def pilot_inclined(heel, trim, point=(2.5, 0., 1.75)):
    """ANYstability's convention: ``R = Ry(-trim) Rx(heel)``, ``n = R[2]``, ``d = n . point``, basis ``R[0], R[1]``."""
    rotation = _ry(-trim) @ _rx(heel)
    n = rotation[2]
    return (tuple(float(x) for x in n), float(n @ np.array(point)), tuple(point), tuple(float(x) for x in rotation[0]),
            tuple(float(x) for x in rotation[1]))


def upright(z):
    return ((0., 0., 1.), float(z), (2.5, 0., float(z)), (1., 0., 0.), (0., 1., 0.))


PLANES = {
    "z1.0": upright(1.0), "z1.75": upright(1.75), "z2.5": upright(2.5),
    "heel+0.1": pilot_inclined(0.1, 0.), "heel-0.1": pilot_inclined(-0.1, 0.),
    "trim+0.1": pilot_inclined(0., 0.1), "trim-0.1": pilot_inclined(0., -0.1),
    "both+0.1": pilot_inclined(0.1, 0.1), "both-0.1": pilot_inclined(-0.1, -0.1),
    "dry z0.25": upright(0.25), "full z3.25": upright(3.25), "bottom z0.500001": upright(0.500001),
}
PARTIAL_PLANES = [name for name in PLANES if name not in ("dry z0.25", "full z3.25")]


@lru_cache(None)
def prepared(which):
    from conftest import b1_full_hull_shared, b1_half_hull_shared
    model, faces = b1_full_hull_shared() if which == "full" else b1_half_hull_shared()
    start = time.perf_counter()
    result = prepare_certified_hydrostatics(model, list(faces.values()))
    TIMES[("prepare", which)] = time.perf_counter() - start
    return result


def make_query(plane, accuracy=ALL_ACCURACY, forms=FORMS):
    n, d, o, e1, e2 = plane
    return HalfSpaceQuery(normal=n, offset=d, waterplane_origin=o, waterplane_axes=(e1, e2),
                          absolute_accuracy=accuracy, forms=forms)


def run(which, plane, **kwargs):
    start = time.perf_counter()
    result = query_certified_hydrostatics(prepared(which), make_query(plane, **kwargs))
    return result, time.perf_counter() - start


@lru_cache(None)
def answer(which, name):
    result, seconds = run(which, PLANES[name])
    TIMES[(which, name)] = seconds
    return result


def claims(result):
    """The claim intervals ``(lo, hi)`` in the order of ``NAMES`` (waterplane ones ``None`` if the section is absent)."""
    out = [result.volume.enclosure] + [m.enclosure for m in result.moments]
    section = result.section("plane")
    out += [q.enclosure for q in section.quantities] if section.quantities else [None] * 6
    return out


def quantity(result, name):
    if name == "V":
        return result.volume
    if name in ("Jx", "Jy", "Jz"):
        return next(m for m in result.moments if m.name == name)
    return result.section("plane").quantity(name)


def as_fraction(x):
    return F(mp.nstr(x, 28))


@lru_cache(None)
def hull(half):
    return ref.Hull(half=half)


@lru_cache(None)
def reference(which, name):
    """The 10 reference values (V, Jx, Jy, Jz, A, Mu, Mv, Iuu, Ivv, Iuv) for a pilot plane."""
    h = hull(which == "half")
    n, d, o, e1, e2 = PLANES[name]
    if name == "dry z0.25":
        return None
    if n[0] == 0. and n[1] == 0.:
        v = h.station_volume_stats(min(d, 3.0))
        cap = h.station_moments(d, (o[0], o[1])) if d <= 3.0 else None
    else:
        v = h.slice_volume_stats(n, d)
        cap = h.slice_cap_stats(n, d, o, e1, e2)
    return [as_fraction(x) for x in v] + ([as_fraction(x) for x in cap] if cap is not None else [])


def assert_encloses(result, name, values, half=False):
    """Every claim has width <= 1e-9, contains the reference, and its binary64 / decimal forms contain it too."""
    for label, value in zip(NAMES, values):
        item = quantity(result, label)
        lo, hi = item.enclosure
        assert hi - lo <= ACC, (name, label, float(hi - lo))
        assert lo <= value <= hi, (name, label, float(lo), float(value), float(hi))
        assert item.numerator is None, (name, label, "an enclosed quantity carries no exact form")
        b_lo, b_hi = item.binary64
        assert F(b_lo) <= lo and hi <= F(b_hi) and F(b_hi) - F(b_lo) <= ACC
        d_lo, d_hi, width = item.decimal
        assert F(d_lo) <= lo and hi <= F(d_hi) and F(d_hi) - F(d_lo) <= width == ACC
        assert item.accuracy == ACC and item.accuracy_met is True


# ------------------------------------------------------------------ the reference itself

def test_the_reference_reproduces_the_h1_numbers_of_the_design():
    h = hull(False)
    assert abs(h.station_volume_stats(1.5)[0] - mp.mpf("22.569943729716")) < 1e-9
    assert abs(h.station_moments(1.5)[0] - mp.mpf("23.319938010999")) < 1e-9
    assert abs(h.station_volume_stats(2.9)[0] - mp.mpf("56.691072446746")) < 1e-9
    assert abs(h.station_volume_stats(3.0)[0] - mp.mpf("59.241180737020")) < 1e-9
    half = hull(True)
    assert abs(2 * half.station_volume_stats(2.9)[0] - h.station_volume_stats(2.9)[0]) < 1e-12


# ------------------------------------------------------------------ F-A: exact values by hand integration

def _polymul(p, q):
    out = [F(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out


def _polypow(p, e):
    out = [F(1)]
    for _ in range(e):
        out = _polymul(out, p)
    return out


def _antiderivative_at(p, x):
    return sum(c * x ** (i + 1) / (i + 1) for i, c in enumerate(p))


def prism_fa_exact(top):
    """``V, Jx, Jy, Jz`` of ``{(x, z) in the parallelogram z-1/8 <= x <= z+1/8, 1/4 <= z <= top; 0 <= y <= x^2}``
    by exact polynomial integration."""
    z = [F(0), F(1)]
    lo, hi = [F(-1, 8), F(1)], [F(1, 8), F(1)]

    def inner(power):                      # int_{z-1/8}^{z+1/8} x^power dx as a polynomial in z
        def anti(bound):
            return [c / (power + 1) for c in _polypow(bound, power + 1)]
        a, b = anti(hi), anti(lo)
        return [(a[i] if i < len(a) else F(0)) - (b[i] if i < len(b) else F(0)) for i in range(max(len(a), len(b)))]

    def outer(poly):
        return _antiderivative_at(poly, top) - _antiderivative_at(poly, F(1, 4))

    v = outer(inner(2))
    jx = outer(inner(3))
    jy = outer([c / 2 for c in inner(4)])
    jz = outer(_polymul(z, inner(2)))
    return v, jx, jy, jz


def fa_waterplane_exact():
    """``A, Mu, Mv, Iuu, Ivv, Iuv`` of the section ``z = 1/2``: ``3/8 <= x <= 5/8``, ``0 <= y <= x^2``, origin 0."""
    def i(power):
        return F(5, 8) ** (power + 1) / (power + 1) - F(3, 8) ** (power + 1) / (power + 1)
    return i(2), i(3), i(4) / 2, i(4), i(6) / 3, i(5) / 2


@pytest.fixture(scope="module")
def fa():
    model, ids = solids.solid_fa(build_face, surface_fa, planes_fa)
    return prepare_certified_hydrostatics(model, ids)


def fa_query(n, d, **kw):
    return HalfSpaceQuery(normal=n, offset=d, absolute_accuracy={"V": 1e-9, "J": 1e-9, "A": 1e-9}, forms=FORMS, **kw)


def test_fa_exact_values_full_clipped_and_waterplane(fa):
    full = query_certified_hydrostatics(fa, fa_query((0., 0., 1.), 1.0))      # the whole prism: z <= 1 covers it
    assert full.occupancy == "FULL" and full.volume.numerator == F(53, 1536)
    assert full.section("plane").section == "ABSENT" and full.section("plane").quantities is None
    assert (full.volume.numerator,) + tuple(m.numerator for m in full.moments) == prism_fa_exact(F(3, 4))
    half = query_certified_hydrostatics(fa, fa_query((0., 0., 1.), 0.5))
    assert half.occupancy == "PARTIAL" and half.volume.numerator == F(29, 3072)
    assert (half.volume.numerator,) + tuple(m.numerator for m in half.moments) == prism_fa_exact(F(1, 2))
    section = half.section("plane")
    assert section.quantity("A").numerator == F(49, 768) and section.quantity("A").norm_squared == 1
    assert tuple(section.quantity(n).numerator for n in ("A", "Mu", "Mv", "Iuu", "Ivv", "Iuv")) == fa_waterplane_exact()
    for item in (half.volume, section.quantity("A")):          # the exact value is inside every form
        assert item.enclosure[0] == item.enclosure[1] == item.numerator
        assert F(item.binary64[0]) <= item.numerator <= F(item.binary64[1])
        assert F(item.decimal[0]) <= item.numerator <= F(item.decimal[1])


def test_fa_events_of_the_design_are_refused_or_answered_as_listed(fa):
    # n = (-3/2, 1, 1), d = 3/16: the deck-edge residual (u - 3/4)^2 has a double root inside the edge: tangency
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        query_certified_hydrostatics(fa, fa_query((F(-3, 2), 1, 1), F(3, 16), waterplane_axes=((1, 0, 0), (0, 1, 0))))
    assert info.value.category is C.CURVED_SECTION_DEGENERACY and info.value.sub_reason == "edge_tangency"
    # n = (1, 0, 0): the zero-beta case (every ruling parallel to the plane). d = 1/2 lies in the forbidden interval
    # [1/8, 7/8] of the heights along the rulings: the plane contains a ruling, refused; d = 15/16 is outside, the
    # face is wholly below and the query answers FULL
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        query_certified_hydrostatics(fa, fa_query((1, 0, 0), F(1, 2), waterplane_axes=((0, 1, 0), (0, 0, 1))))
    assert info.value.category is C.CURVED_SECTION_DEGENERACY and info.value.sub_reason == "ruling_in_plane"
    outside = query_certified_hydrostatics(fa, fa_query((1, 0, 0), F(15, 16), waterplane_axes=((0, 1, 0), (0, 0, 1))))
    assert outside.occupancy == "FULL" and outside.volume.numerator == F(53, 1536)
    # y + z <= 3/4 passes through the planar vertices (7/8, 0, 3/4), (5/8, 0, 3/4): refused (the design called it
    # generic, but those vertices lie in the plane)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        query_certified_hydrostatics(fa, fa_query((0, 1, 1), F(3, 4)))
    assert info.value.sub_reason == "plane_through_vertex"
    # a slanted generic plane y + z <= 7/8 answers; its volume int dz int dx min(x^2, 7/8 - z) (by mpmath, split at
    # the kink x^2 = 7/8 - z) lies inside the claim
    result = query_certified_hydrostatics(fa, fa_query((0, 1, 1), F(7, 8)))
    assert result.occupancy == "PARTIAL"
    mp.mp.dps = 30
    q = mp.mpf(7) / 8

    def inner(z):
        a, b = z - mp.mpf(1) / 8, z + mp.mpf(1) / 8
        kink = mp.sqrt(q - z)
        points = [a] + ([kink] if a < kink < b else []) + [b]
        return mp.quad(lambda x: min(x * x, q - z), points)
    breaks = [mp.mpf(1) / 4, mp.mpf(3) / 4]            # the outer integrand has kinks where x^2 = 7/8 - z meets an end
    for s_ in (mp.mpf(1) / 8, -mp.mpf(1) / 8):
        for r in mp.polyroots([1, 2 * s_ + 1, s_ * s_ - q], maxsteps=200, extraprec=50):
            if abs(mp.im(r)) < 1e-20 and breaks[0] < mp.re(r) < breaks[1]:
                breaks.append(mp.re(r))
    value = F(mp.nstr(mp.quad(inner, sorted(breaks)), 25))
    lo, hi = result.volume.enclosure
    assert lo <= value <= hi and hi - lo <= ACC


# ------------------------------------------------------------------ F-C: exact equality with CH1

def test_fc_equals_ch1_on_the_planar_equivalent_solid():
    model, ids = solids.solid_fc(build_face, surface_fc, planes_fc)
    curved = prepare_certified_hydrostatics(model, ids)
    planar_model, planar_ids = extrusion([(.25, .5), (.5, .25), (.75, .5), (.5, .75)], 1.0)
    planar = prepare_certified_hydrostatics(planar_model, planar_ids)
    assert curved.policy.convention_version == "2" and planar.policy.convention_version == "1"
    for n, d in (((0., 1., 1.), F(11, 8)), ((0., 0., 1.), F(1, 2)), ((1., 2., 3.), F(9, 4)), ((0., 0., 1.), 2.0)):
        q = HalfSpaceQuery(normal=n, offset=d, waterplane_origin=(1, 2, 0), waterplane_axes=((1, 0, 0), (0, 1, 0)),
                           absolute_accuracy={"V": 1e-9, "A": 1e-9}, forms=("exact", "enclosure"))
        a, b = query_certified_hydrostatics(curved, q), query_certified_hydrostatics(planar, q)
        assert a.occupancy == b.occupancy
        assert a.volume.numerator == b.volume.numerator
        assert [m.numerator for m in a.moments] == [m.numerator for m in b.moments]
        sa, sb = a.sections[0], b.sections[0]
        assert sa.section == sb.section
        for qa, qb in zip(sa.quantities or (), sb.quantities or ()):
            assert (qa.numerator, qa.norm_squared) == (qb.numerator, qb.norm_squared), qa.name
    q = HalfSpaceQuery(normal=(0., 1., 1.), offset=F(11, 8), forms=("enclosure",))
    assert query_certified_hydrostatics(curved, q).volume.numerator == F(167, 1536)   # the design's hand value


# ------------------------------------------------------------------ the pilot planes on the full and the half hull

@pytest.mark.parametrize("which", ["full", "half"])
@pytest.mark.parametrize("name", PARTIAL_PLANES)
def test_pilot_planes_enclose_the_independent_references(which, name):
    result = answer(which, name)
    assert result.occupancy == "PARTIAL" and result.generic
    assert TIMES[(which, name)] <= QUERY_CEILING
    assert_encloses(result, name, reference(which, name))
    section = result.section("plane")
    assert section.section == "PRESENT" and len(section.topology.loops) == 1
    assert section.topology.loops[0].role == "outer" and section.topology.components[0][1] == ()
    assert len(section.topology.cut_points) == len(section.topology.loops[0].identity) >= 3


def test_dry_and_full_planes_have_states_and_no_section():
    for which in ("full", "half"):
        dry = answer(which, "dry z0.25")
        assert dry.occupancy == "EMPTY" and dry.generic and dry.volume.numerator == 0
        assert all(m.numerator == 0 for m in dry.moments)
        assert dry.section("plane").section == "ABSENT" and dry.section("plane").quantities is None
        full = answer(which, "full z3.25")
        assert full.occupancy == "FULL" and full.section("plane").occupancy == "FULL"
        assert full.section("plane").section == "ABSENT" and full.section("plane").quantities is None
        values = reference(which, "full z3.25")[:4]
        assert_encloses(full, "full z3.25", values)
        total = F(59241180737020, 10 ** 12) / (1 if which == "full" else 2)       # design section 1 (H1)
        assert abs(values[0] - total) < F(1, 10 ** 9)


def test_the_bottom_adjacent_plane_has_a_tiny_but_positive_volume():
    result = answer("full", "bottom z0.500001")
    lo, hi = result.volume.enclosure
    assert 0 < lo and hi < F(1, 10 ** 3)
    assert result.section("plane").quantity("A").enclosure[0] > 0


def test_cold_preparation_and_query_times_are_recorded():
    start = time.perf_counter()
    from conftest import b1_full_hull_shared
    model, faces = b1_full_hull_shared()
    prepare_certified_hydrostatics(model, list(faces.values()))
    seconds = time.perf_counter() - start
    assert seconds < PREPARE_CEILING
    longest = max((t for key, t in TIMES.items() if key[0] in ("full", "half")), default=0.0)
    print(f"\nfull hull preparation {seconds:.2f} s; longest pilot query so far {longest:.2f} s")
    assert longest <= QUERY_CEILING


# ------------------------------------------------------------------ half hull vs full hull (design 8.2, stated above)

def mirror(plane):
    n, d, o, e1, e2 = plane
    r = lambda v: (v[0], -v[1], v[2])
    return (r(n), d, r(o), r(e1), r(e2))


def add_intervals(a, b):
    return (a[0] + b[0], a[1] + b[1])


@lru_cache(None)
def mirrored_half_answer(name):
    result, seconds = run("half", mirror(PLANES[name]))
    TIMES[("half-mirror", name)] = seconds
    return result


@pytest.mark.parametrize("name", PARTIAL_PLANES + ["dry z0.25", "full z3.25"])
def test_half_and_full_hull_obey_the_mirrored_term_relation(name):
    full, half, mirrored = answer("full", name), answer("half", name), mirrored_half_answer(name)
    assert full.occupancy == half.occupancy == mirrored.occupancy or name in ("dry z0.25", "full z3.25")
    for label in NAMES:
        if label.startswith(("A", "M", "I")) and full.section("plane").quantities is None:
            continue
        f = quantity(full, label).enclosure
        a, b = quantity(half, label).enclosure, quantity(mirrored, label).enclosure
        total = (a[0] - b[1], a[1] - b[0]) if label == "Jy" else add_intervals(a, b)
        assert f[0] <= total[1] and total[0] <= f[1], (name, label, [float(x) for x in (*f, *total)])
        assert f[1] - f[0] <= ACC and (total[1] - total[0]) <= 2 * ACC


def test_symmetric_planes_double_the_half_hull():
    """For ``n_y = 0``, ``o_y = 0`` and ``e2`` along ``y``: ``V, Jx, Jz, A, Mu, Iuu, Ivv`` double, ``Jy, Mv, Iuv`` vanish."""
    for name in ("z1.0", "z1.75", "z2.5", "trim+0.1", "trim-0.1", "bottom z0.500001"):
        full, half = answer("full", name), answer("half", name)
        for label in ("V", "Jx", "Jz", "A", "Mu", "Iuu", "Ivv"):
            f, h = quantity(full, label).enclosure, quantity(half, label).enclosure
            assert f[0] <= 2 * h[1] and 2 * h[0] <= f[1], (name, label)
        for label in ("Jy", "Mv", "Iuv"):
            f = quantity(full, label).enclosure
            assert f[0] <= 0 <= f[1], (name, label)


# ------------------------------------------------------------------ the +- heel reflection pair

def test_the_plus_minus_heel_pair_is_a_reflection():
    """heel -h is the mirror image of +h with ``e2 -> -R e2``: V, Jx, Jz, A, Mu, Iuu, Ivv agree; Jy, Mv, Iuv flip sign."""
    plus, minus = answer("full", "heel+0.1"), answer("full", "heel-0.1")
    for label in NAMES:
        a, b = quantity(plus, label).enclosure, quantity(minus, label).enclosure
        if label in ("Jy", "Mv", "Iuv"):
            b = (-b[1], -b[0])
        assert a[0] <= b[1] and b[0] <= a[1], (label, [float(x) for x in (*a, *b)])
        assert a[1] - a[0] <= ACC and b[1] - b[0] <= ACC


# ------------------------------------------------------------------ group M: two crossings on one deck edge

def _bow():
    from anygeometry.b1_example import b1_bow_surface
    s = b1_bow_surface()
    return ([[tuple(F(c) for c in s.controls[i][j]) for j in (0, 1)] for i in range(3)], [F(s.weights[i][0]) for i in range(3)])


def bow_point(u, v):
    """Exact ``S(u, v)`` and ``S_u(u, v)`` of the starboard bow (rational Bezier, the stored doubles exactly)."""
    p, w = _bow()
    b = [(1 - u) ** 2, 2 * u * (1 - u), u * u]
    db = [-2 * (1 - u), 2 - 4 * u, 2 * u]
    q = [[(1 - v) * p[i][0][c] + v * p[i][1][c] for c in range(3)] for i in range(3)]
    ww = sum(b[i] * w[i] for i in range(3))
    dw = sum(db[i] * w[i] for i in range(3))
    num = [sum(b[i] * w[i] * q[i][c] for i in range(3)) for c in range(3)]
    dnum = [sum(db[i] * w[i] * q[i][c] for i in range(3)) for c in range(3)]
    return tuple(x / ww for x in num), tuple((dnum[c] * ww - num[c] * dw) / (ww * ww) for c in range(3))


def cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot3(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_group_m_two_crossings_on_one_deck_edge():
    """``n = C'(1/4) x e_z`` on the deck arc, ``d = d_t + 2^-10`` (half hull): the vertical plane crosses the deck
    edge twice (u = 0.24658, 0.25342); the engine answers with two cut points of ordinals 0 and 1 on that edge.

    The side ``n . r <= d`` is a thin sliver (V ~ 5e-10) that the slice reference cannot resolve to its own
    width, so it is checked by complementarity: it plus the side ``-n . r <= -d`` is the half hull (station
    reference), the two caps are the same section, and the large side is compared with the slice reference."""
    c, cu = bow_point(F(1, 4), F(3, 4))
    n = cross3(cu, (0, 0, 1))
    d_t = dot3(n, c)
    p = prepared("half")
    deck = [e for e, (_f, trim) in p.snapshot.curved.arcs.items() if trim.plane.N == (3,) and trim.plane.D == (4,)]
    assert len(deck) == 1
    axes = ((0, 0, 1), (1, 0, 0))
    h = hull(True)
    accuracy = {"V": 1e-11, "J": 1e-11, "A": 1e-11, "Mu": 1e-11, "Mv": 1e-11, "Iuu": 1e-11, "Ivv": 1e-11, "Iuv": 1e-11}
    results = {}
    for sign in (1, -1):
        nn = tuple(sign * x for x in n)
        d = sign * (d_t + F(1, 1024))
        q = HalfSpaceQuery(normal=nn, offset=d, waterplane_axes=axes, forms=FORMS, absolute_accuracy=accuracy)
        results[sign] = query_certified_hydrostatics(p, q)
        assert results[sign].occupancy == "PARTIAL"
        crossings = [cp for cp in results[sign].section("plane").topology.cut_points if cp[0] == deck[0]]
        assert crossings == [(deck[0], 0), (deck[0], 1)]
    plus, minus = results[1], results[-1]
    assert quantity(plus, "V").enclosure[1] < F(1, 10 ** 8) < quantity(minus, "V").enclosure[0]
    total = [as_fraction(x) for x in h.station_volume_stats(3.0)]
    for label, whole in zip(NAMES[:4], total):
        a, b = quantity(plus, label).enclosure, quantity(minus, label).enclosure
        assert a[0] + b[0] <= whole <= a[1] + b[1], (label, float(whole))
    for label in NAMES[4:]:
        a, b = quantity(plus, label).enclosure, quantity(minus, label).enclosure
        assert a[0] <= b[1] and b[0] <= a[1], label
    big = [float(-x) for x in n]
    values = [as_fraction(x) for x in h.slice_volume_stats(big, float(-(d_t + F(1, 1024))))]         + [as_fraction(x) for x in h.slice_cap_stats(big, float(-(d_t + F(1, 1024))), (0., 0., 0.), (0., 0., 1.),
                                                      (1., 0., 0.))]
    for label, value in zip(NAMES, values):
        lo, hi = quantity(minus, label).enclosure
        assert lo <= value <= hi, (label, float(lo), float(value), float(hi))
    below = query_certified_hydrostatics(p, HalfSpaceQuery(normal=n, offset=d_t - F(1, 1024), waterplane_axes=axes,
                                                           forms=("enclosure",)))
    assert below.occupancy == "EMPTY"                      # d_t - 2^-10: the plane misses the half hull


# ------------------------------------------------------------------ refusals (constructed from the B1 geometry)

def refusal(which, **kwargs):
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        query_certified_hydrostatics(prepared(which), HalfSpaceQuery(**kwargs))
    assert info.value.phase == "query" and info.value.category is C.CURVED_SECTION_DEGENERACY
    return info.value


def test_a_plane_through_a_corner_is_refused():
    """``x + z = 9/2`` passes through the aft deck corners ``(3/2, +-y*, 3)`` (algebraic points; on the half hull also
    the centreline vertex ``(3/2, 0, 3)``)."""
    for which in ("half", "full"):
        error = refusal(which, normal=(1, 0, 1), offset=F(9, 2), waterplane_axes=((0, 1, 0), (1, 0, 0)))
        assert error.sub_reason == "plane_through_vertex"


def test_a_plane_containing_an_edge_is_refused():
    for plane in (dict(normal=(0, 0, 1), offset=3), dict(normal=(0, 0, 1), offset=F(1, 2)),
                  dict(normal=(1, 0, 0), offset=F(3, 2), waterplane_axes=((0, 1, 0), (0, 0, 1)))):
        error = refusal("full", **plane)
        assert error.sub_reason == "edge_in_plane"


def test_a_tangent_plane_is_refused():
    c, cu = bow_point(F(1, 4), F(3, 4))
    n = cross3(cu, (0, 0, 1))
    error = refusal("half", normal=n, offset=dot3(n, c), waterplane_axes=((0, 0, 1), (1, 0, 0)))
    assert error.sub_reason == "edge_tangency"


def test_a_plane_containing_a_ruling_is_refused():
    a, _ = bow_point(F(1, 4), F(0))
    b, _ = bow_point(F(1, 4), F(1))
    direction = tuple(y - x for x, y in zip(a, b))
    for helper in ((1, 0, 0), (0, 1, 0)):
        n = cross3(direction, helper)
        error = refusal("half", normal=n, offset=dot3(n, a), waterplane_axes=((1, 0, 0), (0, 1, 0))
                        if n[2] else ((0, 0, 1), (1, 0, 0)))
        assert error.sub_reason == "ruling_in_plane"


def test_a_plane_parallel_to_a_ruling_inside_a_cut_face_is_refused_as_outside_the_class():
    """``beta`` vanishes at u = 1/4 (the plane is parallel to that ruling but does not contain it): the cut graph
    has a pole inside the face's extent, which the narrowed engine does not support (documented known limit)."""
    a, _ = bow_point(F(1, 4), F(0))
    b, _ = bow_point(F(1, 4), F(1))
    n = cross3(tuple(y - x for x, y in zip(a, b)), (1, 0, 0))
    for shift in (F(1, 2), F(-1, 10)):
        error = refusal("half", normal=n, offset=dot3(n, a) + shift, waterplane_axes=((0, 0, 1), (1, 0, 0)))
        assert error.sub_reason == "beta_root_in_extent"


def test_the_refusal_keeps_the_request_and_the_structured_cause():
    error = refusal("full", normal=(0, 0, 1), offset=3)
    assert error.request["normal"][2] == [1, 1] and error.identity["model_id"]
    assert error.work["operations"] > 0


# ------------------------------------------------------------------ the claim protocol and the proof

def test_claim_protocol_forms_and_default_width():
    p = prepared("half")
    result = query_certified_hydrostatics(p, HalfSpaceQuery(normal=(0., 0., 1.), offset=1.75, forms=("enclosure", "binary64")))
    for label in NAMES[:4]:
        item = quantity(result, label)
        assert item.accuracy is None and item.accuracy_met is None and item.numerator is None
        assert item.enclosure[1] - item.enclosure[0] <= F(1, 10 ** 12)       # the default claim width
        assert item.decimal is None and F(item.binary64[0]) <= item.enclosure[0]


def test_a_finer_accuracy_than_binary64_refuses_that_form_only():
    p = prepared("half")
    q = dict(normal=(0., 0., 1.), offset=1.75, absolute_accuracy={"V": 1e-15})
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        query_certified_hydrostatics(p, HalfSpaceQuery(forms=("enclosure", "binary64"), **q))
    assert info.value.category is C.ACCURACY_UNACHIEVABLE
    result = query_certified_hydrostatics(p, HalfSpaceQuery(forms=("enclosure",), **q))
    assert result.volume.enclosure[1] - result.volume.enclosure[0] <= F(1, 10 ** 15)


def test_scale_invariance_of_the_claims():
    name = "trim+0.1"
    n, d, o, e1, e2 = PLANES[name]
    scaled = (tuple(2 * x for x in n), 2 * d, o, e1, e2)
    a = answer("half", name)
    b, _ = run("half", scaled)
    for label in NAMES:
        x, y = quantity(a, label).enclosure, quantity(b, label).enclosure
        assert x[0] <= y[1] and y[0] <= x[1], label


def test_repeated_queries_give_identical_claims_and_proofs():
    q = make_query(PLANES["z1.75"], accuracy={"V": 1e-9}, forms=("enclosure",))
    p = prepared("half")
    a, b = query_certified_hydrostatics(p, q), query_certified_hydrostatics(p, q)
    assert a.proof == b.proof and a.claim() == b.claim()


def test_the_proof_carries_the_enclosures_cut_points_and_orientation_convention():
    result = answer("half", "heel+0.1")
    document = result.proof_document()
    assert document["model"]["snapshot_digest"] == prepared("half").snapshot_digest
    assert document["preparation_policy"]["convention_version"] == "2" and document["preparation_policy"]["max_bits"] == 4096
    claim = document["result"]
    assert claim["volume"]["enclosure"]["lo"] == [result.volume.enclosure[0].numerator, result.volume.enclosure[0].denominator]
    assert "exact" not in claim["volume"] and set(claim["volume"]) >= {"enclosure", "binary64", "decimal", "accuracy"}
    topology = claim["sections"]["plane"]["topology"]
    assert topology["cut_points"] == [list(cp) for cp in result.section("plane").topology.cut_points]
    assert topology["loops"][0]["role"] == "outer"
    assert set(claim["execution"]) == {"integrals", "taylor_pieces", "point_bits", "core"}     # untrusted metadata
    # the claim protocol (design 4.4): claim = core +- width of the core, and at most 3/4 of the accuracy wide
    for label in NAMES:
        (clo, chi), item = claim["execution"]["core"][label], quantity(result, label)
        core = (F(*clo), F(*chi))
        width = core[1] - core[0]
        assert item.enclosure == (core[0] - width, core[1] + width) and 3 * width <= F(3, 4) * ACC
    assert "orientation" in document["convention"]["curved"] and "events" in document["convention"]["curved"]


def test_planar_queries_still_ignore_the_enclosure_form():
    model, faces = extrusion([(0., 0.), (2., 0.), (0., 2.)], 1.5)
    p = prepare_certified_hydrostatics(model, faces)
    result = query_certified_hydrostatics(p, HalfSpaceQuery(normal=(0., 0., 1.), offset=0.5, forms=("enclosure",)))
    assert result.volume.numerator == F(1)
    assert result.volume.enclosure is None
    assert result.section("plane").quantity("A").numerator == F(2)


def test_the_policy_bit_cap_is_enforced_when_the_budget_carries_it():
    """A budget made by ``policy.new_budget()`` caps exact values at ``max_bits``: the pilot queries at 1e-9 need
    longer integers and are refused ``BIT_LENGTH_EXCEEDED`` (fail closed) -- the default budget is unlimited."""
    policy = CertifiedHydrostaticsPolicy()
    p = prepared("half")
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        query_certified_hydrostatics(p, make_query(PLANES["z1.75"], accuracy={"V": 1e-9}, forms=("enclosure",)),
                                     budget=policy.new_budget())
    assert info.value.category is C.BIT_LENGTH_EXCEEDED and info.value.phase == "query"
    assert info.value.work["max_bits_seen"] > 4096


def test_cancellation_and_budget_stop_a_curved_query():
    p = prepared("half")
    q = make_query(PLANES["z1.75"], accuracy={"V": 1e-9}, forms=("enclosure",))
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        query_certified_hydrostatics(p, q, budget=WorkBudget(2000, None))
    assert info.value.category is C.BUDGET_EXHAUSTED and info.value.phase == "query"
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        query_certified_hydrostatics(p, q, cancellation_check=lambda: True)
    assert info.value.category is C.CANCELLED
