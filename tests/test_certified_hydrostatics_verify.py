"""Certified hydrostatics, slice CH1c: the independent verifier (design section 6; acceptance manifest sections 2, 5, 6).

Fixtures F1-F9 are rebuilt here from the manifest definitions. Attacks are built by altering a genuine result object
or its proof (regenerating the proof where the attack says so). Nothing is taken from the provider's internals."""
from __future__ import annotations

import copy
import itertools
import json
import math
from dataclasses import replace
from fractions import Fraction as F

import pytest

from test_certified_hydrostatics_admission import build, extrusion
from anygeometry import Plane
from anygeometry.certified_hydrostatics import (
    CertifiedHydrostaticsPolicy, CertifiedHydrostaticsRefusal, HalfSpaceQuery, QuantityResult, RefusalCategory,
    SectionResult, Topology, TopologyLoop, VerifiedHydrostatics, WorkBudget, prepare_certified_hydrostatics,
    query_certified_hydrostatics, verify_certified_hydrostatics)
from anygeometry.certified_hydrostatics._verify import _claim_of

C = RefusalCategory
ALL_FORMS = ("exact", "binary64", "decimal")

# ------------------------------------------------------------------ fixtures


def _normal_outward(points, loop, inside):
    """Reverse ``loop`` when its Newell normal points towards ``inside`` (planar faces)."""
    nx = ny = nz = 0.0
    m = len(loop)
    for i in range(m):
        p, q = points[loop[i]], points[loop[(i + 1) % m]]
        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])
    p0 = points[loop[0]]
    side = nx * (inside[0] - p0[0]) + ny * (inside[1] - p0[1]) + nz * (inside[2] - p0[2])
    assert side != 0
    return loop[::-1] if side > 0 else loop


def oriented_solid(points, loops, inside):
    return build(points, [[_normal_outward(points, list(loop), inside)] for loop in loops])


QUADS = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 4, 6, 2), (1, 3, 7, 5), (0, 1, 5, 4), (2, 6, 7, 3))


def box(lo=(0., 0., 0.), hi=(1., 1., 1.), scale=1.0):
    pts = [(lo[0] + (hi[0] - lo[0]) * i, lo[1] + (hi[1] - lo[1]) * j, lo[2] + (hi[2] - lo[2]) * k)
           for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    return build(pts, [[list(q)] for q in QUADS])


def tetra():
    return build([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)],
                 [[[0, 2, 1]], [[0, 1, 3]], [[1, 2, 3]], [[0, 3, 2]]])


U = [(0., 0.), (3., 0.), (3., 3.), (2., 3.), (2., 1.), (1., 1.), (1., 3.), (0., 3.)]


def holed_plate(x0, x1, y0, y1, holes, z0=0., z1=1.):
    """Plate [x0,x1]x[y0,y1]x[z0,z1] with through bores ``(hx0, hx1, hy0, hy1)``."""
    def rect(a, b, c, d):
        return [(a, c), (b, c), (b, d), (a, d)]

    outer = rect(x0, x1, y0, y1)
    pts = [(x, y, z0) for x, y in outer] + [(x, y, z1) for x, y in outer]
    ob, ot = list(range(0, 4)), list(range(4, 8))
    faces = [[ot], [ob[::-1]]]
    hole_rings = []
    for hx0, hx1, hy0, hy1 in holes:
        start = len(pts)
        ring = rect(hx0, hx1, hy0, hy1)
        pts += [(x, y, z0) for x, y in ring] + [(x, y, z1) for x, y in ring]
        hb, ht = list(range(start, start + 4)), list(range(start + 4, start + 8))
        hole_rings.append((hb, ht))
        faces[0].append(ht[::-1])
        faces[1].append(hb)
        for i in range(4):
            j = (i + 1) % 4
            faces.append([[ht[i], ht[j], hb[j], hb[i]]])
    for i in range(4):
        faces.append([[ob[i], ob[(i + 1) % 4], ot[(i + 1) % 4], ot[i]]])
    return build(pts, [f if isinstance(f[0], list) else [f] for f in faces])


def saddle():
    ring = [(1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1)]
    pts = [(0., 0., 0.)] + [(float(x), float(y), float(abs(x) - abs(y))) for x, y in ring]
    pts += [(float(x), float(y), -2.) for x, y in ((1, -1), (1, 1), (-1, 1), (-1, -1))]
    loops = [(0, 1 + i, 1 + (i + 1) % 8) for i in range(8)]
    # sides: bottom corner pair + the three top points between the two corners
    loops += [(9, 10, 3, 2, 1), (10, 11, 5, 4, 3), (11, 12, 7, 6, 5), (12, 9, 1, 8, 7), (9, 12, 11, 10)]
    return oriented_solid(pts, loops, (0., 0., -1.5))


def recess():
    pts = [(-2., -2., 0.), (2., -2., 0.), (2., 2., 0.), (-2., 2., 0.),
           (-2., -2., 2.), (2., -2., 2.), (2., 2., 2.), (-2., 2., 2.),
           (-1., -1., 2.), (1., -1., 2.), (1., 1., 2.), (-1., 1., 2.), (0., 0., 1.)]
    faces = [[[3, 2, 1, 0]]]
    faces += [[[i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i]] for i in range(4)]
    faces += [[[4, 5, 6, 7], [8, 11, 10, 9]]]
    faces += [[[12, 8 + i, 8 + (i + 1) % 4]] for i in range(4)]
    return build(pts, faces)


def basis_for(n):
    k = max(range(3), key=lambda i: abs(n[i]))
    return {0: ((0, 1, 0), (0, 0, 1)), 1: ((1, 0, 0), (0, 0, 1)), 2: ((1, 0, 0), (0, 1, 0))}[k]


def expected_for(prepared):
    return {"artifact_files": prepared.implementation.files, "convention_version": prepared.policy.convention_version,
            "units": prepared.units, "policy": prepared.policy}


def make_query(n, d, forms=ALL_FORMS, accuracy=None, basis=None):
    e1, e2 = basis or basis_for(n)
    return HalfSpaceQuery(normal=n, offset=F(d), waterplane_origin=(0, 0, 0), waterplane_axes=(e1, e2),
                          absolute_accuracy=accuracy or {}, forms=forms)


class Case:
    """A prepared fixture with helpers to query and verify."""

    def __init__(self, maker, policy=None, units="m"):
        self.model, self.faces = maker()
        self.prepared = prepare_certified_hydrostatics(self.model, self.faces, policy=policy, units=units)
        self.expected = expected_for(self.prepared)

    def query(self, n, d, **kw):
        q = make_query(n, d, **kw)
        return q, query_certified_hydrostatics(self.prepared, q)

    def verify(self, q, r, **kw):
        return verify_certified_hydrostatics(self.model, self.faces, q, r, self.expected, **kw)

    def run(self, n, d, **kw):
        q, r = self.query(n, d, **kw)
        return q, r, self.verify(q, r)


def refusal_of(call):
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        call()
    return info.value


def failed(call, sub):
    error = refusal_of(call)
    assert error.category == C.VERIFICATION_FAILED, error
    assert error.sub_reason == sub, (error.sub_reason, str(error))
    assert error.phase == "verify" and error.work is not None
    return error


# ------------------------------------------------------------------ agreement on F1-F9

Z_, X_, Y_ = (0, 0, 1), (1, 0, 0), (0, 1, 0)

FIXTURES = {
    "F1": (lambda: box((0., 0., 0.), (2., 3., 5.)),
           [(Z_, h) for h in (F(0), F(1), F(5, 2), F(5), F(6), F(-1))]),
    "F2": (tetra, [(Z_, h) for h in (F(0), F(1, 4), F(1, 2), F(1))]),
    "F3": (lambda: extrusion([(0., 0.), (2., 0.), (0., 1.)], 3.0), [(X_, s) for s in (F(0), F(1), F(2))]),
    "F4": (lambda: extrusion(U, 1.0), [(Y_, F(1, 2)), (Y_, F(2))]),
    "F5": (lambda: holed_plate(0., 3., 0., 3., [(1., 2., 1., 2.)]), [(Z_, F(1, 2)), (X_, F(3, 2))]),
    "F6": (saddle, [(Z_, F(0)), (Z_, F(1, 2)), (Z_, F(-1, 2))]),
    "F7": (recess, [(Z_, F(1)), (Z_, F(3, 2)), (Z_, F(1, 2))]),
    "F8": (lambda: holed_plate(-4., 4., -2., 2., [(-3., -2., -1., 1.), (2., 3., -1., 1.)]), [(Z_, F(1, 2))]),
    "F9": (lambda: box(), [(sign * np_, F(sign, 2)) for np_ in (X_, Y_, Z_) for sign in (1, -1)]),
}
FIXTURES["F9"] = (FIXTURES["F9"][0], [(tuple(s * c for c in axis), F(s, 2))
                                      for axis in (X_, Y_, Z_) for s in (1, -1)])

_CASES: dict = {}


def case(name):
    if name not in _CASES:
        _CASES[name] = Case(FIXTURES[name][0])
    return _CASES[name]


AGREEMENT = [(name, n, d) for name, (_m, planes) in FIXTURES.items() for n, d in planes]


@pytest.mark.parametrize("name,n,d", AGREEMENT, ids=[f"{a}-{b}-{c}" for a, b, c in AGREEMENT])
def test_verifier_agrees_with_the_provider(name, n, d):
    c = case(name)
    q, r, verified = c.run(n, d, accuracy={"V": 1e-9, "A": 1e-9})
    assert isinstance(verified, VerifiedHydrostatics)
    assert verified.result is r and verified.face_ids == tuple(c.faces)
    assert verified.work["operations"] > 0


def loops_of(result, side):
    t = result.section(side).topology
    return (t.outer_count, t.hole_count, len(t.components))


def test_topology_of_the_manifest_fixtures_is_the_expected_one():
    # F4 two components at y <= 2, one at y <= 1/2; F5 one hole; F6 below one loop above two; F7 hole above only; F8 two holes
    c = case("F4")
    assert loops_of(c.run(Y_, F(2))[1], "plane") == (2, 0, 2)
    assert loops_of(c.run(Y_, F(1, 2))[1], "plane") == (1, 0, 1)
    assert loops_of(case("F5").run(Z_, F(1, 2))[1], "plane") == (1, 1, 1)
    r = case("F6").run(Z_, F(0))[1]
    assert not r.generic and loops_of(r, "below") == (1, 0, 1) and loops_of(r, "above") == (2, 0, 2)
    r = case("F7").run(Z_, F(1))[1]
    assert loops_of(r, "below") == (1, 0, 1) and loops_of(r, "above") == (1, 1, 1)
    assert loops_of(case("F8").run(Z_, F(1, 2))[1], "plane") == (1, 2, 1)


def test_states_of_the_box_at_its_faces():
    c = case("F1")
    r = c.run(Z_, F(0))[1]
    assert (r.occupancy, r.section("below").occupancy, r.section("below").section,
            r.section("above").section) == ("EMPTY", "EMPTY", "ABSENT", "PRESENT")
    r = c.run(Z_, F(5))[1]
    assert (r.occupancy, r.section("below").section, r.section("above").occupancy,
            r.section("above").section) == ("FULL", "PRESENT", "FULL", "ABSENT")
    r = c.run(Z_, F(-1))[1]
    assert r.generic and r.occupancy == "EMPTY" and r.section("plane").section == "ABSENT"


def rotation(heel, trim):
    ch, sh, ct, st = math.cos(heel), math.sin(heel), math.cos(-trim), math.sin(-trim)
    rx = [[1, 0, 0], [0, ch, -sh], [0, sh, ch]]
    ry = [[ct, 0, st], [0, 1, 0], [-st, 0, ct]]
    return [[sum(ry[i][k] * rx[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


@pytest.mark.parametrize("name,d", [("F1", 1.7), ("F2", 0.3), ("F4", 0.8)])
@pytest.mark.parametrize("heel,trim", [(0.1, 0.02), (0.5, 0.02), (-0.3, 0.0)])
def test_planes_in_the_consumers_convention_verify(name, d, heel, trim):
    r_ = rotation(heel, trim)
    q = HalfSpaceQuery(normal=tuple(r_[2]), offset=d, waterplane_axes=(tuple(r_[0]), tuple(r_[1])),
                       forms=ALL_FORMS)
    c = case(name)
    result = query_certified_hydrostatics(c.prepared, q)
    assert c.verify(q, result).result is result


def test_exact_only_and_binary64_only_requests_verify():
    c = case("F2")
    for forms in (("exact",), ("binary64",), ("decimal",), ("exact", "decimal")):
        q, r, _v = c.run(Z_, F(1, 2), forms=forms, accuracy={"A": 1e-6})
        assert r.section("plane").quantity("A").numerator is None or "exact" in forms


# ------------------------------------------------------------------ attacks


def with_proof(result, edit=None):
    """Regenerate the proof from the (altered) result object; ``edit`` may alter the proof document further."""
    doc = json.loads(result.proof.decode())
    doc["result"] = _claim_of(result)
    if edit:
        edit(doc)
    return replace(result, proof=json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def rewrite(result, edit):
    doc = json.loads(result.proof.decode())
    edit(doc)
    return replace(result, proof=json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def alter_quantity(result, getter, **changes):
    """Result with one QuantityResult replaced (``getter`` is 'V', 'Jx', or ('plane', 'A'))."""
    if isinstance(getter, str):
        if getter == "V":
            return replace(result, volume=replace(result.volume, **changes))
        moments = tuple(replace(m, **changes) if m.name == getter else m for m in result.moments)
        return replace(result, moments=moments)
    side, name = getter
    sections = []
    for s in result.sections:
        if s.side == side:
            s = replace(s, quantities=tuple(replace(q, **changes) if q.name == name else q for q in s.quantities))
        sections.append(s)
    return replace(result, sections=tuple(sections))


@pytest.fixture(scope="module")
def box_case():
    return Case(lambda: box((0., 0., 0.), (2., 3., 5.)))


@pytest.fixture(scope="module")
def box_result(box_case):
    q, r = box_case.query(Z_, F(5, 2), accuracy={"V": F(1, 10 ** 12), "A": F(1, 10 ** 12)})
    return q, r


def test_genuine_result_verifies(box_case, box_result):
    q, r = box_result
    assert box_case.verify(q, r).result is r


def test_attack_result_of_another_query_with_equal_quantities():
    c = Case(lambda: box((0., 0., 0.), (2., 3., 5.)))
    q6, r6 = c.query(Z_, F(6))
    q7, r7 = c.query(Z_, F(7))
    assert r6.volume.numerator == r7.volume.numerator and r6.occupancy == r7.occupancy == "FULL"
    failed(lambda: c.verify(q7, r6), "query")
    failed(lambda: c.verify(q7, with_proof(r6)), "query")
    # a result whose bound query was edited to match, with the proof regenerated
    forged = replace(r6, query=q7)
    failed(lambda: c.verify(q7, forged), "query")
    failed(lambda: c.verify(q7, with_proof(forged)), "query")


@pytest.mark.parametrize("regenerate", [False, True])
def test_attack_altered_values(box_case, box_result, regenerate):
    q, r = box_result
    fix = (lambda x: with_proof(x)) if regenerate else (lambda x: x)
    v = r.volume
    failed(lambda: box_case.verify(q, fix(alter_quantity(r, "V", numerator=v.numerator + F(1, 1000)))), "value")
    failed(lambda: box_case.verify(q, fix(alter_quantity(r, "Jz", numerator=F(0)))), "value")
    a = r.section("plane").quantity("A")
    failed(lambda: box_case.verify(q, fix(alter_quantity(r, ("plane", "A"), numerator=a.numerator + 1))), "value")
    failed(lambda: box_case.verify(q, fix(alter_quantity(r, ("plane", "Iuu"), norm_squared=F(2)))), "value")


@pytest.mark.parametrize("regenerate", [False, True])
def test_attack_altered_bounds(box_case, box_result, regenerate):
    q, r = box_result
    fix = (lambda x: with_proof(x)) if regenerate else (lambda x: x)
    lo, hi = r.section("plane").quantity("Iuu").binary64
    shifted = math.nextafter(hi, math.inf)
    bad = alter_quantity(r, ("plane", "Iuu"), binary64=(shifted, shifted))
    failed(lambda: box_case.verify(q, fix(bad)), "enclosure")
    dlo, dhi, w = r.section("plane").quantity("Iuu").decimal
    bad = alter_quantity(r, ("plane", "Iuu"), decimal=("8.000000000001", "8.000000000001", w))
    failed(lambda: box_case.verify(q, fix(bad)), "enclosure")


@pytest.mark.parametrize("regenerate", [False, True])
def test_attack_altered_states(box_case, box_result, regenerate):
    q, r = box_result
    fix = (lambda x: with_proof(x)) if regenerate else (lambda x: x)
    failed(lambda: box_case.verify(q, fix(replace(r, occupancy="FULL"))), "state")
    sections = tuple(replace(s, occupancy="FULL") for s in r.sections)
    failed(lambda: box_case.verify(q, fix(replace(r, sections=sections))), "state")
    sections = tuple(replace(s, section="ABSENT", quantities=None, topology=None) for s in r.sections)
    failed(lambda: box_case.verify(q, fix(replace(r, sections=sections))), "state")


@pytest.mark.parametrize("regenerate", [False, True])
def test_attack_altered_accuracy_flag(box_case, box_result, regenerate):
    q, r = box_result
    fix = (lambda x: with_proof(x)) if regenerate else (lambda x: x)
    assert r.volume.accuracy_met is True
    failed(lambda: box_case.verify(q, fix(alter_quantity(r, "V", accuracy_met=None))), "accuracy flag")
    failed(lambda: box_case.verify(q, fix(alter_quantity(r, "V", accuracy_met=False))), "accuracy flag")
    assert r.section("plane").quantity("Mu").accuracy_met is None
    failed(lambda: box_case.verify(q, fix(alter_quantity(r, ("plane", "Mu"), accuracy_met=True))), "accuracy flag")


@pytest.mark.parametrize("regenerate", [False, True])
def test_attack_altered_topology(regenerate):
    c = case("F5")
    q, r = c.query(Z_, F(1, 2))
    fix = (lambda x: with_proof(x)) if regenerate else (lambda x: x)
    section = r.section("plane")
    topo = section.topology

    def with_topology(t):
        return replace(r, sections=(replace(section, topology=t),))

    swapped = Topology(loops=tuple(TopologyLoop(l.identity, "outer" if l.role == "hole" else "hole")
                                   for l in topo.loops), components=topo.components)
    failed(lambda: c.verify(q, fix(with_topology(swapped))), "topology")
    ident = tuple(topo.loops[0].identity[:-1]) + (topo.loops[0].identity[-1] + 1000,)
    renamed = Topology(loops=(TopologyLoop(ident, topo.loops[0].role),) + topo.loops[1:], components=topo.components)
    failed(lambda: c.verify(q, fix(with_topology(renamed))), "topology")
    dropped = Topology(loops=topo.loops[:1], components=topo.components)
    failed(lambda: c.verify(q, fix(with_topology(dropped))), "topology")
    split = Topology(loops=topo.loops, components=tuple((o, ()) for o, _h in topo.components))
    failed(lambda: c.verify(q, fix(with_topology(split))), "topology")
    failed(lambda: c.verify(q, fix(with_topology(None))), "topology")


def test_attack_deleted_forms_and_flags_with_regenerated_proof(box_case, box_result):
    q, r = box_result
    stripped = alter_quantity(r, "V", decimal=None)
    failed(lambda: box_case.verify(q, with_proof(stripped)), "request")
    section = r.section("plane")
    no_decimal = replace(section, quantities=tuple(replace(x, decimal=None) for x in section.quantities))
    failed(lambda: box_case.verify(q, with_proof(replace(r, sections=(no_decimal,)))), "request")
    failed(lambda: box_case.verify(q, with_proof(alter_quantity(r, "Jx", binary64=None))), "request")
    failed(lambda: box_case.verify(q, with_proof(alter_quantity(r, ("plane", "A"), numerator=None, norm_squared=None))),
           "request")
    # a form that was not requested must not appear
    q_exact, r_exact = box_case.query(Z_, F(5, 2), forms=("exact",))
    failed(lambda: box_case.verify(q_exact, with_proof(alter_quantity(r_exact, "V", binary64=(1.0, 1.0)))), "request")
    # a removed quantity / removed section / removed moment
    short = replace(section, quantities=section.quantities[:-1])
    failed(lambda: box_case.verify(q, with_proof(replace(r, sections=(short,)))), "request")
    failed(lambda: box_case.verify(q, with_proof(replace(r, moments=r.moments[:2]))), "request")
    failed(lambda: box_case.verify(q, with_proof(replace(r, sections=()))), "state")
    # accuracy field edited away
    failed(lambda: box_case.verify(q, with_proof(alter_quantity(r, "V", accuracy=None))), "request")


def test_attack_enclosure_wider_than_requested():
    c = case("F9")
    q, r = c.query(Z_, F(2), forms=("binary64", "decimal"), accuracy={"V": F(1, 10 ** 12)})   # fully submerged unit cube
    assert r.volume.numerator == 1
    failed(lambda: c.verify(q, with_proof(alter_quantity(r, "V", binary64=(0.0, 2.0)))), "enclosure")
    failed(lambda: c.verify(q, with_proof(alter_quantity(r, "V", binary64=(0.0, 2.0), accuracy_met=True))), "enclosure")
    wide = alter_quantity(r, "V", decimal=("0.9", "1.1", F(1, 10 ** 12)))
    failed(lambda: c.verify(q, with_proof(wide)), "enclosure")
    wrong_width = alter_quantity(r, "V", decimal=("1.0", "1.0", F(1, 10)))
    failed(lambda: c.verify(q, with_proof(wrong_width)), "enclosure")


def test_attack_non_adjacent_binary64_endpoints():
    c = case("F2")
    q, r = c.query(Z_, F(1, 2), forms=("binary64",))
    lo, hi = r.volume.binary64
    assert math.nextafter(lo, math.inf) == hi
    wide = (math.nextafter(lo, -math.inf), hi)
    failed(lambda: c.verify(q, with_proof(alter_quantity(r, "V", binary64=wide))), "enclosure")
    off = (lo, math.nextafter(hi, math.inf))
    failed(lambda: c.verify(q, with_proof(alter_quantity(r, "V", binary64=off))), "enclosure")
    exact_pretender = (lo, lo)
    failed(lambda: c.verify(q, with_proof(alter_quantity(r, "V", binary64=exact_pretender))), "enclosure")
    failed(lambda: c.verify(q, with_proof(alter_quantity(r, "V", binary64=(math.nan, hi)))), "enclosure")


def test_attack_altered_contacts():
    c = case("F1")
    q, r = c.query(Z_, F(5))
    assert r.contacts.vertices
    for contacts in (replace(r.contacts, vertices=r.contacts.vertices[1:]),
                     replace(r.contacts, edges=r.contacts.edges[1:]),
                     replace(r.contacts, faces=()),
                     replace(r.contacts, vertices=r.contacts.vertices + (999,))):
        failed(lambda: c.verify(q, with_proof(replace(r, contacts=contacts))), "value")
        failed(lambda: c.verify(q, replace(r, contacts=contacts)), "value")
    failed(lambda: c.verify(q, with_proof(replace(r, generic=True))), "value")


def test_attack_substituted_model_and_face_set(box_case, box_result):
    q, r = box_result
    other_model, other_faces = box((0., 0., 0.), (2., 3., 5.))
    assert tuple(other_faces) == tuple(box_case.faces)
    failed(lambda: verify_certified_hydrostatics(other_model, other_faces, q, r, box_case.expected), "model")
    scaled_model, scaled_faces = box((0., 0., 0.), (2., 3., 6.))
    failed(lambda: verify_certified_hydrostatics(scaled_model, scaled_faces, q, r, box_case.expected), "snapshot")
    faces = list(box_case.faces)
    failed(lambda: verify_certified_hydrostatics(box_case.model, faces[:-1], q, r, box_case.expected), "face set")
    failed(lambda: verify_certified_hydrostatics(box_case.model, faces[::-1], q, r, box_case.expected), "face set")
    # an unrelated edit of the model advances the revision
    c = Case(lambda: box())
    q, r = c.query(Z_, F(1, 2))
    c.model.add_point(9., 9., 9.)
    failed(lambda: c.verify(q, r), "model")


def test_attack_stale_snapshot():
    c = Case(lambda: box())
    q, r = c.query(Z_, F(1, 2))
    c.verify(q, r)
    face = c.model.faces[c.faces[0]]
    s = face.surface
    c.model.set_face_support_surface(face.id, Plane(s.origin + s.u_vector, s.u_vector, s.v_vector))   # same plane
    failed(lambda: c.verify(q, r), "snapshot")
    c2 = Case(lambda: tetra())
    q, r = c2.query(Z_, F(1, 2))
    c2.model.move_point(4, 0.0, 0.0, 2.0)
    failed(lambda: c2.verify(q, r), "snapshot")


def test_attack_artifact_identity(box_case, box_result):
    q, r = box_result
    failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["artifact"].__setitem__("python", "CPython 2.7.18"))),
           "artifact")
    failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["artifact"].__setitem__(
        "dependencies", [["numpy", "0.0.1"]]))), "artifact")
    failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["artifact"].__setitem__("distribution_version", "9.9"))),
           "artifact")
    failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["artifact"].__setitem__("package_version", "9.9"))),
           "artifact")
    files = [list(f) for f in box_case.expected["artifact_files"]]
    files[0][1] = "0" * 64
    wrong = dict(box_case.expected, artifact_files=files)
    failed(lambda: verify_certified_hydrostatics(box_case.model, box_case.faces, q, r, wrong), "artifact")
    # the proof claims other hashes, regenerated, and the consumer expects those too: the running files disagree
    forged = rewrite(r, lambda d: d["artifact"].__setitem__("files", files))
    failed(lambda: verify_certified_hydrostatics(box_case.model, box_case.faces, q, forged, wrong), "artifact")
    failed(lambda: verify_certified_hydrostatics(box_case.model, box_case.faces, q, forged, box_case.expected),
           "artifact")


def test_attack_convention_policy_and_model_fields(box_case, box_result):
    q, r = box_result
    failed(lambda: box_case.verify(q, r) if False else verify_certified_hydrostatics(
        box_case.model, box_case.faces, q, r, dict(box_case.expected, convention_version="2")), "convention")
    failed(lambda: verify_certified_hydrostatics(box_case.model, box_case.faces, q, r,
                                                 dict(box_case.expected, units="ft")), "convention")
    failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["convention"].__setitem__("units", "ft"))), "convention")
    failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["convention"].__setitem__("moment_order", ["A"]))),
           "convention")
    policy = CertifiedHydrostaticsPolicy(max_proof_bytes=box_case.prepared.policy.max_proof_bytes + 1)
    failed(lambda: verify_certified_hydrostatics(box_case.model, box_case.faces, q, r,
                                                 dict(box_case.expected, policy=policy)), "policy")
    failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["preparation_policy"].__setitem__("max_integer_digits", 5))),
           "policy")
    for key, value in (("revision", 999), ("model_id", "other"), ("document_checksum", "0" * 64)):
        failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["model"].__setitem__(key, value))), "model")
    failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["model"].__setitem__("snapshot_digest", "0" * 64))),
           "snapshot")
    failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["model"].__setitem__("face_ids", [1, 2]))), "face set")
    failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["query"].__setitem__("offset", [3, 1]))), "query")


def test_malformed_and_oversized_proofs(box_case, box_result):
    q, r = box_result
    for junk in (b"", b"not json", b"[]", b"{}", b"\xff\xfe", json.dumps({"artifact": {}}).encode(),
                 r.proof[:-5], r.proof.replace(b'"revision":', b'"revision":0.5,"x":', 1)):
        error = refusal_of(lambda: box_case.verify(q, replace(r, proof=junk)))
        assert error.category == C.MALFORMED_PROOF, (junk[:20], error)
        assert error.phase == "verify"
    assert refusal_of(lambda: box_case.verify(q, replace(r, proof="text"))).category == C.MALFORMED_PROOF
    small = dict(box_case.expected, policy=CertifiedHydrostaticsPolicy(max_proof_bytes=len(r.proof) - 1))
    error = refusal_of(lambda: verify_certified_hydrostatics(box_case.model, box_case.faces, q, r, small))
    assert error.category == C.MALFORMED_PROOF
    ok_size = dict(box_case.expected, policy=CertifiedHydrostaticsPolicy(max_proof_bytes=len(r.proof)))
    assert refusal_of(lambda: verify_certified_hydrostatics(box_case.model, box_case.faces, q, r, ok_size)).category \
        == C.VERIFICATION_FAILED   # size accepted; the policy then differs from the proof's
    digits = dict(box_case.expected, policy=CertifiedHydrostaticsPolicy(max_integer_digits=10))
    error = refusal_of(lambda: verify_certified_hydrostatics(box_case.model, box_case.faces, q, r, digits))
    assert error.category == C.MALFORMED_PROOF
    huge = r.proof.replace(b'"revision":', b'"revision":' + b"9" * 200 + b',"x":', 1)
    mid = dict(box_case.expected, policy=CertifiedHydrostaticsPolicy(max_integer_digits=100))
    error = refusal_of(lambda: verify_certified_hydrostatics(box_case.model, box_case.faces, q, replace(r, proof=huge), mid))
    assert error.category == C.MALFORMED_PROOF and "digit" in error.detail
    # the token is rejected before conversion: a limit below Python's own conversion limit is not needed
    giant = r.proof.replace(b'"revision":', b'"revision":' + b"9" * 6000 + b',"x":', 1)
    error = refusal_of(lambda: box_case.verify(q, replace(r, proof=giant)))
    assert error.category == C.MALFORMED_PROOF


def test_non_canonical_but_equal_proof_is_refused(box_case, box_result):
    q, r = box_result
    pretty = json.dumps(json.loads(r.proof), indent=1, sort_keys=True).encode()
    error = refusal_of(lambda: box_case.verify(q, replace(r, proof=pretty)))
    assert error.category == C.MALFORMED_PROOF            # whitespace is not part of the canonical grammar
    reordered = json.dumps(dict(reversed(list(json.loads(r.proof).items()))), separators=(",", ":")).encode()
    assert reordered != r.proof
    failed(lambda: box_case.verify(q, replace(r, proof=reordered)), "value")


# ------------------------------------------------------------------ resource refusals during verification


def test_budget_exhaustion_with_work_counts():
    c = case("F6")
    q, r = c.query(Z_, F(0), forms=ALL_FORMS)
    total = c.verify(q, r).work["operations"]
    assert c.verify(q, r, budget=WorkBudget(total, None)).work["operations"] == total
    for limit in (0, 1, 5, 50, total // 3, total // 2, total - 1):
        budget = WorkBudget(limit, None)
        error = refusal_of(lambda: c.verify(q, r, budget=budget))
        assert error.category == C.BUDGET_EXHAUSTED and error.phase == "verify"
        assert error.work["operations"] <= limit and error.work == budget.counts()


def test_budget_is_shared_and_reports_units_by_kind():
    c = case("F2")
    q, r = c.query(Z_, F(1, 2))
    budget = WorkBudget.unlimited()
    one = c.verify(q, r, budget=budget).work["operations"]
    two = c.verify(q, r, budget=budget).work["operations"]
    assert two == 2 * one
    kinds = budget.counts()["units"]
    for kind in ("snapshot_read", "package_hash_chunk", "edge_crossing", "green_edge", "box_test", "proof_element"):
        assert kinds[kind] > 0


def test_cancellation_stops_within_one_unit():
    c = case("F4")
    q, r = c.query(Y_, F(2))
    calls = []

    def check():
        calls.append(1)
        return len(calls) > 40

    budget = WorkBudget.unlimited()
    error = refusal_of(lambda: c.verify(q, r, budget=budget, cancellation_check=check))
    assert error.category == C.CANCELLED and error.phase == "verify"
    assert len(calls) == 41 and error.work["operations"] == 40      # exactly one operation per checkpoint
    assert error.work == budget.counts()


def test_deadline_stops_verification():
    c = case("F4")
    q, r = c.query(Y_, F(2))
    ticks = itertools.count()
    budget = WorkBudget(None, None, deadline=25, clock=lambda: next(ticks))
    error = refusal_of(lambda: c.verify(q, r, budget=budget))
    assert error.category == C.DEADLINE_EXCEEDED and error.phase == "verify"
    assert next(ticks) == 26 and error.work["operations"] > 0      # stopped at the first unit past the deadline
    ticks = itertools.count()
    assert c.verify(q, r, budget=WorkBudget(None, None, deadline=10 ** 9, clock=lambda: next(ticks))).work["operations"]


def test_bit_length_cap_during_verification():
    c = case("F2")
    q, r = c.query(Z_, F(1, 2))
    for bits in (0, 1, 2, 8):
        budget = WorkBudget(None, bits)
        error = refusal_of(lambda: c.verify(q, r, budget=budget))
        assert error.category == C.BIT_LENGTH_EXCEEDED and error.phase == "verify"
        assert error.work["operations"] > 0 or error.work["max_bits_seen"] > bits
    ample = WorkBudget(None, 4096)
    assert c.verify(q, r, budget=ample).work["max_bits_seen"] <= 4096
    assert c.verify(q, r, budget=ample).work["max_bits_seen"] > 8


def test_invalid_basis_cannot_be_posed():
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        HalfSpaceQuery(normal=X_, offset=F(1), waterplane_axes=((1, 0, 0), (0, 1, 0)))
    assert info.value.category == C.INVALID_QUERY


# ------------------------------------------------------------------ CH1c review round 1: regression tests


@pytest.mark.parametrize("text", ["5.0\n", "5.0 ", " 5.0", "+5.0", "5.", ".5", "5e-1", "0x5", "５.0", "5.0\x00", ""])
def test_decimal_bounds_must_match_the_grammar_in_full(text):
    c = case("F9")
    q, r = c.query(Z_, F(1, 2), forms=("decimal",))
    assert r.volume.decimal[0].startswith("0.5")
    forged = alter_quantity(r, "V", decimal=(text, text, F(1, 10 ** 12)))
    failed(lambda: c.verify(q, with_proof(forged)), "enclosure")


def test_convention_block_is_compared_exactly(box_case, box_result):
    q, r = box_result
    for key, value in (("half_space", "n.r >= d"), ("body_frame", "rotated"), ("area_element", "x"),
                       ("limits", "below: n.r = d + eps"), ("loop_identity", "ids"),
                       ("moment_order", ["Mu", "A", "Mv", "Iuu", "Ivv", "Iuv"])):
        failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["convention"].__setitem__(key, value))), "convention")
    failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["convention"].__setitem__("extra", "x"))), "convention")
    failed(lambda: box_case.verify(q, rewrite(r, lambda d: d["convention"].pop("limits"))), "convention")


def test_every_checkpoint_is_one_operation_including_parsing_and_hashing():
    c = case("F6")
    q, r = c.query(Z_, F(0), forms=ALL_FORMS)
    total = c.verify(q, r).work["operations"]
    for stop in sorted({1, 2, 3, 7, 50, 333, 1000, total // 4, total // 2, total - 2}):
        calls = []

        def check(stop=stop, calls=calls):
            calls.append(1)
            return len(calls) > stop

        error = refusal_of(lambda: c.verify(q, r, cancellation_check=check))
        assert error.category == C.CANCELLED and error.work["operations"] == stop, (stop, error.work["operations"])
        budget = WorkBudget(stop, None)
        error = refusal_of(lambda: c.verify(q, r, budget=budget))
        assert error.category == C.BUDGET_EXHAUSTED and error.work["operations"] == stop
    budget = WorkBudget.unlimited()
    c.verify(q, r, budget=budget)
    assert budget.counts()["units"]["proof_element"] > 100         # the parser is metered element by element


def test_bit_cap_covers_query_values_and_intermediate_arithmetic():
    c = case("F9")
    q = HalfSpaceQuery(normal=(1, 0, 0), offset=F(1, 2), waterplane_origin=(F(1, 2), 0, 0),
                       waterplane_axes=((2 ** 10000, 0, 1), (0, 1, 0)), forms=("exact",))
    r = query_certified_hydrostatics(c.prepared, q)
    assert c.verify(q, r).work["max_bits_seen"] > 64
    error = refusal_of(lambda: c.verify(q, r, budget=WorkBudget(None, 64)))
    assert error.category == C.BIT_LENGTH_EXCEEDED and error.phase == "verify"
    # intermediate products: the cap must bind at the largest value any operation forms
    q2 = make_query(Z_, F(1, 2))
    r2 = query_certified_hydrostatics(c.prepared, q2)
    seen = c.verify(q2, r2).work["max_bits_seen"]
    assert refusal_of(lambda: c.verify(q2, r2, budget=WorkBudget(None, seen - 1))).category == C.BIT_LENGTH_EXCEEDED
    assert c.verify(q2, r2, budget=WorkBudget(None, seen)).work["max_bits_seen"] == seen


def test_verification_inside_an_open_transaction_is_refused():
    c = Case(lambda: box())
    q, r = c.query(Z_, F(1, 2))
    with c.model.transaction():
        error = refusal_of(lambda: c.verify(q, r))
    assert error.category == C.CONCURRENT_MUTATION and error.phase == "verify"
    assert c.verify(q, r).result is r


def test_huge_integers_do_not_depend_on_pythons_digit_limit():
    import sys
    c = case("F9")
    q, r = c.query(Z_, F(1, 2), forms=("exact",), accuracy={"V": F(1, 10 ** 5000)})
    assert len(r.proof) > 10000
    assert c.verify(q, r).result is r
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(640)
    try:
        assert c.verify(q, r).result is r
        digits = dict(c.expected, policy=CertifiedHydrostaticsPolicy(max_integer_digits=4000))
        error = refusal_of(lambda: verify_certified_hydrostatics(c.model, c.faces, q, r, digits))
        assert error.category == C.MALFORMED_PROOF and "digit" in error.detail
    finally:
        sys.set_int_max_str_digits(old)


def test_non_ascii_units_are_written_as_utf8():
    c = Case(lambda: box(), units="mètres")
    q, r = c.query(Z_, F(1, 2))
    assert "mètres".encode("utf-8") in r.proof
    assert c.verify(q, r).result is r
    wrong = dict(c.expected, units="metres")
    failed(lambda: verify_certified_hydrostatics(c.model, c.faces, q, r, wrong), "convention")


MALFORMED_EDITS = [
    lambda d: d.__setitem__("model", None), lambda d: d.__setitem__("artifact", None),
    lambda d: d.__setitem__("convention", []), lambda d: d.__setitem__("preparation_policy", 5),
    lambda d: d.__setitem__("query", "x"), lambda d: d.__setitem__("result", None),
    lambda d: d["model"].__setitem__("face_ids", None), lambda d: d["model"].pop("revision"),
    lambda d: d["artifact"].__setitem__("files", [1, 2]), lambda d: d["artifact"].__setitem__("files", {}),
    lambda d: d["artifact"].__setitem__("dependencies", "numpy"), lambda d: d["artifact"].pop("python"),
    lambda d: d["preparation_policy"].pop("max_proof_bytes"), lambda d: d.__setitem__("extra", 1),
]


@pytest.mark.parametrize("edit", MALFORMED_EDITS, ids=range(len(MALFORMED_EDITS)))
def test_malformed_nested_shapes_refuse_with_phase_and_work(box_case, box_result, edit):
    q, r = box_result
    error = refusal_of(lambda: box_case.verify(q, rewrite(r, edit)))
    assert error.category == C.MALFORMED_PROOF and error.phase == "verify" and error.work["operations"] > 0


def test_fabricated_work_counts_are_refused(box_case, box_result):
    q, r = box_result
    for work in ({"operations": -100, "units": {"green_edge": -200}},
                 {"operations": 5, "units": {"green_edge": 4}},
                 {"operations": 1.0, "units": {"green_edge": 1.0}},
                 {"operations": True, "units": {"a": True}},
                 {"operations": 0, "units": {"a": 1, "b": -1}}):
        forged = replace(r, work=work)
        failed(lambda: box_case.verify(q, with_proof(forged)), "value")
        failed(lambda: box_case.verify(q, forged), "value")
    for work in ({"operations": 3}, {"operations": 3, "units": []}, None, 5):
        failed(lambda: box_case.verify(q, replace(r, work=work)), "value")
