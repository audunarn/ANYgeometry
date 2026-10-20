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
    assert verified.result == r and verified.face_ids == tuple(c.faces)
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
    assert c.verify(q, result).result == result


def test_exact_only_and_binary64_only_requests_verify():
    c = case("F2")
    for forms in (("exact",), ("binary64",), ("decimal",), ("exact", "decimal")):
        q, r, _v = c.run(Z_, F(1, 2), forms=forms, accuracy={"A": 1e-6})
        assert r.section("plane").quantity("A").numerator is None or "exact" in forms


# ------------------------------------------------------------------ attacks


def _pair(x):
    return [x.numerator, x.denominator]


def _claim_quantity(item):
    out = {"accuracy": None if item.accuracy is None else _pair(item.accuracy), "accuracy_met": item.accuracy_met}
    if item.numerator is not None:
        out["exact"] = {"numerator": _pair(item.numerator),
                        "norm_squared": None if item.norm_squared is None else _pair(item.norm_squared)}
    if item.binary64 is not None:
        out["binary64"] = {"lo": float.hex(item.binary64[0]), "hi": float.hex(item.binary64[1])}
    if item.decimal is not None:
        out["decimal"] = {"lo": item.decimal[0], "hi": item.decimal[1], "width": _pair(item.decimal[2])}
    return out


def _claim_section(section):
    topology = None
    if section.topology is not None:
        topology = {"loops": [{"identity": list(l.identity), "role": l.role} for l in section.topology.loops],
                    "components": [{"outer": list(o), "holes": [list(h) for h in hs]}
                                   for o, hs in section.topology.components]}
    quantities = None
    if section.quantities is not None:
        quantities = {q.name: _claim_quantity(q) for q in section.quantities}
    return {"occupancy": section.occupancy, "section": section.section, "quantities": quantities,
            "topology": topology}


def _claim_of(result):
    """The claim of a result object, built test-side straight from its attributes (independent of the verifier)."""
    return {"occupancy": result.occupancy, "generic": result.generic,
            "contacts": {"vertices": list(result.contacts.vertices), "edges": list(result.contacts.edges),
                         "faces": list(result.contacts.faces)},
            "volume": _claim_quantity(result.volume),
            "moments": {m.name: _claim_quantity(m) for m in result.moments},
            "sections": {s.side: _claim_section(s) for s in result.sections},
            "work": {"operations": result.work["operations"], "units": dict(result.work["units"])}}


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
    assert box_case.verify(q, r).result == r


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
    assert c.verify(q, r).result == r


def test_huge_integers_do_not_depend_on_pythons_digit_limit():
    import sys
    c = case("F9")
    q, r = c.query(Z_, F(1, 2), forms=("exact",), accuracy={"V": F(1, 10 ** 5000)})
    assert len(r.proof) > 10000
    assert c.verify(q, r).result == r
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(640)
    try:
        assert c.verify(q, r).result == r
        digits = dict(c.expected, policy=CertifiedHydrostaticsPolicy(max_integer_digits=4000))
        error = refusal_of(lambda: verify_certified_hydrostatics(c.model, c.faces, q, r, digits))
        assert error.category == C.MALFORMED_PROOF and "digit" in error.detail
    finally:
        sys.set_int_max_str_digits(old)


def test_non_ascii_units_are_written_as_utf8():
    c = Case(lambda: box(), units="mètres")
    q, r = c.query(Z_, F(1, 2))
    assert "mètres".encode("utf-8") in r.proof
    assert c.verify(q, r).result == r
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


# ------------------------------------------------------------------ CH1c review round 2: regression tests

import sys as _sys
import anygeometry.certified_hydrostatics._verify as vm


def comb(teeth):
    """Prism over a comb polygon (a base bar with ``teeth`` upright teeth): many edges crossed by one plane."""
    width = 2 * teeth - 1
    pts = [(0., 0.), (float(width), 0.)]
    for j in range(teeth):
        hi = width - 2 * j
        lo = hi - 1
        pts += [(float(hi), 3.), (float(lo), 3.)]
        if j < teeth - 1:
            pts += [(float(lo), 1.), (float(lo - 1), 1.)]
    return extrusion(pts, 1.0)


def max_lines_between_checkpoints(call):
    """Run ``call(check, budget)`` while counting executed lines of the verifier between consecutive cancellation
    callbacks, from the entry of ``verify_certified_hydrostatics`` (ingestion, parsing, type checks included).

    Arithmetic wrappers are not counted (their cost is one rational operation); returns the largest gap."""
    skip = {"operation", "_wrap", "_fit", "fit", "_guarded", "_ordered", "check_value"}
    state = {"count": 0, "gap": 0}
    budget = WorkBudget.unlimited()

    def local(frame, event, arg):
        if event == "line":
            state["count"] += 1
        return local

    def tracer(frame, event, arg):
        code = frame.f_code
        if code.co_filename.endswith("_verify.py") and code.co_name not in skip:
            return local
        return None

    def check():
        state["gap"] = max(state["gap"], state["count"])
        state["count"] = 0
        return False

    previous = _sys.gettrace()
    _sys.settrace(tracer)
    try:
        call(check, budget)
    finally:
        _sys.settrace(previous)
    return max(state["gap"], state["count"])


# a single checkpoint interval may run one bounded predicate (an edge integral, one quantity comparison)
LINE_BOUND = 450


@pytest.mark.parametrize("plane", [(Z_, F(1, 2)), (Y_, F(2)), (Y_, F(1))], ids=["clip", "sort-and-nesting", "contact"])
def test_no_uninterruptible_span_grows_with_the_data(plane):
    c = Case(lambda: comb(10))
    n, d = plane
    q, r = c.query(n, d, forms=("exact",))
    gap = max_lines_between_checkpoints(
        lambda check, budget: c.verify(q, r, budget=budget, cancellation_check=check))
    assert gap <= LINE_BOUND, gap


def test_every_loop_vertex_is_its_own_checkpoint_in_clipping():
    c = Case(lambda: comb(10))
    q, r = c.query(Z_, F(1, 2), forms=("exact",))
    lt_calls = []
    cls = getattr(vm, "_Q", F)
    original = cls.__lt__

    def counting(self, other):
        lt_calls.append(1)
        return original(self, other)

    trail = []

    def check():
        trail.append(len(lt_calls))
        return False

    cls.__lt__ = counting
    try:
        c.verify(q, r, cancellation_check=check)
    finally:
        cls.__lt__ = original
    worst = max(b - a for a, b in zip(trail, trail[1:]))
    assert worst <= 40, worst        # comparisons between two checkpoints: a bounded number, not a loop's worth


def test_requested_accuracies_are_checked_against_the_bit_cap():
    c = case("F9")
    q = make_query(Z_, F(1, 2), forms=("exact",), accuracy={"V": F(1, 10 ** 100)})
    r = query_certified_hydrostatics(c.prepared, q)
    assert c.verify(q, r).work["max_bits_seen"] >= 330
    error = refusal_of(lambda: c.verify(q, r, budget=WorkBudget(None, 64)))
    assert error.category == C.BIT_LENGTH_EXCEEDED and error.phase == "verify"


def test_products_and_sums_are_checked_as_they_are_formed():
    a = 2 ** 63
    c = case("F9")
    q = HalfSpaceQuery(normal=Z_, offset=F(-1), waterplane_axes=((a, a, 0), (a, a + 1, 0)), forms=("exact",))
    r = query_certified_hydrostatics(c.prepared, q)
    assert r.occupancy == "EMPTY"
    assert c.verify(q, r, budget=WorkBudget(None, 200)).result == r
    error = refusal_of(lambda: c.verify(q, r, budget=WorkBudget(None, 64)))
    assert error.category == C.BIT_LENGTH_EXCEEDED
    # the guard sits in the number type itself: sums, products, quotients and ordering products
    gauge = vm._Gauge(vm.Meter(WorkBudget(None, 64)))
    vm._ACTIVE.gauge, vm._ACTIVE.budget = gauge, gauge.budget
    try:
        big = vm._Q(2 ** 40 + 1, 2 ** 40 + 3)
        other = vm._Q(2 ** 40 + 5, 2 ** 40 + 7)
        for operation in (lambda: big * other, lambda: big + other, lambda: big - other, lambda: big / other,
                          lambda: big < other, lambda: big >= other, lambda: 2 ** 40 * big * 2 ** 30):
            with pytest.raises(CertifiedHydrostaticsRefusal) as info:
                operation()
            assert info.value.category == C.BIT_LENGTH_EXCEEDED
        assert (vm._Q(3, 4) * vm._Q(2, 3)) == F(1, 2) and vm._Q(1, 3) < vm._Q(1, 2)
    finally:
        vm._ACTIVE.gauge = vm._ACTIVE.budget = None


def test_long_decimal_enclosures_span_string_windows():
    c = case("F2")
    q, r = c.query(Z_, F(1, 2), forms=("decimal",), accuracy={"V": F(1, 10 ** 5000)})
    assert max(len(x) for x in r.volume.decimal[:2]) > 5000
    assert c.verify(q, r).result == r


def test_long_strings_with_escapes_cross_windows():
    units = ('m\u00e8"\\x\n' * 900) + "end"
    c = Case(lambda: box(), units=units)
    q, r = c.query(Z_, F(1, 2))
    assert len(units) > 4096 + 1000
    assert c.verify(q, r).result == r


def test_decimal_strings_are_not_limited_by_the_integer_digit_limit():
    policy = CertifiedHydrostaticsPolicy(max_integer_digits=8)
    c = Case(lambda: box(), policy=policy)
    accuracy = {name: F(1, 10 ** 6) for name in ("V", "Jx", "Jy", "Jz", "A", "Mu", "Mv", "Iuu", "Ivv", "Iuv")}
    q = HalfSpaceQuery(normal=Z_, offset=F(1, 2), waterplane_origin=(1, 1, 0), waterplane_axes=((1, 0, 0), (0, 1, 0)),
                       absolute_accuracy=accuracy, forms=ALL_FORMS)
    r = query_certified_hydrostatics(c.prepared, q)
    assert r.section("plane").quantity("Mu").decimal[0].startswith("-0.5")
    assert c.verify(q, r).result == r
    # integer tokens remain limited
    tight = dict(c.expected, policy=CertifiedHydrostaticsPolicy(max_integer_digits=4))
    assert refusal_of(lambda: verify_certified_hydrostatics(c.model, c.faces, q, r, tight)).category == C.MALFORMED_PROOF


@pytest.mark.parametrize("regenerate", [False, True])
def test_booleans_and_integers_must_have_their_exact_types(box_case, box_result, regenerate):
    q, r = box_result
    fix = (lambda x: with_proof(x)) if regenerate else (lambda x: x)
    failed(lambda: box_case.verify(q, fix(replace(r, generic=1))), "value")
    failed(lambda: box_case.verify(q, fix(alter_quantity(r, "V", accuracy_met=1))), "accuracy flag")
    failed(lambda: box_case.verify(q, fix(alter_quantity(r, ("plane", "A"), accuracy_met=1))), "accuracy flag")
    failed(lambda: box_case.verify(q, fix(alter_quantity(r, ("plane", "Mu"), accuracy_met=0))), "accuracy flag")
    c = case("F1")
    q5, r5 = c.query(Z_, F(5))
    assert r5.contacts.vertices
    for field_name in ("vertices", "edges", "faces"):
        ids = getattr(r5.contacts, field_name)
        if ids:
            forged = replace(r5.contacts, **{field_name: (float(ids[0]),) + tuple(ids[1:])})
            failed(lambda: c.verify(q5, fix(replace(r5, contacts=forged))), "value")
    topology_case = case("F5")
    qt, rt = topology_case.query(Z_, F(1, 2))
    section = rt.section("plane")
    loop = section.topology.loops[0]
    twisted = Topology(loops=(TopologyLoop((float(loop.identity[0]),) + tuple(loop.identity[1:]), loop.role),)
                       + section.topology.loops[1:], components=section.topology.components)
    failed(lambda: topology_case.verify(qt, fix(replace(rt, sections=(replace(section, topology=twisted),)))),
           "topology")


# ------------------------------------------------------------------ CH1c review round 3: regression tests

import ast as _ast
import pathlib as _pathlib

VERIFY_SOURCE = _pathlib.Path(vm.__file__)

# Functions whose iteration is bounded by a constant (the lint below is local syntax only, see its docstring) (polynomials in epsilon have at most three coefficients, Green
# polynomials at most four); every entry is justified on its line. Nothing else may loop over input-sized data
# without a checkpoint.
ALLOWED_CONSTANT_SIZE = {
    "_padd": "adds two polynomials in epsilon: at most 3 coefficients each",
    "_psub": "subtracts two polynomials in epsilon: at most 3 coefficients each",
    "_pmul": "multiplies two polynomials in epsilon: at most 3 coefficients each",
    "_psign": "sign of a polynomial in epsilon: at most 5 coefficients",
    "_pmul_list": "multiplies edge polynomials of degree <= 2: at most 4 coefficients",
    "_edge_G.integral": "integrates one polynomial of degree <= 3",
}
TICKING_CALLS = {"_tick", "_each", "_tsorted", "_joined"}
DATA_BUILTINS = {"sorted", "sum", "min", "max", "all", "any", "set", "frozenset", "tuple", "list", "dict", "reversed",
                 "enumerate", "zip", "map", "filter", "bytes", "bytearray"}
CHECKED_METHODS = {"extend", "update", "sort", "join"}


def _is_tick(node):
    if isinstance(node, _ast.Call):
        func = node.func
        if isinstance(func, _ast.Name) and func.id in TICKING_CALLS:
            return True
        if isinstance(func, _ast.Attribute) and func.attr == "unit":
            return True
    return False


def _tick_in(nodes):
    return any(_is_tick(n) for node in nodes for n in _ast.walk(node))


def _constant_iterable(node):
    if isinstance(node, (_ast.Tuple, _ast.List, _ast.Set)):
        return not any(isinstance(e, _ast.Starred) for e in node.elts)
    if isinstance(node, _ast.Constant):
        return True
    if (isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) and node.func.id == "range"
            and not node.keywords):
        return all(isinstance(a, _ast.Constant) for a in node.args)
    return False


def checkpoint_violations(allowed=ALLOWED_CONSTANT_SIZE):
    """Local syntax check only: every For/While/comprehension/generator and every data builtin/method call of
    ``_verify.py`` must iterate a constant, sit in an allowlisted constant-size function, or tick (``_tick``/``_each``/
    ``_tsorted``/``_joined``/``.unit``) within the same loop, comprehension or call arguments. It does NOT follow calls
    (a tick inside a callee does not count for the caller's loop), does not see work inside C-level calls or the
    control flow that decides how often a loop runs, and cannot judge a loop whose body ticks only on some paths. The
    only operations exempt from one-unit spans are the two listed bulk operations of design amendment 6b (the model
    document encoding for ``document_checksum`` and one UTF-8 decode of the proof); both are covered by their own tests
    (units charged per model entity before the encoding; the decode is bounded by ``max_proof_bytes``)."""
    tree = _ast.parse(VERIFY_SOURCE.read_text(encoding="utf-8"))
    for parent in _ast.walk(tree):
        for child in _ast.iter_child_nodes(parent):
            child.parent = parent

    def qualified(node):
        names = []
        while node is not None:
            if isinstance(node, (_ast.FunctionDef, _ast.ClassDef)):
                names.append(node.name)
            node = getattr(node, "parent", None)
        return ".".join(reversed(names))

    def inside_ticking_loop(node):
        node = getattr(node, "parent", None)
        while node is not None:
            if isinstance(node, (_ast.For, _ast.While)) and _tick_in(node.body):
                return True
            if isinstance(node, (_ast.ListComp, _ast.SetComp, _ast.DictComp, _ast.GeneratorExp)) and _tick_in([node]):
                return True
            node = getattr(node, "parent", None)
        return False

    found = []
    for node in _ast.walk(tree):
        where = qualified(node)
        if where in allowed:
            continue
        what = None
        if isinstance(node, _ast.For):
            if not (_constant_iterable(node.iter) or _tick_in(node.body) or _tick_in([node.iter])):
                what = "for"
        elif isinstance(node, _ast.While):
            if not _tick_in(node.body):
                what = "while"
        elif isinstance(node, (_ast.ListComp, _ast.SetComp, _ast.DictComp, _ast.GeneratorExp)):
            if not (_tick_in([node]) or all(_constant_iterable(g.iter) for g in node.generators)):
                what = "comprehension"
        elif isinstance(node, _ast.Call):
            func = node.func
            if isinstance(func, _ast.Name) and func.id in DATA_BUILTINS:
                parent = getattr(node, "parent", None)
                comprehension_args = all(
                    isinstance(a, (_ast.ListComp, _ast.GeneratorExp, _ast.SetComp))
                    and all(_constant_iterable(g.iter) for g in a.generators) for a in node.args)
                ok = ((not node.args and not node.keywords) or _tick_in(node.args)
                      or (func.id in ("min", "max") and len(node.args) >= 2)
                      or (isinstance(parent, _ast.Call) and _is_tick(parent))
                      or (node.args and comprehension_args))
                if not ok:
                    what = f"builtin {func.id}"
            elif isinstance(func, _ast.Attribute) and func.attr in CHECKED_METHODS:
                string_join = func.attr == "join" and (isinstance(func.value, _ast.Constant)
                                                       or (isinstance(func.value, _ast.Name)
                                                           and func.value.id == "separator"))
                if func.attr != "join" or string_join:
                    if not (_tick_in(node.args) or inside_ticking_loop(node)):
                        what = f"method {func.attr}"
        if what:
            found.append((node.lineno, where, what))
    return found


def test_every_data_dependent_iteration_of_the_verifier_ticks():
    violations = checkpoint_violations()
    assert not violations, violations


def test_the_constant_size_allowlist_is_minimal():
    without = {where for _line, where, _what in checkpoint_violations(allowed={})}
    assert set(ALLOWED_CONSTANT_SIZE) <= without, set(ALLOWED_CONSTANT_SIZE) - without
    assert without <= set(ALLOWED_CONSTANT_SIZE), without - set(ALLOWED_CONSTANT_SIZE)


def test_the_source_inspection_catches_an_unticked_loop(tmp_path, monkeypatch):
    source = VERIFY_SOURCE.read_text(encoding="utf-8")
    broken = source.replace('        _tick("clip_vertex")' + chr(10), "", 1)
    assert broken != source
    path = tmp_path / "_verify_broken.py"
    path.write_text(broken, encoding="utf-8")
    monkeypatch.setitem(globals(), "VERIFY_SOURCE", path)
    assert any(where == "_clip_loop" for _l, where, _w in checkpoint_violations())


class Liar(F):
    """A Fraction that claims to equal everything."""

    __eq__ = lambda self, other: True
    __ne__ = lambda self, other: False
    __hash__ = F.__hash__


class LiarInt(int):
    __eq__ = lambda self, other: True
    __ne__ = lambda self, other: False
    __hash__ = int.__hash__


class LiarFloat(float):
    __eq__ = lambda self, other: True
    __ne__ = lambda self, other: False
    __hash__ = float.__hash__
    __lt__ = lambda self, other: True
    __gt__ = lambda self, other: True


class LiarStr(str):
    __eq__ = lambda self, other: True
    __ne__ = lambda self, other: False
    __hash__ = str.__hash__


class LiarBytes(bytes):
    __eq__ = lambda self, other: True


def test_subclassed_rationals_cannot_vouch_for_a_wrong_value():
    c = Case(lambda: box())
    q, r = c.query(Z_, F(1, 2), forms=("exact",))
    forged = alter_quantity(r, "V", numerator=Liar(12345))
    for candidate in (forged, with_proof(forged)):
        error = failed(lambda: c.verify(q, candidate), "value")
    forged = alter_quantity(r, "V", norm_squared=Liar(1))
    failed(lambda: c.verify(q, with_proof(forged)), "value")
    forged = alter_quantity(r, ("plane", "A"), numerator=Liar(7))
    failed(lambda: c.verify(q, with_proof(forged)), "value")
    forged = alter_quantity(r, "V", accuracy=Liar(1))
    failed(lambda: c.verify(q, with_proof(forged)), "request")
    # ... and the verified object reports the recomputed value, not the supplied one
    assert c.verify(q, r).exact["V"] == F(1, 2)


def test_subclassed_ints_floats_strings_and_bytes_are_refused():
    c = Case(lambda: box())
    q, r = c.query(Z_, F(1, 2), forms=("exact", "binary64", "decimal"))
    lo, hi = r.section("plane").quantity("Iuu").binary64
    for bad in (LiarFloat(lo), LiarFloat(1.0)):
        forged = alter_quantity(r, ("plane", "Iuu"), binary64=(bad, bad))
        failed(lambda: c.verify(q, with_proof(forged)), "enclosure")
    forged = alter_quantity(r, "V", binary64=(LiarFloat(0.5), LiarFloat(0.5)))
    failed(lambda: c.verify(q, with_proof(forged)), "enclosure")
    contacts = replace(r.contacts, vertices=(LiarInt(3),))
    failed(lambda: c.verify(q, with_proof(replace(r, contacts=contacts))), "value")
    section = r.section("plane")
    topology = section.topology
    loop = topology.loops[0]
    twisted = Topology(loops=(TopologyLoop((LiarInt(loop.identity[0]),) + tuple(loop.identity[1:]), loop.role),)
                       + topology.loops[1:], components=topology.components)
    failed(lambda: c.verify(q, with_proof(replace(r, sections=(replace(section, topology=twisted),)))), "topology")
    failed(lambda: c.verify(q, with_proof(replace(r, occupancy=LiarStr("PARTIAL")))), "state")
    decimal = r.volume.decimal
    failed(lambda: c.verify(q, with_proof(alter_quantity(r, "V", decimal=(LiarStr(decimal[0]), decimal[1], decimal[2])))),
           "enclosure")
    error = refusal_of(lambda: c.verify(q, replace(r, proof=LiarBytes(r.proof))))
    assert error.category == C.MALFORMED_PROOF
    error = refusal_of(lambda: c.verify(q, replace(r, proof=bytearray(r.proof))))
    assert error.category == C.MALFORMED_PROOF


def test_expected_and_query_values_are_normalised_too():
    c = Case(lambda: box())
    q, r = c.query(Z_, F(1, 2))
    with pytest.raises(ValueError):
        verify_certified_hydrostatics(c.model, c.faces, q, r, dict(c.expected, units=LiarStr("m")))
    with pytest.raises(ValueError):
        verify_certified_hydrostatics(c.model, c.faces, q, r, dict(c.expected, convention_version=LiarStr("1")))
    files = [(LiarStr(p), h) for p, h in c.expected["artifact_files"]]
    with pytest.raises(ValueError):
        verify_certified_hydrostatics(c.model, c.faces, q, r, dict(c.expected, artifact_files=files))

    class Policy:
        convention_version = "1"
        max_proof_bytes = LiarInt(10 ** 9)
        max_integer_digits = 100000

    with pytest.raises(ValueError):
        verify_certified_hydrostatics(c.model, c.faces, q, r, dict(c.expected, policy=Policy()))

    class LyingQuery:
        normal = (0, 0, 1)
        waterplane_origin = (0, 0, 0)
        waterplane_axes = ((1, 0, 0), (0, 1, 0))
        absolute_accuracy = ()
        forms = ("exact",)
        offset = Liar(1, 2)

    failed(lambda: verify_certified_hydrostatics(c.model, c.faces, LyingQuery(), r, c.expected), "query")


def test_guarded_rationals_have_no_unchecked_operations():
    gauge = vm._Gauge(vm.Meter(WorkBudget(None, 8)))
    vm._ACTIVE.gauge, vm._ACTIVE.budget = gauge, gauge.budget
    try:
        big, small = vm._q(255), vm._q(F(1, 255))
        for operation in (lambda: big // small, lambda: divmod(big, small), lambda: big % small,
                          lambda: big ** 2, lambda: 2 ** big, lambda: round(big), lambda: math.trunc(big),
                          lambda: math.floor(big), lambda: math.ceil(big), lambda: int(big), lambda: float(big),
                          lambda: big.limit_denominator(3), lambda: pow(big, 2, 7)):
            with pytest.raises(TypeError):
                operation()
        for operation in (lambda: big * big, lambda: big + big, lambda: big - (-big), lambda: -big * big,
                          lambda: big / small, lambda: 1000 / small, lambda: small * 255 * 256):
            with pytest.raises(CertifiedHydrostaticsRefusal) as info:
                operation()
            assert info.value.category == C.BIT_LENGTH_EXCEEDED
        assert big - big == 0 and abs(-small) == small and big == 255      # within the cap everything works
    finally:
        vm._ACTIVE.gauge = vm._ACTIVE.budget = None


def test_bool_is_not_an_integer_in_the_proofs_query():
    c = Case(lambda: box())
    q, r = c.query(Z_, F(1))
    assert json.loads(r.proof)["query"]["offset"] == [1, 1]
    failed(lambda: c.verify(q, rewrite(r, lambda d: d["query"].__setitem__("offset", [True, True]))), "query")
    failed(lambda: c.verify(q, rewrite(r, lambda d: d["query"]["normal"].__setitem__(0, [False, True]))), "query")
    failed(lambda: c.verify(q, rewrite(r, lambda d: d["query"]["waterplane_origin"].__setitem__(2, [False, True]))),
           "query")


def test_huge_identity_does_not_create_a_long_span_between_checkpoints():
    c = case("F5")
    q, r = c.query(Z_, F(1, 2))
    section = r.section("plane")
    topology = section.topology
    loop = topology.loops[0]
    huge = Topology(loops=(TopologyLoop(tuple(range(10000)), loop.role),) + topology.loops[1:],
                    components=topology.components)
    forged = replace(r, sections=(replace(section, topology=huge),))

    def run(check, budget):
        error = refusal_of(lambda: c.verify(q, forged, budget=budget, cancellation_check=check))
        assert error.category == C.VERIFICATION_FAILED and error.sub_reason in ("value", "topology")

    gap = max_lines_between_checkpoints(run)
    assert gap <= LINE_BOUND, gap
    for field_forge in (replace(r, contacts=replace(r.contacts, vertices=tuple(range(10000)))),
                        replace(r, sections=(replace(section, topology=Topology(
                            loops=topology.loops, components=((tuple(range(10000)), ()),))),))):
        def run_other(check, budget, forged=field_forge):
            refusal_of(lambda: c.verify(q, forged, budget=budget, cancellation_check=check))

        assert max_lines_between_checkpoints(run_other) <= LINE_BOUND


# ------------------------------------------------------------------ CH1c review round 4: regression tests

import types as _types
import dataclasses as _dataclasses
import fractions as _fractions


class ShiftingResult:
    """A result proxy that reports the true attributes on the first read and a forged volume afterwards."""

    def __init__(self, real):
        self._real = real
        self._volume_reads = 0

    def __getattr__(self, name):
        value = getattr(self._real, name)
        if name == "volume":
            self._volume_reads += 1
            if self._volume_reads > 1:
                return replace(value, numerator=F(999))
        return value


def test_the_verified_object_holds_only_checked_values_and_forwards_nothing():
    c = Case(lambda: box())
    q, r = c.query(Z_, F(1, 2), forms=("exact", "binary64", "decimal"), accuracy={"V": F(1, 10 ** 6)})
    proxy = ShiftingResult(r)
    verified = c.verify(q, proxy)
    assert verified.exact["V"] == F(1, 2)
    assert verified.volume.numerator == F(1, 2) and verified.result.volume.numerator == F(1, 2)
    assert verified.result is not proxy and verified.result is not r and verified.query is not q
    assert not hasattr(verified, "no_such_attribute")
    for item in (verified.result, verified.query, verified.exact, verified.work):
        assert not isinstance(item, ShiftingResult)
    seen = []

    def walk(value):
        seen.append(value)
        if _dataclasses.is_dataclass(value) and not isinstance(value, type):
            for f in _dataclasses.fields(value):
                walk(getattr(value, f.name))
        elif isinstance(value, (tuple, list)):
            for x in value:
                walk(x)
        elif isinstance(value, dict):
            for x in value.values():
                walk(x)

    walk(verified)
    assert all(x is not proxy and x is not r and x is not q for x in seen)
    assert all(type(x) is not vm._Q for x in seen)            # no guarded internal numbers leak out
    # every public field equals the recomputed values
    plain = c.verify(q, r)
    assert verified.volume == plain.volume == r.volume
    assert verified.moments == r.moments and verified.sections == r.sections
    assert verified.occupancy == r.occupancy and verified.generic is r.generic and verified.contacts == r.contacts
    assert verified.exact["J"] == (F(1, 4), F(1, 4), F(1, 8)) and verified.solid_volume == F(1)
    assert verified.face_ids == tuple(c.faces) and verified.model_id == str(c.model.model_id)
    assert verified.work["operations"] > 0


def test_document_checksum_is_charged_per_model_entity_before_it_runs(monkeypatch):
    import anygeometry.serialization as serialization

    def charged(extra):
        c = Case(lambda: _box_with_isolated_points(extra))
        q, r = c.query(Z_, F(1, 2))
        budget = WorkBudget.unlimited()
        c.verify(q, r, budget=budget)
        return budget.counts()["units"]["document_entity"], c, q, r

    few, *_ = charged(100)
    many, c, q, r = charged(1000)
    assert many - few >= 900 and many >= 1000
    calls = []
    real = serialization.to_dict
    monkeypatch.setattr(serialization, "to_dict", lambda *a, **k: calls.append(1) or real(*a, **k))
    budget = WorkBudget.unlimited()

    def cancel():
        return budget.counts()["units"].get("document_entity", 0) >= 1

    error = refusal_of(lambda: c.verify(q, r, budget=budget, cancellation_check=cancel))
    assert error.category == C.CANCELLED and not calls      # refused before the encoding ran
    c.verify(q, r)
    assert calls                                            # ... which does run when not cancelled


def test_document_charge_covers_every_collection_the_encoder_walks():
    import inspect
    import re
    import anygeometry.serialization as serialization
    source = inspect.getsource(serialization._serialized_model_state)
    walked = set(re.findall(r"\('([a-z_]+)', [A-Z][A-Za-z]+\)", source))
    assert {"vertices", "edges", "faces", "coedges"} <= walked             # the pattern still finds the loop
    assert walked <= set(vm._MODEL_COLLECTIONS) and "features" in vm._MODEL_COLLECTIONS


def _box_with_isolated_points(extra):
    model, faces = box()
    for i in range(extra):
        model.add_point(10. + i, 10., 10.)
    return model, faces


def test_integer_tokens_are_scanned_to_at_most_the_limit_plus_one_digit(monkeypatch):
    c = Case(lambda: box(), policy=CertifiedHydrostaticsPolicy(max_integer_digits=40))
    q, r = c.query(Z_, F(1, 2))
    huge = r.proof.replace(b'"revision":', b'"revision":' + b"9" * 100001 + b',"x":', 1)
    spans = []
    original = vm._number_pattern

    def recording(limit):
        pattern = original(limit)

        class Spy:
            def match(self, text, pos):
                found = pattern.match(text, pos)
                if found is not None:
                    spans.append(found.end() - pos)
                return found

        return Spy()

    monkeypatch.setattr(vm, "_number_pattern", recording)
    error = refusal_of(lambda: c.verify(q, replace(r, proof=huge)))
    assert error.category == C.MALFORMED_PROOF and "digit" in error.detail
    assert max(spans) <= 42, max(spans)          # a sign and at most 41 digits, never the 100,001
    monkeypatch.undo()
    assert c.verify(q, r).result == r


def test_refusal_context_reads_no_outside_data(monkeypatch):
    c = Case(lambda: box())
    _q0, r = c.query(Z_, F(1, 2))
    reads_after = {}

    def run(pairs, cancel_at):
        accuracy = tuple(("V" if i == 0 else f"V{i}", F(i + 1, 7)) for i in range(pairs))
        query = _types.SimpleNamespace(normal=(0, 0, 1), offset=F(1, 2), waterplane_origin=(0, 0, 0),
                                       waterplane_axes=((1, 0, 0), (0, 1, 0)), absolute_accuracy=accuracy,
                                       forms=("exact",))
        reads = []
        original = _fractions.Fraction.numerator

        def counting(self):
            if type(self) is _fractions.Fraction:
                reads.append(1)
            return original.fget(self)

        calls = []
        marks = []

        def cancel():
            calls.append(1)
            if len(calls) >= cancel_at:
                marks.append(len(reads))
                return True
            return False

        monkeypatch.setattr(_fractions.Fraction, "numerator", property(counting))
        try:
            error = refusal_of(lambda: verify_certified_hydrostatics(
                c.model, c.faces, query, r, c.expected, cancellation_check=cancel))
        finally:
            monkeypatch.undo()
        assert error.category == C.CANCELLED
        assert error.request is None or len(error.request) <= 6      # constant-size context
        return len(reads) - marks[0]

    for cancel_at in (1, 5, 40):
        small, large = run(100, cancel_at), run(10000, cancel_at)
        assert small == 0 and large == 0, (cancel_at, small, large)


# ------------------------------------------------------------------ review round 5: windowed bulk work, limits, context

def _long_decimal_case():
    c = case("F2")
    q, r = c.query(Z_, F(1, 2), forms=("decimal",), accuracy={"V": F(1, 10 ** 9000)})
    assert max(len(x) for x in r.volume.decimal[:2]) > 9000
    return c, q, r


def test_proof_hashing_encoding_and_decimal_scans_tick_per_window():
    c, q, r = _long_decimal_case()
    budget = WorkBudget.unlimited()
    c.verify(q, r, budget=budget)
    units = budget.counts()["units"]
    longest = max(len(x) for x in r.volume.decimal[:2])
    assert units.get("proof_hash_chunk", 0) >= max(1, len(r.proof) // 65536)
    assert units.get("encode_window", 0) >= longest // 4096
    assert units.get("decimal_window", 0) >= 2 * (longest // 4096)


def test_cancellation_is_seen_inside_the_proof_hash():
    c, q, r = _long_decimal_case()
    budget = WorkBudget.unlimited()
    # the artifact-file digest is hashed just before the proof: the proof's first chunk checkpoint sees the request
    error = refusal_of(lambda: c.verify(q, r, budget=budget, cancellation_check=lambda: budget.counts()["units"].get(
        "files_hash_chunk", 0) >= 1))
    assert error.category == C.CANCELLED
    assert budget.counts()["units"].get("proof_hash_chunk", 0) <= 1


@pytest.mark.parametrize("text", [
    "0", "-0", "007", "1.5", "-1.5", "12.000", "", "-", ".", ".5", "5.", "-.5", "1.2.3", "1..2", "+1", "1e5", " 1",
    "1 ", "１", "1.５", "--1", "1-", "9" * 4095 + ".5", "9" * 4096 + ".5", "9" * 4095 + "..5",
    "9" * 4096 + "." + "5" * 4096 + ".5", "-" + "9" * 4096, "9" * 8192 + ".", "." + "9" * 8192])
def test_windowed_decimal_grammar_matches_the_plain_grammar(text):
    import re
    plain = re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", text) is not None
    parts = vm._decimal_parts(text)
    assert (parts is not None) == plain
    if parts is not None:
        negative, whole, frac = parts
        assert ("-" if negative else "") + whole + ("." + frac if frac else "") == text


def test_huge_integer_digit_limits_are_accepted():
    # the proof and the expected policy agree on a 2**32-digit limit (built without the provider, which is not
    # exercised at that limit here)
    c = Case(lambda: box())
    q, r = c.query(Z_, F(1, 2))
    big = rewrite(r, lambda d: d["preparation_policy"].__setitem__("max_integer_digits", 2 ** 32))
    expected = dict(c.expected, policy=replace(c.expected["policy"], max_integer_digits=2 ** 32))
    verified = verify_certified_hydrostatics(c.model, c.faces, q, big, expected)
    assert verified.exact["V"] == F(1, 2)


def test_refusal_context_carries_the_exact_ingested_request():
    c = Case(lambda: box())
    q, r = c.query(Z_, F(1, 2), forms=("exact",), accuracy={"V": F(1, 100)})
    forged = alter_quantity(r, "V", numerator=F(999))
    error = refusal_of(lambda: c.verify(q, with_proof(forged)))
    assert error.request["forms"] == ("exact",)
    assert tuple(error.request["absolute_accuracy"]) == (("V", (1, 100)),)


# ------------------------------------------------------------------ CH fixes 6b: FX1 streaming re-encoding, FX2 decimal bounds

TRICKY_STRINGS = [
    "", "plain", 'a"b', "a\b", '"' * 9000, "\\" * 9000, "\x00\x01\x1f\x7f", "\n\r\t\b\f" * 2000,
    "m\u00e8tres \u00e9\u00e0 \u20ac", "\U0001F600" * 5000, "a\ud83d\ude00b",            # a pair as two code points
    "x" * 4095 + '"', "x" * 4095 + "\n", "x" * 4096 + "\\", "x" * 4094 + "\u00e9" + '"\\',
    "q" * 4095 + "\U0001F600" + "r" * 4097, "\ud83d" + "y" * 4094 + "\ude00",       # pair split across windows
    "".join(chr(i) for i in range(0, 0x250)) * 20,
]


@pytest.mark.parametrize("window", [4096, 7, 1])
@pytest.mark.parametrize("text", TRICKY_STRINGS, ids=range(len(TRICKY_STRINGS)))
def test_window_escaper_equals_json_dumps(text, window, monkeypatch):
    monkeypatch.setattr(vm, "_WINDOW", window)
    pieces = list(vm._json_string_pieces(text))
    assert "".join(pieces) == json.dumps(text, ensure_ascii=False)
    assert len(pieces) <= max(1, -(-len(text) // window)) + 2        # one piece per window plus the quotes


class _DumpsSpy:
    """Stands in for the ``json`` module of the verifier and records the longest string handed to ``dumps``."""

    def __init__(self):
        self.longest = 0

    def dumps(self, obj, *args, **kw):
        if isinstance(obj, str):
            self.longest = max(self.longest, len(obj))
        return json.dumps(obj, *args, **kw)

    def __getattr__(self, name):
        return getattr(json, name)


def _long_units_case(n):
    c = Case(lambda: box(), units="m" * n)
    q, r = c.query(Z_, F(1, 2), forms=("exact",))
    return c, q, r


def test_a_long_string_token_is_escaped_and_compared_window_by_window(monkeypatch):
    c, q, r = _long_units_case(100_000)
    spy = _DumpsSpy()
    monkeypatch.setattr(vm, "json", spy)
    ops = c.verify(q, r).work["operations"]
    assert 0 < spy.longest <= vm._WINDOW, spy.longest             # no single dumps call spans more than one window
    c2, q2, r2 = _long_units_case(200_000)
    ops2 = c2.verify(q2, r2).work["operations"]
    assert ops2 - ops >= 100_000 // vm._WINDOW                    # checkpoints grow with the token length
    c0, q0, r0 = _long_units_case(10)
    assert ops - c0.verify(q0, r0).work["operations"] >= 2 * (100_000 // vm._WINDOW)


def test_the_canonical_comparison_is_incremental_and_never_joins_the_document():
    c, q, r = _long_units_case(100_000)
    ok = c.verify(q, r).work["operations"]
    reordered = json.dumps(dict(reversed(list(json.loads(r.proof).items()))), separators=(",", ":"),
                           ensure_ascii=False).encode()
    error = failed(lambda: c.verify(q, replace(r, proof=reordered)), "value")
    # refused at the first window that differs: far fewer checkpoints than a complete comparison needs
    assert error.work["operations"] < ok - 100_000 // vm._WINDOW + 10, (error.work["operations"], ok)
    source = VERIFY_SOURCE.read_text(encoding="utf-8")
    start = source.index("def _encode")
    assert 'b"".join' not in source[start:source.index("def _same")]
    assert "canonical != res.proof" not in source


def test_a_non_canonical_long_string_is_still_refused_with_the_same_reason():
    c, q, r = _long_units_case(20_000)
    spaced = r.proof.replace(b'"units":', b'"units": ', 1)
    assert spaced != r.proof
    assert refusal_of(lambda: c.verify(q, replace(r, proof=spaced))).category == C.MALFORMED_PROOF
    swapped = r.proof.replace(b"m" * 10, b"m" * 9 + b"n", 1)
    failed(lambda: c.verify(q, replace(r, proof=swapped)), "convention")


def _dry_box_with_decimal(lo, hi):
    c = Case(lambda: box())
    q, r = c.query(Z_, F(-1))                                      # z <= -1: dry
    vol = r.volume
    assert vol.decimal is not None
    bad = replace(r, volume=replace(vol, decimal=(lo, hi, vol.decimal[2])))
    return c, q, with_proof(bad)


@pytest.mark.parametrize("lo,hi", [("0." + "0" * 20000, "0"), ("1" * 20000 + ".5", "0"), ("0", "0." + "0" * 20000),
                                   ("-" + "0" * 5000 + "." + "0" * 20000, "0")],
                         ids=["fraction", "numerator", "hi-fraction", "negative"])
def test_a_long_decimal_bound_is_refused_before_a_power_of_ten_is_built(lo, hi, monkeypatch):
    c, q, r = _dry_box_with_decimal(lo, hi)
    seen = []
    original = vm._fit

    def spy(x):
        seen.append(int(x.numerator if isinstance(x, F) else x).bit_length()
                    if not isinstance(x, F) else max(x.numerator.bit_length(), x.denominator.bit_length()))
        return original(x)

    monkeypatch.setattr(vm, "_fit", spy)
    error = refusal_of(lambda: c.verify(q, r, budget=WorkBudget(None, 128)))
    assert error.category == C.BIT_LENGTH_EXCEEDED and error.phase == "verify"
    assert max(seen, default=0) <= 128 + 1100, max(seen)          # at most one 300-digit chunk beyond the cap
    assert error.work["max_bits_seen"] <= 128 + 1100


def test_the_decimal_precheck_is_exact_at_the_cap(monkeypatch):
    for k in (1, 2, 7, 38, 39, 40, 100, 301, 1000):
        exact = (10 ** k).bit_length()
        assert vm._pow10_bits_lower(k) <= exact <= vm._pow10_bits_lower(k) + 1, k
    c, q, r = _dry_box_with_decimal("0." + "0" * 30, "0")
    assert c.verify(q, r, budget=WorkBudget(None, 128)).result is not None
    assert vm._pow10_bits_lower(30) == (10 ** 30).bit_length() == 100
    tight = refusal_of(lambda: c.verify(q, r, budget=WorkBudget(None, 99)))
    assert tight.category == C.BIT_LENGTH_EXCEEDED


# ------------------------------------------------------------------ 6b: single-string bulk operations are charged first

def test_long_strings_are_charged_per_64_kib_before_they_are_copied_or_compared():
    def units(n):
        c = Case(lambda: box(), units="m" * n)
        q, r = c.query(Z_, F(1, 2))
        budget = WorkBudget.unlimited()
        c.verify(q, r, budget=budget)
        return budget.counts()["units"].get("string_bulk", 0)

    small, large = units(10), units(1_000_000)
    assert large - small >= 2 * (1_000_000 // 65536)      # parsed (joined) and compared at least once each


def test_long_decimal_strings_are_charged_before_they_are_split():
    c = Case(lambda: box())
    q, r = c.query(Z_, F(-1), forms=("decimal",))
    lo, hi, width = r.volume.decimal
    long_lo = "0" * 1_000_000 + ".0"
    candidate = with_proof(alter_quantity(r, "V", decimal=(long_lo, hi, width)))
    budget = WorkBudget(None, 128)
    try:
        c.verify(q, candidate, budget=budget)
    except CertifiedHydrostaticsRefusal:
        pass
    assert budget.counts()["units"].get("string_bulk", 0) >= 2 * (len(long_lo) // 65536)
