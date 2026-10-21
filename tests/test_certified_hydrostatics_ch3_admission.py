"""CH3a1: admission of curved faces of the ruled rational class: C1 class, C2 trims, C3 corners with the vertex hint H1,
the domain/path/direction hint H2, C4 chart injectivity, the policy fields and the new refusal category (design
CERTIFIED_HYDROSTATICS_CH3_DESIGN.md sections 2.2, 2.3 and 9, CH3a row, first half)."""
from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from anygeometry import GeometryModel, Plane
from anygeometry.b1_example import _parameter, b1_bow_surface
from anygeometry.certified_hydrostatics import (CertifiedHydrostaticsPolicy, CertifiedHydrostaticsRefusal,
                                                RefusalCategory, WorkBudget)
from anygeometry.certified_hydrostatics._ch3_admission import classify_surface, plane_graph
from anygeometry.certified_hydrostatics._ch3_admission_trims import admit_curved_faces, rational_enclosure
from anygeometry.certified_hydrostatics._ch3_algebraic import AlgebraicNumber
from anygeometry.certified_hydrostatics._ch3_poly import Kernel, KernelLimits, pmul, poly_from, psub
from anygeometry.certified_hydrostatics._work import Meter
from anygeometry.entities import Face, OrientedEdge
from anygeometry.curves import Straight
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anygeometry.support_curves import PlaneNurbsCurve

C = RefusalCategory
W2 = 0.7071067811865476
POLICY = CertifiedHydrostaticsPolicy(convention_version="2")


def kern(max_bits=4096, budget=None, cancel=None):
    budget = budget if budget is not None else WorkBudget(None, max_bits)
    return Kernel(Meter(budget, cancel), POLICY.kernel_limits())


# ------------------------------------------------------------------ fixtures (design 8.2), built from their data

def horizontal(z):
    return Plane((0., 0., z), (1., 0., 0.), (0., 1., 0.))


def _nd(plane):
    o, u, v = ([F(x) for x in vector] for vector in (plane.origin, plane.u_vector, plane.v_vector))
    m = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    return tuple(m), sum(a * b for a, b in zip(m, o))


def surface_fa(top=((0., 0., 1.), (.5, 0., 1.), (1., 1., 1.))):
    """F-A: the parabolic cylinder (u, u^2, v), degree (2, 1), weights None."""
    bottom = ((0., 0., 0.), (.5, 0., 0.), (1., 1., 0.))
    return NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.), tuple(zip(bottom, top)))


def planes_fa():
    slant_up = lambda z0: Plane((0., 0., z0), (1., 0., 1.), (0., 1., 0.))
    return [horizontal(.25), slant_up(-.125), horizontal(.75), slant_up(.125)]


def surface_fb():
    """F-B: the conic slab's quarter conic (1,0), (1,1), (0,1) with weights (1, w, 1), z = 2 v."""
    bottom = ((1., 0., 0.), (1., 1., 0.), (0., 1., 0.))
    top = ((1., 0., 2.), (1., 1., 2.), (0., 1., 2.))
    return NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.), tuple(zip(bottom, top)),
                        ((1., 1.), (W2, W2), (1., 1.)))


def planes_fb():
    return [horizontal(.5), Plane((0., 0., 2.), (1., 0., -3.), (0., 1., 0.)), horizontal(1.5),
            Plane((0., 0., 2.), (0., 1., -3.), (1., 0., 0.))]


def surface_fc():
    """F-C: the plane top (u, v, 1) as a degree (1, 1) patch."""
    return NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.), (((0., 0., 1.), (0., 1., 1.)),
                                                                   ((1., 0., 1.), (1., 1., 1.))))


def planes_fc():
    up = lambda y0: Plane((0., y0, 0.), (1., 1., 0.), (0., 0., 1.))
    down = lambda y0: Plane((0., y0, 0.), (1., -1., 0.), (0., 0., 1.))
    return [down(.75), up(-.25), down(1.25), up(.25)]


def build_face(surface, planes, *, move=None, domains=None, reverse=(), lines=(), positions=None):
    """A model with one face on ``surface`` bounded by section curves of ``planes`` (edge i on plane i, from the
    corner of planes i-1, i to the corner of planes i, i+1, as ``b1_example`` builds them). Options inject the
    hostile cases: ``move=(vertex, (dx, dy, dz))`` moves a stored vertex, ``domains={edge: fn(a, b)}`` replaces an
    edge's stored domain, ``reverse`` runs the stored curve of an edge against its vertices, ``lines`` makes edges
    straight."""
    n = len(planes)
    model = GeometryModel()
    curves = [PlaneNurbsCurve.from_plane(p, surface, 0) for p in planes]
    nd = [_nd(p) for p in planes]
    given = positions is not None
    if not given:
        t_in = [_parameter(curves[i], *nd[(i - 1) % n]) for i in range(n)]
        t_out = [_parameter(curves[i], *nd[(i + 1) % n]) for i in range(n)]
        positions = [list(curves[i].evaluate(t_in[i])) for i in range(n)]
    else:
        t_in, t_out = [0.] * n, [1.] * n
        positions = [list(p) for p in positions]
    vertices = [model.add_point(*p) for p in positions]
    edges = []
    for i in range(n):
        if given:           # no section roots at the vertices: lines first, the full section curves written below
            edges.append(model.add_line(vertices[i], vertices[(i + 1) % n]))
        else:
            edges.append(model.add_curve(vertices[i], vertices[(i + 1) % n], curves[i].subcurve(t_in[i], t_out[i])))
    loop = model.order_loop(edges)
    face = None if given else model.add_face_from_loop(loop, surface=surface)
    # hostile edits go straight into the stores: the public API checks edge ends against vertices and refuses them
    for i in range(n):
        curve = curves[i] if given else None
        if i in lines:
            curve = Straight()
        elif i in reverse:
            curve = curves[i].subcurve(t_out[i], t_in[i])
        elif domains and i in domains:
            curve = replace(curves[i].subcurve(t_in[i], t_out[i]), domain=domains[i](t_in[i], t_out[i]))
        if curve is not None:
            model._put_entity("edge", replace(model.edges[edges[i]], curve=curve))
    if given:               # the section ends are not roots at these vertices: the face is written, not validated
        face = model._next_id["face"]
        model._next_id["face"] = face + 1
        model._put_entity("face", Face(face, tuple(loop), surface=surface))
    if move is not None:             # after the model checks (it refuses a vertex far from its curve's end)
        record = model.vertices[vertices[move[0]]]
        model._put_entity("vertex", replace(record, position=np.array([a + b for a, b in zip(record.position, move[1])])))
    return model, face, vertices, edges


def admit(model, face, policy=POLICY, budget=None, cancel=None):
    meter = Meter(budget if budget is not None else policy.new_budget(), cancel)
    return admit_curved_faces(model, [face], meter, policy)[face]


def refusal_of(call):
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        call()
    return info.value


_CACHE: dict = {}


def fixture(name):
    if name not in _CACHE:
        surface, planes = {"fa": (surface_fa(), planes_fa), "fb": (surface_fb(), planes_fb),
                           "fc": (surface_fc(), planes_fc)}[name]
        _CACHE[name] = build_face(surface, planes())
    return _CACHE[name]


# ------------------------------------------------------------------ policy and categories

def test_the_policy_keeps_the_planar_dictionary_and_maps_the_curved_fields():
    planar = CertifiedHydrostaticsPolicy()
    assert planar.as_dict() == {"convention_version": "1", "max_proof_bytes": 64 * 1024 * 1024,
                                "max_integer_digits": 100_000}
    full = POLICY.as_dict()
    assert full["convention_version"] == "2" and full["max_taylor_order"] == 16
    assert (full["delta_hint_bits"], full["delta_chart_bits"], full["max_bits"], full["max_isolation_bits"],
            full["max_pieces_per_integral"], full["max_refinement_rounds"], full["default_enclosure_digits"]) \
        == (40, 40, 4096, 4096, 1024, 8, 12)
    assert POLICY.kernel_limits() == KernelLimits(4096, 16, 1024, 8)
    custom = CertifiedHydrostaticsPolicy(convention_version="2", max_taylor_order=32, max_isolation_bits=100,
                                         max_pieces_per_integral=7, max_refinement_rounds=3, max_bits=999)
    assert custom.kernel_limits() == KernelLimits(100, 32, 7, 3) and custom.new_budget().max_bits == 999
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            CertifiedHydrostaticsPolicy(delta_hint_bits=bad)


def test_the_new_refusal_category_exists():
    assert C.TRIM_HINT_MISMATCH.value == "TRIM_HINT_MISMATCH"


# ------------------------------------------------------------------ C1 and C4

@pytest.mark.parametrize("name", ["fa", "fb", "fc"])
def test_the_fixture_supports_pass_c1_and_c4(name):
    surface = {"fa": surface_fa, "fb": surface_fb, "fc": surface_fc}[name]()
    support = classify_surface(kern(), surface)
    assert support.chart_c1 != 0 and any(support.chart_normal)
    assert support.degree_u == (1 if name == "fc" else 2)


def test_b1_support_net_and_chart():
    support = classify_surface(kern(), b1_bow_surface())
    assert support.span_u == (0, 1) and support.span_v == (0, 1) and support.u_range == (0, F(0.9))
    assert support.W == support.A[3] == support.B[3]
    assert support.chart_normal[0] == 0 and support.chart_normal[1] == 0 and support.chart_normal[2] != 0
    assert (support.chart_c0, support.chart_c1 / support.chart_normal[2]) == (0, 4)
    # z~ = 0 on the bottom row and z~ = 4 W on the deck row, as polynomials (design 1)
    assert support.A[2] == () and psub(kern(), support.B[2], tuple(4 * c for c in support.W)) == ()


def test_knot_space_net_agrees_with_the_library_point_formula():
    """S(u, v) = P / W in knot space equals the library's evaluation (rational vs float: 1e-12)."""
    surface = b1_bow_surface()
    k = kern()
    support = classify_surface(k, surface)
    for u, v in ((F(1, 5), F(1, 3)), (F(3, 4), F(7, 8))):
        w = sum(c * u ** i for i, c in enumerate(support.W))
        point = [((1 - v) * sum(c * u ** i for i, c in enumerate(support.A[a])) +
                  v * sum(c * u ** i for i, c in enumerate(support.B[a]))) / w for a in range(3)]
        chart = (float(u - support.u_range[0]) / float(support.u_range[1] - support.u_range[0]), float(v))
        reference = surface.evaluate_many([chart])[0]
        assert max(abs(float(a) - b) for a, b in zip(point, reference)) < 1e-12


def test_c1_unequal_weight_rows_refused():
    rows = tuple(zip(((0., 6., 0.), (5.4, 6., 0.), (5.4, 0., 0.)), ((0., 7.5, 4.), (6., 7.5, 4.), (6., 0., 4.))))
    surface = NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.), rows, ((1., 1.), (W2, .8), (1., 1.)))
    error = refusal_of(lambda: classify_surface(kern(), surface))
    assert error.category is C.UNSUPPORTED_SURFACE and error.sub_reason == "unequal_weight_rows"


def test_c1_interior_knot_degree_and_row_count_refused():
    pts = lambda z: tuple((float(i), float(i * i % 3), z) for i in range(4))
    four_rows = NurbsSurface(2, 1, (0., 0., 0., .5, 1., 1., 1.), (0., 0., 1., 1.), tuple(zip(pts(0.), pts(1.))))
    three_cols = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., .5, 1., 1.),
                              tuple(tuple((float(i), 0., float(j)) for j in range(3)) for i in range(2)))
    cubic = NurbsSurface(3, 1, (0., 0., 0., 0., 1., 1., 1., 1.), (0., 0., 1., 1.), tuple(zip(pts(0.), pts(1.))))
    quad_v = NurbsSurface(1, 2, (0., 0., 1., 1.), (0., 0., 0., 1., 1., 1.),
                          tuple(tuple((float(i), 0., float(j)) for j in range(3)) for i in range(2)))
    seen = [(four_rows, "interior_knot"), (three_cols, "interior_knot"), (cubic, "degree_u"), (quad_v, "degree_v")]
    for surface, sub in seen:
        error = refusal_of(lambda s=surface: classify_surface(kern(), s))
        assert error.category is C.UNSUPPORTED_SURFACE and error.sub_reason == sub, sub


def test_c1_a_non_nurbs_support_is_refused():
    error = refusal_of(lambda: classify_surface(kern(), horizontal(0.)))
    assert error.category is C.UNSUPPORTED_SURFACE and error.sub_reason == "not_nurbs"


def test_c4_collinear_rows_and_non_parallel_rows_are_not_graph_like():
    collinear = surface_fa(top=((0., 0., 1.), (.5, 0., 1.), (1., 0., 1.)))
    tilted = surface_fa(top=((0., 0., 1.), (.5, 0., 2.), (1., 1., 1.)))
    for surface in (collinear, tilted):
        error = refusal_of(lambda s=surface: classify_surface(kern(), s))
        assert error.category is C.UNSUPPORTED_SURFACE and error.sub_reason == "not_graph_like"


def test_c4_degenerate_linear_patch_is_not_graph_like():
    zero_segment = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                                (((0., 0., 0.), (0., 0., 1.)), ((0., 0., 0.), (0., 0., 1.))))
    error = refusal_of(lambda: classify_surface(kern(), zero_segment))
    assert error.category is C.UNSUPPORTED_SURFACE and error.sub_reason == "not_graph_like"


# ------------------------------------------------------------------ plane graphs

def test_plane_graph_of_the_fixture_trims_and_the_b1_station_planes():
    k = kern()
    fa = classify_surface(k, surface_fa())
    graph = plane_graph(k, fa, *(_vec(x) for x in (planes_fa()[1].origin, planes_fa()[1].u_vector,
                                                  planes_fa()[1].v_vector)))
    # z = x - 1/8 on (u, u^2, v): v = u - 1/8, i.e. N = 8u - 1, D = 8
    assert graph.N == (F(-1), F(8)) and graph.D == (F(8),)
    b1 = classify_surface(k, b1_bow_surface())
    z_half = plane_graph(k, b1, *(_vec(x) for x in (horizontal(.5).origin, horizontal(.5).u_vector,
                                                    horizontal(.5).v_vector)))
    assert z_half.N == (F(1),) and z_half.D == (F(8),)             # r = 1/8 (design 1: corners at v = 1/8, 3/4)
    deck = plane_graph(k, b1, *(_vec(x) for x in (horizontal(3.).origin, horizontal(3.).u_vector,
                                                  horizontal(3.).v_vector)))
    assert deck.N == (F(3),) and deck.D == (F(4),)


def _vec(x):
    return tuple(F(c) for c in x)


def test_plane_graph_zero_beta_and_face_in_plane_refused():
    k = kern()
    fa = classify_surface(k, surface_fa())
    ruling = Plane((.5, 0., 0.), (0., 1., 0.), (0., 0., 1.))                # x = 1/2: parallel to the rulings
    error = refusal_of(lambda: plane_graph(k, fa, *(_vec(x) for x in (ruling.origin, ruling.u_vector,
                                                                      ruling.v_vector))))
    assert error.category is C.UNSUPPORTED_TRIM and error.sub_reason == "ruling_trim"
    flat = classify_surface(k, surface_fc())
    in_plane = horizontal(1.)                                             # the fixture top lies in z = 1
    error = refusal_of(lambda: plane_graph(k, flat, *(_vec(x) for x in (in_plane.origin, in_plane.u_vector,
                                                                        in_plane.v_vector))))
    assert error.sub_reason == "face_in_plane"
    zero = (F(0), F(0), F(0))
    error = refusal_of(lambda: plane_graph(k, fa, zero, (F(1), F(0), F(0)), (F(2), F(0), F(0))))
    assert error.sub_reason == "zero_plane"


# ------------------------------------------------------------------ faces: C2, C3, H1, H2 on the fixtures and B1

def _ek(k, face, i):
    """The enclosure of the i-th corner's (u, v) in knot space."""
    trim = face.trims[i]
    vertex = trim.start_vertex if trim.forward else trim.end_vertex
    corner = face.corners[vertex]
    return corner.number, rational_enclosure(k, trim.plane.N, trim.plane.D, corner.number)


@pytest.mark.parametrize("name,expected", [
    ("fa", [(F(1, 8), F(1, 4)), (F(3, 8), F(1, 4)), (F(7, 8), F(3, 4)), (F(5, 8), F(3, 4))]),
    ("fc", [(F(1, 4), F(1, 2)), (F(1, 2), F(1, 4)), (F(3, 4), F(1, 2)), (F(1, 2), F(3, 4))]),
    ("fb", [(0.3411, .25), (0.6589, .25), (0.8853, .75), (0.1147, .75)])])
def test_fixture_faces_pass_c1_to_c4_with_corners_hints_and_h2(name, expected):
    from anygeometry.certified_hydrostatics._ch3_algebraic import compare
    model, face, vertices, edges = fixture(name)
    result = admit(model, face)
    k = kern()
    assert len(result.trims) == 4 and set(result.corners) == set(vertices)
    for i, (u, v) in enumerate(expected):
        number = result.corners[vertices[i]].number
        _, value = _ek(k, result, i)                      # vertex i starts trim i (all forward)
        if name == "fb":
            mid = float(number.lo + number.hi) / 2
            assert abs(mid - u) < 1e-4 and abs(float(value.lo + value.hi) / 2 - v) < 1e-9
        else:
            assert compare(k, number, u) == 0 and value.lo <= v <= value.hi and value.width < F(1, 2 ** 38)
        assert result.corners[vertices[i]].hint_error <= F(1, 2 ** 40) * 3
    for trim in result.trims:
        assert trim.chart_errors and max(trim.chart_errors) <= F(1, 2 ** 40)
        assert trim.range_method in ("bernstein", "exact") and trim.forward
        assert trim.loop_increasing == trim.curve_increasing
    # F-A and F-C have rational corners: the exact u-intervals end at those rationals
    if name != "fb":
        ends = sorted(compare(k, t.u_lo, F(1, 8)) for t in result.trims)
        assert ends[0] >= -1


def test_b1_half_hull_bow_passes_with_the_measured_corners():
    from conftest import b1_half_hull_shared
    model, faces = b1_half_hull_shared()
    result = admit(model, faces["bow"])
    k = kern()
    assert len(result.trims) == 4 and len(result.corners) == 4
    levels = []
    for trim in result.trims:
        vertex = trim.start_vertex if trim.forward else trim.end_vertex
        number = result.corners[vertex].number
        assert F(1749, 10000) <= number.lo and number.hi <= F(4446, 10000)         # design section 1
        value = rational_enclosure(k, trim.plane.N, trim.plane.D, number)
        levels.append(next(v for v in (F(1, 8), F(3, 4)) if value.lo <= v <= value.hi))
        assert max(trim.chart_errors[:2]) < F(1, 2 ** 40) and trim.chart_errors[2] < F(1, 2 ** 40)
    assert sorted(levels) == [F(1, 8), F(1, 8), F(3, 4), F(3, 4)]
    # the two z-plane trims are the graphs r = 1/8 and r = 3/4 (design section 1)
    constants = sorted(t.plane.N[0] / t.plane.D[0] for t in result.trims if len(t.plane.D) == 1)
    assert constants == [F(1, 8), F(3, 4)]
    assert all(c.hint_error <= F(1, 2 ** 40) * 8 for c in result.corners.values())


def test_b1_full_hull_ports_and_starboard_bows_share_corner_polynomials():
    from conftest import b1_full_hull_shared
    model, faces = b1_full_hull_shared()
    both = admit_curved_faces(model, [faces["bow"], faces["bow_port"]], Meter(POLICY.new_budget()), POLICY)
    starboard = sorted((c.poly, c.ordinal) for c in both[faces["bow"]].corners.values())
    port = sorted((c.poly, c.ordinal) for c in both[faces["bow_port"]].corners.values())
    assert starboard == port


# ------------------------------------------------------------------ refusals from minimal cases

def _fresh(name, **options):
    surface, planes = {"fa": (surface_fa, planes_fa), "fb": (surface_fb, planes_fb),
                       "fc": (surface_fc, planes_fc)}[name]
    return build_face(surface(), planes(), **options)


def test_c2_straight_edge_on_the_bow_refused():
    from conftest import b1_half_hull_shared
    model, faces = b1_half_hull_shared()
    clone = model.clone()
    bow = clone.faces[faces["bow"]]
    edge_id = bow.loop[0].edge
    clone._put_entity("edge", replace(clone.edges[edge_id], curve=Straight()))
    error = refusal_of(lambda: admit(clone, faces["bow"]))
    assert error.category is C.UNSUPPORTED_TRIM and error.sub_reason == "straight_edge"


def test_c2_curve_on_surface_edge_and_foreign_support_refused():
    model, face, vertices, edges = _fresh("fa")
    surface = model.faces[face].surface
    pcurve = NurbsCurve(1, (0., 0., 1., 1.), ((.1, .5), (.9, .5)))
    model._put_entity("edge", replace(model.edges[edges[0]], curve=CurveOnSurface(surface, pcurve)))
    error = refusal_of(lambda: admit(model, face))
    assert error.category is C.UNSUPPORTED_TRIM and error.sub_reason == "curve_on_surface"
    model, face, vertices, edges = _fresh("fa")
    other = surface_fa(top=((0., 0., 2.), (.5, 0., 2.), (1., 1., 2.)))                 # another support
    model._put_entity("face", replace(model.faces[face], surface=other))
    error = refusal_of(lambda: admit(model, face))
    assert error.category is C.UNSUPPORTED_TRIM and error.sub_reason == "support_mismatch"
    for narrower in ({"u_range": (0., .5)}, {"v_range": (0., .5)}):           # the same support, another range box
        model, face, vertices, edges = _fresh("fa")
        model._put_entity("face", replace(model.faces[face], surface=replace(model.faces[face].surface, **narrower)))
        error = refusal_of(lambda: admit(model, face))
        assert error.category is C.UNSUPPORTED_TRIM and error.sub_reason == "support_mismatch"


def test_c3_vertex_hint_moved_1e6_refused_and_1e13_accepted():
    model, face, vertices, edges = _fresh("fa", move=(1, (1e-6, 0., 0.)))
    error = refusal_of(lambda: admit(model, face))
    assert error.category is C.TRIM_HINT_MISMATCH and error.sub_reason == "vertex_hint"
    model, face, vertices, edges = _fresh("fa", move=(1, (1e-13, 0., 0.)))              # inside delta_hint = 2^-40 (1 + max|x|)
    result = admit(model, face)
    assert 5e-14 < result.corners[vertices[1]].hint_error <= 2 ** -40 * 3


def test_h2_shortened_and_distant_domains_and_reversed_direction_refused():
    shortened = lambda a, b: ((3 * a + b) / 4, (a + 3 * b) / 4)
    distant = lambda a, b: (0.95, 0.99)
    for options in ({"domains": {0: shortened}}, {"domains": {2: distant}}, {"reverse": (3,)}):
        model, face, vertices, edges = _fresh("fa", **options)
        error = refusal_of(lambda: admit(model, face))
        assert error.category is C.TRIM_HINT_MISMATCH and error.sub_reason == "domain_start", options
    model, face, vertices, edges = _fresh("fa", domains={1: lambda a, b: (a, (a + 3 * b) / 4)})   # right start, short end
    error = refusal_of(lambda: admit(model, face))
    assert error.category is C.TRIM_HINT_MISMATCH and error.sub_reason == "domain_end"
    # the unmodified fixture passes the same checks
    model, face, vertices, edges = _fresh("fa")
    admit(model, face)


def test_h2_midpoint_conditions(monkeypatch):
    model, face, vertices, edges = _fresh("fa")
    original = PlaneNurbsCurve._knot_points

    def shifted(self, t, du=0., dv=0., only_bottom=False):
        us, vs = original(self, t)
        if only_bottom and self.plane_record[0][2] != .25:           # the graph v = 1/4 is constant: a shift of u
            return us, vs                                            # is invisible to the graph test
        us, vs = us.copy(), vs.copy()
        us[1] += du
        vs[1] += dv
        return us, vs

    monkeypatch.setattr(PlaneNurbsCurve, "_knot_points", lambda self, t: shifted(self, t, 0., 1e-3))
    error = refusal_of(lambda: admit(model, face))
    assert error.category is C.TRIM_HINT_MISMATCH and error.sub_reason == "domain_mid"
    monkeypatch.setattr(PlaneNurbsCurve, "_knot_points", lambda self, t: shifted(self, t, 0.5, 0., True))
    error = refusal_of(lambda: admit(model, face))
    assert error.category is C.TRIM_HINT_MISMATCH and error.sub_reason == "domain_mid"
    monkeypatch.setattr(PlaneNurbsCurve, "_knot_points", lambda self, t: shifted(self, t, 0., 0.))
    admit(model, face)


def surface_fa_wide(v_range=None):
    """F-A with the knot domain ``v`` in [0, 2], z = v, and the range box ``v`` in ``v_range`` (default [0, 2])."""
    bottom = ((0., 0., 0.), (.5, 0., 0.), (1., 1., 0.))
    top = ((0., 0., 2.), (.5, 0., 2.), (1., 1., 2.))
    return NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 2., 2.), tuple(zip(bottom, top)),
                        v_range=v_range)


S1 = Plane((0., 0., .4375), (1., 0., -.75), (0., 1., 0.))               # z = 7/16 - 3x/4 (through (1/4, v = 1/4))
S2 = Plane((0., 0., -.3125), (1., 0., .75), (0., 1., 0.))               # z = 3x/4 - 5/16 (through (3/4, v = 1/4))
REVIEW_TRIM = Plane((0., 0., -2.), (1., 1., 0.), (1., 0., 12.))         # 12 x - 12 y - z = 2, i.e. z = 12x - 12y - 2
LOWER_TRIM = Plane((0., 0., -1.), (1., 1., 0.), (1., 0., 6.))           # 6 x - 6 y - z = 1: z = 6u - 6u^2 - 1 <= 1/2


def _triangle(trim, narrow_to=None):
    """F-A with the knot domain ``v`` in [0, 2] (z = v): the triangle bounded by ``trim`` (over the top) and the
    planes S2, S1 (meeting at (1/2, v = 1/16)). With ``narrow_to`` the sections are built on the wide range box and
    the stored face and curves then claim ``v`` in ``narrow_to``: the library refuses to build a curve that touches
    the range box, so the hostile record is written straight into the stores."""
    model, face, vertices, edges = build_face(surface_fa_wide(), [trim, S2, S1])
    if narrow_to:
        narrow = surface_fa_wide(narrow_to)
        for e in edges:
            model._put_entity("edge", replace(model.edges[e], curve=replace(model.edges[e].curve, surface=narrow)))
        model._put_entity("face", replace(model.faces[face], surface=narrow))
    return model, face


def test_c2_review_trim_reaching_the_range_box_is_refused_but_a_lower_trim_is_admitted():
    """z <= 12x - 12y - 2 on (u, u^2, v): the corners r(1/4) = r(3/4) = 1/4 are interior, the extremum r(1/2) = 1 = v_1."""
    model, face = _triangle(REVIEW_TRIM, narrow_to=(0., 1.))
    error = refusal_of(lambda: admit(model, face))
    assert error.category is C.UNSUPPORTED_TRIM and error.sub_reason == "leaves_range_box"
    # a top whose extremum stays inside the box (6u - 6u^2 - 1 <= 1/2) is admitted on the same range box
    model, face = _triangle(LOWER_TRIM, narrow_to=(0., 1.))
    result = admit(model, face)
    assert result.trims[0].range_method == "bernstein" or result.trims[0].range_method == "exact"
    # and the review trim is admitted when the range box is wide enough to contain its extremum
    model, face = _triangle(REVIEW_TRIM)
    admit(model, face)


def test_c2_denominator_vanishing_on_an_edge_refused():
    """The flared patch S = (u, u v, v), v in [-1, 1]: the plane 2y - z = 1/4 has D = 2u - 1, a pole inside its edge."""
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (-1., -1., 1., 1.),
                           (((0., 0., -1.), (0., 0., 1.)), ((1., -1., -1.), (1., 1., 1.))))
    pole = Plane((0., .125, 0.), (1., 0., 0.), (0., 1., 2.))
    slant = Plane((0., 0., -.625), (1., 0., 1.5), (0., 1., 0.))              # z = 3x/2 - 5/8
    planes = [pole, slant, horizontal(-.5)]
    positions = [(.25, -.125, -.5), (.75, .375, .5), (1 / 12, -1 / 24, -.5)]
    model, face, vertices, edges = build_face(surface, planes, positions=positions)
    error = refusal_of(lambda: admit(model, face))
    assert error.category is C.UNSUPPORTED_TRIM and error.sub_reason == "denominator_vanishes"


def test_c3_parallel_and_coincident_adjacent_trims_are_refused():
    slant = planes_fa()[1]
    positions = [(.375, .140625, .25), (.2, .04, .75), (.3, .09, .5)]   # vertex 0 is the true corner
    for planes, sub in (([horizontal(.25), horizontal(.75), slant], "no_corner_in_range_box"),
                        ([horizontal(.25), horizontal(.25), slant], "coincident_trims")):
        model, face, vertices, edges = build_face(surface_fa(), planes, positions=positions)
        error = refusal_of(lambda: admit(model, face))
        assert error.category is C.UNSUPPORTED_TRIM and error.sub_reason == sub


def test_a_face_with_holes_or_a_short_loop_is_refused():
    model, face, vertices, edges = _fresh("fa")
    record = model.faces[face]
    model._put_entity("face", replace(record, holes=(record.loop,)))
    assert refusal_of(lambda: admit(model, face)).category is C.INVALID_HOLES
    model._put_entity("face", replace(record, holes=(), loop=record.loop[:2]))
    error = refusal_of(lambda: admit(model, face))
    assert error.category is C.INVALID_LOOP and error.sub_reason == "loop_too_short"


# ------------------------------------------------------------------ work: hard-coded counts, cancellation, budget

def _admission_ops():
    b1 = classify_surface(kern(), b1_bow_surface())
    deck = horizontal(3.)
    model, face, _vertices, _edges = fixture("fc")
    return {
        "classify": lambda k: classify_surface(k, b1_bow_surface()),
        "plane_graph": lambda k: plane_graph(k, b1, *(_vec(x) for x in (deck.origin, deck.u_vector, deck.v_vector))),
        "face": lambda k: admit_curved_faces(model, [face], k.meter, POLICY),
    }


def _units(op):
    budget = WorkBudget(None, 4096)
    op(kern(budget=budget))
    return budget.counts()


def test_hard_coded_unit_counts_of_the_admission_loops():
    ops = _admission_ops()
    assert _units(ops["classify"])["operations"] == 275
    assert _units(ops["classify"])["units"] == {"admission_element": 55, "bernstein_coefficient": 14,
                                                "poly_coefficient": 206}
    assert _units(ops["plane_graph"])["units"] == {"admission_element": 9, "poly_coefficient": 123}
    face = _units(ops["face"])
    assert face["operations"] == 1378
    assert face["units"] == {"admission_element": 121, "bernstein_coefficient": 219, "face_read": 1,
                             "poly_coefficient": 970, "root_refine": 27, "sign_count": 40}


@pytest.mark.parametrize("name", ["classify", "plane_graph", "face"])
def test_cancellation_at_element_k_and_deadline_per_admission_loop(name):
    op = _admission_ops()[name]
    total = _units(op)["operations"]
    for stop in sorted({0, 1, total // 3, total // 2, total - 1}):
        budget = WorkBudget(None, 4096)
        error = refusal_of(lambda b=budget, s=stop: op(kern(budget=b, cancel=lambda: b.operations >= s)))
        assert error.category is C.CANCELLED and budget.operations == stop
        holder = []
        budget = WorkBudget(None, 4096, deadline=5.0, clock=lambda s=stop: 10.0 if holder[0].operations >= s else 0.0)
        holder.append(budget)
        error = refusal_of(lambda b=budget: op(kern(budget=b)))
        assert error.category is C.DEADLINE_EXCEEDED and budget.operations == stop
    # every unit kind the loops reserve is a checkpoint: cancel at the first unit of each kind
    for kind in _units(op)["units"]:
        budget = WorkBudget(None, 4096)
        error = refusal_of(lambda b=budget, kd=kind: op(kern(budget=b, cancel=lambda: b.counts()["units"].get(kd, 0) >= 1)))
        assert error.category is C.CANCELLED


@pytest.mark.parametrize("name", ["classify", "plane_graph", "face"])
def test_every_admission_unit_is_a_budget_checkpoint(name):
    op = _admission_ops()[name]
    total = _units(op)["operations"]
    for limit in sorted({0, 1, 2, total // 3, total // 2, total - 2, total - 1}):
        budget = WorkBudget(limit, 4096)
        error = refusal_of(lambda b=budget: op(kern(budget=b)))
        assert error.category is C.BUDGET_EXHAUSTED and budget.operations == limit
    budget = WorkBudget(total, 4096)
    op(kern(budget=budget))
    assert budget.operations == total


def test_the_bit_cap_refuses_before_an_over_cap_value_is_formed():
    for cap in (10, 40, 70):
        error = refusal_of(lambda c=cap: classify_surface(kern(max_bits=c), b1_bow_surface()))
        assert error.category is C.BIT_LENGTH_EXCEEDED
    model, face, _v, _e = fixture("fc")
    error = refusal_of(lambda: admit(model, face, budget=WorkBudget(None, 30)))
    assert error.category is C.BIT_LENGTH_EXCEEDED
