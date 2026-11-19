"""K6: the rigorous error bound of the float fold coefficients (``exact_curves._fold_coefficient_bound``) and the
undecidable-fold classification it drives (``_fold_undecidable``)."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from anygeometry.analytic_supports import undecidable_contact
from anygeometry.exact_curves import _CylinderSupport, _fold_coefficient_bound, _fold_undecidable
from anygeometry.surfaces import Cylinder


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector/np.linalg.norm(vector)


def _pair(rng, scale):
    a1 = _unit(rng.normal(size=3))
    radial = _unit(np.cross(a1, rng.normal(size=3)))
    a2 = _unit(rng.normal(size=3))
    first = Cylinder(origin=tuple(scale*rng.normal(size=3)), axis=tuple(a1), radial_direction=tuple(radial),
                     radius=float(abs(rng.normal())*scale/4+.1), height=10., start_angle=0., sweep_angle=2*math.pi)
    second = Cylinder(origin=tuple(scale*rng.normal(size=3)), axis=tuple(a2),
                      radial_direction=tuple(_unit(np.cross(a2, rng.normal(size=3)))),
                      radius=float(abs(rng.normal())*scale/4+.1), height=10., start_angle=0., sweep_angle=2*math.pi)
    return _CylinderSupport.from_surface(first), _CylinderSupport.from_surface(second)


def _discriminants(first, second, sign):
    """``(D of the float coefficients, exact D of the stored doubles)`` as exact rationals; the irrational
    ``|a1 x a2|`` of the exact D is replaced by a 60-digit rational, far inside the tested bounds."""
    q = lambda vector: [Fraction(float(value)) for value in np.asarray(vector, dtype=float)]
    cross = lambda u, v: [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    dot = lambda u, v: sum(x*y for x, y in zip(u, v))
    a1, a2 = np.asarray(first.axis), np.asarray(second.axis)
    c = np.cross(a1, a2)
    magnitude = float(np.linalg.norm(c))
    e1 = np.asarray(first.radial_direction)
    n0 = float((np.asarray(first.origin)-np.asarray(second.origin)) @ c)
    nc, ns = first.radius*float(e1 @ c), first.radius*float(np.cross(a1, e1) @ c)
    c0 = n0-sign*magnitude*second.radius
    floated = Fraction(nc)**2+Fraction(ns)**2-Fraction(c0)**2
    A1, E1 = q(a1), q(e1)
    C, E2 = cross(A1, q(a2)), cross(A1, E1)
    N0 = dot([x-y for x, y in zip(q(first.origin), q(second.origin))], C)
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    norm = Fraction((Decimal(dot(C, C).numerator)/Decimal(dot(C, C).denominator)).sqrt())
    R1, R2 = Fraction(float(first.radius)), Fraction(float(second.radius))
    exact = (R1*dot(E1, C))**2+(R1*dot(E2, C))**2-(N0-sign*norm*R2)**2
    return floated, exact


@pytest.mark.parametrize("scale", [2.**-40, 1., 50., 1e4])
def test_the_bound_covers_the_float_coefficient_error_on_random_pairs(scale):
    rng = np.random.default_rng(20261119)
    for _ in range(200):
        first, second = _pair(rng, scale)
        for sign in (-1, 1):
            bound = _fold_coefficient_bound(first, second, sign)
            floated, exact = _discriminants(first, second, sign)
            assert abs(floated-exact) <= Fraction(bound), (scale, sign)


@pytest.mark.parametrize("scale", [1., 50., 1e4])
def test_ordinary_random_pairs_have_no_undecidable_fold(scale):
    rng = np.random.default_rng(7)
    for _ in range(200):
        first, second = _pair(rng, scale)
        assert _fold_undecidable(first, second) == {}


def test_the_bound_is_none_for_non_finite_input():
    rng = np.random.default_rng(1)
    first, second = _pair(rng, 1.)
    broken = _CylinderSupport(origin=(math.inf, 0., 0.), axis=second.axis, radial_direction=second.radial_direction,
                              radius=second.radius, height=second.height, start_angle=second.start_angle,
                              sweep_angle=second.sweep_angle)
    assert _fold_coefficient_bound(first, broken, 1) is None


def test_the_magnitude_gate_declines_tiny_radii():
    """Codex K6 R1 M7: radii 2^-530 underflow the bound's terms; such pairs are outside the K3 magnitude range and
    are never classified undecidable (main's path)."""
    r1 = 2.**-530
    first = _CylinderSupport.from_surface(Cylinder(origin=(0., 0., 0.), axis=(0., 0., 1.), radial_direction=(0., 1., 0.),
                                                   radius=r1, height=1., start_angle=0., sweep_angle=2*math.pi))
    second = _CylinderSupport.from_surface(Cylinder(origin=(0., 0., 0.), axis=tuple(_unit((1., 1., 0.))),
                                                    radial_direction=(0., 0., 1.), radius=float(np.nextafter(r1, 1.)),
                                                    height=1., start_angle=0., sweep_angle=2*math.pi))
    assert _fold_undecidable(first, second) == {}


def test_an_ill_conditioned_undecidable_pair_offers_no_chart():
    """Codex K6 R1 M5: axes 1.7 degrees apart (sin^2 < 2^-8), equal radii within one ulp, axes intersecting: both folds
    undecidable but no admitted chart (the pair stage then refuses wherever a contact point is in both faces)."""
    u = _unit((.03, 0., 1.))
    first = Cylinder(origin=(0., 0., -1.), axis=(0., 0., 1.), radial_direction=(0., 1., 0.), radius=1., height=2.,
                     start_angle=-.5, sweep_angle=1.)
    second = Cylinder(origin=tuple(-u), axis=tuple(u), radial_direction=(0., 1., 0.), radius=float(np.nextafter(1., 2.)),
                      height=2., start_angle=-.5, sweep_angle=1.)
    contact = undecidable_contact(first, second, tolerance=1e-9)
    assert contact is not None and len(contact.points) == 4 and contact.support is None


def test_the_bound_never_rounds_to_zero():
    """Codex K6 R2 M5: the error propagation is exact rational arithmetic, so a pair whose float terms are tiny still
    gets a positive bound that covers the coefficient error (here axes 2^-29 apart, radii 2^-64)."""
    x = 2.**-29
    r = 2.**-64
    first = _CylinderSupport.from_surface(Cylinder(origin=(0., 0., 0.), axis=(1., 0., 0.), radial_direction=(0., 0., 1.),
                                                   radius=r, height=1., start_angle=0., sweep_angle=2*math.pi))
    second = _CylinderSupport.from_surface(Cylinder(origin=(0., 0., 0.), axis=tuple(_unit((1., x, 0.))),
                                                    radial_direction=(0., 0., 1.), radius=float(np.nextafter(r, 1.)),
                                                    height=1., start_angle=0., sweep_angle=2*math.pi))
    for sign in (-1, 1):
        bound = _fold_coefficient_bound(first, second, sign)
        floated, exact = _discriminants(first, second, sign)
        assert bound > 0 and abs(floated-exact) <= bound


def test_axes_nearer_parallel_than_the_floor_are_not_classified():
    """Codex K6 R2 M5: axes 2^-537 apart are below the classification floor (main's path)."""
    x = 1.25*2.**-537
    r = 2.**-64
    first = _CylinderSupport.from_surface(Cylinder(origin=(0., 0., 0.), axis=(1., 0., 0.), radial_direction=(0., 0., 1.),
                                                   radius=r, height=1., start_angle=0., sweep_angle=2*math.pi))
    second = _CylinderSupport.from_surface(Cylinder(origin=(0., 0., 0.), axis=(1., x, 0.), radial_direction=(0., 0., 1.),
                                                    radius=float(np.nextafter(r, 1.)), height=1., start_angle=0.,
                                                    sweep_angle=2*math.pi))
    assert _fold_undecidable(first, second) == {}


def test_the_ellipse_residual_allowance_uses_unprojected_lengths():
    """Codex K6 R2 H3: origins 1e7 m along the axes; projecting the long axial offset cancels, so an allowance on
    projected lengths certified ellipses 3e-10 m off the cylinder at tolerance 1e-10. They are not certified now."""
    L = 1e7+1/13
    first = Cylinder(origin=tuple(-L*np.asarray((.6, .8, 0.))), axis=(.6, .8, 0.), radial_direction=(0., 0., 1.),
                     radius=1., height=2*L, start_angle=0., sweep_angle=2*math.pi)
    second = Cylinder(origin=(0., 0., -L), axis=(0., 0., 1.), radial_direction=(1., 0., 0.),
                      radius=float(np.nextafter(1., 0.)), height=2*L, start_angle=0., sweep_angle=2*math.pi)
    contact = undecidable_contact(first, second, tolerance=1e-10)
    assert contact is not None and contact.support is None


def _chord_rectangle(face_id, cylinder, half_angle, base_edge):
    """A cylinder face trimmed by four straight chords between its corner points (allowed within the model's surface
    residual), with its own edge ids."""
    from anygeometry.arrangement_geometry import LinePath
    from anygeometry.material_arrangement import ArrangementPath, MaterialDomain
    origin, axis = np.asarray(cylinder.origin), np.asarray(cylinder.axis)
    radial, circumferential = np.asarray(cylinder.radial_direction), cylinder.circumferential_direction
    corners = []
    for angle, height in ((-half_angle, 0.), (half_angle, 0.), (half_angle, cylinder.height),
                          (-half_angle, cylinder.height)):
        corners.append(tuple(origin+cylinder.radius*(math.cos(angle)*radial+math.sin(angle)*circumferential)
                             + height*axis))
    loop = tuple(ArrangementPath(LinePath(corners[k], corners[(k+1) % 4]), base_edge+k) for k in range(4))
    return MaterialDomain(face_id, Cylinder(origin=cylinder.origin, axis=cylinder.axis,
                                            radial_direction=cylinder.radial_direction, radius=cylinder.radius,
                                            height=cylinder.height, start_angle=-half_angle,
                                            sweep_angle=2*half_angle), (loop,))


def test_chord_trims_that_stop_short_of_the_contact_still_need_a_witness():
    """Codex K6 R3 H1: micro cylinders (r = 1e-3) trimmed by straight chords: both boundary boxes end about 8e-9 m short
    of the contact (beyond the old bound-derived radius). The near radius r / 4 covers it: the pair refuses."""
    from anygeometry.batch_intersections import UNDECIDABLE_CONTACT, _pair_paths_exact
    from anygeometry.errors import GeometryError
    r = .001
    first = Cylinder(origin=(0., 0., -.002), axis=(0., 0., 1.), radial_direction=(0., 1., 0.), radius=r, height=.004,
                     start_angle=-.004, sweep_angle=.008)
    second = Cylinder(origin=(-.002, 0., 0.), axis=(1., 0., 0.), radial_direction=(0., 1., 0.),
                      radius=float(np.nextafter(r, 1.)), height=.004, start_angle=-.004, sweep_angle=.008)
    contact = undecidable_contact(first, second, tolerance=1e-10)
    assert contact is not None and contact.near_radius == .25*r
    with pytest.raises(GeometryError, match=UNDECIDABLE_CONTACT):
        _pair_paths_exact(_chord_rectangle(1, first, .004, 10), _chord_rectangle(2, second, .004, 20), None, 1e-10,
                          lambda: None, point_contacts=[], face_tolerances=(1e-10, 1e-10))


def test_a_contact_point_with_a_large_location_error_is_not_admitted():
    """Codex K6 R3 H2: r = 1e-6 with the second origin 2^30 m out: the axial solve can move the computed contact point
    by far more than its reach, and the coordinates are 2^30 r: outside the scale gates (coarse box test, no chart)."""
    r, L = 1e-6, 2.**30
    u = (.6, 0., .8)
    first = Cylinder(origin=(0., 0., -1e-8), axis=(0., 0., 1.), radial_direction=(0., 1., 0.), radius=r, height=2e-8,
                     start_angle=-.2, sweep_angle=.4)
    second = Cylinder(origin=tuple(-L*np.asarray(u)), axis=u, radial_direction=(0., 1., 0.),
                      radius=float(np.nextafter(r, 1.)), height=2*L, start_angle=-.2, sweep_angle=.4)
    contact = undecidable_contact(first, second, tolerance=1e-10)
    assert contact is not None and not contact.admitted and contact.support is None


def test_the_reach_covers_the_axial_travel_of_a_steep_branch():
    """Codex K6 R3 M3: axes 3.7 degrees apart, radii 1 and 1 - 7 ulp, tolerance 1e-10: a point 7.8e-4 m from the
    contact still differs from the stored branch by 4.9e-10 m. The admitted near radius (r / 4) is far beyond it."""
    first = Cylinder(origin=(0., 0., -1.), axis=(0., 0., 1.), radial_direction=(0., 1., 0.), radius=1., height=2.,
                     start_angle=-.5, sweep_angle=1.)
    axis = _unit((.064, 0., 1.))
    second = Cylinder(origin=tuple(-axis), axis=tuple(axis), radial_direction=(0., 1., 0.), radius=1.-7*2.**-53,
                      height=2., start_angle=-.5, sweep_angle=1.)
    contact = undecidable_contact(first, second, tolerance=1e-10)
    assert contact is not None
    assert contact.near_radius == math.inf or contact.near_radius >= 16*7.8245e-4


def test_classification_does_not_depend_on_the_order_of_the_faces():
    """Codex K6 R3 M4: |D| = 3.286e-15 against B = 3.220e-15 in one order, undecidable in the other. The contact is
    classified from both orders, so both orders see the same contact points."""
    axis = _unit((1., 1., 0.))
    radial = _unit((-1., 1., 0.))
    first = Cylinder(origin=(0., 0., -2.), axis=(0., 0., 1.), radial_direction=tuple(radial), radius=1., height=4.,
                     start_angle=-.2, sweep_angle=.4)
    second = Cylinder(origin=tuple(-2*axis), axis=tuple(axis), radial_direction=(0., 0., 1.), radius=1.+14*2.**-53,
                      height=4., start_angle=-.2, sweep_angle=.4)
    forward = undecidable_contact(first, second, tolerance=1e-10)
    backward = undecidable_contact(second, first, tolerance=1e-10)
    assert forward is not None and backward is not None
    for point in forward.points:
        assert min(np.linalg.norm(np.asarray(point)-np.asarray(other)) for other in backward.points) < 1e-9
    # Codex K6 R4 M3: every point of both orders is kept (no deduplication), so both orders test the same points.
    assert sorted(zip(forward.points, forward.reaches)) == sorted(zip(backward.points, backward.reaches))
    assert forward.admitted == backward.admitted


def _arc_rectangle(face_id, cylinder, start, sweep, base_edge):
    """A cylinder face over azimuth ``[start, start + sweep]`` (of the cylinder's radial basis) and its full height,
    trimmed by its two circular arcs and two generators, with its own edge ids."""
    from anygeometry.arrangement_geometry import EllipticArc, LinePath
    from anygeometry.material_arrangement import ArrangementPath, MaterialDomain
    origin, axis = np.asarray(cylinder.origin), np.asarray(cylinder.axis)
    radial, circumferential = np.asarray(cylinder.radial_direction), cylinder.circumferential_direction
    at = lambda angle, height: tuple(origin+cylinder.radius*(math.cos(angle)*radial+math.sin(angle)*circumferential)
                                     + height*axis)
    top = origin+cylinder.height*axis
    loop = (ArrangementPath(EllipticArc(tuple(origin), tuple(cylinder.radius*radial),
                                        tuple(cylinder.radius*circumferential), start, sweep), base_edge),
            ArrangementPath(LinePath(at(start+sweep, 0.), at(start+sweep, cylinder.height)), base_edge+1),
            ArrangementPath(EllipticArc(tuple(top), tuple(cylinder.radius*radial),
                                        tuple(cylinder.radius*circumferential), start+sweep, -sweep), base_edge+2),
            ArrangementPath(LinePath(at(start, cylinder.height), at(start, 0.)), base_edge+3))
    return MaterialDomain(face_id, Cylinder(origin=cylinder.origin, axis=cylinder.axis,
                                            radial_direction=cylinder.radial_direction, radius=cylinder.radius,
                                            height=cylinder.height, start_angle=start, sweep_angle=sweep), (loop,))


@pytest.mark.parametrize("offset, required", [(0., True), (8e-6, True), (3.2e-5, False), (1e-3, False)])
def test_the_near_test_on_both_sides_of_the_reach_margin(offset, required):
    """Owner decision 2026-11-19 (option 1): a face requires the witness where the contact point lies in it or within
    ``_REACH_MARGIN`` (16) reaches of its boundary curves. Reach 1e-6: a point in the face (offset 0) and a point
    outside it 8e-6 from its generator (8 reaches) require it; 3.2e-5 (32 reaches) and 1e-3 do not."""
    from anygeometry.batch_intersections import _requires_witness
    cylinder = Cylinder(origin=(0., 0., -1.), axis=(0., 0., 1.), radial_direction=(1., 0., 0.), radius=1., height=2.,
                        start_angle=0., sweep_angle=1.)
    domain = _arc_rectangle(1, cylinder, offset if offset else -.5, .5 if offset else 1., 10)
    point = (1., 0., 0.)                             # azimuth 0: distance to the generator at azimuth ``offset``
    assert _requires_witness(domain, point, 1e-6, 1e-10) is required


def test_a_contact_in_one_face_and_far_from_the_other_takes_mains_path():
    """N2c 551/552 shape (the R3 regression): the contact point is in the first face but a whole azimuth step
    (1e-3 m) away from the second face's boundary, far beyond 16 reaches: no witness is required, the pair takes
    main's path (``_undecidable_contact_support`` is None) and the pair stage succeeds."""
    from anygeometry.batch_intersections import _pair_paths_exact, _undecidable_contact_support
    r = 1.
    first = Cylinder(origin=(0., 0., -2.), axis=(0., 0., 1.), radial_direction=(0., 1., 0.), radius=r, height=4.,
                     start_angle=-.2, sweep_angle=.4)
    second = Cylinder(origin=(-2., 0., 0.), axis=(1., 0., 0.), radial_direction=(0., 1., 0.),
                      radius=float(np.nextafter(r, 2.)), height=4., start_angle=-.2, sweep_angle=.4)
    one, other = _arc_rectangle(1, first, -.2, .4, 10), _arc_rectangle(2, second, .001, .2, 20)
    assert _undecidable_contact_support(one, other, 1e-10) is None
    _pair_paths_exact(one, other, None, 1e-10, lambda: None, point_contacts=[], face_tolerances=(1e-10, 1e-10))
    contact = undecidable_contact(first, second, tolerance=1e-10)
    assert contact is not None and contact.admitted


def test_a_contact_in_both_faces_without_a_shared_edge_refuses():
    """The same pair with the second face over the contact too: both faces require the witness and none is stored."""
    from anygeometry.batch_intersections import UNDECIDABLE_CONTACT, _pair_paths_exact
    from anygeometry.errors import GeometryError
    r = 1.
    first = Cylinder(origin=(0., 0., -2.), axis=(0., 0., 1.), radial_direction=(0., 1., 0.), radius=r, height=4.,
                     start_angle=-.2, sweep_angle=.4)
    second = Cylinder(origin=(-2., 0., 0.), axis=(1., 0., 0.), radial_direction=(0., 1., 0.),
                      radius=float(np.nextafter(r, 2.)), height=4., start_angle=-.2, sweep_angle=.4)
    with pytest.raises(GeometryError, match=UNDECIDABLE_CONTACT):
        _pair_paths_exact(_arc_rectangle(1, first, -.2, .4, 10), _arc_rectangle(2, second, -.2, .4, 20), None, 1e-10,
                          lambda: None, point_contacts=[], face_tolerances=(1e-10, 1e-10))


def test_micro_chord_trims_over_the_contact_still_need_a_witness():
    """Codex K6 R4 H1(a): r = 2e-8, chord trims whose boxes stop 7.6e-9 m short of the contact, which lies in both
    material domains: inside the gates the point-in-face test requires the witness, and the pair refuses."""
    from anygeometry.batch_intersections import UNDECIDABLE_CONTACT, _pair_paths_exact
    from anygeometry.errors import GeometryError
    r, h = 2e-8, 4e-8
    first = Cylinder(origin=(0., 0., -h), axis=(0., 0., 1.), radial_direction=(0., 1., 0.), radius=r, height=2*h,
                     start_angle=-.9, sweep_angle=1.8)
    second = Cylinder(origin=(-h, 0., 0.), axis=(1., 0., 0.), radial_direction=(0., 1., 0.),
                      radius=float(np.nextafter(r, 1.)), height=2*h, start_angle=-.9, sweep_angle=1.8)
    contact = undecidable_contact(first, second, tolerance=1e-10)
    assert contact is not None
    with pytest.raises(GeometryError, match=UNDECIDABLE_CONTACT):
        _pair_paths_exact(_chord_rectangle(1, first, .9, 10), _chord_rectangle(2, second, .9, 20), None, 1e-10,
                          lambda: None, point_contacts=[], face_tolerances=(1e-10, 1e-10))


def test_an_ellipse_far_from_the_origin_is_not_offered():
    """Codex K6 R4 M2: origins 3000 m out, axes 3.8 degrees apart, radii 1 and 1 - 1 ulp: the ellipses lie within
    6.9e-11 m of both cylinders but up to 1.4e-10 m from their intersection (the residual is amplified by 1 / sin of
    the normals' angle). The certificate is required within tolerance |a1 x a2|: no chart is offered."""
    u = _unit((.066, 0., 1.))
    first = Cylinder(origin=(0., 0., 3000.), axis=(0., 0., 1.), radial_direction=(0., 1., 0.), radius=1., height=8000.,
                     start_angle=-2., sweep_angle=4.)
    second = Cylinder(origin=tuple(-2000*u), axis=tuple(u), radial_direction=(0., 1., 0.),
                      radius=float(np.nextafter(1., 0.)), height=8000., start_angle=-2., sweep_angle=4.)
    contact = undecidable_contact(first, second, tolerance=1e-10)
    assert contact is None or contact.support is None


def test_an_exact_double_in_either_order_stays_on_mains_path():
    """Codex K6 R4 M4: unit radii, axes +Z and +X through -0.7 their axes, radial inputs +Y and (0, 1, 1): with +Z
    first both discriminants are exactly zero (K3's exact doubles); the other order rounds the normalised radial
    and classifies them undecidable. Either way the pair is not moved onto the witness path."""
    from anygeometry.analytic_supports import _certified_nodes
    first = Cylinder(origin=(0., 0., -.7), axis=(0., 0., 1.), radial_direction=(0., 1., 0.), radius=1., height=1.4,
                     start_angle=-.1, sweep_angle=.2)
    second = Cylinder(origin=(-.7, 0., 0.), axis=(1., 0., 0.), radial_direction=tuple(_unit((0., 1., 1.))), radius=1.,
                      height=1.4, start_angle=-1., sweep_angle=1.)
    assert _certified_nodes(first, second) or _certified_nodes(second, first)
    assert undecidable_contact(first, second, tolerance=1e-10) is None
    assert undecidable_contact(second, first, tolerance=1e-10) is None


def test_the_far_ellipse_is_withheld_inside_the_gates_by_the_scaled_certificate():
    """Codex K6 R5 low: the R4 M2 geometry at tolerance 5e-10 passes the scale gates in both orders; the 6.9e-11
    residual exceeds tolerance |a1 x a2| = 3.3e-11, so the chart is withheld by the scaled certificate itself."""
    u = _unit((.066, 0., 1.))
    first = Cylinder(origin=(0., 0., 3000.), axis=(0., 0., 1.), radial_direction=(0., 1., 0.), radius=1., height=8000.,
                     start_angle=-2., sweep_angle=4.)
    second = Cylinder(origin=tuple(-2000*u), axis=tuple(u), radial_direction=(0., 1., 0.),
                      radius=float(np.nextafter(1., 0.)), height=8000., start_angle=-2., sweep_angle=4.)
    for a, b in ((first, second), (second, first)):
        contact = undecidable_contact(a, b, tolerance=5e-10)
        assert contact is not None and contact.admitted and contact.support is None


def test_ellipse_admission_does_not_depend_on_the_order_of_the_faces():
    """Codex K6 R5 M1: axes +Z and (1, 0, 1)/sqrt 2 through origins 3000 m apart, radii 1 and 1 - 1 ulp, at a
    tolerance where the two orders' residual bounds straddle the threshold. The ellipses are built from a canonical
    order of the supports: both orders get the same ellipses and the same decision."""
    u = _unit((1., 0., 1.))
    first = Cylinder(origin=(0., 0., -3000.), axis=(0., 0., 1.), radial_direction=(0., 1., 0.), radius=1.,
                     height=8000., start_angle=-2., sweep_angle=4.)
    second = Cylinder(origin=tuple(-2000*u), axis=tuple(u), radial_direction=(0., 1., 0.),
                      radius=float(np.nextafter(1., 0.)), height=8000., start_angle=-2., sweep_angle=4.)
    tolerance = 9.154286229041723e-11
    forward = undecidable_contact(first, second, tolerance=tolerance)
    backward = undecidable_contact(second, first, tolerance=tolerance)
    assert (forward is None) == (backward is None)
    if forward is not None:
        assert (forward.support is None) == (backward.support is None)
        if forward.support is not None:
            assert forward.support.curves == backward.support.curves


def test_the_boundary_near_test_decides_a_just_missing_generator_quickly():
    """Codex K6 R5 M2: a generator that misses the cutoff by a few 1e-15 m stalled the subdivision (2.5 million
    evaluations); straight edges use the closed-form nearest point, other curves a work budget."""
    import time
    from anygeometry.batch_intersections import _boundary_within
    cylinder = Cylinder(origin=(0., 0., -1.), axis=(0., 0., 1.), radial_direction=(1., 0., 0.), radius=1., height=2.,
                        start_angle=0., sweep_angle=1.)
    offset = 7.460698909929148e-5
    domain = _arc_rectangle(1, cylinder, offset, .1, 10)
    distance = 2*math.sin(offset/2)                  # chord from (1, 0, 0) to the generator at azimuth ``offset``
    started = time.perf_counter()
    assert _boundary_within(domain, (1., 0., 0.), distance*(1-1e-10)) is False
    assert _boundary_within(domain, (1., 0., 0.), distance*(1+1e-10)) is True
    assert time.perf_counter()-started < 2.
