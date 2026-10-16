"""Scalar fast paths decide exactly as the array routes they replace (slice S1).

Owner rule D1 for this programme: topology, classification and certificates
must not change. Each case below is a counterexample raised in independent
review of an earlier draft (tolerance boundaries, norm underflow/overflow,
forged private state, floating-point error handling); the fast paths must fall
back to the original route for all of them.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry import Cone, Cylinder, EllipticArc
from anygeometry._cross3 import _gauss_legendre
from anygeometry.material_arrangement import _VertexLookup
from anygeometry.surfaces import _angle_on_sweep, _vector3


def scan(points, tolerance):
    """The original lookup: lowest earlier index within tolerance, else a new vertex."""
    vertices, result = [], []
    for point in points:
        point = np.asarray(point, dtype=float)
        for index, candidate in enumerate(vertices):
            if np.linalg.norm(point - candidate) <= tolerance:
                result.append(index)
                break
        else:
            vertices.append(point)
            result.append(len(vertices) - 1)
    return result


def lookup(points, tolerance):
    vertices, result = [], []
    index = _VertexLookup(vertices, tolerance)
    for point in points:
        point = np.asarray(point, dtype=float)
        found = index.find(point)
        if found is None:
            index.add(point)
            vertices.append(point)
            found = len(vertices) - 1
        result.append(found)
    return result


@pytest.mark.parametrize("points,tolerance", [
    ([(np.nextafter(1., 0.), 0., 0.), (2., 0., 0.)], 1.),          # distance rounds to exactly the tolerance
    ([(0., 0., 0.), (1e-170, 0., 0.)], 1e-200),                       # the norm underflows to zero
    ([(1., 0., 0.), (1. + 1e-320, 0., 0.)], 1e-320),                  # cell index would overflow
    ([(0., 0., 0.), (np.nan, 0., 0.), (0., 0., 1e-12)], 1e-9),        # non-finite point
    ([(1e250, 0., 0.), (1e250, 0., 0.)], 1e-9),                        # huge coordinates
])
def test_vertex_lookup_matches_the_scan_at_numerical_boundaries(points, tolerance):
    assert lookup(points, tolerance) == scan(points, tolerance)


def test_vertex_lookup_keeps_the_scan_error_for_mixed_dimensions():
    with pytest.raises(ValueError):
        scan([(0., 0.), (0., 0., 0.)], 1e-9)
    with pytest.raises(ValueError):
        lookup([(0., 0.), (0., 0., 0.)], 1e-9)


@pytest.mark.parametrize("seed", range(4))
def test_vertex_lookup_matches_the_scan_on_clustered_points(seed):
    rng = np.random.default_rng(seed)
    tolerance = 10.0 ** rng.uniform(-9, -2)
    centres = rng.uniform(-5, 5, (40, 3))
    points = centres[rng.integers(0, 40, 600)] + rng.normal(0, tolerance, (600, 3)) * rng.choice([0, .5, 1, 2], (600, 1))
    assert lookup(points, tolerance) == scan(points, tolerance)


def old_axial_uv(surface, point):
    """The array route of Cylinder/Cone.local_uv before S1."""
    offset = _vector3(point, "point") - surface.origin
    axial = float(offset @ surface.axis)
    radial = offset - axial * surface.axis
    angle = float(np.arctan2(radial @ surface.circumferential_direction, radial @ surface.radial_direction))
    delta = _angle_on_sweep(angle, surface.start_angle, surface.sweep_angle)
    return delta / surface.sweep_angle, axial / surface.height


SURFACES = (
    Cylinder((0.1, 0.2, 0.3), (0.1, 0.0, 1.0), (1, 0, 0), 0.8, 2.0, 0.3, 4.0),
    Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 0.5, 1.5, 3.0, -1.0, 5.0),
)


@pytest.mark.parametrize("surface", SURFACES)
def test_axial_local_uv_agrees_with_the_array_route(surface):
    rng = np.random.default_rng(5)
    for point in rng.standard_normal((2000, 3)):
        np.testing.assert_allclose(surface.local_uv(point), old_axial_uv(surface, point), rtol=0, atol=1e-14)


def test_forged_tuple_origin_still_evaluates():
    surface = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1., 2.)
    object.__setattr__(surface, "origin", (0., 0., 0.))
    assert surface.local_uv((0., 1., 1.)) == pytest.approx(old_axial_uv(surface, (0., 1., 1.)))


def test_overflow_keeps_numpy_floating_point_error_handling():
    surface = Cylinder((1e308, 0., 0.), (0., 0., 1.), (1., 0., 0.), 1., 2.)
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(FloatingPointError):
            surface.local_uv((-1e308, 0., 0.))


def test_forged_float32_ellipse_keeps_float64_results():
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    for name in ("center", "u_vector", "v_vector"):
        object.__setattr__(arc, name, np.asarray(getattr(arc, name), dtype=np.float32))
    for parameter in (0.125, np.float64(0.125)):
        point = arc.evaluate(parameter)
        assert point.dtype == np.float64
        assert point[0] == np.cos(np.float64(math.tau) * 0.125)


def test_forged_infinite_angle_keeps_the_array_result():
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    object.__setattr__(arc, "start_angle", math.inf)
    with np.errstate(invalid="ignore"):
        assert np.all(np.isnan(arc.evaluate(0.5)))
        assert np.all(np.isnan(arc.derivative(0.5)))


def test_ellipse_scalar_route_matches_the_array_route():
    arc = EllipticArc((1., 2., 3.), (2., 0.5, 0.), (0., 1., 0.25), 0.4, -3.)
    for t in np.linspace(0., 1., 97):
        np.testing.assert_allclose(arc.evaluate(float(t)), arc.evaluate(np.array([t]))[0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(arc.derivative(float(t)), arc.derivative(np.array([t]))[0], rtol=0, atol=1e-14)
    with pytest.raises(Exception, match="curve parameters must be finite and in"):
        arc.evaluate(1.5)


@pytest.mark.parametrize("count", [16, 24, 64])
def test_cached_gauss_legendre_rules_are_identical_fresh_copies(count):
    nodes, weights = _gauss_legendre(count)
    fresh = np.polynomial.legendre.leggauss(count)
    assert np.array_equal(nodes, fresh[0]) and np.array_equal(weights, fresh[1])
    nodes[:] = 0.
    weights[:] = 0.                                   # the caller's copy, not the cache
    again = _gauss_legendre(count)
    assert np.array_equal(again[0], fresh[0]) and np.array_equal(again[1], fresh[1])


# Second review round: NumPy floating-point error modes and forged shapes.


def outcome(route, points, tolerance):
    try:
        return route(points, tolerance)
    except FloatingPointError as error:
        return type(error), str(error)


def test_vertex_lookup_keeps_overflow_errors_of_far_vertices():
    # Whether np.linalg.norm raises on overflow depends on the platform's BLAS and NumPy (it does
    # not on macOS arm64 or NumPy 1.26): the lookup must behave as the original scan does there.
    points = [(0., 0., 0.), (1e199, 0., 0.)]
    with np.errstate(over="raise"):
        try:
            np.linalg.norm(np.asarray(points[1]) - np.asarray(points[0]))
            raises = False
        except FloatingPointError:
            raises = True
        expected = outcome(scan, points, 1.)
        assert isinstance(expected, tuple) == raises
        assert outcome(lookup, points, 1.) == expected


def test_vertex_lookup_matches_the_scan_with_underflow_raising():
    points = [(0., 0., 0.), (1., 1e-170, 0.), (1., 2e-170, 0.)]
    with np.errstate(under="raise"):
        assert outcome(lookup, points, 1e-9) == outcome(scan, points, 1e-9)


def test_axial_local_uv_keeps_underflow_errors():
    surface = Cylinder((0., 0., 0.), (.1, 0., 1.), (1., 0., 0.), 1., 2.)
    with np.errstate(under="raise"):
        with pytest.raises(FloatingPointError):
            surface.local_uv((0., 0., 1e-320))


def test_ellipse_keeps_underflow_errors():
    arc = EllipticArc((0., 0., 0.), (1., 0., 1e-320), (0., 1., 0.))
    with np.errstate(under="raise"):
        with pytest.raises(FloatingPointError):
            arc.evaluate(.125)
        with pytest.raises(FloatingPointError):
            arc.derivative(.125)


def test_ellipse_overflowing_forged_angle_takes_the_array_route():
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    object.__setattr__(arc, "start_angle", 1e308)
    object.__setattr__(arc, "sweep_angle", 1e308)
    with np.errstate(over="ignore", invalid="ignore"):
        assert np.all(np.isnan(arc.evaluate(1.0)))
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(FloatingPointError):
            arc.evaluate(1.0)


def test_ellipse_forged_short_center_broadcasts_as_before():
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    object.__setattr__(arc, "center", (0.,))
    np.testing.assert_allclose(arc.evaluate(0.0), [1., 0., 0.], atol=1e-15)


def test_vertex_lookup_keeps_float32_norm_behaviour():
    # float32 differences underflow to zero in the norm; the scan then reuses vertex 0.
    points = [np.zeros(3, dtype=np.float32), np.array([1e-23, 0, 0], dtype=np.float32)]

    def run(route_points, use_lookup):
        vertices, result = [], []
        index = _VertexLookup(vertices, 1e-30) if use_lookup else None
        for point in route_points:
            found = index.find(point) if use_lookup else next(
                (i for i, c in enumerate(vertices) if np.linalg.norm(point - c) <= 1e-30), None)
            if found is None:
                if use_lookup:
                    index.add(point)
                vertices.append(point)
                found = len(vertices) - 1
            result.append(found)
        return result

    assert run(points, True) == run(points, False)


def test_definition_value_fast_path_leaves_sequence_subclasses_alone():
    from anygeometry.definition_binding import definition_value

    class L(list):
        def __len__(self):
            raise RuntimeError("unexpected len")

    assert definition_value(L([1, True, "a", None])) == [1, True, "a", None]
    assert definition_value((1.5, 2, None, "x", False)) == [1.5, 2, None, "x", False]


def test_definition_value_fast_path_uses_type_identity():
    from dataclasses import dataclass
    from anygeometry.definition_binding import definition_checksum, definition_value

    class Meta(type):
        def __eq__(cls, other):
            return other is type(None)
        __hash__ = type.__hash__

    @dataclass(frozen=True)
    class D(metaclass=Meta):
        x: int

    assert definition_value([D(1)]) == [{"type": D.__qualname__, "fields": {"x": 1}}]
    definition_checksum([D(1)])                       # serialisable as on main


def test_definition_value_fast_path_runs_no_user_equality():
    from dataclasses import dataclass
    from anygeometry.definition_binding import definition_value

    class Meta(type):
        def __eq__(cls, other):
            if other is str:
                cls.probes += 1
                if cls.probes > 1:
                    raise RuntimeError("duplicate primitive probe")
            return False
        __hash__ = type.__hash__

    @dataclass(frozen=True)
    class D(metaclass=Meta):
        x: int

    D.probes = 0
    assert definition_value([D(1)]) == [{"type": D.__qualname__, "fields": {"x": 1}}]
    assert D.probes == 1                              # exactly as many probes as main
