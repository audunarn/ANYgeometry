"""NURBS-domain paths that read pcurve points must carry an iso-line's constant coordinate exactly.

``CurveOnSurface.chart`` sets an iso-line pcurve's constant chart coordinate
exactly (``_iso_fixed``); a raw float blend ``(1 - t) c + t c`` can be an ulp
off, and a narrow surface knot span amplifies that ulp in the world. These
regressions cover the level-root endpoint test, the world-area quadrature, the
certified chart polyline and the ``uv`` clip (Codex review of nurbs_domain.py,
2026-10-29).
"""
from types import SimpleNamespace

import numpy as np

from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anygeometry import nurbs_domain

_NARROW = 2. ** -44


def _narrow_surface():
    """Degree 1, a v knot span of width 2**-44 at v = .3 (as tests/test_nurbs_domain_iso_chart.py)."""
    return NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., .3, .3 + _NARROW, 1., 1.),
                        tuple(tuple((float(u), float(y), 0.) for y in range(4)) for u in (0, 1)))


def _narrow_u_surface():
    """The transpose: a u knot span of width 2**-44 at u = .3, control net P[i, j] = (i, j, 0)."""
    return NurbsSurface(1, 1, (0., 0., .3, .3 + _NARROW, 1., 1.), (0., 0., 1., 1.),
                        tuple(tuple((float(i), float(j), 0.) for j in range(2)) for i in range(4)))


def _trimmed_iso_pcurve():
    """``v = .3`` on the trimmed domain (.08, 1): the raw blend at t = 0 gives v = .30000000000000004."""
    return NurbsCurve(1, (0., 0., 1., 1.), ((0., .3), (1., .3)), domain=(.08, 1.))


def test_iso_pcurve_end_does_not_create_a_false_level_root():
    surface = _narrow_surface()
    pcurve = _trimmed_iso_pcurve()
    assert float(pcurve.evaluate(0.)[1]) != .3          # the raw end is an ulp off (the case is live)
    domain = SimpleNamespace(support=surface)
    level = float(np.nextafter(.3, 1.))
    # The exact pcurve has v = .3 everywhere, an ulp below the level; in the world that ulp is
    # about 2**44 * 5.6e-17 ~ 1e-3 >> 1e-12, so there is no root, not even at an end.
    assert nurbs_domain.horizontal_roots(domain, CurveOnSurface(surface, pcurve), level, 1e-12) == ()
    assert nurbs_domain.horizontal_roots(domain, CurveOnSurface(surface, pcurve), .3, 1e-12) is None


def test_world_area_of_an_iso_rectangle_across_a_narrow_u_span_is_exact():
    surface = _narrow_u_surface()
    corners = [(0., 0.), (.3, 0.), (.3, 1.), (0., 1.)]
    loop = [SimpleNamespace(curve=CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.),
                                                                     (corners[k], corners[(k + 1) % 4]))))
            for k in range(4)]
    domain = SimpleNamespace(support=surface)
    # The chart rectangle [0, .3] x [0, 1] maps onto the world square [0, 1] x [0, 1].
    value, change = nurbs_domain.world_area_estimate(domain, loop, 1e-12)
    assert abs(value - 1.0) <= 1e-12 and change <= 1e-12
    assert abs(nurbs_domain.area_loop(domain, loop, 1e-12) - .3) <= 4 * np.finfo(float).eps


def test_certified_polyline_of_a_trimmed_iso_pcurve_on_a_narrow_span_is_reachable():
    surface = _narrow_surface()
    pcurve = _trimmed_iso_pcurve()
    polyline, bound = nurbs_domain.certified_chart_polyline(surface, pcurve, 1e-6)
    assert bound <= 1e-6
    assert np.all(polyline[:, 1] == .3)
    assert polyline[0, 0] == .08 and polyline[-1, 0] == 1.


def test_uv_of_an_ordinary_pcurve_is_the_clipped_chart_not_the_raw_evaluation():
    surface = _narrow_surface()
    pcurve = NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((1., .1), (1., .3), (1. - 2. ** -53, .9)))
    curve = CurveOnSurface(surface, pcurve)
    domain = SimpleNamespace(support=surface)
    assert float(pcurve.evaluate(.061)[0]) > 1.0         # raw rounding leaves the chart (the case is live)
    point = nurbs_domain.uv(domain, curve, .061)
    assert point.tobytes() == np.asarray(curve.chart(.061), dtype=float).tobytes()
    assert point[0] == 1.0
    for t in np.linspace(0., 1., 101):
        assert nurbs_domain.uv(domain, curve, float(t)).tobytes() == np.asarray(
            curve.chart(float(t)), dtype=float).tobytes()
