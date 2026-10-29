"""``nurbs_domain.uv`` must go through ``CurveOnSurface.chart`` (exact iso coordinate), as before the integration."""
from types import SimpleNamespace

import numpy as np

from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anygeometry import nurbs_domain


def _narrow_surface():
    d = 2. ** -44
    return NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., .3, .3 + d, 1., 1.),
                        tuple(tuple((float(u), float(y), 0.) for y in range(4)) for u in (0, 1)))


def test_iso_pcurve_uv_carries_the_constant_coordinate_exactly():
    surface = _narrow_surface()
    curve = CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), ((0., .3), (1., .3))))
    domain = SimpleNamespace(support=surface)
    point = nurbs_domain.uv(domain, curve, .08)
    assert point[1] == .3 and point[0] == .08
    for t in np.linspace(0., 1., 101):
        assert tuple(nurbs_domain.uv(domain, curve, float(t))) == tuple(curve.chart(float(t)))
    assert float(surface.evaluate(float(point[0]), float(point[1]))[1]) == float(curve.evaluate(.08)[1])


def test_ordinary_curved_pcurve_uv_is_bit_identical_to_chart():
    surface = _narrow_surface()
    curve = CurveOnSurface(surface, NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((.1, .1), (.8, .3), (.6, .9))))
    domain = SimpleNamespace(support=surface)
    for t in np.linspace(0., 1., 101):
        assert tuple(nurbs_domain.uv(domain, curve, float(t))) == tuple(curve.chart(float(t)))
