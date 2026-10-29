"""Joint trim clearance must consume actual COS geometry, never the iso cache."""
from fractions import Fraction as Q

import pytest

from anygeometry import GeometryModel, nurbs
from anygeometry.definition_binding import definition_checksum
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anygeometry._canonical_plate_joint import (
    JointRefused, _certified_world_bounds, trims_clear,
)
from anygeometry.plane_nurbs import CURRENT_WORK, PlaneNurbsNotCertified
from anygeometry.section_audit import Work


def _trim_fixture(diagonal):
    surface = NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
        tuple(((x, 0., z), (x, 1., z)) for x, z in ((0., 0.), (.5, 2.), (1., 0.))))
    uv = ((0., 0.), (1., 1.), (0., 1.)) if diagonal else (
        (0., 0.), (1., 0.), (1., 1.), (0., 1.))
    model = GeometryModel()
    vertices = model.add_points(tuple(tuple(surface.evaluate(*p)) for p in uv))
    curves = tuple(CurveOnSurface(surface,
        NurbsCurve(1, (0., 0., 1., 1.), (p, uv[(i+1) % len(uv)])))
        for i, p in enumerate(uv))
    edges = [model.add_curve(vertices[i], vertices[(i+1) % len(uv)], curve)
             for i, curve in enumerate(curves)]
    face = model.add_face(edges, surface=surface)
    far = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 10.), (0., 1., 10.)), ((1., 0., 10.), (1., 1., 10.))))
    far_curve = CurveOnSurface(far,
        NurbsCurve(1, (0., 0., 1., 1.), ((0., 0.), (1., 0.))))
    return model, face, curves[0], far_curve._iso_world(fresh=True)


@pytest.mark.parametrize('diagonal', [False, True], ids=['iso', 'noniso'])
def test_intersecting_actual_trim_refuses_far_iso_cache_clearance(monkeypatch, diagonal):
    model, face, curve, poison = _trim_fixture(diagonal)
    assert (curve._iso_world(fresh=True) is None) is diagonal
    signature = definition_checksum(curve)
    support, pcurve = curve.support, curve.pcurve
    cache = {curve: poison}
    monkeypatch.setattr(nurbs, '_ISO_CACHE', cache)
    # Show the adversarial entry would incorrectly clear the real neighbourhood.
    assert curve.bounds(.49, .51)[0][2] > 9
    low, high = _certified_world_bounds(curve, .49, .51)
    assert low[2] <= 1 <= high[2]
    y = Q(1, 2) if diagonal else Q(0)
    lower, upper = (Q(49, 100), y-Q(1, 100), Q(99, 100)), (
        Q(51, 100), y+Q(1, 100), Q(101, 100))
    calls = []
    token = CURRENT_WORK.set(Work(100, 2))
    try:
        with pytest.raises(JointRefused, match='not certified clear'):
            trims_clear(model, face, lower, upper, None, lambda: calls.append(1))
    finally:
        CURRENT_WORK.reset(token)
    assert calls
    assert curve.support is support and curve.pcurve is pcurve
    assert definition_checksum(curve) == signature
    assert nurbs._ISO_CACHE is cache and tuple(cache) == (curve,) and cache[curve] is poison


def test_fresh_joint_bounds_preserve_charge_before_enclosure(monkeypatch):
    _model, _face, curve, poison = _trim_fixture(False)
    cache = {curve: poison}
    monkeypatch.setattr(nurbs, '_ISO_CACHE', cache)
    token = CURRENT_WORK.set(Work(0, 0))
    try:
        with pytest.raises(PlaneNurbsNotCertified, match='budget exhausted'):
            _certified_world_bounds(curve, 0., 1.)
    finally:
        CURRENT_WORK.reset(token)
    assert nurbs._ISO_CACHE is cache and cache[curve] is poison


def test_trim_clearance_callback_exception_propagates_before_bounds(monkeypatch):
    model, face, curve, poison = _trim_fixture(True)
    cache = {curve: poison}
    monkeypatch.setattr(nurbs, '_ISO_CACHE', cache)
    original = BaseException('cancel joint clearance')
    def cancel():
        raise original
    with pytest.raises(BaseException) as caught:
        trims_clear(model, face, (Q(0),)*3, (Q(2),)*3, None, cancel)
    assert caught.value is original
    assert nurbs._ISO_CACHE is cache and cache[curve] is poison
