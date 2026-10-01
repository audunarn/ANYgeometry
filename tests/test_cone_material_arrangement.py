"""A Cone face is a material chart like a Cylinder face: the angular/axial chart carries the arrangement.

Exact generator lines and rings (both analytic) split a conical facet; the native cell areas must be the
exact products of the cut fractions, in a tilted cone as well as an upright one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry.arrangement_geometry import LinePath
from anygeometry.exact_curves import EllipticArc
from anygeometry.generators import cone
from anygeometry.material_arrangement import ArrangementPath, MaterialDomain, arrange_material


def _facet(axis=(0., 0., 1.), radial=(1., 0., 0.), segments=8, r0=.5, r1=1.5, height=2.):
    model = cone(r0, r1, height, origin=(.3, -.2, .1), axis=axis, radial_direction=radial,
                 circumferential_segments=segments)
    face_id = sorted(model.faces)[2]
    return model, face_id, MaterialDomain.from_model(model, face_id)


def _generator(support, u):
    return ArrangementPath(LinePath(tuple(support.evaluate(u, 0.)), tuple(support.evaluate(u, 1.))))


def _ring(support, v):
    radius = (1 - v) * support.radius_start + v * support.radius_end
    center = np.asarray(support.origin) + v * support.height * np.asarray(support.axis)
    return ArrangementPath(EllipticArc(center, radius * np.asarray(support.radial_direction),
                                       radius * np.asarray(support.circumferential_direction),
                                       support.start_angle, support.sweep_angle))


def _native_areas(arrangement, domain, tolerance):
    result = []
    for cell in arrangement.cells:
        loop = tuple(ArrangementPath(arrangement.paths[i].curve if forward else arrangement.paths[i].curve.subcurve(1., 0.))
                     for i, forward in cell.outer)
        result.append(abs(domain.area_loop(loop, tolerance)))
    return sorted(result)


@pytest.mark.parametrize("axis,radial", [((0., 0., 1.), (1., 0., 0.)),
                                         ((math.cos(math.radians(10)), 0., math.sin(math.radians(10))), (0., 1., 0.)),
                                         ((.3, -.5, .8), (1., 0., 0.))])
def test_generator_and_ring_cuts_split_a_cone_facet_exactly(axis, radial):
    model, _face, domain = _facet(axis, radial)
    support = domain.support
    tolerance = model.tolerance.length
    u, v = .3, .6
    cases = {
        "generator": ([_generator(support, u)], [u, 1 - u]),
        "ring": ([_ring(support, v)], [v, 1 - v]),
        "both": ([_generator(support, u), _ring(support, v)], [u * v, u * (1 - v), (1 - u) * v, (1 - u) * (1 - v)]),
    }
    for name, (traces, fractions) in cases.items():
        arrangement = arrange_material(domain, traces, tolerance=tolerance, area_tolerance=model.tolerance.effective_area(2.))
        assert len(arrangement.cells) == len(fractions), name
        assert not any(cell.holes for cell in arrangement.cells)
        areas = _native_areas(arrangement, domain, 1e-14)
        assert areas == pytest.approx(sorted(fractions), abs=1e-11), name
        assert sum(areas) == pytest.approx(1., abs=1e-11)


def test_cone_chart_metric_scales_with_the_local_radius():
    _model, _face, domain = _facet(r0=.5, r1=1.5)
    support = domain.support
    slant = math.hypot(support.height, support.radius_end - support.radius_start)
    for v in (0., .5, 1.):
        radius = (1 - v) * support.radius_start + v * support.radius_end
        world = domain.world_delta(np.array([1., 1.]), v)
        assert world[0] == pytest.approx(radius * support.sweep_angle)
        assert world[1] == pytest.approx(slant)
    assert domain.area_jacobian >= abs(support.radius_end * support.sweep_angle * slant) - 1e-12


def test_cone_facet_without_traces_is_one_cell_of_unit_native_area():
    model, _face, domain = _facet()
    arrangement = arrange_material(domain, [], tolerance=model.tolerance.length,
                                   area_tolerance=model.tolerance.effective_area(2.))
    assert len(arrangement.cells) == 1
    assert _native_areas(arrangement, domain, 1e-14) == pytest.approx([1.], abs=1e-12)
