"""Programme step 3, P3a: the two-patch bow (owner Q2: two patches meeting at the stem)."""
from __future__ import annotations

import math

import numpy as np

from anygeometry.bow_example import bow_two_patch


def _curve_definition(curve):
    return curve.degree, tuple(curve.knots), tuple(map(tuple, curve.controls)), curve.weights


def test_the_stem_and_keel_iso_curves_of_both_patches_are_the_same_exact_curves():
    starboard, port = bow_two_patch()
    for fixed, value in (("u", 0.), ("v", 0.)):              # stem (u = 0) and keel (v = 0)
        assert _curve_definition(starboard.iso_curve(fixed, value)) == _curve_definition(port.iso_curve(fixed, value))
    stem = np.asarray(starboard.iso_curve("u", 0.).controls)
    keel = np.asarray(starboard.iso_curve("v", 0.).controls)
    assert np.all(stem[:, 1] == 0.) and np.all(keel[:, 1] == 0.)          # both in the centreline plane


def test_the_port_patch_is_the_exact_mirror_of_the_starboard_patch():
    starboard, port = bow_two_patch()
    assert (port.degree_u, port.degree_v, port.knots_u, port.knots_v) == \
        (starboard.degree_u, starboard.degree_v, starboard.knots_u, starboard.knots_v)
    for row_s, row_p in zip(starboard.controls, port.controls):
        for (xs, ys, zs), (xp, yp, zp) in zip(row_s, row_p):
            assert (xp, yp, zp) == (xs, -ys, zs)
            assert yp != 0. or math.copysign(1., yp) == 1.               # a zero stays +0.0
    uv = np.random.default_rng(3).uniform(0., 1., (10000, 2))
    uv = np.vstack([uv, [[0., t] for t in np.linspace(0., 1., 101)], [[t, 0.] for t in np.linspace(0., 1., 101)]])
    assert np.array_equal(port.evaluate_many(uv), starboard.evaluate_many(uv) * np.array([1., -1., 1.]))


def test_the_patches_meet_along_the_stem_and_the_keel():
    starboard, port = bow_two_patch()
    for points in ([[0., t] for t in np.linspace(0., 1., 201)], [[t, 0.] for t in np.linspace(0., 1., 201)]):
        assert np.array_equal(starboard.evaluate_many(points), port.evaluate_many(points))


def test_the_aft_boundary_is_the_midship_section_at_x_15():
    starboard, _port = bow_two_patch()
    aft = starboard.controls[-1]
    assert all(x == 15. for x, _y, _z in aft)
    bottom, side = aft[:4], aft[3:]
    assert all(z == 0. for _x, _y, z in bottom) and bottom[0][1] == 0. and bottom[-1][1] == 5.5   # keel to chine
    assert all(y == 5.5 for _x, y, _z in side) and side[-1][2] == 3.                              # chine to above the deck
    assert starboard.knots_v.count(.5) == 3                                                        # the chine crease
