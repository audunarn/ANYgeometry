"""Exact level sets of NURBS pcurve coordinates (A1e-lite, owner option a).

References: constructed curves whose crossings, touches and runs are known in
closed form (rational knots and controls), checked exactly.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from anygeometry.exact_levels import level_set
from anygeometry.nurbs import NurbsCurve


def summary(events):
    return None if events is None else [(e.t0, e.t1, e.before, e.after) for e in events]


def quadratic(lift=0.):
    return NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((.1, .75 + lift), (.5, .25 + lift), (.9, .75 + lift)))


def test_tangency_is_one_exact_touch():
    assert summary(level_set(quadratic(), 1, .5)) == [(Fraction(1, 2), Fraction(1, 2), 1, 1)]


def test_a_dip_below_the_level_by_1e_14_gives_two_crossings():
    events = level_set(quadratic(-1e-14), 1, .5)
    assert [(e.before, e.after) for e in events] == [(1, -1), (-1, 1)]
    assert events[1].t0 - events[0].t0 < Fraction(1, 10 ** 6)
    assert level_set(quadratic(1e-14), 1, .5) == ()


def test_four_close_crossings_are_kept():
    curve = NurbsCurve(1, (0., 0., .4999999, .50000005, .5000003, .50000055, .5000007, 1., 1.),
                       ((.1, .3), (.2, .3), (.3, .6), (.4, .1), (.5, .6), (.6, .3), (.9, .3)))
    events = level_set(curve, 1, .5)
    assert len(events) == 4 and all(e.crossing for e in events)


def test_runs_whole_curves_and_reversed_domains():
    run = NurbsCurve(1, (0., 0., .5, 1., 1.), ((.1, .5), (.5, .5), (.9, .9)))
    assert summary(level_set(run, 1, .5)) == [(Fraction(0), Fraction(1, 2), 0, 1)]
    reversed_run = NurbsCurve(1, (0., 0., .1, 1., 1.), ((.1, .3), (.5, .3), (.9, .35)), domain=(1., 0.))
    events = level_set(reversed_run, 1, .3)
    assert len(events) == 1 and events[0].run and events[0].t1 == 1 and events[0].before == 1
    assert level_set(NurbsCurve(1, (0., 0., 1., 1.), ((.1, .5), (.9, .5))), 1, .5) is None


def test_high_order_touch_and_rational_curves():
    quartic = NurbsCurve(4, (0.,) * 5 + (1.,) * 5, ((.1, .5625), (.3, .4375), (.5, .5625), (.7, .4375), (.9, .5625)))
    assert summary(level_set(quartic, 1, .5)) == [(Fraction(1, 2), Fraction(1, 2), 1, 1)]
    arc = NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((1., 0.), (1., 1.), (0., 1.)), (1., .5, 1.))   # quarter circle
    events = level_set(arc, 1, .6)
    assert len(events) == 1 and events[0].crossing
    x = float(arc.evaluate(float(events[0].t0))[1])
    assert abs(x - .6) <= 1e-15


def test_roots_at_knots_and_curve_ends():
    curve = NurbsCurve(1, (0., 0., .5, 1., 1.), ((.1, .2), (.5, .5), (.9, .2)))
    assert summary(level_set(curve, 1, .5)) == [(Fraction(1, 2), Fraction(1, 2), -1, -1)]
    start = NurbsCurve(1, (0., 0., 1., 1.), ((.1, .5), (.9, .8)))
    assert summary(level_set(start, 1, .5)) == [(Fraction(0), Fraction(0), 0, 1)]


def test_a_crossing_closer_to_a_curve_end_than_the_refinement_width():
    line = NurbsCurve(1, (0., 0., 1., 1.), ((0., 0.), (0., 1.)))
    events = level_set(line, 1, 2. ** -100)
    assert len(events) == 1 and events[0].crossing and events[0].lo <= Fraction(2) ** -100 <= events[0].hi


def test_event_side_is_exact_at_irrational_and_rational_roots():
    from anygeometry.exact_levels import event_side
    diagonal = NurbsCurve(1, (0., 0., 1., 1.), ((0., 0.), (1., 1.)), (1., 1.3))
    event = level_set(diagonal, 1, .5)[0]
    assert event_side(diagonal, event, 0, .5) == 0
    import math
    assert event_side(diagonal, event, 0, math.nextafter(.5, 0.)) == 1
    assert event_side(diagonal, event, 0, math.nextafter(.5, 1.)) == -1
    conic = NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((1., 0.), (1., 1.), (0., 1.)), (1., .5, 1.))
    event = level_set(conic, 1, .5)[0]
    assert event.lo < event.hi                                          # an irrational root (isolating interval)
    x = float(conic.evaluate(float(event.t0))[0])
    assert event_side(conic, event, 0, x - 1e-12) == 1 and event_side(conic, event, 0, x + 1e-12) == -1


def test_two_roots_on_either_side_of_a_knot_stay_separate():
    curve = NurbsCurve(1, (0., 0., .5, 1., 1.), ((.1, .2), (.5, .2), (.9, .2)))
    assert level_set(curve, 1, .3) == ()
    zigzag = NurbsCurve(1, (0., 0., .5, 1., 1.), ((.1, .4), (.5, .6), (.9, .4)))
    events = level_set(zigzag, 1, .5)
    assert len(events) == 2 and all(e.crossing for e in events)
