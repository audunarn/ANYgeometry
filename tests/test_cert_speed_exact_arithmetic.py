"""Exact oracles and generic fallbacks for certification arithmetic."""
from fractions import Fraction as Q
from itertools import product
from math import comb
import random

import pytest

from anygeometry.support_curves import _de_casteljau
from anygeometry.nurbs_metric import mul, scale
from anygeometry.nurbs import NurbsSurface, _exact_chart, _exact_span, _exact_basis


def old_casteljau(values, t):
    work = list(values)
    for _ in range(len(work) - 1):
        work = [(1-t)*a+t*b for a, b in zip(work, work[1:])]
    return work[0]


@pytest.mark.parametrize('degree', range(9))
def test_casteljau_exact_two_oracles(degree):
    rng = random.Random(481 + degree)
    for _ in range(12):
        values = [Q(rng.randrange(-2**180, 2**180), rng.randrange(1, 1000))
                  for _ in range(degree + 1)]
        for t in (Q(0), Q(1), Q(-3, 7), Q(9, 4), Q(17, 31)):
            actual = _de_casteljau(values, t)
            bernstein = sum(comb(degree, k) * (1-t)**(degree-k) * t**k * value
                            for k, value in enumerate(values))
            assert actual == old_casteljau(values, t) == bernstein
            assert type(actual) is Q


def test_casteljau_empty_single_and_generic_fallback():
    with pytest.raises(IndexError):
        _de_casteljau([], Q(1, 3))
    sentinel = object()
    assert _de_casteljau([sentinel], sentinel) is sentinel
    class Observed(Q):
        def __rmul__(self, other):
            calls.append(('mul', other))
            return super().__rmul__(other)
    calls = []
    values = [Observed(1, 3), Observed(7, 9)]
    expected = old_casteljau(values, Q(2, 7))
    trace = calls[:]
    calls.clear()
    assert _de_casteljau(values, Q(2, 7)) == expected
    assert calls == trace
    for values, t in (([1, 2, 4], .3), ([Q(1), Q(2)], Observed(1, 2))):
        assert _de_casteljau(values, t) == old_casteljau(values, t)


def test_interval_all_sign_cases_unordered_and_generic():
    endpoints = [Q(-7, 3), Q(-1, 5), Q(0), Q(2, 7), Q(11, 3)]
    intervals = list(product(endpoints, repeat=2))
    for a, b in product(intervals, repeat=2):
        products = [x*y for x in a for y in b]
        assert mul(a, b) == (min(products), max(products))
    for a, factor in product(intervals, endpoints):
        products = [x*factor for x in a]
        assert scale(a, factor) == (min(products), max(products))
    for a, b in (([1, 2, 3], [-2, 5]), ((.3, .7), (-.2, .9)), ((Q(3),), (Q(2),))):
        products = [x*y for x in a for y in b]
        assert mul(a, b) == (min(products), max(products))


def old_surface(surface, u, v, net):
    s, t = _exact_chart(u, surface.u_range), _exact_chart(v, surface.v_range)
    ku, kv = list(map(Q, surface.knots_u)), list(map(Q, surface.knots_v))
    p, q = surface.degree_u, surface.degree_v
    su, sv = _exact_span(p, ku, len(net), s), _exact_span(q, kv, len(net[0]), t)
    bu, bv = _exact_basis(p, ku, su, s), _exact_basis(q, kv, sv, t)
    total = [Q(0)]*4
    for a, nu in enumerate(bu):
        for b, nv in enumerate(bv):
            entry = net[su-p+a][sv-q+b]
            factor = nu*nv
            total = [x+factor*y for x, y in zip(total, entry)]
    return tuple(value/total[3] for value in total[:3])


@pytest.mark.parametrize('degree', [1, 2, 3])
def test_surface_exact_accumulation_and_bernstein(degree):
    n = degree + 1
    knots = (0.,)*n + (1.,)*n
    surface = NurbsSurface(degree, degree, knots, knots,
        tuple(tuple((float(a), float(b), float(a*b)) for b in range(n)) for a in range(n)))
    net = tuple(tuple(tuple(Q((a+1)*(axis+1)-b, 2**(a+b+1)+axis+1) for axis in range(3))
                      + (Q(a+b+1, 7),) for b in range(n)) for a in range(n))
    for u, v in product((Q(0), Q(1), Q(2, 7), Q(5, 9)), repeat=2):
        actual = surface.exact_point(u, v, net)
        total = [sum(comb(degree, a)*u**a*(1-u)**(degree-a)
                     *comb(degree, b)*v**b*(1-v)**(degree-b)*net[a][b][axis]
                     for a in range(n) for b in range(n)) for axis in range(4)]
        assert actual == old_surface(surface, u, v, net)
        assert actual == tuple(value/total[3] for value in total[:3])
        assert surface.exact_point(u, v, list(net)) == actual
        assert surface.exact_point(u, v) == old_surface(surface, u, v, surface.exact_net())


def test_surface_subclass_fallback_and_errors():
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                          (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    calls = []
    class Observed(Q):
        def __rmul__(self, other):
            calls.append(other)
            return super().__rmul__(other)
    net = tuple(tuple(tuple(Observed(x) for x in entry) for entry in row) for row in surface.exact_net())
    expected = old_surface(surface, Q(1, 3), Q(2, 7), net)
    trace = calls[:]
    calls.clear()
    assert surface.exact_point(Q(1, 3), Q(2, 7), net) == expected
    assert calls == trace
    zero = tuple(tuple((Q(1), Q(2), Q(3), Q(0)) for _ in range(2)) for _ in range(2))
    for net in (zero, (), (((),),)):
        with pytest.raises(Exception) as reference:
            old_surface(surface, Q(1, 3), Q(2, 7), net)
        with pytest.raises(type(reference.value)):
            surface.exact_point(Q(1, 3), Q(2, 7), net)
