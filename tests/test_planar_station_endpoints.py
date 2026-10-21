"""Endpoint reuse has the same exact values and conservative arithmetic scope."""
from fractions import Fraction
import random
import struct

import pytest

from anygeometry.planar_stations import Segment, at, junctions, point_parameters


def reference(segment, t):
    return tuple(a + t * (b-a) for a, b in zip(*segment))


@pytest.mark.parametrize('derived', [False, True])
def test_endpoint_interpolation_matches_exact_formula(derived):
    rng = random.Random(41021)
    for _ in range(80):
        points = [tuple(Fraction(rng.randrange(-10000, 10001), rng.randrange(1, 1000))
                        for _ in range(3)) for _ in range(2)]
        segment = Segment(*points, derived=derived)
        for t in (0, 1, Fraction(0), Fraction(1), Fraction(1, 3), Fraction(-1, 7)):
            actual = at(segment, t)
            assert actual == reference(segment, t)
            assert all(type(value) is Fraction for value in actual)


def test_float_coordinates_keep_signed_zero_arithmetic():
    segment = Segment((-0., 1., 2.), (0., 4., 6.))
    for t in (0, 1, 0., 1.):
        assert [struct.pack('d', x) for x in at(segment, t)] == [
            struct.pack('d', x) for x in reference(segment, t)]


def test_custom_segment_operators_are_not_bypassed():
    calls=[]
    class Number(Fraction):
        def __sub__(self, other):
            calls.append('subtract')
            return super().__sub__(other)
    segment=Segment((Number(0), Number(0), Number(0)),
                    (Number(1), Number(2), Number(3)))
    calls.clear()
    assert at(segment, 0) == (0, 0, 0)
    assert calls == ['subtract']*3


def test_near_endpoint_incidence_and_degenerate_refusal_unchanged():
    segment=Segment(tuple(map(Fraction,(0,0,0))),tuple(map(Fraction,(1,0,0))))
    for x in (Fraction(0),Fraction(1),Fraction(1,10**12),1-Fraction(1,10**12)):
        point=(x,Fraction(0),Fraction(0))
        assert point_parameters(segment, point, 1e-9) == ((Fraction(0),) if x<.5 else (Fraction(1),))
    reversed_segment=Segment(segment[1],segment[0])
    assert junctions(segment,reversed_segment,1e-9) == ((Fraction(0),Fraction(1)),(Fraction(1),Fraction(0)))
    with pytest.raises(Exception, match='degenerate arrangement line'):
        point_parameters(Segment(segment[0],segment[0]),segment[0],1e-9)
