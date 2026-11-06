"""Chain evaluation must not refuse a rounding-level overshoot of the local edge parameter.

``GeometryModel._chain_eval`` maps a chain fraction to one edge and a local parameter. In real
arithmetic that parameter lies in [0, 1]; the float sum/cumsum/subtract/divide can carry it a few
ulp past an end (S0d: ``1.0000000000000002`` at fraction 1 on a two-arc side), and an
``EllipticArc`` then refuses. Only a rounding-level excursion is snapped; a larger one still refuses
with the same error, and every evaluation that worked before is bit-identical.
"""

from fractions import Fraction
import math

import numpy as np
import pytest

from anygeometry.errors import GeometryError
from anygeometry.exact_curves import EllipticArc
import anygeometry.model as model_module
from anygeometry.model import GeometryModel, OrientedEdge

EPS = float(np.finfo(float).eps)

# Original qualified two-arc producer capture cb007f9f65c46444695c70d228d0adbb4d70795986e89755adb0f897618735a5.
# The circle/sweeps below are real geometry; these are its complete historical
# chain-arithmetic inputs, not a demand that every CPU quadrature return them.
_SPLIT = '0x1.3333333333333p-2'
_LENGTHS = ('0x1.3333333333332p-2', '0x1.4cccccccccccdp+0')
_TOTAL = '0x1.999999999999ap+0'
_CUMULATIVE = ('0x0.0p+0', _LENGTHS[0], _TOTAL)
_OLD_LOCAL = '0x1.0000000000001p+0'


def _rounding_bound(data, index):
    """The snap limit: the derived rounding bound of the index/local arithmetic (S0e replaced S0d's
    blanket 8-ulp constant, which refused real rounding on chains with a short last edge)."""

    chain, lengths, total, _cumulative, _edges = data
    return float(model_module._chain_local_rounding_bound(len(chain), total, float(lengths[index])))


def test_snap_limit_is_the_derived_bound_not_a_blanket_constant():
    assert not hasattr(model_module, "_CHAIN_LOCAL_ROUNDING_SLACK")
    geometry, chain = _two_arc_side()
    data = _captured_chain_data(geometry, chain)
    _chain, lengths, total, _cumulative, _edges = data
    bound = _rounding_bound(data, 1)
    # First order eps * ((n - 1) * T/L + 1) for n = 2; it covers the observed 1-ulp overshoot.
    assert abs(bound / (EPS * (total / float(lengths[1]) + 1.0)) - 1.0) < 1e-9
    assert _old_local(data, 1.0)[1] - 1.0 <= bound


def _arc_side(split, *, reverse_second=False):
    """A side of two unit-circle arcs, angles 0..split and split..1.6."""

    geometry = GeometryModel()
    first = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., split)
    second = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), split, 1.6 - split)
    p0, p1 = geometry.add_points(first.evaluate(np.asarray((0., 1.))))
    (p2,) = geometry.add_points(second.evaluate(np.asarray((1.,))))
    e0 = geometry.add_curve(p0, p1, first)
    if reverse_second:
        back = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 1.6, split - 1.6)
        e1 = geometry.add_curve(p2, p1, back)
        chain = (OrientedEdge(e0, True), OrientedEdge(e1, False))
    else:
        e1 = geometry.add_curve(p1, p2, second)
        chain = (OrientedEdge(e0, True), OrientedEdge(e1, True))
    return geometry, chain


def _two_arc_side(*, reverse_second=False):
    """Fixed original two-arc geometry; overshoot premises use captured inputs."""
    return _arc_side(float.fromhex(_SPLIT), reverse_second=reverse_second)


def _captured_chain_data(geometry, chain):
    live = geometry._chain_data(chain)
    lengths = np.asarray([float.fromhex(value) for value in _LENGTHS])
    total = float.fromhex(_TOTAL)
    cumulative = np.asarray([float.fromhex(value) for value in _CUMULATIVE])
    assert len(chain) == 2 and chain[0].forward
    reverse = not chain[1].forward
    expected = ((0., float.fromhex(_SPLIT)),
                (1.6, -float.fromhex(_LENGTHS[1])) if reverse else
                (float.fromhex(_SPLIT), float.fromhex(_LENGTHS[1])))
    for item, length, (start, sweep) in zip(chain, lengths, expected, strict=True):
        curve = geometry.edges[item.edge].curve
        assert type(curve) is EllipticArc
        assert (curve.center, curve.u_vector, curve.v_vector) == (
            (0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
        assert (curve.start_angle, curve.sweep_angle) == (start, sweep)
        # Exact unit-circle length is |sweep|. This historical quadrature's
        # <=1-ulp approximation error is a fixture fact, no production tolerance.
        assert abs(Fraction(float(length)) - abs(Fraction(sweep))) <= Fraction(math.ulp(float(length)))
    assert total == float(sum((Fraction(float(value)) for value in lengths), Fraction(0)))
    assert cumulative.tolist() == [0., float(lengths[0]), total]
    data = (live[0], lengths, total, cumulative, live[4])
    index, local = _old_local(data, 1.)
    assert index == 1 and float(local).hex() == _OLD_LOCAL
    # At fraction1 the exact stored-length ratio is exactly1. The captured
    # sum/subtract/divide overshoots by exactlyeps, within the existing bound.
    bound = model_module._chain_local_rounding_bound(2, total, float(lengths[1]))
    assert Fraction(float(local)) - 1 == Fraction(EPS)
    assert Fraction(float(local)) - 1 <= bound < 8 * Fraction(EPS)
    return data


def _old_local(data, fraction):
    """The unchanged index/local arithmetic of ``_chain_eval``."""

    chain, lengths, total, cumulative, _edges = data
    target = min(max(float(fraction), 0.0), 1.0) * total
    index = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(chain) - 1)
    return index, (target - cumulative[index]) / lengths[index]


@pytest.mark.parametrize("reverse_second", (False, True))
def test_two_arc_side_end_overshoot_is_snapped_not_refused(reverse_second):
    geometry, chain = _two_arc_side(reverse_second=reverse_second)
    data = _captured_chain_data(geometry, chain)
    index, local = _old_local(data, 1.0)
    # Precondition: the arithmetic really overshoots by rounding only (else this test proves nothing).
    assert index == 1 and 1.0 < local <= 1.0 + _rounding_bound(data, index)
    point, derivative = geometry._chain_eval(data, 1.0)
    curve = geometry.edges[chain[-1].edge].curve
    endpoint = 1.0 if chain[-1].forward else 0.0
    assert np.array_equal(point, curve.evaluate(np.asarray([endpoint]))[0])
    expected = np.asarray(curve.derivative(endpoint), dtype=float) * (data[2] / data[1][1])
    if not chain[-1].forward:
        expected = -expected
    assert np.array_equal(derivative, expected)
    point_only, _zero = geometry._chain_eval(data, 1.0, _point_only=True)
    assert np.array_equal(point_only, point)
    # Today's genuine producer is checked separately; its rounding bits need
    # not equal a capture made with a different CPU quadrature implementation.
    live = geometry._chain_data(chain)
    assert live[1].tolist() == [geometry.edge_length(item.edge) for item in chain]
    assert np.isfinite(live[1]).all() and (live[1] > 0).all()
    live_point, live_derivative = geometry._chain_point_and_derivative(chain, 1.0)
    assert np.array_equal(live_point, curve.evaluate(np.asarray([endpoint]))[0])
    live_expected = np.asarray(curve.derivative(endpoint), dtype=float) * (live[2] / live[1][1])
    if not chain[-1].forward:
        live_expected = -live_expected
    assert np.array_equal(live_derivative, live_expected)


def test_topology_face_with_two_arc_side_evaluates():
    """Face level: a Coons topology face whose first side is the two-arc chain (corner at fraction 1)."""

    geometry, chain = _two_arc_side()
    p0 = geometry.edges[chain[0].edge].start
    p2 = geometry.edges[chain[1].edge].end
    first = geometry.edges[chain[0].edge].curve
    second = geometry.edges[chain[1].edge].curve
    top_points = [np.asarray(q, dtype=float) + (0., 0., 1.) for q in
                  (first.evaluate(np.asarray((0.,)))[0], second.evaluate(np.asarray((1.,)))[0])]
    q0, q2 = geometry.add_points(top_points)
    right = geometry.add_line(p2, q2)
    top = geometry.add_curve(q2, q0, EllipticArc((0., 0., 1.), (1., 0., 0.), (0., 1., 0.), 1.6, -1.6))
    left = geometry.add_line(q0, p0)
    face = geometry.add_face_from_loop(
        (chain[0], chain[1], OrientedEdge(right, True), OrientedEdge(top, True), OrientedEdge(left, True)),
        corners=(0, 2, 3, 4),
    )
    point = geometry.face_point(face, 1.0, 0.0)
    assert np.allclose(point, geometry.vertex_position(p2), atol=1e-14)
    middle = geometry.face_point(face, 0.5, 0.5)
    assert np.isfinite(middle).all()


@pytest.mark.parametrize("excursion", (1e-9, -1e-9, "4bound", "-4bound", "8eps", "-8eps"))
def test_larger_excursion_still_refuses_with_same_error(excursion):
    geometry, chain = _two_arc_side()
    chain_items, lengths, total, cumulative, edges = _captured_chain_data(geometry, chain)
    blanket_case = isinstance(excursion, str) and excursion.endswith('8eps')
    if isinstance(excursion, str):  # four times the derived snap limit of this chain
        sign = -1.0 if excursion.startswith("-") else 1.0
        excursion = sign * (8 * EPS if blanket_case else
                            4.0 * _rounding_bound(_captured_chain_data(geometry, chain), 1))
    # Fabricate chain data whose last-edge local parameter is 1 + excursion (forward edge) or
    # whose reversed-edge parameter 1 - local is -|excursion| (reverse the last item).
    if excursion > 0:
        items = chain_items
        wanted = 1.0 + excursion
    else:
        items = (chain_items[0], OrientedEdge(chain_items[1].edge, False))
        wanted = 1.0 - excursion
    fake_lengths = np.asarray((lengths[0], (total - cumulative[1]) / wanted))
    data = (items, fake_lengths, total, cumulative, edges)
    _index, local = _old_local(data, 1.0)
    parameter = local if items[-1].forward else 1.0 - local
    limit = _rounding_bound(data, 1)
    assert parameter > 1.0 + limit or parameter < -limit
    if blanket_case:
        distance = Fraction(float(parameter)) - 1 if parameter > 1. else -Fraction(float(parameter))
        assert distance <= 8 * Fraction(EPS)
    with pytest.raises(GeometryError, match=r"curve parameters must be finite and in \[0, 1\]"):
        geometry._chain_eval(data, 1.0)
    with pytest.raises(GeometryError, match=r"curve parameters must be finite and in \[0, 1\]"):
        geometry._chain_eval(data, 1.0, _point_only=True)


def test_in_range_and_extrapolating_evaluations_are_bit_identical():
    geometry, chain = _two_arc_side()
    data = _captured_chain_data(geometry, chain)
    _chain, lengths, total, _cumulative, edges = data
    for fraction in np.linspace(0.0, 1.0, 257)[:-1]:
        index, local = _old_local(data, fraction)
        assert 0.0 <= local <= 1.0
        point, derivative = geometry._chain_eval(data, float(fraction))
        curve = edges[index].curve
        assert np.array_equal(point, curve.evaluate(np.asarray([local]))[0])
        assert np.array_equal(derivative, np.asarray(curve.derivative(local), dtype=float) * (total / lengths[index]))

    # A straight edge extrapolates out-of-range parameters without refusing; that path is unchanged.
    straight = GeometryModel()
    a, b, c = straight.add_points(((0., 0., 0.), (1., 0., 0.), (1., 2., 0.)))
    lines = (straight.add_line(a, b), straight.add_line(b, c))
    items = (OrientedEdge(lines[0], True), OrientedEdge(lines[1], True))
    items_, lens, tot, cum, eds = straight._chain_data(items)
    wanted = 1.0 + 2 * EPS
    fake = (items_, np.asarray((lens[0], (tot - cum[1]) / wanted)), tot, cum, eds)
    _index, local = _old_local(fake, 1.0)
    assert local > 1.0
    point, _derivative = straight._chain_eval(fake, 1.0)
    start = straight.vertex_position(b)
    end = straight.vertex_position(c)
    assert np.array_equal(point, model_module.sample_straight(start, end, np.asarray([local]))[0])
