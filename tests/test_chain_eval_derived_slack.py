"""The ``_chain_eval`` rounding snap uses a derived bound, not a blanket slack (S0e).

S0d snapped an excursion of the local edge parameter past [0, 1] only up to 8 eps. On the split cap
rails of model A (faces 23/63/89 of the 121-face cap subset after apply) the last sub-arc is short
(0.005-0.027 m of 0.68 m) and fraction 1 overshoots by 9-12 eps, so ``face_point`` raised
``GeometryError``. The snap limit is now ``_chain_local_rounding_bound(n, T, L)``, a proven bound on
the rounding of the index/local arithmetic; a larger excursion keeps the original refusal.
"""

from fractions import Fraction

import numpy as np
import pytest

from anygeometry.errors import GeometryError
from anygeometry.exact_curves import EllipticArc
import anygeometry.model as model_module
from anygeometry.model import GeometryModel, OrientedEdge

EPS = float(np.finfo(float).eps)
_REFUSAL = r"curve parameters must be finite and in \[0, 1\]"

# Captured from redesign/runs/ADM2-seamrot-sub1081-194247/owner_post.pickle (model A, 121-face cap
# subset of 1081 after Option S and apply). One rail side per face: a chain of forward EllipticArc edges
# on the circle centre (50, 0, -17.5), radius 6.5, axis z. Row: (start angle, sweep, stored edge length,
# end vertex); the first entry is the start vertex. All floats are exact (hex).
_CENTER = (50.0, 0.0, -17.5)
_U = (6.5, 0.0, 0.0)
_V = (0.0, 6.5, 0.0)
_CAPTURED_RAILS = {
    23: (  # n=4, last 0.005162 m of 0.680678 m, old overshoot 12 eps
        ['0x1.a5267a6ad4f5dp+5', '0x1.7c08efe2f97a9p+2', '-0x1.1800000000000p+4'],
        ('0x1.26e40d7653222p+0', '0x1.33b2264434500p-8', '0x1.f4017e2ed5020p-6', ['0x1.a4ed524bba9ebp+5', '0x1.7cd33c1589d28p+2', '-0x1.1800000000000p+4']),
        ('0x1.2817bf9c97567p+0', '0x1.20941d7191b10p-4', '0x1.d4f0afd88cbfap-2', ['0x1.a18629ba908acp+5', '0x1.87aa6ca8faa98p+2', '-0x1.1800000000000p+4']),
        ('0x1.3a210173b0718p+0', '0x1.d77a483146580p-6', '0x1.7f135aa809278p-3', ['0x1.a01bb02a9047dp+5', '0x1.8b898bb38fe80p+2', '-0x1.1800000000000p+4']),
        ('0x1.417eea94758aep+0', '0x1.a066aacc18a62p-11', '0x1.52536ac5d4070p-8', ['0x1.a011a25cd6ed9p+5', '0x1.8ba3b6fa7850fp+2', '-0x1.1800000000000p+4']),
    ),
    63: (  # n=7, last 0.018713 m of 0.680678 m, old overshoot 10 eps
        ['0x1.5c48ecadec576p+5', '-0x1.5bdee7e2106efp-1', '-0x1.1800000000000p+4'],
        ('0x1.9f872a3e00c76p+1', '0x1.f4eff55e84c00p-9', '0x1.9702f75ccbdc0p-6', ['0x1.5c4e56d883179p+5', '-0x1.688480c481a00p-1', '-0x1.1800000000000p+4']),
        ('0x1.a004663b58689p+1', '0x1.503e3296b4d80p-6', '0x1.1132891a72ef8p-3', ['0x1.5c6eb81235bb9p+5', '-0x1.ac559647ce303p-1', '-0x1.1800000000000p+4']),
        ('0x1.a2a4e2a085d24p+1', '0x1.5985cce76b280p-6', '0x1.18bcb67c07108p-3', ['0x1.5c95c8d385ce8p+5', '-0x1.f1d59fc7e5718p-1', '-0x1.1800000000000p+4']),
        ('0x1.a557ee3a54a89p+1', '0x1.cfaaf2deb6600p-6', '0x1.78bae554f42e0p-3', ['0x1.5cd366fc8c410p+5', '-0x1.275f6c165e759p+0', '-0x1.1800000000000p+4']),
        ('0x1.a8f7442012155p+1', '0x1.358bc54365c00p-8', '0x1.f703208d8557fp-6', ['0x1.5cdeb59e69ef5p+5', '-0x1.2f1aaaf109d68p+0', '-0x1.1800000000000p+4']),
        ('0x1.a9920a02b3c83p+1', '0x1.7f1fc80926c00p-6', '0x1.3749d2876f7c0p-3', ['0x1.5d1afc3799cc4p+5', '-0x1.5547234c5792bp+0', '-0x1.1800000000000p+4']),
        ('0x1.ac904992c615bp+1', '0x1.795693e29e68fp-9', '0x1.3296582820b54p-6', ['0x1.5d22e62df9bf3p+5', '-0x1.59f70dbf5e840p+0', '-0x1.1800000000000p+4']),
    ),
    89: (  # n=4, last 0.026845 m of 0.680678 m, old overshoot 9 eps
        ['0x1.8a90846077be3p+5', '-0x1.9db89a909d451p+2', '-0x1.1800000000000p+4'],
        ('0x1.26e40d7653222p+2', '0x1.dd5cad26ed6ffp-5', '0x1.83db4cafa0eafp-2', ['0x1.8d95e7fcd188ep+5', '-0x1.9f8d2474b784ep+2', '-0x1.1800000000000p+4']),
        ('0x1.2a9ec6d0a0fd0p+2', '0x1.d4f62fc72fd00p-6', '0x1.7d0806d1d6d90p-3', ['0x1.8f12ba5232b7dp+5', '-0x1.9fef1508a4f4fp+2', '-0x1.1800000000000p+4']),
        ('0x1.2c73bd00682cdp+2', '0x1.c0c060d87ac00p-7', '0x1.6c9c4eafe3bc0p-4', ['0x1.8fc9054c522c8p+5', '-0x1.9fff177b56582p+2', '-0x1.1800000000000p+4']),
        ('0x1.2d541d30d46a3p+2', '0x1.0eab0976cb959p-8', '0x1.b7d5ef610ad30p-6', ['0x1.9000000000000p+5', '-0x1.a000000000000p+2', '-0x1.1800000000000p+4']),
    ),
}
_CAPTURED_OVERSHOOT_EPS = {23: 12, 63: 10, 89: 9}


def _hex3(values):
    return tuple(float.fromhex(x) for x in values)


def _old_local(data, fraction):
    """The unchanged index/local arithmetic of ``_chain_eval`` (72dac6d), written out."""

    chain, lengths, total, cumulative, _edges = data
    target = min(max(float(fraction), 0.0), 1.0) * total
    index = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(chain) - 1)
    return index, (target - cumulative[index]) / lengths[index]


def _captured_rail(face):
    rows = _CAPTURED_RAILS[face]
    geometry = GeometryModel()
    vertices = geometry.add_points([_hex3(rows[0])] + [_hex3(row[3]) for row in rows[1:]])
    chain, stored = [], []
    for k, (start, sweep, length, _end) in enumerate(rows[1:]):
        arc = EllipticArc(_CENTER, _U, _V, float.fromhex(start), float.fromhex(sweep))
        edge = geometry.add_curve(vertices[k], vertices[k + 1], arc)
        chain.append(OrientedEdge(edge, True))
        stored.append(float.fromhex(length))
    return geometry, tuple(chain), stored


def _assert_end_snapped(geometry, chain, data):
    point, derivative = geometry._chain_eval(data, 1.0)
    point_only, _zero = geometry._chain_eval(data, 1.0, _point_only=True)
    curve = geometry.edges[chain[-1].edge].curve
    endpoint = 1.0 if chain[-1].forward else 0.0
    expected_point = curve.evaluate(np.asarray([endpoint]))[0]
    assert np.array_equal(point, expected_point)
    assert np.array_equal(point_only, expected_point)
    expected = np.asarray(curve.derivative(endpoint), dtype=float) * (data[2] / data[1][-1])
    if not chain[-1].forward:
        expected = -expected
    assert np.array_equal(derivative, expected)


@pytest.mark.parametrize("face", sorted(_CAPTURED_RAILS))
def test_captured_cap_rail_end_overshoot_is_snapped(face):
    geometry, chain, stored = _captured_rail(face)
    # Preconditions: the rebuilt rail has the captured lengths bit for bit, and the unchanged
    # arithmetic overshoots by the captured amount, beyond S0d's 8 eps (else this proves nothing).
    assert [geometry.edge_length(item.edge) for item in chain] == stored
    data = geometry._chain_data(chain)
    index, local = _old_local(data, 1.0)
    assert index == len(chain) - 1
    assert (local - 1.0) / EPS == _CAPTURED_OVERSHOOT_EPS[face] > 8
    _assert_end_snapped(geometry, chain, data)


@pytest.mark.parametrize("face", sorted(_CAPTURED_RAILS))
def test_captured_cap_rail_strip_face_evaluates(face):
    """Face level: a Coons topology strip whose first side is the captured split rail (the S0e failure
    was ``face_point`` computing the corner at fraction 1 of that side)."""

    geometry, chain, _stored = _captured_rail(face)
    rows = _CAPTURED_RAILS[face][1:]
    start = float.fromhex(rows[0][0])
    sweep = float.fromhex(rows[-1][0]) + float.fromhex(rows[-1][1]) - start
    lifted = (_CENTER[0], _CENTER[1], _CENTER[2] + 0.105)
    top = EllipticArc(lifted, _U, _V, start + sweep, -sweep)
    q_last, q_first = geometry.add_points(top.evaluate(np.asarray((0.0, 1.0))))
    p_first = geometry.edges[chain[0].edge].start
    p_last = geometry.edges[chain[-1].edge].end
    right = geometry.add_line(p_last, q_last)
    top_edge = geometry.add_curve(q_last, q_first, top)
    left = geometry.add_line(q_first, p_first)
    n = len(chain)
    face_id = geometry.add_face_from_loop(
        chain + (OrientedEdge(right, True), OrientedEdge(top_edge, True), OrientedEdge(left, True)),
        corners=(0, n, n + 1, n + 2),
    )
    corner = geometry.face_point(face_id, 1.0, 0.0)
    # The Coons blend of the four sides reproduces the corner up to its own rounding (coordinates ~50 m).
    assert np.allclose(corner, geometry.vertex_position(p_last), rtol=0.0, atol=4e-14)
    grid = [geometry.face_point(face_id, float(u), float(v)) for u in np.linspace(0, 1, 5) for v in np.linspace(0, 1, 5)]
    assert np.isfinite(grid).all()


def _split_rail(cuts, *, reverse_last=False):
    """A rail on the captured circle, angle 1.15..1.255 split at ``cuts`` (fractions of the sweep)."""

    geometry = GeometryModel()
    start, sweep = 1.15, 0.105
    angles = [start] + [start + sweep * float(c) for c in cuts] + [start + sweep]
    circle = EllipticArc(_CENTER, _U, _V, start, sweep)
    vertices = geometry.add_points(circle.evaluate(np.asarray([(a - start) / sweep for a in angles])))
    chain = []
    for k in range(len(angles) - 1):
        a, b = angles[k], angles[k + 1]
        if reverse_last and k == len(angles) - 2:
            edge = geometry.add_curve(vertices[k + 1], vertices[k], EllipticArc(_CENTER, _U, _V, b, a - b))
            chain.append(OrientedEdge(edge, False))
        else:
            edge = geometry.add_curve(vertices[k], vertices[k + 1], EllipticArc(_CENTER, _U, _V, a, b - a))
            chain.append(OrientedEdge(edge, True))
    return geometry, tuple(chain)


def _synthetic_overshooting_rails(reverse_last):
    """Deterministic scan: rails split at 1-7 interior angles with a last sub-arc under 4 % of the sweep
    whose fraction-1 overshoot exceeds S0d's 8 eps."""

    rng = np.random.default_rng(20261029)
    found = []
    for _ in range(4000):
        interior = int(rng.integers(1, 8))
        last = float(rng.uniform(0.005, 0.04))
        cuts = np.sort(rng.uniform(0.0, 1.0 - last, interior - 1)).tolist() + [1.0 - last]
        if min(np.diff([0.0] + cuts + [1.0])) < 1e-3:
            continue
        geometry, chain = _split_rail(cuts, reverse_last=reverse_last)
        data = geometry._chain_data(chain)
        _index, local = _old_local(data, 1.0)
        if local - 1.0 > 8 * EPS:
            found.append((geometry, chain, data))
            if len(found) == 3:
                break
    assert found, "no synthetic rail overshoots beyond 8 eps; the regression precondition is lost"
    return found


@pytest.mark.parametrize("reverse_last", (False, True))
def test_synthetic_split_rail_end_overshoot_is_snapped(reverse_last):
    for geometry, chain, data in _synthetic_overshooting_rails(reverse_last):
        _assert_end_snapped(geometry, chain, data)


def _bound(count, total, length):
    return model_module._chain_local_rounding_bound(count, total, length)


@pytest.mark.parametrize("face", sorted(_CAPTURED_RAILS))
def test_captured_overshoot_lies_within_the_derived_bound(face):
    geometry, chain, _stored = _captured_rail(face)
    chain_items, lengths, total, cumulative, _edges = geometry._chain_data(chain)
    bound = _bound(len(chain), total, float(lengths[-1]))
    overshoot = Fraction(_CAPTURED_OVERSHOOT_EPS[face]) * Fraction(EPS)
    assert overshoot <= bound
    # First-order form stated in the docstring: eps * ((n - 1) * T/L + 1).
    first_order = EPS * ((len(chain) - 1) * total / float(lengths[-1]) + 1.0)
    assert abs(float(bound) / first_order - 1.0) < 1e-9


@pytest.mark.parametrize("multiple", (4.0, -4.0, 1e-9, -1e-9))
def test_excursion_beyond_bound_still_refuses_with_same_error(multiple):
    """A real out-of-range parameter (1 + 4 bound, 1 + 1e-9, and the mirrored reversed-edge cases)
    keeps the original refusal, with and without ``_point_only``."""

    geometry, chain, _stored = _captured_rail(23)
    chain_items, lengths, total, cumulative, edges = geometry._chain_data(chain)
    n = len(chain)
    span = total - cumulative[n - 1]
    base = float(_bound(n, total, float(lengths[-1])))
    excursion = abs(multiple) * base if abs(multiple) >= 1.0 else abs(multiple)
    items = chain_items if multiple > 0 else chain_items[:-1] + (OrientedEdge(chain_items[-1].edge, False),)
    fake = lengths.copy()
    fake[-1] = span / (1.0 + excursion)
    data = (items, fake, total, cumulative, edges)
    _index, local = _old_local(data, 1.0)
    parameter = local if items[-1].forward else 1.0 - local
    bound = _bound(n, total, float(fake[-1]))
    distance = Fraction(parameter) - 1 if parameter > 1.0 else -Fraction(parameter)
    assert distance > bound and distance >= Fraction(excursion) / 2
    with pytest.raises(GeometryError, match=_REFUSAL):
        geometry._chain_eval(data, 1.0)
    with pytest.raises(GeometryError, match=_REFUSAL):
        geometry._chain_eval(data, 1.0, _point_only=True)


def _random_lengths(rng, n):
    kind = rng.integers(0, 3)
    if kind == 0:  # comparable lengths
        lengths = rng.uniform(0.2, 1.0, n)
    elif kind == 1:  # magnitudes over six decades
        lengths = 10.0 ** rng.uniform(-3.0, 3.0, n)
    else:  # captured shape: comparable pieces and a short last piece
        lengths = rng.uniform(0.05, 0.5, n)
        lengths[-1] = lengths[:-1].sum() * float(rng.uniform(1e-4, 0.05)) if n > 1 else lengths[-1]
    return lengths * float(10.0 ** rng.uniform(-2.0, 2.0))


def _exact_check(lengths, fractions):
    """Max of (overshoot past 1, |error vs exact rational value|) / bound over ``fractions``."""

    n = len(lengths)
    total = float(lengths.sum())
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    exact_prefix = [Fraction(0)]
    for value in lengths:
        exact_prefix.append(exact_prefix[-1] + Fraction(float(value)))
    worst = 0.0
    for fraction in fractions:
        index, local = model_module._chain_local_parameter(lengths, total, cumulative, n, fraction)
        assert (index, local) == _old_local((range(n), lengths, total, cumulative, None), fraction)
        bound = _bound(n, total, float(lengths[index]))
        c = Fraction(min(max(float(fraction), 0.0), 1.0))
        exact = (c * exact_prefix[-1] - exact_prefix[index]) / Fraction(float(lengths[index]))
        computed = Fraction(float(local))
        assert computed >= 0
        error = abs(computed - exact)
        overshoot = max(computed - 1, Fraction(0))
        assert error <= bound, (n, fraction, float(error), float(bound))
        assert overshoot <= bound, (n, fraction, float(overshoot), float(bound))
        worst = max(worst, float(max(error, overshoot) / bound))
    return worst


def test_bound_is_never_exceeded_against_exact_rational_evaluation():
    """Empirical check of the analysis on random chains: n in 1..12 and NumPy pairwise-sum sizes
    (8, 9, 16, 17, 129, 130), fractions at 0, 1, uniformly random and at every computed breakpoint
    and its float neighbours."""

    rng = np.random.default_rng(7)
    counts = list(range(1, 13)) * 60 + [8, 9, 16, 17, 129, 130] * 6
    worst = 0.0
    for n in counts:
        lengths = _random_lengths(rng, n)
        total = float(lengths.sum())
        cumulative = np.cumsum(lengths)
        breaks = [float(x) / total for x in cumulative[-min(n, 4):]]
        fractions = [0.0, 1.0, *rng.uniform(0.0, 1.0, 4).tolist()]
        for b in breaks:
            fractions += [b, float(np.nextafter(b, 0.0)), float(np.nextafter(b, 2.0))]
        worst = max(worst, _exact_check(lengths, fractions))
    assert worst <= 1.0
