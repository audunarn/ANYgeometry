"""Independent CH3 oracle roots: no production root/intersection implementation."""
import mpmath as mp
import pytest

import ch3_reference as reference


# Descending coefficients printed by the failed adaptive-quadrature CI frame.
CI_COEFFICIENTS = (
    '0.99999999999999999999999999999999999999999999999999999999999999370919575772838957012520846829646494',
    '-2.000000000000000000000000000000000000000000000000000000000000008896540677654709528720919349997969',
    '1.0',
)


def coefficient_hull(a, b, c):
    # Bypass Hull's data-only production import; these are synthetic numerators.
    hull = object.__new__(reference.Hull)
    hull.coeffs = lambda z: ([c, b, a], [mp.mpf(0)] * 3, [mp.mpf(1), mp.mpf(0), mp.mpf(0)])
    return hull


def assert_residuals_and_viete(a, b, c, roots):
    tolerance = 128 * mp.eps
    with mp.extraprec(60):
        for root in roots:
            terms = (a * root * root, b * root, c)
            scale = sum(abs(term) for term in terms)
            residual = abs(sum(terms))
            assert residual <= tolerance * scale if scale else residual == 0
        assert abs(sum(roots) + b / a) <= tolerance * max(1, abs(b / a))
        assert abs(roots[0] * roots[1] - c / a) <= tolerance * max(1, abs(c / a))


def test_actual_ci_nearly_double_coefficients_do_not_need_iterative_convergence():
    with mp.workdps(100):
        a, b, c = map(mp.mpf, CI_COEFFICIENTS)
        hull = coefficient_hull(a, b, c)
        # Both nearly coincident u roots are interior to this synthetic arc.
        parameters = hull.cut_params(mp.mpf(0), ('arc', 1, mp.mpf(0), mp.mpf(2)),
                                     mp.mpf(1), mp.mpf(0), mp.mpf(0))
        assert len(parameters) == 2
        assert parameters[0] < mp.mpf('.5') < parameters[1]
        assert_residuals_and_viete(a, b, c, [2 * value for value in parameters])


@pytest.mark.parametrize('dps', [15, 30, 80, 120])
@pytest.mark.parametrize('case', ['distinct', 'double', 'imaginary', 'mixed-sign', 'zero', 'complex-coefficients'])
def test_closed_form_preserves_known_roots_residuals_and_viete(dps, case):
    with mp.workdps(dps):
        expected = {
            'distinct': (mp.mpf(2), mp.mpf(3)),
            'double': (mp.mpf(1), mp.mpf(1)),
            'imaginary': (mp.j, -mp.j),
            'mixed-sign': (mp.mpf(-3), mp.mpf(2)),
            'zero': (mp.mpf(0), mp.mpf(4)),
            'complex-coefficients': (mp.mpc(1, 2), mp.mpc(-3, 4)),
        }[case]
        a = mp.mpc(2, -1) if case == 'complex-coefficients' else mp.mpf(1)
        b, c = -a * sum(expected), a * expected[0] * expected[1]
        roots = reference._quadratic_roots(a, b, c)
        assert len(roots) == 2
        tolerance = 128 * mp.eps
        assert all(any(abs(root - wanted) <= tolerance * max(1, abs(wanted)) for root in roots)
                   for wanted in expected)
        assert_residuals_and_viete(a, b, c, roots)


@pytest.mark.parametrize('dps', [30, 80])
@pytest.mark.parametrize('exponent', [-20000, 20000])
def test_common_extreme_scale_does_not_change_roots(dps, exponent):
    with mp.workdps(dps):
        scale = mp.ldexp(mp.mpf(1), exponent)
        roots = reference._quadratic_roots(scale, -3 * scale, 2 * scale)
        assert sorted(roots) == [1, 2]
        assert_residuals_and_viete(scale, -3 * scale, 2 * scale, roots)


@pytest.mark.parametrize('dps', [30, 80])
@pytest.mark.parametrize('exponent', [200, 20000])
def test_widely_separated_roots_keep_the_small_nonzero_root(dps, exponent):
    with mp.workdps(dps):
        large = mp.ldexp(mp.mpf(1), exponent)
        roots = reference._quadratic_roots(mp.mpf(1), -large, mp.mpf(1))
        small = min(roots, key=abs)
        assert small != 0
        assert abs(small * large - 1) <= 128 * mp.eps
        assert_residuals_and_viete(mp.mpf(1), -large, mp.mpf(1), roots)


@pytest.mark.parametrize('coefficients,expected', [
    ((0, 2, -6), [3]), ((0, 0, 5), []), ((1, 0, 0), [0, 0]),
])
def test_linear_constant_and_zero_cases_have_only_isolated_roots(coefficients, expected):
    assert reference._quadratic_roots(*coefficients) == expected


def test_zero_polynomial_is_not_a_finite_complete_root_set():
    with pytest.raises(ValueError, match='zero polynomial'):
        reference._quadratic_roots(0, 0, 0)


@pytest.mark.parametrize('dps', [15, 30, 100])
def test_working_precision_uses_old_sixty_guard_bits_and_is_restored(monkeypatch, dps):
    with mp.workdps(dps):
        precision = mp.mp.prec
        original = mp.sqrt
        observed = []
        def sqrt(value):
            observed.append(mp.mp.prec)
            return original(value)
        monkeypatch.setattr(mp, 'sqrt', sqrt)
        reference._quadratic_roots(1, 0, 1)
        assert observed == [precision + 60]
        assert mp.mp.prec == precision
    assert mp.mp.dps == 30


def test_precision_is_restored_after_conversion_and_arithmetic_errors(monkeypatch):
    precision = mp.mp.prec
    with pytest.raises(TypeError, match='cannot create mpf'):
        reference._quadratic_roots('not-a-number', 1, 1)
    assert mp.mp.prec == precision
    failure = ArithmeticError('oracle square root failed')
    def fail(value):
        raise failure
    monkeypatch.setattr(mp, 'sqrt', fail)
    with pytest.raises(ArithmeticError) as caught:
        reference._quadratic_roots(1, 0, 1)
    assert caught.value is failure and mp.mp.prec == precision


def test_tiny_negative_discriminant_is_not_clamped_to_a_real_double_root():
    with mp.workdps(100):
        a, b, c = mp.mpf(1), mp.mpf(-2), 1 + mp.mpf('1e-60')
        roots = reference._quadratic_roots(a, b, c)
        assert all(mp.im(root) != 0 for root in roots)
        assert mp.im(roots[0]) * mp.im(roots[1]) < 0
        assert_residuals_and_viete(a, b, c, roots)


def test_u_at_x_keeps_real_and_domain_filters_without_polyroots(monkeypatch):
    monkeypatch.setattr(mp, 'polyroots', lambda *a, **k: pytest.fail('iterative root solver used'))
    assert coefficient_hull(1, 0, 0).u_at_x(0, mp.mpf('.25')) == mp.mpf('.5')
    assert coefficient_hull(0, 1, 0).u_at_x(0, mp.mpf('.25')) == mp.mpf('.25')


@pytest.mark.parametrize('coefficients,expected', [
    ((1, -1, 0), []), ((1, -1, mp.mpf('.1875')), [mp.mpf('.25'), mp.mpf('.75')]),
    ((1, 0, 1), []), ((0, 1, mp.mpf('-.5')), [mp.mpf('.5')]), ((0, 0, 0), []),
])
def test_cut_params_keeps_existing_open_domain_and_nonreal_filters(coefficients, expected):
    hull = coefficient_hull(*coefficients)
    assert hull.cut_params(0, ('arc', 1, mp.mpf(0), mp.mpf(1)), 1, 0, 0) == expected


def test_bounded_slice_integrand_and_chords_exercise_both_callers(monkeypatch):
    monkeypatch.setattr(mp, 'polyroots', lambda *a, **k: pytest.fail('iterative root solver used'))
    hull = object.__new__(reference.Hull)
    hull.half = True
    hull.x1, hull.x2 = mp.mpf(0), mp.mpf(1)
    # Independent polynomial arc x=u, y=1-u**2, not production geometry.
    hull.coeffs = lambda z: ([mp.mpf(0), mp.mpf(1), mp.mpf(0)],
                             [mp.mpf(1), mp.mpf(0), mp.mpf(-1)],
                             [mp.mpf(1), mp.mpf(0), mp.mpf(0)])
    root = mp.sqrt(mp.mpf('.5'))
    expected = ((2 - root) / 3, mp.mpf(3) / 16, mp.mpf(4) / 15 - 7 * root / 30)
    actual = hull.slice_stats(mp.mpf('.25'), mp.mpf(0), mp.mpf(1), mp.mpf('.5'))
    # Gauss nodes remain binary64 exactly as in the existing reference contract.
    assert all(abs(got - wanted) < mp.mpf('1e-13') for got, wanted in zip(actual, expected))
    chords = hull.chords(mp.mpf('.25'), mp.mpf(0), mp.mpf(1), mp.mpf('.5'))
    assert len(chords) == 1
    assert abs(chords[0][0] + root) <= 128 * mp.eps and chords[0][1] == 0
