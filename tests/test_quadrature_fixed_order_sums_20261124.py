"""Quadrature sums are fixed-order, correctly rounded sums (0.4.7 gate, class A, 2026-11-24).

On Intel macOS numpy links Apple Accelerate, whose 1-D dot product (``weights @ samples``) is not
repeatable: the same values at different memory offsets give results that differ in the last bits
(measured on the hosted macos-15-intel runner for every length 9..1000; Linux OpenBLAS repeatable).
Loop areas, and through them plan checksums, then varied between two calls in one process.

Contract: every quadrature reduction is ``math.fsum`` of the elementwise products, so its value is
defined by IEEE products and one correctly rounded sum on every platform and memory layout.
"""
import ast
import math
import pathlib
from types import SimpleNamespace

import numpy as np

from anygeometry import material_arrangement as ma

SRC = pathlib.Path(ma.__file__).resolve().parent


def _samples(curve, parameters):
    t = np.asarray(parameters, dtype=float)
    return curve.scale*(np.sin(37.*t + curve.phase)*1e3 + 1./(1.+t) + t**7)


def _fixed(weights, samples):
    return math.fsum((np.asarray(weights)*np.asarray(samples)).tolist())


def test_loop_area_rule_is_the_correctly_rounded_sum_of_the_weighted_samples():
    loop = [SimpleNamespace(curve=SimpleNamespace(scale=s, phase=p)) for s, p in ((1., .1), (-.7, 1.3), (2.3, -.4))]
    boundary = lambda curve: curve.scale*.125
    tolerance = 1e300                                   # stop at the second rule: both rules are compared
    expected = None
    previous = None
    for count in (16, 32):
        nodes, weights = ma._gauss_legendre(count)
        value = 0.
        for path in loop:
            value += boundary(path.curve) + .5*_fixed(weights, _samples(path.curve, .5*(nodes+1)))
        if previous is not None:
            expected = value
            break
        previous = value
    got = ma._integrate_loop(loop, boundary, _samples, tolerance)
    assert float(got).hex() == float(expected).hex()


def _is_weight(name):
    return 'weight' in name or name == 'ws'


def _weight_dots(path):
    """Source sites where a weight array (a name containing 'weight', or ``ws``) is an operand of ``@``,
    ``np.matmul``, ``np.dot``, ``.dot``, ``np.inner`` or ``np.vdot``."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    def names(node):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} |                {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            if any(_is_weight(n) for n in names(node.left) | names(node.right)):
                hits.append(node.lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and                 node.func.attr in ('dot', 'inner', 'vdot', 'matmul'):
            operands = [*node.args, *(k.value for k in node.keywords)]
            if node.func.attr == 'dot' and not (isinstance(node.func.value, ast.Name) and node.func.value.id == 'np'):
                operands.append(node.func.value)
            if any(_is_weight(n) for arg in operands for n in names(arg)):
                hits.append(node.lineno)
    return hits


def test_no_quadrature_weight_reduction_goes_through_a_blas_dot():
    offenders = {}
    for path in sorted(SRC.rglob('*.py')):
        hits = _weight_dots(path)
        if hits:
            offenders[path.relative_to(SRC).as_posix()] = hits
    assert offenders == {}


# ---- Codex round 1 (2026-11-24) ----

def test_fixed_sums_keep_the_ieee_value_of_non_finite_sums_instead_of_raising():
    from anygeometry._fixed_sums import weighted_columns, weighted_rows, weighted_sum
    inf = math.inf
    assert math.isnan(weighted_sum([1., 1.], [inf, -inf]))
    assert weighted_sum([1., 1.], [1e308, 1e308]) == inf
    assert weighted_sum([1., 1.], [-1e308, -1e308]) == -inf
    assert math.isnan(weighted_sum([1., 2.], [math.nan, 1.]))
    assert weighted_sum([1., 1.], [inf, 1.]) == inf
    rows = weighted_rows(np.array([[inf, -inf], [1e308, 1e308], [1., 2.]]), np.array([1., 1.]))
    assert math.isnan(rows[0]) and rows[1] == inf and rows[2] == 3.
    columns = weighted_columns(np.array([1., 1.]), np.array([[inf, 1.], [-inf, 2.]]))
    assert math.isnan(columns[0]) and columns[1] == 3.


def test_fixed_sums_are_correctly_rounded_and_never_broadcast():
    from anygeometry._fixed_sums import weighted_columns, weighted_rows, weighted_sum
    import pytest
    w, v = np.array([.1, .2, .3, 1e16, -1e16]), np.array([.7, .11, .13, 1., 1.])
    assert weighted_sum(w, v) == math.fsum((w*v).tolist())
    for bad in ((w, v[:4]), (w, np.ones((5, 1))), (w[:, None], v)):
        with pytest.raises(ValueError):
            weighted_sum(*bad)
    with pytest.raises(ValueError):
        weighted_rows(np.ones((2, 4)), w)
    with pytest.raises(ValueError):
        weighted_columns(w, np.ones((4, 2)))


def test_a_single_field_column_has_the_value_it_has_among_other_columns():
    """Face moments of one field column equal that column of a several-column call, bit for bit
    (a (n,) @ (n, 1) product took another BLAS route than (n,) @ (n, k))."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_face_moments import box
    from anygeometry.face_moments import Frame, face_moments, volume_fields
    model, faces, _corner = box()
    for column in range(4):
        def single(local, column=column):
            return volume_fields(local)[:, column:column + 1]
        for face in faces:
            alone = face_moments(model, face, Frame.world(), single)
            together = face_moments(model, face, Frame.world(), volume_fields)
            assert float(alone[0]).hex() == float(together[column]).hex(), (face, column)


def test_a_nan_boundary_area_fails_material_conservation(monkeypatch):
    """``abs(nan - retained) > tolerance`` is False: a NaN original area must refuse, not pass."""
    import sys
    import pytest
    from anygeometry.errors import GeometryError
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_cone_material_arrangement import _facet
    model, _face, domain = _facet()
    original = ma._float_material_cells

    def nan_original(*args, **kwargs):
        cells, retained, outer, _original = original(*args, **kwargs)
        return cells, retained, outer, math.nan
    monkeypatch.setattr(ma, '_float_material_cells', nan_original)
    with pytest.raises(GeometryError):
        ma.arrange_material(domain, [], tolerance=model.tolerance.length,
                            area_tolerance=model.tolerance.effective_area(2.))


# ---- Codex round 2 (2026-11-24) ----

def test_a_finite_sum_with_intermediate_overflow_is_its_correctly_rounded_exact_value():
    """math.fsum raises on intermediate overflow although the exact sum is finite; the value is the
    exact sum rounded once (no nan, no warning), or a signed infinity when the exact sum is beyond range."""
    import warnings
    from fractions import Fraction
    from anygeometry._fixed_sums import weighted_columns, weighted_rows, weighted_sum
    ones = np.ones(5)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert weighted_sum(ones[:4], [1e308, 1e308, -1e308, -1e308]) == 0.
        values = [1e308, 1e308, -1e308, 3.]
        assert weighted_sum(ones[:4], values) == float(sum(map(Fraction, values)))
        assert weighted_sum(ones[:4], [1e308, 1e308, 1e308, -1e308]) == math.inf
        assert weighted_sum(ones[:4], [-1e308, -1e308, -1e308, 1e308]) == -math.inf
        assert weighted_rows(np.array([[1e308, 1e308, -1e308, -1e308]]), ones[:4])[0] == 0.
        assert weighted_columns(ones[:4], np.array([[1e308], [1e308], [-1e308], [-1e308]]))[0] == 0.


def test_no_module_imports_a_blas_product_under_another_name():
    offenders = []
    for path in sorted(SRC.rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'numpy':
                offenders += [(path.name, node.lineno, a.name) for a in node.names
                              if a.name in ('dot', 'matmul', 'inner', 'vdot', 'einsum_path')]
    assert offenders == []


def test_curve_length_refuses_a_derivative_that_is_not_one_vector_per_parameter():
    """A (3,) derivative for a batch of parameters must not broadcast into a length (0502f62 gave ~1.0)."""
    import pytest
    from anygeometry.exact_curves import analytic_curve_length

    class Bad:
        def derivative(self, parameters):
            return np.array([1., 0., 0.])
    with pytest.raises(ValueError):
        analytic_curve_length(Bad(), 1e-9)


# ---- Codex round 3 (2026-11-24): strict helper contract adopted by the owner ----
# Inputs are real float64 arrays only; non-finite products are classified before any summation
# (any nan, or +inf with -inf -> nan; infinities of one sign -> that infinity); only all-finite
# products reach the correctly rounded sum.

def test_non_finite_products_are_classified_before_any_summation():
    from anygeometry._fixed_sums import weighted_columns, weighted_rows, weighted_sum
    inf, nan = math.inf, math.nan
    one = np.ones(3)
    assert math.isnan(weighted_sum(one, [1e308, 1e308, nan]))
    assert weighted_sum(one, [1e308, 1e308, inf]) == inf
    assert weighted_sum(one, [1e308, 1e308, -inf]) == -inf  # finite + -inf; the old dot overflowed to +inf first (nan)
    assert weighted_sum(one, [-1e308, -1e308, -inf]) == -inf
    assert math.isnan(weighted_sum(one, [inf, -inf, 1.]))
    assert math.isnan(weighted_rows(np.array([[1e308, 1e308, nan]]), one)[0])
    assert weighted_columns(one, np.array([[1e308], [1e308], [inf]]))[0] == inf


# Revised contract (owner, 2026-11-24, after the strict-contract review): every real numeric input is
# converted to native float64 before the products are formed; only complex and object input is refused.

REAL_DTYPES = (np.float64, np.float32, np.float16, np.longdouble, np.int64, np.int32, np.bool_,
               np.dtype('>f8'), np.dtype('<f8'))


def test_real_numeric_inputs_are_converted_to_native_float64():
    from anygeometry._fixed_sums import weighted_columns, weighted_rows, weighted_sum
    w = np.array([.5, .25, 2.])
    v = np.array([1., 0., 1.])                     # exact in every dtype, bool included
    expected = math.fsum((w*v).tolist())
    for dtype in REAL_DTYPES:
        values = v.astype(dtype)
        assert weighted_sum(w, values) == expected, dtype
        if np.dtype(dtype).kind == 'f':
            assert weighted_sum(w.astype(dtype), v) == expected, dtype
        assert weighted_rows(values[None, :], w)[0] == expected, dtype
        assert weighted_columns(w, values[:, None])[0] == expected, dtype


def test_the_overflow_path_works_for_every_accepted_dtype():
    from anygeometry._fixed_sums import weighted_columns, weighted_rows, weighted_sum
    one = np.ones(3)
    for dtype in (np.float64, np.longdouble, np.dtype('>f8')):
        values = np.array([1e308, 1e308, -1e308]).astype(dtype)
        assert weighted_sum(one, values) == 1e308, dtype
        assert weighted_rows(values[None, :], one)[0] == 1e308, dtype
        assert weighted_columns(one, values[:, None])[0] == 1e308, dtype


def test_complex_and_object_inputs_are_refused():
    import pytest
    from anygeometry._fixed_sums import weighted_columns, weighted_rows, weighted_sum
    w = np.ones(3)
    for values in (np.ones(3, dtype=complex), np.ones(3, dtype=object)):
        with pytest.raises(TypeError):
            weighted_sum(w, values)
        with pytest.raises(TypeError):
            weighted_sum(values, w)
        with pytest.raises(TypeError):
            weighted_rows(values[None, :], w)
        with pytest.raises(TypeError):
            weighted_columns(w, values[:, None])


def test_a_float32_derivative_still_gives_a_curve_length():
    """A curve whose derivative() returns float32 had a length on d71c5bc; the strict contract refused it."""
    import pytest
    from anygeometry.exact_curves import analytic_curve_length

    class Float32Line:
        def derivative(self, parameters):
            t = np.atleast_1d(np.asarray(parameters, dtype=float))
            return np.tile(np.array([3., 4., 0.], dtype=np.float32), (len(t), 1))
    assert analytic_curve_length(Float32Line(), 1e-9) == pytest.approx(5., rel=1e-12)


def test_a_complex_face_field_is_refused_as_before():
    """d71c5bc refused a complex field (float() of a complex sum); the helper must not cast it to real."""
    import sys
    import pytest
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_face_moments import box
    from anygeometry.face_moments import Frame, face_moments
    model, faces, _corner = box()
    with pytest.raises(TypeError):
        face_moments(model, faces[0], Frame.world(), lambda local: np.ones((len(local), 1)) * (1 + 1j))


QUADRATURE_MODULES = ('material_arrangement.py', 'nurbs_domain.py', 'nurbs_section_faces.py',
                      'section_faces.py', 'exact_curves.py', 'face_moments.py')


def test_quadrature_modules_never_name_a_numpy_product_routine():
    """Covers aliases (``op = np.dot``, ``import numpy as n; n.dot``, ``a.dot``): no attribute of these names."""
    offenders = []
    for name in QUADRATURE_MODULES:
        for node in ast.walk(ast.parse((SRC / name).read_text(encoding='utf-8'))):
            if isinstance(node, ast.Attribute) and node.attr in ('dot', 'matmul', 'inner', 'vdot'):
                offenders.append((name, node.lineno))
    assert offenders == []
