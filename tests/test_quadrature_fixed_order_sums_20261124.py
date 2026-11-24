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
