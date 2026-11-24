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


def _weight_dots(path):
    """Source sites where a name containing 'weight' is an operand of ``@``, ``np.dot`` or ``.dot``."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    def names(node):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
               {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            if any('weight' in n for n in names(node.left) | names(node.right)):
                hits.append(node.lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('dot', 'inner', 'vdot'):
            if any('weight' in n for arg in node.args for n in names(arg)) or \
                    any('weight' in n for n in names(node.func.value)):
                hits.append(node.lineno)
    return hits


# Vector-matrix products are repeatable on Accelerate (measured on macos-15-intel for inner lengths 2..1000,
# run 45364202290): only 1-D x 1-D dots, 1-D norms and (rows x long inner) matrix-vector products are not.
ALLOWED = {'face_moments.py': 1}        # (.5*(hi-lo)*weights) @ g, g of shape (order, k): vector-matrix


def test_no_quadrature_weight_reduction_goes_through_a_blas_dot():
    offenders = {}
    for path in sorted(SRC.rglob('*.py')):
        hits = _weight_dots(path)
        name = path.relative_to(SRC).as_posix()
        if hits and len(hits) != ALLOWED.get(name, 0):
            offenders[name] = hits
    assert offenders == {}
