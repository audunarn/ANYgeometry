"""Exact conic/line exclusion; contacts and restricted budgets still refuse."""
from fractions import Fraction as Q

import pytest

from anygeometry import branch_predicates as predicates
from anygeometry.plane_nurbs import PlaneNurbsNotCertified
from anygeometry.support_curves import PlaneNurbsCurve
from test_plane_split import bowl


def circle():
    # Tensor Bernstein coefficients of u**2 + v**2 - 1/2 on the unit square.
    squares = (Q(0), Q(0), Q(1))
    return tuple(tuple(a + b - Q(1, 2) for b in squares) for a in squares)


def line(a, b, c):
    # Tensor degree (1,1) coefficients of a*u+b*v+c.
    return ((c, c + b), (c + a, c + a + b))


@pytest.mark.parametrize('tiny', (Q(1, 2**54), Q(1, 2**200)))
def test_exact_positive_separation_never_rounds_to_tangency(tiny):
    # min(u**2+v**2 | u+(1-tiny)*v=1)=1/(1+(1-tiny)**2)>1/2.
    assert 1 / (1 + (1-tiny)**2) > Q(1, 2)
    assert predicates._quadratic_affine_disjoint(circle(), line(Q(1), 1-tiny, Q(-1)))


@pytest.mark.parametrize('tiny', (Q(0), -Q(1, 2**54)))
def test_exact_contact_and_crossing_cannot_receive_no_root_certificate(tiny):
    assert 1 / (1 + (1-tiny)**2) <= Q(1, 2)
    assert not predicates._quadratic_affine_disjoint(circle(), line(Q(1), 1-tiny, Q(-1)))


def test_both_elimination_axes_keep_the_same_no_root_proof():
    assert predicates._quadratic_affine_disjoint(circle(), line(Q(1), Q(0), Q(-2)))
    assert predicates._quadratic_affine_disjoint(circle(), line(Q(0), Q(1), Q(-2)))
    assert not predicates._quadratic_affine_disjoint(circle(), line(Q(1), Q(0), Q(0)))


def test_unsupported_degree_and_float_coefficients_cannot_certify():
    assert not predicates._quadratic_affine_disjoint(
        ((Q(0),), (Q(0),), (Q(0),), (Q(1),)), line(Q(1), Q(1), Q(-1)))
    assert not predicates._quadratic_affine_disjoint(
        ((0.0, 0.0), (0.0, 1.0)), line(Q(1), Q(1), Q(-1)))
    assert not predicates._quadratic_affine_disjoint(circle(), circle())


def test_lower_explicit_depth_limit_retains_the_original_refusal():
    model, face, plane = bowl()
    curve = PlaneNurbsCurve.from_plane(plane, model.faces[face].surface, 0)
    with pytest.raises(PlaneNurbsNotCertified):
        predicates.section_roots(curve, (Q(1), 1-Q(1, 2**54), Q(0)), Q(1), max_depth=1)


def test_constant_second_equation_uses_the_unchanged_interval_proof():
    # The certificate is conservative, not a complete polynomial solver.
    # A nonzero constant G2 already exits through the original sign-range test.
    assert not predicates._quadratic_affine_disjoint(circle(), ((Q(1),),))
    model, face, plane = bowl()
    curve = PlaneNurbsCurve.from_plane(plane, model.faces[face].surface, 0)
    assert predicates.section_roots(curve, (Q(0), Q(0), Q(1)), Q(2)) == ()
