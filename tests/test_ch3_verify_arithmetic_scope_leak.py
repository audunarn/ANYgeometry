"""Regression: a test that builds ``_ch3_verify_math.Arithmetic`` directly must not leave the verifier scope
active for later tests (it made the planning-threads route decline in 14 later tests, 2026-11-12)."""
from fractions import Fraction as F

from anygeometry import _planning_threads as pt
from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic, Rational, _active
from anygeometry.certified_hydrostatics import CertifiedHydrostaticsPolicy, WorkBudget
from anygeometry.certified_hydrostatics._work import Meter


def test_direct_arithmetic_use_like_test_left_taylor_rational_oracle():
    # Same body as tests/test_certified_hydrostatics_ch3_verify.py::test_left_taylor_rational_oracle.
    ar = Arithmetic(Meter(WorkBudget.unlimited()), CertifiedHydrostaticsPolicy(convention_version='2').as_dict())
    value = ar.integrate(Rational((1,), (1, 2, 1)), F(0), F(1), F(1, 10**12))
    assert value.lo <= F(1, 2) <= value.hi


def test_no_verify_arithmetic_scope_survives_the_previous_test():
    # FAILS on 2e1967c: _active still holds the previous test's (meter, max_bits).
    assert _active.get() is None
    assert not pt._scope_active()
