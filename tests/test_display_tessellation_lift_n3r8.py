"""N3 Codex round 8 (codex-n3r8.md; SPEC-N3-R8-FIXES.md): the reproductions as regression tests.

H1/M1/L1  A subclass (copy hook, field overrides, a changing read) could make the lift certify other
          numbers than N1 saw.  Option A (owner decision): the lift runs only for supports of EXACT type
          Plane, Cylinder or Cone (else ``SUPPORT_UNSUPPORTED``), whose declared fields are read once
          and are exactly ``float`` / float64 ``ndarray`` of shape (3,) (else ``OUT_OF_DOMAIN``).
M2        N1 reads the support too: the same check runs before N1 sees it, so no raw ``TypeError``.
"""
from __future__ import annotations

import copy
import math
from fractions import Fraction

import numpy as np
import pytest

import test_display_tessellation_lift as T
from anygeometry import display_tessellation as dt
from anygeometry.display_tessellation import DisplayTolerance, face_display_mesh
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Cone, Cylinder, Plane
from anygeometry.tolerance import TolerancePolicy
from test_display_tessellation_lift_n3r7 import TOL, _BASE, _face, _with, _run


class ShiftOnCopy(Plane):
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "u_vector", self.u_vector.astype(np.int64))

    def __copy__(self):
        return Plane((0, 0, 0), self.u_vector, self.v_vector)


class Uncopyable(ShiftOnCopy):
    def __copy__(self):
        raise TypeError("copy disabled")


class ChangingHeight(Cone):
    def __getattribute__(self, name):
        if name == "height":
            reads = vars(self).get("reads", 0) + 1
            object.__setattr__(self, "reads", reads)
            if reads >= 4:
                return 1j
        return super().__getattribute__(name)


class PlainPlane(Plane):
    pass


class PlainCylinder(Cylinder):
    pass


class PlainCone(Cone):
    pass


def _plane_mesh(cls):
    g = GeometryModel(tolerance=TolerancePolicy(surface_residual=.01))
    p = [g.add_point(*xyz) for xyz in ((0, 0, 0), (1, 0, 0), (0, 1, 0))]
    edges = [g.add_line(a, b) for a, b in zip(p, p[1:] + p[:1])]
    f = g.add_face(edges, surface=cls((0, 0, .005), (1, 0, 0), (0, 1, 0)))
    return face_display_mesh(g, f, DisplayTolerance(.25, sag=.001))


def _refused(call, code):
    with pytest.raises(dt.DisplayTrimRefused) as e:
        call()
    assert e.value.code == code
    return str(e.value)


@pytest.mark.parametrize("cls", [ShiftOnCopy, Uncopyable, PlainPlane])
def test_h1_m1_a_plane_subclass_is_unsupported(cls):
    _refused(lambda: _plane_mesh(cls), "SUPPORT_UNSUPPORTED")


def test_l1_a_cone_subclass_with_a_changing_read_is_unsupported():
    s = ChangingHeight((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 2., 1.)
    object.__setattr__(s, "reads", 0)
    g, f = _face("cone", s)
    _refused(lambda: face_display_mesh(g, f, TOL), "SUPPORT_UNSUPPORTED")


@pytest.mark.parametrize("kind,cls,args", [
    ("cylinder", PlainCylinder, ((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 1.)),
    ("cone", PlainCone, ((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 2., 1.))])
def test_cylinder_and_cone_subclasses_are_unsupported(kind, cls, args):
    g, f = _face(kind, cls(*args))
    _refused(lambda: face_display_mesh(g, f, TOL), "SUPPORT_UNSUPPORTED")


def test_m2_injected_fraction_and_longdouble_fields_are_out_of_domain_not_raw():
    s = _with("cylinder", "start_angle", Fraction(0))
    object.__setattr__(s, "sweep_angle", np.longdouble(2 * np.pi))
    g, f = _face("cylinder", s)
    _refused(lambda: face_display_mesh(g, f, TOL), "OUT_OF_DOMAIN")


@pytest.mark.parametrize("kind", list(_BASE))
def test_scalar_and_array_field_types_are_exact(kind):
    names = [n for n in ("radius", "height", "radius_start", "radius_end", "start_angle", "sweep_angle")
             if hasattr(_BASE[kind], n)]
    for name in names:
        v = getattr(_BASE[kind], name)
        for bad in (int(v), Fraction(v), np.float64(v), np.float32(v), np.longdouble(v)):
            r = _run(kind, _with(kind, name, bad))
            assert r[0] == "refused" and r[1] == "OUT_OF_DOMAIN", (kind, name, type(bad))
    for bad in (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.int64), [0., 0., 0.],
                np.zeros((3, 1)), np.zeros(4)):
        r = _run(kind, _with(kind, "origin", bad))
        assert r[0] == "refused" and r[1] == "OUT_OF_DOMAIN", (kind, type(bad))


@pytest.mark.parametrize("kind", list(_BASE))
def test_ordinarily_built_supports_are_accepted(kind):
    assert _run(kind)[0] == "ok"
    s = copy.copy(_BASE[kind])                                  # exact type, plain fields
    assert _run(kind, s)[0] == "ok"


def test_neighbour_face_subclass_support_is_refused_before_n1_reads_it():
    g = GeometryModel(tolerance=TolerancePolicy(surface_residual=.01))
    p = [g.add_point(*q) for q in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0))]
    e = {k: g.add_line(*k) for k in ((p[0], p[1]), (p[1], p[2]), (p[2], p[0]), (p[1], p[3]), (p[3], p[2]))}
    a = g.add_face([e[(p[0], p[1])], e[(p[1], p[2])], e[(p[2], p[0])]],
                   surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    g.add_face([e[(p[1], p[3])], e[(p[3], p[2])], e[(p[1], p[2])]],
               surface=PlainPlane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    _refused(lambda: face_display_mesh(g, a, DisplayTolerance(.25, sag=.001)), "SUPPORT_UNSUPPORTED")
