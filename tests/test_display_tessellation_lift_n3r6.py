"""N3 Codex round 6 (codex-n3r6.md; SPEC-N3-R6-FIXES.md): the reproductions as regression tests.

M1  class "the scalar fast path duplicates array-path arithmetic with other primitives" (``math.hypot``
    vs ``np.hypot`` differ by an ulp at the strict ``rho > 16 err`` cutoff): the scalar path must give
    exactly the array path's decision and values.  Differential tests run faces with the fast path on
    and forced off (``_SMALL_POINTS = 0``) and demand an identical outcome.
L1  class "the upper guard's parameter list misses a stored support field" (``Cylinder.height``):
    the list is derived from the stored attributes; an introspection test fails on a missed field.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import display_tessellation_corpus as corpus
import test_display_tessellation_lift as T
from anygeometry import _display_lift as lift
from anygeometry import display_tessellation as dt
from anygeometry.line_ranges import LineRange
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Cone, Cylinder, Plane
from anygeometry.tolerance import TolerancePolicy

_FIELDS = ("points", "triangles", "chart", "source")
_SCALARS = ("face_id", "ring_vertex_count", "sense", "face_tolerance", "deviation_bound", "trim_sag_bound",
            "degenerate_count")


def outcome(call):
    """The outcome of a display call: the refusal (code, text), or the mesh's bytes and scalars."""
    try:
        m = call()
    except dt.DisplayTrimRefused as e:
        return ("refused", e.code, str(e))
    return ("ok", {f: getattr(m, f).tobytes() for f in _FIELDS} | {f: getattr(m, f) for f in _SCALARS})


def both_paths(monkeypatch, call):
    """``(fast, array)``: the outcome with the scalar path on (default) and forced off."""
    fast = outcome(call)
    monkeypatch.setattr(lift, "_SMALL_POINTS", 0)
    array = outcome(call)
    monkeypatch.setattr(lift, "_SMALL_POINTS", 40)
    return fast, array


def _near_axis_face():
    """Codex N3 r6 M1: a vertex whose ``rho`` sits at the strict ``16 err`` cutoff, where ``math.hypot``
    is one ulp above ``np.hypot``."""
    g = GeometryModel(tolerance=TolerancePolicy(length=3e-9, surface_residual=.001))
    s = Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 1e-5, 1.)

    def point(t, r, z):
        return (-r * math.sin(t), r * math.cos(t), z)

    x = float.fromhex("0x1.fd712f9a8185dp-47")
    y = float.fromhex("0x1.98eaecb8bcba9p-50")
    base = [point(.1, 1e-5, 0.), point(.4, 1e-5, 0.)]
    p = [g.add_point(*q) for q in [*base, (-y, x, 1.)]]
    carrier = [*base, point(.1, 2e-9, 1.)]
    f = g.add_face([g.add_curve(p[i], p[j], LineRange(carrier[i], carrier[j]))
                    for i, j in ((0, 1), (1, 2), (2, 0))], surface=s)
    assert not g.validate_topology()
    return g, f, dt.DisplayTolerance(.5, sag=.001)


def test_m1_the_codex_face_has_one_outcome_on_both_paths(monkeypatch):
    """On 17c0de7 the scalar path published the triangle and the array path refused TRIM_INVALID."""
    g, f, tol = _near_axis_face()
    fast, array = both_paths(monkeypatch, lambda: dt.face_display_mesh(g, f, tol))
    assert fast == array


def _corpus_calls():
    for name, make in corpus.ALL.items():
        case = make()
        for label, face in case.faces.items():
            if isinstance(case.model.faces[face].surface, (Plane, Cylinder, Cone)):
                yield f"{name}/{label}", case.model, face


def test_m1_corpus_faces_decide_identically_on_both_paths(monkeypatch):
    """Every corpus plane / cylinder / cone face of up to 40 points per mesh, both corpus steps (a
    bounded subset: the full differential over the fuzz sets is a script, evidence n3r6-diff.txt)."""
    count = 0
    for label, model, face in _corpus_calls():
        if count >= 120:
            break
        for step in corpus.STEPS:
            tol = dt.DisplayTolerance(step)
            fast, array = both_paths(monkeypatch, lambda: dt.face_display_mesh(model, face, tol))
            assert fast == array, (label, step)
        count += 1
    assert count > 0


def test_m1_small_curved_projection_matches_the_array_projection_bit_for_bit():
    """The scalar and the array projections hold the same P and error bounds (the arrays' values)."""
    rng = np.random.default_rng(6)
    s = Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 2.)
    chart = lift._CylinderChart(s)
    for _ in range(40):
        n = int(rng.integers(3, 30))
        t = np.sort(rng.uniform(-1, 1, n))
        z = rng.uniform(0, 1, n)
        r = rng.uniform(.5, 2, n)
        pts = (np.asarray(chart.o, dtype=float) + np.outer(r * np.cos(t), chart.e1) + np.outer(r * np.sin(t), chart.e2)
               + np.outer(z, chart.a))
        theta = t.copy()
        small = lift._SmallCurvedProjection(1, chart, pts, theta)
        full = lift._CurvedProjection(1, chart, pts, theta)
        assert np.array(small.ul).tobytes() == np.asarray(full.u).tobytes()
        assert np.array(small.aerrl).tobytes() == np.asarray(full.aerr).tobytes()
        assert np.array(small.errl).tobytes() == np.asarray(full.err).tobytes()


def test_m1_primitive_audit_the_fast_path_uses_no_math_transcendentals():
    """The scalar path's source calls numpy for hypot, atan2, cos and sin (the audit table)."""
    import inspect
    text = inspect.getsource(lift._SmallCurvedProjection) + inspect.getsource(lift._projection_triangles_small)
    for banned in ("math.hypot", "math.atan2", "math.cos", "math.sin"):
        assert banned not in text, banned


# --------------------------------------------------------------------------------------------- L1
def _cylinder_height_face(height):
    g = GeometryModel(tolerance=TolerancePolicy(surface_residual=.01))
    s = Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., height)
    p = [g.add_point(math.cos(t), math.sin(t), z) for t, z in ((0., 0.), (.125, 0.), (.125, 1.), (0., 1.))]
    f = g.add_face([g.add_line(a, b) for a, b in zip(p, p[1:] + p[:1])], surface=s)
    assert not g.validate_topology()
    return g, f


def test_l1_cylinder_height_above_the_domain_is_out_of_domain():
    """On 17c0de7 the face with stored height 2**201 was accepted (2 triangles)."""
    g, f = _cylinder_height_face(2.0 ** 201)
    text = T._refused("OUT_OF_DOMAIN", lambda: T._mesh(g, f, dt.DisplayTolerance(.25, sag=.02)))
    assert "above 2**200" in text


def test_l1_cylinder_height_inside_the_domain_is_accepted():
    g, f = _cylinder_height_face(2.0 ** 199)
    assert len(T._mesh(g, f, dt.DisplayTolerance(.25, sag=.02)).triangles) == 2


def _supports():
    return (Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 2.),
            Cone((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 2., 3.))


def _numeric_attributes(support):
    """Every stored numeric attribute of a support instance: (name, index or None)."""
    out = []
    for name in (f.name for f in dataclasses.fields(type(support))):      # r7 G1: the declared fields of the base class
        flat = np.asarray(getattr(support, name))
        if flat.dtype.kind not in "fiu":
            continue
        out.extend((name, i) for i in range(flat.size)) if flat.ndim else out.append((name, None))
    return out


@pytest.mark.parametrize("support", _supports(), ids=lambda s: type(s).__name__)
def test_l1_the_upper_guard_covers_every_stored_numeric_attribute(support):
    """Class: set each stored numeric attribute (each component) to 2**201 on a copy; the guard's
    parameter list must then exceed the bound.  A missed or newly added field fails here."""
    attributes = _numeric_attributes(support)
    assert attributes
    for name, index in attributes:
        copy = object.__new__(type(support))
        for key, value in vars(support).items():
            object.__setattr__(copy, key, np.array(value, dtype=float) if isinstance(value, np.ndarray) else value)
        value = getattr(copy, name)
        if index is None:
            object.__setattr__(copy, name, 2.0 ** 201)
        else:
            value[index] = 2.0 ** 201
        listed = np.abs(np.asarray(lift._upper_parameters(copy), dtype=float))
        assert bool(np.any(listed > lift._MAX_MAGNITUDE)), (type(support).__name__, name, index)
