"""N3 Codex round 7 (codex-n3r7.md; SPEC-N3-R7-FIXES.md): the reproductions as regression tests.

M1/L1  class "caller-supplied support data is converted outside the typed-refusal handler or skipped
       by the guard": ragged, big-int and string-bearing stored fields.  G1: the guard covers exactly
       the declared fields of the base support class; a subclass's extra attributes are never read and
       are outside the domain.  G2: every guarded field converts to binary64 or the face is refused
       ``OUT_OF_DOMAIN`` with fixed text, never a raw exception.
"""
from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest

import test_display_tessellation_lift as T
from anygeometry import _display_lift as lift
from anygeometry import display_tessellation as dt
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Cone, Cylinder, Plane
from anygeometry.tolerance import TolerancePolicy

TOL = dt.DisplayTolerance(.25, sag=.02)
_FIELDS = ("points", "triangles", "chart", "source")
_SCALARS = ("face_id", "ring_vertex_count", "sense", "face_tolerance", "deviation_bound", "trim_sag_bound",
            "degenerate_count")


def outcome(call):
    try:
        m = call()
    except dt.DisplayTrimRefused as e:
        return ("refused", e.code, str(e))
    return ("ok", {f: getattr(m, f).tobytes() for f in _FIELDS} | {f: getattr(m, f) for f in _SCALARS})


def _face(kind, support=None):
    g = GeometryModel(tolerance=TolerancePolicy(surface_residual=.01))
    if kind == "plane":
        s = support
        p = [g.add_point(*q) for q in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]]
    elif kind == "cylinder":
        s = support
        p = [g.add_point(math.cos(t), math.sin(t), z) for t, z in ((0., 0.), (.125, 0.), (.125, 1.), (0., 1.))]
    else:
        s = support
        p = [g.add_point(math.cos(t) * r, math.sin(t) * r, z)
             for t, r, z in ((0., 1., 0.), (.125, 1., 0.), (.125, 2., 1.), (0., 2., 1.))]
    f = g.add_face([g.add_line(a, b) for a, b in zip(p, p[1:] + p[:1])], surface=_BASE[kind])
    if support is not None:
        T._force(g, f, surface=s)                      # outside the model's commit validation
    return g, f


def _run(kind, support=None):
    g, f = _face(kind, support)
    return outcome(lambda: T._mesh(g, f, TOL))


_BASE = {"plane": Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)),
         "cylinder": Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 1.),
         "cone": Cone((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 2., 1.)}


def test_baselines_are_accepted():
    for kind in _BASE:
        assert _run(kind)[0] == "ok", kind


# ---- Codex reproduction: a subclass's extra attribute is outside the contract (G1)
@dataclass(frozen=True)
class AnnotatedPlane(Plane):
    extra: object = None


@pytest.mark.parametrize("extra", [2 ** 201, [2. ** 201, "label"], [[0], [1, 2]], "label", None, 5, 2 ** 5000,
                                   1j, {"a": 1}])
def test_m1_l1_subclass_extra_attribute_never_changes_the_outcome(extra):
    plain = _run("plane")
    s = AnnotatedPlane((0, 0, 0), (1, 0, 0), (0, 1, 0), extra)
    assert _run("plane", s) == plain


# ---- G2: every guarded base field converts or is refused, never a raw exception
def _fields(kind):
    return [f.name for f in dataclasses.fields(type(_BASE[kind]))]


def _numeric_base_fields(kind):
    out = []
    for name in _fields(kind):
        v = getattr(_BASE[kind], name)
        if isinstance(v, (np.ndarray, float, int)):
            out.append(name)
    return out


def _with(kind, name, value):
    s = copy.copy(_BASE[kind])
    object.__setattr__(s, name, value)
    return s


def _convert(value, f):
    """The same value with every number passed through ``f`` (scalar or vector)."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return f(float(arr))
    return [f(float(x)) for x in arr]


_EQUIVALENT = {"int": lambda x: int(x), "fraction": lambda x: Fraction(x), "np.int64": lambda x: np.int64(int(x)),
               "np.float32": lambda x: np.float32(x), "np.float64": lambda x: np.float64(x)}


def _cases():
    for kind in _BASE:
        for name in _numeric_base_fields(kind):
            value = getattr(_BASE[kind], name)
            for label in _EQUIVALENT:
                if label in ("int", "np.int64") and any(float(x) != int(x) for x in np.ravel(value)):
                    continue
                if label == "fraction" and name in ("radial_direction",):
                    pass
                yield pytest.param(kind, name, label, id=f"{kind}-{name}-{label}")


@pytest.mark.parametrize("kind,name,label", list(_cases()))
def test_g2_equivalent_numeric_types_give_the_float_mesh(kind, name, label):
    base = _run(kind)
    value = _convert(getattr(_BASE[kind], name), _EQUIVALENT[label])
    assert _run(kind, _with(kind, name, value)) == base


def _bad_values(name, value):
    """Values that are not finite real arrays of the field's shape."""
    scalar = np.asarray(value).ndim == 0
    huge = 2 ** 5000
    if scalar:
        return {"huge": huge, "2**201": 2 ** 201, "string": "label", "None": None, "complex": 1j,
                "list": [1.0, 2.0], "obj": object(), "nan": float("nan"), "inf": float("inf")}
    return {"huge": [huge, 0, 0], "2**201": [2 ** 201, 0, 0], "string": "label", "None": None,
            "complex": [1j, 0, 0], "ragged": [[0], [1, 2]], "mixed": [2. ** 201, "label", 0],
            "short": [1.0, 0.0], "none-elem": [1.0, None, 0.0], "nan": [float("nan"), 0., 0.],
            "inf": [float("inf"), 0., 0.], "obj": [object(), 0, 0]}


def _bad_cases():
    for kind in _BASE:
        for name in _numeric_base_fields(kind):
            for label, bad in _bad_values(name, getattr(_BASE[kind], name)).items():
                yield pytest.param(kind, name, label, bad, id=f"{kind}-{name}-{label}")


@pytest.mark.parametrize("kind,name,label,bad", list(_bad_cases()))
def test_g2_unconvertible_or_oversized_fields_are_a_typed_refusal(kind, name, label, bad):
    result = _run(kind, _with(kind, name, bad))
    assert result[0] == "refused", (kind, name, label)
    assert result[1] in ("OUT_OF_DOMAIN",), result[:2]
    assert "label" not in result[2] and "object at" not in result[2]       # fixed text, no caller data


def test_g1_the_guard_covers_the_declared_fields_of_the_base_class():
    """Introspection: every dataclass field of each base class is read by the guard (a new base field
    that the guard misses fails here); a subclass's extra attribute is not."""
    for kind, support in _BASE.items():
        declared = [f.name for f in dataclasses.fields(type(support))]
        assert list(lift._guarded_fields(support)) == declared
        sub = copy.copy(support)
        object.__setattr__(sub, "zzz_extra", 2.0 ** 300)
        assert list(lift._guarded_fields(sub)) == declared
