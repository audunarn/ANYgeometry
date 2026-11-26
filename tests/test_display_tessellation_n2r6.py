"""N2 Codex round 6 (owner decisions): model-produced ids are formatted only below 10**18; the N3
deviation contract is parametric; coverage and allocation wording."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from anygeometry import display_tessellation as dt
from anygeometry.errors import GeometryError
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Plane

_DOCS = Path(__file__).resolve().parents[1] / "docs" / "DISPLAY_TESSELLATION.md"
_TEXT = " ".join(_DOCS.read_text(encoding="utf-8").split())


def _model(face_id_exp, parameterized):
    g = GeometryModel()
    a, b, c = [g.add_point(x, y) for x, y in [(0., 0.), (1., 0.), (0., 1.)]]
    edges = [g.add_line(a, b), g.add_line(b, c), g.add_line(c, a)]
    g.reserve_id_state({**g.id_state(), "face": 10 ** face_id_exp})
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    face = g.add_face(edges, surface=plane)
    if parameterized:
        g.set_face_parameterization(face, plane)
    return g, edges


@pytest.mark.parametrize("parameterized", [True, False])
@pytest.mark.parametrize("call", ["edge_display_samples", "display_rounding_allowance"])
def test_r6_1_unbounded_face_id_is_a_typed_refusal(call, parameterized):
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    try:
        g, edges = _model(5000, parameterized)
        if not parameterized:
            # a support that is not a plane, cylinder or cone, so the second branch is taken
            from anygeometry.surfaces import ExtrudedSurface  # noqa: F401
            face = next(iter(g.faces.values()))
            object.__setattr__(face, "surface", object())
        try:
            getattr(dt, call)(g, edges[0], dt.DisplayTolerance(.1))
        except (dt.DisplayTrimRefused, GeometryError) as exc:
            assert type(exc) is not ValueError
            assert "<id out of display range>" in str(exc)
        except ValueError as exc:
            raise AssertionError(f"raw ValueError escaped: {exc!r}")
    finally:
        sys.set_int_max_str_digits(old)


def test_r6_1_id_text_is_arithmetic():
    assert dt._id_text(10 ** 18 - 1) == str(10 ** 18 - 1)
    assert dt._id_text(-(10 ** 18 - 1)) == str(-(10 ** 18 - 1))
    assert dt._id_text(10 ** 18) == "<id out of display range>"
    assert dt._id_text(-(10 ** 5000)) == "<id out of display range>"


def test_r6_2_n3_contract_is_parametric():
    for phrase in ("PARAMETRIC", "same barycentric weights", "max over the triangle", "rigorously",
                   "not by sampling", "stricter than radial projection", "0.380", "0.293"):
        assert phrase in _TEXT, phrase


def test_r6_3_coverage_closure_wording():
    assert "not in the closure of the positive-area interior" in _TEXT
    assert "may appear as vertices" in _TEXT


def test_r6_4_allocation_claim_is_scoped():
    assert "form refusals are O(1) memory" in _TEXT
    assert "one bounded copy" in _TEXT
