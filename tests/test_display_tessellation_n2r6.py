"""N2 Codex round 6 (owner decisions): model-produced ids are formatted only below 10**18;
coverage and allocation wording (the N3 lift items moved to the evidence note N3-LIFT-INPUTS.md)."""
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
        with pytest.raises(dt.DisplayTrimRefused) as info:
            getattr(dt, call)(g, edges[0], dt.DisplayTolerance(.1))
        assert info.value.code == "SUPPORT_UNSUPPORTED"
        assert "<id out of display range>" in str(info.value)
    finally:
        sys.set_int_max_str_digits(old)


def test_r6_1_id_text_is_arithmetic():
    assert dt._id_text(10 ** 18 - 1) == str(10 ** 18 - 1)
    assert dt._id_text(-(10 ** 18 - 1)) == str(-(10 ** 18 - 1))
    assert dt._id_text(10 ** 18) == "<id out of display range>"
    assert dt._id_text(-(10 ** 5000)) == "<id out of display range>"


def test_r6_3_coverage_closure_wording():
    assert "not in the closure of the positive-area interior" in _TEXT
    assert "may appear as vertices" in _TEXT


def test_r6_4_allocation_claim_is_scoped():
    assert "one bounded copy" in _TEXT


def test_r7_allocation_wording():
    assert "O(number of rings)" in _TEXT and "16,666 = 50_000 // 3" in _TEXT and "135,355 bytes" in _TEXT
    assert "12,500" not in _TEXT and "102,027" not in _TEXT
    assert "Lifting this mesh onto a support is N3's contract" in _TEXT
    assert "must not rely on cut_fraction reconstruction for orientation or coverage" in _TEXT
    # N2's own contract states no N3 obligation; since N3c the N3 section defines T_face itself
    n2_part, n3_part = _TEXT.split("## Face lift (N3)")
    assert "T_face" not in n2_part and "T_face" in n3_part
