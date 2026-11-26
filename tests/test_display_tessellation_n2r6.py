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


def test_r6_2_n3_contract_is_parametric():
    for phrase in ("PARAMETRIC", "same barycentric weights", "max over the triangle", "rigorously",
                   "not by sampling", "stricter than radial projection", "0.380", "0.293"):
        assert phrase in _TEXT, phrase


def test_r6_2_frustum_numbers():
    import numpy as np
    from anygeometry._slab_triangulation import triangulate_rings
    from anygeometry.surfaces import Cone
    uv = np.array([(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    mesh = triangulate_rings([uv])
    cone = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .01, 1., 1., 0., np.pi / 2)
    xyz = np.array([cone.evaluate(*p) for p in mesh.points])
    assert sum(0 in t and 2 in t for t in mesh.triangles) == 2
    midpoint = (xyz[0] + xyz[2]) / 2
    radial = 1 - 2 ** -.5
    parametric = float(np.linalg.norm(midpoint - cone.evaluate(.5, .5)))
    assert abs(radial - 0.2929) < 1e-4
    assert abs(parametric - 0.3800) < 1e-4, parametric
    assert radial < .30 < parametric


def test_r6_3_coverage_closure_wording():
    assert "not in the closure of the positive-area interior" in _TEXT
    assert "may appear as vertices" in _TEXT


def test_r6_4_allocation_claim_is_scoped():
    assert "one bounded copy" in _TEXT


def test_r7_n3_face_tolerance_rule_is_pinned():
    import numpy as np
    from anygeometry._slab_triangulation import triangulate_rings
    g = GeometryModel()
    plane = Plane((0, 0, 0), (1, 0, 0), (0, 1, 0))
    uv = np.array([(0., 0.), (1., 0.), (0., 1.)])
    vertices = [g.add_point(u, v, 4e-9) for u, v in uv]
    edges = [g.add_line(a, b) for a, b in zip(vertices, vertices[1:] + vertices[:1])]
    g.add_face(edges, surface=plane)
    tol = dt.DisplayTolerance(.1)
    assert all(dt.display_rounding_allowance(g, e, tol).sag is None for e in edges)
    xyz = np.array([dt.edge_display_samples(g, e, tol).points[0] for e in edges])
    np.testing.assert_array_equal(xyz[:, :2], uv)
    np.testing.assert_array_equal(triangulate_rings([uv]).triangles, [[0, 1, 2]])
    error = max(np.linalg.norm(xyz[i] - plane.evaluate(*uv[i])) for i in range(3))
    t_face = g.tolerance.surface_residual
    assert g.tolerance.length < error <= t_face
    assert "`T_face = tol.sag`" in _TEXT and "never certified against a smaller per-edge value" in _TEXT


def test_r7_allocation_wording():
    assert "O(number of rings)" in _TEXT and "102,027 bytes" in _TEXT
