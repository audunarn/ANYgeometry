"""The display-tessellation acceptance corpus loads and every face lies on its support (plan S0)."""
import numpy as np
import pytest

import display_tessellation_corpus as corpus


@pytest.mark.parametrize("name", sorted(corpus.ALL))
def test_corpus_case_builds_valid_faces_on_their_supports(name):
    case = corpus.ALL[name]()
    g = case.model
    assert g.faces or "edge" in case.info
    for face_id in case.faces.values():
        face = g.faces[face_id]
        support = face.surface
        assert support is not None
        t = np.linspace(0., 1., 33)
        for loop in (face.loop, *face.holes):
            for use in loop:
                points = g.evaluate_edge_many(use.edge, t)
                back = np.array([support.evaluate(*support.local_uv(p)) for p in points])
                scale = max(1.0, float(np.abs(points).max()))
                assert np.abs(back - points).max() <= 1e-9 * scale, (name, face_id, use.edge)


def test_tripod_fixture_is_the_menu_example_skins():
    case = corpus.tripod_skins()
    assert len(case.faces) == 18
    shared = [e for e in case.model.edges if len(case.model.faces_using_edge(e)) > 1]
    assert len(shared) == 33


def _boundary(g, face, loop, count=129):
    t = np.linspace(0., 1., count)[:-1]
    return np.vstack([g.evaluate_edge_many(use.edge, t if use.forward else 1. - t) for use in loop])


def _uv_area(support, points, period=None):
    uv = np.array([support.local_uv(p) for p in points])
    if period is not None:
        uv[:, 0] = np.unwrap(uv[:, 0] * 2 * np.pi / period) * period / (2 * np.pi)
    uv -= uv.min(axis=0)
    return 0.5 * abs(np.dot(uv[:, 0], np.roll(uv[:, 1], -1)) - np.dot(np.roll(uv[:, 0], -1), uv[:, 1]))


@pytest.mark.parametrize("name,holes", [("seam_hole_0", 1), ("seam_hole_1", 1), ("seam_hole_2", 1), ("seam_hole_3", 1),
                                        ("holed_cone_comb", 1), ("cone_comb_r1", 0), ("cone_comb_r3", 0),
                                        ("l_shape", 0)])
def test_holes_and_trim_areas_are_the_stated_ones(name, holes):
    case = corpus.ALL[name]()
    g = case.model
    (face_id,) = case.faces.values()
    face = g.faces[face_id]
    assert len(face.holes) == holes
    support = face.surface
    period = 1.0 if abs(abs(getattr(support, "sweep_angle", 0.0)) - 2 * np.pi) < 1e-12 else None
    area = _uv_area(support, _boundary(g, face, face.loop), period) - sum(
        _uv_area(support, _boundary(g, face, h), period) for h in face.holes)
    assert area == pytest.approx(case.info["uv_area"], rel=1e-9)


@pytest.mark.parametrize("name", ["box_corner_cut", "wavy_band", "wavy_band_cornerless", "wavy_band_split", "steep_cuts"])
def test_band_boundaries_are_their_defining_sections(name):
    """Every boundary point lies on the lower section, the upper section, or a side ruling."""
    case = corpus.ALL[name]()
    g = case.model
    (face_id,) = case.faces.values()
    points = _boundary(g, g.faces[face_id], g.faces[face_id].loop)
    theta = np.arctan2(points[:, 1], points[:, 0])
    low, high = case.info["low"](theta), case.info["high"](theta)
    ends = (0., np.pi) if name == "box_corner_cut" else (-case.info["half_angle"], case.info["half_angle"])
    on_low = np.abs(points[:, 2] - low) <= 1e-9
    on_high = np.abs(points[:, 2] - high) <= 1e-9
    on_side = np.min(np.abs(np.mod(theta[:, None] - np.array(ends)[None, :] + np.pi, 2 * np.pi) - np.pi), axis=1) <= 1e-9
    assert np.all(on_low | on_high | on_side)
    middle = np.abs(theta - 0.5 * (ends[0] + ends[1])) <= 0.3 * (ends[1] - ends[0])
    assert np.any(on_low & middle) and np.any(on_high & middle)
    if name == "box_corner_cut":
        assert np.any((np.abs(points[:, 2] - .4) <= 1e-9) & (np.abs(theta - np.pi / 2) <= 1e-9))


def test_standalone_ellipse_and_lens_sections():
    case = corpus.ellipse_edge()
    assert np.allclose(case.model.evaluate_edge_many(case.info["edge"], [0., .5, 1.]),
                       [(20., 0., 0.), (0., 1., 0.), (-20., 0., 0.)], atol=1e-12)
    for half in (.05, .001):
        lens = corpus.lens(half)
        mid = lens.model.evaluate_edge_many(lens.info["upper"], [.5])[0]
        assert mid[2] == pytest.approx(.4 + .5 * (1 - np.cos(half)), abs=1e-12)
        assert np.arctan2(mid[1], mid[0]) == pytest.approx(.5, abs=1e-12)
