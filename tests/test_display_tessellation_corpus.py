"""The display-tessellation acceptance corpus loads and every face lies on its support (plan S0)."""
import numpy as np
import pytest

import display_tessellation_corpus as corpus


BUILDERS = {**corpus.ALL, **{name: builder for name, (builder, _code) in corpus.REFUSED.items()}}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_corpus_case_builds_valid_faces_on_their_supports(name):
    case = BUILDERS[name]()
    g = case.model
    assert g.faces or "edge" in case.info
    for face_id in case.faces.values():
        face = g.faces[face_id]
        support = face.surface
        if name in corpus.REFUSED:
            continue
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
    """Signed support-uv area of a loop (positive counter-clockwise in the support's u, v)."""
    uv = np.array([support.local_uv(p) for p in points])
    if period is not None:
        uv[:, 0] = np.unwrap(uv[:, 0] * 2 * np.pi / period) * period / (2 * np.pi)
    uv -= uv.min(axis=0)
    return 0.5 * (np.dot(uv[:, 0], np.roll(uv[:, 1], -1)) - np.dot(np.roll(uv[:, 0], -1), uv[:, 1]))


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
    outer = _uv_area(support, _boundary(g, face, face.loop), period)
    holes = [_uv_area(support, _boundary(g, face, h), period) for h in face.holes]
    # holes wind against the outer loop, and the trim area is the stated one
    assert all(np.sign(h) == -np.sign(outer) for h in holes)
    assert abs(outer) - sum(abs(h) for h in holes) == pytest.approx(case.info["uv_area"], rel=1e-9)


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


@pytest.mark.parametrize("name,bottom,corners", [("wavy_band", 1, 4), ("wavy_band_cornerless", 1, 0),
                                                 ("wavy_band_split", 4, 4)])
def test_band_variants_are_the_stated_topology(name, bottom, corners):
    """Counted on the geometry: edges lying on the lower section, on the upper one, rulings."""
    case = corpus.ALL[name]()
    g = case.model
    (face_id,) = case.faces.values()
    face = g.faces[face_id]
    on = {"low": 0, "high": 0, "side": 0}
    for use in face.loop:
        p = g.evaluate_edge_many(use.edge, np.linspace(0., 1., 17))
        theta = np.arctan2(p[:, 1], p[:, 0])
        if np.all(np.abs(p[:, 2] - case.info["low"](theta)) <= 1e-9):
            on["low"] += 1
        elif np.all(np.abs(p[:, 2] - case.info["high"](theta)) <= 1e-9):
            on["high"] += 1
        else:
            assert np.ptp(theta) <= 1e-12          # a ruling
            on["side"] += 1
    assert on == {"low": bottom, "high": 1, "side": 2}
    assert len(face.corners) == corners


def test_cell_outline_refuses_loops_sharing_a_lattice_point():
    cells = [(i, j) for i in range(3) for j in range(3) if (i, j) not in ((2, 0), (1, 1))]
    with pytest.raises(ValueError):
        corpus._cell_outline(cells, (0., 0.), 1.)
    with pytest.raises(ValueError):
        corpus._cell_outline([(0, 0), (1, -1)], (0., 0.), 1.)


@pytest.mark.parametrize("rotation", range(4))
def test_seam_holes_cross_the_seam(rotation):
    """Both rings span azimuth 0 of the full-turn support (the case is about the seam)."""
    case = corpus.seam_hole(rotation)
    g = case.model
    face = g.faces[case.faces["holed"]]
    for loop in (face.loop, *face.holes):
        p = _boundary(g, face, loop)
        theta = np.arctan2(p[:, 1], p[:, 0])
        assert theta.min() < 0.0 < theta.max() and np.ptp(theta) < 1e-3


@pytest.mark.parametrize("name", ["cone_comb_r1", "cone_comb_r3", "l_shape"])
def test_combs_are_concave(name):
    """A comb or L is far from its convex hull (a same-area rectangle would not be)."""
    case = corpus.ALL[name]()
    g = case.model
    (face_id,) = case.faces.values()
    face = g.faces[face_id]
    uv = np.array([face.surface.local_uv(p) for p in _boundary(g, face, face.loop)])
    hull = _convex_hull_area(uv)
    assert hull >= 1.2 * case.info["uv_area"]
    if "rings" in case.info:
        corners = len(case.info["rings"][0])
        assert corners == {"cone_comb_r1": 10, "cone_comb_r3": 10}[name]


def _convex_hull_area(points):
    pts = sorted(map(tuple, np.round(points, 12)))
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out
    hull = np.array(half(pts)[:-1] + half(pts[::-1])[:-1])
    return 0.5 * abs(np.dot(hull[:, 0], np.roll(hull[:, 1], -1)) - np.dot(np.roll(hull[:, 0], -1), hull[:, 1]))
