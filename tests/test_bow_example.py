"""Programme step 3, P3a: the two-patch bow (owner Q2: two patches meeting at the stem)."""
from __future__ import annotations

import math

import numpy as np

from anygeometry.bow_example import bow_two_patch


def _curve_definition(curve):
    return curve.degree, tuple(curve.knots), tuple(map(tuple, curve.controls)), curve.weights, curve.domain


def test_the_stem_and_keel_iso_curves_of_both_patches_are_the_same_exact_curves():
    starboard, port = bow_two_patch()
    for fixed, value in (("u", 0.), ("v", 0.)):              # stem (u = 0) and keel (v = 0)
        assert _curve_definition(starboard.iso_curve(fixed, value)) == _curve_definition(port.iso_curve(fixed, value))
    # the exact boundary rows of the nets (iso_curve() rounds its controls) and the ranges
    assert starboard.controls[0] == port.controls[0]                                       # stem row
    assert [row[0] for row in starboard.controls] == [row[0] for row in port.controls]     # keel column
    assert (starboard.u_range, starboard.v_range) == (port.u_range, port.v_range)
    stem = np.asarray(starboard.iso_curve("u", 0.).controls)
    keel = np.asarray(starboard.iso_curve("v", 0.).controls)
    assert np.all(stem[:, 1] == 0.) and np.all(keel[:, 1] == 0.)          # both in the centreline plane


def test_the_port_patch_is_the_exact_mirror_of_the_starboard_patch():
    starboard, port = bow_two_patch()
    assert (port.degree_u, port.degree_v, port.knots_u, port.knots_v) == \
        (starboard.degree_u, starboard.degree_v, starboard.knots_u, starboard.knots_v)
    for row_s, row_p in zip(starboard.controls, port.controls):
        for (xs, ys, zs), (xp, yp, zp) in zip(row_s, row_p):
            assert (xp, yp, zp) == (xs, -ys, zs)
            assert yp != 0. or math.copysign(1., yp) == 1.               # a zero stays +0.0
    uv = np.random.default_rng(3).uniform(0., 1., (10000, 2))
    uv = np.vstack([uv, [[0., t] for t in np.linspace(0., 1., 101)], [[t, 0.] for t in np.linspace(0., 1., 101)]])
    assert np.array_equal(port.evaluate_many(uv), starboard.evaluate_many(uv) * np.array([1., -1., 1.]))


def test_the_patches_meet_along_the_stem_and_the_keel():
    starboard, port = bow_two_patch()
    for points in ([[0., t] for t in np.linspace(0., 1., 201)], [[t, 0.] for t in np.linspace(0., 1., 201)]):
        assert np.array_equal(starboard.evaluate_many(points), port.evaluate_many(points))


def test_the_aft_boundary_is_the_midship_section_at_x_15():
    starboard, _port = bow_two_patch()
    aft = starboard.controls[-1]
    assert all(x == 15. for x, _y, _z in aft)
    bottom, side = aft[:4], aft[3:]
    assert all(z == 0. for _x, _y, z in bottom) and bottom[0][1] == 0. and bottom[-1][1] == 5.5   # keel to chine
    assert all(y == 5.5 for _x, y, _z in side) and side[-1][2] == 3.                              # chine to above the deck
    assert starboard.knots_v.count(.5) == 3                                                        # the chine crease


# ------------------------------------------------------------------ P3b probe: the faces below the deck
# These record the current behaviour (refusals included); they change when the stem-adjacency design lands.

def test_the_deck_meets_the_stem_at_one_exact_point_on_both_patches():
    from anygeometry.bow_example import build_bow_faces_below_deck
    model, faces, edges = build_bow_faces_below_deck()
    starboard_end = model.edges[edges["starboard"]["deck"]].curve.evaluate(np.array([1.]))
    port_end = model.edges[edges["port"]["deck"]].curve.evaluate(np.array([1.]))
    assert np.array_equal(starboard_end, port_end) and starboard_end[0][1] == 0.


def test_one_bow_face_audits_clean():
    from anygeometry.bow_example import build_bow_faces_below_deck
    from anygeometry.nurbs_section_faces import validate_nurbs_section_face
    model, faces, _edges = build_bow_faces_below_deck()
    assert validate_nurbs_section_face(model, faces["starboard"]) == []
    assert validate_nurbs_section_face(model, faces["port"]) == []
    single, _faces, _edges = build_bow_faces_below_deck(sides=("starboard",))
    assert single.strict_audit().clean


def test_a_shared_stem_curve_needs_the_also_on_proof():
    import pytest
    from anygeometry import GeometryError
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve
    starboard, port = bow_two_patch()
    with pytest.raises(GeometryError, match="also_on requires an exact shared-boundary proof"):
        CurveOnSurface(starboard, NurbsCurve(1, (0., 0., 1., 1.), ((0., 1.), (0., 0.))), (port,))


def test_the_two_bow_faces_are_not_classified_across_the_stem():
    from anygeometry.bow_example import build_bow_faces_below_deck
    model, faces, edges = build_bow_faces_below_deck()
    report = model.strict_audit()
    assert not report.clean
    codes = {issue.code.value for issue in report.issues}
    assert codes == {"unclassified_candidate", "unverified_classification"}
    pairs = {tuple(sorted((entity.kind, entity.id) for entity in issue.entities))
             for issue in report.issues if issue.code.value == "unclassified_candidate"}
    s, p = faces["starboard"], faces["port"]
    assert (("face", p), ("face", s)) in pairs or (("face", s), ("face", p)) in pairs
    duplicates = {tuple(sorted((edges["starboard"][k], edges["port"][k]))) for k in ("stem", "keel")}
    assert duplicates <= {tuple(sorted(e[1] for e in pair)) for pair in pairs if all(e[0] == "edge" for e in pair)}


# ------------------------------------------------------------------ P3c-P3e probe findings (current behaviour)

def test_the_deck_bounded_by_both_patches_sections_validates():
    from anygeometry import Plane
    from anygeometry.bow_example import build_bow_faces_below_deck
    from anygeometry.section_faces import validate_plane_section_face
    model, _faces, edges = build_bow_faces_below_deck(make_faces=False)
    aft = {side: model.edges[edges[side]["deck"]].start for side in edges}
    line = model.add_line(aft["starboard"], aft["port"])
    deck = model.add_face([edges["starboard"]["deck"], edges["port"]["deck"], line],
                          surface=Plane((0., 0., 2.5), (1., 0., 0.), (0., 1., 0.)))
    assert validate_plane_section_face(model, deck) == []


def test_a_plane_face_bounded_by_iso_curves_in_its_plane_is_refused():
    from anygeometry import Plane
    from anygeometry.bow_example import build_bow_faces_below_deck
    from anygeometry.section_faces import validate_plane_section_face
    model, _faces, edges = build_bow_faces_below_deck(make_faces=False)
    aft = {side: model.edges[edges[side]["deck"]].start for side in edges}
    line = model.add_line(aft["starboard"], aft["port"])
    bulkhead = model.add_face([edges["starboard"]["aft"], edges["port"]["aft"], line],
                              surface=Plane((15., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    problems = validate_plane_section_face(model, bulkhead)
    assert len(problems) == 2 and all("only straight and plane x NURBS section edges" in p for p in problems)


import pytest


_SECTION_PLANES = {
    "deck": ((0., 0., 2.5), (1., 0., 0.), (0., 1., 0.)),
    "waterline": ((0., 0., 1.2), (1., 0., 0.), (0., 1., 0.)),
    "frame": ((17., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
    "longitudinal": ((0., 2., 0.), (1., 0., 0.), (0., 0., 1.)),      # across the chine
    "bulkhead": ((15., 0., 0.), (0., 1., 0.), (0., 0., 1.)),         # contains the aft boundary
    "centreline": ((0., 0., 0.), (1., 0., 0.), (0., 0., 1.)),        # contains the stem and keel
    "side": ((0., 5.5, 0.), (1., 0., 0.), (0., 0., 1.)),             # aft side boundary (G1: tangent)
    "bottom": ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)),            # aft bottom boundary (G1: tangent)
}
_BOX_EDGE = "the section runs along a box edge"
# expected outcome per (plane, join), absent = certified; a string = the exact refusal message; a chart
# point = a singular box (smaller than 1e-3) containing that chart corner
_SECTION_OUTCOMES = {
    ("bulkhead", "G0"): _BOX_EDGE, ("bulkhead", "G1"): _BOX_EDGE,
    ("centreline", "G0"): (0., 0.), ("centreline", "G1"): (0., 0.),   # the forefoot corner
    ("side", "G0"): _BOX_EDGE, ("side", "G1"): (1., 1.),
    ("bottom", "G0"): _BOX_EDGE, ("bottom", "G1"): (1., .5),
}


@pytest.mark.parametrize("join", ["G0", "G1"])
@pytest.mark.parametrize("name", list(_SECTION_PLANES))
def test_which_planes_section_the_bow(join, name):
    import re
    from anygeometry import Plane
    from anygeometry.bow_example import bow_starboard_surface
    from anygeometry.plane_nurbs import PlaneNurbsNotCertified, plane_nurbs_section
    plane, surface = Plane(*_SECTION_PLANES[name]), bow_starboard_surface(join)
    expected = _SECTION_OUTCOMES.get((name, join))
    if expected is None:
        assert plane_nurbs_section(plane, surface).paths
        return
    with pytest.raises(PlaneNurbsNotCertified) as refusal:
        plane_nurbs_section(plane, surface)
    message = str(refusal.value)
    if isinstance(expected, str):
        assert message == expected
    else:
        found = re.fullmatch(r"singular box at depth \d+: u \(([^,]+), ([^)]+)\), v \(([^,]+), ([^)]+)\)", message)
        assert found, message
        u0, u1, v0, v1 = map(float, found.groups())
        assert u0 <= expected[0] <= u1 and v0 <= expected[1] <= v1 and u1 - u0 < 1e-3 and v1 - v0 < 1e-3


def test_the_g1_variant_leaves_the_midship_tangentially():
    from anygeometry.bow_example import bow_starboard_surface
    surface = bow_starboard_surface("G1")
    # exactly: the row next to X = 15 differs from the aft row in x only, so S_u is along -x there
    assert [(y, z) for _x, y, z in surface.controls[2]] == [(y, z) for _x, y, z in surface.controls[3]]
    su, _sv = surface.derivatives_many(np.array([[1., v] for v in np.linspace(0., 1., 41)]))
    assert np.all(su[:, 0] < -1.) and np.abs(su[:, 1:]).max() <= 1e-15 * np.abs(su[:, 0]).max()


def test_the_halves_meet_bluntly_at_the_stem_tangentially_at_the_keel_and_in_a_cusp_at_the_forefoot():
    """Review of the probe: the interior angle between the halves (from outward normals) is blunt
    along the stem, 180 degrees towards X = 15 along the keel, and 0 at the forefoot."""
    starboard, port = bow_two_patch()

    def interior_angles(points):
        a, b = starboard.derivatives_many(points)
        c, d = port.derivatives_many(points)
        outward_s, outward_p = np.cross(a, b), -np.cross(c, d)      # mirroring reverses the port orientation
        cos = np.einsum("ij,ij->i", outward_s, outward_p) / (
            np.linalg.norm(outward_s, axis=1) * np.linalg.norm(outward_p, axis=1))
        return 180. - np.degrees(np.arccos(np.clip(cos, -1., 1.)))

    stem = interior_angles(np.array([[0., v] for v in np.linspace(.05, 1., 20)]))
    keel = interior_angles(np.array([[u, 0.] for u in np.linspace(.8, 1., 5)]))
    foot = interior_angles(np.array([[0., v] for v in (0., .0025, .005, .01, .02, .04)]))
    assert 110. < stem.min() and stem.max() < 177.                    # blunt along most of the stem
    assert keel.min() > 179.                                          # the flat bottom: a tangent seam
    assert foot[0] < 1e-6                                             # tangent plane = centreline plane
    # (review of the probe) the angle falls continuously to 0 towards the forefoot: no positive bound
    assert np.all(np.diff(foot) > 0.) and abs(foot[3] - 37.5) < 2. and abs(foot[4] - 68.3) < 2.


def test_the_centreline_plane_separates_the_halves_exactly():
    """The symmetry certificate for the audit across the stem (report, response 2): starboard
    controls have y >= 0, zero only in the stem row and the keel column, so (polynomial weights)
    y > 0 inside the patch and y = 0 exactly on the shared stem and keel curves."""
    starboard, _port = bow_two_patch()
    controls = np.asarray(starboard.controls)
    assert starboard.weights is None
    assert np.all(controls[..., 1] >= 0.)
    zero = controls[..., 1] == 0.
    assert np.all(zero[0, :]) and np.all(zero[:, 0])                  # stem row, keel column
    assert not np.any(zero[1:, 1:])                                   # positive everywhere else


def _closed_models():
    """(full bow with consistently oriented loops, starboard half closed by the centreline plane)."""
    from anygeometry import Plane
    from anygeometry.b1_example import _consistent_loops
    from anygeometry.bow_example import build_bow_faces_below_deck
    starboard, port = bow_two_patch()
    deck_plane = Plane((0., 0., 2.5), (1., 0., 0.), (0., 1., 0.))
    bulkhead_plane = Plane((15., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    model, _faces, edges = build_bow_faces_below_deck(make_faces=False)
    s, p = edges["starboard"], edges["port"]
    line = model.add_line(model.edges[s["deck"]].start, model.edges[p["deck"]].start)
    loops = _consistent_loops(model, [list(s.values()), list(p.values()), [s["deck"], p["deck"], line],
                                      [s["aft"], p["aft"], line]])
    full = (model, [model.add_face_from_loop(loop, surface=support)
                    for loop, support in zip(loops, (starboard, port, deck_plane, bulkhead_plane))], edges)
    half, _faces, edges = build_bow_faces_below_deck(sides=("starboard",), make_faces=False)
    s = edges["starboard"]
    centre_aft = half.add_point(15., 0., 2.5)
    deck_centre = half.add_line(half.edges[s["deck"]].end, centre_aft)
    deck_aft = half.add_line(centre_aft, half.edges[s["deck"]].start)
    bulkhead_centre = half.add_line(half.edges[s["keel"]].end, centre_aft)
    loops = _consistent_loops(half, [list(s.values()), [s["deck"], deck_centre, deck_aft],
                                     [s["aft"], deck_aft, bulkhead_centre],
                                     [s["stem"], s["keel"], bulkhead_centre, deck_centre]])
    supports = (starboard, deck_plane, bulkhead_plane, Plane((0., 0., 0.), (1., 0., 0.), (0., 0., 1.)))
    return full, (half, [half.add_face_from_loop(loop, surface=sp) for loop, sp in zip(loops, supports)])


def test_the_full_bow_fails_to_close_only_on_the_duplicate_stem_and_keel_edges():
    from anygeometry.shell_volume import closed_shell
    (model, faces, edges), _half = _closed_models()
    duplicates = sorted(edges[side][k] for side in edges for k in ("stem", "keel"))
    assert closed_shell(model, faces) == [f"edge {edge} bounds 1 faces of the shell (needs exactly 2)"
                                          for edge in duplicates]


def test_the_starboard_half_closes_but_its_hydrostatics_refuse_the_in_plane_iso_curves():
    from anygeometry import GeometryError
    from anygeometry.hydrostatics import hydrostatics
    from anygeometry.shell_volume import closed_shell
    _full, (half, faces) = _closed_models()
    assert closed_shell(half, faces) == []
    with pytest.raises(GeometryError, match="only straight and section edges bound plane faces"):
        hydrostatics(half, faces, (17., 2., 1.))


def test_a_port_face_cannot_use_the_starboard_stem_edge():
    from anygeometry import GeometryError
    from anygeometry.bow_example import build_bow_faces_below_deck
    model, _faces, edges = build_bow_faces_below_deck(make_faces=False)
    p, s = edges["port"], edges["starboard"]
    _starboard, port = bow_two_patch()
    with pytest.raises(GeometryError, match="is not an exact chart curve on the face's support"):
        model.add_face([p["keel"], p["aft"], p["deck"], s["stem"]], surface=port)


def test_equal_boundary_rows_prove_a_shared_boundary_only_at_a_clamped_end():
    """Review round 2 of the probe: restricted to u_range = (0.25, 1), both patches keep equal stem
    rows, knots and ranges, yet their chart boundaries u = 0 are mirror curves, not one curve."""
    from anygeometry.nurbs import NurbsSurface
    starboard, port = bow_two_patch()

    def cropped(surface):
        return NurbsSurface(surface.degree_u, surface.degree_v, surface.knots_u, surface.knots_v,
                            surface.controls, surface.weights, (.25, 1.), surface.v_range)
    s, p = cropped(starboard), cropped(port)
    assert s.controls[0] == p.controls[0] and (s.u_range, s.v_range) == (p.u_range, p.v_range)
    assert s.u_range[0] != s.knots_u[0]                                   # not a clamped end
    ends = s.evaluate_many(np.array([[0., .5]])), p.evaluate_many(np.array([[0., .5]]))
    assert ends[0][0][1] > 0. and ends[1][0][1] == -ends[0][0][1]          # mirror points, not shared
    # at the clamped end the same rows are the shared curve
    assert np.array_equal(starboard.evaluate_many(np.array([[0., .5]])), port.evaluate_many(np.array([[0., .5]])))
