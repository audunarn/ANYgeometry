"""Programme step 3, P3a: the two-patch bow (owner Q2: two patches meeting at the stem)."""
from __future__ import annotations

import math

import numpy as np

from anygeometry.bow_example import bow_two_patch


def _curve_definition(curve):
    return curve.degree, tuple(curve.knots), tuple(map(tuple, curve.controls)), curve.weights


def test_the_stem_and_keel_iso_curves_of_both_patches_are_the_same_exact_curves():
    starboard, port = bow_two_patch()
    for fixed, value in (("u", 0.), ("v", 0.)):              # stem (u = 0) and keel (v = 0)
        assert _curve_definition(starboard.iso_curve(fixed, value)) == _curve_definition(port.iso_curve(fixed, value))
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


@pytest.mark.parametrize("join", ["G0", "G1"])
@pytest.mark.parametrize("origin, u_vector, v_vector, certified", [
    ((0., 0., 2.5), (1., 0., 0.), (0., 1., 0.), True),          # deck
    ((0., 0., 1.2), (1., 0., 0.), (0., 1., 0.), True),          # waterline
    ((17., 0., 0.), (0., 1., 0.), (0., 0., 1.), True),          # frame
    ((0., 2., 0.), (1., 0., 0.), (0., 0., 1.), True),           # longitudinal across the chine
    ((15., 0., 0.), (0., 1., 0.), (0., 0., 1.), False),         # bulkhead: contains the aft boundary
    ((0., 0., 0.), (1., 0., 0.), (0., 0., 1.), False),          # centreline: contains the stem and keel
    ((0., 5.5, 0.), (1., 0., 0.), (0., 0., 1.), False),         # flat side: aft boundary (G1: tangent)
    ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), False),          # flat bottom: aft boundary (G1: tangent)
])
def test_which_planes_section_the_bow(join, origin, u_vector, v_vector, certified):
    from anygeometry import Plane
    from anygeometry.bow_example import bow_starboard_surface
    from anygeometry.plane_nurbs import PlaneNurbsNotCertified, plane_nurbs_section
    plane, surface = Plane(origin, u_vector, v_vector), bow_starboard_surface(join)
    if certified:
        assert plane_nurbs_section(plane, surface).paths
    else:
        with pytest.raises(PlaneNurbsNotCertified, match="runs along a box edge|singular box"):
            plane_nurbs_section(plane, surface)


def test_the_g1_variant_leaves_the_midship_tangentially():
    from anygeometry.bow_example import bow_starboard_surface
    surface = bow_starboard_surface("G1")
    # exactly: the row next to X = 15 differs from the aft row in x only, so S_u is along -x there
    assert [(y, z) for _x, y, z in surface.controls[2]] == [(y, z) for _x, y, z in surface.controls[3]]
    su, _sv = surface.derivatives_many(np.array([[1., v] for v in np.linspace(0., 1., 41)]))
    assert np.all(su[:, 0] < -1.) and np.abs(su[:, 1:]).max() <= 1e-15 * np.abs(su[:, 0]).max()
