"""Analytic geometry checks; no hydrodynamic solver or pressure qualification."""
from dataclasses import replace

import numpy as np
from anygeometry.floating_overlap import polygon_plane
import pytest

from anygeometry import (MappingLimits, PanelProjectionError, PanelProjectionCancelled,
                         prepare_panel_projection, validate_panel_geometry)


def square(x0=0., x1=1., z=0.):
    return np.array([[x0, 0, z], [x1, 0, z], [x1, 1, z], [x0, 1, z]], float)


def limits(**kw):
    return MappingLimits(**dict(gap_m=.2, max_angle=.3, coverage_tolerance=1e-10,
                                convergence_tolerance=.15, max_depth=6, max_triangles=20000, **kw))


def areas(result):
    t = result.triangles
    return np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1)/2


def sheet_projection(source,target,**kw):
    return prepare_panel_projection(source,target,limits=limits(),source_regions=['dipole_jump']*len(source),
        target_regions=['dipole_jump']*len(target),two_sided_regions=('dipole_jump',),
        source_coverage_regions=('dipole_jump',),**kw)


def test_dipole_two_sided_exact_and_source_coverage():
    r=sheet_projection([square()[::-1]],[square()])
    assert areas(r).sum()==pytest.approx(1)
    assert r.diagnostics['duplicate_target_coverage_checked']
    assert r.diagnostics['uncovered_source_area_m2']==pytest.approx(0.)
    with pytest.raises(PanelProjectionError):
        prepare_panel_projection([square()[::-1]],[square()],limits=limits())
    with pytest.raises(PanelProjectionError):sheet_projection([square()],[square(0,.5)])


@pytest.mark.parametrize('offset',[0.,.02])
def test_duplicate_dipole_targets_refused_across_seams(offset):
    targets=[square(),square(z=offset)[::-1]]
    with pytest.raises(PanelProjectionError,match='duplicate|ambiguous'):
        sheet_projection([square(0,.3),square(.3,1)],targets)


def test_distinct_parallel_dipole_supports_are_not_collapsed():
    source=[square(z=0.),square(z=.05)[::-1]]
    target=[square(z=0.)[::-1],square(z=.05)]
    r=sheet_projection(source,target)
    assert areas(r).sum()==pytest.approx(2.)
    assert set(r.source_indices)=={0,1}


def test_colocated_body_and_dipole_region_separation():
    r=prepare_panel_projection([square(),square()[::-1]],[square(),square()],limits=limits(),
        source_regions=['body_pressure','dipole_jump'],target_regions=['body_pressure','dipole_jump'],
        two_sided_regions=('dipole_jump',),source_coverage_regions=('dipole_jump',))
    assert set(zip(r.source_indices,r.target_indices))=={(0,0),(1,1)}


def test_exact_subdivision_area_and_first_moment_bind_pressure_boundary():
    result = prepare_panel_projection([square(0, .3), square(.3, 1)], [square()], limits=limits())
    a = areas(result)
    for i, expected_area, expected_moment in [(0, .3, .045), (1, .7, .455)]:
        pick = result.source_indices == i
        assert a[pick].sum() == pytest.approx(expected_area, abs=1e-14)
        assert (a[pick]*result.triangles[pick].mean(axis=1)[:, 0]).sum() == pytest.approx(expected_moment, abs=1e-14)
    assert result.diagnostics['exact_targets'] == 1
    assert result.diagnostics['unresolved_area_bound_m2'] == 0
    result.validate()


def test_offset_projection_keeps_target_surface_and_area_no_normalization():
    result = prepare_panel_projection([square(z=.1)], [square()], limits=limits())
    assert np.all(result.triangles[:, :, 2] == 0)
    assert areas(result).sum() == pytest.approx(1.)
    assert result.diagnostics['projected_targets'] == 1
    assert result.diagnostics['max_gap_m'] == pytest.approx(.1)


def test_independent_facets_capture_both_sides_with_bounded_assignment_area():
    left, right = square(0, .5), square(.5, 1)
    left[:, 2] = .02+.1*(.5-left[:, 0])
    right[:, 2] = .02+.1*(right[:, 0]-.5)
    result = prepare_panel_projection([left, right], [square()], limits=limits())
    assert set(result.source_indices) == {0, 1}
    assert areas(result).sum() == pytest.approx(1.)
    assert np.max(np.abs(result.triangles[:, :, 2])) == 0
    uncertain = result.diagnostics['unresolved_area_bound_m2']
    assert uncertain <= .15
    assert abs(areas(result)[result.source_indices == 0].sum()-.5) <= uncertain
    assert result.diagnostics['max_depth_used'] > 0


def test_hashes_and_immutability_and_geometry_identity():
    source = square()
    result = prepare_panel_projection([source], [square()], limits=limits())
    identity = result.content_sha256
    source[:] = 42
    result.validate()
    assert identity == result.content_sha256
    for a in (result.triangles, result.source_indices, result.target_indices):
        with pytest.raises(ValueError): a.setflags(write=True)
    with pytest.raises(TypeError): result.diagnostics['triangle_count'] = 0
    with pytest.raises(PanelProjectionError, match='identity'):
        replace(result, target_sha256='changed').validate()
    altered = prepare_panel_projection([square(z=.1)], [square()], limits=limits())
    assert altered.source_sha256 != result.source_sha256
    assert altered.content_sha256 != result.content_sha256


def test_body_region_restriction_prevents_ambiguous_cross_body_transfer():
    result = prepare_panel_projection([square(z=.1), square(z=-.1)], [square()], limits=limits(),
                                      source_regions=['A/hull', 'B/hull'], target_regions=['B/hull'])
    assert set(result.source_indices) == {1}
    with pytest.raises(PanelProjectionError, match='region'):
        prepare_panel_projection([square()], [square()], limits=limits(), source_regions=['A'], target_regions=['B'])


@pytest.mark.parametrize('source, match', [
    ([square()[::-1]], 'max_angle'),
    ([square(z=.3)], 'gap_m'),
    ([square(z=.1), square(z=-.1)], 'ambiguous'),
    ([square(), square()], 'ambiguous'),
    ([square(0, .4), square(.6, 1)], 'coverage'),
])
def test_refusal(source, match):
    with pytest.raises(PanelProjectionError, match=match):
        prepare_panel_projection(source, [square()], limits=limits())


def test_depth_and_triangle_budgets_fail_closed():
    with pytest.raises(PanelProjectionError, match='max_depth'):
        prepare_panel_projection([square(0, .4, .1), square(.4, 1, .1)], [square()],
                                 limits=replace(limits(), max_depth=0, convergence_tolerance=0))
    with pytest.raises(PanelProjectionError, match='max_triangles'):
        prepare_panel_projection([square()], [square()], limits=replace(limits(), max_triangles=1))


def test_cancellation_before_and_during_preparation():
    with pytest.raises(PanelProjectionCancelled):
        prepare_panel_projection([square()], [square()], limits=limits(), cancel=lambda: True)
    calls = []
    def cancel():
        calls.append(1)
        return len(calls) > 4
    with pytest.raises(PanelProjectionCancelled):
        prepare_panel_projection([square(0, .4, .1), square(.4, 1, .1)], [square()], limits=limits(), cancel=cancel)


def test_planar_transfer_translation_covariance():
    shift = np.array([20., -30., 9.])
    original = prepare_panel_projection([square(0, .3), square(.3, 1)], [square()], limits=limits())
    moved = prepare_panel_projection([square(0, .3)+shift, square(.3, 1)+shift], [square()+shift], limits=limits())
    np.testing.assert_array_equal(original.source_indices, moved.source_indices)
    np.testing.assert_allclose(original.triangles, moved.triangles-shift, atol=1e-14)


def test_projection_rigid_rotation_covariance_and_region_hash_binding():
    q = np.array([[1., 0., 0.], [0., .8, -.6], [0., .6, .8]])
    original = prepare_panel_projection([square(z=.1)], [square()], limits=limits())
    rotated = prepare_panel_projection([square(z=.1)@q.T], [square()@q.T], limits=limits())
    np.testing.assert_allclose(rotated.triangles, original.triangles@q.T, atol=1e-14)
    bound = prepare_panel_projection([square(z=.1)], [square()], limits=limits(),
                                    source_regions=['body/hull'], target_regions=['body/hull'])
    assert bound.source_sha256 != original.source_sha256
    assert bound.target_sha256 != original.target_sha256
    parallel = prepare_panel_projection([square(z=.1)@q.T], [square()@q.T],
                                        limits=replace(limits(), max_angle=0.))
    np.testing.assert_allclose(parallel.triangles, original.triangles@q.T, atol=1e-14)


def test_small_interior_source_patch_cannot_hide_from_corner_only_sampling():
    # A closer island is invisible at target corners; Lipschitz certification
    # forces refinement and bounds even a pressure boundary inside one triangle.
    island = np.array([[.15, .55, .02], [.35, .55, .02], [.35, .75, .02], [.15, .75, .02]])
    result = prepare_panel_projection([square(z=.1), island], [square()],
                                     limits=replace(limits(), convergence_tolerance=.1, max_depth=8))
    assert 1 in result.source_indices
    assert result.diagnostics['max_depth_used'] > 0
    assert result.diagnostics['unresolved_area_bound_m2'] <= .1


@pytest.mark.parametrize('change', [dict(gap_m=-1), dict(max_angle=np.pi), dict(max_depth=True),
                                  dict(convergence_tolerance=1), dict(coverage_tolerance=np.nan)])
def test_invalid_limits(change):
    with pytest.raises(PanelProjectionError): replace(limits(), **change)


def test_invalid_polygon_and_region_inventory():
    bad = square(); bad[2, 2] = .1
    for polygon in (bad, np.zeros((3, 3)), [[0, 0, 0], [1, 0, 0], [np.nan, 1, 0]]):
        with pytest.raises(PanelProjectionError):
            prepare_panel_projection([polygon], [square()], limits=limits())
    with pytest.raises(PanelProjectionError, match='together'):
        prepare_panel_projection([square()], [square()], limits=limits(), source_regions=['A'])


def test_public_panel_geometry_nonrectangular_centroid_and_immutable_winding():
    # Trapezoid = unit square plus a right triangle of area 1/2.
    p = np.array([[0., 0., 2.], [2., 0., 2.], [1., 1., 2.], [0., 1., 2.]])
    polygons, normals, centers, area = validate_panel_geometry([p, p[::-1]])
    np.testing.assert_allclose(centers, [[7/9, 4/9, 2.]]*2, atol=1e-14)
    np.testing.assert_allclose(area, [1.5, 1.5], atol=1e-14)
    np.testing.assert_allclose(normals, [[0, 0, 1], [0, 0, -1]], atol=1e-14)
    p[:] = 99
    assert polygons[0][0, 0] == 0
    for a in (*polygons, normals, centers, area):
        with pytest.raises(ValueError): a.setflags(write=True)


@pytest.mark.parametrize('polygon', [
    [[0, 0, 0], [2, 0, 0], [.5, .5, 0], [0, 2, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 1, .01], [0, 1, 0]],
])
def test_public_panel_geometry_refuses_nonconvex_or_warped_facets(polygon):
    with pytest.raises(PanelProjectionError): validate_panel_geometry([polygon])


def test_missing_offset_sheet_strip_cannot_use_nearest_edge_as_coverage():
    source=[square()]
    for lo in (0.,.25,.5,.75):
        source.append(np.array([[1.,lo,.01],[1.1,lo,.01],[1.1,lo+.25,.01],[1.,lo+.25,.01]]))
    # The four omitted strips could carry self-equilibrated [+1,-1,-1,+1]
    # pressures: source support must be checked independently of resultant.
    with pytest.raises(PanelProjectionError,match='interior support'):
        sheet_projection(source,[square()])


def test_translated_adjacent_facets_keep_exact_clips_nondegenerate():
    from scipy.spatial.transform import Rotation
    rotation=Rotation.from_euler('xyz',[.17,.43,.29]).as_matrix()
    source=[square(0,.3),square(.3,1)]
    targets=[square(0,.2),square(.2,.7),square(.7,1)]
    move=lambda p:p@rotation.T+np.array([12.345,-8.765,3.123])
    result=sheet_projection([move(p) for p in source],[move(p) for p in targets])
    assert result.diagnostics['mapped_area_m2']==pytest.approx(1.)
    assert np.all(areas(result)>0)


def test_dipole_precision_overlap_is_deducted_from_coverage_allowance():
    source=[square()];targets=[square(0,.5),square(.5-1e-14,1)]
    result=sheet_projection(source,targets)
    assert result.diagnostics['source_coverage_precision_area_bound_m2']>0
    with pytest.raises(PanelProjectionError,match='interior support'):
        prepare_panel_projection(source,targets,limits=replace(limits(),coverage_tolerance=1e-16),
            source_regions=['dipole'],target_regions=['dipole']*2,two_sided_regions=('dipole',),source_coverage_regions=('dipole',))


def test_noncoplanar_sheet_seam_duplicate_bridge_refused():
    left=square(0,1)
    right=square(1,2);right[:,2]=.1*(right[:,0]-1)
    bridge=square(.8,1.2,z=.01)[::-1]
    # Two genuine facets plus a second midsurface bridging their angled seam.
    with pytest.raises(PanelProjectionError,match='duplicate'):
        sheet_projection([left,right],[left,right,bridge])


def test_nonfinite_generated_fragment_is_refused():
    from anygeometry.panel_pressure_projection import _support_area
    polygon=square();polygon[0,0]=np.inf
    with pytest.raises(PanelProjectionError,match='nonfinite'):_support_area(polygon,np.array([0.,0.,1.]))


@pytest.mark.parametrize('coverage',[0.,1e-18])
def test_exact_sheet_match_cannot_certify_sub_roundoff_coverage(coverage):
    from scipy.spatial.transform import Rotation
    polygon=square()@Rotation.from_euler('xyz',[.17,.43,.29]).as_matrix().T+[12.345,-8.765,3.123]
    kwargs=dict(source_regions=['sheet'],target_regions=['sheet'],two_sided_regions=('sheet',),source_coverage_regions=('sheet',))
    with pytest.raises(PanelProjectionError,match='interior support'):
        prepare_panel_projection([polygon],[polygon],limits=replace(limits(),coverage_tolerance=coverage),**kwargs)
    admitted=prepare_panel_projection([polygon],[polygon],limits=limits(),**kwargs)
    assert admitted.diagnostics['source_coverage_precision_area_bound_m2']>0


# --- coordinate-magnitude planarity floor and sharp-edge tie assignment -------------------

def tilted_facet(origin, a, b, scale=2., quad=True, ulps=0, seed=0):
    """Planar facet of ~``scale`` extent at ``origin``; every coordinate then moves by up to ``ulps`` ulp,
    the rounding of a few rigid-transform operations (the refined L01 bank is off a plane by up to ~20 eps*max|p|)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    uv = [(0, 0), (1, 0), (1, 1), (0, 1)] if quad else [(0, 0), (1, 0), (0, 1)]
    p = np.array([np.asarray(origin, float)+scale*(u*a+v*b) for u, v in uv])
    k = np.random.default_rng(seed).integers(-ulps, ulps+1, size=p.shape) if ulps else np.zeros(p.shape, int)
    for _ in range(int(np.abs(k).max())):
        step = np.sign(k)
        p = np.where(step > 0, np.nextafter(p, np.inf), np.where(step < 0, np.nextafter(p, -np.inf), p))
        k = k-step
    return p


FAR = (46.1234567891234, 3.3141592653589, -1.7182818284590)


@pytest.mark.parametrize('quad', [True, False])
def test_planarity_gate_admits_rounding_of_far_from_origin_coordinates(quad):
    rng = np.random.default_rng(7)
    for _ in range(200):
        a, b = rng.normal(size=3), rng.normal(size=3)
        b -= a*(a@b)/(a@a)
        polygon = tilted_facet(FAR, a, b, quad=quad, ulps=12, seed=_)
        validate_panel_geometry([polygon])  # coordinate rounding (~1e-14 m at 46 m) is not a physics defect


def test_planarity_gate_still_refuses_real_warp_far_from_origin():
    polygon = tilted_facet(FAR, [1, .3, .2], [-.3, 1, .1])
    n = np.cross(polygon[1]-polygon[0], polygon[3]-polygon[0])
    polygon[2] += 1e-9*n/np.linalg.norm(n)
    with pytest.raises(PanelProjectionError, match='planar'):
        validate_panel_geometry([polygon])


def test_planarity_floor_scales_with_coordinate_magnitude_only():
    polygon = tilted_facet((0., 0., 0.), [1, .3, .2], [-.3, 1, .1])
    n = np.cross(polygon[1]-polygon[0], polygon[3]-polygon[0])
    polygon[2] += 1e-12*n/np.linalg.norm(n)  # far above rounding at the origin
    with pytest.raises(PanelProjectionError, match='planar'):
        validate_panel_geometry([polygon])


def sharp_edge_sources():
    deck = np.array([[-1, 0, 1], [0, 0, 1], [0, 1, 1], [-1, 1, 1]], float)       # normal +z, index 0
    bow = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], float)          # normal +x, index 1
    return [deck, bow]


def test_sharp_edge_vertex_tie_prefers_compatible_source_panel():
    target = np.array([[.005, 0, 0], [.005, 1, 0], [.005, 1, 1], [.005, 0, 1]], float)  # normal +x
    r = prepare_panel_projection(sharp_edge_sources(), [target], limits=limits())
    assert r.diagnostics['max_angle_rad'] < 1e-6
    assert set(np.unique(r.source_indices)) == {1}


def test_sharp_edge_tie_still_refuses_when_no_tied_panel_is_compatible():
    target = np.array([[.005, 0, .9], [.005, 1, .9], [.005, 1, 1], [.005, 0, 1]], float)
    deck, _ = sharp_edge_sources()  # the only source is perpendicular to the target
    with pytest.raises(PanelProjectionError, match='max_angle'):
        prepare_panel_projection([deck], [target], limits=limits())


# --- planarity gate vs the local-frame overlap gate; float32; tie-rule boundaries ----------

def test_far_translated_quad_with_real_warp_is_refused_at_the_first_gate():
    M, h = 1e9, 1e-6
    quad = np.array([[M, 0, 0], [M+1, 0, 0], [M+1, 1, h], [M, 1, 0]], float)
    with pytest.raises(PanelProjectionError, match='planar'):
        validate_panel_geometry([quad])
    with pytest.raises(PanelProjectionError, match='planar'):  # was an uncaught GeometryError from the overlap call
        prepare_panel_projection([quad], [quad], limits=limits())


def test_rounding_floor_is_bounded_relative_to_facet_size():
    quad = tilted_facet((1e6, 1e6, 1e6), [1, .3, .2], [-.3, 1, .1], scale=2e-3)
    n = np.cross(quad[1]-quad[0], quad[3]-quad[0])
    quad[2] += 5e-9*n/np.linalg.norm(n)   # 2.5e-6 of the extent; 32*eps*max|p| alone (7e-9) would admit it
    with pytest.raises(PanelProjectionError, match='planar'):
        validate_panel_geometry([quad])


def test_admitted_far_quad_projects_without_geometry_error():
    quad = tilted_facet(FAR, [1, .3, .2], [-.3, 1, .1], ulps=12, seed=3)
    r = prepare_panel_projection([quad], [quad], limits=limits())
    assert areas(r).sum() == pytest.approx(polygon_area_of(quad))


def polygon_area_of(p):
    return float(np.linalg.norm(0.5*np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)))


def test_float32_input_is_judged_by_the_strict_gate_not_only_the_legacy_one():
    # warp 1e-8 m on a 46 m-offset quad: inside the legacy 1e-9*max|p| gate, far outside the strict float64 gate
    quad = np.array([[46, 0, 0], [48, 0, 0], [48, 2, 1e-8], [46, 2, 0]], np.float32)
    assert quad.dtype == np.float32
    polygon_plane(quad.astype(float))          # the upstream legacy gate admits it
    with pytest.raises(PanelProjectionError, match='planar'):
        validate_panel_geometry([quad])


def test_large_triangle_is_planar_by_construction_and_projects():
    tri = np.array([[0, 0, 0], [1e8, 3e7, 2e7], [-3e7, 1e8, 1e7]], float)
    validate_panel_geometry([tri])
    r = prepare_panel_projection([tri], [tri], limits=limits())
    assert areas(r).sum() == pytest.approx(polygon_area_of(tri), rel=1e-9)


def test_relative_cap_refuses_small_facet_far_from_origin():
    quad = tilted_facet((1e6, 1e6, 1e6), [1, .3, .2], [-.3, 1, .1], scale=1e-4)
    n = np.cross(quad[1]-quad[0], quad[3]-quad[0])
    quad[2] += 1.5e-9*n/np.linalg.norm(n)    # residual ~7e-10 m: below 32*eps*max|p| (7e-9), above the cap
    with pytest.raises(PanelProjectionError, match='planar'):
        validate_panel_geometry([quad])


def test_admitted_implies_accepted_by_overlap_property():
    rng = np.random.default_rng(11)
    admitted = 0
    for k in range(300):
        scale = 10.**rng.uniform(-3, 2)
        offset = rng.normal(size=3)*10.**rng.uniform(0, 9)
        a, b = rng.normal(size=3), rng.normal(size=3)
        b -= a*(a@b)/(a@a)
        quad = tilted_facet(offset, a, b, scale=scale, ulps=int(rng.integers(0, 40)), seed=k)
        quad = quad + rng.normal(size=3)*0.0
        try:
            validate_panel_geometry([quad])
        except PanelProjectionError:
            continue
        admitted += 1
        try:
            prepare_panel_projection([quad], [quad], limits=limits())
        except PanelProjectionError:
            pass            # typed refusals (coverage etc.) are fine; an uncaught GeometryError is not
    assert admitted > 30


def edge_target():
    return np.array([[.005, 0, .9], [.005, 1, .9], [.005, 1, 1], [.005, 0, 1]], float)  # normal +x


def edge_centre_target():
    """Triangle (normal +x) whose centre (.005, .5, 1) lies exactly on the deck/bow edge line."""
    return np.array([[.005, .2, .94], [.005, .8, .94], [.005, .5, 1.12]], float)


def bow_panel(x, z0=0., z1=1.):
    return np.array([[x, 0, z0], [x, 1, z0], [x, 1, z1], [x, 0, z1]], float)  # normal +x


def test_tie_rule_does_not_replace_a_strictly_nearer_incompatible_panel():
    deck, _ = sharp_edge_sources()
    with pytest.raises(PanelProjectionError, match='max_angle'):
        prepare_panel_projection([deck, bow_panel(-.02)], [edge_target()], limits=limits())


def test_tie_rule_does_not_use_a_compatible_panel_just_beyond_the_tie_tolerance():
    deck, _ = sharp_edge_sources()
    with pytest.raises(PanelProjectionError, match='max_angle'):
        prepare_panel_projection([deck, bow_panel(-1e-9)], [edge_centre_target()], limits=limits())


def test_tie_rule_still_reports_ambiguity_after_replacement():
    deck, _ = sharp_edge_sources()
    sources = [deck, bow_panel(0.), np.array([[.01, 0, 1], [.01, 1, 1], [.01, 1, 2], [.01, 0, 2]], float)]
    with pytest.raises(PanelProjectionError, match='ambiguous'):
        prepare_panel_projection(sources, [edge_centre_target()], limits=limits())


# --- final round: legacy path frozen, local-frame centroid, private admitted path, mutants ---------------
import anygeometry.floating_overlap as fo
from anygeometry.errors import GeometryError
from fractions import Fraction

# Verbatim copy of the ORIGINAL floating_overlap.polygon_plane / overlap_polygon (git HEAD before this work).
def frozen_polygon_plane(poly):
    """(unit outward normal, centroid, area); refuses a non-planar polygon (relative 1e-9) or zero area."""
    p = np.asarray(poly, dtype=np.float64)
    av = fo.area_vector(p)
    a = float(np.linalg.norm(av))
    if a <= 0.0:
        raise fo.GeometryError("degenerate polygon")
    n = av / a
    scale = max(1.0, float(np.abs(p).max()))
    if np.abs((p - p[0]) @ n).max() > 1e-9 * scale:
        raise fo.GeometryError("floating overlap requires planar polygons")
    # area-weighted centroid by fan triangles
    c = np.zeros(3); tot = 0.0
    for k in range(1, len(p) - 1):
        t = p[[0, k, k + 1]]; ta = 0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0]))
        c += ta * t.mean(axis=0); tot += ta
    return n, c / tot, a




def frozen_overlap_polygon(source, target, *, distance_tol: float = 1e-6, normal_cos_tol: float = 1.0 - 1e-6):
    """Physical overlap polygon (in the source plane) of two coplanar same-orientation convex polygons, or None."""
    ns, cs, _ = frozen_polygon_plane(source)
    nt, ct, _ = frozen_polygon_plane(target)
    if float(ns @ nt) < normal_cos_tol:
        return None
    t = np.asarray(target, dtype=np.float64)
    s = np.asarray(source, dtype=np.float64)
    if np.abs((t - cs) @ ns).max() > distance_tol * max(1.0, float(np.abs(s).max())):
        return None
    q = t - np.outer((t - cs) @ ns, ns)
    for i in range(len(s)):
        e = s[(i + 1) % len(s)] - s[i]
        m = np.cross(e, ns); ml = np.linalg.norm(m)
        if ml == 0.0:
            continue
        m = m / ml
        q = fo.clip_halfspace(q, m, float(m @ s[i]))     # outward edge normal for a CCW (outward-normal) polygon
        if q is None:
            return None
    return q




SKINNY = np.array([[0.15588125739342668, 0.19376654712868652, -0.010209089088953016],
                   [-0.19049694608667933, -0.03540470372994975, 0.16300372202437743],
                   [0.026649335892430737, 0.10826393471914507, 0.0544157178216305]])


def test_legacy_mode_still_gates_triangles_like_the_original():
    with pytest.raises(GeometryError, match='requires planar'):
        polygon_plane(SKINNY)
    with pytest.raises(GeometryError, match='requires planar'):
        frozen_polygon_plane(SKINNY)
    polygon_plane(SKINNY, planarity='strict')          # strict: a triangle is planar by construction
    # mutation: gating triangles only for len>3 in legacy mode makes the first assertion fail


def _outcome(fn, *a, **k):
    try:
        return ('ok', fn(*a, **k))
    except GeometryError as e:
        return ('refused', str(e))


def _same(x, y):
    if x[0] != y[0]:
        return False
    if x[0] == 'refused':
        return x[1] == y[1]
    u, v = x[1], y[1]
    if isinstance(u, tuple):
        return all(np.array_equal(np.asarray(i), np.asarray(j)) for i, j in zip(u, v))
    return (u is None and v is None) or (u is not None and v is not None and np.array_equal(u, v))


def test_legacy_mode_matches_frozen_original_on_random_polygons():
    rng = np.random.default_rng(2024)
    refused = ok = 0
    for k in range(600):
        scale = 10.**rng.uniform(-3, 2)
        offset = rng.normal(size=3)*10.**rng.uniform(-1, 4)
        a, b = rng.normal(size=3), rng.normal(size=3)
        b -= a*(a@b)/(a@a)
        quad = k % 3 != 0
        poly = tilted_facet(offset, a, b, scale=scale, quad=quad, ulps=int(rng.integers(0, 6)), seed=k)
        if k % 5 == 0:
            poly = poly+rng.normal(size=poly.shape)*10.**rng.uniform(-12, -7)    # real warp around the gate
        new, old = _outcome(polygon_plane, poly), _outcome(frozen_polygon_plane, poly)
        assert _same(new, old)
        refused += new[0] == 'refused'; ok += new[0] == 'ok'
        other = poly+rng.normal(size=3)*1e-4*scale if k % 2 else poly[::-1]
        assert _same(_outcome(fo.overlap_polygon, poly, other), _outcome(frozen_overlap_polygon, poly, other))
        assert _same(_outcome(fo.overlap_polygon, poly, poly), _outcome(frozen_overlap_polygon, poly, poly))
    assert refused > 5 and ok > 100                   # the comparison exercises both outcomes
    for tri in (SKINNY, SKINNY[::-1]):
        assert _same(_outcome(polygon_plane, tri), _outcome(frozen_polygon_plane, tri))


def _exact_quad(m, du, dv, plane):
    """Vertices are dyadic rationals, hence exact floats; coplanarity is verified in rational arithmetic."""
    (a, b, c) = plane
    uv = [(0, 0), (du, 0), (du, dv), (0, dv)]
    q = np.array([[m+u, w, a*(m+u)+b*w+c] for u, w in uv], float)
    ex = [[Fraction(float(x)) for x in row] for row in q]
    sub = lambda i, j: [ex[i][k]-ex[j][k] for k in range(3)]
    u1, u3, u2 = sub(1, 0), sub(3, 0), sub(2, 0)
    n = [u1[1]*u3[2]-u1[2]*u3[1], u1[2]*u3[0]-u1[0]*u3[2], u1[0]*u3[1]-u1[1]*u3[0]]
    assert sum(n[i]*u2[i] for i in range(3)) == 0      # exactly coplanar
    return q


@pytest.mark.parametrize('m, du, dv', [(1e9, 1., 1.), (1e9+.1, .7, 1.3), (2.**30, 2., 3.), (1e8, 1e7, 5e6)])
def test_exactly_coplanar_quad_is_admitted_at_any_offset(m, du, dv):
    quad = _exact_quad(m, du, dv, (.5, .25, 3.))
    validate_panel_geometry([quad])
    # mutation: centroid taken in the original frame (p - p.mean()) leaves a residual ~1e-7 at 1e9 and refuses it


def test_no_public_entry_point_accepts_a_polygon_both_gates_reject():
    quad = tilted_facet(FAR, [1, .3, .2], [-.3, 1, .1])
    n = np.cross(quad[1]-quad[0], quad[3]-quad[0])
    quad[2] += 1e-4*n/np.linalg.norm(n)
    for mode in ('legacy', 'strict'):
        with pytest.raises(GeometryError):
            polygon_plane(quad, planarity=mode)
        with pytest.raises(GeometryError):
            fo.overlap_polygon(quad, quad, planarity=mode)
    with pytest.raises(ValueError):
        polygon_plane(quad, planarity='certified')
    with pytest.raises(ValueError):
        fo.overlap_polygon(quad, quad, planarity='certified')
    with pytest.raises(PanelProjectionError):
        validate_panel_geometry([quad])
    with pytest.raises(PanelProjectionError):
        prepare_panel_projection([quad], [quad], limits=limits())
    assert not any(not name.startswith('_') and 'admitted' in name for name in dir(fo))


def test_replacement_itself_produces_an_ambiguity():
    """The ambiguity must arise AT the replaced vertex. Target vertex (.005, 0, 1) lies on the deck/bow edge: the deck
    (index 0, normal +z) is incompatible and ties with bow1 (x=0, compatible), which replaces it. bow2 (x=.01, y<=0, z>=1,
    corner at (.01, 0, 1)) is compatible and exactly as close (.005) with a different closest point, so the
    post-replacement check must raise 'ambiguous'. Nothing else in the target is tied, so a mutant that skips the
    check after a replacement maps the target without error."""
    deck, _ = sharp_edge_sources()
    bow2 = np.array([[.01, 0, 1], [.01, 1, 1], [.01, 1, 2], [.01, 0, 2]], float)     # normal +x, above the bow, 1 cm ahead
    sources = [deck, bow_panel(0.), bow2]
    with pytest.raises(PanelProjectionError, match='ambiguous'):
        prepare_panel_projection(sources, [edge_centre_target()], limits=limits())
    # control: without bow2 the same target maps through the replacement
    ok = prepare_panel_projection(sources[:2], [edge_centre_target()], limits=replace(limits(), convergence_tolerance=.6))   # the edge strip never certifies
    assert set(np.unique(ok.source_indices)) == {1}
    assert ok.diagnostics['max_gap_m'] > .1          # the far, still unresolved strip above the edge counts: it is published


# --- coordinate_scale validation (strict gate must not be bypassable through a NaN tolerance) -----------
WARPED = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1e-6], [0, 1, 0]], float)
BAD_SCALES = [float('nan'), float('inf'), -float('inf'), 0., -1., -1e-300, True, False, np.float64('nan'),
              np.float32('inf'), np.float64(0.), np.int64(0), np.int64(-3), '46', [46.], np.array([46.])]


@pytest.mark.parametrize('scale', BAD_SCALES, ids=repr)
def test_invalid_coordinate_scale_is_refused_at_every_public_entry_point(scale):
    for mode in ('legacy', 'strict'):
        with pytest.raises(GeometryError, match='scale'):
            polygon_plane(WARPED, planarity=mode, coordinate_scale=scale)
        with pytest.raises(GeometryError, match='scale'):
            fo.overlap_polygon(WARPED, WARPED, planarity=mode, source_scale=scale)
        with pytest.raises(GeometryError, match='scale'):
            fo.overlap_polygon(WARPED, WARPED, planarity=mode, target_scale=scale)
    with pytest.raises(GeometryError, match='scale'):
        fo.planarity_tolerance(WARPED, scale)


def test_valid_scale_types_are_accepted_and_warped_quad_stays_refused():
    flat = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    for scale in (46, 46., np.float64(46.), np.float32(46.), np.int64(46)):
        polygon_plane(flat, planarity='strict', coordinate_scale=scale)
        with pytest.raises(GeometryError, match='planar'):
            polygon_plane(WARPED, planarity='strict', coordinate_scale=scale)
        with pytest.raises(GeometryError, match='planar'):
            fo.overlap_polygon(WARPED, WARPED, planarity='strict', source_scale=scale, target_scale=scale)


def test_warped_quad_is_refused_by_the_projection_entry_points():
    with pytest.raises(PanelProjectionError, match='planar'):
        validate_panel_geometry([WARPED])
    with pytest.raises(PanelProjectionError, match='planar'):
        prepare_panel_projection([WARPED], [WARPED], limits=limits())


# ---- Bow-corner and centreline-mirror assignment contract -------------------------------
def wide(**kw):
    return MappingLimits(**dict(gap_m=2., max_angle=1.2, coverage_tolerance=1e-10,
                                convergence_tolerance=.15, max_depth=8, max_triangles=200000, **kw))


def per_source_area(result, n):
    t = result.triangles
    a = np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1)/2
    return np.array([a[result.source_indices == i].sum() for i in range(n)])


BOW_R = [[0, 0, -1], [0, 1, -1], [0, 1, 0], [0, 0, 0]]                       # normal +x
KEEL_R = [[0, 0, -1], [-2, 0, -1.5], [-2, 1, -1.5], [0, 1, -1]]              # normal (.2425, 0, -.97)
BOW_L = [[0, 0, -1], [0, 0, 0], [0, -1, 0], [0, -1, -1]]
KEEL_L = [[0, 0, -1], [0, -1, -1], [-2, -1, -1.5], [-2, 0, -1.5]]


def corner_target(x, z0, z1, y0=.2, y1=.8):
    return np.array([[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]], float)  # normal +x


@pytest.mark.parametrize('order', [(0, 1), (1, 0)])
def test_corner_point_outside_a_convex_edge_uses_the_adjacent_compatible_panel(order):
    panels = [np.array(BOW_R, float), np.array(KEEL_R, float)]
    sources = [panels[i] for i in order]
    r = prepare_panel_projection(sources, [corner_target(.003, -1.02, -.5)], limits=wide())
    assert set(np.unique(r.source_indices)) == {order.index(0)}            # always the bow face
    assert areas(r).sum() == pytest.approx(.6*.52)
    assert r.diagnostics['max_angle_rad'] < 1e-6


def test_corner_rule_is_not_an_index_tie_break():
    keel = np.array(KEEL_R, float)
    far_keel_copy = keel + [0, 5, 0]                # an unrelated incompatible panel changes nothing
    a = prepare_panel_projection([np.array(BOW_R, float), keel], [corner_target(.003, -1.02, -.5)], limits=wide())
    b = prepare_panel_projection([far_keel_copy, np.array(BOW_R, float), keel], [corner_target(.003, -1.02, -.5)],
                                 limits=wide())
    assert np.array_equal(a.triangles, b.triangles) and set(b.source_indices) == {1}


def test_corner_rule_refuses_a_point_far_from_the_corner():
    sources = [np.array(BOW_R, float), np.array(KEEL_R, float)]
    # Inside the hull, 0.5 m behind the bow face: the keel is the surface here, the bow face is far.
    with pytest.raises(PanelProjectionError, match='max_angle'):
        prepare_panel_projection(sources, [corner_target(-.5, -1.2, -1.05)], limits=wide())


def corner_centre_target():
    """Triangle (normal +x) whose centre (.003, .5, -1.02) is the sample 2 cm outside the bow's bottom corner."""
    return np.array([[.003, .2, -1.1], [.003, .8, -1.1], [.003, .5, -.86]], float)


def test_corner_rule_refuses_when_the_nearer_panel_does_not_touch_the_compatible_one():
    keel = np.array(KEEL_R, float)
    keel[[0, 3], 0] = .01                           # keel starts 1 cm ahead of the bow face: they do not touch
    with pytest.raises(PanelProjectionError, match='max_angle'):
        prepare_panel_projection([np.array(BOW_R, float), keel], [corner_centre_target()], limits=wide())


def test_corner_rule_refuses_when_no_adjacent_panel_is_compatible():
    keel = np.array(KEEL_R, float)
    with pytest.raises(PanelProjectionError, match='max_angle'):
        prepare_panel_projection([keel], [corner_target(.003, -1.02, -.5)], limits=wide())


VALLEY_R = [[0, 0, 0], [1, 0, 0], [1, 1, .3], [0, 1, .3]]                    # normal (0, -.287, .958)
VALLEY_L = [[0, 0, 0], [0, -1, .3], [1, -1, .3], [1, 0, 0]]


def flat_target(y0, y1, z=.02, x0=.2, x1=.8):
    return np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)    # normal +z


def valley_limits():
    return MappingLimits(gap_m=.2, max_angle=.5, coverage_tolerance=1e-10, convergence_tolerance=.15,
                         max_depth=8, max_triangles=200000)


def test_equal_distance_tie_between_distinct_closest_points_is_ambiguous():
    s1 = np.array([[1, -.5, 0], [2, -.5, 0], [2, .5, 0], [1, .5, 0]], float)
    s2 = np.array([[-2, -.9, 0], [-1, -.9, 0], [-1, .9, 0], [-2, .9, 0]], float)     # wider: no mirror image
    t = np.array([[-.3, .1, .05], [-.3, -.1, .05], [.6, 0., .05]], float)                # centre (0, 0) on the tie
    lim = MappingLimits(gap_m=1.2, max_angle=.5, coverage_tolerance=1e-10, convergence_tolerance=.15,
                        max_depth=8, max_triangles=200000)
    with pytest.raises(PanelProjectionError, match='ambiguous'):
        prepare_panel_projection([s1, s2], [t], limits=lim)


def test_corner_rule_at_a_bow_apex_with_two_bow_faces():
    sources = [np.array(p, float) for p in (BOW_R, BOW_L, KEEL_R, KEEL_L)]
    r = prepare_panel_projection(sources, [corner_target(.003, -1.02, -.5, y0=-.5, y1=.5)], limits=wide())
    got = per_source_area(r, 4)
    assert got[2] == 0 and got[3] == 0
    assert got[0] == pytest.approx(got[1], rel=1e-9) and got.sum() == pytest.approx(.52, rel=1e-9)
    assert r.diagnostics['max_angle_rad'] < 1e-6


# ---- direct checks of the corner decision helper ------------------------------
def _rows_and_source(polys, distances):
    source = [(np.array(p, float), np.array([0., 0., 1.]), 1.) for p in polys]
    values = [(i, d, np.zeros(3), False) for i, d in distances]
    return source, values


def test_corner_helper_follows_a_chain_of_touching_panels_but_not_a_detached_one():
    from anygeometry import panel_pressure_projection as ppp
    bow = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    a1 = [[1, 0, 0], [1.05, 0, 0], [1.05, 1, 0], [1, 1, 0]]    # shares a local edge with the bow
    a2 = [[1.05, 0, 0], [2, 0, 0], [2, 1, 0], [1.05, 1, 0]]    # shares the next local edge with a1
    far = [[5, 0, 0], [6, 0, 0], [6, 1, 0], [5, 1, 0]]
    comp = lambda row: row[0] == 0
    source, values = _rows_and_source([bow, a1, a2], [(2, .01), (1, .02), (0, .03)])
    assert ppp._corner_row(np.array([1.025, .5, .02]), values, source, comp, 1e-12)[0] == 0
    source, values = _rows_and_source([bow, a1, far], [(2, .01), (1, .02), (0, .03)])
    assert ppp._corner_row(np.zeros(3), values, source, comp, 1e-12) is None
    source, values = _rows_and_source([bow, a1, a2], [(2, .01), (1, .02), (0, .3)])       # too far from the junction
    assert ppp._corner_row(np.zeros(3), values, source, comp, 1e-12) is None


def test_corner_helper_returns_nothing_when_no_panel_is_compatible():
    from anygeometry import panel_pressure_projection as ppp
    source, values = _rows_and_source([[[0, 0, 0], [1, 0, 0], [1, 1, 0]]], [(0, .01)])
    assert ppp._corner_row(np.zeros(3), values, source, lambda row: False, 1e-12) is None


def test_remote_single_shared_vertex_cannot_override_nearest_panel_angle():
    flat = np.array([[0., 0., 0.], [10., 0., 0.], [10., 1., 0.]])
    steep = np.array([[0., 0., 0.], [10., .01, .02], [10., .5, 1.]])
    target = np.array([[8.99, .019, .04], [9.01, .019, .04], [9., .022, .04]])
    with pytest.raises(PanelProjectionError, match='max_angle'):
        prepare_panel_projection([flat, steep], [target], limits=replace(limits(), gap_m=.1, max_angle=.1))


# ---- slice (a): target vertex samples supply distances only and decide nothing ------------------
def test_vertex_on_the_bisector_of_non_mirror_panels_does_not_block_the_mapping():
    # The planes are exact mirror images but the outlines differ, so the two panels tie at every
    # point of y = 0 yet are not mirror panels. The vertices on y = 0 are never published labels.
    right = np.array(VALLEY_R, float)
    left = np.array([[-.5, 0, 0], [-.5, -1, .3], [1.5, -1, .3], [1.5, 0, 0]], float)
    tri = np.array([[.3, 0., .02], [.7, 0., .02], [.5, .5, .02]], float)
    for order in [(0, 1), (1, 0)]:
        sources = [[right, left][i] for i in order]
        r = prepare_panel_projection(sources, [tri], limits=valley_limits())
        assert set(np.unique(r.source_indices)) == {order.index(0)}
        assert areas(r).sum() == pytest.approx(.4*.5/2)


def test_diagnostics_describe_the_published_labels_only():
    # Flat target 3 cm above and tilted 0.1 rad against one flat source panel: gap and angle are those of the
    # published label; the vertex samples contribute distances to the bound, not their own angle.
    tilt = .1
    t = np.array([[0, 0, .03], [1, 0, .03], [1, 1, .03+np.tan(tilt)], [0, 1, .03+np.tan(tilt)]], float)
    r = prepare_panel_projection([square()], [t], limits=MappingLimits(gap_m=.5, max_angle=.5, coverage_tolerance=1e-10,
                                 convergence_tolerance=.15, max_depth=6, max_triangles=20000))
    assert r.diagnostics['max_angle_rad'] == pytest.approx(tilt, rel=1e-9)
    assert r.diagnostics['max_gap_m'] == pytest.approx(.03+np.tan(tilt), rel=1e-9)


def test_a_target_vertex_beyond_gap_m_is_still_refused_even_when_the_centre_is_within_it():
    t = np.array([[0, 0, 0], [1, 0, 0], [0, 1, .5]], float)       # centre .17 above the panel, one vertex .5
    with pytest.raises(PanelProjectionError, match='gap_m|max_depth'):
        prepare_panel_projection([square()], [t], limits=MappingLimits(gap_m=.2, max_angle=1., coverage_tolerance=1e-10,
                                 convergence_tolerance=.15, max_depth=6, max_triangles=20000))


# ---- coplanar seam: found in the L01 bank (targets 133 342 473 483 623 830 961) ----------------------------
SEAM_SHIFT = np.array([20., 0., 0.])        # coordinates of size 20 make eps ~6e-13, as in the real bank


def seam_sources():
    left = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float) + SEAM_SHIFT
    right = np.array([[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], float) + SEAM_SHIFT
    return left, right


@pytest.mark.parametrize('order', [(0, 1), (1, 0)])
def test_coplanar_neighbours_tie_to_second_order_resolves_to_the_panel_containing_the_foot(order):
    # The foot of a sample 4.5e-8 inside the left panel: distances to both panels agree to ~1e-13
    # (second order), but the closest points differ by 4.5e-8. Same surface, no ambiguity.
    panels = seam_sources()
    sources = [panels[i] for i in order]
    t = np.array([[1-4.5e-8, .2, .007], [1.5, .2, .007], [1.5, .8, .007], [1-4.5e-8, .8, .007]], float) + SEAM_SHIFT
    lim = MappingLimits(gap_m=.05, max_angle=.3, coverage_tolerance=1e-10, convergence_tolerance=.15,
                        max_depth=8, max_triangles=200000)
    r = prepare_panel_projection(sources, [t], limits=lim)
    assert areas(r).sum() == pytest.approx(polygon_area_of(t))
    assert set(np.unique(r.source_indices)) <= {0, 1}


def test_coplanar_seam_rule_does_not_cover_two_interior_overlapping_sheets():
    left, _ = seam_sources()
    t = np.array([[.2, .2, .007], [.8, .2, .007], [.8, .8, .007], [.2, .8, .007]], float) + SEAM_SHIFT
    lim = MappingLimits(gap_m=.05, max_angle=.3, coverage_tolerance=1e-10, convergence_tolerance=.15,
                        max_depth=8, max_triangles=200000)
    with pytest.raises(PanelProjectionError, match='ambiguous'):
        prepare_panel_projection([left, left.copy()], [t], limits=lim)


@pytest.mark.parametrize('tilt', [1e-6, 2e-7])
def test_seam_rule_needs_exactly_coplanar_panels(tilt):
    # 1e-6: the normals already disagree beyond 128 eps; 2e-7: the normals still agree, but the vertices of the
    # 1 m wide panel sit 2e-7 off the other plane (eps ~ 6e-13), so the plane residual alone must refuse the seam.
    left, right = seam_sources()
    right = right.copy()
    right[:, 2] += (right[:, 0]-(1+SEAM_SHIFT[0]))*tilt      # the right panel is tilted about the seam
    x = 1-4.5e-8
    t = np.array([[x-.4, .2, .007], [x+.8, .5, .007], [x-.4, .8, .007]], float) + SEAM_SHIFT   # centre at x
    lim = MappingLimits(gap_m=.05, max_angle=.3, coverage_tolerance=1e-10, convergence_tolerance=.15,
                        max_depth=8, max_triangles=200000)
    with pytest.raises(PanelProjectionError, match='ambiguous'):
        prepare_panel_projection([left, right], [t], limits=lim)

@pytest.mark.parametrize('order', [(0, 1), (1, 0)])
def test_coplanar_neighbours_tied_at_a_triangle_centre_resolve_to_the_panel_containing_the_foot(order):
    # Real-bank class (7 of 976 targets): a refinement-triangle centre whose foot is 4.5e-8 inside the left panel.
    # Distances to both coplanar panels agree to ~1e-13, the closest points differ by 4.5e-8: one surface.
    panels = seam_sources()
    sources = [panels[i] for i in order]
    x = 1-4.5e-8
    t = np.array([[x-.4, .2, .007], [x+.8, .5, .007], [x-.4, .8, .007]], float) + SEAM_SHIFT      # centre at x
    lim = MappingLimits(gap_m=.05, max_angle=.3, coverage_tolerance=1e-10, convergence_tolerance=.15,
                        max_depth=8, max_triangles=200000)
    r = prepare_panel_projection(sources, [t], limits=lim)
    assert areas(r).sum() == pytest.approx(polygon_area_of(t))
    assert set(np.unique(r.source_indices)) == {0, 1}


# ---- REVIEW5 class: one final validation of the published row, on every publication path ------------------
def seam_quad(S, inset, tilt=0., z=.007, y=.5):
    """Quad with two fan triangles: the first has its centre at x = S - inset (the seam tie), the second lies far
    to the left over the panels' interior. Centre of triangle one is (S-inset, y, z) in the target plane."""
    x0 = S-.4
    x1 = 3*(S-inset)-2*x0
    pts = [(x0, y-.3), (x1, y), (x0, y+.3), (S-.7, y)]
    h = lambda x: z-np.tan(tilt)*(x-(S-inset))
    return np.array([[x, yy, h(x)] for x, yy in pts], float)


def test_seam_replacement_that_fails_max_angle_is_refused():
    # A (right, edge-only) and B (left, holds the foot) are 1e-4 wide; B is tilted 1e-9 rad past A, so A is exactly at
    # max_angle while B exceeds it. The distances tie to ~1e-18 (eps ~ 6e-13) and both panels are coplanar within eps.
    # The seam replacement must not publish B. C is a compatible flat panel under the second fan triangle.
    S = 21.
    a = np.array([[S, .4999, 0], [S+1e-4, .4999, 0], [S+1e-4, .5001, 0], [S, .5001, 0]], float)
    b = np.array([[S-1e-4, .4999, 0], [S, .4999, 0], [S, .5001, 0], [S-1e-4, .5001, 0]], float)
    b[:, 2] = 1e-9*(b[:, 0]-S)
    c = np.array([[S-1.2, 0, 0], [S-.3, 0, 0], [S-.3, 1, 0], [S-1.2, 1, 0]], float)
    t = seam_quad(S, 1e-11, tilt=.1)
    lim = MappingLimits(gap_m=3., max_angle=.1, coverage_tolerance=1e-10, convergence_tolerance=.85,
                        max_depth=0, max_triangles=20000)
    for order in [(0, 1, 2), (1, 0, 2)]:
        with pytest.raises(PanelProjectionError, match='max_angle'):
            prepare_panel_projection([[a, b, c][i] for i in order], [t], limits=lim)


def detached_seam_sources(gap):
    left, right = seam_sources()
    return left, right+np.array([gap, 0., 0.])


@pytest.mark.parametrize('order', [(0, 1), (1, 0)])
def test_seam_exemption_needs_the_panels_to_share_a_boundary(order):
    # The connected twin of this fixture is test_coplanar_neighbours_tied_at_a_triangle_centre...; moving the right
    # panel 1e-8 away leaves two detached coplanar sheets: equally close, different closest points -> ambiguous.
    panels = detached_seam_sources(1e-8)
    x = 1-4.5e-8
    t = np.array([[x-.4, .2, .007], [x+.8, .5, .007], [x-.4, .8, .007]], float) + SEAM_SHIFT
    lim = MappingLimits(gap_m=.05, max_angle=.3, coverage_tolerance=1e-10, convergence_tolerance=.15,
                        max_depth=8, max_triangles=200000)
    with pytest.raises(PanelProjectionError, match='ambiguous'):
        prepare_panel_projection([panels[i] for i in order], [t], limits=lim)


def test_exact_clip_publication_reports_gap_and_angle_diagnostics():
    # Unit squares far from the origin: eps ~ 2.8e-5, so a source 1e-5 above the target still takes the exact path.
    x0 = 1e9
    src = square(x0, x0+1, 1e-5)
    tgt = square(x0, x0+1, 0.)
    r = prepare_panel_projection([src], [tgt], limits=limits())
    assert r.diagnostics['exact_targets'] == 1
    assert r.diagnostics['max_gap_m'] == pytest.approx(1e-5, abs=2e-7)
    assert r.diagnostics['max_angle_rad'] == pytest.approx(0., abs=1e-7)


def test_diagnostics_of_a_vertex_nearest_to_another_panel_than_the_published_one():
    # Target flat 5 cm above S1 (x in [0,1]); S2 is a thin strip beyond x = 1 tilted by .2 rad. The target's far
    # vertex is nearest to S2, but every published leaf is labelled S1: the diagnostics are those of S1 only.
    s1 = square(0., 1., 0.)
    s2 = np.array([[1, 0, 0], [1.03, 0, .006], [1.03, 1, .006], [1, 1, 0]], float)
    t = np.array([[0, 0, .05], [1.015, 0, .05], [0, 1, .05]], float)
    lim = MappingLimits(gap_m=.5, max_angle=.5, coverage_tolerance=1e-10, convergence_tolerance=.4,
                        max_depth=6, max_triangles=20000)
    r = prepare_panel_projection([s1, s2], [t], limits=lim)
    assert set(np.unique(r.source_indices)) == {0}
    assert r.diagnostics['max_angle_rad'] == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize('order', [(0, 1), (1, 0)])
def test_floating_point_seam_tie_publishes_the_panel_containing_the_foot(order):
    # The distances to the two coplanar panels are equal to the last bit (difference ~1e-21), the closest points differ
    # by 1e-11. In the (1, 0) order the edge-only (right) panel is first in index order; the label must be the left panel.
    left, right = seam_sources()
    t = seam_quad(21., 1e-11)
    lim = MappingLimits(gap_m=3., max_angle=.3, coverage_tolerance=1e-10, convergence_tolerance=.85,
                        max_depth=0, max_triangles=20000)
    panels = [[right, left][i] for i in order]
    r = prepare_panel_projection(panels, [t], limits=lim)
    assert set(r.source_indices.tolist()) == {order.index(1)}
    assert len(r.source_indices) == 2


# ---- the corner boundary is the SMALLEST diameter in the cluster, in the production path ------------------
BIG_FLAT = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]                  # compatible, diameter 14.1
SMALL_STEEP = [[0, 0, 0], [0, 1.5, 0], [-1.2, 1.5, 2], [-1.2, 0, 2]]         # incompatible, diameter 2.6


def corner_boundary(height):
    lim = MappingLimits(gap_m=.5, max_angle=.3, coverage_tolerance=1e-10, convergence_tolerance=.5,
                        max_depth=4, max_triangles=200000)
    t = np.array([[-.1, .2, height], [.5, .1, height], [.5, 1.4, height]], float)
    return prepare_panel_projection([np.array(BIG_FLAT, float), np.array(SMALL_STEEP, float)], [t], limits=lim)


def test_corner_boundary_uses_the_smallest_diameter_so_a_far_compatible_panel_is_refused():
    # Compatible panel >= .3 away inside the corner: within 0.1 * the big panel's diameter (1.41) but outside
    # 0.1 * the small incompatible panel's (.26).
    with pytest.raises(PanelProjectionError, match='max_angle'):
        corner_boundary(.3)


def test_corner_boundary_control_inside_the_small_diameter_threshold_maps():
    r = corner_boundary(.02)
    assert set(r.source_indices.tolist()) == {0}
    assert r.diagnostics['unresolved_area_bound_m2'] > 0          # the corner leaves were published, not avoided


def test_seam_rule_needs_parallel_normals_even_when_tiny_panels_stay_within_the_plane_tolerance():
    # 1 micron panels tilted by 5e-7 against each other: every vertex is within eps of the other plane (5e-13 < 6e-13)
    # but the normals differ by far more than 128 eps in cosine. Distances tie; the seam exemption must not apply.
    S = 21.
    a = np.array([[S, .4999, 0], [S+1e-6, .4999, 0], [S+1e-6, .5001, 0], [S, .5001, 0]], float)
    b = np.array([[S-1e-6, .4999, 0], [S, .4999, 0], [S, .5001, 0], [S-1e-6, .5001, 0]], float)
    b[:, 2] = 5e-7*(b[:, 0]-S)
    x = S-1e-10                     # 1e-4 above the panels the tilt moves the foot by 5e-11 only: still inside B
    t = np.array([[x-.4, .4, 1e-4], [x+.8, .5, 1e-4], [x-.4, .6, 1e-4]], float)
    lim = MappingLimits(gap_m=.05, max_angle=.3, coverage_tolerance=1e-10, convergence_tolerance=.15,
                        max_depth=8, max_triangles=200000)
    for order in [(0, 1), (1, 0)]:
        with pytest.raises(PanelProjectionError, match='ambiguous'):
            prepare_panel_projection([[a, b][i] for i in order], [t], limits=lim)
