from itertools import combinations
import numpy as np
import pytest
from anygeometry.intersection_candidates import swept_face_pairs


def exhaustive(bounds, padding, **kwargs):
    return tuple((a, b) for a, b in combinations(sorted(bounds), 2)
                 if np.all(bounds[a][1] >= bounds[b][0] - padding)
                 and np.all(bounds[b][1] >= bounds[a][0] - padding))


@pytest.mark.parametrize('seed', range(12))
def test_sweep_matches_exhaustive_order_and_boundary_comparisons(seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(90, 3))
    widths = rng.uniform(0, 1.5, size=(90, 3))
    bounds = {3*i+7: (c-w, c+w) for i, (c, w) in enumerate(zip(centers, widths))}
    for padding in (0., 1e-10, .2):
        assert swept_face_pairs(bounds, padding) == exhaustive(bounds, padding)


def test_float_expiry_matches_original_subtraction_at_extreme_coordinates():
    bounds = {1: (np.array([1e16, 0., 0.]), np.array([1e16, 1., 1.])),
              2: (np.array([1e16+2, 0., 0.]), np.array([1e16+4, 1., 1.])),
              3: (np.array([1e16+4, 0., 0.]), np.array([1e16+6, 1., 1.]))}
    assert swept_face_pairs(bounds, 1.) == exhaustive(bounds, 1.)


def test_sweep_polls_while_constructing_candidates():
    bounds = {i: (np.zeros(3), np.ones(3)) for i in range(20)}
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        swept_face_pairs(bounds, 0., poll=cancel)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_boxes_are_refused_instead_of_silently_pruned(bad):
    from anygeometry import GeometryError
    bounds = {1: (np.array([bad,0.,0.]), np.ones(3)),
              2: (np.zeros(3), np.ones(3))}
    with pytest.raises(GeometryError, match='finite'):
        swept_face_pairs(bounds, 0.)


def test_exact_touch_and_equal_lows_are_retained():
    bounds = {1: (np.zeros(3), np.ones(3)),
              2: (np.array([1.,0.,0.]), np.array([2.,1.,1.])),
              3: (np.zeros(3), np.array([0.,1.,1.]))}
    assert swept_face_pairs(bounds, 0.) == exhaustive(bounds, 0.)


def test_skipped_pair_and_tail_charges_match_original_budget():
    from anygeometry import GeometryModel, plan_intersections, GeometryError, to_dict
    from anygeometry.batch_intersections import IntersectionBatchPolicy
    model = GeometryModel()
    for x in (0., 10., 20., 30.):
        model.add_plate(model.add_points(((x,0,0),(x+1,0,0),(x+1,1,0),(x,1,0))))
    before = to_dict(model)
    accepted = plan_intersections(model, tuple(model.faces), policy=IntersectionBatchPolicy(max_predicates=6))
    with pytest.raises(GeometryError, match='budget'):
        plan_intersections(model, tuple(model.faces), policy=IntersectionBatchPolicy(max_predicates=5))
    reference = plan_intersections(model, tuple(model.faces), policy='connect')
    assert accepted.content_checksum == reference.content_checksum
    assert to_dict(model) == before


def plates():
    from anygeometry import GeometryModel
    model = GeometryModel()
    faces = []
    for points in (((0,0,0),(5,0,0),(5,5,0),(0,5,0)),
                   ((2,-1,-1),(2,6,-1),(2,6,1),(2,-1,1)),
                   ((0,2,-1),(5,2,-1),(5,2,1),(0,2,1))):
        faces.append(model.add_plate(model.add_points(points)))
    return model, faces


def test_public_plans_match_reference_and_preserve_budgets(monkeypatch):
    from anygeometry import plan_intersections, to_dict, GeometryError
    from anygeometry.batch_intersections import IntersectionBatchPolicy
    import anygeometry.intersection_candidates as candidates
    model, faces = plates()
    before = to_dict(model)
    swept = plan_intersections(model, faces, policy='connect')
    monkeypatch.setattr(candidates, 'swept_face_pairs', exhaustive)
    reference = plan_intersections(model, faces, policy='connect')
    assert swept == reference
    for factory in (swept_face_pairs, exhaustive):
        monkeypatch.setattr(candidates, 'swept_face_pairs', factory)
        with pytest.raises(GeometryError, match='budget'):
            plan_intersections(model, faces, policy=IntersectionBatchPolicy(max_predicates=1))
        with pytest.raises(GeometryError, match='maximum_candidate_pairs=1'):
            plan_intersections(model, faces, policy=IntersectionBatchPolicy(max_candidate_pairs=1))
        with pytest.raises(GeometryError, match='cancelled'):
            plan_intersections(model, faces, policy=IntersectionBatchPolicy(cancellation_check=lambda: True))
        assert to_dict(model) == before
