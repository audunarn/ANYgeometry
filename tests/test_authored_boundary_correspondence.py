"""Exact boundary-chain evidence is narrower than partition or mesh acceptance."""
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace

import pytest

from anygeometry import (GeometryError, GeometryModel, apply_intersections,
    clone_prepared_geometry, from_dict, plan_intersections, to_dict)
from anygeometry.authored_boundary_correspondence import (
    _tile, query_prepared_authored_boundary_correspondence as query,
    validate_prepared_authored_boundary_correspondence_binding as validate)


def fixture(cubic=False):
    model = GeometryModel()
    if cubic:
        points = model.add_points(((0, 0, 0), (1, 2, 0), (2, -1, 0), (3, 1, 0)))
        edge = model.add_spline(points[0], points[1:-1], points[-1])
        model.extrude((edge,), (.25, 0, 1.5))
        original = min(model.faces)
        cutter = ((1.5, -3, -1), (1.5, 3, -1), (1.5, 3, 3), (1.5, -3, 3))
    else:
        original = model.add_plate(model.add_points(((-1, -1, 0), (1, -1, 0),
                                                     (1, 1, 0), (-1, 1, 0))))
        cutter = ((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1))
    model.add_plate(model.add_points(cutter))
    source = to_dict(model)
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    apply_intersections(model, plan, policy='connect')
    return model, original, source


@pytest.mark.parametrize('cubic', (False, True))
def test_complete_descendant_exterior_preserves_original_loops_and_exact_tiling(cubic):
    model, face, original = fixture(cubic)
    committed = to_dict(model)
    result = query(model, face)
    expected = next(row for row in original['faces'] if row['id'] == face)['loop']
    assert [(root, direction) for root, direction, _ in result.exterior_loops[0]] == [tuple(row) for row in expected]
    assert all(children for _, _, children in result.exterior_loops[0])
    assert result.interior_incidence
    exterior = {child for loop in result.exterior_loops for _, _, children in loop for child in children}
    assert exterior.isdisjoint(edge for edge, _ in result.interior_incidence)
    assert len(result.descendants) > 1
    assert result.authored_definition.source_checksum == original['checksum']['value']
    validate(model, result)
    assert to_dict(model) == committed
    assert not hasattr(result, 'material_coverage_qualified')
    assert query(clone_prepared_geometry(model), face) == result


def test_every_original_hole_loop_is_present_after_fragmentation():
    from test_authored_face_definitions import authored, prepare
    model, face, _, _ = authored(with_cutter=True)
    original = to_dict(model)
    prepare(model)
    result = query(model, face)
    expected = next(row for row in original['faces'] if row['id'] == face)
    assert len(result.exterior_loops) == 3
    assert tuple(tuple((edge, direction) for edge, direction, _ in loop)
                 for loop in result.exterior_loops) == tuple(
        tuple(map(tuple, loop)) for loop in (expected['loop'], *expected['holes']))


def row(edge, first, second):
    return SimpleNamespace(edge_id=edge, interval=((F(first).numerator, F(first).denominator),
                                                (F(second).numerator, F(second).denominator)))


def test_rational_tiling_does_not_weld_distinct_stations_or_require_display_precision():
    seam = F(1, 2)
    near = seam+F(1, 2**100)
    assert float(seam) == float(near)
    assert _tile(((row(2, seam, 1), True), (row(1, 0, seam), True)), True) == (1, 2)
    assert _tile(((row(1, 0, seam), False), (row(2, seam, 1), False)), False) == (2, 1)
    with pytest.raises(GeometryError, match='gap'):
        _tile(((row(1, 0, seam), True), (row(2, near, 1), True)), True)
    with pytest.raises(GeometryError, match='duplicate'):
        _tile(((row(1, 0, near), True), (row(2, seam, 1), True)), True)
    with pytest.raises(GeometryError, match='reversed'):
        _tile(((row(1, 0, 1), False),), True)
    with pytest.raises(GeometryError, match='incomplete'):
        _tile(((row(1, 0, seam), True),), True)


def test_forged_scope_stale_source_and_unqualified_copy_refuse():
    model, face, _ = fixture()
    result = query(model, face)
    with pytest.raises(GeometryError, match='definition binding'):
        validate(model, replace(result, descendants=result.descendants[:-1]))
    with pytest.raises(GeometryError):
        query(from_dict(to_dict(model)), face)
    model.add_point(8, 9, 10)
    with pytest.raises(GeometryError, match='stale'):
        validate(model, result)


def test_final_callback_source_change_rejects_without_query_edits():
    model, face, _ = fixture()
    original = to_dict(model)
    def cancelled(_):
        return True
    with pytest.raises(GeometryError, match='cancelled'):
        query(model, face, cancellation_check=cancelled)
    assert to_dict(model) == original


@pytest.mark.parametrize('mutation', ('face_receipt', 'document'))
def test_false_returning_last_callback_cannot_invalidate_either_owner_receipt(monkeypatch, mutation):
    from anygeometry import EntityRef
    import anygeometry.authored_boundary_correspondence as module
    model, face, _ = fixture()
    original = to_dict(model)
    last = False
    done = False
    edge_validation = module.validate_prepared_edge_subcurve_preimages_binding
    def final_validation(*args, **kwargs):
        nonlocal last
        last = True
        return edge_validation(*args, **kwargs)
    monkeypatch.setattr(module, 'validate_prepared_edge_subcurve_preimages_binding', final_validation)
    def callback(_):
        nonlocal done
        if last and not done:
            done = True
            if mutation == 'face_receipt':
                del model._prepared_face_preimages_receipt
            else:
                model.tag(EntityRef('edge', min(model.edges)), 'callback-edit')
        return False
    with pytest.raises(GeometryError):
        query(model, face, cancellation_check=callback)
    assert done
    if mutation == 'face_receipt':
        assert to_dict(model) == original
