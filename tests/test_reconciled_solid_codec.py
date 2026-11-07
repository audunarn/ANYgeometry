"""Compatible schema-8 records from both reconciliation parents coexist."""
from copy import deepcopy
from fractions import Fraction
import json

import pytest

from anygeometry import GeometryError, GeometryModel, LineRange, from_dict, to_dict
from anygeometry.serialization import _checksum, required_version


def add_line_range(model):
    curve = LineRange((10., 0., 0.), (12., 0., 0.)).subcurve(
        Fraction(1, 7), Fraction(5, 7))
    start, end = model.add_points(curve.evaluate((0., 1.)))
    return model.add_curve(start, end, curve)


@pytest.mark.parametrize("solids,lines", [(False, True), (True, False), (True, True)])
def test_original_and_combined_v8_documents_roundtrip_and_clone(solids, lines):
    model = GeometryModel()
    region = model.add_solid_box((0, 0, 0), (1, 2, 3)) if solids else None
    edge = add_line_range(model) if lines else None
    document = json.loads(json.dumps(to_dict(model)))
    assert document["version"] == required_version(document) == 8
    assert ("solid_regions" in document) == solids
    loaded = from_dict(document)
    assert to_dict(loaded) == document
    for candidate in (loaded, model.clone()):
        if solids:
            snapshot = candidate.solid_region_snapshot(region)
            assert snapshot.volume == pytest.approx(6.)
            assert all(handle.model_id == candidate.model_id for handle in snapshot.face_refs)
        if lines:
            assert candidate.edges[edge].curve == model.edges[edge].curve


@pytest.mark.parametrize("version", [5, 6, 7])
@pytest.mark.parametrize("solids,lines", [(False, True), (True, False), (True, True)])
def test_lower_schema_cannot_silently_drop_either_v8_domain(version, solids, lines):
    model = GeometryModel()
    if solids:
        model.add_solid_box((0, 0, 0), (1, 2, 3))
    if lines:
        add_line_range(model)
    document = to_dict(model)
    document["version"] = version
    document["checksum"] = _checksum(document)
    with pytest.raises(GeometryError):
        from_dict(document)


@pytest.mark.parametrize("corruption", ["wrong_owner", "orientation", "unknown_field", "not_array"])
def test_line_range_does_not_bypass_present_solid_validation(corruption):
    model = GeometryModel()
    model.add_solid_box((0, 0, 0), (1, 2, 3))
    add_line_range(model)
    document = to_dict(model)
    if corruption == "wrong_owner":
        document["solid_regions"][0]["model_id"] = str(GeometryModel().model_id)
    elif corruption == "orientation":
        document["solid_regions"][0]["face_orientations"][0] = False
    elif corruption == "unknown_field":
        document["solid_regions"][0]["unknown"] = True
    else:
        document["solid_regions"] = {}
    document["checksum"] = _checksum(document)
    with pytest.raises(GeometryError):
        from_dict(document)


@pytest.mark.parametrize("solids,lines", [(False, True), (True, False), (True, True)])
def test_required_version_sees_both_domains_in_nested_feature_baselines(solids, lines):
    model = GeometryModel()
    if solids:
        model.add_solid_box((0, 0, 0), (1, 2, 3))
    if lines:
        add_line_range(model)
    model.features.capture_baseline(model)
    model.features.append("geometry.point", parameters={"position": [20., 0., 0.]})
    assert model.regenerate_features().success
    document = to_dict(model)
    # The recursive requirement must follow baseline records, even if a
    # caller embeds the feature history in a different top-level container.
    assert required_version({"features": document["features"]}) == 8
    assert to_dict(from_dict(document)) == document


@pytest.mark.parametrize("removal", ["absent", "empty", "unrelated"])
def test_materialized_prism_feature_requires_its_own_region_even_with_line_ranges(removal):
    model = GeometryModel()
    add_line_range(model)
    model.features.capture_baseline(model)
    feature = model.features.append("geometry.solid_prism", parameters={
        "profile": [[0., 0., 0.], [2., 0., 0.], [2., 3., 0.], [0., 3., 0.]],
        "vector": [0., 0., 1.],
    })
    assert model.regenerate_features().success
    assert model.features.get(feature.feature_id).outputs
    document = deepcopy(to_dict(model))
    if removal == "absent":
        document.pop("solid_regions")
    elif removal == "empty":
        document["solid_regions"] = []
    else:
        document["solid_regions"][0]["provenance"].pop("feature_id")
    document["checksum"] = _checksum(document)
    with pytest.raises(GeometryError, match="solid.*feature|feature.*solid"):
        from_dict(document)


def test_cancelled_loft_preserves_combined_geometry_and_region_snapshot():
    model = GeometryModel()
    region = model.add_solid_box((0, 0, 0), (1, 2, 3))
    first = add_line_range(model)
    a, b = model.add_points(((10., 1., 1.), (12., 1., 1.)))
    second = model.add_line(a, b)
    snapshot = model.solid_region_snapshot(region)
    before = to_dict(model)
    with pytest.raises(GeometryError, match="cancelled"):
        model.loft((first,), (second,), cancellation_check=lambda: True)
    assert to_dict(model) == before
    model.validate_solid_region_snapshot(snapshot)


def test_nurbs_loft_and_solid_survive_joint_feature_staging_and_json():
    from anygeometry.nurbs import NurbsCurve

    model = GeometryModel()
    region = model.add_solid_box((0, 0, 0), (1, 2, 3))
    faces = model.solid_region_snapshot(region).face_refs
    profiles = []
    for y in (10., 12.):
        curve = NurbsCurve(2, (0, 0, 0, 1, 1, 1),
                           ((0, y, 0), (1, y, 1), (2, y, 0)), (1, .7, 1))
        start, end = model.add_points(curve.evaluate((0., 1.)))
        profiles.append(model.add_curve(start, end, curve))
    loft, = model.loft((profiles[0],), (profiles[1],))
    expected = model.face_point(loft, .25, .4)
    model.features.capture_baseline(model)
    model.features.append("geometry.point", parameters={"position": [20., 0., 0.]})
    assert model.regenerate_features().success
    for candidate in (model, from_dict(to_dict(model))):
        assert candidate.solid_region_snapshot(region).face_refs == faces
        assert candidate.solid_region_snapshot(region).volume == pytest.approx(6.)
        assert candidate.face_point(loft, .25, .4) == pytest.approx(expected)


def test_prepared_local_field_binding_preserves_unselected_solid_ownership():
    from anygeometry import clone_prepared_geometry, query_material_surface_regions
    from anygeometry.local_approximation_field import (
        bind_local_approximation_field, query_local_approximation_field)
    from test_prepared_material_curvilinear import square_cells

    model, _, _, _ = square_cells()
    face = next(iter(model.faces))
    region = model.add_solid_box((10, 0, 0), (11, 2, 3))
    field = query_local_approximation_field(
        model, (face,), relative_accuracy=.001, scale_error=.001, policy_floor=1e-8)
    prepared = clone_prepared_geometry(model, new_preparation_epoch=True)
    materials = query_material_surface_regions(prepared, (face,))
    allocation = bind_local_approximation_field(model, prepared, field, materials)
    assert allocation.fingerprint
    snapshot = prepared.solid_region_snapshot(region)
    assert snapshot.volume == pytest.approx(6.)
    assert all(handle.model_id == prepared.model_id for handle in snapshot.face_refs)


@pytest.mark.parametrize("new_epoch", [False, True])
def test_prepared_clone_rebind_never_hides_changed_solid_geometry(monkeypatch, new_epoch):
    from anygeometry import clone_prepared_geometry

    model = GeometryModel()
    region = model.add_solid_box((10, 0, 0), (11, 2, 3))
    before = to_dict(model)
    original_clone = GeometryModel.clone

    def altered_clone(self, **kwargs):
        made = original_clone(self, **kwargs)
        made.regenerate_solid_prism(region,
            ((10, 0, 0), (11, 0, 0), (11, 2, 0), (10, 2, 0)), (0, 0, 4))
        return made

    monkeypatch.setattr(GeometryModel, "clone", altered_clone)
    with pytest.raises(GeometryError, match="clone changed"):
        clone_prepared_geometry(model, new_preparation_epoch=new_epoch)
    assert to_dict(model) == before
