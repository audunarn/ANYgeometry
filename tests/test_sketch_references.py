import numpy as np
import pytest

from anygeometry import (
    ReferenceLink, SketchReference, SketchReferenceError, bind_references,
    capture_reference_link, capture_sketch_reference, reference_links_from_parameters,
    reference_links_to_parameters, surviving_reference_links,
)
from anygeometry.entities import EntityRef
from anygeometry.model import GeometryModel
from anygeometry.sketch import SketchConstraint, SketchDefinition, plane_sketch_plane
from anygeometry.sketch_entities import SketchEntity, SketchWire, materialize_entities


def _plane(origin=(0, 0, 0)):
    return plane_sketch_plane(origin, (0, 0, 1), (1, 0, 0))


def test_vertex_in_plane_is_captured_as_fixed_snapshot():
    model = GeometryModel()
    source = EntityRef("vertex", model.add_point(0.25, 0.5, 0.0))
    reference = capture_sketch_reference(model, _plane(), source, "a")
    assert reference.kind == "vertex" and reference.points == ((0.25, 0.5),)
    assert reference.anchors() == {"ref:a": (0.25, 0.5)}


def test_vertex_off_plane_is_explicit():
    model = GeometryModel()
    source = EntityRef("vertex", model.add_point(0.0, 0.0, 1.0))
    with pytest.raises(SketchReferenceError) as caught:
        capture_sketch_reference(model, _plane(), source, "a")
    assert caught.value.reason == "no_intersection"


def test_far_origin_vertex_off_plane_is_rejected_by_local_tolerance():
    model = GeometryModel()
    source = EntityRef("vertex", model.add_point(1.0e9, 0.0, 0.25))
    with pytest.raises(SketchReferenceError) as caught:
        capture_sketch_reference(model, _plane((1.0e9, 0.0, 0.0)), source, "a")
    assert caught.value.reason == "no_intersection"


@pytest.mark.parametrize("origin", [(0.0, 0.0, 0.0), (1.0e9, 0.0, 0.0), (-3.0e8, 7.0e8, 0.0)])
def test_vertex_membership_does_not_depend_on_plane_origin(origin):
    model = GeometryModel()
    source = EntityRef("vertex", model.add_point(1.0e9, 0.0, 0.25))
    with pytest.raises(SketchReferenceError) as caught:
        capture_sketch_reference(model, _plane(origin), source, "a")
    assert caught.value.reason == "no_intersection"
    inside = EntityRef("vertex", model.add_point(1.0e9, 0.0, 0.0))
    assert capture_sketch_reference(model, _plane(origin), inside, "b").kind == "vertex"


def test_vertex_membership_honours_custom_coincidence_policy():
    from anygeometry.tolerance import TolerancePolicy

    model = GeometryModel()
    source = EntityRef("vertex", model.add_point(0.5, 0.5, 1.0e-6))
    with pytest.raises(SketchReferenceError):
        capture_sketch_reference(model, _plane(), source, "a")
    model.set_document_settings(tolerance=TolerancePolicy(length=1.0e-3, merge_length=1.0e-2))
    assert capture_sketch_reference(model, _plane(), source, "a").points == ((0.5, 0.5),)
    model.set_document_settings(tolerance=TolerancePolicy(length=1.0e-9, coincidence=1.0e-8))
    with pytest.raises(SketchReferenceError):
        capture_sketch_reference(model, _plane(), source, "a")


def test_legacy_user_point_with_reference_prefix_is_still_materialized():
    definition = SketchDefinition(
        points={"ref:custom": (0.0, 0.0), "b": (1.0, 0.0)},
        entities=(SketchEntity("e", "line", ("ref:custom", "b")),),
        wires=(SketchWire("w", ("e",)),), version=2)
    model = GeometryModel()
    outputs = materialize_entities(model, _plane(), bind_references(definition))
    assert "point/ref:custom" in outputs and "point/b" in outputs
    bound = bind_references(_definition(SketchReference("a", "vertex", 1, ((0.4, 0.6),))))
    assert bound.bound_reference_keys == frozenset({"ref:a"})
    assert "point/ref:a" not in materialize_entities(GeometryModel(), _plane(), bound)


def test_crossing_edge_and_inplane_edge():
    model = GeometryModel()
    low, high = model.add_point(0.3, 0.2, -1.0), model.add_point(0.3, 0.2, 1.0)
    crossing = EntityRef("edge", model.add_line(low, high))
    reference = capture_sketch_reference(model, _plane(), crossing, "c")
    assert reference.kind == "crossing" and reference.count == 1
    assert reference.points[0] == pytest.approx((0.3, 0.2))
    start, end = model.add_point(0, 0, 0), model.add_point(1, 0, 0)
    flat = EntityRef("edge", model.add_line(start, end))
    reference = capture_sketch_reference(model, _plane(), flat, "f")
    assert reference.kind == "inplane" and reference.keys == ("ref:f:start", "ref:f:end")
    assert reference.points[1] == pytest.approx((1.0, 0.0)) and len(reference.curve) > 2


def test_missing_face_is_explicit():
    with pytest.raises(SketchReferenceError) as caught:
        capture_sketch_reference(GeometryModel(), _plane(), EntityRef("face", 1), "x")
    assert caught.value.reason == "missing"


def _plate(model, corners):
    return EntityRef("face", model.add_plate([model.add_point(*item) for item in corners]))


def test_planar_plate_section_is_captured_as_exact_straight_segment():
    model = GeometryModel()
    face = _plate(model, ((0, 0, -1), (2, 0, -1), (2, 0, 1), (0, 0, 1)))
    reference = capture_sketch_reference(model, _plane(), face, "s")
    assert reference.kind == "section" and reference.source_kind == "face"
    (start, end), = reference.segments
    assert np.asarray(sorted((start, end)), dtype=float) == pytest.approx(np.array([[0.0, 0.0], [2.0, 0.0]]))
    assert reference.keys == ("ref:s:0", "ref:s:1")
    assert SketchReference.from_dict(reference.to_dict()) == reference


def test_plate_missing_the_plane_and_coplanar_plate():
    model = GeometryModel()
    high = _plate(model, ((0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1)))
    with pytest.raises(SketchReferenceError) as caught:
        capture_sketch_reference(model, _plane(), high, "s")
    assert caught.value.reason == "no_intersection"
    flat = _plate(model, ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
    reference = capture_sketch_reference(model, _plane(), flat, "f")
    assert reference.kind == "section" and len(reference.segments) == 4


def test_plate_section_link_detaches_when_plate_moves():
    model = GeometryModel()
    face = _plate(model, ((0, 0, -1), (2, 0, -1), (2, 0, 1), (0, 0, 1)))
    link = capture_reference_link(model, face, "s")
    assert surviving_reference_links(model, [link]) == (link,)
    from anygeometry.editing import translate_entities

    translate_entities(model, [face], (0.0, 0.5, 0.0))
    assert surviving_reference_links(model, [link]) == ()


def _definition(reference):
    return SketchDefinition(
        points={"p": (1.0, 1.0), "q": (2.0, 1.0)},
        entities=(SketchEntity("e", "line", ("p", "q")),),
        wires=(SketchWire("w", ("e",)),),
        constraints=(SketchConstraint("coincident", "p", "ref:a"),),
        version=2, references=(reference,),
    )


def test_snapshot_persists_and_binds_hard_anchor_without_model():
    reference = SketchReference("a", "vertex", 1, ((0.4, 0.6),))
    definition = _definition(reference)
    data = definition.to_parameters()
    assert data["references"][0]["points"] == [[0.4, 0.6]]
    assert "ref:a" not in data["points"]
    assert SketchDefinition.from_parameters(data) == definition
    model = GeometryModel()
    outputs = materialize_entities(model, _plane(), bind_references(definition))
    assert "point/ref:a" not in outputs and "point/p" in outputs
    assert model.vertex_position(outputs["point/p"].id) == pytest.approx((0.4, 0.6, 0.0))


def test_reference_without_snapshot_is_rejected():
    with pytest.raises(Exception):
        SketchReference("a", "vertex")


def test_old_definitions_unchanged_and_unbound_solve_is_rejected():
    plain = SketchDefinition(points={"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (0.0, 1.0)}, path=("a", "b", "c"))
    assert "references" not in plain.to_parameters()
    from anygeometry.sketch import solve_sketch

    with pytest.raises(Exception):
        solve_sketch(_definition(SketchReference("a", "vertex", 1, ((0.0, 0.0),))), _plane())


def test_link_survives_unrelated_change_and_detaches_on_source_change():
    model = GeometryModel()
    other = model.add_point(5.0, 5.0, 5.0)
    vertex = model.add_point(0.25, 0.5, 0.0)
    source = EntityRef("vertex", vertex)
    link = capture_reference_link(model, source, "a")
    assert reference_links_from_parameters({"reference_links": reference_links_to_parameters([link])}) == (link,)
    model.add_point(9.0, 9.0, 9.0)
    assert surviving_reference_links(model, [link]) == (link,)
    from anygeometry.editing import translate_entities

    translate_entities(model, [source], (0.05, 0.0, 0.0))  # public geometry command
    assert surviving_reference_links(model, [link]) == ()
    assert other


def test_metadata_parameter_refuses_geometry_defining_keys_atomically():
    model = GeometryModel()
    point = model.features.append("geometry.point", parameters={"position": [0.0, 0.0, 0.0]})
    sketch = model.features.append("geometry.sketch.plane", parameters={"points": {}})
    before = model.features.snapshot()
    revision = model.revision
    for feature, key in ((point.feature_id, "reference_links"), (point.feature_id, "position"),
                         (sketch.feature_id, "points"), (sketch.feature_id, "plane")):
        with pytest.raises(Exception):
            model.features.set_metadata_parameter(feature, key, [{"x": 1}] if key == "reference_links" else {"a": 1})
    assert model.features.snapshot() == before and model.revision == revision
    model.features.set_metadata_parameter(sketch.feature_id, "reference_links", [{"reference": "a"}])
    assert model.features.parameter_values("reference_links") == {sketch.feature_id: [{"reference": "a"}]}


def test_link_detaches_when_source_is_missing():
    model = GeometryModel()
    link = ReferenceLink("a", EntityRef("vertex", 999), "0" * 64)
    assert surviving_reference_links(model, [link]) == ()
