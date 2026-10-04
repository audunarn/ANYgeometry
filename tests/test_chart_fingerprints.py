"""Public definition snapshots; no meshing or live-model certification."""

from copy import deepcopy
from dataclasses import replace
from uuid import uuid4

import pytest

from anygeometry import (
    GeometryError, GeometryModel, chart_definition_fingerprint,
    query_material_surface_regions, query_trimmed_surface_charts, to_dict,
    validate_trimmed_surface_charts_binding,
)


@pytest.fixture
def results():
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0, 0, 0), (2, 0, 0),
                                            (2, 1, 0), (0, 1, 0))))
    charts = query_trimmed_surface_charts(model, (face,))
    regions = query_material_surface_regions(model, (face,))
    return model, charts, regions


@pytest.mark.parametrize("selection", ("charts", "regions", "region"))
def test_fingerprint_is_content_based_and_does_not_mutate(results, selection):
    model, charts, regions = results
    result = {"charts": charts, "regions": regions, "region": regions.regions[0]}[selection]
    before = to_dict(model)
    token = chart_definition_fingerprint(result)
    assert isinstance(token, str)
    assert chart_definition_fingerprint(deepcopy(result)) == token
    assert chart_definition_fingerprint(result) == token
    assert to_dict(model) == before


@pytest.mark.parametrize("field,value", (("model_id", uuid4()), ("revision", -1),
                                         ("source_checksum", "refreshed")))
def test_collection_identity_and_source_content_are_bound(results, field, value):
    _, charts, regions = results
    changed = replace(charts, **{field: value})
    assert chart_definition_fingerprint(changed) != chart_definition_fingerprint(charts)
    assert chart_definition_fingerprint(replace(regions, source=changed)) != chart_definition_fingerprint(regions)


def test_nested_chart_support_and_region_reference_definitions_are_bound(results):
    model, charts, regions = results
    token = chart_definition_fingerprint(charts)
    chart = charts.charts[0]
    altered_domain = replace(chart.domain, support=replace(chart.support, origin=(0, 0, 1)))
    altered_chart = replace(chart, domain=altered_domain)
    assert chart_definition_fingerprint(replace(charts, charts=(altered_chart,))) != token
    region = regions.regions[0]
    region_token = chart_definition_fingerprint(region)
    # These are syntactically distinct receipts, not new source-validation claims.
    vertex = model.handle("vertex", min(model.vertices))
    for altered in (replace(region, retained_vertices=(vertex,)),
                    replace(region, source_attachments=(vertex,)),
                    replace(region, material_area=region.material_area + 1)):
        assert chart_definition_fingerprint(altered) != region_token
        assert chart_definition_fingerprint(replace(regions, regions=(altered,))) != chart_definition_fingerprint(regions)


def test_original_token_detects_same_object_content_refresh(results):
    _, charts, _ = results
    entry = chart_definition_fingerprint(charts)
    object.__setattr__(charts, "source_checksum", "rebound")
    assert chart_definition_fingerprint(charts) != entry


def test_fingerprint_does_not_replace_live_source_validation(results, monkeypatch):
    model, charts, _ = results
    entry = chart_definition_fingerprint(charts)
    model.add_point(10, 10, 10)
    # Definition-only query intentionally retains an unchanged stale token.
    import anygeometry.serialization as serialization
    monkeypatch.setattr(serialization, "to_dict", lambda *_: pytest.fail("live model serialization"))
    assert chart_definition_fingerprint(charts) == entry
    with pytest.raises(GeometryError, match="stale"):
        validate_trimmed_surface_charts_binding(model, charts)


@pytest.mark.parametrize("value", (None, {}, (), "token"))
def test_unrelated_inputs_are_refused(value):
    with pytest.raises(TypeError, match="public chart or material-region"):
        chart_definition_fingerprint(value)


def test_nonfinite_definition_is_a_typed_failure(results):
    _, _, regions = results
    with pytest.raises(GeometryError, match="cannot be fingerprinted"):
        chart_definition_fingerprint(replace(regions.regions[0], material_area=float("nan")))


def test_public_root_export_is_declared():
    import anygeometry
    assert "chart_definition_fingerprint" in anygeometry.__all__
    assert anygeometry.chart_definition_fingerprint is chart_definition_fingerprint
