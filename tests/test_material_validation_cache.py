"""Live singleton scope reuse preserves genuine qualification and freshness."""
from dataclasses import replace
from fractions import Fraction as Q
import gc
import weakref

import pytest

from anygeometry import GeometryModel, GeometryError, to_dict
from anygeometry import query_material_surface_regions, validate_material_surface_regions_binding
from anygeometry import validate_trimmed_surface_charts_binding
from anygeometry.definition_binding import definition_checksum
from anygeometry import material_regions, trimmed_charts
from anygeometry.material_arrangement import MaterialDomain


def _plates(count=11):
    model = GeometryModel()
    faces = [model.add_plate(model.add_points(((2*x, 0, 0), (2*x+1, 0, 0),
              (2*x+1, 1, 0), (2*x, 1, 0)))) for x in range(count)]
    return model, faces


def _measure(monkeypatch):
    calls = {'area': 0, 'regions': 0, 'document': 0}
    for owner, key, name in ((MaterialDomain, 'area', 'original_world_area'),
                             (material_regions, 'regions', '_regions'),
                             (trimmed_charts, 'document', 'to_dict')):
        original = getattr(owner, name)
        def measured(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)  # genuine full certifier, never mocked proof
        monkeypatch.setattr(owner, name, measured)
    return calls


def _validate(kind, model, result, callback=None):
    if kind == 'chart':
        validate_trimmed_surface_charts_binding(model, result.source, cancellation_check=callback)
    else:
        validate_material_surface_regions_binding(model, result, cancellation_check=callback)


def test_eleven_public_singletons_cycle_without_repeating_full_certifiers(monkeypatch):
    model, faces = _plates()
    before = to_dict(model)
    counts = _measure(monkeypatch)
    results = [query_material_surface_regions(model, [face]) for face in faces]
    assert all(counts[name] > 0 for name in counts)
    cold = dict(counts)
    for _ in range(3):
        for result in results:
            validate_material_surface_regions_binding(model, result)
            validate_trimmed_surface_charts_binding(model, replace(result.source))
    assert counts == cold
    for cache in (material_regions._validated, trimmed_charts._validated_collections):
        assert set(cache[model].singletons) == set(faces)
        assert len(cache[model].recent) <= 8
    assert to_dict(model) == before


def test_non_singleton_recent_scopes_stay_bounded_and_requalify_evicted_result(monkeypatch):
    model, faces = _plates()
    results = [query_material_surface_regions(model, [faces[0], face]) for face in faces[1:]]
    counts = _measure(monkeypatch)
    validate_material_surface_regions_binding(model, results[-1])
    assert counts == {'area': 0, 'regions': 0, 'document': 0}
    validate_material_surface_regions_binding(model, results[0])
    assert all(counts[name] > 0 for name in counts)
    for cache in (material_regions._validated, trimmed_charts._validated_collections):
        assert len(cache[model].recent) == 8
        assert not cache[model].singletons


def test_new_same_revision_source_epoch_retires_all_old_singletons():
    model, faces = _plates(3)
    results = [query_material_surface_regions(model, [face]) for face in faces]
    old = (material_regions._validated[model], trimmed_charts._validated_collections[model])
    revision = model.revision
    model._faces[faces[0]] = replace(model.faces[faces[0]], metadata={'new': True})
    assert model.revision == revision
    with pytest.raises(GeometryError, match='source binding changed'):
        validate_material_surface_regions_binding(model, results[0])
    fresh = query_material_surface_regions(model, [faces[0]])
    for previous, cache in zip(old, (material_regions._validated, trimmed_charts._validated_collections)):
        assert cache[model] is not previous
        assert cache[model].epoch[0] == fresh.source.source_checksum
        assert set(cache[model].singletons) == {faces[0]}


@pytest.mark.parametrize('kind', ('chart', 'region'))
def test_warm_result_tamper_cancellation_and_source_callback_refuse(kind):
    model, faces = _plates(1)
    result = query_material_surface_regions(model, faces)
    forged = (replace(result, source=replace(result.source, charts=(replace(result.source.charts[0], material_area=9.),)))
              if kind == 'chart' else replace(result, regions=(replace(result.regions[0], material_area=9.),)))
    with pytest.raises(GeometryError, match='binding changed'):
        _validate(kind, model, forged)
    before = to_dict(model)
    with pytest.raises(GeometryError, match='cancelled'):
        _validate(kind, model, result, lambda phase: True)
    assert to_dict(model) == before
    marker = RuntimeError('caller exception identity')
    def throwing(phase):
        raise marker
    with pytest.raises(RuntimeError) as caught:
        _validate(kind, model, result, throwing)
    assert caught.value is marker
    face = model.faces[faces[0]]
    def mutate(phase):
        model._faces[face.id] = replace(face, metadata={'callback': True})
    try:
        with pytest.raises(GeometryError, match='changed'):
            _validate(kind, model, result, mutate)
    finally:
        model._faces[face.id] = face


def _cylinder():
    from test_cylinder_angular_lifts import panel
    model, face, _ruling, _seam, _bottom, surface = panel()
    return model, face, surface, query_material_surface_regions(model, [face])


@pytest.mark.parametrize('kind', ('chart', 'region'))
@pytest.mark.parametrize('callback', (False, True))
def test_warm_cylinder_hidden_native_edit_refuses(kind, callback):
    model, _face, surface, result = _cylinder()
    before = surface._circumferential.copy()
    def mutate(phase):
        surface._circumferential[:] = 2*before
    try:
        if callback:
            with pytest.raises(GeometryError, match='native cylinder'):
                _validate(kind, model, result, mutate)
        else:
            mutate(None)
            with pytest.raises(GeometryError, match='native cylinder'):
                _validate(kind, model, result)
    finally:
        surface._circumferential[:] = before


@pytest.mark.parametrize('kind', ('chart', 'region', 'region_source'))
def test_detached_equivalent_cylinder_evidence_owes_actual_native_audit(kind):
    model, _face, surface, result = _cylinder()
    detached = replace(surface)
    if kind == 'chart':
        chart = result.source.charts[0]
        good = replace(result, source=replace(result.source, charts=(
            replace(chart, domain=replace(chart.domain, support=detached)),)))
    elif kind == 'region':
        region = result.regions[0]
        good = replace(result, regions=(replace(region, domain=replace(region.domain, support=detached)),))
    else:
        region = result.regions[0]
        chart = region.sources[0]
        good = replace(result, regions=(replace(region, sources=(
            replace(chart, domain=replace(chart.domain, support=detached)),)),))
    assert definition_checksum(good) == definition_checksum(result)
    _validate(kind, model, good)  # equivalent canonical copies remain supported
    detached._circumferential[:] *= 2
    assert definition_checksum(good) == definition_checksum(result)
    with pytest.raises(GeometryError, match='native cylinder'):
        _validate(kind, model, good)
    repaired = []
    def repair(phase):
        repaired.append(True)
        detached._circumferential[:] = surface._circumferential
    with pytest.raises(GeometryError, match='native cylinder'):
        _validate(kind, model, good, repair)
    assert not repaired  # poisoned initial evidence cannot be callback-repaired
    _validate(kind, model, result)  # actual model and original evidence untouched


@pytest.mark.parametrize('kind', ('chart', 'region'))
def test_cold_noncanonical_section_law_refuses_before_full_certifier(monkeypatch, kind):
    from test_sweep_world_adjacency import _cut_adjacent_sweeps
    from anygeometry.support_curves import PlaneNurbsCurve
    model, traces = _cut_adjacent_sweeps()
    face = min(model.faces_using_edge(traces[0]))
    result = query_material_surface_regions(model, [face])
    material_regions._validated.pop(model, None)
    trimmed_charts._validated_collections.pop(model, None)
    entered = []
    def forbidden(*args, **kwargs):
        entered.append(True)
        raise AssertionError('noncanonical law reached full certifier')
    monkeypatch.setattr(trimmed_charts, 'to_dict', forbidden)
    monkeypatch.setattr(PlaneNurbsCurve, '_INITIAL_ROOT_BISECTIONS', 0)
    with pytest.raises(GeometryError, match='evaluator law is not certified'):
        _validate(kind, model, result)
    assert not entered


def test_cache_values_do_not_retain_model_or_evidence():
    model, faces = _plates(1)
    result = query_material_surface_regions(model, faces)
    reference = weakref.ref(model)
    scopes = (material_regions._validated[model], trimmed_charts._validated_collections[model])
    del model
    gc.collect()
    assert reference() is None
    assert all(scope.singletons for scope in scopes)
    assert result.source.charts  # evidence alive does not keep its model alive


@pytest.mark.parametrize('memo', ('chart', 'boundary'))
@pytest.mark.parametrize('callback', (False, True))
def test_legitimate_domain_memos_reuse_but_poisoned_consumed_entries_refuse(memo, callback):
    model, faces = _plates(1)
    result = query_material_surface_regions(model, faces)
    domain = result.regions[0].domain
    loop = domain.boundaries[0]
    assert domain.contains_loop(loop, (Q(1, 2), Q(1, 2)), 1e-9, boundary=False)
    validate_material_surface_regions_binding(model, result)
    if memo == 'chart':
        target = domain._exact_chart_cache[1]
        field, original = '_scale', domain._exact_chart_cache[1]._scale
    else:
        prepared = domain._exact_boundary_cache[2][id(loop)]
        target = next(iter(prepared.values()))[1]
        field, original = 'length2', next(iter(prepared.values()))[1].length2
    def poison(phase):
        object.__setattr__(target, field, original+1)
    try:
        if callback:
            with pytest.raises(GeometryError, match='cache differs'):
                validate_material_surface_regions_binding(model, result, cancellation_check=poison)
        else:
            poison(None)
            with pytest.raises(GeometryError, match='cache differs'):
                validate_material_surface_regions_binding(model, result)
    finally:
        object.__setattr__(target, field, original)
    validate_material_surface_regions_binding(model, result)


def test_generic_topology_coons_uses_full_qualification_without_reusable_scope(monkeypatch):
    from anygeometry.surfaces import CoonsSurface
    from anygeometry.entities import OrientedEdge
    from anygeometry.native_support_snapshots import capture_material_validation_read_pin
    model = GeometryModel()
    points = model.add_points(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)))
    edges = [model.add_line(points[i], points[(i+1)%4]) for i in range(4)]
    face = model.add_face_from_loop(tuple(OrientedEdge(edge, True) for edge in edges),
        (0, 1, 2, 3), surface=CoonsSurface())
    result = query_material_surface_regions(model, [face])
    assert capture_material_validation_read_pin(model) is None
    counts = _measure(monkeypatch)
    model._topology_corner_cache.clear()  # unbound native cache cannot supply reusable authority
    validate_material_surface_regions_binding(model, result)
    assert all(counts[name] > 0 for name in counts)
    assert model not in material_regions._validated
    assert model not in trimmed_charts._validated_collections


def test_qualified_exact_sweep_retains_singleton_reuse_without_chart_reconstruction(monkeypatch):
    from test_sweep_world_adjacency import _cut_adjacent_sweeps
    from anygeometry import nurbs_sweep_chart
    from anygeometry.native_support_snapshots import capture_material_validation_read_pin
    model, traces = _cut_adjacent_sweeps()
    face = min(model.faces_using_edge(traces[0]))
    result = query_material_surface_regions(model, [face])
    assert capture_material_validation_read_pin(model) is not None
    counts = _measure(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('warm exact-sweep binding reconstructed its chart')
    monkeypatch.setattr(nurbs_sweep_chart, 'chart_model', forbidden)
    validate_material_surface_regions_binding(model, result)
    assert counts == {'area': 0, 'regions': 0, 'document': 0}
    assert face in material_regions._validated[model].singletons
    assert face in trimmed_charts._validated_collections[model].singletons


def test_partial_coons_marker_with_recipe_cannot_acquire_read_pin():
    import numpy as np
    from test_sweep_world_adjacency import _cut_adjacent_sweeps
    from anygeometry.native_support_snapshots import capture_material_validation_read_pin, _material_support_read_pin
    model, traces = _cut_adjacent_sweeps()
    face = min(model.faces_using_edge(traces[0]))
    marker = model.faces[face].surface
    object.__setattr__(marker, 'right', np.array(((0., 0., 0.), (0., 1., 0.))))
    try:
        assert _material_support_read_pin(marker) is None
        assert capture_material_validation_read_pin(model) is None
    finally:
        object.__setattr__(marker, 'right', None)
