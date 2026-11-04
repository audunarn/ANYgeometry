"""Ineligible chart families retain the direct, source-owned occurrence path."""
from fractions import Fraction as Q

import pytest
import anygeometry as owner
import anygeometry.prepared_material_curvilinear as module
from anygeometry.arrangement_geometry import freeze_edge
from anygeometry.definition_binding import definition_checksum
from anygeometry.nurbs import CurveOnSurface, NurbsSurface
from test_adaptive_rational_sampling import rounded


@pytest.mark.parametrize('reverse', (False, True))
def test_full_ineligible_query_never_enters_chart_reuse(monkeypatch, reverse):
    model, regions, seeds = rounded()
    original = owner.to_dict(model)
    if reverse:
        seeds = tuple((handle, parameters[::-1]) for handle, parameters in seeds)
    references = module._sampling_references(model, regions)
    chart = regions.regions[0].sources[0]
    reference = references[chart.face.id][1]
    old_owner = reference.owner if hasattr(reference, 'owner') else reference.model
    old_curves = tuple(old_owner.edges[handle.id].curve if old_owner is not model
                       else freeze_edge(model, handle.id) for handle, _ in seeds)
    assert type(chart.support) is NurbsSurface
    assert all(type(curve) is CurveOnSurface for curve in old_curves)
    canonical = {definition_checksum(curve): curve for curve in old_curves}
    actual = module._uv_box
    observed = []

    def record(support, curve, t):
        value = actual(support, curve, t)
        observed.append((support, definition_checksum(curve), t, value))
        return value

    def unexpected(*args, **kwargs):
        raise AssertionError('ineligible occurrence must not construct a chart producer')

    monkeypatch.setattr(module, '_uv_box', record)
    monkeypatch.setattr(module, '_sampling_chart_boxes', unexpected)
    result = module.query_prepared_material_boundary_sampling(
        model, regions, seeds, chord_error=.05, error_measure='cell_reference',
        proposal_fraction=.25)
    assert observed
    assert {t for _, _, t, _ in observed} >= {Q(0), Q(1)}
    for support, signature, t, value in observed:
        assert support is chart.support
        # This is the old owner/occurrence selection, evaluated by the same
        # original enclosure builder, without detached operands entering a bank.
        assert value == actual(support, canonical[signature], t)
    assert all(chord.world_error <= .05*.25 for chord in result.chords.intervals)
    assert all((parameters[0] > parameters[-1]) == reverse
               for _, parameters in result.edge_parameters)
    module.validate_prepared_material_boundary_sampling_binding(model, regions, result)
    assert owner.to_dict(model) == original
