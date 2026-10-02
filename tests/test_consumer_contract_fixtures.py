"""Portable owner fixtures bind consumer expectations to public behavior."""
from pathlib import Path
import runpy

import numpy as np
import pytest

from anygeometry import to_dict, from_dict, query_trimmed_surface_charts

BUILDERS = runpy.run_path(str(Path(__file__).parents[1] / 'tools/general_intersections/consumer_contract_fixtures.py'))


@pytest.mark.parametrize('builder,args', [('cubic_oblique_pipe', ()), ('parabolic_pipe', ()),
    ('tangent_pipe', ()), ('tangent_pipe', (-1e-6,)), ('tangent_pipe', (1e-6,)),
    ('second_cut_with_attachments', ())])
def test_consumer_fixture_contract(builder, args):
    fixture = BUILDERS[builder](*args)
    authored = to_dict(fixture.model)
    model, result = BUILDERS['prepare'](fixture)
    assert to_dict(fixture.model) == authored
    assert model.validate_topology() == ()
    assert to_dict(from_dict(to_dict(model))) == to_dict(model)
    expected = fixture.expected
    if 'required_curve_type' in expected:
        curves = [e.curve for e in model.edges.values() if type(e.curve).__name__ == expected['required_curve_type']]
        assert curves
        if expected.get('requires_regularized_fold_end'):
            assert any(c.parameterization != 'linear' for c in curves)
    if 'generator_lines' in expected:
        lines = expected['generator_lines']
        assert len(result.joint_edges) == len(lines)
        for edge in result.joint_edges:
            endpoints = model.sample_edge(edge.id, np.array([0., 1.]))
            assert any(np.allclose(endpoints, line, rtol=0., atol=1e-10)
                       or np.allclose(endpoints[::-1], line, rtol=0., atol=1e-10) for line in lines)
        area = sum(c.material_area for c in query_trimmed_surface_charts(model).charts)
        assert area == pytest.approx(expected['material_area'], rel=1e-12)
    for station in expected.get('attachment_stations', ()):
        attachment = model.attachments[station['attachment_id']]
        assert attachment.target_id != expected['old_joint_edge']
        actual = model.sample_edge(attachment.target_id, np.array([attachment.target_parameters[0].start]))[0]
        np.testing.assert_allclose(actual, station['position'], rtol=0., atol=1e-10)
