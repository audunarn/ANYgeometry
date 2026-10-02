"""Small representatives only; growing acceptance runs use the bounded tool."""
from pathlib import Path

import pytest

from anygeometry import (IntersectionBatchPolicy, GeometryError, apply_intersections,
    from_dict, plan_intersections, query_trimmed_surface_charts, to_dict)


@pytest.fixture
def builders(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'tools/general_intersections'))
    import large_connected_fixtures
    return large_connected_fixtures


def test_centered_bezier_split_preserves_original_endpoints():
    from anygeometry.arrangement_geometry import BezierPath
    curve = BezierPath(((1e16, 0., 7.), (5e15, 1., 7.), (1., 2., 7.)))
    left, right = curve._split(.4)
    assert left.controls[0] == curve.controls[0]
    assert right.controls[-1] == curve.controls[-1]
    assert left.controls[-1] == right.controls[0]
    assert all(point[2] == 7. for child in (left, right) for point in child.controls)


def test_bounded_runner_exercises_chart_cancellation_signature(builders, tmp_path):
    import json
    from types import SimpleNamespace
    from check_large_connected import run
    report = tmp_path/'result.json'
    run(SimpleNamespace(family='strip', count=10, profile=None, export=None,
                        report=str(report), deadline_seconds=30.))
    result = json.loads(report.read_text())
    assert result['status'] == 'passed'
    assert 'charts' in result['stages']


@pytest.mark.parametrize('family,count', [('connected_strip', 10), ('connected_hub', 10),
                                          ('connected_mixed', 10), ('concave_boundary_junctions', None)])
def test_connected_source_owners_material_and_joint_oracles(builders, family, count):
    from check_large_connected import verify
    fixture = getattr(builders, family)(count) if count is not None else getattr(builders, family)()
    authored = to_dict(fixture.model)
    model = from_dict(authored)
    plan = plan_intersections(model, tuple(reversed(fixture.operands)), policy='connect')
    assert to_dict(model) == authored
    apply_intersections(model, plan, policy='connect')
    verify(fixture, model, query_trimmed_surface_charts(model))
    committed = to_dict(model)
    assert apply_intersections(model, plan, policy='connect').reused
    assert to_dict(model) == committed == to_dict(from_dict(committed))
    assert to_dict(fixture.model) == authored


def test_connected_planning_budget_and_cancellation_leave_source_unchanged(builders):
    fixture = builders.connected_strip(10)
    source = to_dict(fixture.model)
    for policy in (IntersectionBatchPolicy(max_candidate_pairs=1),
                   IntersectionBatchPolicy(max_predicates=1),
                   IntersectionBatchPolicy(cancellation_check=lambda: True)):
        with pytest.raises(GeometryError):
            plan_intersections(fixture.model, fixture.operands, policy=policy)
        assert to_dict(fixture.model) == source


@pytest.mark.parametrize('family', ['connected_strip', 'connected_mixed'])
def test_operand_order_produces_identical_document(builders, family):
    fixture = getattr(builders, family)(10)
    original = to_dict(fixture.model)
    documents = []
    for operands in (fixture.operands, tuple(reversed(fixture.operands))):
        model = from_dict(original)
        plan = plan_intersections(model, operands, policy='connect')
        apply_intersections(model, plan, policy='connect')
        documents.append(to_dict(model))
    assert documents[0] == documents[1]


@pytest.mark.parametrize('height', [4.5, 5.])
def test_boundary_member_decomposition_preserves_point_attachment(builders, height):
    import numpy as np
    from anygeometry.structural import ParameterRange
    fixture = builders.concave_boundary_junctions()
    model = fixture.model
    member = model.members[max(model.members)]
    edge = model.member_edge_uses[member.edge_use_ids[0]].edge_id
    point = (0., height, 0.)
    vertex = model.add_point(*point)
    identifier = model.add_attachment(None, 'vertex_on_edge', 'edge', edge,
        ParameterRange.point(0.), (ParameterRange.point(height/6.),),
        source_kind='vertex', source_id=vertex, evidence='exact', tolerance_used=1e-9)
    plan = plan_intersections(model, fixture.operands, policy='connect')
    apply_intersections(model, plan, policy='connect')
    attachment = model.attachments[identifier]
    position = model.sample_edge(attachment.target_id,
        np.asarray([attachment.target_parameters[0].start]))[0]
    np.testing.assert_allclose(position, point, rtol=0., atol=1e-12)
    assert model.validate_topology() == ()


@pytest.mark.parametrize('bay', [60, 69])
def test_translated_cubic_bay_retains_material_without_relaxed_tolerance(builders, bay):
    from check_large_connected import verify
    fixture = builders.connected_mixed(10, start_bay=bay)
    authored = to_dict(fixture.model)
    model = from_dict(authored)
    plan = plan_intersections(model, fixture.operands, policy='connect')
    apply_intersections(model, plan, policy='connect')
    verify(fixture, model, query_trimmed_surface_charts(model))
    assert to_dict(fixture.model) == authored
