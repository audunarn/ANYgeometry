"""Owner receipts for an attached member on a split plate boundary.

This proves geometry ancestry and retained coordinates, not consumer meshing.
"""
from fractions import Fraction

import numpy as np
import pytest

from anygeometry import (
    GeometryError, GeometryModel, apply_intersections, plan_intersections,
    query_prepared_edge_subcurve_preimages, to_dict,
    query_prepared_sheet_joint_component,
    validate_prepared_edge_subcurve_preimages_binding,
)
from anygeometry.structural import Orientation, ParameterRange


@pytest.mark.parametrize('reverse', [False, True])
def test_boundary_member_and_vertex_attachment_have_complete_split_ancestry(reverse):
    model = GeometryModel()
    for points in (
        ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)),
        ((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)),
    ):
        face = model.add_plate(model.add_points(points))
        model.add_sheet((face,))
    edge = next(e.id for e in model.edges.values()
                if all(model.vertex_position(v)[0] == -1
                       and model.vertex_position(v)[2] == 0
                       for v in (e.start, e.end)))
    member = model.add_member((edge,))
    if reverse:
        model.reverse_member(member)
    expected = model.sample_edge(edge, np.array([.75]))[0]
    vertex = model.add_point(*expected)
    attachment = model.add_attachment(
        None, 'vertex_on_edge', 'edge', edge, ParameterRange.point(0.),
        (ParameterRange.point(.75),), source_kind='vertex', source_id=vertex,
        evidence='exact', tolerance_used=1e-9,
    )
    original_attachment = model.attachments[attachment]
    before = to_dict(model)
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    apply_intersections(model, plan, policy='connect')
    receipt = query_prepared_edge_subcurve_preimages(model)
    records = sorted((r for r in receipt.records
                      if r.ancestor.definition.edge_id == edge),
                     key=lambda r: Fraction(*r.interval[0]))
    assert len(records) == 2
    assert records[0].interval[0] == (0, 1)
    assert records[0].interval[1] == records[1].interval[0]
    assert records[1].interval[1] == (1, 1)
    assert all(r.ancestor.source_checksum == before['checksum']['value']
               for r in records)
    uses = [model.member_edge_uses[i] for i in model.members[member].edge_use_ids]
    assert {u.edge_id for u in uses} == {r.edge_id for r in records}
    traversal = list(reversed(records)) if reverse else records
    assert [u.edge_id for u in uses] == [r.edge_id for r in traversal]
    assert all(u.orientation is (Orientation.REVERSED if reverse else Orientation.FORWARD)
               for u in uses)
    intervals = [(u.parent_range.start, u.parent_range.end) for u in uses]
    assert intervals == [(0., .5), (.5, 1.)]
    assert all(Fraction(*r.squared_distance_bound) == 0 for r in records)
    retained = model.attachments[attachment]
    assert retained.target_id == records[1].edge_id
    assert retained.target_parameters == (ParameterRange.point(.5),)
    for field in ('source_id', 'source_kind', 'evidence', 'tolerance_used', 'metadata'):
        assert getattr(retained, field) == getattr(original_attachment, field)
    np.testing.assert_allclose(
        model.sample_edge(retained.target_id,
                          np.array([retained.target_parameters[0].start]))[0],
        expected, rtol=0., atol=1e-12,
    )
    assert model.validate_topology() == ()
    joint = next(a.target_id for a in model.attachments.values()
                 if a.kind == 'sheet_on_joint')
    with pytest.raises(GeometryError, match='unsupported Member semantics'):
        query_prepared_sheet_joint_component(model, joint)
    validate_prepared_edge_subcurve_preimages_binding(model, receipt)
    committed = to_dict(model)
    assert apply_intersections(model, plan, policy='connect').reused
    assert query_prepared_edge_subcurve_preimages(model) == receipt
    assert to_dict(model) == committed
