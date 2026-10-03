"""Small complete persisted-reference fixture; no mesh or remap claim.

Use build_prepared_scope() to obtain the prepared model, owner scope receipt
and a detached original document for comparison. Queries are public owner APIs.
"""
from anygeometry import (
    GeometryModel, apply_intersections, plan_intersections,
    query_prepared_model_scope, to_dict, validate_prepared_model_scope_binding,
)
from anygeometry.structural import ParameterRange


def build_prepared_scope():
    model = GeometryModel()
    plate = model.add_plate(model.add_points(((0,0,0),(2,0,0),(2,2,0),(0,2,0))))
    model.add_plate(model.add_points(((1,-1,-1),(1,3,-1),(1,3,1),(1,-1,1))))
    model.add_plate(model.add_points(((10,0,0),(12,0,0),(12,2,0),(10,2,0))))
    edge = model.faces[plate].loop[0].edge
    vertex = model.add_point(.5,0,0)
    endpoint = model.add_point(.5,0,1)
    member = model.add_member((model.add_line(vertex, endpoint),), metadata={'section': 'fixture'})
    model.add_attachment(member, 'member_on_boundary', 'edge', edge,
        ParameterRange.point(0), (ParameterRange.point(.25),), evidence='exact', tolerance_used=1e-9)
    model.add_attachment(None, 'vertex_on_edge', 'edge', edge, ParameterRange.point(0),
        (ParameterRange.point(.25),), source_kind='vertex', source_id=vertex,
        evidence='exact', tolerance_used=1e-9)
    model.add_point(20,20,20)  # Isolated: visibility is not a surface mapping.
    original = to_dict(model)
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    apply_intersections(model, plan, policy='connect')
    scope = query_prepared_model_scope(model)
    validate_prepared_model_scope_binding(model, scope)
    assert scope.authored_document == original
    assert scope.current_document == to_dict(model)
    return model, scope, original


if __name__ == '__main__':
    model, scope, original = build_prepared_scope()
    print({'original_faces': len(original['faces']), 'current_faces': len(model.faces),
           'original_attachments': len(original['structural']['attachments']),
           'original_vertices': len(original['vertices']), 'scope_valid': True})
