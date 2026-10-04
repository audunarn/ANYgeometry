"""Rebuild transient inventory using installed public APIs; never mesh."""
from anygeometry import (
    GeometryModel, apply_intersections, plan_intersections, to_dict,
    query_prepared_authored_constraint_scope,
    validate_prepared_authored_constraint_scope_binding,
)


def build():
    model = GeometryModel()
    points = model.add_points(((0, 0, 0), (1, 2, 0), (2, -1, 0), (3, 1, 0)))
    edge = model.add_spline(points[0], points[1:-1], points[-1])
    wall = model.extrude((edge,), (.25, 0, 1.5))[0]
    cutter = model.add_plate(model.add_points(((1.5, -3, -1), (1.5, 3, -1),
                                               (1.5, 3, 3), (1.5, -3, 3))))
    original = to_dict(model)
    plan = plan_intersections(model, (wall, cutter), policy='connect')
    applied = apply_intersections(model, plan, policy='connect')
    prepared = to_dict(model)
    full = query_prepared_authored_constraint_scope(model, (wall, cutter))
    partial = query_prepared_authored_constraint_scope(model, (wall,))
    assert full.scope.authored_document == original
    assert full.scope.current_document == prepared
    assert full.selected_root_ids == tuple(sorted((wall, cutter)))
    assert not full.outside_root_ids
    assert partial.outside_root_ids == (cutter,)
    assert not full.publication_qualified
    for receipt in (full, partial):
        validate_prepared_authored_constraint_scope_binding(model, receipt)
    assert to_dict(model) == prepared
    return model, (wall, cutter), tuple(edge.id for edge in applied.joint_edges), full, partial


if __name__ == '__main__':
    model, roots, joints, full, partial = build()
    print({'authored_roots': roots, 'joint_edges': joints,
           'complete_typed_visibility': True,
           'constraint_preservation_qualified': False,
           'accepted_mesh': False})
