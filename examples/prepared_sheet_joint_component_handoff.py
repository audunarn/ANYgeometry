"""Portable current Sheet-joint visibility; no mesher or publication route."""
from anygeometry import (
    GeometryModel, Plane, apply_intersections, plan_intersections,
    query_prepared_sheet_joint_component,
    validate_prepared_sheet_joint_component_selection,
)


def build(*, third=False, unrelated=False, explicit_sheets=True, before_prepare=None):
    model = GeometryModel()
    root = model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    model.set_face_surface(root, Plane((0,0,0),(1,0,0),(0,1,0)))
    cutter = model.add_plate(model.add_points(((3,-1,-1),(3,5,-1),(3,5,1),(3,-1,1))))
    roots = [root, cutter]
    if third:
        roots.append(model.add_plate(model.add_points(((1,-1,-1),(1,5,-1),(1,5,1),(1,-1,1)))))
    if unrelated:
        roots.append(model.add_plate(model.add_points(((10,0,0),(14,0,0),(14,4,0),(10,4,0)))))
    sheets = [model.add_sheet((face,), name=f'authored Sheet {face}') for face in roots] if explicit_sheets else []
    if before_prepare is not None:
        before_prepare(model, roots, sheets)
    apply_intersections(model, plan_intersections(model, tuple(model.faces), policy='connect'), policy='connect')
    # Select the declared straight joint at x=3 by its persisted endpoints.
    edges = [edge for edge in model.edges if len(model.sheets_using_edge(edge)) >= 2 and
             all(model.vertex_position(vertex)[0] == 3 for vertex in
                 (model.edges[edge].start, model.edges[edge].end))]
    assert len(edges) == 1
    return model, tuple(roots), tuple(sheets), edges[0]


def verify():
    model, roots, sheets, edge = build()
    receipt = query_prepared_sheet_joint_component(model, edge)
    validate_prepared_sheet_joint_component_selection(model, receipt, roots)
    return {'joint_edge':edge, 'authored_roots':receipt.authored_face_ids,
            'current_faces':receipt.current_face_ids, 'sheets':sheets,
            'occurrence_mapping_qualified':receipt.occurrence_mapping_qualified,
            'semantic_mapping_qualified':receipt.semantic_mapping_qualified,
            'publication_qualified':receipt.publication_qualified}


if __name__ == '__main__':
    print(verify())
