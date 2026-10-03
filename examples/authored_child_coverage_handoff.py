"""Portable selected-child discriminator; no mesh acceptance or permission."""
from anygeometry import (
    GeometryError, GeometryModel, Plane, apply_intersections, plan_intersections,
    query_prepared_authored_boundary_correspondence,
    validate_prepared_authored_face_triangles,
    validate_prepared_authored_face_child_triangles,
)


def build():
    model = GeometryModel()
    original = model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    model.set_face_surface(original, Plane((0,0,0),(1,0,0),(0,1,0)))
    model.add_plate(model.add_points(((3,-1,-1),(3,5,-1),(3,5,1),(3,-1,1))))
    apply_intersections(model, plan_intersections(model, tuple(model.faces), policy='connect'), policy='connect')
    correspondence = query_prepared_authored_boundary_correspondence(model, original)
    left, right = sorted(correspondence.descendants, key=lambda face: min(
        model.vertex_position(model.oriented_start_vertex(use))[0] for use in model.faces[face].loop))
    return model, correspondence, left, right


def verify():
    model, correspondence, left, right = build()
    crossing = [[[2,.5],[3.5,.5],[2,1.5]]]
    validate_prepared_authored_face_triangles(model, correspondence, crossing)
    for child in (left, right):
        try:
            validate_prepared_authored_face_child_triangles(model, correspondence, child, crossing)
        except GeometryError:
            pass
        else:
            raise AssertionError('original-root coverage must not grant a child scope')
    validate_prepared_authored_face_child_triangles(model, correspondence, left,
                                                   [[[1,.5],[2,.5],[1,1.5]]])
    validate_prepared_authored_face_child_triangles(model, correspondence, right,
                                                   [[[3.1,.5],[3.8,.5],[3.1,1.5]]])
    return {'original': correspondence.authored_definition.face_id, 'left': left, 'right': right,
            'cut_x': 3, 'original_area': 16, 'left_area': 12, 'right_area': 4,
            'accepted_mesh': False}


if __name__ == '__main__':
    print(verify())
