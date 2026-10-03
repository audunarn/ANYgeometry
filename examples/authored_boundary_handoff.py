"""Portable owner-contract fixture; this is not a meshing acceptance harness.

Transient owner evidence must be rebuilt with the exact candidate package.
Saving/loading only the geometry document intentionally drops that evidence.
"""
from fractions import Fraction

from anygeometry import (
    GeometryModel, apply_intersections, plan_intersections,
    query_prepared_authored_boundary_correspondence,
    query_prepared_authored_boundary_stations,
    validate_prepared_authored_face_triangles,
)


def build():
    model = GeometryModel()
    points = model.add_points(((0, 0, 0), (1, 2, 0), (2, -1, 0), (3, 1, 0)))
    edge = model.add_spline(points[0], points[1:-1], points[-1])
    authored_wall = model.extrude((edge,), (.25, 0, 1.5))[0]
    cutter = model.add_plate(model.add_points(((1.5, -3, -1), (1.5, 3, -1),
                                               (1.5, 3, 3), (1.5, -3, 3))))
    plan = plan_intersections(model, (authored_wall, cutter), policy='connect')
    application = apply_intersections(model, plan, policy='connect')
    correspondence = query_prepared_authored_boundary_correspondence(model, authored_wall)
    # Explicit ORIGINAL analytic domain: the full normalized ruled rectangle.
    validate_prepared_authored_face_triangles(model, correspondence,
        (((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1))))
    current_edge = correspondence.exterior_loops[0][0][2][0]
    stations = query_prepared_authored_boundary_stations(model, correspondence,
        current_edge, (Fraction(0), Fraction(1, 3), Fraction(1)))
    return model, authored_wall, correspondence, stations, application.joint_edges


if __name__ == '__main__':
    model, root, boundary, stations, joints = build()
    print({'authored_wall': root, 'descendants': boundary.descendants,
           'physical_joints': tuple(edge.id for edge in joints),
           'exterior_loops': len(boundary.exterior_loops),
           'station_parameters': stations.parameters,
           'accepted_mesh': False})
