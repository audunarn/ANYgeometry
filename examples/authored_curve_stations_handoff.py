"""Small public owner fixture; no discretization or meshing acceptance claim."""
from fractions import Fraction

import numpy as np

from anygeometry import (
    BezierQuadricCurve, GeometryModel, IntersectionBatchPolicy, apply_intersections, plan_intersections,
    evaluate_prepared_authored_face, query_prepared_authored_boundary_correspondence,
    query_prepared_authored_curve_stations,
    validate_prepared_authored_curve_station_coordinates, to_dict,
)
from anygeometry.generators import cylinder


def build(*, cancellation_check=None):
    design = GeometryModel()
    controls = design.add_points(((0, 0, 0), (1, 2, 0), (2, -1, 0), (3, 1, 0)))
    edge = design.add_spline(controls[0], controls[1:-1], controls[-1])
    wall = design.extrude((edge,), (.25, 0, 1.5))[0]
    design.insert_model(cylinder(.7, 8, origin=(-2, .4, .6), axis=(1, 0, 0),
        radial_direction=(0, 1, 0), circumferential_segments=8))
    source = to_dict(design)
    prepared = design.clone(preserve_identity=True)
    policy = IntersectionBatchPolicy(cancellation_check=cancellation_check)
    plan = plan_intersections(prepared, tuple(prepared.faces), policy=policy)
    application = apply_intersections(prepared, plan, policy=policy)
    correspondence = query_prepared_authored_boundary_correspondence(prepared, wall,
        cancellation_check=cancellation_check)
    assert to_dict(design) == source
    return design, prepared, wall, application, correspondence


def check_stations(prepared, correspondence, *, cancellation_check=None):
    parameters = (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1))
    receipts = []
    for edge, _occurrences in correspondence.interior_incidence:
        assert isinstance(prepared.edges[edge].curve, BezierQuadricCurve)
        receipt = query_prepared_authored_curve_stations(prepared, correspondence, edge, parameters,
            cancellation_check=cancellation_check)
        xyz = prepared.sample_edge(edge, np.array(tuple(map(float, parameters))))
        original = xyz.tobytes()
        validate_prepared_authored_curve_station_coordinates(prepared, receipt, xyz,
            cancellation_check=cancellation_check)
        assert xyz.tobytes() == original
        receipts.append(receipt)
    return tuple(receipts)


if __name__ == '__main__':
    design, prepared, root, application, correspondence = build()
    receipts = check_stations(prepared, correspondence)
    uv = np.array(((0, 0), (.25, .75), (1, 1)), float)
    xyz = evaluate_prepared_authored_face(prepared, correspondence, uv)
    du, dv = evaluate_prepared_authored_face(prepared, correspondence, uv, derivatives=True)
    print(dict(authored_faces=len(design.faces), prepared_faces=len(prepared.faces), root=root,
        descendants=correspondence.descendants, internal_station_edges=len(receipts),
        physical_joint_ids=tuple(edge.id for edge in application.joint_edges),
        point_shape=xyz.shape, derivative_shapes=(du.shape, dv.shape), accepted_mesh=False))
