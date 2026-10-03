"""Analytically constructed double-fold wall/pipe; no mesher execution."""
from anygeometry import (
    GeometryModel, IntersectionBatchPolicy, BezierQuadricCurve,
    plan_intersections, apply_intersections, to_dict,
    query_prepared_authored_boundary_correspondence,
)
from anygeometry.generators import cylinder


def build(*, cancellation_check=None):
    # B(t)=(3t,25/16-3t+3t^2,-1), D=(0,0,2).
    # Unit pipe equation gives z^2=1-B_y(t)^2. Its two folds
    # t=1/4,3/4 lie inside the wall, with |angle|<pi/4.
    design = GeometryModel()
    points = design.add_points(((0, 25/16, -1), (1, 9/16, -1),
                                (2, 9/16, -1), (3, 25/16, -1)))
    edge = design.add_spline(points[0], points[1:-1], points[-1])
    root = design.extrude((edge,), (0, 0, 2))[0]
    design.insert_model(cylinder(1, 5, origin=(-1, 0, 0), axis=(1, 0, 0),
        radial_direction=(0, 1, 0), circumferential_segments=8))
    source = to_dict(design)
    prepared = design.clone(preserve_identity=True)
    policy = IntersectionBatchPolicy(cancellation_check=cancellation_check)
    plan = plan_intersections(prepared, tuple(prepared.faces), policy=policy)
    application = apply_intersections(prepared, plan, policy=policy)
    correspondence = query_prepared_authored_boundary_correspondence(prepared, root,
        cancellation_check=cancellation_check)
    assert to_dict(design) == source
    return design, prepared, root, application, correspondence


def double_fold_edges(prepared, correspondence):
    return tuple(edge for edge, _ in correspondence.interior_incidence
        if type(prepared.edges[edge].curve) is BezierQuadricCurve
        and prepared.edges[edge].curve.parameterization == 'both_sine')


if __name__ == '__main__':
    design, prepared, root, application, correspondence = build()
    print(dict(authored_faces=len(design.faces), prepared_faces=len(prepared.faces),
        root=root, both_sine_edges=double_fold_edges(prepared, correspondence),
        joint_ids=tuple(edge.id for edge in application.joint_edges), accepted_mesh=False))
