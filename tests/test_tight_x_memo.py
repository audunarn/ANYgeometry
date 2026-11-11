"""``_tight_x`` is memoised per plan (value key, cache owned by the plan's Work): repeated boxes in one
plane x NURBS plan are charged once (kernel strategy review proposal 1)."""
from anygeometry import GeometryModel, Plane, plan_plane_split
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anygeometry.predicates import IntersectionQualificationPolicy


def _bowl():
    a = (1., -1., 1.)
    return NurbsSurface(2, 2, (0., 0., 0., 1., 1., 1.), (0., 0., 0., 1., 1., 1.),
                        tuple(tuple((i - 1., j - 1., a[i] + a[j]) for j in range(3)) for i in range(3)))


def _square_face():
    surface, model = _bowl(), GeometryModel()
    pts = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
    vs = [model.add_point(*surface.evaluate(*p)) for p in pts]
    edges = [model.add_curve(vs[i], vs[(i + 1) % 4],
                             CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (pts[i], pts[(i + 1) % 4]))))
             for i in range(4)]
    return model, model.add_face(edges, surface=surface)


def _plan(z, cap):
    model, face = _square_face()
    return plan_plane_split(model, [face], Plane((0., 0., z), (1., 0., 0.), (0., 1., 0.)),
                            qualification=IntersectionQualificationPolicy(max_boxes_per_pair=cap))


def test_repeated_tight_x_boxes_are_not_recharged():
    # refused with 3000 boxes before the memo (about 8000 were needed); the result is the same plan
    plan = _plan(.5, 3000)
    assert plan is not None

