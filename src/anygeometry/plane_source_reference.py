"""True implicit boundary images in an affine Plane chart.

No closest-point inversion or polygon substitution establishes material.
These maps feed the existing oriented source-cell complex proof.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
from types import SimpleNamespace

from .errors import GeometryError
from .surfaces import Plane
from .curves import Straight, LineRange
from .support_curves import PlaneNurbsCurve
from .regular_source_reference import SourceNode, SourceSide, source_node, source_side, parameter_transport_bound
from .planar_exact import ExactPlane
from .nurbs_metric import add, mul, derivative_intervals
from .plane_nurbs import charge_box, charge_split, depth_limit
from .definition_binding import definition_checksum


def project(frame, box, direction=False):
    points = [(frame.uv_direction(p) if direction else frame.uv(p)) for p in product(*box)]
    return tuple((min(p[i] for p in points), max(p[i] for p in points)) for i in (0, 1))


@dataclass(frozen=True)
class PlaneSide:
    frame: object
    native: object
    start: object
    end: object

    @property
    def curve(self):
        return self.native.curve

    def box(self, lo, hi):
        if lo == hi == 0:
            return self.start.box
        if lo == hi == 1:
            return self.end.box
        from .curvilinear_reference_maps import _source_box
        return project(self.frame, _source_box(self.native.curve.surface, self.native.box(lo, hi)))

    def derivative(self, lo, hi):
        chart = self.native.box(lo, hi)
        rates = self.native.derivative(lo, hi)
        du, dv = derivative_intervals(self.native.curve.surface, chart)
        world = tuple(add(mul(a, rates[0]), mul(b, rates[1])) for a, b in zip(du, dv))
        return project(self.frame, world, direction=True)

    def transport_bound(self):
        # Projection onto the SAME generating plane is identity on the true
        # image; the existing world parameter-transport bound still applies.
        return parameter_transport_bound(self.native)


class PlaneReference:
    def __init__(self, model, face):
        self.model, self.face = model, face
        self.frame = ExactPlane.from_plane(face.surface)
        self.key = definition_checksum(face.surface)
        self.vertices, self.joints = {}, {}
        uses = [use for loop in (face.loop, *face.holes) for use in loop]
        if not {type(model.edges[use.edge].curve) for use in uses} <= {Straight, LineRange, PlaneNurbsCurve}:
            raise GeometryError('planar source curve family is not qualified')
        # A vertex where a straight piece and a section block meet is a MIXED joint: it is the canonical true
        # root of the section against the plate line (recomputed from complete definitions, the same symbol and
        # enclosure the NURBS protocol consumes); anything else mixed refuses.
        kinds_at = {}
        for use in uses:
            edge = model.edges[use.edge]
            for vertex in (edge.start, edge.end):
                kinds_at.setdefault(vertex, set()).add(type(edge.curve))
        from ._canonical_plate_joint import plate_joint
        for use in uses:
            edge = model.edges[use.edge]
            for t, vertex in ((Q(0), edge.start), (Q(1), edge.end)):
                if PlaneNurbsCurve in kinds_at[vertex] and kinds_at[vertex] & {Straight, LineRange}:
                    section = next(u.edge for u in uses if type(model.edges[u.edge].curve) is PlaneNurbsCurve
                                   and vertex in (model.edges[u.edge].start, model.edges[u.edge].end))
                    joint = plate_joint(model, vertex, section)
                    if joint is None:
                        raise GeometryError('mixed planar boundary true-joint reconstruction is not qualified')
                    self.joints[vertex] = joint
                    box = project(self.frame, tuple(zip(joint.lower, joint.upper)))
                    node = SourceNode(('plane_root', self.key, joint.identity), box, (joint.isolator[0] + joint.isolator[1])/2,
                                      joint.isolator)
                else:
                    node = self._node(edge, t, vertex)
                old = self.vertices.get(vertex)
                if old is not None and old.identity != node.identity:
                    raise GeometryError('planar source vertex has incompatible true root identities')
                self.vertices[vertex] = node

    def _node(self, edge, t, vertex):
        curve = edge.curve
        if type(curve) in (Straight, LineRange):
            if type(curve) is LineRange:
                curve.certify_plane(self.frame, min(1.e-8, self.model.tolerance.effective_curve_fit_residual(1.)))
                points = (curve.exact_point(0), curve.exact_point(1))
            else:
                points = [tuple(map(Q, self.model.vertex_position(v))) for v in (edge.start, edge.end)]
            world = tuple(a+t*(b-a) for a, b in zip(*points))
            uv = self.frame.uv(world)
            key = ('plane_vertex', self.key, self.face.id, vertex) if vertex is not None else ('plane_line_station', self.key, edge.id, t)
            return SourceNode(key, tuple((x, x) for x in uv))
        if not isinstance(curve, PlaneNurbsCurve):
            raise GeometryError('planar source curve family is not qualified')
        original = source_node(curve, t, vertex=vertex)
        from .curvilinear_reference_maps import _source_box
        box = project(self.frame, _source_box(curve.surface, original.box))
        key = ('plane_root', self.key, original.identity)
        # At a declared transverse tensor-profile joint, derive the SAME
        # exact algebraic world point on every incident strip. Vertex IDs
        # select correspondence; coordinate equality only verifies it.
        s = curve.surface
        if (vertex is not None and s.degree_v == 1 and s.knots_v == (0., 0., 1., 1.)
                and s.v_range == (0., 1.) and all(len(r) == 2 for r in s.controls)):
            u = original.box[0]
            if u in ((Q(0), Q(0)), (Q(1), Q(1))):
                d = tuple(Q(b)-Q(a) for a, b in zip(s.controls[0][0], s.controls[0][1]))
                if (all(tuple(Q(b)-Q(a) for a, b in zip(r[0], r[1])) == d for r in s.controls)
                        and (s.weights is None or all(w[0] == w[1] for w in s.weights))):
                    denominator = sum(a*b for a, b in zip(self.frame.normal, d))
                    if denominator:
                        p = s.exact_point(u[0], Q(0))
                        v = (self.frame.offset()-sum(a*b for a, b in zip(self.frame.normal, p)))/denominator
                        if not original.box[1][0] <= v <= original.box[1][1]:
                            raise GeometryError('transverse true joint lies outside its certified source root')
                        world = tuple(a+v*b for a, b in zip(p, d))
                        uv = self.frame.uv(world)
                        box = tuple((x, x) for x in uv)
                        key = ('plane_vertex', self.key, self.face.id, vertex, world)
        return SourceNode(key, box)

    def node(self, edge_id, t, vertex=None):
        if vertex is not None:
            return self.vertices[vertex]
        return self._node(self.model.edges[edge_id], Q(t), None)

    def side(self, edge_id, parameters, first, last):
        edge = self.model.edges[edge_id]
        if type(edge.curve) in (Straight, LineRange):
            # the stored segment ends at the station vertex q while the TRUE source joint is the root: charge the
            # L1 diameter of the union of q and the root's world enclosure at every joint end (exact, no new slack)
            from ._canonical_plate_joint import joint_endpoint_distance
            transport = sum((joint_endpoint_distance(self.model, vertex, self.joints[vertex])
                             for vertex in (edge.start, edge.end) if vertex in self.joints), Q(0))
            if type(edge.curve) is LineRange:
                transport = Q(0)
                for parameter, vertex in ((0, edge.start), (1, edge.end)):
                    raw = edge.curve.exact_point(parameter)
                    projected = tuple(self.frame.world(*self.frame.uv(raw)))
                    proxy = tuple(map(Q, self.model.vertex_position(vertex)))
                    joint = self.joints.get(vertex)
                    low = joint.lower if joint is not None else projected
                    high = joint.upper if joint is not None else projected
                    transport += sum(max(hi, r, s, q)-min(lo, r, s, q)
                                     for lo, hi, r, s, q in zip(low, high, raw, projected, proxy))
            return SourceSide(first, last, endpoint_transport=transport)
        original = []
        for t in parameters:
            vertex = edge.start if t == 0 else edge.end if t == 1 else None
            node = source_node(edge.curve, t, vertex=vertex)
            if vertex is not None:
                # The native chart-coordinate side must start/end at the SAME true joint the plane node uses,
                # so its interior map and derivatives are transported from the canonical isolator's chart
                # enclosure (not merely its endpoint box overridden afterwards). Regenerated, never trusted.
                from ._canonical_plate_joint import plate_joint
                from .regular_source_reference import _joint_source_node
                joint = plate_joint(self.model, vertex, edge_id)
                if joint is not None:
                    node = _joint_source_node(edge.curve, joint)
            original.append(node)
        return PlaneSide(self.frame, source_side(edge.curve, parameters, *original), first, last)


def contains_section_loop(plane, loop, point, tolerance, *, boundary=True):
    """Certified contact followed by exact branch-root parity for pure loops."""
    from .nurbs_domain import ChartContainmentUndecided
    from .plane_section_plate import _contains
    from .entities import OrientedEdge
    frame = ExactPlane.from_plane(plane)
    curves = [p.curve for p in loop]
    if not all(isinstance(c, PlaneNurbsCurve) for c in curves):
        raise GeometryError('mixed planar implicit containment is not qualified')
    world = frame.world(*point)
    tol2 = Q(float(tolerance))**2
    for curve in curves:
        pending = [(Q(0), Q(1), 0)]
        while pending:
            lo, hi, depth = pending.pop()
            charge_box()
            lower, upper = curve.bounds(float(lo), float(hi))
            box = tuple((Q(float(a)), Q(float(b))) for a, b in zip(lower, upper))
            gap = sum(max(a-x, x-b, Q(0))**2 for (a, b), x in zip(box, world))
            if gap > tol2:
                continue
            radius = sum(max(abs(a-x), abs(b-x))**2 for (a, b), x in zip(box, world))
            if radius <= tol2:
                return boundary
            if depth >= depth_limit(32):
                raise ChartContainmentUndecided('planar implicit boundary contact is undecided')
            charge_split()
            mid = (lo+hi)/2
            pending.extend(((lo, mid, depth+1), (mid, hi, depth+1)))
    proxy = SimpleNamespace(edges={i: SimpleNamespace(curve=c) for i, c in enumerate(curves)})
    return _contains(proxy, tuple(OrientedEdge(i, True) for i in range(len(curves))), tuple(map(Q, point)), frame, lambda: None)
