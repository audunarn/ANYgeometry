"""True source stations and regular-image parameterizations of section paths.

Public sampled parameters remain approximation targets. These proof-only
curves use the same source image with shared algebraic arc-end stations.
"""
from dataclasses import dataclass
from fractions import Fraction as Q

from .errors import GeometryError
from .nurbs import CurveOnSurface
from .support_curves import PlaneNurbsCurve
from .definition_binding import definition_checksum
from .plane_nurbs import charge_box
from .nurbs_metric import add, sub, mul, scale
from .nurbs_section_faces import _Chart
from .branch_predicates import _Arc


def _plate_joint(owner, vertex, edge_id):
    """The canonical plate-line joint at a section vertex shared with a straight plate piece, else ``None``."""
    if not hasattr(owner, 'topological_edges_using_vertex'):
        return None
    from ._canonical_plate_joint import plate_joint
    return plate_joint(owner, vertex, edge_id)


def _joint_source_node(curve, joint, path_parameter=None):
    """The NURBS-protocol node of a canonical plate joint: the chart enclosure of the unique root isolator,
    with the isolator midpoint as its path parameter and the WHOLE isolator as ``path_interval`` (the stored
    edge ends at an older proxy; ``parameter_transport_bound`` charges the distance from every point of this
    interval to the nominal proxy, so the full root uncertainty reaches the surface-error budget).
    ``path_parameter`` is accepted for call compatibility and ignored."""
    from .nurbs_section_faces import _piece_rect
    low, high = joint.isolator
    rect = _piece_rect(joint.full_curve, _Chart(curve.surface), float(low), float(high))
    return SourceNode(joint.identity, tuple((Q(lo), Q(hi)) for lo, hi in rect), (low + high)/2, (low, high))


def _constant(x):
    return x, x


def _blend(a, b, interval):
    lo, hi = interval
    values = [x*(1-t)+y*t for x in a for y in b for t in (lo, hi)]
    return min(values), max(values)


@dataclass(frozen=True)
class SourceNode:
    identity: tuple
    box: tuple
    path_parameter: object = None
    # private transient (never persisted, defaulted so open records are unchanged): the exact path-parameter
    # interval of a canonical plate joint's isolating root; its WHOLE width enters the transport budget
    path_interval: tuple = None


def _section_key(curve):
    return curve.plane_record, definition_checksum(curve.surface), curve.path


def _root_node(curve, root, parameter):
    box = _Chart(curve.surface).rect(*curve._root_rect(root))
    identity = ('arc_root', *_section_key(curve), root.axis, root.value,
                tuple(root.poly), tuple(root.edge), root.patch, root.ident)
    return SourceNode(identity, box, parameter)


def _path_node(curve, parameter, *, encoded_joint=False):
    arcs = curve._arcs()
    count = len(arcs)
    for k in range(count+1):
        joint = Q(k, count)
        if parameter == joint or (encoded_joint and float(parameter) == float(joint)):
            root = arcs[0].first if k == 0 else arcs[k-1].second
            return _root_node(curve, root, joint)
    k = min(count-1, int(parameter*count))
    arc = arcs[k]
    s = arc.s0+(parameter*count-k)*(arc.s1-arc.s0)
    for root, joint in ((arc.first, Q(k, count)), (arc.second, Q(k+1, count))):
        rs, _ = curve._ordered_root_rect(arc, root)
        if rs[0] <= s <= rs[1]:
            return _root_node(curve, root, joint)
    view = _view(arc)
    bracket = view.bracket(s, Q(1, 2**100))
    if bracket is None:
        raise GeometryError('true source station is not on its regular section graph')
    box = _Chart(curve.surface).rect(*view.uv((s, s), bracket))
    return SourceNode(('section_slice', *_section_key(curve), k, s), box, parameter)


def source_node(curve, parameter, *, vertex=None):
    parameter = Q(parameter)
    if isinstance(curve, CurveOnSurface):
        point = tuple(curve.pcurve.exact_point(parameter))
        return SourceNode(('chart_point', definition_checksum(curve.support), *point), tuple((x, x) for x in point))
    if isinstance(curve, PlaneNurbsCurve):
        a, b = map(Q, curve.domain)
        return _path_node(curve, a+(b-a)*parameter, encoded_joint=vertex is not None)
    raise GeometryError('true source station curve family is not qualified')


class NurbsReference:
    """Declared source vertices, including exact axis/section root joints.

    An axis line selects a root by its exact constant chart coordinate and
    the incident section definition. No approximate endpoint selects identity.
    """
    def __init__(self, owner, face):
        self.owner, self.face = owner, face
        self.vertices = {}
        uses = [use for loop in (face.loop, *face.holes) for use in loop]
        for use in uses:
            edge = owner.edges[use.edge]
            if isinstance(edge.curve, PlaneNurbsCurve):
                for t, vertex in ((Q(0), edge.start), (Q(1), edge.end)):
                    node = source_node(edge.curve, t, vertex=vertex)
                    joint = _plate_joint(owner, vertex, use.edge)
                    if joint is not None:
                        node = _joint_source_node(edge.curve, joint)
                    old = self.vertices.get(vertex)
                    if old is not None and old.identity != node.identity:
                        raise GeometryError('section/section true joint reconstruction is not qualified')
                    self.vertices[vertex] = node
        for use in uses:
            edge = owner.edges[use.edge]
            if isinstance(edge.curve, CurveOnSurface):
                for t, vertex in ((Q(0), edge.start), (Q(1), edge.end)):
                    native = source_node(edge.curve, t, vertex=vertex)
                    if vertex in self.vertices and self.vertices[vertex].identity != native.identity:
                        node, p = self.vertices[vertex], edge.curve.pcurve
                        if p.degree != 1 or len(p.controls) != 2:
                            raise GeometryError('axis source boundary requires a rational segment')
                        axes = [i for i in (0, 1) if Q(p.controls[0][i]) == Q(p.controls[1][i])]
                        if not any(node.box[i] == (Q(p.controls[0][i]),)*2 for i in axes):
                            raise GeometryError('source joint is not the certified exact axis/section root')
                    else:
                        self.vertices[vertex] = native

    def node(self, edge_id, parameter, vertex=None):
        if vertex is None:
            return source_node(self.owner.edges[edge_id].curve, parameter)
        canonical = self.vertices[vertex]
        edge = self.owner.edges[edge_id]
        if (isinstance(edge.curve, PlaneNurbsCurve)
                and ((parameter == 0 and edge.start == vertex) or (parameter == 1 and edge.end == vertex))):
            # One topological vertex, one canonical identity and box; but each incident edge occurrence
            # keeps ITS OWN endpoint path parameter (a closed seam is the same root at path parameters 0
            # and 1). The occurrence is derived from the exact owner edge/curve and accepted only when it
            # is the same true source station; otherwise the canonical node is returned unchanged.
            occurrence = source_node(edge.curve, Q(parameter), vertex=vertex)
            if canonical.identity[0] == 'plate_joint':
                # a canonical plate joint is the TRUE root shared with the plate protocol; the node carries the
                # root's own midpoint and whole isolator (the transient interval is retained by this copy)
                return SourceNode(canonical.identity, canonical.box, canonical.path_parameter,
                                  canonical.path_interval)
            if occurrence.identity == canonical.identity:
                return SourceNode(canonical.identity, canonical.box, occurrence.path_parameter,
                                  occurrence.path_interval)
        return canonical

    def side(self, edge_id, parameters, first, last):
        curve = self.owner.edges[edge_id].curve
        if isinstance(curve, PlaneNurbsCurve):
            return source_side(curve, parameters, first, last)
        side = source_side(curve, parameters, first, last)
        # Geometric-line reparameterization, with full-span endpoint image
        # transport. Its derivative correction is included in side.derivative
        # and hence in the strict fan Jacobian, not silently discarded.
        delta = Q(0)
        for t, node in zip(parameters, (first, last)):
            point = curve.pcurve.exact_point(t)
            delta = max(delta, sum(max(abs(a-Q(x)), abs(b-Q(x))) for (a, b), x in zip(node.box, point)))
        if delta:
            from .surface_capabilities import capabilities
            sigma = Q(float(capabilities(curve.support).jacobian_bounds(((0., 1.), (0., 1.))).sigma_max))
            side = SourceSide(first, last, endpoint_transport=delta*sigma)
        return side


def _view(arc):
    view = object.__new__(_Arc)
    view.arc = arc
    view.slicing, view.u_box, view.v_box = arc.slicing, arc.box.u_range, arc.box.v_range
    view.x_box = arc.box.v_range if arc.slicing == 0 else arc.box.u_range
    view.g1 = arc.box.coefficients
    # rectangle uses the curve's established true-root bracketing methods.
    return view


def _point_rectangle(view, point):
    """A knot rectangle ``(s, x)`` containing the section point at the exact slicing value ``point``.

    ``_Arc.rectangle`` only brackets the graph when ``s_hi > s_lo``, so a zero-width slice fell back to the
    arc's whole transverse x range. Here the point is bracketed exactly: on the line ``s = point`` the
    column polynomial ``G1(point, .)`` is strictly monotone in x on a regular box (``G_x`` has one sign),
    its values at the two box faces have opposite signs inside the arc's gap, and exact bisection by the
    sign of ``G1`` (Fractions, no float) keeps the unique zero inside an x interval of width ``2**-100``
    in knot units, the same contract as the true source nodes (``_path_node``). Where the slice is outside
    the gap (``bracket`` is ``None``) the established ``rectangle`` fallback, including its end-root
    windows, is used unchanged. When a root's slicing interval contains the point, that root's bracket is
    also included (the section may sit at the true end root there), exactly as ``rectangle`` does.
    The step is charged like any other examined box.
    """
    charge_box()
    bracket = view.bracket(point, Q(1, 2**100))
    if bracket is None:
        return view.rectangle(point, point, parent=view.x_box)
    s_range, x_range = (point, point), bracket
    target = (view.x_box[1]-view.x_box[0])/2**60
    for root in (view.arc.first, view.arc.second):
        rs, _rx = view.curve._ordered_root_rect(view.arc, root)
        if rs[1] < point or rs[0] > point:
            continue
        (bs, bx) = view.root_bracket(root, target)
        s_range = (min(s_range[0], bs[0]), max(s_range[1], bs[1]))
        x_range = (min(x_range[0], bx[0]), max(x_range[1], bx[1]))
    return s_range, x_range


@dataclass(frozen=True)
class SourceSide:
    start: SourceNode
    end: SourceNode
    curve: object = None
    pieces: tuple = ()
    parameters: tuple = ()
    endpoint_transport: object = Q(0)

    def _ranges(self, lo, hi):
        if self.curve is None:
            return tuple(_blend(a, b, (lo, hi)) for a, b in zip(self.start.box, self.end.box)), \
                   tuple(sub(b, a) for a, b in zip(self.start.box, self.end.box))
        chart = _Chart(self.curve.surface)
        boxes, derivatives = [], []
        for t0, t1, arc, first, last in self.pieces:
            a, b = max(lo, t0), min(hi, t1)
            if a > b:
                continue
            local = (a-t0)/(t1-t0), (b-t0)/(t1-t0)
            axis = arc.slicing
            source_range = (self.curve.surface.u_range, self.curve.surface.v_range)[axis]
            origin, span = Q(source_range[0]), Q(source_range[1])-Q(source_range[0])
            s0 = add(_constant(origin), scale(first.box[axis], span))
            s1 = add(_constant(origin), scale(last.box[axis], span))
            s = _blend(s0, s1, local)
            ds = scale(sub(s1, s0), 1/(t1-t0))
            if ds[0] <= 0 <= ds[1]:
                raise GeometryError('true section endpoint order is not certified')
            view = _view(arc)
            view.curve = self.curve
            if s[0] == s[1]:
                s, x = _point_rectangle(view, s[0])
            else:
                s, x = view.rectangle(*s, parent=view.x_box)
            du, dv = view.derivative_ranges(view.g1, s, x)
            gs, gx = (du, dv) if axis == 0 else (dv, du)
            if gx[0] <= 0 <= gx[1]:
                raise GeometryError('true source image derivative is not regular')
            dx = scale(mul(mul(gs, tuple(sorted((1/gx[0], 1/gx[1])))), ds), -1)
            physical = (ds, dx) if axis == 0 else (dx, ds)
            derivatives.append(tuple(scale(d, 1/(Q(r[1])-Q(r[0]))) for d, r in
                                     zip(physical, (self.curve.surface.u_range, self.curve.surface.v_range))))
            boxes.append(chart.rect(*view.uv(s, x)))
        if not boxes:
            raise GeometryError('true source reference interval is empty')
        combine = lambda rows: tuple((min(row[i][0] for row in rows), max(row[i][1] for row in rows)) for i in (0, 1))
        return combine(boxes), combine(derivatives)

    def box(self, lo, hi):
        if lo == hi == 0:
            return self.start.box
        if lo == hi == 1:
            return self.end.box
        return self._ranges(lo, hi)[0]

    def derivative(self, lo, hi):
        return self._ranges(lo, hi)[1]


def source_side(curve, parameters, first, last):
    if isinstance(curve, CurveOnSurface):
        p = curve.pcurve
        if p.degree != 1 or len(p.controls) != 2:
            raise GeometryError('curved polynomial source pcurves are not yet qualified')
        # A rational segment has the same oriented image as its endpoint
        # chord. Reference parameterization is independent of sample speed.
        return SourceSide(first, last)
    if not isinstance(curve, PlaneNurbsCurve):
        raise GeometryError('source reference side family is not qualified')
    a, b = first.path_parameter, last.path_parameter
    if a == b:
        raise GeometryError('source boundary interval collapses at a true arc station')
    arcs = curve._arcs()
    count = len(arcs)
    cuts = sorted({Q(0), Q(1), *((Q(k, count)-a)/(b-a) for k in range(count+1)
                               if 0 < (Q(k, count)-a)/(b-a) < 1)})
    pieces = []
    for lo, hi in zip(cuts, cuts[1:]):
        middle = a+(b-a)*(lo+hi)/2
        index = min(count-1, int(middle*count))
        start = first if lo == 0 else _path_node(curve, a+(b-a)*lo)
        end = last if hi == 1 else _path_node(curve, a+(b-a)*hi)
        pieces.append((lo, hi, arcs[index], start, end))
    return SourceSide(first, last, curve, tuple(pieces), tuple(parameters))


def parameter_transport_bound(side):
    """World bound of true-endpoint transport versus the P2 parameter map.

    Piecewise graph Lipschitz bounds cover endpoint slicing displacement;
    all certified arc-joint enclosures cover possible nominal-map jumps.
    This changes reference speed, never the source image.
    """
    from .plane_source_reference import PlaneSide
    if type(side) is PlaneSide:
        return side.transport_bound()
    if side.curve is None:
        return Q(side.endpoint_transport)
    import math
    from .surface_capabilities import capabilities
    from .exact_predicates import correctly_rounded_sqrt
    curve = side.curve
    arcs = curve._arcs()
    a, b = map(Q, curve.domain)
    nominal = tuple(a+(b-a)*t for t in side.parameters)

    def spread(node, old):
        # a canonical joint's WHOLE isolating interval (not only its midpoint) deviates from the nominal proxy
        low, high = node.path_interval if node.path_interval is not None else (node.path_parameter,)*2
        return max(abs(node.path_parameter-old), abs(low-old), abs(high-old))

    shift = max(spread(node, old) for node, old in zip((side.start, side.end), nominal))
    sigma = Q(float(capabilities(curve.surface).jacobian_bounds(((0., 1.), (0., 1.))).sigma_max))
    maximum, path_lipschitz, joints = Q(0), Q(0), Q(0)
    for arc in arcs:
        view = _view(arc)
        s = arc.box.u_range if arc.slicing == 0 else arc.box.v_range
        x = arc.box.v_range if arc.slicing == 0 else arc.box.u_range
        du, dv = view.derivative_ranges(view.g1, s, x)
        gs, gx = (du, dv) if arc.slicing == 0 else (dv, du)
        if gx[0] <= 0 <= gx[1]:
            raise GeometryError('source parameter transport graph is not regular')
        slope = max(abs(v) for v in mul(gs, tuple(sorted((1/gx[0], 1/gx[1])))))
        spans = tuple(abs(Q(r[1])-Q(r[0])) for r in (curve.surface.u_range, curve.surface.v_range))
        lip = 1/spans[arc.slicing]+slope/spans[1-arc.slicing]
        width = Q(0)
        for root in (arc.first, arc.second):
            sr, _xr = curve._ordered_root_rect(arc, root)
            width = max(width, sr[1]-sr[0])
            box = _Chart(curve.surface).rect(*curve._root_rect(root))
            joints += sum(hi-lo for lo, hi in box)
        maximum = max(maximum, lip*width)
        path_lipschitz = max(path_lipschitz, lip*len(arcs)*abs(arc.s1-arc.s0))
    delta2 = curve._world_bound()[3]
    delta = Q(math.nextafter(correctly_rounded_sqrt(delta2), math.inf))
    return sigma*(maximum+path_lipschitz*shift+2*joints)+2*delta
