"""Canonical exact plate-line joints (bounded sections, conservative route C).

A plate-boundary station is the TRUE root of a regular section path against the plate edge's exact line. This
module (1) chooses the single float station vertex ``q`` that lies EXACTLY on the original 3D plate line and on
the plate support and inside the root's exact world enclosure, and (2) re-derives, from declared topology and
complete exact definitions only, the one canonical joint symbol both source protocols (NURBS and Plane) must
consume. Nothing here is persisted or trusted: a loaded model regenerates the joint from its definitions, and
metadata, closeness or the same vertex id never grant identity.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as Q
import math
from itertools import product

from .curves import Straight, LineRange
from .definition_binding import definition_checksum
from .errors import GeometryError
from .support_curves import PlaneNurbsCurve


class JointRefused(GeometryError):
    """A plate joint is not exactly representable or its identity/clearance cannot be certified."""


def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def exact(position):
    return tuple(Q(float(x)) for x in position)


def line_key(normal, offset):
    """The plate-chart line equation normalised by its first nonzero normal component."""
    pivot = next(x for x in normal if x != 0)
    return tuple(x/pivot for x in normal) + (offset/pivot,)


def choose_station(a, b, plate_normal, plate_offset, lower, upper, effective_length, *, retained=False):
    """The float point ``q`` on the EXACT 3D line through the original plate edge ``a -> b`` (Fractions), on the
    plate support, inside the root's world enclosure ``[lower, upper]`` whose diameter is at most the existing
    ``effective_length``. Returns ``(q_floats, lam)`` with ``q = a + lam (b - a)`` exactly, or refuses.

    No tolerance is added: if no float coordinate triple lies exactly on the line (general oblique edges)
    the case is refused before anything is published."""
    if retained:
        norm2 = _dot(plate_normal, plate_normal)
        projected = [tuple(x - n*(_dot(plate_normal, p)-plate_offset)/norm2
                           for x, n in zip(p, plate_normal)) for p in (a, b)]
        sa, sb = projected
        e = _sub(sb, sa)
        length2 = _dot(e, e)
        if not length2:
            raise JointRefused('degenerate projected plate carrier')
        for p in (a, b):
            if (_dot(plate_normal, p)-plate_offset)**2 > Q(1.e-8)**2*norm2:
                raise JointRefused('plate carrier exceeds native support residual')
        centre = tuple((lo+hi)/2 for lo, hi in zip(lower, upper))
        lam = _dot(_sub(centre, sa), e)/length2
        if not 0 < lam < 1:
            raise JointRefused('plate carrier station is outside its retained range')
        raw = tuple(x+lam*(y-x) for x, y in zip(a, b))
        projected_point = tuple(x+lam*y for x, y in zip(sa, e))
        q = tuple(float(x) for x in raw)
        lows = tuple(min(lo, r, s, Q(f)) for lo, r, s, f in zip(lower, raw, projected_point, q))
        highs = tuple(max(hi, r, s, Q(f)) for hi, r, s, f in zip(upper, raw, projected_point, q))
        if sum(hi-lo for lo, hi in zip(lows, highs)) > Q(effective_length):
            raise JointRefused('carrier, projection, topology and root exceed joint tolerance')
        return q, lam
    if _dot(plate_normal, a) != plate_offset or _dot(plate_normal, b) != plate_offset:
        raise JointRefused('plate edge endpoints are not exactly on the plate support')
    # creation guarantees the L1 precision the source node-residual check will demand (never relaxes it)
    if sum(hi - lo for lo, hi in zip(lower, upper)) > Q(effective_length):
        raise JointRefused('the root enclosure is wider than the existing length tolerance')
    e = _sub(b, a)
    if not any(e):
        raise JointRefused('degenerate plate edge')
    pivot = max(range(3), key=lambda k: abs(e[k]))
    centre = float((lower[pivot] + upper[pivot])/2)
    candidates = [centre]
    low_f, high_f = centre, centre
    for _ in range(96):
        low_f, high_f = math.nextafter(low_f, -math.inf), math.nextafter(high_f, math.inf)
        candidates.extend((high_f, low_f))
    for y in candidates:
        if not lower[pivot] <= Q(y) <= upper[pivot]:
            continue
        lam = (Q(y) - a[pivot])/e[pivot]
        if not 0 < lam < 1:
            continue
        q = tuple(a[k] + lam*e[k] for k in range(3))
        if all(Q(float(x)) == x for x in q) and all(lo <= x <= hi for x, lo, hi in zip(q, lower, upper)):
            return tuple(float(x) for x in q), lam
    raise JointRefused('no exactly representable station lies on the plate line inside the root enclosure')


def _certified_world_bounds(curve, lo, hi):
    """Examine one finite ordered world box from an explicitly admitted kernel."""
    from .nurbs import CurveOnSurface, NurbsCurve
    from .plane_nurbs import charge_box
    if type(curve) not in (CurveOnSurface, NurbsCurve, PlaneNurbsCurve):
        raise JointRefused('a source trim family has no certified world enclosure')
    charge_box()
    # The shared COS iso cache is not proof authority, including its cached
    # eligibility decision for a non-iso pcurve. Reconstruct without reading
    # or writing that cache; the other registered kernels keep their contract.
    low, high = (curve.bounds(lo, hi, fresh=True) if type(curve) is CurveOnSurface
                 else curve.bounds(lo, hi))
    if (len(low) != 3 or len(high) != 3
            or any(not math.isfinite(float(x)) for row in (low, high) for x in row)
            or any(a > b for a, b in zip(low, high))):
        raise JointRefused('a certified trim kernel returned an invalid world enclosure')
    return low, high


def _classify_gap(curve, lower, upper, a, b, check):
    """'in' / 'out' for the contact-free parameter gap ``(a, b)`` of the section path against the box.

    Private to ``local_certificate`` and valid ONLY there: the complete boundary-contact list (all six finite
    faces, isolated, strictly separated, nonzero endpoints) has already been verified, so the continuous
    mathematical section C* (authenticated owner path, exact root IDs - never the approximate P2 joint
    coordinates) does not meet the boundary of the box anywhere on the open gap. Its connected image therefore
    lies wholly strictly inside the box or wholly outside the closed box, and ONE certified witness decides the
    whole gap. The witness is an exact rational parameter whose OUTWARD float neighbourhood stays strictly inside
    ``(a, b)`` (otherwise the neighbourhood is halved, or the next of a small fixed set of rational fractions is
    tried); its world enclosure comes from the admitted kernel with an explicit Work box charge. Strictly inside
    or disjoint decides; the bounded halving charges ``charge_split`` and respects ``depth_limit(20)`` and the
    caller's ``check``. No decision: a typed refusal - never a guess, never an empty success. This is not a
    sampled tessellation or a general sampling helper."""
    from .nurbs import _float_down, _float_up
    from .plane_nurbs import charge_split, depth_limit
    width = b - a
    for fraction in (Q(1, 2), Q(1, 3), Q(2, 3), Q(1, 4), Q(3, 4)):
        centre, half = a + fraction*width, width/8
        for _ in range(depth_limit(20)):
            check()
            low_f, high_f = _float_down(centre - half), _float_up(centre + half)
            if not (a < Q(low_f) and Q(high_f) < b and low_f >= 0. and high_f <= 1.):
                half /= 2                       # the outward-rounded neighbourhood must be strictly inside (a, b)
                charge_split()
                continue
            low, high = _certified_world_bounds(curve, low_f, high_f)
            low, high = tuple(Q(float(x)) for x in low), tuple(Q(float(x)) for x in high)
            if all(lo < l and h < up for l, h, lo, up in zip(low, high, lower, upper)):
                return 'in'
            if any(h < lo or l > up for l, h, lo, up in zip(low, high, lower, upper)):
                return 'out'
            half /= 2
            charge_split()
    raise JointRefused('no certified interior witness classifies the contact-free gap (or the budget is exhausted)')


def _edge_piece_clear(curve, lower, upper, lo, hi, check, depth=0):
    from .nurbs import _float_down, _float_up
    from .plane_nurbs import charge_split, depth_limit
    check()
    low, high = _certified_world_bounds(curve, max(0., _float_down(lo)), min(1., _float_up(hi)))
    if any(Q(float(h)) < l or Q(float(w)) > u for w, h, l, u in zip(low, high, lower, upper)):
        return True
    if depth >= depth_limit(24):
        raise JointRefused('a source trim is not certified clear of the joint neighbourhood '
                           '(or the subdivision budget is exhausted)')
    charge_split()
    middle = (lo + hi)/2
    return (_edge_piece_clear(curve, lower, upper, lo, middle, check, depth + 1)
            and _edge_piece_clear(curve, lower, upper, middle, hi, check, depth + 1))


def edge_clear(model, edge_id, lower, upper, check):
    """Strict disjointness of an actual model edge from the box (its conservative existing world enclosures:
    exact endpoint box for a straight edge, ``curve.bounds`` for CurveOnSurface / NurbsCurve / PlaneNurbsCurve,
    refined by bounded subdivision). Any other family refuses; a trim touching the box refuses."""
    curve = model.edges[edge_id].curve
    if type(curve) in (Straight, LineRange):
        from .plane_nurbs import charge_box
        check()
        charge_box()
        item = model.edges[edge_id]
        points = ([curve.exact_point(0), curve.exact_point(1)] if type(curve) is LineRange
                  else [exact(model.vertex_position(v)) for v in (item.start, item.end)])
        low = tuple(min(p[k] for p in points) for k in range(3))
        high = tuple(max(p[k] for p in points) for k in range(3))
        if any(h < l or w > u for w, h, l, u in zip(low, high, lower, upper)):
            return
        raise JointRefused('a straight source trim is not clear of the joint neighbourhood')
    if not _edge_piece_clear(curve, lower, upper, Q(0), Q(1), check):
        raise JointRefused('a source trim meets the joint neighbourhood')


def trims_clear(model, face_id, lower, upper, own_path, check):
    """Every boundary edge of the NURBS face (outer, holes, prior section trims, new cut edges) must be strictly
    clear of the box, EXCEPT section edges of the joint's own path and plane (``own_path`` = (plane_record,
    path, surface checksum)): those are the authenticated local curve segments, whose behaviour inside the box
    is exactly what ``local_certificate`` accounts for the whole path. Identity here is the complete curve
    definition, never a label or proximity."""
    face = model.faces[face_id]
    for loop in (face.loop, *face.holes):
        for use in loop:
            curve = model.edges[use.edge].curve
            if (isinstance(curve, PlaneNurbsCurve) and own_path is not None
                    and (curve.plane_record, curve.path, definition_checksum(curve.surface)) == own_path):
                continue
            edge_clear(model, use.edge, lower, upper, check)


def local_certificate(full, isolator, lower, upper, normal, offset, tolerance, closed, proxies=(), check=None):
    """Conservative certificate that the section path is a single regular transverse branch through the
    canonical joint inside its neighbourhood ``B`` (the root's exact world enclosure expanded by the existing
    length tolerance, no other slack).

    Proof: every contact of the full path with the six exact axis planes of ``B`` is accounted with
    ``section_roots`` (an isolated simple root, whose other coordinates are certified inside the finite face's
    ranges; roots certified beyond the face are not contacts; straddling/tangent/non-isolated ones refuse). The
    contacts partition the parameter domain into pieces, none of which meets the boundary of ``B``, so each is
    wholly strictly inside or wholly clear of ``B`` (a closed path's first and last pieces are one component
    only by seam identity). Exactly ONE piece (the GAP) may be inside; it must contain the whole intended isolator
    AND every endpoint proxy path parameter STRICTLY in its interior, and every other piece is strictly outside.
    The inside component of the path physically extends into the two adjacent contact BANDS, so the proof runs
    over the full local ENVELOPE (the preceding contact's lower endpoint to the following contact's upper
    endpoint; the path ends 0 / 1 only where the gap touches the actual domain end of an open path), on an
    OUTWARD rounded float domain: the signed distance to the plate line must be strictly monotone over the whole
    envelope (regular transverse branch, no derivative reversal in a band) and the plate line must have exactly
    ONE root there, the intended one (an event in a band overlaps the envelope and refuses). Every recursion
    charges the active Work, is depth-bounded and calls ``check``; exhaustion is a typed refusal."""
    from .branch_predicates import projection_monotone, section_roots
    from .nurbs import _float_down, _float_up
    check = check or (lambda *_: None)
    tol = Q(tolerance)
    blo = tuple(x - tol for x in lower)
    bhi = tuple(x + tol for x in upper)
    contacts = []
    for axis in range(3):
        for value in (blo[axis], bhi[axis]):
            unit = tuple(Q(1) if k == axis else Q(0) for k in range(3))
            others = [k for k in range(3) if k != axis]
            for root in section_roots(full, unit, value):
                rl = [Q(float(root.lower[k])) for k in others]
                ru = [Q(float(root.upper[k])) for k in others]
                if any(h < blo[k] or l > bhi[k] for l, h, k in zip(rl, ru, others)):
                    continue                                          # beyond this finite face: not a contact
                if not root.isolated or len(root.exact) != 1 or not all(
                        blo[k] <= l and h <= bhi[k] for l, h, k in zip(rl, ru, others)):
                    raise JointRefused('a contact with the joint neighbourhood is tangent, ambiguous or on an edge')
                contacts.append(tuple(root.exact[0]))
    contacts.sort()
    for first, second in zip(contacts, contacts[1:]):
        if first[1] >= second[0]:
            raise JointRefused('two contacts with the joint neighbourhood are not separated')
    edges = [Q(0)] + [x for pair in contacts for x in pair] + [Q(1)]
    pieces = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
    if any(a >= b for a, b in pieces):
        raise JointRefused('a contact isolator reaches the end of the section path')
    status = [_classify_gap(full, blo, bhi, a, b, check) for a, b in pieces]
    intended = [i for i, (a, b) in enumerate(pieces) if a < isolator[0] and isolator[1] < b]
    if len(intended) != 1 or status[intended[0]] != 'in':
        raise JointRefused('the intended joint is not inside a strictly interior component of its neighbourhood')
    gap_lo, gap_hi = pieces[intended[0]]
    if any(not gap_lo < Q(p) < gap_hi for p in proxies):
        raise JointRefused('an endpoint proxy parameter is not strictly inside the joint gap')
    inside = [i for i, s in enumerate(status) if s == 'in']
    if closed and len(pieces) > 1 and status[0] == 'in' and status[-1] == 'in':
        # a component through the closed-path seam: its two end pieces are one component only by seam identity;
        # that is not certified here (conservative refusal)
        raise JointRefused('the joint neighbourhood component spans the closed-path seam')
    if inside != [intended[0]]:
        raise JointRefused('the section path re-enters the joint neighbourhood elsewhere')
    index = inside[0]
    envelope_lo = contacts[index - 1][0] if index > 0 else Q(0)             # preceding contact's LOWER endpoint
    envelope_hi = contacts[index][1] if index < len(contacts) else Q(1)     # following contact's UPPER endpoint
    domain = (max(0., _float_down(envelope_lo)), min(1., _float_up(envelope_hi)))
    check()
    if projection_monotone(replace(full, domain=domain), normal) is None:
        raise JointRefused('the signed distance to the plate line is not certified strictly monotone over the '
                           'whole local envelope')
    roots = [r for r in section_roots(full, normal, offset)
             if r.exact and envelope_lo <= r.exact[0][1] and r.exact[0][0] <= envelope_hi]
    if len(roots) != 1 or tuple(roots[0].exact[0]) != tuple(isolator):
        raise JointRefused('the local envelope does not hold exactly the intended plate-line event')
    return True


def joint_l1(q, lower, upper):
    """L1 diameter of the union of the point ``q`` and the box ``[lower, upper]`` (exact)."""
    return sum(max(hi, x) - min(lo, x) for x, lo, hi in zip(q, lower, upper))


def select_roots(roots):
    """The complete globally ordered root list: EVERY root must be one isolated simple root with separated
    isolators, otherwise the ordinal used in the symbol could shift under later refinement (typed refusal)."""
    for r in roots:
        if not r.isolated or len(r.exact) != 1:
            raise JointRefused('a root of the plate line is not isolated, so its ordinal would not be stable')
    ordered = sorted(roots, key=lambda r: tuple(r.exact[0]))
    for first, second in zip(ordered, ordered[1:]):
        if first.exact[0][1] >= second.exact[0][0]:
            raise JointRefused('two roots of the plate line are not separated')
    return ordered


def select_candidate(roots, q, tolerance):
    """The unique root that can correspond to the stored station ``q``: a root whose world enclosure is at a
    STRICT rational distance greater than the length tolerance from ``q`` is incompatible; every other root is
    possible; exactly one must remain (zero or several refuse). Nothing is nearest/first/welded."""
    tol2 = Q(tolerance)**2
    possible = []
    for index, root in enumerate(roots):
        lower = tuple(Q(float(x)) for x in root.lower)
        upper = tuple(Q(float(x)) for x in root.upper)
        gap2 = sum(max(lo - x, x - hi, Q(0))**2 for x, lo, hi in zip(q, lower, upper))
        if gap2 <= tol2:
            possible.append((index, root))
    if len(possible) != 1:
        raise JointRefused('the stored station does not correspond to exactly one root of the plate line')
    return possible[0]


def check_hull(full, proxies, isolator, tolerance, check):
    """Old incident proxies plus the new isolator: strictly inside ONE owner arc interval k/N .. (k+1)/N (a hull
    crossing an arc joint refuses in this slice), with certified world bounds of L1 diameter <= tolerance."""
    from .nurbs import _float_down, _float_up
    low = min((*proxies, isolator[0]))
    high = max((*proxies, isolator[1]))
    count = len(full._arcs())
    k = int(math.floor(low * count))
    if not (Q(k, count) < low and high < Q(k + 1, count)):
        raise JointRefused('the joint parameter hull is not strictly inside one owner arc interval')
    check()
    lower, upper = _certified_world_bounds(full, max(0., _float_down(low)), min(1., _float_up(high)))
    diameter = sum(Q(float(h)) - Q(float(l)) for l, h in zip(lower, upper))
    if diameter > Q(tolerance):
        raise JointRefused('the local endpoint movement hull is wider than the length tolerance')


def joint_endpoint_distance(model, vertex, joint):
    """Stored station -> true root: L1 diameter of the union of the vertex and the root's world enclosure."""
    return joint_l1(exact(model.vertex_position(vertex)), joint.lower, joint.upper)


@dataclass(frozen=True)
class PlateJoint:
    identity: tuple
    lower: tuple
    upper: tuple
    isolator: tuple
    full_curve: object
    proxies: tuple = ()


def plate_joint(model, vertex, edge_id, check=None):
    """The canonical joint at ``vertex`` between the section edge ``edge_id`` and the plate line, or ``None``
    when the vertex is not incident to a straight edge (an ordinary section/section or arc vertex).

    Derived afresh from complete exact definitions: the straight edges at the vertex must lie on ONE exact
    line (after the plate-chart projection) and the vertex must be exactly its projection; the unique complete
    isolator of the full section path against that line must contain the edge end's path parameter (as an exact
    Fraction) with the vertex inside its exact world enclosure, whose diameter is at most the existing length
    tolerance. Missing, multiple, tangent or non-isolated roots refuse."""
    from .branch_predicates import section_roots
    from .planar_exact import ExactPlane
    from .nurbs_sweep_chart import SweepChartModel
    if type(model) is SweepChartModel:
        from .model import GeometryModel
        if type(model.raw) is not GeometryModel:
            raise JointRefused('plate joint sweep view has no literal world model')
        # The explicitly requested edge still owes THIS face's chart
        # correspondence. Global incident edges, however, are world adjacency
        # and must never all be pulled through that one chart.
        model.edges[edge_id]
        model = model.raw
    edge = model.edges[edge_id]
    curve = edge.curve
    if not isinstance(curve, PlaneNurbsCurve):
        return None
    plane_key = definition_checksum(curve.plane)
    # only a straight edge that bounds a face lying on the SECTION'S OWN plane is a plate boundary line; other
    # straight edges at a section vertex (connectors of swept supports, trims) keep their existing protocols
    straights = [s for s in model.topological_edges_using_vertex(vertex) if type(model.edges[s].curve) in (Straight, LineRange)
                 and any(definition_checksum(model.faces[f].surface) == plane_key
                         for f in model.faces_using_edge(s))]
    if not straights:
        return None
    if edge.start == edge.end:
        raise JointRefused('a closed single-edge section has an ambiguous joint end')
    domain = tuple(Q(x) for x in curve.domain)
    path_parameter = domain[0] if edge.start == vertex else domain[1]
    frame = ExactPlane.from_plane(curve.plane)
    plate_normal = tuple(Q(x) for x in frame.normal)
    q = tuple(Q(x) for x in frame.world(*frame.uv(model.vertex_position(vertex))))
    if q != exact(model.vertex_position(vertex)) and not any(type(model.edges[s].curve) is LineRange for s in straights):
        raise JointRefused('the joint vertex is not exactly on the plate support')
    keys, carriers, endpoint_points, equation = set(), set(), [], None
    for s in straights:
        item = model.edges[s]
        if type(item.curve) is LineRange:
            item.curve.certify_plane(frame, min(1.e-8, model.tolerance.effective_curve_fit_residual(1.)))
            raw_a, raw_b = item.curve.exact_point(0), item.curve.exact_point(1)
            a = tuple(frame.world(*frame.uv(raw_a)))
            b = tuple(frame.world(*frame.uv(raw_b)))
            endpoint = raw_a if item.start == vertex else raw_b
            endpoint_points.extend((endpoint, tuple(frame.world(*frame.uv(endpoint)))))
            carriers.add((item.curve.a, item.curve.b))
            normal = _cross(_sub(b, a), plate_normal)
            if not any(normal):
                raise JointRefused('degenerate projected carrier at a plate joint')
            offset = _dot(normal, a)
            keys.add(line_key(normal, offset))
            equation = normal, offset
            continue
        # BOTH raw endpoints must equal their exact projections onto P (a tiny out-of-plane edit of the remote
        # endpoint must refuse, never regenerate an old projected proof); the line is derived from the raw points
        a, b = exact(model.vertex_position(item.start)), exact(model.vertex_position(item.end))
        for raw in (item.start, item.end):
            projection = tuple(Q(x) for x in frame.world(*frame.uv(model.vertex_position(raw))))
            if projection != exact(model.vertex_position(raw)):
                raise JointRefused('a straight plate endpoint is not exactly on the plate support')
        normal = _cross(_sub(b, a), plate_normal)
        if not any(normal):
            raise JointRefused('degenerate straight edge at a plate joint')
        offset = _dot(normal, a)
        keys.add(line_key(normal, offset))
        equation = (normal, offset)
    if len(keys) != 1:
        raise JointRefused('the straight edges at a section vertex do not lie on one exact line')
    full = replace(curve, domain=(0.0, 1.0))
    check = check or (lambda *_: None)
    tolerance = Q(float(model.tolerance.effective_length(1.)))
    # One consistently normalised exact line for the roots, the symbol and the local proof.
    line = next(iter(keys))
    normal, offset = line[:3], line[3]
    roots = select_roots(section_roots(full, normal, offset))
    raw_q = exact(model.vertex_position(vertex))
    order, root = select_candidate(roots, q, tolerance)
    lower = tuple(Q(float(x)) for x in root.lower)
    upper = tuple(Q(float(x)) for x in root.upper)
    joint_lower = tuple(min([lo, raw_q[k], q[k]]+[p[k] for p in endpoint_points]) for k, lo in enumerate(lower))
    joint_upper = tuple(max([hi, raw_q[k], q[k]]+[p[k] for p in endpoint_points]) for k, hi in enumerate(upper))
    if sum(hi-lo for lo, hi in zip(joint_lower, joint_upper)) > tolerance:
        raise JointRefused('retained native carrier, projected station and root exceed joint tolerance')
    # stored station -> true root: L1 diameter of union(q, new root enclosure) within the existing tolerance
    if joint_l1(q, lower, upper) > tolerance:
        raise JointRefused('the stored station is farther than the length tolerance from the true root enclosure')
    from .plane_nurbs import plane_nurbs_section
    from .nurbs import _float_down, _float_up
    closed = plane_nurbs_section(curve.plane, curve.surface).paths[curve.path].closed
    # every section edge of THIS path and plane that ends at the vertex contributes its own endpoint proxy
    own = (curve.plane_record, curve.path, definition_checksum(curve.surface))
    incident = [s for s in model.topological_edges_using_vertex(vertex)
                if isinstance(model.edges[s].curve, PlaneNurbsCurve)
                and (model.edges[s].curve.plane_record, model.edges[s].curve.path,
                     definition_checksum(model.edges[s].curve.surface)) == own]
    proxies = []
    for s in incident:
        item = model.edges[s]
        domain_s = tuple(Q(x) for x in item.curve.domain)
        if item.start == vertex:
            proxies.append(domain_s[0])
        if item.end == vertex:
            proxies.append(domain_s[1])
    isolator = tuple(root.exact[0])
    # ALL old incident proxies and the whole new isolator lie strictly inside ONE owner arc interval, and the
    # certified world bounds of their hull are within the length tolerance (local endpoint movement)
    check_hull(full, tuple(proxies), isolator, tolerance, check)
    local_certificate(full, isolator, lower, upper, normal, offset, tolerance, closed, tuple(proxies), check)
    # The NURBS fragments that use this joint's section edges must have every NON-own boundary edge (original
    # outer/hole trims, prior section trims) strictly clear of the neighbourhood; the own-path section edges are
    # the authenticated local curve segments accounted by the certificate above (identified by complete curve
    # definition, never by label). A root on an original boundary therefore refuses.
    blo = tuple(x - tolerance for x in lower)
    bhi = tuple(x + tolerance for x in upper)
    _clear_incident_fragments(model,incident,own,blo,bhi,check)
    identity = ('plate_joint', definition_checksum(curve.surface), curve.plane_record, curve.path,
                next(iter(keys)), order, tuple(sorted(carriers)))
    return PlateJoint(identity, lower, upper, isolator, full, tuple(proxies))


def _clear_incident_fragments(world, incident, own, lower, upper, check):
    """World box against every owned fragment's individually certified trims."""
    from .nurbs_sweep_chart import has_sweep_chart,chart_model
    faces = {face for edge in incident for face in world.faces_using_edge(edge)}
    for face_id in sorted(faces):
        face = world.faces[face_id]
        if definition_checksum(face.surface) == own[2]:
            owner = world
        elif (face.parameterization is not None
              and definition_checksum(face.parameterization) == own[2]):
            if not has_sweep_chart(face):
                raise JointRefused('incident section fragment has no certified sweep chart')
            owner = chart_model(world,face_id)
            if definition_checksum(owner.support) != own[2]:
                raise JointRefused('incident section fragment changed its own support')
        else:
            continue  # the Plate/other support is not this section's NURBS trim domain
        trims_clear(owner,face_id,lower,upper,own,check)
