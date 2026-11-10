"""Sufficient contact proofs for untrimmed authored tensor charts."""
from fractions import Fraction
from itertools import product

from .source_tensor_chart import source_tensor_chart


def _difference(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])


def _hull_bounds(points):
    return tuple((min(c), max(c)) for c in zip(*points))


def _overlapping_boxes(bounds, tolerance):
    active = []
    for f in sorted(bounds, key=lambda face: (bounds[face][0][0], face)):
        active = [g for g in active if bounds[g][0][1] + tolerance >= bounds[f][0][0]]
        for g in active:
            if all(lo <= hi2 + tolerance and lo2 <= hi + tolerance
                   for (lo, hi), (lo2, hi2) in zip(bounds[f], bounds[g])):
                yield f, g
        active.append(f)


def disjoint_plane_nurbs_faces(model):
    """Prove separation of finite planar polygons from positive-weight NURBS control hulls."""
    from .curves import Straight
    from .nurbs import NurbsSurface
    from .surfaces import Plane
    from .trimmed_charts import query_trimmed_surface_charts
    from .native_support_snapshots import capture_native_supports

    if {type(f.surface) for f in model.faces.values()} != {Plane, NurbsSurface}:
        return False
    if any(f.parameterization is not None or f.holes or len(f.loop) != 4 for f in model.faces.values()):
        return False
    capture_native_supports(model)
    bounds = {}
    for f in model.faces.values():
        if type(f.surface) is Plane:
            if not all(type(model.edges[u.edge].curve) is Straight for u in f.loop):
                return False
            points = (model.vertices[model.oriented_start_vertex(u)].position for u in f.loop)
        else:
            if f.surface.weights is not None and any(w <= 0 for row in f.surface.weights for w in row):
                return False
            points = (p for row in f.surface.controls for p in row)
        bounds[f.id] = _hull_bounds(tuple(map(Fraction, p)) for p in points)
    if next(_overlapping_boxes(bounds, Fraction(float(model.tolerance.length))), None) is not None:
        return False
    query_trimmed_surface_charts(model, tuple(model.handle('face', f) for f in bounds),
                                 expected_revision=model.revision)
    return True


def _boundary_and_interior(net, side):
    last_u, last_v = len(net) - 1, len(net[0]) - 1
    rim, interior = [], []
    for u, v in product(range(last_u + 1), range(last_v + 1)):
        at_side = (v == 0, u == last_u, v == last_v, u == 0)[side]
        (rim if at_side else interior).append(net[u][v])
    return rim, interior


def _opposite_open_halfspaces(first, second, tolerance):
    rim, inside = first
    other_rim, other_inside = second
    origin = rim[0]
    axis = _difference(rim[-1], origin)
    length2 = _dot(axis, axis)
    if length2 == 0:
        return False
    # Candidate normals only. Acceptance requires an exact sign test on the
    # entire positive Bernstein/B-spline control net, including both edges.
    normals = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # A non-collinear seam determines its plane even at arbitrary orientation.
    for p in (*rim, *other_rim):
        plane = _cross(axis, _difference(p, origin))
        if any(plane):
            normals.append(plane)
            break
    else:
        # For a straight seam the separator can rotate about its tangent.
        # Intersect the strict linear inequalities in the perpendicular plane.
        perpendiculars = (_cross(axis, basis) for basis in normals)
        u = next(n for n in perpendiculars if any(n))
        v = _cross(axis, u)
        signed = [_difference(p, origin) for p in inside] + [_difference(origin, p) for p in other_inside]
        normals.append(v)
        for sign in (-1, 1):
            lo, hi = None, None
            for p in signed:
                a, b = sign * _dot(u, p), _dot(v, p)
                if not b and a <= 0:
                    break
                if b > 0:
                    lo = -a/b if lo is None else max(lo, -a/b)
                elif b < 0:
                    hi = -a/b if hi is None else min(hi, -a/b)
            else:
                if lo is None or hi is None or lo < hi:
                    if lo is not None and hi is not None:
                        t = (lo + hi) / 2
                    elif lo is not None:
                        t = lo + 1
                    elif hi is not None:
                        t = hi - 1
                    else:
                        t = 0
                    normals.append(tuple(sign * a + t * b for a, b in zip(u, v)))
    for p in (inside[0], other_inside[0]):
        d = _difference(p, origin)
        normals.append(tuple(length2*x - _dot(d, axis)*y for x, y in zip(d, axis)))
    for normal in normals:
        def height(p):
            return _dot(_difference(p, origin), normal)
        if not any(normal) or any(height(p) != 0 for p in (*rim, *other_rim)):
            continue
        a, b = [height(p) for p in inside], [height(p) for p in other_inside]
        clearance = tolerance**2 * _dot(normal, normal)
        if ((min(a) > 0 > max(b) or min(b) > 0 > max(a))
                and all(h*h > clearance for h in (*a,*b))):
            return True
    return False


def tensor_boundary_contacts(model):
    """Return all contacting face pairs and their single shared topological edge.

    None means this proof cannot classify the complete model. Certified charts
    are clamped polynomial tensor products. A positive off-edge control weight
    places every point away from that rim strictly inside its open half-space.
    """
    uses = {f: {use.edge: i for i, use in enumerate(face.loop)} for f, face in model.faces.items()}
    if sum(map(len, uses.values())) == len({e for edges in uses.values() for e in edges}):
        return None
    nets, bounds = {}, {}
    for f in uses:
        view = source_tensor_chart(model, f)
        if view is None:
            return None
        net = tuple(tuple(tuple(Fraction(x) for x in p) for p in row) for row in view.support.controls)
        nets[f] = net
        bounds[f] = _hull_bounds(p for row in net for p in row)
    tolerance = Fraction(float(model.tolerance.length))
    contacts = {}
    for f, g in _overlapping_boxes(bounds, tolerance):
        common = uses[f].keys() & uses[g].keys()
        if len(common) != 1:
            return None
        edge, = common
        if not _opposite_open_halfspaces(_boundary_and_interior(nets[f], uses[f][edge]),
                                         _boundary_and_interior(nets[g], uses[g][edge]), tolerance):
            return None
        contacts[tuple(sorted((f, g)))] = edge
    return contacts or None
