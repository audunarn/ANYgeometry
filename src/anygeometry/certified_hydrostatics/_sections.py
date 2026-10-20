"""Volume sums and symbolic sections (design sections 4 and 5).

Everything is exact (``Fraction``) and bit-capped: every product, sum and quotient goes through ``Ar``, which
checks the raw operand sizes before the operation is formed. Two separate pieces:

* ``clipped_sums``: signed tetrahedron sums over the faces clipped with the ``below`` convention (a vertex on
  the plane counts as above). The clipped chains are used *only* as signed boundary chains.
* ``build_section``: the section of one side (``sigma = +1`` for ``below``, plane ``n.r = d - eps``; ``sigma = -1``
  for ``above``, plane ``d + eps``). Cut points are affine points ``c0 + eps c1`` identified by their shell
  edge; predicates are polynomials in ``eps`` decided by the first nonzero coefficient. Sections are never
  built from clipped-face output.

Every primitive unit reserves a count of one."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key

from ._arith import Ar
from ._errors import RefusalCategory, refuse
from ._snapshot import ExactSnapshot
from ._work import Meter

__all__ = ["vertex_heights", "contacts", "exact_occupancy", "clipped_sums", "cap_sums", "build_section",
           "waterplane_sums", "SectionData", "LoopData", "projection_axis"]

ZERO = Fraction(0)


def _first_sign(*coefficients) -> int:
    """The sign of the polynomial ``c0 + c1 eps + c2 eps^2`` for small ``eps > 0``: its first nonzero coefficient."""
    for c in coefficients:
        if c:
            return 1 if c > 0 else -1
    return 0


def projection_axis(normal) -> int:
    """The index ``k`` maximising ``|n[k]|`` (lowest index on ties): the coordinate dropped by the projection."""
    magnitude = [abs(c) for c in normal]
    return magnitude.index(max(magnitude))


# ------------------------------------------------------------------ heights, contacts, occupancy


def vertex_heights(snapshot: ExactSnapshot, query, meter: Meter) -> dict:
    """``h(p) = n . p - d`` of every shell vertex (exact)."""
    ar = Ar(meter)
    n, d = query.normal, query.offset
    heights = {}
    for vertex in sorted(snapshot.positions):
        meter.unit("vertex_height")
        heights[vertex] = ar.sub(ar.dot(n, snapshot.positions[vertex]), d)
    return heights


def contacts(snapshot: ExactSnapshot, heights: dict, meter: Meter) -> tuple:
    """``(vertices, edges, faces)`` (sorted id tuples) lying in the plane ``n.r = d``."""
    vertices = tuple(v for v in sorted(heights) if heights[v] == 0)
    edges = []
    for edge in sorted(snapshot.edges):
        meter.unit("contact_test")
        start, end = snapshot.edges[edge]
        if heights[start] == 0 and heights[end] == 0:
            edges.append(edge)
    faces = []
    for face in snapshot.faces:
        in_plane = True
        for loop in face.loops:
            for vertex in loop.vertices:
                meter.unit("contact_test")
                in_plane = in_plane and heights[vertex] == 0
        if in_plane:
            faces.append(face.id)
    return vertices, tuple(edges), tuple(faces)


def exact_occupancy(heights: dict) -> str:
    """Occupancy of the exact half-space: ``EMPTY`` when no vertex is strictly below, ``FULL`` when none is above."""
    values = heights.values()
    if all(h >= 0 for h in values):
        return "EMPTY"
    if all(h <= 0 for h in values):
        return "FULL"
    return "PARTIAL"


# ------------------------------------------------------------------ clipped faces: V and J


def _clip_chain(ar: Ar, points, hs, meter: Meter) -> list:
    """Sutherland-Hodgman against ``h < 0`` (vertices on the plane are outside): a signed boundary chain."""
    out = []
    count = len(points)
    for i in range(count):
        meter.unit("clip_vertex")
        j = (i + 1) % count
        cur_in, next_in = hs[i] < 0, hs[j] < 0
        if cur_in and next_in:
            out.append(points[j])
        elif cur_in != next_in:
            meter.unit("edge_crossing")
            t = ar.div(hs[i], ar.sub(hs[i], hs[j]))
            cur, nxt = points[i], points[j]
            out.append(tuple(ar.add(cur[k], ar.mul(t, ar.sub(nxt[k], cur[k]))) for k in range(3)))
            if next_in:
                out.append(points[j])
    return out


def _fan(ar: Ar, chain, meter: Meter, total, moment) -> None:
    """Add ``det(a,b,c)`` and ``det(a,b,c) (a+b+c)`` of a fan over ``chain`` to ``total`` / ``moment``."""
    for i in range(1, len(chain) - 1):
        meter.unit("triangle_sum")
        a, b, c = chain[0], chain[i], chain[i + 1]
        det = ar.det(a, b, c)
        total[0] = ar.add(total[0], det)
        for k in range(3):
            moment[k] = ar.add(moment[k], ar.mul(det, ar.add(ar.add(a[k], b[k]), c[k])))


def clipped_sums(snapshot: ExactSnapshot, heights: dict, meter: Meter) -> tuple:
    """``(sum det, [sum det (a+b+c)] x 3)`` over all faces clipped to ``h < 0`` (without the cap)."""
    ar = Ar(meter)
    total, moment = [ZERO], [ZERO, ZERO, ZERO]
    positions = snapshot.positions
    for face in snapshot.faces:
        for loop in face.loops:
            points = [positions[v] for v in loop.vertices]
            hs = [heights[v] for v in loop.vertices]
            _fan(ar, _clip_chain(ar, points, hs, meter), meter, total, moment)
    return total[0], moment


def cap_sums(section: "SectionData", meter: Meter) -> tuple:
    """The cap's tetrahedron sums: a fan from a point on the plane over the directed cut segments."""
    ar = Ar(meter)
    total, moment = [ZERO], [ZERO, ZERO, ZERO]
    if section.segments:
        q = section.points[section.segments[0][1]][0]
        for _face, a, b in section.segments:
            _fan(ar, (q, section.points[a][0], section.points[b][0]), meter, total, moment)
    return total[0], moment


# ------------------------------------------------------------------ the section of one side


@dataclass(frozen=True)
class LoopData:
    edges: tuple         # crossed edges in cycle order
    identity: tuple      # sorted crossed-edge ids
    role: str            # "outer" | "hole"


@dataclass(frozen=True)
class SectionData:
    sigma: int
    occupancy: str        # occupancy of the side's perturbed half-space: EMPTY, FULL or PARTIAL
    points: dict          # crossed edge id -> (c0, c1)
    segments: tuple       # (face id, start edge id, end edge id), directed with the cap on the left seen from +n
    loops: tuple          # LoopData, sorted by identity
    components: tuple     # (outer identity, (hole identities...)), sorted


def _newell(ar: Ar, points, meter: Meter):
    base = points[0]
    total = (ZERO, ZERO, ZERO)
    for i in range(1, len(points) - 1):
        meter.unit("loop_area")
        c = ar.cross(ar.sub3(points[i], base), ar.sub3(points[i + 1], base))
        total = (ar.add(total[0], c[0]), ar.add(total[1], c[1]), ar.add(total[2], c[2]))
    return total


def _cut_points(ar: Ar, snapshot, heights, sign, meter: Meter, sigma: int) -> dict:
    points = {}
    for edge in sorted(snapshot.edges):
        meter.unit("edge_crossing")
        start, end = snapshot.edges[edge]
        if sign[start] == sign[end]:
            continue
        meter.unit("cut_point")
        p, q = snapshot.positions[start], snapshot.positions[end]
        denominator = ar.sub(heights[end], heights[start])       # nonzero: the effective signs differ
        t0 = ar.div(-heights[start], denominator)
        t1 = ar.div(Fraction(-sigma), denominator)
        direction = ar.sub3(q, p)
        c0 = tuple(ar.add(p[k], ar.mul(t0, direction[k])) for k in range(3))
        c1 = tuple(ar.mul(t1, direction[k]) for k in range(3))
        points[edge] = (c0, c1)
    return points


def _face_segments(ar: Ar, snapshot, query, points, meter: Meter) -> list:
    n = query.normal
    segments = []
    for face in snapshot.faces:
        crossed = [edge for loop in face.loops for edge, _forward in loop.edges if edge in points]
        if not crossed:
            continue
        if len(crossed) % 2:
            raise RuntimeError(f"face {face.id} has an odd number of crossed edges")
        normal = _newell(ar, [snapshot.positions[v] for v in face.loops[0].vertices], meter)
        direction = ar.cross(n, normal)                       # cap on the left seen from +n
        entries = []
        for edge in crossed:
            meter.unit("cut_param")
            c0, c1 = points[edge]
            entries.append((edge, ar.dot(direction, c0), ar.dot(direction, c1)))

        def compare(x, y):
            meter.unit("sort_comparison")
            order = ar.cmp(x[1], y[1]) or ar.cmp(x[2], y[2])
            if order:
                return order
            raise refuse(RefusalCategory.SYMBOLIC_DEGENERACY,
                         f"two cut points of face {face.id} coincide for every epsilon (edges {x[0]} and {y[0]})")

        entries.sort(key=cmp_to_key(compare))
        for k in range(0, len(entries), 2):
            meter.unit("segment_build")
            segments.append((face.id, entries[k][0], entries[k + 1][0]))
    return segments


def _loops(points, segments, meter: Meter) -> list:
    successor = {}
    for _face, a, b in segments:
        if a in successor:
            raise RuntimeError(f"cut point of edge {a} starts two segments")
        successor[a] = b
    if sorted(successor) != sorted(successor.values()) or set(successor) != set(points):
        raise RuntimeError("the cut-point graph is not a union of cycles")
    seen, loops = set(), []
    for start in sorted(points):
        if start in seen:
            continue
        cycle, edge = [], start
        while edge not in seen:
            meter.unit("loop_walk")
            seen.add(edge)
            cycle.append(edge)
            edge = successor[edge]
        if edge != start:
            raise RuntimeError("the cut-point graph is not a union of cycles")
        loops.append(tuple(cycle))
    return loops


# --- affine (polynomial in eps) predicates in the projection coordinates (a, b)


def _project_points(points, axis: int):
    keep = [i for i in range(3) if i != axis]
    return {edge: (c0[keep[0]], c1[keep[0]], c0[keep[1]], c1[keep[1]]) for edge, (c0, c1) in points.items()}


def _loop_area_sign(ar: Ar, cycle, projected, factor: int, meter: Meter) -> int:
    """Sign of the loop's signed area about ``n`` (a polynomial of degree 2 in eps); refuses a zero polynomial."""
    c0 = c1 = c2 = ZERO
    count = len(cycle)
    for i in range(count):
        meter.unit("loop_area")
        pa0, pa1, pb0, pb1 = projected[cycle[i]]
        qa0, qa1, qb0, qb1 = projected[cycle[(i + 1) % count]]
        c0 = ar.add(c0, ar.sub(ar.mul(pa0, qb0), ar.mul(qa0, pb0)))
        c1 = ar.add(c1, ar.sub(ar.add(ar.mul(pa0, qb1), ar.mul(pa1, qb0)),
                               ar.add(ar.mul(qa0, pb1), ar.mul(qa1, pb0))))
        c2 = ar.add(c2, ar.sub(ar.mul(pa1, qb1), ar.mul(qa1, pb1)))
    sign = _first_sign(c0, c1, c2)
    if sign == 0:
        raise refuse(RefusalCategory.SYMBOLIC_DEGENERACY, "a section loop has zero signed area for every epsilon")
    return sign * factor


def _b_not_above(ar: Ar, x, y) -> bool:
    """``x.b <= y.b`` as polynomials in eps (equal polynomials count as ``<=``)."""
    return _first_sign(ar.sub(x[2], y[2]), ar.sub(x[3], y[3])) <= 0


def _inside(ar: Ar, p, loop_cycle, projected, meter: Meter) -> bool:
    """Ray parity of ``p`` in the loop with the half-open rule (design section 5)."""
    crossings = 0
    count = len(loop_cycle)
    for i in range(count):
        meter.unit("pip_edge_test")
        q, r = projected[loop_cycle[i]], projected[loop_cycle[(i + 1) % count]]
        q_low, r_low = _b_not_above(ar, q, p), _b_not_above(ar, r, p)
        if q_low == r_low:
            continue
        da0, da1 = ar.sub(r[0], q[0]), ar.sub(r[1], q[1])
        db0, db1 = ar.sub(r[2], q[2]), ar.sub(r[3], q[3])
        pb0, pb1 = ar.sub(p[2], q[2]), ar.sub(p[3], q[3])
        pa0, pa1 = ar.sub(p[0], q[0]), ar.sub(p[1], q[1])
        sign = _first_sign(
            ar.sub(ar.mul(da0, pb0), ar.mul(db0, pa0)),
            ar.sub(ar.add(ar.mul(da0, pb1), ar.mul(da1, pb0)), ar.add(ar.mul(db0, pa1), ar.mul(db1, pa0))),
            ar.sub(ar.mul(da1, pb1), ar.mul(db1, pa1)))
        if sign == 0:
            raise refuse(RefusalCategory.SYMBOLIC_DEGENERACY,
                         "a section loop vertex lies on another loop for every epsilon")
        if (sign > 0) == q_low:          # upward edge (q low, r high): right of p when positive; downward: negative
            crossings += 1
    return bool(crossings % 2)


def _topology(ar: Ar, cycles, points, normal, meter: Meter) -> tuple:
    axis = projection_axis(normal)
    projected = _project_points(points, axis)
    factor = (-1) ** axis * (1 if normal[axis] > 0 else -1)
    outer_by_area = [_loop_area_sign(ar, cycle, projected, factor, meter) > 0 for cycle in cycles]
    containers = [[] for _ in cycles]
    if len(cycles) > 1:
        for i, cycle in enumerate(cycles):
            probe = projected[cycle[0]]
            for j, other in enumerate(cycles):
                if i != j and _inside(ar, probe, other, projected, meter):
                    containers[i].append(j)
    depth = [len(c) for c in containers]
    for i in range(len(cycles)):
        if (depth[i] % 2 == 0) != outer_by_area[i]:
            raise RuntimeError("loop nesting parity disagrees with the loop orientation")
    loops = [LoopData(cycles[i], tuple(sorted(cycles[i])), "outer" if outer_by_area[i] else "hole")
             for i in range(len(cycles))]
    components = []
    for i, loop in enumerate(loops):
        if loop.role == "outer":
            holes = sorted(loops[h].identity for h in range(len(cycles))
                           if loops[h].role == "hole" and containers[h]
                           and max(containers[h], key=lambda c: depth[c]) == i)
            components.append((loop.identity, tuple(holes)))
    return tuple(sorted(loops, key=lambda loop: loop.identity)), tuple(sorted(components))


def build_section(snapshot: ExactSnapshot, query, heights: dict, sigma: int, meter: Meter) -> SectionData:
    """The section of the ``below`` (``sigma = +1``) or ``above`` (``-1``) limit, or an empty one when absent."""
    ar = Ar(meter)
    sign = {}
    for vertex in sorted(heights):
        meter.unit("side_sign")
        h = heights[vertex]
        sign[vertex] = -1 if h < 0 else (1 if h > 0 else sigma)
    values = set(sign.values())
    occupancy = "FULL" if values == {-1} else ("EMPTY" if values == {1} else "PARTIAL")
    points = _cut_points(ar, snapshot, heights, sign, meter, sigma)
    if not points:
        return SectionData(sigma, occupancy, {}, (), (), ())
    segments = _face_segments(ar, snapshot, query, points, meter)
    cycles = _loops(points, segments, meter)
    loops, components = _topology(ar, cycles, points, query.normal, meter)
    return SectionData(sigma, occupancy, points, tuple(segments), loops, components)


# ------------------------------------------------------------------ waterplane integrals (limit)


def waterplane_sums(section: SectionData, query, meter: Meter) -> dict:
    """Numerators ``Q`` of ``A, Mu, Mv, Iuu, Ivv, Iuv`` (each value is ``Q / |n|``) from the limit cut points."""
    ar = Ar(meter)
    n, origin = query.normal, query.waterplane_origin
    e1, e2 = query.waterplane_axes
    result = {name: ZERO for name in ("A", "Mu", "Mv", "Iuu", "Ivv", "Iuv")}
    if not section.segments:
        return result
    coords = {}

    def uv(edge):
        if edge not in coords:
            c0 = section.points[edge][0]
            rel = ar.sub3(c0, origin)
            coords[edge] = (c0, ar.dot(e1, rel), ar.dot(e2, rel))
        return coords[edge]

    def total(values):
        out = ZERO
        for value in values:
            out = ar.add(out, value)
        return out

    q = uv(section.segments[0][1])
    pairs = ((0, 1), (0, 2), (1, 2))
    for _face, a_edge, b_edge in section.segments:
        meter.unit("waterplane_triangle")
        corners = (q, uv(a_edge), uv(b_edge))
        area2 = ar.dot(ar.cross(ar.sub3(corners[1][0], corners[0][0]), ar.sub3(corners[2][0], corners[0][0])), n)
        weight = ar.div(area2, 2)
        us, vs = [c[1] for c in corners], [c[2] for c in corners]
        suu = total([ar.mul(u, u) for u in us] + [ar.mul(us[i], us[j]) for i, j in pairs])
        svv = total([ar.mul(v, v) for v in vs] + [ar.mul(vs[i], vs[j]) for i, j in pairs])
        suv = total([ar.mul(2, ar.mul(u, v)) for u, v in zip(us, vs)]
                    + [ar.mul(us[i], vs[j]) for i in range(3) for j in range(3) if i != j])
        terms = {"A": Fraction(1), "Mu": ar.div(total(us), 3), "Mv": ar.div(total(vs), 3),
                 "Iuu": ar.div(suu, 6), "Ivv": ar.div(svv, 6), "Iuv": ar.div(suv, 12)}
        for name, factor in terms.items():
            result[name] = ar.add(result[name], ar.mul(weight, factor))
    return result
