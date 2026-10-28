"""Generic continuing product cover with cached per-surface pieces and section-root continuation.

Every node of the lazy 4D hierarchy (two 2D knot boxes, exact Fractions, one factor split at a time, knot seams
preferred) is exactly one of:
* excluded - certified exact hull separation, or an exact Bernstein-range sign of the factored
  ``F = A_xyz B_w - B_xyz A_w`` (no 4D tensor; ranges of the two 2D pieces combined by interval products);
* covered - inside one knot span per axis, a parameterized Krawczyk certificate with strict whole-free-interval
  inclusion on a PADDED neighborhood (clipped to the span and to the selected support, never extrapolated)
  proves every zero of the cell lies on one regular graph. Alternate free coordinates are tried
  on strongly anisotropic candidates or at the depth limit after the first proposal fails;
* unresolved - counted and sampled (resolution floor, depth, uncertified); the cover CONTINUES, and a budget
  stop leaves the remaining stack counted as frontier.
Pieces, derivative nets, exact nets and hulls are cached once per query per surface.

prune_seconds measures visited/residual node hull retrieval, node hull separation
and homogeneous F exclusion. Initial root-hull preparation belongs to
prepare_seconds; K contraction/exclusion belongs to certification_seconds.
exclusion_seconds is the F-only subset of prune_seconds, not additive time.

Only after complete local coverage does section-root continuation attempt ordered
components. Unresolved correspondence, knot seams and simultaneous physical
support faces refuse; partial coverage cannot certify either topology or absence.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from fractions import Fraction

from ..errors import GeometryError
from ._bernstein import bezier_piece, derivative_grid, subbox
from ._certificate import _Pair, certify_prepared
from ._intervals import Interval, float_up, sqrt_up
from ._records import AtlasReport, CoverWitness, CellReduction, CoverageUse
from ._topology import chart_box

__all__ = ["run_atlas"]

_SAMPLE = 16


class _Surface:
    def __init__(self, source, stats):
        self.surface = surface = source.rebuild()
        self.stats = stats
        self._net = None
        self._hulls: dict = {}
        self._pieces: dict = {}
        self.ranges = (tuple(sorted(Fraction(x) for x in surface.u_range)),
                       tuple(sorted(Fraction(x) for x in surface.v_range)))
        self.spans = (self._spans(surface.knots_u, surface.degree_u, len(surface.controls)),
                      self._spans(surface.knots_v, surface.degree_v, len(surface.controls[0])))
        self.knots = (tuple(sorted({Fraction(k) for k in surface.knots_u})),
                      tuple(sorted({Fraction(k) for k in surface.knots_v})))

    @staticmethod
    def _spans(knots, degree, count):
        ks = sorted({Fraction(k) for k in knots[degree: count + 1]})
        return list(zip(ks, ks[1:]))

    @property
    def net(self):
        if self._net is None:
            self._net = self.surface.exact_net()
        return self._net

    def hull(self, box):
        cached = self._hulls.get(box)
        if cached is not None:
            self.stats.hull_cache_hits += 1
            return cached
        indices = tuple(self.locate(k, *box[k]) for k in range(2))
        if None not in indices:
            grid = self.piece(*indices)[0]
            (u0, u1), (v0, v1) = self.local(box, indices)
            grid = subbox(grid, u0, u1, v0, v1)
            points = [tuple(p[k] / p[3] for k in range(3)) for row in grid for p in row]
            lower = tuple(min(p[k] for p in points) for k in range(3))
            upper = tuple(max(p[k] for p in points) for k in range(3))
        else:
            chart = chart_box(box, self.surface.u_range, self.surface.v_range)
            lower, upper, _sigma2 = self.surface.patch_bounds(chart, self.net)
        self.stats.hull_evaluations += 1
        self._hulls[box] = made = (lower, upper)
        self.stats.cell_cache_peak=max(self.stats.cell_cache_peak,
            self.stats.hull_evaluations+self.stats.pieces_prepared)
        return made

    def locate(self, axis, lo, hi):
        for index, (k0, k1) in enumerate(self.spans[axis]):
            if k0 <= lo and hi <= k1:
                return index
        return None

    def piece(self, iu, iv):
        key = (iu, iv)
        cached = self._pieces.get(key)
        if cached is not None:
            self.stats.piece_reuses += 1
            return cached
        grid = bezier_piece(self.surface, self.net, self.spans[0][iu], self.spans[1][iv])
        self._pieces[key] = made = (grid, derivative_grid(grid, 0), derivative_grid(grid, 1))
        self.stats.pieces_prepared += 1
        self.stats.cell_cache_peak=max(self.stats.cell_cache_peak,
            self.stats.hull_evaluations+self.stats.pieces_prepared)
        return made

    def split(self, box, axis=None):
        widths = [(box[k][1] - box[k][0]) / (self.ranges[k][1] - self.ranges[k][0]) for k in range(2)]
        if axis is None:
            axis = 0 if widths[0] >= widths[1] else 1
        lo, hi = box[axis]
        seams = [k for k in self.knots[axis] if lo < k < hi]
        middle = (lo + hi) / 2
        at = min(seams, key=lambda k: (abs(k - middle), k)) if seams else middle
        left, right = list(box), list(box)
        left[axis], right[axis] = (lo, at), (at, hi)
        return tuple(left), tuple(right)

    def local(self, box, indices):
        """Span-local [0, 1] coordinates of a knot box inside the spans ``indices``."""
        out = []
        for axis in range(2):
            k0, k1 = self.spans[axis][indices[axis]]
            out.append(((box[axis][0] - k0) / (k1 - k0), (box[axis][1] - k0) / (k1 - k0)))
        return out

    def clip(self, indices):
        """Selected-support clip of the span-local coordinates."""
        out = []
        for axis in range(2):
            k0, k1 = self.spans[axis][indices[axis]]
            lo, hi = max(k0, self.ranges[axis][0]), min(k1, self.ranges[axis][1])
            out.append(((lo - k0) / (k1 - k0), (hi - k0) / (k1 - k0)))
        return out


def _separated(a, b) -> bool:
    return any(a[1][k] < b[0][k] or b[1][k] < a[0][k] for k in range(3))


def _diameter(hull) -> Fraction:
    return max(hi - lo for lo, hi in zip(*hull))


def _determinant(m):
    return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
            -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
            +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))


def _free_order(pair, cell, interval_mode="exact"):
    """Strongest exact midpoint cofactor first; only a proposal, never proof."""
    if interval_mode=="filtered":
        from ._filtered import free_order,Unknown
        try:
            return free_order(pair,cell)
        except (Unknown,OverflowError,ZeroDivisionError):
            pass
    midpoint = tuple(((a+b)/2,)*2 for a,b in cell)
    _,jac = pair.value_and_jacobian(pair.enclose(midpoint))
    scores=[]
    for free in range(4):
        minor=[[row[k].lo for k in range(4) if k!=free] for row in jac]
        width=cell[free][1]-cell[free][0]
        scores.append(abs(_determinant(minor))/width if width else Fraction(-1))
    return sorted((k for k in range(4) if cell[k][0]<cell[k][1]),key=lambda k:(-scores[k],k))


def _intersect_image(cell, free, image):
    """Outward 96-bit snapped X intersection K; None proves no root in X."""
    out=list(cell)
    columns=[k for k in range(4) if k!=free]
    scale=1<<96
    for k,x in zip(columns,image):
        lo,hi=max(cell[k][0],x.lo),min(cell[k][1],x.hi)
        if lo>hi:
            return None
        lo=max(cell[k][0],Fraction(math.floor(lo*scale),scale))
        hi=min(cell[k][1],Fraction(-math.floor(-hi*scale),scale))
        out[k]=(lo,hi)
    return tuple(out)


def _graph_axes(order, cell, depth, max_depth, resolution_stop=False):
    """Bound proposal work, without assigning any proof authority to scores.

    A very narrow free coordinate can rank highly after root contraction while
    being a poor whole-interval chart near an extremum. Retry the other valid
    axes there, and at every depth-limit leaf, before declaring it unresolved.
    """
    widths = [hi-lo for lo, hi in cell]
    order = tuple(k for k in order if widths[k] > 0)
    if not order:
        return ()
    if resolution_stop or depth >= max_depth or widths[order[0]]*8 < max(widths):
        return tuple(order)
    return tuple(order[:1])


def _graph_region(cell, clips, free):
    """Existing asymmetric padding, recomputed for each proposed free axis."""
    tube_width = max(hi-lo for lo, hi in cell)
    region = []
    for k, ((lo, hi), (c_lo, c_hi)) in enumerate(zip(cell, clips)):
        pad = (hi-lo)/4 if k == free else max(hi-lo, tube_width/4)/2
        region.append((max(c_lo, lo-pad), min(c_hi, hi+pad)))
    return tuple(region)


def run_atlas(source_a, source_b, tolerances, limits, stats, tick) -> AtlasReport:
    started = time.perf_counter()
    first, second = _Surface(source_a, stats), _Surface(source_b, stats)
    tick()
    root_a = (first.ranges[0], first.ranges[1])
    root_b = (second.ranges[0], second.ranges[1])
    tick()
    hull_a = first.hull(root_a)
    tick()
    hull_b = second.hull(root_b)
    extents = []
    for lower, upper in (hull_a, hull_b):
        squared = sum(((b - a) ** 2 for a, b in zip(lower, upper)), Fraction(0))
        extents.append(float_up(sqrt_up(squared)))
    if not all(math.isfinite(e) for e in extents):
        raise GeometryError("surface hull extent is not representable in float64; query refused")
    length = tolerances.effective_length(min(extents))
    if not math.isfinite(length):
        raise GeometryError("effective length tolerance is not finite; query refused")
    resolution = Fraction(length)
    stats.prepare_seconds = time.perf_counter() - started

    stack = [(root_a, root_b, 0)]
    excluded_hull = excluded_f = covered = unresolved = 0
    excluded_k = 0
    witnesses, sample, reductions,coverage_uses = [], [], [], []
    proof_index={}
    def bucket(value):
        return max(0,min(15,math.floor(value*16)))
    def reuse(spans,cell):
        # Same-span 16-bin interval index, followed by exact full-box
        # containment. Bounded search misses cause recomputation, never loss.
        candidates=proof_index.get(spans,{}).get(bucket(cell[0][0]),())
        for index in candidates[:64]:
            stats.coverage_lookup_checks+=1
            w=witnesses[index]
            if all(a<=lo<=hi<=b for (lo,hi),(a,b) in zip(cell,w.region)):
                coverage_uses.append(CoverageUse(spans,tuple(cell),index))
                stats.reused_graph_cells+=1
                stats.covered_cells+=1
                return True
        return False
    def index_witness(index):
        w=witnesses[index]
        bins=proof_index.setdefault(w.spans,{})
        for b in range(bucket(w.region[0][0]),bucket(w.region[0][1])+1):
            bins.setdefault(b,[]).append(index)
    pairs = {}
    stopped = False
    reason = ""
    while stack:
        stats.peak_stack = max(stats.peak_stack, len(stack))
        tick()
        node = stack.pop()
        if stats.boxes_visited >= limits.max_boxes_per_pair:           # authority checked before consuming
            stack.append(node)
            stopped, reason = True, "box budget exhausted; the remaining frontier is counted"
            break
        stats.boxes_visited += 1
        box_a, box_b, depth = node
        stats.max_depth = max(stats.max_depth, depth)
        pruning_at=time.perf_counter()
        ha, hb = first.hull(box_a), second.hull(box_b)
        hull_separated=_separated(ha,hb)
        stats.prune_seconds+=time.perf_counter()-pruning_at
        if hull_separated:
            stats.excluded_pairs += 1
            excluded_hull += 1
            continue
        ia = tuple(first.locate(k, *box_a[k]) for k in range(2))
        ib = tuple(second.locate(k, *box_b[k]) for k in range(2))
        if None not in ia + ib:
            cell = first.local(box_a, ia) + second.local(box_b, ib)
            if reuse((ia,ib),cell):
                continue
            piece_a, piece_b = first.piece(*ia), second.piece(*ib)
            key = (ia, ib)
            if key not in pairs:
                pairs[key] = _Pair.from_pieces(piece_a, piece_b)
            pair = pairs[key].anchored(cell)
            began = time.perf_counter()
            values_a,values_b = pair.enclose_values(cell)
            value=[values_a[k]*values_b[3]-values_b[k]*values_a[3] for k in range(3)]
            f_excluded=values_a[3].lo>0 and values_b[3].lo>0 and any(v.excludes_zero() for v in value)
            exclusion_elapsed=time.perf_counter()-began
            stats.exclusion_seconds+=exclusion_elapsed
            stats.prune_seconds+=exclusion_elapsed
            if f_excluded:
                stats.excluded_by_f += 1
                excluded_f += 1
                continue
            clips = first.clip(ia) + second.clip(ib)
            knot_box = (first.spans[0][ia[0]], first.spans[1][ia[1]], second.spans[0][ib[0]],
                        second.spans[1][ib[1]])
            # Preserve all candidate roots while removing correlated empty
            # volume before splitting. No strict graph certificate is required
            # for the Krawczyk identity X_roots subset X intersection K(X).
            began = time.perf_counter()
            empty = False
            order = _free_order(pair,cell,limits.interval_mode)
            for _ in range(min(4,limits.max_newton_iterations)):
                if not order:
                    break
                free = order[0]
                stats.certificate_attempts += 1
                tick()
                cert = certify_prepared(pair,knot_box,free,
                    [Interval(*cell[k]) for k in range(4) if k!=free],Interval(*cell[free]),
                    for_root_filter=True,interval_mode=limits.interval_mode,stats=stats)
                if not cert.unknown_enclosure:
                    break
                residual = _intersect_image(cell,free,cert.unknown_enclosure)
                if residual is None:
                    excluded_k += 1
                    stats.excluded_by_k += 1
                    empty = True
                    break
                if residual == tuple(cell):
                    break
                reductions.append(CellReduction((ia,ib),tuple(cell),residual,free,
                    tuple((x.lo,x.hi) for x in cert.unknown_enclosure)))
                stats.contractions += 1
                old_width = sum(hi-lo for lo,hi in cell)
                new_width = sum(hi-lo for lo,hi in residual)
                cell = list(residual)
                pair = pairs[key].anchored(cell)
                order = _free_order(pair,cell,limits.interval_mode)
                if new_width*8 >= old_width*7:
                    break
            stats.certification_seconds += time.perf_counter()-began
            if empty:
                continue
            if reuse((ia,ib),cell):
                continue
            native = tuple((k0+lo*(k1-k0),k0+hi*(k1-k0))
                           for (lo,hi),(k0,k1) in zip(cell,knot_box))
            box_a,box_b = native[:2],native[2:]
            node = (box_a,box_b,depth)
            pruning_at=time.perf_counter()
            ha,hb = first.hull(box_a),second.hull(box_b)
            stats.prune_seconds+=time.perf_counter()-pruning_at
            began = time.perf_counter()
            done = False
            for free in _graph_axes(order, cell, depth, limits.max_subdivision_depth,
                                   max(_diameter(ha), _diameter(hb)) <= resolution):
                if cell[free][0] == cell[free][1]:
                    continue  # a graph must have a nonzero free interval
                # A dependent identity such as u=s needs more room in Y than
                # in Phi. Equal padding can never give strict inclusion.
                region = _graph_region(cell, clips, free)
                stats.certificate_attempts += 1
                tick()
                unknown = [Interval(*region[k]) for k in range(4) if k != free]
                certificate = certify_prepared(pair, knot_box, free, unknown, Interval(*region[free]),
                    interval_mode=limits.interval_mode,stats=stats)
                # The exact Krawczyk identity encloses EVERY zero in the proof
                # box, even when its strict-inclusion test failed. The proof
                # box contains the owned candidate in all four coordinates.
                # A disjoint unknown projection therefore excludes that owned
                # candidate (not the entire padded neighborhood).
                assert all(a <= lo <= hi <= b for (lo,hi),(a,b) in zip(cell,region))
                columns = [k for k in range(4) if k != free]
                if certificate.unknown_enclosure and any(
                    not x.intersects(Interval(*cell[k]))
                    for x,k in zip(certificate.unknown_enclosure,columns)):
                    excluded_k += 1
                    stats.excluded_by_k += 1
                    done = True
                    break
                if certificate.certified:
                    if len(witnesses) >= limits.max_trace_segments:
                        stopped, reason = True, "witness limit exhausted"
                        stack.append(node)  # this cell has not been consumed
                        break
                    stats.certificates += 1
                    covered += 1
                    witnesses.append(CoverWitness((ia, ib), tuple(cell), free, tuple(region)))
                    index_witness(len(witnesses)-1)
                    stats.covered_cells+=1
                    done = True
                    break
            stats.certification_seconds += time.perf_counter() - began
            if stopped:
                break
            if done:
                continue
        if max(_diameter(ha), _diameter(hb)) <= resolution or depth >= limits.max_subdivision_depth:
            unresolved += 1
            stats.unresolved_cells += 1
            if len(sample) < _SAMPLE:
                sample.append((box_a, box_b))
            continue
        native = box_a+box_b
        domains = first.ranges+second.ranges
        widths=[(hi-lo)/(b-a) for (lo,hi),(a,b) in zip(native,domains)]
        coordinate=max(range(4),key=lambda k:(widths[k],-k))
        split_first = coordinate<2
        stats.splits += 1
        children = first.split(box_a,coordinate) if split_first else second.split(box_b,coordinate-2)
        for child in reversed(children):
            stack.append((child, box_b, depth + 1) if split_first else (box_a, child, depth + 1))
    frontier = len(stack) if stopped else 0
    complete = not stopped and unresolved == 0
    if not reason:
        if unresolved:
            reason = f"{unresolved} cells unresolved at the length resolution or depth limit"
        elif covered:
            reason = ("every cell is excluded or covered by a local regular-graph witness; "
                      "ordered section-root component accounting is pending")
        else:
            reason = "every cell excluded"
    report = AtlasReport(excluded_hull, excluded_f, covered, unresolved, frontier, complete, False,
                         tuple(witnesses), tuple(sample), reason,excluded_by_k=excluded_k,
                         reductions=tuple(reductions),coverage_uses=tuple(coverage_uses),
                         covered_cells=stats.covered_cells)
    if complete and covered:
        from ._continuation import stitch
        from ._source import NurbsSurfaceIntersectionRefused
        began = time.perf_counter()
        try:
            roots, components = stitch(first,second,report.witnesses,limits,stats,tick)
        except NurbsSurfaceIntersectionRefused as error:
            report = replace(report,stitching_implemented=True,reason=f"generic continuation refused: {error}")
        else:
            report = replace(report,stitching_implemented=True,roots=roots,components=components,
                             reason="complete regular graph cover with certified oriented section-root continuation")
            stats.roots,stats.components = len(roots),len(components)
            stats.arcs = sum(len(c.arcs) for c in components)
        finally:
            stats.topology_seconds += time.perf_counter()-began
    return report
