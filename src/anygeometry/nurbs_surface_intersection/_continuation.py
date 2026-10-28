"""Section-root continuation for a complete finite regular graph cover.

Incidence is certified by an existing isolated root lying strictly inside a
second graph's uniqueness region. Rectangular overlap is only a search filter.
The same cofactor tangent transports orientation between free coordinates.
No result is issued unless every whole graph joins an ordered closed path or a
path ending on transverse physical support faces. Simultaneous faces require
an exact constrained root and a common inward one-sided continuation.
"""
from __future__ import annotations

from fractions import Fraction
import math

from ._certificate import _Pair, certify_prepared
from ._intervals import Interval
from ._records import SectionRoot, GenericArc, ComponentRecord
from ._source import NurbsSurfaceIntersectionRefused, NurbsSurfaceIntersectionIntegrityError
from ._topology import chart_box


def _det(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _snap(x):
    scale = 1 << 96
    return Interval(Fraction(math.floor(x.lo * scale), scale),
                    Fraction(-math.floor(-x.hi * scale), scale))


class Graphs:
    def __init__(self, first, second, witnesses, limits, stats, tick):
        self.first, self.second, self.witnesses = first, second, witnesses
        self.limits, self.stats, self.tick = limits, stats, tick
        self.pairs, self.knots, self.views = {}, [], []
        self.native_regions,self.edge_cache=[],{}
        self.attempts = 0
        for w in witnesses:
            self.charge()
            ia, ib = w.spans
            if w.spans not in self.pairs:
                self.pairs[w.spans] = _Pair.from_pieces(first.piece(*ia), second.piece(*ib))
            self.views.append(self.pairs[w.spans].anchored(w.region))
            self.knots.append((first.spans[0][ia[0]], first.spans[1][ia[1]],
                               second.spans[0][ib[0]], second.spans[1][ib[1]]))
            self.native_regions.append(tuple((a+lo*(b-a),a+hi*(b-a))
                for (lo,hi),(a,b) in zip(w.region,self.knots[-1])))
        indices=sorted(range(len(witnesses)),key=lambda i:(self.native_regions[i][0][0],i))
        def tree(lo,hi):
            if lo>=hi:
                return None
            mid=(lo+hi)//2
            i=indices[mid]
            left,right=tree(lo,mid),tree(mid+1,hi)
            maximum=max(self.native_regions[i][0][1],
                        left[3] if left else self.native_regions[i][0][1],
                        right[3] if right else self.native_regions[i][0][1])
            return i,left,right,maximum
        self.index=tree(0,len(indices))
        self.filter_limit=limits.max_boxes_per_pair*max(1,len(indices).bit_length())
        self.filter_checks=0

    def candidates(self,root):
        """Exact 4D interval broadphase; no root/topology conclusion from overlap."""
        native=self.native(root)
        stack=[self.index] if self.index else []
        found=[]
        while stack:
            node=stack.pop()
            self.filter_checks+=1
            if hasattr(self.stats,"broadphase_checks"):
                self.stats.broadphase_checks+=1
            if self.filter_checks>self.filter_limit:
                raise NurbsSurfaceIntersectionRefused("native broadphase work limit exhausted")
            if self.filter_checks%32==0:
                self.tick()
            i,left,right,maximum=node
            if maximum<native[0][0]:
                continue
            box=self.native_regions[i]
            if box[0][0]<=native[0][1]:
                if all(a<=d and c<=b for (a,b),(c,d) in zip(native,box)):
                    found.append(i)
                if right:
                    stack.append(right)
            if left:
                stack.append(left)
        return tuple(sorted(found))

    def charge(self):
        self.tick()
        if self.attempts >= self.limits.max_boxes_per_pair:
            raise NurbsSurfaceIntersectionRefused("section/transition attempt limit exhausted")
        self.attempts += 1
        self.stats.section_attempts += 1

    def pair(self, wi):
        return self.views[wi]

    def contract(self, wi, free, phi, region):
        self.charge()
        y = [Interval(*region[k]) for k in range(4) if k != free]
        cert = certify_prepared(self.pair(wi), self.knots[wi], free, y, phi,
                               interval_mode=self.limits.interval_mode,stats=self.stats)
        if not cert.unknown_enclosure:
            raise NurbsSurfaceIntersectionRefused("section contraction lacks a regular Krawczyk enclosure")
        out, j = [], 0
        for k in range(4):
            if k == free:
                out.append((phi.lo, phi.hi))
            else:
                x = _snap(cert.unknown_enclosure[j])
                lo, hi = max(region[k][0], x.lo), min(region[k][1], x.hi)
                if lo > hi:
                    raise NurbsSurfaceIntersectionIntegrityError("a proven graph section contracted to empty")
                out.append((lo, hi))
                j += 1
        return tuple(out)

    def endpoint(self, wi, side):
        w = self.witnesses[wi]
        value = w.region[w.free_coordinate][side]
        region = w.region
        # The whole-interval witness already proves existence and uniqueness at
        # this endpoint; contraction preserves that root even without new strictness.
        # Precision is refined only when incidence/order/evaluation needs it;
        # existence already follows from the whole-interval witness.
        region = self.contract(wi, w.free_coordinate, Interval(value), region)
        return SectionRoot(wi, w.free_coordinate, value, region)

    def refine(self, root):
        region = self.contract(root.witness, root.free_coordinate, Interval(root.value), root.region)
        return SectionRoot(root.witness, root.free_coordinate, root.value, region)

    def native(self, root):
        return tuple((k0 + lo * (k1-k0), k0 + hi * (k1-k0))
                     for (lo, hi), (k0, k1) in zip(root.region, self.knots[root.witness]))

    def in_graph(self, root, wi):
        """Return span-local enclosure when strict incidence is proved, else None."""
        native = self.native(root)
        local = tuple(((lo-k0)/(k1-k0), (hi-k0)/(k1-k0))
                      for (lo, hi), (k0, k1) in zip(native, self.knots[wi]))
        region = self.witnesses[wi].region
        self.charge()
        if all(a < lo <= hi < b for (lo, hi), (a, b) in zip(local, region)):
            return local
        f=self.witnesses[wi].free_coordinate
        if (all(a<lo<=hi<b for k,((lo,hi),(a,b)) in enumerate(zip(local,region)) if k!=f)
                and region[f][0]<=local[f][0]<=local[f][1]<=region[f][1]):
            if root.witness==wi:
                return local  # its own whole-interval graph proves the endpoint
            if self.seam_incidence(root,wi):
                return local
        return None

    def seam_incidence(self,root,wi):
        """Exactly one adjacent original span face, with identical homogeneous edge."""
        old,new=self.witnesses[root.witness],self.witnesses[wi]
        a=old.spans[0]+old.spans[1]
        b=new.spans[0]+new.spans[1]
        changed=[k for k in range(4) if a[k]!=b[k]]
        if len(changed)!=1:
            return False
        f=changed[0]
        if root.free_coordinate!=f or old.free_coordinate!=f or new.free_coordinate!=f:
            return False
        old_span,new_span=self.knots[root.witness][f],self.knots[wi][f]
        if old_span[1]==new_span[0]:
            seam,old_side,new_side=old_span[1],1,0
        elif new_span[1]==old_span[0]:
            seam,old_side,new_side=old_span[0],0,1
        else:
            return False
        if self.native(root)[f]!=(seam,seam):
            return False
        key=old.spans,new.spans
        if key not in self.edge_cache:
            surface=self.first if f<2 else self.second
            old_indices=old.spans[0 if f<2 else 1]
            new_indices=new.spans[0 if f<2 else 1]
            ga,gb=surface.piece(*old_indices)[0],surface.piece(*new_indices)[0]
            axis=f%2
            def edge(grid,side):
                j=0 if side==0 else -1
                return grid[j] if axis==0 else tuple(row[j] for row in grid)
            self.edge_cache[key]=edge(ga,old_side)==edge(gb,new_side)
        return self.edge_cache[key]

    def orientation(self, root, old, new, direction):
        local = self.in_graph(root, new)
        if local is None:
            return None
        f=self.witnesses[new].free_coordinate
        region=self.witnesses[new].region
        if (local[f][0]==local[f][1]==region[f][0] and direction<0
                or local[f][0]==local[f][1]==region[f][1] and direction>0):
            return None  # a seam transition must extend ahead, never backtrack
        self.charge()
        _f, jac = self.pair(new).value_and_jacobian(self.pair(new).enclose(local))
        a, b = self.witnesses[old].free_coordinate, self.witnesses[new].free_coordinate
        # Cofactors are with respect to the new span chart. Span widths are
        # positive, so the native and span-local tangent component signs agree.
        minors = []
        for col in (a, b):
            m = [[row[k] for k in range(4) if k != col] for row in jac]
            minors.append(_det(m).scale(-1 if col % 2 else 1))
        product = minors[0] * minors[1]
        if not product.excludes_zero():
            return None
        return direction if product.lo > 0 else -direction

    def physical(self, root):
        native = self.native(root)
        ranges = self.first.ranges + self.second.ranges
        f = root.free_coordinate
        if not (native[f][0] == native[f][1] and native[f][0] in ranges[f]):
            return False
        if all(lo < a <= b < hi for k, ((a,b),(lo,hi)) in enumerate(zip(native, ranges)) if k != f):
            return True
        # An interval touching a second face is only a candidate. Prove an
        # exact constrained root inside the already unique section instead.
        active, point = {}, []
        for k, ((a,b),(lo,hi)) in enumerate(zip(native,ranges)):
            faces = [x for x in (lo,hi) if a <= x <= b]
            if len(faces) > 1:
                return False
            if faces:
                active[k] = faces[0]
                point.append(faces[0])
            elif lo < a <= b < hi:
                point.append((a+b)/2)
            else:
                return False
        local = tuple(((x-lo)/(hi-lo),)*2 for x,(lo,hi) in zip(point,self.knots[root.witness]))
        pair = self.pair(root.witness)
        value, _ = pair.value_and_jacobian(pair.enclose(local))
        if not all(x.lo == x.hi == 0 for x in value):
            return False
        # The fixed section's uniqueness, not rounded root proximity, identifies
        # the constrained point. A common signed tangent enters EVERY face.
        self.charge()
        _, jac = pair.value_and_jacobian(pair.enclose(root.region))
        cofactors = tuple(_det([[row[j] for j in range(4) if j != k] for row in jac]).scale(
            -1 if k % 2 else 1) for k in range(4))
        if not cofactors[f].excludes_zero():
            return False
        inward = 1 if active[f] == ranges[f][0] else -1
        for k, face in active.items():
            product = (cofactors[k]*cofactors[f]).scale(inward)
            if face == ranges[k][0] and product.lo <= 0:
                return False
            if face == ranges[k][1] and product.hi >= 0:
                return False
        return True

    def enclosures(self, wi):
        w = self.witnesses[wi]
        native = tuple((k0+lo*(k1-k0), k0+hi*(k1-k0))
                       for (lo,hi),(k0,k1) in zip(w.region,self.knots[wi]))
        ca = chart_box(native[:2], self.first.surface.u_range, self.first.surface.v_range)
        cb = chart_box(native[2:], self.second.surface.u_range, self.second.surface.v_range)
        a, b = self.first.hull(native[:2]), self.second.hull(native[2:])
        low = tuple(max(x,y) for x,y in zip(a[0],b[0]))
        high = tuple(min(x,y) for x,y in zip(a[1],b[1]))
        if any(x>y for x,y in zip(low,high)):
            raise NurbsSurfaceIntersectionIntegrityError("graph original hull intersection is empty")
        from ._intervals import float_down, float_up
        return ca, cb, tuple(float_down(x) for x in low), tuple(float_up(x) for x in high)


def stitch(first, second, witnesses, limits, stats, tick):
    """Return immutable section roots and ordered components, or a typed refusal."""
    g = Graphs(first, second, witnesses, limits, stats, tick)
    roots, components, endpoints = [], [], {}
    assigned = {}  # whole connected graphs already joined to maximal components

    def endpoint(wi, side):
        key = wi, side
        if key not in endpoints:
            endpoints[key] = len(roots)
            roots.append(g.endpoint(wi, side))
        return endpoints[key]

    def inside_arc(root, arc):
        if root == roots[arc.first_root] or root == roots[arc.second_root]:
            return True
        local = g.in_graph(root, arc.witness)
        if local is None:
            return False
        f = witnesses[arc.witness].free_coordinate
        k0,k1 = g.knots[arc.witness][f]
        a,b = g.native(roots[arc.first_root])[f], g.native(roots[arc.second_root])[f]
        lo,hi = (k0+local[f][0]*(k1-k0), k0+local[f][1]*(k1-k0))
        if lo==hi and ((a[0]==a[1]==lo) or (b[0]==b[1]==lo)):
            return True  # same fixed section and graph uniqueness identify one root
        if arc.direction > 0:
            return a[1] < lo <= hi < b[0]
        return b[1] < lo <= hi < a[0]

    def outside_arc(root, arc):
        """Prove absence from an issued segment; uncertainty is never absence."""
        native = g.native(root)
        region = witnesses[arc.witness].region
        bounds = tuple((k0+lo*(k1-k0),k0+hi*(k1-k0))
                       for (lo,hi),(k0,k1) in zip(region,g.knots[arc.witness]))
        if any(b<c or d<a for (a,b),(c,d) in zip(native,bounds)):
            return True
        f = witnesses[arc.witness].free_coordinate
        a,b = g.native(roots[arc.first_root])[f],g.native(roots[arc.second_root])[f]
        low,high = (a,b) if arc.direction>0 else (b,a)
        return native[f][1]<low[0] or high[1]<native[f][0]

    def walk(start, wi, direction, allow_close):
        arcs, entry = [], start
        for _ in range(limits.max_trace_segments):
            exit_id = endpoint(wi, 1 if direction > 0 else 0)
            ca,cb,lo,hi = g.enclosures(wi)
            provisional = GenericArc(wi, entry, exit_id, direction, ca,cb,lo,hi)
            # Closing root is the original isolated root, never a nearby endpoint.
            if allow_close and arcs:
                for precision in range(limits.max_newton_iterations):
                    if inside_arc(roots[start],provisional):
                        arcs.append(GenericArc(wi,entry,start,direction,ca,cb,lo,hi))
                        return arcs,True
                    if outside_arc(roots[start],provisional):
                        break
                    # Ambiguous order is a precision obligation, never licence
                    # to pass the initial root and silently trace another lap.
                    for ident in sorted({start,entry,exit_id}):
                        roots[ident]=g.refine(roots[ident])
                else:
                    raise NurbsSurfaceIntersectionRefused("closing-section order remains unresolved")
            if entry != exit_id:
                arcs.append(provisional)
            if g.physical(roots[exit_id]):
                return arcs, False
            found = None
            for refinement in range(limits.max_newton_iterations):
                for candidate in g.candidates(roots[exit_id]):
                    if candidate == wi:
                        continue
                    transported = g.orientation(roots[exit_id], wi, candidate, direction)
                    if transported is not None:
                        found = candidate, transported
                        break
                if found is not None:
                    break
                roots[exit_id] = g.refine(roots[exit_id])
            if found is None:
                raise NurbsSurfaceIntersectionRefused("no certified oriented endpoint transition (boundary/seam or unresolved incidence)")
            entry, (wi,direction) = exit_id, found
        raise NurbsSurfaceIntersectionRefused("ordered continuation segment limit exhausted")

    for wi in range(len(witnesses)):
        if wi in assigned:
            continue
        start = endpoint(wi,0)
        joined = False
        separate = not components
        for _ in range(limits.max_newton_iterations):
            owner = assigned.get(wi)
            if owner is None:
                candidates=g.candidates(roots[start])
                for used in candidates:
                    if used in assigned and g.in_graph(roots[start],used) is not None:
                        owner = assigned[used]
                        break
            if owner is not None:
                assigned[wi] = owner
                joined = True
                break
            candidates=set(g.candidates(roots[start]))
            separate = all(outside_arc(roots[start],arc) for c in components for arc in c.arcs
                           if arc.witness in candidates)
            if separate:
                break
            if not components:
                break
            roots[start] = g.refine(roots[start])
        if joined:
            continue
        if not separate:
            raise NurbsSurfaceIntersectionRefused("duplicate-component incidence is unresolved")
        if len(components) >= limits.max_components:
            raise NurbsSurfaceIntersectionRefused("generic component limit exhausted")
        forward,closed = walk(start,wi,1,True)
        if not closed:
            backward,back_closed = walk(start,wi,-1,False)
            if back_closed:
                raise NurbsSurfaceIntersectionIntegrityError("open backward trace unexpectedly closed")
            backward = [GenericArc(a.witness,a.second_root,a.first_root,-a.direction,
                                   a.first_chart,a.second_chart,a.world_lo,a.world_hi) for a in reversed(backward)]
            arcs = backward + forward
        else:
            arcs = forward
        if not arcs or sum(len(c.arcs) for c in components)+len(arcs)>limits.max_trace_segments:
            raise NurbsSurfaceIntersectionRefused("empty path or total generic segment limit exhausted")
        ids = (arcs[0].first_root,) + tuple(a.second_root for a in arcs)
        if closed and ids[0] != ids[-1]:
            raise NurbsSurfaceIntersectionIntegrityError("closed path does not repeat its isolated root")
        ca = tuple((min(a.first_chart[k][0] for a in arcs),max(a.first_chart[k][1] for a in arcs)) for k in range(2))
        cb = tuple((min(a.second_chart[k][0] for a in arcs),max(a.second_chart[k][1] for a in arcs)) for k in range(2))
        low = tuple(min(a.world_lo[k] for a in arcs) for k in range(3))
        high = tuple(max(a.world_hi[k] for a in arcs) for k in range(3))
        components.append(ComponentRecord(len(components),closed,ids,tuple(arcs),ca,cb,low,high))
        assigned[wi] = len(components)-1
        for arc in arcs:
            assigned[arc.witness] = len(components)-1
    return tuple(roots),tuple(components)
