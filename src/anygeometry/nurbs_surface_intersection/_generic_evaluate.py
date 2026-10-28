"""Original-surface enclosure evaluation of generic algebraic section arcs."""
from __future__ import annotations

import dataclasses
import time
from fractions import Fraction

from ..errors import GeometryError
from ..tolerance import TolerancePolicy
from ._atlas import _Surface
from ._continuation import Graphs
from ._intervals import Interval, float_down, float_up, diameter_upper
from ._records import digest_of
from ._source import (IntersectionStatus, IntersectionResult, SourceDefinition, Limits, WorkCounters, isolate_callback,
                      NurbsSurfaceIntersectionIntegrityError as Integrity,
                      NurbsSurfaceIntersectionRefused as Refused,
                      NurbsSurfaceIntersectionCancelled as Cancelled)
from ._topology import chart_box


def evaluate_generic(result, component_index, arc_index, station, **kwargs):
    return evaluate_generic_batch(result,((component_index,arc_index,station),),**kwargs)[0]


def evaluate_generic_batch(result, requests, *, first, second,
                           tolerances, qualification=None, cancellation_check=None):
    started = time.perf_counter()
    from ._evaluate import ArcEvaluation
    from .query import query_nurbs_surface_intersection
    if not isinstance(tolerances,TolerancePolicy):
        raise GeometryError("tolerances must be a TolerancePolicy")
    if cancellation_check is not None and not callable(cancellation_check):
        raise GeometryError("cancellation_check must be callable")
    if (type(result) is not IntersectionResult or type(result.first_source) is not SourceDefinition
            or type(result.second_source) is not SourceDefinition):
        raise Integrity("generic evaluation requires canonical result/source record types")
    expected_a,expected_b = SourceDefinition.capture(first),SourceDefinition.capture(second)
    if result.first_source!=expected_a or result.second_source!=expected_b:
        raise Integrity("generic result source differs from actual input definitions")
    tolerances = TolerancePolicy(**{f.name:getattr(tolerances,f.name) for f in dataclasses.fields(tolerances)})
    limits = Limits.read(qualification)
    callback = isolate_callback(cancellation_check)
    def live():
        if not expected_a.matches(first) or not expected_b.matches(second):
            raise Integrity("an input surface differs from the generic result source")
    def tick():
        if callback is not None and callback():
            raise Cancelled("generic arc evaluation cancelled")
        live()
    tick()

    # Canonical re-derivation checks proof authority independently of a digest;
    # changing and rehashing a record cannot create a valid root or path.
    canonical = query_nurbs_surface_intersection(first,second,tolerances=tolerances,
                                                qualification=qualification,cancellation_check=tick)
    if (canonical.status is not IntersectionStatus.CERTIFIED or canonical.algorithm!="generic-atlas"
        or (result.roots,result.components,result.atlas)!=(canonical.roots,canonical.components,canonical.atlas)
        or digest_of(result.atlas.witnesses,result.roots,(),result.components)!=result.descriptor_digest):
        raise Integrity("generic descriptors differ from the canonical complete cover and continuation")
    # From this boundary onward, only newly derived trusted records are used.
    # The caller may mutate its frozen input using object.__setattr__ in a
    # callback or request iterator; it cannot change these invocation locals.
    return _evaluate_validated_generic(canonical,requests,tolerances=tolerances,limits=limits,
        tick=tick,live=live,started=started)


def _evaluate_validated_generic(canonical, requests, *, tolerances, limits, tick, live, started):
    """Internal invocation-local continuation; canonical must be freshly rederived."""
    from ._evaluate import ArcEvaluation
    result = canonical
    validation_seconds = time.perf_counter()-started
    stats = WorkCounters()
    a,b = _Surface(result.first_source,stats),_Surface(result.second_source,stats)
    g = Graphs(a,b,result.atlas.witnesses,limits,stats,tick)
    root_cache=dict(enumerate(result.roots))
    endpoint_cache={}
    target=min(tolerances.length,tolerances.surface_residual)
    target2=Fraction(target)**2
    def evaluate_one(component_index,arc_index,station):
        station=Fraction(station)
        if not 0<=station<=1:
            raise GeometryError("station must lie in [0,1]")
        try:
            if component_index<0 or arc_index<0:
                raise IndexError
            arc = result.components[component_index].arcs[arc_index]
        except (IndexError,TypeError) as error:
            raise GeometryError("no such component or arc") from error
        endpoint_ids=(arc.first_root,arc.second_root)
        endpoint_id=endpoint_ids[0 if station==0 else 1] if station in (0,1) else None
        def issue(payload,steps):
            native,ca,cb,low,high,witness,diameter=payload
            tick()
            live()
            return ArcEvaluation(component_index,arc_index,station,native,ca,cb,low,high,witness,
                                 False,diameter,target,steps,validation_seconds,time.perf_counter()-started)
        if endpoint_id is not None and endpoint_id in endpoint_cache:
            return issue(endpoint_cache[endpoint_id],0)
        w = result.atlas.witnesses[arc.witness]
        free = w.free_coordinate
        region = w.region
        cap = 4*limits.max_newton_iterations
        for step in range(cap+1):
            tick()
            endpoints=[root_cache[ident] for ident in endpoint_ids]
            if station in (0,1):
                root = endpoints[0 if station==0 else 1]
                native = g.native(root)
            else:
                x,y = (Interval(*g.native(r)[free]) for r in endpoints)
                value = x.scale(1-station)+y.scale(station)
                k0,k1 = g.knots[arc.witness][free]
                phi = Interval((value.lo-k0)/(k1-k0),(value.hi-k0)/(k1-k0))
                if not (w.region[free][0]<=phi.lo<=phi.hi<=w.region[free][1]):
                    raise Refused("algebraic station cannot yet be enclosed in its certified graph interval")
                region = g.contract(arc.witness,free,phi,region)
                native = tuple((lo0+x*(hi0-lo0),lo0+y*(hi0-lo0))
                               for (x,y),(lo0,hi0) in zip(region,g.knots[arc.witness]))
            ha,hb = a.hull(native[:2]),b.hull(native[2:])
            low = tuple(float_down(max(x,y)) for x,y in zip(ha[0],hb[0]))
            high = tuple(float_up(min(x,y)) for x,y in zip(ha[1],hb[1]))
            if any(x>y for x,y in zip(low,high)):
                raise Integrity("original surface point hulls have empty intersection")
            squared = sum((Fraction(y)-Fraction(x))**2 for x,y in zip(low,high))
            if squared<=target2:
                ca = chart_box(native[:2],a.surface.u_range,a.surface.v_range)
                cb = chart_box(native[2:],b.surface.u_range,b.surface.v_range)
                chart = tuple((x+y)/2 for x,y in ca)
                witness = tuple(float(v) for v in a.surface.evaluate(*chart))
                witness = tuple(min(max(v,l),h) for v,l,h in zip(witness,low,high))
                payload=(native,ca,cb,low,high,witness,diameter_upper(squared))
                if endpoint_id is not None:
                    endpoint_cache[endpoint_id]=payload
                return issue(payload,step)
            if step<cap:
                needed=endpoint_ids if endpoint_id is None else (endpoint_id,)
                for ident in sorted(set(needed)):
                    root_cache[ident]=g.refine(root_cache[ident])
        raise Refused("generic point enclosure cannot meet the absolute length/residual target within its work bound")
    output=[]
    for request in requests:
        tick()
        if len(output)>=limits.max_trace_segments:
            raise Refused("evaluation batch point limit exhausted")
        output.append(evaluate_one(*request))
    tick()
    live()
    return tuple(output)
