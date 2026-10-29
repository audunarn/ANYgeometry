"""Literal, source-bound intersection curves with an oriented common parameter.

The persistent records contain geometric sections and proof domains, never
adaptive array indices. Proofs are rederived for each operation. A changed
partition that cannot reproduce these literal sections refuses; it must never
silently select another branch by overlap or a digest.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import weakref

from ..errors import GeometryError
from ._source import (SourceDefinition, IntersectionStatus, isolate_callback,
    NurbsSurfaceIntersectionIntegrityError as Integrity,
    NurbsSurfaceIntersectionRefused as Refused,
    NurbsSurfaceIntersectionCancelled as Cancelled)
from ._intervals import Interval

_INTERVAL_PAYLOAD = object()
_RATIONAL_PAYLOAD = object()


def _covered_arc_indices(start, end, count):
    """Exact ordered overlap window; a singleton seam belongs to BOTH arcs.

    Inputs are already validated Fractions in [0,count]. Non-singleton
    intervals exclude arcs whose only intersection is a parameter endpoint,
    exactly as the original per-arc overlap predicate did.
    """
    first = start.numerator // start.denominator
    if start == end:
        last = first
        if start.denominator == 1:
            first -= 1
    else:
        last = -((-end.numerator) // end.denominator)-1
    return range(max(0, first), min(count-1, last)+1)


def _bound_literal(value):
    """No caller-visible interval or rational object is retained by the cache."""
    if type(value) is Interval:
        return (_INTERVAL_PAYLOAD,(value.lo.numerator,value.lo.denominator),
                (value.hi.numerator,value.hi.denominator))
    if type(value) is Fraction:
        return (_RATIONAL_PAYLOAD,value.numerator,value.denominator)
    if type(value) is tuple:
        return tuple(_bound_literal(x) for x in value)
    return value


def _restore_bound(value):
    if type(value) is tuple:
        if len(value) == 3 and value[0] is _INTERVAL_PAYLOAD:
            return Interval(Fraction(*value[1]),Fraction(*value[2]))
        if len(value) == 3 and value[0] is _RATIONAL_PAYLOAD:
            return Fraction(value[1],value[2])
        return tuple(_restore_bound(x) for x in value)
    return value

# Issuer identities only; no reusable proof/result cache. Entries disappear with
# their invocation-local context. Duck objects cannot manufacture proof authority.
_ISSUED_CONTEXTS = {}


def owner_binding(owner):
    return (owner.first_source.values,owner.second_source.values,owner.branch_anchor,owner.segments,owner.closed)


def prepared_context_binding(context):
    row = _ISSUED_CONTEXTS.get(id(context))
    if row is None or row[0]() is not context or type(context) is not row[2]():
        raise Integrity("prepared curve context was not issued by a source owner")
    return row[1]


def prepared_context_configuration(context):
    prepared_context_binding(context)
    return _ISSUED_CONTEXTS[id(context)][3]


def _root_key(result, ident):
    root = result.roots[ident]
    if result.algorithm != "generic-atlas":
        return ("scalar-section", root.axis, root.value, root.lo, root.hi,
                root.poly, root.edge)
    # Native span definitions, rather than witness/span array positions.
    w = result.atlas.witnesses[root.witness]
    from ._atlas import _Surface
    from ._source import WorkCounters
    prepared = tuple(_Surface(s, WorkCounters()) for s in (result.first_source,result.second_source))
    knots = tuple(prepared[j].spans[k][w.spans[j][k]] for j in range(2) for k in range(2))
    native = tuple((a+lo*(b-a),a+hi*(b-a)) for (lo,hi),(a,b) in zip(root.region,knots))
    a,b = knots[root.free_coordinate]
    return ("graph-section", root.free_coordinate, a+root.value*(b-a), native)


def _arc_key(result, arc):
    ends = (_root_key(result,arc.first_root),_root_key(result,arc.second_root))
    if result.algorithm != "generic-atlas":
        box = result.boxes[arc.box_index]
        return ("scalar",box.u_range,box.v_range,box.coefficients,box.kind,
                arc.slicing_axis,ends)
    from ._atlas import _Surface
    from ._source import WorkCounters
    prepared = tuple(_Surface(s,WorkCounters()) for s in (result.first_source,result.second_source))
    w = result.atlas.witnesses[arc.witness]
    knots = tuple(prepared[j].spans[k][w.spans[j][k]] for j in range(2) for k in range(2))
    return ("generic",knots,w.region,w.free_coordinate,arc.direction,ends)


@dataclass(frozen=True, slots=True)
class NurbsIntersectionCurve:
    """One ordered component; parameter range [0, number of fixed segments].

    Integer parameters are mandatory stations. A closed component has two
    distinct parameter occurrences of its seam, with one geometric seam key.
    ``segments`` and ``branch_anchor`` are literal geometric descriptors, not
    transient certificate objects. Construction alone grants no proof authority.
    """
    first_source: SourceDefinition
    second_source: SourceDefinition
    branch_anchor: tuple
    segments: tuple
    closed: bool

    @property
    def parameter_range(self):
        return (Fraction(0),Fraction(len(self.segments)))

    @property
    def domain(self):
        return self.parameter_range

    @property
    def breakpoints(self):
        return self.mandatory_stations

    @property
    def mandatory_stations(self):
        return tuple(Fraction(i) for i in range(len(self.segments)+1))

    @property
    def seam_identity(self):
        return self.branch_anchor if self.closed else None

    @classmethod
    def from_result(cls, result, component_index=0):
        """Capture descriptors; the first operation rederives all proof authority."""
        if result.status is not IntersectionStatus.CERTIFIED:
            raise Refused("a source curve requires a complete certified intersection")
        if isinstance(component_index,bool) or not isinstance(component_index,int) or component_index < 0:
            raise GeometryError("invalid component index")
        try:
            component = result.components[component_index]
        except IndexError as error:
            raise GeometryError("invalid component index") from error
        keys = tuple(_arc_key(result,arc) for arc in component.arcs)
        if not keys:
            raise Refused("an empty component is not a curve")
        return cls(SourceDefinition(tuple(result.first_source.values)),
                   SourceDefinition(tuple(result.second_source.values)),keys[0][-1][0],keys,component.closed)

    def _operation(self, first, second, tolerances, qualification, cancellation_check):
        from .query import query_nurbs_surface_intersection
        # Capture before executing user code, including request iterators.
        snapshot = (self.first_source,self.second_source,self.branch_anchor,self.segments,self.closed)
        a,b,anchor,segments,closed = snapshot
        if type(a) is not SourceDefinition or type(b) is not SourceDefinition:
            raise Integrity("curve needs literal canonical support definitions")
        expected_a,expected_b = SourceDefinition.capture(first),SourceDefinition.capture(second)
        if a != expected_a or b != expected_b:
            raise Integrity("curve support definitions differ from live sources")
        callback = isolate_callback(cancellation_check)
        def tick():
            if callback is not None and callback():
                raise Cancelled("source curve operation cancelled")
            if not expected_a.matches(first) or not expected_b.matches(second):
                raise Integrity("a curve support changed during its operation")
            if self.first_source != expected_a or self.second_source != expected_b:
                raise Integrity("literal curve sources changed during their operation")
            if snapshot != (self.first_source,self.second_source,self.branch_anchor,self.segments,self.closed):
                raise Integrity("curve descriptors changed during their operation")
        tick()
        result = query_nurbs_surface_intersection(first,second,tolerances=tolerances,
            qualification=qualification,cancellation_check=tick)
        if result.status is not IntersectionStatus.CERTIFIED:
            raise Refused("source curve canonical revalidation did not certify a complete intersection")
        matches = []
        for ci, component in enumerate(result.components):
            if component.closed != closed or len(component.arcs) != len(segments):
                continue
            keys = tuple(_arc_key(result,arc) for arc in component.arcs)
            # Cyclic rotations preserve the persistent seam occurrence. Reversal
            # is deliberately not adopted: oriented pcurves have one direction.
            offsets = range(len(keys)) if closed else (0,)
            for offset in offsets:
                rotated = keys[offset:]+keys[:offset]
                if rotated == segments and rotated[0][-1][0] == anchor:
                    matches.append((ci,tuple((i+offset)%len(keys) for i in range(len(keys)))))
        if len(matches) != 1:
            raise Integrity("literal curve sections have no unique oriented canonical branch correspondence")
        tick()
        return result,matches[0],tick

    def evaluate_many(self, parameters, *, first, second, tolerances,
                      qualification=None, cancellation_check=None):
        """Certified points on both oriented pcurves and their common world curve.

        Validation is per operation, shared by all requested parameters through
        the freshly validated continuation context.
        """
        return self.prepare(first=first,second=second,tolerances=tolerances,
            qualification=qualification,cancellation_check=cancellation_check).evaluate_many(parameters)

    def evaluate(self, parameter, **kwargs):
        return self.evaluate_many((parameter,),**kwargs)[0]

    def enclose(self, start, end, *, first, second, tolerances,
                qualification=None, cancellation_check=None):
        """Certified continuous subarc and derivative bounds, split at stations."""
        return self.prepare(first=first,second=second,tolerances=tolerances,
            qualification=qualification,cancellation_check=cancellation_check).enclose(start,end)

    def prepare(self, *, first, second, tolerances, qualification=None,cancellation_check=None):
        """Fresh operation context: one canonical proof, many local evaluations.

        Keep this context within one consumer operation. It always checks live
        definitions and cancellation; it is never a persisted or crossquery cache.
        """
        import dataclasses
        import time
        from ..tolerance import TolerancePolicy
        from ._source import Limits
        from ._curve_bounds import prepare_bound_evaluator
        from ._structural_evaluate import _evaluate_validated_structural
        from ._generic_evaluate import _evaluate_validated_generic
        from ._curve_witness import FirstOrderChartWitness
        tolerances = TolerancePolicy(**{f.name:getattr(tolerances,f.name) for f in dataclasses.fields(tolerances)})
        limits = Limits.read(qualification)
        started = time.perf_counter()
        result,(ci,indices),tick = self._operation(first,second,tolerances,limits,cancellation_check)
        n = len(indices)
        support_values = (tuple(result.first_source.values),tuple(result.second_source.values))
        point_cache,bounds_cache,world_cache,chart_cache = {},{},{},{}
        bound_evaluator = prepare_bound_evaluator(result,ci)
        first_order_witness = FirstOrderChartWitness(result,ci,tick)
        membership = []
        class PreparedCurve:
            __slots__ = ('__weakref__',)
            domain = (Fraction(0),Fraction(n))
            breakpoints = tuple(Fraction(i) for i in range(n+1))
            def support_index(context, surface):
                tick()
                values = SourceDefinition.capture(surface).values
                matches = [i for i,v in enumerate(support_values) if values == v]
                if len(matches) != 1:
                    raise Integrity("pcurve needs one uniquely identified literal support")
                return matches[0]
            def evaluate_many(context, parameters):
                requests = []
                values = []
                for value in parameters:
                    tick()
                    if len(requests) >= limits.max_trace_segments:
                        raise Refused("source curve evaluation batch limit exhausted")
                    value = Fraction(value)
                    if not 0 <= value <= n:
                        raise GeometryError("curve parameter leaves its fixed range")
                    i = min(int(value),n-1)
                    requests.append((ci,indices[i],value-i))
                    values.append(value)
                missing = []
                pending = set()
                for value,request in zip(values,requests):
                    if value not in point_cache and value not in pending:
                        missing.append((value,request)); pending.add(value)
                args = dict(tolerances=tolerances,limits=limits,tick=tick,live=tick,started=started)
                if not missing:
                    output = ()
                elif result.algorithm == "generic-atlas":
                    output = _evaluate_validated_generic(result,[r for value,r in missing],**args)
                else:
                    output = _evaluate_validated_structural(result,[r for value,r in missing],
                        source_a=result.first_source,source_b=result.second_source,**args)
                from ._evaluate import ArcEvaluation
                for (value,request),point in zip(missing,output):
                    point_cache[value] = tuple(getattr(point,f.name) for f in dataclasses.fields(point))
                tick()
                # No caller-visible mutable record is ever the cached authority.
                issued = tuple(ArcEvaluation(*point_cache[value]) for value in values)
                if len(point_cache) > 16384:
                    point_cache.clear()
                return issued
            def evaluate(context, parameter):
                return context.evaluate_many((parameter,))[0]
            def chart_derivative_witness_many(context, parameters):
                """Approximate first-order values ONLY for advisory integration."""
                values = []
                for parameter in parameters:
                    tick()
                    if len(values) >= limits.max_trace_segments:
                        raise Refused("source derivative-witness batch limit exhausted")
                    values.append(Fraction(parameter))
                points = context.evaluate_many(values)
                output = []
                for value,point in zip(values,points):
                    tick()
                    i = min(int(value),n-1)
                    output.append(first_order_witness.evaluate(indices[i],point))
                tick()
                return tuple(output)
            def enclose(context, start, end):
                start,end = Fraction(start),Fraction(end)
                if not 0 <= start <= end <= n:
                    raise GeometryError("invalid source curve interval")
                output = []
                for i in _covered_arc_indices(start,end,n):
                    arc_index = indices[i]
                    lo,hi = max(start,Fraction(i)),min(end,Fraction(i+1))
                    if lo > hi or lo == hi and start != end:
                        continue
                    tick()
                    key = (arc_index,lo,hi)
                    if key not in bounds_cache:
                        bound = bound_evaluator(arc_index,(lo,hi),(lo-i,hi-i),tick)
                        if len(bounds_cache) >= 2048:
                            bounds_cache.clear()
                        bounds_cache[key] = _bound_literal(tuple(getattr(bound,f.name) for f in dataclasses.fields(bound)))
                    from ._curve_bounds import CurveSegmentBounds
                    output.append(CurveSegmentBounds(*_restore_bound(bounds_cache[key])))
                tick()
                return tuple(output)
            def _world_bounds(context,start,end):
                tick()
                start,end = Fraction(start),Fraction(end)
                tick()
                if not 0 <= start <= end <= n:
                    raise GeometryError("invalid source curve world interval")
                output = []
                for i in _covered_arc_indices(start,end,n):
                    arc_index = indices[i]
                    lo,hi = max(start,Fraction(i)),min(end,Fraction(i+1))
                    # Exactly the full-bound endpoint/seam accounting: a
                    # singleton at an arc joint includes BOTH incident arcs.
                    # Keep original-support hull precision instead of the
                    # coarser point-evaluation stopping target.
                    if lo > hi or lo == hi and start != end:
                        continue
                    tick()
                    key = (arc_index,lo.numerator,lo.denominator,hi.numerator,hi.denominator)
                    if key not in world_cache:
                        world = bound_evaluator(arc_index,(lo,hi),(lo-i,hi-i),tick,world_only=True)
                        if len(world_cache) >= 2048:
                            world_cache.clear()
                        world_cache[key] = world  # detached immutable float tuples
                    output.append(world_cache[key])
                tick()
                return (tuple(min(box[0][k] for box in output) for k in range(3)),
                        tuple(max(box[1][k] for box in output) for k in range(3)))
            def _chart_bounds(context,start,end):
                """Same contracted original-support charts, without unused jets."""
                tick()
                start,end = Fraction(start),Fraction(end)
                tick()
                if not 0 <= start <= end <= n:
                    raise GeometryError("invalid source curve chart interval")
                output = []
                for i in _covered_arc_indices(start,end,n):
                    arc_index = indices[i]
                    lo,hi = max(start,Fraction(i)),min(end,Fraction(i+1))
                    tick()
                    key = (arc_index,lo.numerator,lo.denominator,hi.numerator,hi.denominator)
                    if key not in chart_cache:
                        charts = bound_evaluator(arc_index,(lo,hi),(lo-i,hi-i),tick,chart_only=True)
                        if len(chart_cache) >= 2048:
                            chart_cache.clear()
                        chart_cache[key] = _bound_literal(charts)
                    output.append(_restore_bound(chart_cache[key]))
                tick()
                return tuple(tuple((min(box[support][k][0] for box in output),
                                    max(box[support][k][1] for box in output)) for k in range(2))
                             for support in range(2))
            def shape(context):
                return PreparedCurveShape(context,first,second)
            def _closed_sign_membership(context,point,clearance):
                tick()
                if not membership:
                    from ._closed_membership import ClosedSignMembership
                    membership.append(ClosedSignMembership(result,ci,context._chart_bounds,tick))
                return membership[0].contains(point,clearance)
        context = PreparedCurve()
        identity = id(context)
        _ISSUED_CONTEXTS[identity] = (weakref.ref(context,lambda ref:_ISSUED_CONTEXTS.pop(identity,None)),
            owner_binding(self),weakref.ref(type(context)),
            (tuple((f.name,getattr(tolerances,f.name)) for f in dataclasses.fields(tolerances)),
             tuple((f.name,getattr(limits,f.name)) for f in dataclasses.fields(limits))))
        return context


class PreparedCurveShape:
    """Operation-local normalized world curve and two oriented support pcurves.

    World/chart point and derivative methods return approximate witnesses. Use
    ``enclose``/``chart_enclosure`` for proof-bearing consumer decisions.
    """
    def __init__(self, context, first, second, start=None, end=None):
        binding = prepared_context_binding(context)
        self._context = context
        self._sources = (SourceDefinition.capture(first),SourceDefinition.capture(second))
        if tuple(s.values for s in self._sources) != binding[:2]:
            raise Integrity("prepared shape supports differ from its issued owner context")
        self._start = context.domain[0] if start is None else Fraction(start)
        self._end = context.domain[1] if end is None else Fraction(end)
        if self._start == self._end:
            raise Refused("a constant subcurve has no regular curve parameter")
        self._definition = (context,tuple(s.values for s in self._sources),self._start,self._end)

    def _validate(self):
        prepared_context_binding(self._context)
        expected_context,expected_sources,expected_start,expected_end = self._definition
        if (self._context is not expected_context or tuple(s.values for s in self._sources) != expected_sources
                or self._start != expected_start or self._end != expected_end):
            raise Integrity("prepared curve view changed after its invocation-local binding")

    @property
    def domain(self):
        return (0.0,1.0)

    @property
    def breakpoints(self):
        delta = self._end-self._start
        return tuple(sorted({Fraction(0),Fraction(1)} | {
            (x-self._start)/delta for x in self._context.breakpoints
            if min(self._start,self._end) < x < max(self._start,self._end)}))

    def _parameter(self, value):
        self._validate()
        value = Fraction(value)
        self._validate()
        if not 0 <= value <= 1:
            raise GeometryError("normalized curve parameter leaves [0,1]")
        result = self._start+value*(self._end-self._start)
        self._validate()
        return result

    def evaluate(self, parameter):
        if isinstance(parameter,(list,tuple)):
            return tuple(x.witness for x in self._context.evaluate_many(self._parameter(t) for t in parameter))
        return self._context.evaluate(self._parameter(parameter)).witness

    def enclose(self, start=0, end=1):
        import dataclasses
        self._validate()
        if Fraction(start) > Fraction(end):
            raise GeometryError("normalized interval endpoints are reversed")
        self._validate()
        a,b = self._parameter(start),self._parameter(end)
        delta = self._end-self._start
        output = []
        for bound in self._context.enclose(min(a,b),max(a,b)):
            first = {name:tuple(x.scale(delta) for x in getattr(bound,name)) for name in (
                "first_chart_derivative","second_chart_derivative","world_derivative")}
            second = {name:tuple(x.scale(delta*delta) for x in getattr(bound,name)) for name in (
                "first_chart_second_derivative","second_chart_second_derivative","world_second_derivative")}
            parameter_range = tuple(sorted((x-self._start)/delta for x in bound.parameter_range))
            output.append(dataclasses.replace(bound,parameter_range=parameter_range,**first,**second))
        self._validate()
        return tuple(output)

    def _world_bounds(self,start=0,end=1):
        self._validate()
        if Fraction(start) > Fraction(end):
            raise GeometryError("normalized world interval endpoints are reversed")
        self._validate()
        a,b = self._parameter(start),self._parameter(end)
        result = self._context._world_bounds(min(a,b),max(a,b))
        self._validate()
        return result

    def _chart_bounds(self,start=0,end=1):
        self._validate()
        if Fraction(start) > Fraction(end):
            raise GeometryError("normalized chart interval endpoints are reversed")
        self._validate()
        a,b = self._parameter(start),self._parameter(end)
        result = self._context._chart_bounds(min(a,b),max(a,b))
        self._validate()
        return result

    def derivative(self, parameter):
        boxes = self.enclose(parameter,parameter)
        return tuple(float(x.mid) for x in boxes[-1].world_derivative)

    def chart_derivative_witness_many(self, surface, parameters):
        """Advisory first-order chart values, never proof-bearing intervals."""
        import math
        from ._curve_witness import FirstOrderWitnessNotQualified
        self._validate()
        support = self._support(surface)
        values = tuple(self._parameter(value) for value in parameters)
        self._validate()
        rows = self._context.chart_derivative_witness_many(values)
        delta = float(self._end-self._start)
        if not math.isfinite(delta) or delta == 0.:
            raise FirstOrderWitnessNotQualified("derivative witness view scale is not representable")
        output = tuple(tuple(value*delta for value in row[2*support:2*support+2]) for row in rows)
        if not all(math.isfinite(x) for row in output for x in row):
            raise FirstOrderWitnessNotQualified("scaled chart derivative witness is not finite")
        self._validate()
        return output

    def sample(self, count=33):
        if isinstance(count,bool) or not isinstance(count,int) or count < 2:
            raise GeometryError("sample count must be an integer at least two")
        return self.evaluate(tuple(Fraction(i,count-1) for i in range(count)))

    def subcurve(self, start, end):
        return PreparedCurveShape._from_sources(self._context,self._sources,
            self._parameter(start),self._parameter(end))

    @classmethod
    def _from_sources(cls, context, sources, start, end):
        return cls(context,*(s.rebuild() for s in sources),start,end)

    def transformed(self, transform):
        raise Refused("transform both literal supports and rederive their intersection curve")

    def _support(self, surface):
        self._validate()
        return self._context.support_index(surface)

    def chart(self, surface, parameter):
        return self.chart_many(surface,(parameter,))[0]

    def chart_many(self, surface, parameters):
        """One certified point batch mapped onto one unchanged literal support."""
        self._validate()
        expected = SourceDefinition.capture(surface)
        support = self._support(surface)
        if not expected.matches(surface):
            raise Integrity("pcurve support changed during batch selection")
        # The existing bounded batch consumes this iterator. Do not first
        # materialize an unbounded caller iterable outside its work limit.
        points = self._context.evaluate_many(self._parameter(value) for value in parameters)
        output = []
        for point in points:
            box = (point.first_chart,point.second_chart)[support]
            output.append(tuple((a+b)/2 for a,b in box))
        self._validate()
        if not expected.matches(surface) or self._support(surface) != support:
            raise Integrity("pcurve support changed during point evaluation")
        self._validate()
        if not expected.matches(surface):
            raise Integrity("pcurve support changed at batch issuance")
        return tuple(output)

    def chart_enclosure(self, surface, start=0, end=1):
        key = ("first_chart","second_chart")[self._support(surface)]
        boxes = tuple(getattr(x,key) for x in self.enclose(start,end))
        return tuple((min(box[k][0] for box in boxes),max(box[k][1] for box in boxes)) for k in range(2))


def encode_curve(curve):
    """JSON-compatible literal encoding; no cached certificates or ordinal IDs."""
    def encode(value):
        if isinstance(value,Fraction):
            return {"rational":[str(value.numerator),str(value.denominator)]}
        if isinstance(value,tuple):
            return [encode(x) for x in value]
        if value is None or type(value) in (str,int,float,bool):
            return value
        raise GeometryError("unsupported curve literal")
    return {"schema":"anygeometry.nurbs-intersection-curve.v1",
            "first":encode(curve.first_source.values),"second":encode(curve.second_source.values),
            "anchor":encode(curve.branch_anchor),"segments":encode(curve.segments),"closed":curve.closed}


def decode_curve(payload):
    """Decode untrusted literals. Evaluation always rederives branch proofs."""
    def decode(value):
        if isinstance(value,list):
            return tuple(decode(x) for x in value)
        if isinstance(value,dict) and set(value)=={"rational"}:
            n,d = value["rational"]
            return Fraction(int(n),int(d))
        if value is None or type(value) in (str,int,float,bool):
            return value
        raise GeometryError("malformed curve literal")
    if (type(payload) is not dict or set(payload)!={"schema","first","second","anchor","segments","closed"}
            or payload["schema"]!="anygeometry.nurbs-intersection-curve.v1" or type(payload["closed"]) is not bool):
        raise GeometryError("unsupported curve encoding")
    a,b = SourceDefinition(decode(payload["first"])),SourceDefinition(decode(payload["second"]))
    a,b = SourceDefinition.capture(a.rebuild()),SourceDefinition.capture(b.rebuild())
    return NurbsIntersectionCurve(a,b,decode(payload["anchor"]),decode(payload["segments"]),payload["closed"])
