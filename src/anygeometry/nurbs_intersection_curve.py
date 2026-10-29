"""Persistent two-support intersection edges and operation-local proof views."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace, field, fields
from fractions import Fraction as Q
from functools import wraps

import numpy as np

from .errors import GeometryError
from .tolerance import TolerancePolicy
from .nurbs_surface_intersection import NurbsIntersectionCurve, encode_curve, decode_curve

_OPERATIONS = ContextVar('nurbs_surface_intersection_operation',default=None)
_FACT_ISSUER = ContextVar('nurbs_surface_intersection_fact_issuer',default=None)
_SCRATCH_NAMESPACE = object()


def _new_facts():
    # The expected payload registry is closed over, separate from the bank a
    # caller can inspect. Replacing/tampering with a stored row cannot issue proof.
    rows,issued = {},{}
    def access(key,builder):
        if key in rows:
            row = rows[key]
            expected = issued.get(id(row))
            if expected is None or expected[0] is not row or row != expected[1] or row[0] != key:
                raise GeometryError('operation proof cache binding changed')
            return row[1]
        value = builder()
        row = (key,value)
        if len(rows) >= 4096:
            rows.clear(); issued.clear()
        rows[key] = row
        issued[id(row)] = (row,row)
        return value
    return access


def _new_checker_runner(checks):
    # Shared CLOSED state, not a ContextVar value: isolate_callback may capture
    # a copied Context during reentrant checksum validation. Unwind must still
    # clear the exact active identities seen by all those copied contexts.
    active = set()
    def run():
        for check in tuple(checks):
            identity = id(check)
            if identity in active:
                continue
            active.add(identity)
            try:
                if check():
                    raise GeometryError('two-support curve consumer operation cancelled')
            finally:
                active.remove(identity)
        return False
    return run


def _operation_fact(namespace,key_factory,builder):
    bank = _OPERATIONS.get()
    if bank is None:
        with intersection_curve_operation():
            return _operation_fact(namespace,key_factory,builder)
    _operation_check()
    key = (namespace,key_factory())
    if bank['_facts'] is not _FACT_ISSUER.get():
        raise GeometryError('operation proof cache issuer changed')
    value = bank['_facts'](key,builder)
    _operation_check()
    if (namespace,key_factory()) != key:
        raise GeometryError('operation proof source changed before issuance')
    return value


def operation_fact(key_factory,builder):
    """Caller scratch reuse, isolated from every owner proof/estimate producer.

    This legacy generic helper grants no owner proof authority. A caller cannot
    seed an owner's closed producer namespace by reconstructing a literal key.
    """
    return _operation_fact(_SCRATCH_NAMESPACE,key_factory,builder)


def operation_cached(key_factory):
    """Capture one trusted builder and opaque namespace, once at definition.

    The resulting callable accepts only its real geometry arguments. Ordinary
    callers can make their own producers but cannot replace an owner's builder
    or reconstruct another producer's namespace using literal geometry keys.
    """
    def decorate(builder):
        namespace = object()
        @wraps(builder)
        def evaluate(*args,**kwargs):
            return _operation_fact(namespace,lambda:key_factory(*args,**kwargs),
                                   lambda:builder(*args,**kwargs))
        return evaluate
    return decorate


@contextmanager
def intersection_curve_operation(check=None):
    """Reuse proofs only within this explicit geometry/mesh consumer operation."""
    existing = _OPERATIONS.get()
    if existing is not None:
        if check is not None:
            existing['_checks'].append(check)
        try:
            yield
        finally:
            if check is not None:
                existing['_checks'].pop()
        return
    facts = _new_facts()
    issuer_token = _FACT_ISSUER.set(facts)
    checks = [] if check is None else [check]
    token = _OPERATIONS.set({'_checks':checks,'_facts':facts,'_runchecks':_new_checker_runner(checks)})
    try:
        yield
    finally:
        _OPERATIONS.reset(token)
        _FACT_ISSUER.reset(issuer_token)


def _operation_check():
    bank = _OPERATIONS.get()
    if bank is not None:
        bank['_runchecks']()
    return False


@dataclass(frozen=True)
class NurbsSurfaceIntersectionCurve:
    owner: NurbsIntersectionCurve
    domain: tuple = (Q(0),Q(1))
    chart_support: int = 0
    tolerance_values: tuple = field(default_factory=lambda:tuple((f.name,getattr(TolerancePolicy(),f.name))
                                                               for f in fields(TolerancePolicy)))
    qualification_values: tuple = ()

    def __post_init__(self):
        if type(self.owner) is not NurbsIntersectionCurve:
            raise GeometryError('intersection edge needs its literal geometry owner')
        a,b = map(Q,self.domain)
        if not (0 <= a <= 1 and 0 <= b <= 1 and a != b):
            raise GeometryError('intersection edge needs a nonzero normalized source interval')
        if type(self.chart_support) is not int or self.chart_support not in (0,1):
            raise GeometryError('invalid intersection pcurve support')
        object.__setattr__(self,'domain',(a,b))
        TolerancePolicy(**dict(self.tolerance_values))

    @property
    def dimension(self):
        return 3

    @property
    def surface(self):
        return (self.owner.first_source,self.owner.second_source)[self.chart_support].rebuild()

    @property
    def source_key(self):
        return (self.owner.first_source.values,self.owner.second_source.values,
                self.owner.branch_anchor,self.owner.segments,self.owner.closed)

    def for_support(self,surface):
        from .nurbs_surface_intersection import SourceDefinition
        source = SourceDefinition.capture(surface)
        indices = [i for i,s in enumerate((self.owner.first_source,self.owner.second_source)) if s == source]
        if len(indices) != 1:
            raise GeometryError('edge has no uniquely identified pcurve on this support')
        return replace(self,chart_support=indices[0])

    def _shape(self):
        _operation_check()
        from .nurbs_surface_intersection.curve import (prepared_context_binding,prepared_context_configuration,
                                                       owner_binding,PreparedCurveShape)
        from .nurbs_surface_intersection._source import Limits
        from types import SimpleNamespace
        effective_limits = Limits.read(SimpleNamespace(**dict(self.qualification_values)))
        configuration = (self.tolerance_values,tuple((f.name,getattr(effective_limits,f.name)) for f in fields(effective_limits)))
        bank = _OPERATIONS.get()
        key = ('owner',self.owner,self.tolerance_values,self.qualification_values)
        context = bank.get(key) if bank is not None else None
        if context is not None and prepared_context_binding(context) != owner_binding(self.owner):
            raise GeometryError('cached prepared context belongs to another literal owner')
        if context is not None and prepared_context_configuration(context) != configuration:
            raise GeometryError('cached prepared context has different effective tolerance or work limits')
        if context is None:
            from types import SimpleNamespace
            context = self.owner.prepare(first=self.owner.first_source.rebuild(),
                second=self.owner.second_source.rebuild(),tolerances=TolerancePolicy(**dict(self.tolerance_values)),
                qualification=SimpleNamespace(**dict(self.qualification_values)),cancellation_check=_operation_check)
            if bank is not None:
                bank[key] = context
        key = ('shape',self)
        shape = bank.get(key) if bank is not None else None
        if shape is not None:
            if type(shape) is not PreparedCurveShape or prepared_context_binding(shape._context) != owner_binding(self.owner):
                raise GeometryError('cached prepared shape belongs to another literal owner')
            shape._validate()
            n = len(self.owner.segments)
            if (shape._start,shape._end) != (self.domain[0]*n,self.domain[1]*n):
                raise GeometryError('cached prepared shape has a different persistent parameter interval')
            if prepared_context_configuration(shape._context) != configuration:
                raise GeometryError('cached prepared shape has different effective tolerance or work limits')
        if shape is None:
            shape = context.shape().subcurve(*self.domain)
            if bank is not None:
                bank[key] = shape
        return shape

    def evaluate(self,parameters):
        values = np.asarray(parameters,dtype=float)
        with intersection_curve_operation():
            shape = self._shape()
            flat = values.reshape(-1)
            output = np.asarray(shape.evaluate(tuple(float(t) for t in flat)),dtype=float)
            return output.reshape(values.shape+(3,))

    def derivative(self,parameters):
        values = np.asarray(parameters,dtype=float)
        with intersection_curve_operation():
            shape = self._shape()
            output = np.asarray([shape.derivative(float(t)) for t in values.reshape(-1)],dtype=float)
            return output.reshape(values.shape+(3,))

    def sample(self,count=33):
        return self.evaluate(np.linspace(0.,1.,count))

    def breakpoints(self):
        a,b = self.domain
        n = len(self.owner.segments)
        return tuple(float((Q(i,n)-a)/(b-a)) for i in range(n+1) if 0 < (Q(i,n)-a)/(b-a) < 1)

    def subcurve(self,lower,upper):
        lower,upper = Q(lower),Q(upper)
        if not 0 <= lower <= 1 or not 0 <= upper <= 1 or lower == upper:
            raise GeometryError('invalid intersection subcurve interval')
        a,b = self.domain
        return replace(self,domain=(a+(b-a)*lower,a+(b-a)*upper))

    def transformed(self,matrix):
        if np.array_equal(np.asarray(matrix,dtype=float),np.eye(4)):
            return self
        raise GeometryError('intersection edges require transformed literal supports and rederived branch correspondence')

    def enclose(self,lower=0,upper=1):
        with intersection_curve_operation():
            return self._shape().enclose(lower,upper)

    def bounds(self,lower=0,upper=1):
        with intersection_curve_operation():
            low,high = self._shape()._world_bounds(lower,upper)
            return np.asarray(low),np.asarray(high)

    def end_box(self,end):
        return self.bounds(end,end)

    def chart(self,parameters):
        values = np.asarray(parameters,dtype=float)
        with intersection_curve_operation():
            shape = self._shape()
            output = np.asarray(shape.chart_many(self.surface,tuple(float(t) for t in values.reshape(-1))),dtype=float)
            return output.reshape(values.shape+(2,))

    def chart_enclosure(self,parameter):
        return self.chart_box(parameter,parameter)

    def chart_box(self,lower=0,upper=1):
        with intersection_curve_operation():
            box = self._shape()._chart_bounds(lower,upper)[self.chart_support]
            return tuple((Q(lo),Q(hi)) for lo,hi in box)

    def chart_bounds(self,lower=0,upper=1):
        from .nurbs_surface_intersection._intervals import float_down,float_up
        box = self.chart_box(lower,upper)
        return (np.asarray([float_down(lo) for lo,hi in box]),
                np.asarray([float_up(hi) for lo,hi in box]))

    def chart_derivative_bounds(self,lower=0,upper=1):
        bounds = self.enclose(lower,upper)
        key = ('first_chart_derivative','second_chart_derivative')[self.chart_support]
        boxes = [getattr(b,key) for b in bounds]
        return tuple((min(box[k].lo for box in boxes),max(box[k].hi for box in boxes)) for k in range(2))

    def chart_derivative(self,parameters):
        values = np.asarray(parameters,dtype=float)
        with intersection_curve_operation():
            output = np.asarray([[(float(a)+float(b))/2 for a,b in self.chart_derivative_bounds(float(t),float(t))]
                for t in values.reshape(-1)],dtype=float)
            return output.reshape(values.shape+(2,))

    def chart_derivative_witness(self,parameters):
        """Approximate first-order chart values for advisory Gauss estimates."""
        values = np.asarray(parameters,dtype=float)
        with intersection_curve_operation():
            shape = self._shape()
            rows = shape.chart_derivative_witness_many(self.surface,tuple(float(t) for t in values.reshape(-1)))
            return np.asarray(rows,dtype=float).reshape(values.shape+(2,))

    def point_identity(self,parameter):
        parameter = Q(parameter)
        a,b = self.domain
        station = a+(b-a)*parameter
        if self.owner.closed and station == 1:
            station = Q(0)
        return ('nurbs_intersection_station',self.source_key,station)


def intersection_curve_record(curve):
    return {'type':'nurbs_surface_intersection','owner':encode_curve(curve.owner),
            'domain':[[str(x.numerator),str(x.denominator)] for x in curve.domain],
            'chart_support':curve.chart_support,'tolerances':[list(x) for x in curve.tolerance_values],
            'qualification':[list(x) for x in curve.qualification_values]}


def intersection_curve_from_record(record):
    if type(record) is not dict or set(record) != {'type','owner','domain','chart_support','tolerances','qualification'}:
        raise GeometryError('malformed NURBS surface intersection curve record')
    return NurbsSurfaceIntersectionCurve(decode_curve(record['owner']),
        tuple(Q(int(n),int(d)) for n,d in record['domain']),record['chart_support'],
        tuple(tuple(x) for x in record['tolerances']),tuple(tuple(x) for x in record['qualification']))
