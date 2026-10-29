"""Operation-local exact graph residual lower distance; no contact exclusions."""
from dataclasses import dataclass,replace
from fractions import Fraction as Q
from collections import Counter
from .errors import GeometryError
from .definition_binding import definition_checksum
from .nurbs import NurbsSurface
from . import _source_box_hull as native
from .nurbs_surface_intersection import _source,_structural,_bernstein
from .nurbs_surface_intersection._source import SourceDefinition
from .nurbs_surface_intersection._structural import derive_structural,ScalarField
from .nurbs_surface_intersection._records import StructuralCertificate
from .nurbs_surface_intersection._bernstein import bezier_piece,derivative_grid
from ._local_piecewise_surface_star import span_runtime_signature


def _pin(value):
    try:return definition_checksum(value)
    except (TypeError,ValueError,OverflowError) as error:
        raise GeometryError('graph clearance operand or runtime changed') from error


def _pieces(field,box,take):
    cuts=tuple(sorted({*pair,*(Q(k) for k in knots if pair[0]<Q(k)<pair[1])})
        for pair,knots in zip(box,(field.knots_u,field.knots_v)))
    net=field.exact_net()
    for a,b in zip(cuts[0],cuts[0][1:]):
        for c,d in zip(cuts[1],cuts[1][1:]):
            take()
            yield ((a,b),(c,d)),bezier_piece(field,net,(a,b),(c,d))


def _range(grid):
    values=tuple(p[2] for row in grid for p in row)
    return min(values),max(values)


@dataclass(frozen=True)
class _GraphPair:
    first:object
    certificate:object
    gradient_squared_upper:Q


def _derive(first,second,take):
    a,b=first.values,second.values
    if any(w is not None and (any(Q(x)<=0 for row in w for x in row) or len({Q(x) for row in w for x in row})!=1)
           for w in (a[5],b[5])):return None
    if tuple(tuple(sorted(map(Q,r))) for r in a[6:])!=tuple(tuple(sorted(map(Q,r))) for r in b[6:]):return None
    for values in (a,b):
        for degree,knots,domain in ((values[0],values[2],values[6]),(values[1],values[3],values[7])):
            lo,hi=sorted(map(Q,domain))
            if degree<1 or lo>=hi or any(n>degree for k,n in Counter(map(Q,knots)).items() if lo<k<hi):return None
        for row in values[4]:
            for point in row:take()
    cert,_=derive_structural(first,second,tick=take)
    if type(cert) is not StructuralCertificate or cert.domain is None:return None
    D=cert.direction;norm=sum(x*x for x in D);height=tuple(x/norm for x in D)
    r0,r1=cert.rows;(_,b0,c0),(_,b1,c1)=cert.affine;det=cert.determinant
    ur=tuple((c1*x-c0*y)/det for x,y in zip(r0,r1))
    vr=tuple((-b1*x+b0*y)/det for x,y in zip(r0,r1))
    heights=tuple(tuple(sum(Q(x)*h for x,h in zip(point,height)) for point in row) for row in b[4])
    field=ScalarField(second,replace(cert,field=heights));bounds=[]
    for box,grid in _pieces(field,cert.domain,take):
        ds=tuple(tuple(x/(box[k][1]-box[k][0]) for x in _range(derivative_grid(grid,k))) for k in (0,1))
        squared=Q(0)
        for h,u,v in zip(height,ur,vr):
            lo=h-max(u*ds[0][0],u*ds[0][1])-max(v*ds[1][0],v*ds[1][1])
            hi=h-min(u*ds[0][0],u*ds[0][1])-min(v*ds[1][0],v*ds[1][1])
            squared+=max(abs(lo),abs(hi))**2
        bounds.append(squared)
    if not bounds or max(bounds)<=0:return None
    return _GraphPair(first,cert,max(bounds))


class GraphClearance:
    """Detached arithmetic inside an owner operation with callback/final guards."""
    def __init__(self,supports,budget,slacks=()):
        self.budget=budget;self.expected=runtime_signature();self.sources={};self.pairs={};self.pins={}
        self.enabled=self.expected==_RUNTIME
        excluded=dict(slacks)
        if self.enabled:
            for face,support in supports.items():
                if (type(support) is NurbsSurface and not excluded.get(id(support),0)
                        and native._context(support) and native._canonical_literals(support)):
                    self.sources[face]=SourceDefinition.capture(support)
        self.source_pin=_pin(self.sources)
    def guard(self):
        if runtime_signature()!=self.expected or _pin(self.sources)!=self.source_pin:
            raise GeometryError('graph clearance context or detached source changed')
        if set(self.pairs)!=set(self.pins) or any(_pin(value)!=self.pins[key] for key,value in self.pairs.items()):
            raise GeometryError('graph clearance detached pair changed')
    def query(self,face,other,box):
        self.budget.take()
        if face not in self.sources or other not in self.sources:return None
        key=face,other
        if key not in self.pairs:
            value=_derive(self.sources[face],self.sources[other],self.budget.take)
            self.pairs[key]=value;self.pins[key]=_pin(value)
        pair=self.pairs[key]
        if pair is None:return None
        ranges=pair.first.values[6:]
        box=tuple(tuple(sorted(Q(a)+(Q(b)-Q(a))*Q(t) for t in part)) for (a,b),part in zip(ranges,box))
        if any(not lo<=a<b<=hi for (a,b),(lo,hi) in zip(box,pair.certificate.domain)):return None
        field=ScalarField(pair.first,pair.certificate)
        values=tuple(_range(grid) for _,grid in _pieces(field,box,self.budget.take))
        low=min(x[0] for x in values);high=max(x[1] for x in values)
        if low<=0<=high:return None
        lower=min(abs(low),abs(high))**2/pair.gradient_squared_upper
        return lower,((((Q(0),Q(1)),(Q(0),Q(1))),lower),)


def runtime_signature():
    functions=(runtime_signature,_pin,_pieces,_range,_derive,_GraphPair.__init__,GraphClearance.__init__,GraphClearance.guard,
        GraphClearance.query,SourceDefinition.capture,SourceDefinition.__init__,_source._freeze,
        derive_structural,_structural.derive_structural,_structural._greville,_structural._uniform,
        _structural._dot,_structural._cross,ScalarField.__init__,ScalarField.exact_net,
        StructuralCertificate.__init__,bezier_piece,derivative_grid,_bernstein.bezier_piece,
        _bernstein.derivative_grid,native._context,native._canonical_literals)
    return ((Q,NurbsSurface,SourceDefinition,ScalarField,StructuralCertificate,_GraphPair,GraphClearance,
        _GraphPair.__getattribute__,GraphClearance.__getattribute__,_source.NurbsSurface,
        _source.SourceDefinition,_source._FIELDS,_structural.StructuralCertificate,
        _structural.Fraction,_bernstein.Fraction,replace,Counter),span_runtime_signature(),
        tuple((f,getattr(f,'__code__',None),_pin(getattr(f,'__defaults__',None)),
               _pin(getattr(f,'__kwdefaults__',None))) for f in functions))


_RUNTIME=runtime_signature()
