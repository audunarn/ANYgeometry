"""UNWIRED NumPy directed-interval restriction experiment; root executes.

Production kernel/rounding code is untouched. No BLAS, fitting, fused operations
or sampled geometry supplies bounds. ``enclose(float_pair, box)`` is a test-only
drop-in for _filtered.Pair.enclose, with its existing four-entry LRU semantics.
"""
from fractions import Fraction as Q
import time
import json

import numpy as np
import pytest

from anygeometry.nurbs_surface_intersection import _filtered as scalar
from anygeometry.nurbs_surface_intersection import _bernstein as exact

_ORIGINAL_PAIR_ENCLOSE = scalar.Pair.enclose


def _finite(lo,hi):
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(lo>hi):
        raise scalar.Unknown("nonfinite or reversed array interval")
    return lo,hi


def arrays(grid):
    return _finite(np.array([[[x.lo for x in p] for p in row] for row in grid],dtype=np.float64),
                   np.array([[[x.hi for x in p] for p in row] for row in grid],dtype=np.float64))


def _multiply(lo,hi,b):
    """Array interval times scalar Bound, with EXACT scalar zero/one identities."""
    if b.lo==b.hi==0:
        return np.zeros_like(lo),np.zeros_like(hi)
    if b.lo==b.hi==1:
        return lo,hi
    with np.errstate(over="ignore",under="ignore",invalid="ignore"):
        ac=np.multiply(lo,b.lo)
        ad=np.multiply(lo,b.hi)
        bc=np.multiply(hi,b.lo)
        bd=np.multiply(hi,b.hi)
        lower=np.minimum(np.minimum(ac,ad),np.minimum(bc,bd))
        upper=np.maximum(np.maximum(ac,ad),np.maximum(bc,bd))
        lower=np.nextafter(lower,-np.inf)
        upper=np.nextafter(upper,np.inf)
        zero=(lo==0)&(hi==0)
        one=(lo==1)&(hi==1)
        lower=np.where(zero,0.,np.where(one,b.lo,lower))
        upper=np.where(zero,0.,np.where(one,b.hi,upper))
    return _finite(lower,upper)


def _add(lo,hi,other_lo,other_hi):
    with np.errstate(over="ignore",under="ignore",invalid="ignore"):
        lower=np.nextafter(np.add(lo,other_lo),-np.inf)
        upper=np.nextafter(np.add(hi,other_hi),np.inf)
        zero=(lo==0)&(hi==0)
        other_zero=(other_lo==0)&(other_hi==0)
        lower=np.where(zero,other_lo,np.where(other_zero,lo,lower))
        upper=np.where(zero,other_hi,np.where(other_zero,hi,upper))
    return _finite(lower,upper)


def _level(lo,hi,t,one_minus):
    # Match scalar order: (ONE-t)*left, then t*right, then their rounded sum.
    left_lo,left_hi=_multiply(lo[:-1],hi[:-1],one_minus)
    right_lo,right_hi=_multiply(lo[1:],hi[1:],t)
    return _add(left_lo,left_hi,right_lo,right_hi)


def _split(lo,hi,t):
    left_lo,left_hi=[lo[0]],[hi[0]]
    right_lo,right_hi=[lo[-1]],[hi[-1]]
    one_minus=scalar.ONE-t
    while len(lo)>1:
        lo,hi=_level(lo,hi,t,one_minus)
        left_lo.append(lo[0]);left_hi.append(hi[0])
        right_lo.append(lo[-1]);right_hi.append(hi[-1])
    return ((np.stack(left_lo),np.stack(left_hi)),
            (np.stack(right_lo[::-1]),np.stack(right_hi[::-1])))


def _reference_axis(lo,hi,a,b):
    n=len(lo)-1
    if a.lo==b.lo and a.hi==b.hi:
        one_minus=scalar.ONE-a
        for _ in range(n):
            lo,hi=_level(lo,hi,a,one_minus)
        return np.repeat(lo,n+1,axis=0),np.repeat(hi,n+1,axis=0)
    lower,upper=[],[]
    for k in range(n+1):
        work_lo,work_hi=lo,hi
        for tau in [a]*(n-k)+[b]*k:
            work_lo,work_hi=_level(work_lo,work_hi,tau,scalar.ONE-tau)
        lower.append(work_lo[0]);upper.append(work_hi[0])
    return np.stack(lower),np.stack(upper)


def _axis(lo,hi,a,b,axis):
    lo,hi=np.moveaxis(lo,axis,0),np.moveaxis(hi,axis,0)
    if len(lo)==1:
        result=lo,hi
    elif a.lo==b.lo and a.hi==b.hi:
        result=_reference_axis(lo,hi,a,b)
    elif not (0<=a.lo<=a.hi<b.lo<=b.hi<=1):
        result=_reference_axis(lo,hi,a,b)
    else:
        try:
            starts_zero=a.lo==a.hi==0
            right=(lo,hi) if starts_zero else _split(lo,hi,a)[1]
            if b.lo==b.hi==1:
                result=right
            else:
                tau=b if starts_zero else (b-a)*(scalar.ONE-a).reciprocal()
                result=_split(*right,tau)[0]
        except (scalar.Unknown,OverflowError,ZeroDivisionError):
            result=_reference_axis(lo,hi,a,b)
    return np.moveaxis(result[0],0,axis),np.moveaxis(result[1],0,axis)


def restrict(converted,u0,u1,v0,v1):
    """Whole 2D coefficient arrays; each interpolation operation rounds out."""
    u0,u1,v0,v1=(scalar.Bound.exact(x) for x in (u0,u1,v0,v1))
    lo,hi=_axis(*converted,u0,u1,0)
    return _axis(lo,hi,v0,v1,1)


def restrict_ranges(converted,*box):
    lo,hi=restrict(converted,*box)
    low,high=np.min(lo,axis=(0,1)),np.max(hi,axis=(0,1))
    return tuple(scalar.Bound(float(a),float(b)) for a,b in zip(low,high))


def _multiply_many(lo,hi,blo,bhi):
    """Broadcast interval products; retain scalar zero/one precedence per item."""
    with np.errstate(over="ignore",under="ignore",invalid="ignore"):
        ac=np.multiply(lo,blo)
        ad=np.multiply(lo,bhi)
        bc=np.multiply(hi,blo)
        bd=np.multiply(hi,bhi)
        lower=np.nextafter(np.minimum(np.minimum(ac,ad),np.minimum(bc,bd)),-np.inf)
        upper=np.nextafter(np.maximum(np.maximum(ac,ad),np.maximum(bc,bd)),np.inf)
        zero=((lo==0)&(hi==0))|((blo==0)&(bhi==0))
        one=(lo==1)&(hi==1)
        other_one=(blo==1)&(bhi==1)
        lower=np.where(zero,0.,np.where(one,blo,np.where(other_one,lo,lower)))
        upper=np.where(zero,0.,np.where(one,bhi,np.where(other_one,hi,upper)))
    return _finite(lower,upper)


def _parameters(values):
    # The coefficient axis is first, the batch axis second; all remaining
    # (other parameter, homogeneous coordinate) operations broadcast together.
    shape=(1,len(values),1,1)
    return (np.array([x.lo for x in values],dtype=np.float64).reshape(shape),
            np.array([x.hi for x in values],dtype=np.float64).reshape(shape))


def _split_many(lo,hi,values):
    tlo,thi=_parameters(values)
    olo,ohi=_parameters([scalar.ONE-t for t in values])
    left_lo,left_hi=[lo[0]],[hi[0]]
    right_lo,right_hi=[lo[-1]],[hi[-1]]
    while len(lo)>1:
        a,b=_multiply_many(lo[:-1],hi[:-1],olo,ohi)
        c,d=_multiply_many(lo[1:],hi[1:],tlo,thi)
        lo,hi=_add(a,b,c,d)
        left_lo.append(lo[0]);left_hi.append(hi[0])
        right_lo.append(lo[-1]);right_hi.append(hi[-1])
    return ((np.stack(left_lo),np.stack(left_hi)),
            (np.stack(right_lo[::-1]),np.stack(right_hi[::-1])))


def _interior_axis_many(lo,hi,starts,ends,axis):
    lo,hi=np.moveaxis(lo,axis,0),np.moveaxis(hi,axis,0)
    if len(lo)>1:
        right=_split_many(lo,hi,starts)[1]
        ratios=[(b-a)*(scalar.ONE-a).reciprocal() for a,b in zip(starts,ends)]
        lo,hi=_split_many(*right,ratios)[0]
    return np.moveaxis(lo,0,axis),np.moveaxis(hi,0,axis)


def restrict_many_coefficients(converted,boxes,*,_grids=None):
    """One cached grid, independent parameter boxes; unsupported items fall back.

    The bulk path admits ONLY strictly separated outward endpoint intervals
    strictly within (0,1). Point/boundary/reversal/uncertain order/extrapolation
    use the previously tested per-box path. Unknown/nonfinite cannot issue bounds.
    The ordinary entry shares one grid; the stacked entry supplies different
    cached grids with the same shape. Neither entry queues queries.
    """
    _finite(*converted)
    boxes=tuple(tuple(box) for box in boxes)
    if _grids is not None:
        _grids=tuple(_grids)
        if len(_grids)!=len(boxes) or any(g[0].shape!=converted[0].shape or g[1].shape!=converted[1].shape for g in _grids):
            raise ValueError("stacked grids must match box count and coefficient shape")
        for grid in _grids:
            _finite(*grid)
    lower,upper=[None]*len(boxes),[None]*len(boxes)
    selected,parameters=[],[]
    for i,box in enumerate(boxes):
        if len(box)!=4:
            raise ValueError("expected (u0,u1,v0,v1) per restriction")
        values=tuple(scalar.Bound.exact(x) for x in box)
        interior=all(0<a.lo<=a.hi<b.lo<=b.hi<1 for a,b in (values[:2],values[2:]))
        if interior:
            selected.append(i);parameters.append(values)
        else:
            lower[i],upper[i]=restrict(converted if _grids is None else _grids[i],*box)
    if selected:
        shape=(len(selected),)+converted[0].shape
        if _grids is None:
            lo,hi=np.broadcast_to(converted[0],shape),np.broadcast_to(converted[1],shape)
        else:
            lo=np.stack([_grids[i][0] for i in selected])
            hi=np.stack([_grids[i][1] for i in selected])
        try:
            lo,hi=_interior_axis_many(lo,hi,[v[0] for v in parameters],[v[1] for v in parameters],1)
            lo,hi=_interior_axis_many(lo,hi,[v[2] for v in parameters],[v[3] for v in parameters],2)
        except (scalar.Unknown,OverflowError,ZeroDivisionError):
            # Same conservative reference opportunity as the single-box path.
            # A fallback that also fails still propagates Unknown, never a hull.
            for index in selected:
                lower[index],upper[index]=restrict(converted if _grids is None else _grids[index],*boxes[index])
        else:
            for local,index in enumerate(selected):
                lower[index],upper[index]=lo[local],hi[local]
    if not boxes:
        shape=(0,)+converted[0].shape
        return np.empty(shape),np.empty(shape)
    return np.stack(lower),np.stack(upper)


def restrict_ranges_many(converted,boxes):
    """Ordered four-Bound ranges for independent restrictions of ONE cached grid."""
    boxes=tuple(boxes)
    if not boxes:
        return ()
    lo,hi=restrict_many_coefficients(converted,boxes)
    low,high=np.min(lo,axis=(1,2)),np.max(hi,axis=(1,2))
    return tuple(tuple(scalar.Bound(float(a),float(b)) for a,b in zip(l,h)) for l,h in zip(low,high))


def restrict_stacked_coefficients(converted_grids,boxes):
    """Independent cached grids of equal degree shape, each with its own box."""
    converted_grids,boxes=tuple(converted_grids),tuple(boxes)
    if not converted_grids:
        if boxes:
            raise ValueError("missing stacked grids")
        return np.empty((0,)),np.empty((0,))
    return restrict_many_coefficients(converted_grids[0],boxes,_grids=converted_grids)


def restrict_ranges_stacked(converted_grids,boxes):
    converted_grids,boxes=tuple(converted_grids),tuple(boxes)
    if not boxes and not converted_grids:
        return ()
    lo,hi=restrict_stacked_coefficients(converted_grids,boxes)
    low,high=np.min(lo,axis=(1,2)),np.max(hi,axis=(1,2))
    return tuple(tuple(scalar.Bound(float(a),float(b)) for a,b in zip(l,h)) for l,h in zip(low,high))


def _interior_box(box):
    values=tuple(scalar.Bound.exact(x) for x in box)
    return all(0<a.lo<=a.hi<b.lo<=b.hi<1 for a,b in (values[:2],values[2:]))


def paired_enclose(self,box):
    """Test-only replacement, grouping different A/B factors by coefficient shape.

    Point, boundary, unsafe and unpaired shapes use ORIGINAL scalar restriction;
    the captured original method prevents recursion when monkeypatched. Arrays
    belong only to this prepared object's fixed source/origin/C grid identity.
    """
    key=tuple(tuple(axis) for axis in box)
    cache=self.enclosure_cache
    if key in cache:
        result=cache.pop(key)
        cache[key]=result
        self.cache_hits+=1
        return result
    (u0,u1),(v0,v1),(s0,s1),(t0,t1)=box
    domains=((u0,u1,v0,v1),(s0,s1,t0,t1))
    eligible=tuple(_interior_box(domain) for domain in domains)
    groups={}
    for i,grid in enumerate(self.grids):
        shape=(len(grid),len(grid[0]),len(grid[0][0]))
        if eligible[0 if i in (0,2,3) else 1]:
            groups.setdefault(shape,[]).append(i)
    groups=[indices for indices in groups.values() if len(indices)>1]
    if not groups:
        self._paired_prototype_fallbacks=getattr(self,"_paired_prototype_fallbacks",0)+1
        return _ORIGINAL_PAIR_ENCLOSE(self,box)
    self.cache_misses+=1
    state=getattr(self,"_paired_prototype_arrays",None)
    if state is None or state[0] is not self.grids:
        state=(self.grids,tuple(arrays(grid) for grid in self.grids))
        self._paired_prototype_arrays=state
    converted=state[1]
    bounds=[None]*6
    for indices in groups:
        domains_for_group=[domains[0 if i in (0,2,3) else 1] for i in indices]
        results=restrict_ranges_stacked([converted[i] for i in indices],domains_for_group)
        for i,result in zip(indices,results):
            bounds[i]=result
        self._paired_prototype_groups=getattr(self,"_paired_prototype_groups",0)+1
        self._paired_prototype_items=getattr(self,"_paired_prototype_items",0)+len(indices)
    for i in range(6):
        if bounds[i] is None:
            domain=domains[0 if i in (0,2,3) else 1]
            bounds[i]=scalar.ranges(scalar.subbox(self.grids[i],*domain))
    result=bounds[0],bounds[1],(bounds[2],bounds[3]),(bounds[4],bounds[5])
    if len(cache)>=4:
        cache.pop(next(iter(cache)))
    cache[key]=result
    return result


def enclose(prepared_float_pair,box):
    """Unwired replacement preserving exact four-entry original LRU protocol."""
    key=tuple(tuple(axis) for axis in box)
    cache=prepared_float_pair.enclosure_cache
    if key in cache:
        result=cache.pop(key)
        cache[key]=result
        prepared_float_pair.cache_hits+=1
        return result
    prepared_float_pair.cache_misses+=1
    converted=getattr(prepared_float_pair,"_array_prototype_grids",None)
    if converted is None:
        converted=tuple(arrays(g) for g in prepared_float_pair.grids)
        prepared_float_pair._array_prototype_grids=converted
    (u0,u1),(v0,v1),(s0,s1),(t0,t1)=box
    bounds=[restrict_ranges(g,*((u0,u1,v0,v1) if i in (0,2,3) else (s0,s1,t0,t1)))
            for i,g in enumerate(converted)]
    result=bounds[0],bounds[1],(bounds[2],bounds[3]),(bounds[4],bounds[5])
    if len(cache)>=4:
        cache.pop(next(iter(cache)))
    cache[key]=result
    return result


def _grid(degree):
    return tuple(tuple(tuple(Q((-1)**(i+j+k)*(i+2)*(j+1)*(k+1),i+j+k+2)
                             for k in range(4)) for j in range(degree+1)) for i in range(degree+1))


def _bits_pair(lo,hi):
    return tuple((float(a).hex(),float(b).hex()) for a,b in zip(lo.flat,hi.flat))


@pytest.mark.parametrize("degree",(1,2,4))
def test_bulk_coefficients_bit_identical_to_scalar_and_enclose_exact(degree):
    grid=_grid(degree)
    rounded=scalar.convert(grid)
    converted=arrays(rounded)
    domains=((Q(0),Q(1),Q(0),Q(1)),(Q(1,5),Q(4,5),Q(1,7),Q(6,7)),
             (Q(1,3),Q(1,3),Q(1,4),Q(3,4)),(Q(0),Q(0),Q(1),Q(1)),
             (Q(3,4),Q(1,4),Q(-1,4),Q(5,4)),
             (Q(1)-Q(1,2**100),Q(1),Q(1,3),Q(1,3)+Q(1,2**100)))
    for domain in domains:
        lo,hi=restrict(converted,*domain)
        old=scalar.subbox(rounded,*domain)
        old_lo,old_hi=arrays(old)
        assert _bits_pair(lo,hi)==_bits_pair(old_lo,old_hi)
        reference=exact.subbox(grid,*domain)
        for i,row in enumerate(reference):
            for j,point in enumerate(row):
                for k,value in enumerate(point):
                    assert Q(float(lo[i,j,k]))<=value<=Q(float(hi[i,j,k]))
        bounds=restrict_ranges(converted,*domain)
        original=scalar.ranges(old)
        assert [(x.lo.hex(),x.hi.hex()) for x in bounds]==[(x.lo.hex(),x.hi.hex()) for x in original]


def test_bulk_primitive_zero_one_extreme_signs_and_nonfinite_refusal():
    values=[(0.,0.),(1.,1.),(-1.,-1.),(-2.,3.),(5e-324,1e-300),(-1e200,-1e100)]
    lo=np.array([x[0] for x in values]);hi=np.array([x[1] for x in values])
    for b in (scalar.Bound(0.),scalar.Bound(1.),scalar.Bound(-2.,3.),scalar.Bound(5e-324),scalar.Bound(-1.,0.)):
        actual=_multiply(lo,hi,b)
        reference=[scalar.Bound(a,c)*b for a,c in values]
        assert _bits_pair(*actual)==tuple((x.lo.hex(),x.hi.hex()) for x in reference)
    with pytest.raises(scalar.Unknown):
        _multiply(np.array([1e308]),np.array([1e308]),scalar.Bound(1e308))


def test_prototype_pair_enclose_lru_and_scalar_reference():
    from types import SimpleNamespace
    rounded=scalar.convert(_grid(4))
    pair=SimpleNamespace(grids=(rounded,)*6,enclosure_cache={},cache_hits=0,cache_misses=0)
    boxes=[((Q(i,16),Q(i+4,16)),(Q(1,4),Q(3,4)))*2 for i in range(6)]
    first=enclose(pair,boxes[0])
    assert enclose(pair,boxes[0]) is first and pair.cache_hits==1
    for box in boxes[1:]:
        result=enclose(pair,box)
        expected=scalar.ranges(scalar.subbox(rounded,*box[0],*box[1]))
        assert [(x.lo,x.hi) for x in result[0]]==[(x.lo,x.hi) for x in expected]
    assert len(pair.enclosure_cache)==4 and tuple(boxes[0]) not in pair.enclosure_cache


@pytest.mark.parametrize("degree",(1,2,4))
def test_many_boxes_coefficients_ranges_mixed_fallback_and_input_immutability(degree):
    grid=_grid(degree)
    rounded=scalar.convert(grid)
    converted=arrays(rounded)
    before=converted[0].tobytes(),converted[1].tobytes()
    boxes=tuple((Q(i,16),Q(i+4,16),Q(i+1,20),Q(i+9,20)) for i in (1,3,7)) + (
        (Q(1,3),Q(1,3),Q(1,4),Q(3,4)),
        (Q(0),Q(1),Q(0),Q(1)),
        (Q(3,4),Q(1,4),Q(-1,4),Q(5,4)),
        (Q(1)-Q(1,2**100),Q(1),Q(1,3),Q(1,3)+Q(1,2**100)))
    lo,hi=restrict_many_coefficients(converted,boxes)
    ranges=restrict_ranges_many(converted,boxes)
    for batch,box in enumerate(boxes):
        old=scalar.subbox(rounded,*box)
        assert _bits_pair(lo[batch],hi[batch])==_bits_pair(*arrays(old))
        expected=exact.subbox(grid,*box)
        for i,row in enumerate(expected):
            for j,point in enumerate(row):
                for k,value in enumerate(point):
                    assert Q(float(lo[batch,i,j,k]))<=value<=Q(float(hi[batch,i,j,k]))
        assert [(x.lo.hex(),x.hi.hex()) for x in ranges[batch]] == [
            (x.lo.hex(),x.hi.hex()) for x in scalar.ranges(old)]
    assert before==(converted[0].tobytes(),converted[1].tobytes())
    assert restrict_ranges_many(converted,())==()


def test_many_nonfinite_and_failed_extrapolation_remain_unknown():
    converted=arrays(scalar.convert(_grid(1)))
    lo,hi=converted[0].copy(),converted[1].copy()
    lo[0,0,0]=np.inf
    hi[0,0,0]=np.inf
    with pytest.raises(scalar.Unknown):
        restrict_ranges_many((lo,hi),((Q(1,4),Q(3,4),Q(1,4),Q(3,4)),))
    huge=np.full((2,2,4),1e308)
    box=(Q(-100),Q(100),Q(-100),Q(100))
    with pytest.raises(scalar.Unknown):
        restrict((huge,huge),*box)
    with pytest.raises(scalar.Unknown):
        restrict_ranges_many((huge,huge),(box,))


def test_stacked_different_grids_match_scalar_fraction_and_preserve_inputs():
    first=_grid(4)
    second=tuple(tuple(tuple(2*x+Q((-1)**k,3) for k,x in enumerate(p)) for p in row) for row in first)
    grids=(first,second,first)
    converted=tuple(arrays(scalar.convert(grid)) for grid in grids)
    before=tuple((lo.tobytes(),hi.tobytes()) for lo,hi in converted)
    boxes=((Q(1,7),Q(5,7),Q(1,6),Q(5,6)),
           (Q(1,5),Q(3,5),Q(2,7),Q(6,7)),
           (Q(1,3),Q(1,3),Q(0),Q(1)))
    lo,hi=restrict_stacked_coefficients(converted,boxes)
    bounds=restrict_ranges_stacked(converted,boxes)
    for batch,(grid,box) in enumerate(zip(grids,boxes)):
        old=scalar.subbox(scalar.convert(grid),*box)
        assert _bits_pair(lo[batch],hi[batch])==_bits_pair(*arrays(old))
        reference=exact.subbox(grid,*box)
        for i,row in enumerate(reference):
            for j,p in enumerate(row):
                for k,value in enumerate(p):
                    assert Q(float(lo[batch,i,j,k]))<=value<=Q(float(hi[batch,i,j,k]))
        assert [(x.lo.hex(),x.hi.hex()) for x in bounds[batch]]==[
            (x.lo.hex(),x.hi.hex()) for x in scalar.ranges(old)]
    assert before==tuple((lo.tobytes(),hi.tobytes()) for lo,hi in converted)


def _actual_prepared_pair():
    from test_chart_fallback import nested_rational_pair,CELL,CLIPS
    from anygeometry.nurbs_surface_intersection._certificate import _Pair
    a,b=nested_rational_pair()
    return scalar.Pair(_Pair(a,b,CLIPS).anchored(CELL))


def _enclosure_bits(value):
    return tuple(tuple((x.lo.hex(),x.hi.hex()) for x in row)
                 for row in (value[0],value[1],*value[2],*value[3]))


def test_actual_paired_enclose_derivative_shapes_lru_fallback_and_immutability():
    reference,candidate=_actual_prepared_pair(),_actual_prepared_pair()
    before=tuple(_bits_pair(*arrays(grid)) for grid in candidate.grids)
    assert {tuple(arrays(g)[0].shape) for g in candidate.grids}=={(5,5,4),(4,5,4),(5,4,4)}
    boxes=[((Q(i,16),Q(i+3,16)),(Q(1,5),Q(4,5)),
            (Q(i+1,16),Q(i+4,16)),(Q(2,7),Q(5,7))) for i in range(1,7)]
    boxes += [((Q(1,3),Q(1,3)),)*4,((Q(0),Q(1)),)*4]
    for box in boxes:
        expected=_ORIGINAL_PAIR_ENCLOSE(reference,box)
        actual=paired_enclose(candidate,box)
        assert _enclosure_bits(actual)==_enclosure_bits(expected)
        assert paired_enclose(candidate,box) is actual
        assert _ORIGINAL_PAIR_ENCLOSE(reference,box) is expected
    assert candidate.cache_hits==reference.cache_hits
    assert candidate.cache_misses==reference.cache_misses
    assert len(candidate.enclosure_cache)==len(reference.enclosure_cache)==4
    assert candidate._paired_prototype_groups==18 and candidate._paired_prototype_items==36
    assert candidate._paired_prototype_fallbacks==2
    assert before==tuple(_bits_pair(*arrays(grid)) for grid in candidate.grids)


def test_paired_enclose_single_shape_fallback_and_unknown_do_not_cache_partial(monkeypatch):
    from types import SimpleNamespace
    grids=tuple(scalar.convert(_grid(i)) for i in range(1,7))
    candidate=SimpleNamespace(grids=grids,enclosure_cache={},cache_hits=0,cache_misses=0)
    box=((Q(1,4),Q(3,4)),)*4
    actual=paired_enclose(candidate,box)
    assert len(actual)==4 and candidate._paired_prototype_fallbacks==1
    candidate=_actual_prepared_pair()
    def failed(*args,**kwargs):
        raise scalar.Unknown("prototype nonfinite bulk work")
    monkeypatch.setitem(globals(),"restrict_ranges_stacked",failed)
    with pytest.raises(scalar.Unknown,match="nonfinite"):
        paired_enclose(candidate,box)
    assert candidate.cache_misses==1 and not candidate.enclosure_cache


def test_paired_backend_fixed64_public_generic_records_and_counts_identical(monkeypatch,capsys):
    """Bounded actual-caller A/B; not a full nested-loop acceptance query."""
    from dataclasses import fields
    from importlib import import_module
    from types import SimpleNamespace
    from test_chart_fallback import nested_rational_pair
    from anygeometry.tolerance import TolerancePolicy
    from anygeometry.nurbs_surface_intersection import query_nurbs_surface_intersection,IntersectionStatus
    from anygeometry.nurbs_surface_intersection._source import SourceDefinition
    query=import_module("anygeometry.nurbs_surface_intersection.query")
    first,second=nested_rational_pair()
    before=SourceDefinition.capture(first),SourceDefinition.capture(second)
    reports,records,counts=[],[],[]
    for bulk in (False,True):
        metrics=dict(calls=0,groups=0,items=0,fallbacks=0)
        def backend(self,box):
            metrics["calls"]+=1
            old=tuple(getattr(self,name,0) for name in
                ("_paired_prototype_groups","_paired_prototype_items","_paired_prototype_fallbacks"))
            result=paired_enclose(self,box)
            for key,name,previous in zip(("groups","items","fallbacks"),
                    ("_paired_prototype_groups","_paired_prototype_items","_paired_prototype_fallbacks"),old):
                metrics[key]+=getattr(self,name,0)-previous
            return result
        with monkeypatch.context() as patch:
            patch.setattr(query,"derive_structural",lambda *a:(None,"explicit prototype generic A/B"))
            patch.setattr(scalar.Pair,"enclose",backend if bulk else _ORIGINAL_PAIR_ENCLOSE)
            started=time.perf_counter()
            result=query_nurbs_surface_intersection(first,second,tolerances=TolerancePolicy(),
                qualification=SimpleNamespace(max_boxes_per_pair=64,max_subdivision_depth=40))
            elapsed=time.perf_counter()-started
        assert result.status is IntersectionStatus.UNCLASSIFIED
        assert result.stats.boxes_visited==64 and result.atlas.frontier_remaining>0
        assert before[0].matches(first) and before[1].matches(second)
        records.append((result.status,result.algorithm,result.structural,result.roots,result.boxes,
                        result.components,result.atlas,result.descriptor_digest))
        counts.append(tuple((field.name,getattr(result.stats,field.name)) for field in fields(result.stats)
                            if not field.name.endswith("seconds")))
        reports.append(dict(bulk=bulk,seconds=elapsed,metrics=metrics,
                            certificate_attempts=result.stats.certificate_attempts,
                            certificates=result.stats.certificates,escalations=result.stats.exact_escalations))
    assert records[0]==records[1] and counts[0]==counts[1]
    with capsys.disabled():
        print(json.dumps(reports,sort_keys=True))


def test_bulk_degree4_fixed_restriction_throughput(capsys):
    """Timing evidence only; scalar equality is mandatory, no speed threshold."""
    rounded=scalar.convert(_grid(4))
    converted=arrays(rounded)
    # 32 identical work items in EVERY route, different per-box endpoint bounds.
    # Interior-only here isolates the supported batched path; correctness tests
    # above separately require mixed fallback and every coefficient reference.
    domains=[(Q(i+1,32),Q(i+13,32),Q(i+1,40),Q(i+21,40)) for i in range(8)]*4
    results=[]
    payloads=[]
    for route in ("scalar","per_cell_array",2,4,32):
        started=time.perf_counter()
        if type(route) is int:
            output=[]
            for offset in range(0,len(domains),route):
                output.extend(restrict_ranges_many(converted,domains[offset:offset+route]))
        else:
            output=[]
            for domain in domains:
                result=(restrict_ranges(converted,*domain) if route=="per_cell_array"
                        else scalar.ranges(scalar.subbox(rounded,*domain)))
                output.append(result)
        elapsed=time.perf_counter()-started
        bits=tuple(tuple((x.lo.hex(),x.hi.hex()) for x in result) for result in output)
        payloads.append(bits)
        results.append(dict(route=str(route),seconds=elapsed,restrictions=len(output),
                            common_cached_grid=True))
    assert all(values==payloads[0] for values in payloads[1:])
    with capsys.disabled():
        print(json.dumps(results,sort_keys=True))
