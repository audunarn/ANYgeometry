"""Independent rectangular areas and rational clipping check completeness."""
from fractions import Fraction as F
from itertools import combinations
from dataclasses import replace

import numpy as np
import pytest

from anygeometry import (GeometryError,Plane,to_dict,
                        validate_prepared_authored_face_partition as validate)
from anygeometry.arrangement_geometry import LinePath
from anygeometry.authored_partition_coverage import (_positive_overlap,_overlap_candidates,_simple_area)
from anygeometry.material_arrangement import MaterialDomain,ArrangementPath
from anygeometry.material_cell_coverage import _frame
from test_authored_domain_coverage import planar,cubic
from test_authored_child_coverage import children


def rectangle(x0,y0,x1,y1):
    return np.asarray([[(x0,y0),(x1,y0),(x1,y1)],[(x0,y0),(x1,y1),(x0,y1)]],float)


def partition():
    model,binding=planar()
    left,right=children(model,binding)
    # Independent fixture contract: x=3 splits a 4x4 source into 12+4.
    return model,binding,{left:rectangle(0,0,3,4),right:rectangle(3,0,4,4)}


def test_complete_partition_reordering_repeat_and_source_nonmutation():
    model,binding,rows=partition();before=to_dict(model)
    copies={key:value.copy() for key,value in rows.items()}
    assert validate(model,binding,rows) is None
    validate(model,binding,{key:value[::-1,::-1] for key,value in reversed(tuple(rows.items()))})
    validate(model,binding,rows)
    assert to_dict(model)==before
    assert all(np.array_equal(value,copies[key]) for key,value in rows.items())


def test_complete_concave_straight_domain():
    model,binding=planar(concave=True,fragment=False)
    # Independent L domain has area3+2; two nonoverlapping rectangles tile it.
    rows=np.concatenate((rectangle(0,0,3,1),rectangle(0,1,1,3)))
    validate(model,binding,{binding.descendants[0]:rows})


@pytest.mark.parametrize('failure',['missing','empty','partial','duplicate','crossing','wrong'])
def test_containment_or_area_alone_cannot_accept_a_partial_or_overlapping_mesh(failure):
    model,binding,rows=partition()
    left,right=children(model,binding)
    if failure=='missing':del rows[right]
    if failure=='empty':rows[right]=np.empty((0,3,2))
    if failure=='partial':rows[left]=rows[left][:1]
    if failure=='duplicate':rows[left]=np.repeat(rows[left][:1],2,axis=0)
    if failure=='crossing':rows[left]=rectangle(0,0,3.5,4)
    if failure=='wrong':rows[left],rows[right]=rows[right],rows[left]
    before=to_dict(model)
    with pytest.raises(GeometryError):validate(model,binding,rows)
    assert to_dict(model)==before


@pytest.mark.parametrize('source',['holes','cubic'])
def test_incomplete_or_unqualified_material_does_not_gain_area_only_acceptance(source):
    model,binding=planar(holes=True,fragment=False) if source=='holes' else cubic(cropped=True)
    with pytest.raises(GeometryError):
        validate(model,binding,{key:rectangle(0,0,1,1) for key in binding.descendants})


@pytest.mark.parametrize('bad',[True,'3',999])
def test_bad_mapping_keys_refuse(bad):
    model,binding,rows=partition()
    rows[bad]=rectangle(0,0,1,1)
    with pytest.raises(GeometryError):validate(model,binding,rows)


def test_stateful_key_conversion_cannot_silently_replace_supplied_cells():
    from collections.abc import Mapping
    model,binding,rows=partition()
    left,right=children(model,binding)
    class Stateful(int):
        def __int__(self):
            result=self.values.pop(0)
            return result
    key=Stateful(left)
    key.values=[right,left]
    class Input(Mapping):
        def __len__(self):return 3
        def __iter__(self):return iter((left,key,right))
        def __getitem__(self,key):raise AssertionError('items are supplied directly')
        def items(self):return ((left,rows[left][:1]),(key,rows[left]),(right,rows[right]))
    # The first incomplete child batch must not disappear during key coercion.
    with pytest.raises(GeometryError):validate(model,binding,Input())


def test_cancellation_and_stale_mutation_never_accept_partial_proof():
    model,binding,rows=partition();before=to_dict(model)
    with pytest.raises(GeometryError,match='cancelled'):
        validate(model,binding,rows,cancellation_check=lambda _:True)
    assert to_dict(model)==before
    child=next(iter(rows))
    def mutate(_):
        model._faces[child]=replace(model.faces[child],metadata={'changed':True})
        return False
    with pytest.raises(GeometryError):validate(model,binding,rows,cancellation_check=mutate)


def test_arrays_are_detached_before_callbacks():
    model,binding,rows=partition();before=to_dict(model)
    def mutate_inputs(_):
        for value in rows.values():value[:]=99
        return False
    validate(model,binding,rows,cancellation_check=mutate_inputs)
    assert to_dict(model)==before


def test_transient_loop_swap_cannot_corrupt_completeness_snapshot(monkeypatch):
    import anygeometry.authored_partition_coverage as owner
    model,binding,rows=partition();before=to_dict(model)
    left,right=children(model,binding)
    original=model.faces[left]
    real=owner._original_domain
    stage={'reconstruct':False,'changed':False}
    def reconstruct(definition,check):
        stage['reconstruct']=True
        result=real(definition,check)
        stage['reconstruct']=False
        return result
    def mutate(_):
        if stage['reconstruct']:
            stage['changed']=True
            model._faces[left]=replace(original,loop=model.faces[right].loop)
        else:model._faces[left]=original
        return False
    monkeypatch.setattr(owner,'_original_domain',reconstruct)
    try:
        validate(model,binding,rows,cancellation_check=mutate)
        assert stage['changed'] and to_dict(model)==before
    finally:model._faces[left]=original


@pytest.mark.parametrize('points',[
    [(0,0),(2,2),(0,2),(2,0)],
    [(0,0),(2,0),(1,0),(2,2),(0,2)],
    [(0,0),(2,0),(2,2),(0,0),(0,2)],
    [(0,0),(2,0),(2,2),(1,0),(0,2)],
])
def test_exact_regular_boundary_proof_refuses_invalid_loops(points):
    support=Plane((0,0,0),(1,0,0),(0,1,0))
    positions=[(*point,0) for point in points]
    loop=tuple(ArrangementPath(LinePath(a,b),index) for index,(a,b) in
               enumerate(zip(positions,(*positions[1:],positions[0]))))
    with pytest.raises(GeometryError):
        _simple_area(MaterialDomain(1,support,(loop,)),_frame(support),lambda:None)


def clipped_area(first,second):
    """Independent rational Sutherland-Hodgman intersection-area oracle."""
    polygon=list(first)
    for a,b in zip(second,(*second[1:],second[0])):
        def side(p):return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
        output=[]
        for p,q in zip(polygon,(*polygon[1:],polygon[0])) if polygon else ():
            sp,sq=side(p),side(q)
            if sp>=0:output.append(p)
            if (sp<0 and sq>0) or (sp>0 and sq<0):
                t=sp/(sp-sq)
                output.append(tuple(x+t*(y-x) for x,y in zip(p,q)))
        polygon=output
    if not polygon:return F(0)
    return abs(sum(p[0]*q[1]-p[1]*q[0] for p,q in zip(polygon,(*polygon[1:],polygon[0]))))/2


def test_sweep_and_separating_axes_match_independent_exact_clipping():
    base=((F(0),F(0)),(F(1),F(0)),(F(0),F(1)))
    triangles=[tuple((x+dx,y+dy) for x,y in base) for dx in map(F,(-1,0,.25,1))
               for dy in map(F,(-1,0,.25,1))]
    # Also include the complementary half: diagonal contact has zero area.
    triangles.append(((F(1),F(0)),(F(1),F(1)),(F(0),F(1))))
    candidates={tuple(sorted(pair)) for pair in _overlap_candidates(triangles,lambda:None)}
    for i,j in combinations(range(len(triangles)),2):
        expected=clipped_area(triangles[i],triangles[j])>0
        assert _positive_overlap(triangles[i],triangles[j])==expected
        if expected:assert (i,j) in candidates


def test_cancellation_during_pair_enumeration_returns_no_pairs():
    calls=[]
    def cancel():
        calls.append(1)
        if len(calls)==3:raise GeometryError('original work exhausted')
    with pytest.raises(GeometryError,match='original work exhausted'):
        list(_overlap_candidates([((F(0),F(0)),(F(1),F(0)),(F(0),F(1)))]*2,cancel))
