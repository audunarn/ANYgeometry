"""Exact nonbinary planar UV must not disappear into binary64 rounding."""
from fractions import Fraction as F

import numpy as np
import pytest

from anygeometry import GeometryError, to_dict
from anygeometry.authored_partition_coverage import validate_prepared_authored_face_partition as validate
from anygeometry.authored_child_coverage import validate_prepared_authored_face_child_triangles
from anygeometry.authored_domain_coverage import validate_prepared_authored_face_triangles
from anygeometry.material_cell_coverage import _triangle_rows, _coordinate_fraction
from examples.authored_material_stations_handoff import build, partition, verify


def test_both_roots_all_eight_children_exact_rational_partition():
    model,bindings,_,_=build();before=to_dict(model)
    assert tuple(len(binding.descendants) for binding in bindings)==(2,6)
    for binding in bindings:
        rows=partition(model,binding)
        for child,value in rows.items():
            validate_prepared_authored_face_triangles(model,binding,value)
            validate_prepared_authored_face_child_triangles(model,binding,child,value)
        validate(model,binding,rows)
        validate(model,binding,{key:tuple(triangle[::-1] for triangle in value[::-1])
                               for key,value in reversed(tuple(rows.items()))})
    assert to_dict(model)==before
    assert len(verify()['current_faces'])==8


def test_float_only_root2_does_not_gain_exact_area_acceptance():
    model,bindings,_,_=build();binding=bindings[1]
    rows={key:np.array(value,float) for key,value in partition(model,binding).items()}
    with pytest.raises(GeometryError):validate(model,binding,rows)


@pytest.mark.parametrize('sign',(-1,1))
def test_nondyadic_gap_or_overlap_that_float_rounds_away_refuses(sign):
    model,bindings,_,_=build();binding=bindings[0]
    rows=partition(model,binding)
    child=next(key for key,value in rows.items() if min(p[0] for t in value for p in t)==0)
    epsilon=F(1,10**80)
    changed=tuple(tuple((p[0]+sign*epsilon if p[0]==3 else p[0],p[1]) for p in triangle)
                  for triangle in rows[child])
    assert np.array_equal(np.array(changed,float),np.array(rows[child],float))
    rows[child]=changed
    with pytest.raises(GeometryError):validate(model,binding,rows)


def test_fraction_detachment_and_existing_binary64_conversion_semantics():
    input=np.array([[[F(1,3),0],[1,0],[1,1]]],dtype=object)
    copied=_triangle_rows(input)
    input[0,0,0]=F(1,5)
    assert copied[0,0,0]==F(1,3)
    assert _coordinate_fraction(.1)==F(.1)
    for dtype in (np.float16,np.float32,np.float64,np.int64):
        numeric=np.array([[[0,0],[1,0],[1,1]]],dtype=dtype)
        assert _triangle_rows(numeric).tobytes()==np.array(numeric,dtype=float).tobytes()


@pytest.mark.parametrize('bad',(np.inf,np.nan,complex(1,1),np.complex64(1+1j),object()))
def test_fraction_object_arrays_reject_nonfinite_or_nonreal_values(bad):
    rows=np.array([[[F(1,3),bad],[1,0],[1,1]]],dtype=object)
    with pytest.raises(GeometryError):_triangle_rows(rows)


def test_rational_partition_cancellation_nonmutation():
    model,bindings,_,_=build();binding=bindings[1]
    rows=partition(model,binding);before=to_dict(model)
    error=RuntimeError('stop exact partition')
    def cancel(_):raise error
    with pytest.raises(RuntimeError) as caught:validate(model,binding,rows,cancellation_check=cancel)
    assert caught.value is error and to_dict(model)==before


def test_explicit_fraction_outside_extruded_child_range_cannot_round_inside():
    from anygeometry.authored_child_coverage import _support_correspondence
    from anygeometry import BezierDirectrix, ExtrudedSurface
    support=ExtrudedSurface(BezierDirectrix(((0,0,0),(.5,.25,0),(1,0,0))), (0,0,1))
    row=(F(1)+F(1,10**80),F(1,2))
    assert float(row[0])==1.0
    cells=np.array([[(F(0),F(0)),row,(F(0),F(1))]],dtype=object)
    with pytest.raises(GeometryError,match='support range'):
        _support_correspondence(support,support,cells)
