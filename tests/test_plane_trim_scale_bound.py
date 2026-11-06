"""Pure binary-width arithmetic; no GeometryModel or geometry qualification."""
import ast
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from anygeometry import nurbs_plane_arrangement as owner
BBOX=((Q(0),Q(1)),(Q(0),Q(1)),(Q(0),Q(0)))
MAC_Y=float.fromhex('-0x1.99999d3340000p-56')
def points(y=MAC_Y):return [(0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.),(.6,y,0.),(.6,1.,0.)]
def gate(values,*,plate=False,closed=False,arcs=1,paths=1,missing=False,entries=2,duplicate=False,endpoint=False,trap=False):
    # Execute the exact original/new bound decision, not a substitute geometry provider.
    tree=ast.parse(Path(owner.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fragment_nurbs_plane')
    block=next(n for n in fn.body if isinstance(n,ast.If) and ast.unparse(n.test)=='joint_bound is not None')
    bound=(BBOX,Q(1,64));vertices={('vertex',i):p for i,p in enumerate(values)}
    keys=list(vertices)
    event_rows=[(keys[i%len(keys)],NS(section_interval=(Q(i+1,3),Q(i+1,3)))) for i in range(entries)]
    if missing:event_rows=[(('absent',),event) for key,event in event_rows]
    if duplicate:event_rows=[(keys[0],event) for key,event in event_rows]
    if endpoint:event_rows[0]=(event_rows[0][0],NS(section_interval=(Q(0),Q(0))))
    namespace=dict(vars(owner),joint_bound=bound,model=NS(vertex_position=lambda p:p),vertices=vertices,
        section=NS(paths=[NS(closed=closed,arcs=tuple(range(arcs))) for _ in range(paths)]),
        plate_stations={'plate':[]} if plate else {},events_by_path=[event_rows])
    if trap:namespace['_staged_joint_widths_bounded']=lambda *a:pytest.fail('inside path evaluated new width fallback')
    exec(compile(ast.Module(body=[block],type_ignores=[]),owner.__file__,'exec'),namespace)
    return namespace['joint_bound'],bound

def test_captured_mac_escape_keeps_exact_same_bound():
    assert MAC_Y<0 and tuple(np.ptp(np.asarray(points()),axis=0))==(1.,1.,0.)
    kept,original=gate(points());assert kept is original

def test_original_inside_fast_path_never_calls_new_arithmetic():
    kept,original=gate(points(0.),trap=True);assert kept is original

@pytest.mark.parametrize('fault',['plate','closed','arcs','paths','missing','entries','duplicate','endpoint'])
def test_extra_vertex_scope_never_gains_escaped_fallback(fault):
    options={'plate':False,'closed':False,'arcs':1,'paths':1,'missing':False,'entries':2,'duplicate':False,'endpoint':False}
    options[fault]={'plate':True,'closed':True,'arcs':2,'paths':2,'missing':True,'entries':3,'duplicate':True,'endpoint':True}[fault]
    assert gate(points(),**options)[0] is None

@pytest.mark.parametrize('value',[np.nextafter(-2.**-53,-np.inf),-1e-4])
def test_genuinely_wider_actual_width_refuses(value):
    assert np.ptp(np.asarray(points(value)),axis=0)[1]>1.
    assert not owner._staged_joint_widths_bounded(points(value),BBOX)
    assert gate(points(value))[0] is None

@pytest.mark.parametrize('bad',[float('inf'),float('-inf'),float('nan')])
def test_nonfinite_width_data_refuses(bad):
    assert not owner._staged_joint_widths_bounded([(bad,0.,0.),(1.,1.,0.)],BBOX)


def test_zero_signedzero_and_validator_zero_scale_fallback():
    data=[(-0.,0.,-0.),(0.,-0.,0.)]
    assert owner._staged_joint_widths_bounded(data,BBOX)
    parent=float(np.linalg.norm(np.array([1.,1.,0.])))
    child=float(np.linalg.norm(np.ptp(np.asarray(data),axis=0))) or 1.
    assert child==1. and child<=parent


def test_all_small_subsets_have_bounded_actual_validator_norm():
    from itertools import combinations
    data=points();parent=float(np.linalg.norm(np.array([1.,1.,0.])))
    assert owner._staged_joint_widths_bounded(data,BBOX)
    for size in (1,2,3):
        for subset in combinations(data,size):
            widths=np.ptp(np.asarray(subset),axis=0)
            assert np.all(widths<=np.array([1.,1.,0.]))
            assert (float(np.linalg.norm(widths)) or 1.)<=parent


def test_overflow_and_empty_shape_fail_closed():
    with np.errstate(over='ignore',invalid='ignore'):
        assert not owner._staged_joint_widths_bounded([(-np.finfo(float).max,0.,0.),(np.finfo(float).max,1.,0.)],BBOX)
    assert not owner._staged_joint_widths_bounded([],BBOX)


def test_nextafter_interior_nearbound_is_accepted_without_tolerance():
    x=np.nextafter(1.,0.)
    data=[(0.,0.,0.),(x,x,0.)]
    assert owner._staged_joint_widths_bounded(data,BBOX)
    assert tuple(np.ptp(np.asarray(data),axis=0))==(x,x,0.)


def test_readonly_array_identity_bytes_and_flags_are_unchanged():
    data=np.asarray(points());data.flags.writeable=False
    before=data.tobytes();identity=id(data);flags=(data.flags.writeable,data.flags.c_contiguous)
    assert owner._staged_joint_widths_bounded(data,BBOX)
    assert id(data)==identity and data.tobytes()==before
    assert (data.flags.writeable,data.flags.c_contiguous)==flags
