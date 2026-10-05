"""Exact hole-aware area proof for straight planar authored partitions.

Independent dyadic fixtures: every coordinate is a multiple of 1/2**k, so
floats, Fractions and the exact kernel agree without any rounding oracle.
Partition-only exact hole exclusion permits complete closed contact while
retaining the shared curved kernel's refusals. These tests prove coverage and
refusal stages without sampling or tolerance.
"""
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from anygeometry import (GeometryError,GeometryModel,Plane,apply_intersections,
                         plan_intersections,to_dict,
                         query_prepared_authored_boundary_correspondence,
                         validate_prepared_authored_face_partition as validate)
from anygeometry.arrangement_geometry import BezierPath,LinePath
from anygeometry.authored_child_coverage import _literal_child_domain
from anygeometry.authored_domain_coverage import _original_domain
from anygeometry.authored_partition_coverage import _simple_area
from anygeometry.entities import OrientedEdge
from anygeometry.material_arrangement import ArrangementPath,MaterialDomain
from anygeometry.material_cell_coverage import _frame

OUTER=((0,0,0),(4,0,0),(4,4,0),(0,4,0))
# Hole loops follow the arrangement's clockwise convention; _simple_area
# itself accepts either orientation (proved at unit level below).
HOLE_A=((1,1,0),(1,1.5,0),(1.5,1.5,0),(1.5,1,0))              # area 1/4
HOLE_B=((2.25,2.25,0),(2.25,2.5,0),(2.5,2.5,0),(2.5,2.25,0))   # area 1/16
CONCAVE_OUTER=((0,0,0),(4,0,0),(4,1,0),(1,1,0),(1,4,0),(0,4,0))
CONCAVE_HOLE=((0.25,2,0),(0.25,2.5,0),(0.75,2.5,0),(0.75,2,0)) # area 1/4

# Known material areas: left child 12-5/16, right child 4, whole 16-5/16,
# concave 7-1/4, three-hole whole 16-21/64.
LEFT_AREA=F(187,16)
WHOLE_AREA=F(251,16)
CONCAVE_AREA=F(27,4)
THREE_HOLE_AREA=F(1003,64)

# Independently constructed rectangle tilings of each material region.
LEFT_BANDS=((0,0,3,1),(0,1,1,1.5),(1.5,1,3,1.5),(0,1.5,3,2.25),
            (0,2.25,2.25,2.5),(2.5,2.25,3,2.5),(0,2.5,3,4))
RIGHT_BANDS=((3,0,4,4),)
WHOLE_BANDS=((0,0,4,1),(0,1,1,1.5),(1.5,1,4,1.5),(0,1.5,4,2.25),
             (0,2.25,2.25,2.5),(2.5,2.25,4,2.5),(0,2.5,4,4))
CONCAVE_BANDS=((0,0,4,1),(0,1,1,2),(0,2,0.25,2.5),(0.75,2,1,2.5),(0,2.5,1,4))


def holed(*,fragment=True,outer=OUTER,holes=(HOLE_A,HOLE_B)):
    model=GeometryModel()
    face=model.add_plate(model.add_points(outer))
    model.set_face_surface(face,Plane((0,0,0),(1,0,0),(0,1,0)))
    loops=[]
    for points in holes:
        edges=model.add_polyline(model.add_points(points),close=True)
        loops.append(tuple(OrientedEdge(edge,True) for edge in edges))
    with model.transaction():
        model._put_entity('face',replace(model.faces[face],holes=tuple(loops)))
    if fragment:
        model.add_plate(model.add_points(((3,-1,-1),(3,5,-1),(3,5,1),(3,-1,1))))
    apply_intersections(model,plan_intersections(model,tuple(model.faces),
                                                 policy='connect'),policy='connect')
    return model,query_prepared_authored_boundary_correspondence(model,face)


def cells(bands):
    return np.asarray([[(x0,y0),(x1,y0),(x1,y1)] for x0,y0,x1,y1 in bands]
                      +[[(x0,y0),(x1,y1),(x0,y1)] for x0,y0,x1,y1 in bands],float)


def band_area(bands):
    """Independent dyadic shoelace oracle; no display sampling is involved."""
    return sum(F(x1-x0)*F(y1-y0) for x0,y0,x1,y1 in bands)


def straight_domain(outer,*holes):
    support=Plane((0,0,0),(1,0,0),(0,1,0))
    loops=[]
    for points in (outer,*holes):
        positions=[(*point,0) for point in points]
        loops.append(tuple(ArrangementPath(LinePath(a,b),index) for index,(a,b) in
                           enumerate(zip(positions,(*positions[1:],positions[0])))))
    return MaterialDomain(1,support,tuple(loops)),_frame(support)


def literal_area(model,binding,child):
    """The validator's own re-anchored literal child area, computed exactly."""
    original=_original_domain(binding.authored_definition,lambda:None)
    _,domain=_literal_child_domain(model,child)
    literal=MaterialDomain(child,original.support,domain.boundaries)
    return _simple_area(literal,_frame(original.support),lambda:None)


def children(model,binding):
    return sorted(binding.descendants,key=lambda face:min(
        model.vertex_position(model.oriented_start_vertex(use))[0]
        for use in model.faces[face].loop))


def test_simple_area_subtracts_valid_holes_exactly_with_any_orientation():
    square=((0,0),(4,0),(4,4),(0,4))
    hole_a=((1,1),(1.5,1),(1.5,1.5),(1,1.5))
    hole_b=((2.25,2.25),(2.5,2.25),(2.5,2.5),(2.25,2.5))
    hole_c=((3.25,3.25),(3.375,3.25),(3.375,3.375),(3.25,3.375))
    domain,frame=straight_domain(square,hole_a,hole_b)
    assert _simple_area(domain,frame,lambda:None)==WHOLE_AREA
    # Reversed outer loop, reversed holes and reordered holes change nothing.
    domain,frame=straight_domain(square[::-1],hole_a[::-1],hole_b[::-1])
    assert _simple_area(domain,frame,lambda:None)==WHOLE_AREA
    domain,frame=straight_domain(square,hole_b,hole_a)
    assert _simple_area(domain,frame,lambda:None)==WHOLE_AREA
    # A finite arbitrary hole count, not a fixed cap.
    domain,frame=straight_domain(square,hole_a,hole_b,hole_c)
    assert _simple_area(domain,frame,lambda:None)==THREE_HOLE_AREA


def test_concave_outer_with_hole_subtracts_exactly():
    concave=((0,0),(4,0),(4,1),(1,1),(1,4),(0,4))
    hole=((0.25,2),(0.75,2),(0.75,2.5),(0.25,2.5))
    domain,frame=straight_domain(concave,hole)
    assert _simple_area(domain,frame,lambda:None)==CONCAVE_AREA
    domain,frame=straight_domain(concave[::-1],hole[::-1])
    assert _simple_area(domain,frame,lambda:None)==CONCAVE_AREA


def test_fragmented_and_unfragmented_children_carry_exact_material_areas():
    model,binding=holed()
    left,right=children(model,binding)
    assert (literal_area(model,binding,left),literal_area(model,binding,right)) \
        ==(LEFT_AREA,F(4))
    assert band_area(LEFT_BANDS)==LEFT_AREA and band_area(RIGHT_BANDS)==F(4)
    model,binding=holed(fragment=False)
    assert literal_area(model,binding,binding.descendants[0])==WHOLE_AREA
    assert band_area(WHOLE_BANDS)==WHOLE_AREA
    model,binding=holed(fragment=False,outer=CONCAVE_OUTER,holes=(CONCAVE_HOLE,))
    assert literal_area(model,binding,binding.descendants[0])==CONCAVE_AREA
    assert band_area(CONCAVE_BANDS)==CONCAVE_AREA


def test_complete_grid_partitions_allow_exact_hole_contact_without_mutation():
    model,binding=holed();before=to_dict(model)
    left,right=children(model,binding)
    rows={left:cells(LEFT_BANDS),right:cells(RIGHT_BANDS)}
    copies={key:value.copy() for key,value in rows.items()}
    validate(model,binding,rows)
    validate(model,binding,{key:value[::-1,::-1] for key,value in reversed(tuple(rows.items()))})
    validate(model,binding,rows)
    assert to_dict(model)==before
    model,binding=holed(fragment=False);before=to_dict(model)
    validate(model,binding,{binding.descendants[0]:cells(WHOLE_BANDS)})
    assert to_dict(model)==before
    assert all(np.array_equal(value,copies[key]) for key,value in rows.items())


def test_complete_concave_hole_material_partition():
    model,binding=holed(fragment=False,outer=CONCAVE_OUTER,holes=(CONCAVE_HOLE,))
    before=to_dict(model)
    rows=cells(CONCAVE_BANDS)
    assert band_area(CONCAVE_BANDS)==CONCAVE_AREA
    validate(model,binding,{binding.descendants[0]:rows})
    validate(model,binding,{binding.descendants[0]:rows[::-1,::-1]})
    assert to_dict(model)==before


@pytest.mark.parametrize('collinear',[False,True])
def test_genuinely_concave_hole_complete_material_partition(collinear):
    # An L-shaped void, rather than a rectangular void in a concave plate.
    hole=((1,1,0),(3,1,0),(3,2,0),(2,2,0),(2,3,0),(1,3,0))
    if collinear:
        hole=(hole[0],(2,1,0),*hole[1:])
    bands=((0,0,4,1),(0,1,1,3),(3,1,4,2),(2,2,4,3),(0,3,4,4))
    assert band_area(bands)==13
    model,binding=holed(fragment=False,holes=(hole[::-1],))
    before=to_dict(model)
    assert literal_area(model,binding,binding.descendants[0])==13
    rows=cells(bands)
    validate(model,binding,{binding.descendants[0]:rows})
    validate(model,binding,{binding.descendants[0]:rows[::-1,::-1]})
    assert to_dict(model)==before


def test_partial_collinear_outer_contacts_do_not_hide_compensated_notch_fill():
    outer=((0,0,0),(4,0,0),(4,2,0),(3,2,0),(3,1,0),(1,1,0),(1,2,0),(0,2,0))
    model,binding=holed(fragment=False,outer=outer,holes=())
    before=to_dict(model)
    child=binding.descendants[0]
    # Areas 4+2 equal material area 6, but the first triangle fills the notch
    # and omits equal material at bottom right. Partial outer overlap emits
    # exact side events; it cannot certify the whole side as covered.
    incorrect=np.asarray([[(0,2),(4,2),(2,0)],[(0,0),(0,2),(2,0)]],float)
    assert literal_area(model,binding,child)==6
    with pytest.raises(GeometryError,match='side outside'):
        validate(model,binding,{child:incorrect})
    valid=((0,0,4,1),(0,1,1,2),(3,1,4,2))
    assert band_area(valid)==6
    validate(model,binding,{child:cells(valid)})
    assert to_dict(model)==before


def test_hole_filling_cells_refuse_with_or_without_exact_area_compensation():
    model,binding=holed(fragment=False);child=binding.descendants[0];before=to_dict(model)
    # Uncompensated: a cell strictly covering hole A adds 9/16 of area.
    filled=cells(WHOLE_BANDS+((0.875,0.875,1.625,1.625),))
    assert band_area(WHOLE_BANDS+((0.875,0.875,1.625,1.625),))==WHOLE_AREA+F(9,16)
    with pytest.raises(GeometryError):validate(model,binding,{child:filled})
    # Compensated: the hole band is covered and exactly 5/16 is removed above
    # and below it, so the total still equals the material area 251/16.
    compensated=((0,0,4,1),(0,1,1,1.5),(1.5,1,4,1.5),(0,1.5,4,2.171875),
                 (0,2.1875,4,2.5625),(0,2.5625,4,4))
    assert band_area(compensated)==WHOLE_AREA
    with pytest.raises(GeometryError):validate(model,binding,{child:cells(compensated)})
    assert to_dict(model)==before


def test_duplicate_and_compensated_overlap_cannot_buy_holed_coverage():
    model,binding=holed(fragment=False);child=binding.descendants[0]
    # Two copies of one valid cell double the area sum: only the exact
    # per-child area equality can refuse this containment-clean mesh.
    small=np.asarray([[[.05,.05],[.3,.05],[.05,.3]]]*2)
    with pytest.raises(GeometryError,match='do not cover the literal child area'):
        validate(model,binding,{child:small})
    # [0,2]x[0,1] covered twice, [2,4]x[0,1] left uncovered: the total area
    # still matches, so containment/overlap proofs must refuse it.
    overlapped=((0,0,2,1),(0,0,2,1),(0,1,1,1.5),(1.5,1,4,1.5),(0,1.5,4,2.25),
                (0,2.25,2.25,2.5),(2.5,2.25,4,2.5),(0,2.5,4,4))
    assert band_area(overlapped)==WHOLE_AREA
    with pytest.raises(GeometryError):validate(model,binding,{child:cells(overlapped)})


def test_outside_in_hole_and_straddling_cells_refuse_for_exact_reasons():
    model,binding=holed(fragment=False);child=binding.descendants[0]
    outside=np.asarray([[[5,5],[6,5],[5,6]]])
    with pytest.raises(GeometryError,match='vertex outside material'):
        validate(model,binding,{child:outside})
    in_hole=np.asarray([[[2.3,2.3],[2.45,2.3],[2.3,2.45]]])
    with pytest.raises(GeometryError,match='cell intersects hole interior'):
        validate(model,binding,{child:in_hole})
    straddling=np.asarray([[[2,2.3],[2.7,2.3],[2.7,2.45]]])
    with pytest.raises(GeometryError,match='cell intersects hole interior'):
        validate(model,binding,{child:straddling})
    enclosing=np.asarray([[[2,2.1],[2.7,2.1],[2.7,2.8]]])
    with pytest.raises(GeometryError,match='cell intersects hole interior'):
        validate(model,binding,{child:enclosing})


@pytest.mark.parametrize('holes,message',[
    ((((1,1),(3,1),(3,3),(1,3)),((1.5,1.5),(2.5,1.5),(2.5,2.5),(1.5,2.5))), 'nested or overlapping'),
    ((((1,1),(2,1),(2,2),(1,2)),((1.5,1.5),(2.5,1.5),(2.5,2.5),(1.5,2.5))), 'hole boundaries touch or cross'),
    ((((1,1),(2,1),(2,2),(1,2)),((2,1),(3,1),(3,2),(2,2))), 'hole boundaries touch or cross'),
    ((((1,1),(2,1),(2,2),(1,2)),((2,2),(3,2),(3,3),(2,3))), 'hole boundaries touch or cross'),
    ((((1,3),(1,4),(2,4),(2,3)),), 'touches or crosses the outer boundary'),
    ((((3,3),(5,3),(5,5),(3,5)),), 'touches or crosses the outer boundary'),
    ((((5,5),(6,5),(6,6),(5,6)),), 'not strictly contained'),
    ((((-1,-1),(5,-1),(5,5),(-1,5)),), 'not strictly contained'),
    ((((1,1),(2,2),(2,1),(1,2)),), 'not simple'),
    ((((1,1),(2,2)),), 'degenerate or repeated boundary vertex'),
    ((((1,1),(2,2),(3,3)),), 'backtracks'),
])
def test_invalid_hole_families_refuse_before_any_area_is_subtracted(holes,message):
    domain,frame=straight_domain(((0,0),(4,0),(4,4),(0,4)),*holes)
    with pytest.raises(GeometryError,match=message):
        _simple_area(domain,frame,lambda:None)


def test_curved_hole_loops_refuse_explicitly():
    support=Plane((0,0,0),(1,0,0),(0,1,0))
    corners=((1,1,0),(2,1,0),(2,2,0),(1,2,0))
    loop=tuple(ArrangementPath(BezierPath((a,((a[0]+b[0])/2,(a[1]+b[1])/2,0),b)),index)
               for index,(a,b) in enumerate(zip(corners,(*corners[1:],corners[0]))))
    domain=MaterialDomain(1,support,(loop,))
    with pytest.raises(GeometryError,match='straight line-segment'):
        _simple_area(domain,_frame(support),lambda:None)


def test_model_layer_refuses_authoring_nested_holes():
    with pytest.raises(GeometryError,match='holes 1 and 2 overlap'):
        holed(fragment=False,holes=(((1,1,0),(3,1,0),(3,3,0),(1,3,0)),
                                     ((1.5,1.5,0),(2.5,1.5,0),(2.5,2.5,0),(1.5,2.5,0))))


def test_cancellation_and_stale_source_never_accept_a_holed_partial_proof():
    model,binding=holed();before=to_dict(model)
    left,right=children(model,binding)
    rows={left:cells(LEFT_BANDS),right:cells(RIGHT_BANDS)}
    with pytest.raises(GeometryError,match='cancelled'):
        validate(model,binding,rows,cancellation_check=lambda _:True)
    assert to_dict(model)==before
    error=RuntimeError('hole partition work exhausted')
    def exhaust(_):raise error
    with pytest.raises(RuntimeError) as caught:
        validate(model,binding,rows,cancellation_check=exhaust)
    assert caught.value is error
    def mutate(_):
        model._faces[left]=replace(model.faces[left],metadata={'changed':True})
        return False
    with pytest.raises(GeometryError):
        validate(model,binding,rows,cancellation_check=mutate)
