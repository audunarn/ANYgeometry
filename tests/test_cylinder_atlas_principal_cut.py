"""Independent regression for the atlas's lifted principal-cut interval union."""
from fractions import Fraction
import math
import numpy as np
import pytest

from anygeometry import (
    AffineTransform, Cylinder, GeometryModel, OrientedEdge,
    CylinderAtlasPolicy, CylinderAtlasStatus, CylinderOccurrenceRequest,
    query_cylinder_atlas, validate_cylinder_atlas_binding,
    evaluate_cylinder_occurrences,
)
from anygeometry import cylinder_charts as charts


def authored_ring(transform, *, sweep_sign=1, height_sign=1):
    """Exact retained public authoring recipe; no private stores/coordinate weld."""
    model=GeometryModel()
    vertices={};edges={};faces=[]
    def vertex(angle,z):
        key=(Fraction(angle)%8,Fraction(z))
        if key not in vertices:
            theta=float(key[0])*math.pi/4
            vertices[key]=model.add_point(*transform.apply_points((math.cos(theta),math.sin(theta),float(z))))
        return vertices[key]
    def edge(a,b):
        ka,kb=(a[0]%8,a[1]),(b[0]%8,b[1])
        circular=a[1]==b[1]
        key=(circular,*sorted((ka,kb)))
        start,end=vertex(*a),vertex(*b)
        if key not in edges:
            edges[key]=(model.add_arc(start,vertex((a[0]+b[0])/2,a[1]),end)
                        if circular else model.add_line(start,end))
        identifier=edges[key]
        return OrientedEdge(identifier,model.edges[identifier].start==start)
    matrix=transform.matrix[:3,:3]
    handedness=1 if np.linalg.det(matrix)>0 else -1
    with model.transaction():
        for index in range(8):
            points=tuple((Fraction(sweep_sign*(index+u)),Fraction(height_sign*z))
                         for u,z in ((0,0),(1,0),(1,2),(0,2)))
            loop=tuple(edge(a,b) for a,b in zip(points,points[1:]+points[:1]))
            surface=Cylinder(transform.apply_points((0.,0.,0.)),matrix@np.array((0.,0.,1.)),
                matrix@np.array((1.,0.,0.)),1.,2.*height_sign,
                handedness*sweep_sign*index*math.pi/4,handedness*sweep_sign*math.pi/4)
            faces.append(model.add_face_from_loop(loop,surface=surface))
        part=model.add_part(name='atlas cut regression')
        sheet=model.add_sheet(faces,part_id=part,orientations=(1,)*8)
    return model,tuple(model.handle('face_use',key) for key in model.sheets[sheet].face_use_ids)


@pytest.mark.parametrize('kind',['identity','rotate_translate','reflect'])
def test_public_atlas_cut_covariance(kind):
    transform=(AffineTransform.rotation((0.,0.,0.),(1.,1.,1.),2*np.pi/3).then(
        AffineTransform.translation((2.,-3.,5.))) if kind=='rotate_translate' else
        AffineTransform.reflection((0.,0.,0.),(1.,0.,0.)) if kind=='reflect' else
        AffineTransform(np.eye(4)))
    model,selected=authored_ring(transform)
    before=repr(model.__dict__)
    result=query_cylinder_atlas(model,selected,reference_face_use=selected[0],expected_revision=model.revision)
    assert result.status is CylinderAtlasStatus.QUALIFIED,result.diagnostics
    assert result.certificate.complete
    validate_cylinder_atlas_binding(model,result,selected,reference_face_use=selected[0],expected_revision=model.revision)
    requests=tuple(CylinderOccurrenceRequest(occurrence.id,.5) for occurrence in result.occurrences)
    rows=evaluate_cylinder_occurrences(model,result,requests,face_uses=selected,
        reference_face_use=selected[0],expected_revision=model.revision)
    assert len(rows.samples)==len(requests)
    assert repr(model.__dict__)==before


@pytest.mark.parametrize('turn',[-2,-1,0,1,2])
def test_split_cut_has_connected_certified_lift(turn):
    proof=charts._Proof(CylinderAtlasPolicy(),None)
    geometry=object.__new__(charts._GeometryProof)
    geometry.p=proof
    eps=Fraction(1,2**80)
    target=proof.mul(2*turn+1,proof.pi_bound())
    angle=geometry.lifted_vertex_angle(proof.i(-1),proof.i(-eps,eps),target.mid)
    assert angle.lo<=target.lo<=target.hi<=angle.hi
    assert angle.hi-angle.lo<Fraction(1,10**20)


def test_origin_box_retains_fail_closed_refusal():
    proof=charts._Proof(CylinderAtlasPolicy(),None)
    geometry=object.__new__(charts._GeometryProof)
    geometry.p=proof
    with pytest.raises(charts._Refusal,match='angular_branch_unqualified'):
        geometry.lifted_vertex_angle(proof.i(-1,1),proof.i(-1,1),0)
