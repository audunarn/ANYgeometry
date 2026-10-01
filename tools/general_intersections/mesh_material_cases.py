"""Default application meshes with independent material and joint oracles."""
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from anygeometry import GeometryModel,Cylinder,to_dict
from anygeometry.entities import OrientedEdge
from anygeometry.generators import cylinder
from anygeometry.operations import trim_face
from anyfem import Project,steel,fixed
from anyfem.model import BeamSection
from anyfem.io.project_file import save_project,load_project,project_to_dict
from anymesher import MeshAutomationOptions,mesh_to_dict,mesh_from_dict
from anymesher.errors import MeshError


def plate(model,points):
    return model.add_plate(model.add_points(points))


def make_case(name):
    if name.startswith('cylinder-holes-'):
        count=int(name.rsplit('-',1)[1])
        model=GeometryModel()
        surface=Cylinder((0.,0.,0.),(0.,0.,1.),(1.,0.,0.),1.,3.,0.,math.pi/2)
        def loop(points):
            vertices=model.add_points(tuple(surface.evaluate(*point) for point in points))
            edges=[]
            for i,(a,b) in enumerate(zip(points,points[1:]+points[:1])):
                if a[1]==b[1]:
                    middle=model.add_point(*surface.evaluate((a[0]+b[0])/2,a[1]))
                    edge=model.add_arc(vertices[i],middle,vertices[(i+1)%4])
                else:
                    edge=model.add_line(vertices[i],vertices[(i+1)%4])
                edges.append(OrientedEdge(edge,True))
            return tuple(edges)
        face=model.add_face_from_loop(loop(((0.,0.),(1.,0.),(1.,1.),(0.,1.))),surface=surface)
        width=.4; height=.4/count
        starts=tuple((i+.3)/count for i in range(count))
        trim_face(model,face,tuple(loop(((.3,v),(.3,v+height),(.7,v+height),(.7,v))) for v in starts))
        model.add_sheet((face,),part_id=model.add_part(name='cylinder'))
        assert model.validate_topology()==()
        return Project(name,geometry=model),((face,),),(1.5*math.pi*(1-width*height*count),),()
    if name.startswith('growing-'):
        count=int(name.split('-')[1])
        model=GeometryModel()
        host=plate(model,((-2.,-2.,0.),(2.,-2.,0.),(2.,2.,0.),(-2.,2.,0.)))
        walls=tuple(plate(model,((x,-1.,-.5),(x,1.,-.5),(x,1.,.5),(x,-1.,.5)))
                    for x in (-2.+4.*(index+1)/(count+1) for index in range(count)))
        return Project(name,geometry=model),tuple((face,) for face in (host,*walls)),(16.,*([2.]*count)),()
    if name=='multiple-cuts':
        model=GeometryModel()
        host=plate(model,((-4.,-2.,0.),(4.,-2.,0.),(4.,2.,0.),(-4.,2.,0.)))
        owners=[(host,)]
        for center,axis in (((-2.5,0.,0.),(0.,0.,1.)),
                            ((0.,0.,0.),(.3,0.,1.)),
                            ((2.5,0.,0.),(0.,.4,1.))):
            axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
            before=set(model.faces)
            model.insert_model(cylinder(.5,2.,circumferential_segments=12,
                                        origin=tuple(np.asarray(center)-axis),axis=tuple(axis)))
            owners.append(tuple(sorted(set(model.faces)-before)))
        return Project('multiple closed cuts',geometry=model),tuple(owners),(32.,*(2*math.pi for _ in range(3))),()
    if name=='oblique':
        model=cylinder(.75,2.,circumferential_segments=12,origin=(0.,0.,-1.))
        cylinders=tuple(sorted(model.faces))
        face=plate(model,((-2.,-2.,-1.),(2.,-2.,1.),(2.,2.,1.),(-2.,2.,-1.)))
        return Project('oblique material',geometry=model),((face,),cylinders),(16*math.sqrt(1.25),3*math.pi),()
    project=Project('concave holed stiffeners')
    model=project.geometry
    face=plate(model,((0.,0.,0.),(6.,0.,0.),(6.,2.,0.),(3.,2.,0.),(3.,4.,0.),(0.,4.,0.)))
    holes=[]
    for points in (((.5,.5,0.),(1.,.5,0.),(1.,1.,0.),(.5,1.,0.)),
                   ((1.5,2.5,0.),(2.5,2.5,0.),(2.5,3.5,0.),(1.5,3.5,0.))):
        vertices=model.add_points(points)
        holes.append(tuple(OrientedEdge(model.add_line(vertices[i],vertices[(i+1)%4]),True) for i in range(4)))
    trim_face(model,face,tuple(holes))
    walls=(plate(model,((1.25,.3,-.5),(1.25,1.75,-.5),(1.25,1.75,.5),(1.25,.3,.5))),
           plate(model,((.25,1.5,-.5),(5.75,1.5,-.5),(5.75,1.5,.5),(.25,1.5,.5))),
           plate(model,((2.75,.3,-.5),(2.75,3.75,-.5),(2.75,3.75,.5),(2.75,.3,.5))))
    beams=tuple(model.add_line(*model.add_points(points)) for points in (
        ((0.,1.5,0.),(6.,1.5,0.)),((.1,.1,0.),(2.9,2.9,0.))))
    return project,tuple((value,) for value in (face,*walls)),(16.75,1.45,5.5,3.45),beams


def topology(mesh,faces):
    edges=Counter()
    nodes=set()
    for face in faces:
        for element in mesh.elements_of_face[face]:
            corners=mesh.corners_of(element);nodes.update(corners)
            edges.update(tuple(sorted((corners[i],corners[(i+1)%len(corners)]))) for i in range(len(corners)))
    return set(edges),nodes


def intervals(values,start,end):
    values=sorted(values)
    assert abs(values[0][0]-start)<1e-8 and abs(values[-1][1]-end)<1e-8
    assert all(abs(first[1]-second[0])<1e-8 for first,second in zip(values,values[1:])),values


def verify(mesh,name,owners,expected,beams):
    edge_sets=[];node_sets=[];areas=[]
    for index,faces in enumerate(owners):
        edges,nodes=topology(mesh,faces);edge_sets.append(edges);node_sets.append(nodes)
        area=0.
        for face in faces:
            for element in mesh.elements_of_face[face]:
                xyz=np.asarray([mesh.nodes[node] for node in mesh.corners_of(element)])
                if name.startswith('cylinder-holes-'):
                    np.testing.assert_allclose(np.linalg.norm(xyz[:,:2],axis=1),1.,rtol=0.,atol=1e-8)
                    angle=np.unwrap(np.arctan2(xyz[:,1],xyz[:,0]))
                    area+=abs(np.dot(angle,np.roll(xyz[:,2],-1))-np.dot(xyz[:,2],np.roll(angle,-1)))/2
                    center=np.mean(xyz,axis=0)
                    u=math.atan2(center[1],center[0])/(math.pi/2);v=center[2]/3
                    count=int(name.rsplit('-',1)[1])
                    assert not any(.3+1e-8<u<.7-1e-8 and start+1e-8<v<start+.4/count-1e-8
                                   for start in ((i+.3)/count for i in range(count)))
                elif name=='multiple-cuts' and index:
                    center,axis=(((-2.5,0.,0.),(0.,0.,1.)),
                                 ((0.,0.,0.),(.3,0.,1.)),
                                 ((2.5,0.,0.),(0.,.4,1.)))[index-1]
                    axis=np.asarray(axis);axis=axis/np.linalg.norm(axis)
                    radial=np.asarray((1.,0.,0.));radial-=axis*(axis@radial);radial/=np.linalg.norm(radial)
                    tangent=np.cross(axis,radial)
                    offsets=xyz-np.asarray(center);height=offsets@axis
                    first,second=offsets@radial,offsets@tangent
                    np.testing.assert_allclose(np.hypot(first,second),.5,rtol=0.,atol=1e-8)
                    angle=np.unwrap(np.arctan2(second,first))
                    area+=.5*abs(np.dot(angle,np.roll(height,-1))-np.dot(height,np.roll(angle,-1)))/2
                elif name=='oblique' and index==1:
                    np.testing.assert_allclose(np.linalg.norm(xyz[:,:2],axis=1),.75,rtol=0.,atol=1e-8)
                    angle=np.unwrap(np.arctan2(xyz[:,1],xyz[:,0]))
                    area+=.75*abs(np.dot(angle,np.roll(xyz[:,2],-1))-np.dot(xyz[:,2],np.roll(angle,-1)))/2
                else:
                    area+=sum(np.linalg.norm(np.cross(xyz[i]-xyz[0],xyz[i+1]-xyz[0]))/2
                              for i in range(1,len(xyz)-1))
        areas.append(float(area))
    np.testing.assert_allclose(areas,expected,rtol=0.,atol=1e-8)
    joints=[]
    for a,b in combinations(range(len(owners)),2):
        shared=edge_sets[a]&edge_sets[b]
        shared_nodes={node for edge in shared for node in edge}
        assert shared_nodes==node_sets[a]&node_sets[b]
        if name.startswith('growing-'):
            if a:
                assert not shared
                continue
            assert shared
            intervals([tuple(sorted((float(mesh.nodes[p][1]),float(mesh.nodes[q][1]))))
                       for p,q in shared],-1.,1.)
        elif name=='multiple-cuts':
            if a:
                assert not shared
                continue
            assert shared and set(Counter(node for edge in shared for node in edge).values())=={2}
            center,axis=(((-2.5,0.,0.),(0.,0.,1.)),((0.,0.,0.),(.3,0.,1.)),
                         ((2.5,0.,0.),(0.,.4,1.)))[b-1]
            axis=np.asarray(axis);axis/=np.linalg.norm(axis)
            radial=np.asarray((1.,0.,0.));radial-=axis*(axis@radial);radial/=np.linalg.norm(radial)
            tangent=np.cross(axis,radial)
            values=[]
            for first,second in shared:
                offsets=np.asarray([mesh.nodes[first],mesh.nodes[second]])-np.asarray(center)
                np.testing.assert_allclose(offsets[:,2],0.,rtol=0.,atol=1e-8)
                pair=np.column_stack((offsets@radial,offsets@tangent))
                begin=math.atan2(pair[0,1],pair[0,0])%(2*math.pi)
                delta=math.atan2(np.linalg.det(pair),float(pair[0]@pair[1]))
                if delta<0:begin=(begin+delta)%(2*math.pi);delta=-delta
                finish=begin+delta
                if finish>2*math.pi:values.extend(((begin,2*math.pi),(0.,finish-2*math.pi)))
                else:values.append((begin,finish))
            intervals(values,0.,2*math.pi)
        elif name=='oblique':
            assert shared and set(Counter(node for edge in shared for node in edge).values())=={2}
            values=[]
            for first,second in shared:
                p,q=mesh.nodes[first],mesh.nodes[second]
                np.testing.assert_allclose((p[2]-.5*p[0],q[2]-.5*q[0]),0.,rtol=0.,atol=1e-8)
                begin=math.atan2(p[1],p[0])%(2*math.pi)
                delta=math.atan2(p[0]*q[1]-p[1]*q[0],float(p[:2]@q[:2]))
                if delta<0:begin=(begin+delta)%(2*math.pi);delta=-delta
                finish=begin+delta
                if finish>2*math.pi:values.extend(((begin,2*math.pi),(0.,finish-2*math.pi)))
                else:values.append((begin,finish))
            intervals(values,0.,2*math.pi)
        else:
            specification={(0,1):(1,.3,1.75),(0,2):(0,.25,5.75),(0,3):(1,.3,3.75),
                           (1,2):(2,-.5,.5),(2,3):(2,-.5,.5)}
            if (a,b) not in specification:
                assert not shared
                continue
            dimension,start,end=specification[a,b]
            assert shared
            intervals([tuple(sorted((float(mesh.nodes[p][dimension]),float(mesh.nodes[q][dimension]))))
                       for p,q in shared],start,end)
        joints.append({'owners':[a,b],'shared_segments':len(shared),'shared_nodes':len(shared_nodes)})
    beam_checks=[]
    for index,edge in enumerate(beams):
        cells=[mesh.beams[element][:2] for element in mesh.elements_of_edge[edge] if element in mesh.beams]
        beam_edges={tuple(sorted(nodes)) for nodes in cells}
        length=sum(np.linalg.norm(mesh.nodes[a]-mesh.nodes[b]) for a,b in cells)
        assert abs(length-(6. if index==0 else 2.8*math.sqrt(2)))<1e-8
        shared=beam_edges&edge_sets[0]
        shared_length=sum(np.linalg.norm(mesh.nodes[a]-mesh.nodes[b]) for a,b in shared)
        assert abs(shared_length-(6. if index==0 else 2.3*math.sqrt(2)))<1e-8
        values=[tuple(sorted((float(mesh.nodes[a][0]),float(mesh.nodes[b][0])))) for a,b in shared]
        if index==0:intervals(values,0.,6.)
        else:
            assert all(b<=.5+1e-8 or a>=1.-1e-8 for a,b in values)
            intervals([value for value in values if value[1]<=.5+1e-8],.1,.5)
            intervals([value for value in values if value[0]>=1.-1e-8],1.,2.9)
        beam_checks.append({'edge':edge,'elements':len(cells),'length':float(length),
                            'shared_material_length':float(shared_length)})
    return {'material_areas':areas,'joints':joints,'beams':beam_checks}


def run_case(name):
    project,owners,expected,beams=make_case(name)
    project.add_material(steel());project.add_plate_section('plate',.01,'S355')
    for faces in owners:
        for face in faces:project.assign_plate(face,'plate')
    if beams:
        project.add_beam_section(BeamSection(name='beam',profile='Flatbar',material='S355',
                                            flange_width=.01,flange_thickness=.1))
        for edge in beams:project.assign_beam(edge,'beam')
    face=owners[0][0]
    project.add_support(fixed(project.edge(project.geometry.faces[face].loop[0].edge)))
    project.load_case().add_surface_traction(project.face(face),(0.,0.,-1000.))
    before=project_to_dict(project);start=perf_counter();last=start
    def cancellation(stage):
        nonlocal last
        now=perf_counter()
        if now-start>900.:raise MeshError('material fixture exceeded its 900-second budget')
        if now-last>10.:print('PROGRESS',name,stage,now-start,flush=True);last=now
    print('START',name,'DEFAULT AUTOMATIC LINEAR',flush=True)
    root=Path(__file__).parent
    try:
        mesh=project.generate_mesh(.5,strategy='auto',order='linear',automation=MeshAutomationOptions(),
                                   cancellation_check=cancellation)
    finally:
        assert project_to_dict(project)==before
    (root/f'candidate-material-{name}-mesh.json').write_text(json.dumps(mesh_to_dict(mesh),allow_nan=False))
    automation=mesh.hybrid_diagnostics['automation']
    print('AUTOMATION',json.dumps({k:automation.get(k) for k in
        ('status','solver_admission','quality_warnings','selected_method')},default=str),flush=True)
    assert mesh.hybrid_diagnostics['automation']['solver_admission']=='ADMITTED'
    assert mesh.hybrid_diagnostics['automation']['status']=='ready'
    checks=verify(mesh,name,owners,expected,beams)
    assert mesh_to_dict(mesh_from_dict(mesh_to_dict(mesh)))==mesh_to_dict(mesh)
    restored=load_project(save_project(project,root/f'accepted-material-{name}.anyfem'))
    assert project_to_dict(restored)==before
    report={'case':name,'target_size':.5,'source_unchanged':True,'save_load':True,
            'nodes':len(mesh.nodes),'elements':len(mesh.shells),'automation':mesh.hybrid_diagnostics['automation'],
            'checks':checks,'seconds':perf_counter()-start}
    (root/f'accepted-material-{name}-checks.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (root/f'accepted-material-{name}-mesh.json').write_text(json.dumps(mesh_to_dict(mesh),allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='automation'},indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    cases={'growing':('growing-1','growing-8','growing-25'),
           'cylinder-holes':('cylinder-holes-1','cylinder-holes-2','cylinder-holes-4')}
    for name in cases.get(sys.argv[1],(sys.argv[1],)):
        run_case(name)
