"""Public native meshing beyond the historical atlas cardinality ceiling."""
import math,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from anygeometry.generators import cylinder
from anyfem import Project,steel
from anyfem.mesh_controls import MeshControls
from anyfem.io.project_file import project_to_dict,save_project,load_project
from anymesher import MeshAutomationOptions,mesh_to_dict,mesh_from_dict
from anymesher.errors import MeshError
m=cylinder(1.,1.,circumferential_segments=260)
p=Project('260 cylinder sectors',geometry=m);p.add_material(steel());p.add_plate_section('plate',.01,'S355')
for face in m.faces:p.assign_plate(face,'plate')
before=project_to_dict(p);start=time.perf_counter();last=start
print('START 260 faces',flush=True)
def cancel(stage):
 global last
 now=time.perf_counter()
 if now-start>900:raise MeshError('capacity fixture exceeded its 900-second budget')
 if now-last>10:print('PROGRESS',stage,now-start,flush=True);last=now
mesh=p.generate_mesh(.05,strategy='native',order='linear',automation=MeshAutomationOptions(),mesh_controls=MeshControls(point_placement='frontal_delaunay'),cancellation_check=cancel)
assert project_to_dict(p)==before
assert mesh.hybrid_diagnostics['automation']['solver_admission']=='ADMITTED'
area=0.
for face in m.faces:
 for element in mesh.elements_of_face[face]:
  xyz=np.asarray([mesh.nodes[n] for n in mesh.corners_of(element)])
  np.testing.assert_allclose(np.linalg.norm(xyz[:,:2],axis=1),1.,rtol=0.,atol=1e-8)
  theta=np.unwrap(np.arctan2(xyz[:,1],xyz[:,0]))
  area+=abs(np.dot(theta,np.roll(xyz[:,2],-1))-np.dot(xyz[:,2],np.roll(theta,-1)))/2
assert abs(area-2*math.pi)<1e-8,area
face_nodes={};face_edges={}
for face in m.faces:
    nodes=set();edges=Counter()
    for element in mesh.elements_of_face[face]:
        corners=mesh.corners_of(element);nodes.update(corners)
        edges.update(tuple(sorted((a,b))) for a,b in zip(corners,corners[1:]+corners[:1]))
    face_nodes[face]=nodes;face_edges[face]=set(edges)
shared=0
for edge in m.edges:
    owners=m.faces_using_edge(edge)
    if len(owners)!=2:continue
    sequence=mesh.nodes_of_edge[edge]
    assert face_nodes[owners[0]]&face_nodes[owners[1]]==set(sequence)
    assert face_edges[owners[0]]&face_edges[owners[1]]=={tuple(sorted(pair)) for pair in zip(sequence,sequence[1:])}
    heights=np.asarray([mesh.nodes[node][2] for node in sequence])
    assert abs(min(heights))<1e-8 and abs(max(heights)-1.)<1e-8
    assert np.all(np.diff(heights)>0.) or np.all(np.diff(heights)<0.)
    shared+=1
assert shared==260
root=Path(__file__).parent
assert mesh_to_dict(mesh_from_dict(mesh_to_dict(mesh)))==mesh_to_dict(mesh)
assert project_to_dict(load_project(save_project(p,root/'accepted-cylinder-capacity.anyfem')))==before
report={'case':'cylinder-capacity','faces':260,'target_size':.05,'strategy':'native','point_placement':'frontal_delaunay',
        'nodes':len(mesh.nodes),'elements':len(mesh.shells),'source_unchanged':True,'save_load':True,
        'checks':{'material_area':float(area),'shared_panel_boundaries':shared},
        'automation':mesh.hybrid_diagnostics['automation'],'seconds':time.perf_counter()-start}
(root/'accepted-material-cylinder-capacity-checks.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in report.items() if k!='automation'},indent=2),flush=True)
