"""Application meshing of the whole-domain cylinder fixtures."""
import json
import sys
import cProfile
import pstats
import io
import faulthandler
from pathlib import Path
from time import perf_counter
from anygeometry import to_dict
from anygeometry.generators import cylinder
from anyfem import Project
from anyfem.mesh_controls import MeshControls
from anymesher import MeshAutomationOptions, mesh_to_dict
from anymesher.errors import MeshError
from cylinder_cases import cases

name=sys.argv[1]
model=cylinder(1.,3.,origin=(0.,0.,-1.5),circumferential_segments=8)
model.insert_model(cylinder(height=3.,circumferential_segments=8,**cases[name]))
project=Project('cylinder '+name,geometry=model)
before=to_dict(model)
start=perf_counter()
print('START',name,'TARGET',.5,'SOURCE FACES',len(model.faces),flush=True)
controls=(MeshControls(recombine=False,point_placement='frontal_delaunay',metric_mode='isotropic_spatial')
          if len(sys.argv)>2 and sys.argv[2]=='frontal' else None)
print('CONTROLS',None if controls is None else controls.parameters(),flush=True)
last_progress=start
profile=cProfile.Profile() if len(sys.argv)>3 and sys.argv[3]=='profile' else None
def check(phase):
    global last_progress
    elapsed=perf_counter()-start
    if perf_counter()-last_progress>10.:
        values={"phase":phase,"seconds":elapsed}
        frame=sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_name=='_mesh_native_face':
                values['face_id']=frame.f_locals.get('face_id')
            if frame.f_code.co_name=='frontal_delaunay_refine':
                for key in ('insertions','operations','geometry_limited'):
                    values[key]=frame.f_locals.get(key)
                values['point_count']=len(frame.f_locals.get('points',()))
            frame=frame.f_back
        print('PROGRESS',values,flush=True)
        last_progress=perf_counter()
    if profile is not None and perf_counter()-start>180.:
        raise MeshError('bounded cylinder mesh cancelled at '+phase)
try:
    # Preserve the exact phase/stack if an installed-platform case stalls;
    # these diagnostics do not change the fixed subprocess/resource budgets.
    faulthandler.dump_traceback_later(180., repeat=True)
    if profile is not None:profile.enable()
    mesh=project.generate_mesh(.5,strategy='auto',order='linear',automation=MeshAutomationOptions(),
                              mesh_controls=controls,cancellation_check=check)
finally:
    faulthandler.cancel_dump_traceback_later()
    if profile is not None:
        profile.disable()
        profile.dump_stats(str(Path(__file__).with_name('skew-frontal.pstats')))
        stream=io.StringIO()
        pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(35)
        Path(__file__).with_name('skew-frontal-profile.txt').write_text(stream.getvalue())
    assert to_dict(model)==before
owners=[]
for selected in (range(1,9),range(9,17)):
    owners.append({tuple(sorted((nodes[index],nodes[(index+1)%len(nodes)])))
        for face in selected for element in mesh.elements_of_face[face]
        for nodes in (mesh.corners_of(element),) for index in range(len(nodes))})
shared=owners[0]&owners[1]
assert shared
Path(__file__).with_name('accepted-cylinders-'+name+'.json').write_text(json.dumps(mesh_to_dict(mesh)))
print('MESH',len(mesh.nodes),len(mesh.shells),'SHARED SEGMENTS',len(shared),
      'AUTOMATION',mesh.hybrid_diagnostics['automation'],'SECONDS',perf_counter()-start,flush=True)
