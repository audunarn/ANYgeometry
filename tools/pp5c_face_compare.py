import json,os,pathlib,sys
W=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src'))
from anygeometry import _planning_executor as a
from anygeometry._planning_stages import WorkerModel,stage_task
from anygeometry._planning_values import encode_value,decode_value
from anygeometry.material_arrangement import MaterialDomain,arrange_material
from anygeometry.batch_intersections import _domain_bounds
base=W/'.pp1-evidence/pp5c-host-green1/pytest-temp/test_public_real_pair_and_face0/parallel/face'
request=a.read_packet(base/'worker-0/request.json','assignment');context=WorkerModel(request['model'])
_,task=request['tasks'][0]
actual=json.loads(stage_task(context,task));domain=MaterialDomain.from_model(context.model,task['face']);_domain_bounds(domain)
traces,points=decode_value(task['inputs'])
reference=encode_value(arrange_material(domain,traces,points=points,tolerance=task['tolerance'],area_tolerance=task['area_tolerance'],cancellation_check=lambda:False))
diffs=[]
def compare(a,b,path='root'):
    if type(a) is list and type(b) is list:
        if len(a)!=len(b):diffs.append([path,'length',len(a),len(b)])
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'['+str(i)+']')
    elif type(a) is not type(b) or a!=b:diffs.append([path,a,b])
compare(reference,actual)
out=pathlib.Path(os.environ['TASK_EVIDENCE_DIRECTORY']);(out/'comparison.json').write_text(json.dumps(dict(reference=reference,actual=actual,differences=diffs),indent=2));print(json.dumps(dict(differences=len(diffs),first=diffs[:12])))
