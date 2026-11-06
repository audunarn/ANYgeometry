"""PP5c real source/closure admission and owned constructor decode; no planning."""
import hashlib,json,os,pathlib,sys
REPO=pathlib.Path(__file__).resolve().parents[1];ROOT=REPO.parents[1]
sys.path.insert(0,str(ROOT/'resume'))
import env_setup
original=env_setup.repo_root
env_setup.repo_root=lambda name: REPO if name=='ANYgeometry' else (ROOT/'int/ANYmesh' if name=='ANYmesh' else original(name))
env_setup.setup()
from anyfem.io import load_project
from anygeometry._planning_executor import freeze_model,thaw_model,unpack,pack
from anygeometry.serialization import to_dict
from anygeometry.surfaces import Cylinder
from anygeometry.closure import extract_model_closure
out=pathlib.Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
rows={}
def check(label,model,expected):
    before=json.dumps(to_dict(model),sort_keys=True,separators=(',',':'),allow_nan=False)
    packet=freeze_model(model);body=unpack(packet,'model')
    assert len(body['constructor_state'])==expected
    restored=thaw_model(packet)
    again=freeze_model(restored)
    expected_body=dict(body)
    if body['extension_record']==dict(present=False,sha256=None):
        expected_body['extension_record']=dict(present=True,sha256=hashlib.sha256(b'{}').hexdigest())
    assert again==pack(expected_body,'model')
    assert json.dumps(to_dict(model),sort_keys=True,separators=(',',':'),allow_nan=False)==before
    rows[label]=dict(faces=len(model.faces),derived_records=expected,payload_bytes=len(packet['payload'].encode()),transport_sha256=packet['sha256'],model_document_sha256=hashlib.sha256(before.encode()).hexdigest(),constructor_roundtrip_equal=True,source_unchanged=True,extension_transition=('absent-to-empty' if not body['extension_record']['present'] else 'unchanged'))
    (out/'admission.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
source=pathlib.Path('C:/ANYworkspaceWork/AIP Gen2/ANYfem-local-assessment/hydrodynamics/AIP-Gen2-full-tripod-no-tower.anyfem')
p=load_project(source);g=p.geometry
check('sector60_r7_source',g,540)
face=min(key for key,value in g.faces.items() if type(value.surface) is Cylinder)
small=extract_model_closure(g,(g.handle('face',face),),include_structural_closure=False,include_features=False).working_model
fixture=to_dict(small)
(out/'cylinder-input.json').write_text(json.dumps(fixture,sort_keys=True,indent=2),encoding='utf-8')
rows['fixture']=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_face=face,faces=len(small.faces),document_sha256=hashlib.sha256(json.dumps(fixture,sort_keys=True,separators=(',',':')).encode()).hexdigest())
source=ROOT/'model-n2-20261030/AIP-Gen2-full-tripod-no-tower.MODEL-N2c-rotated-arms.SIMPLIFIED-EXERCISE.anyfem'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='969cc5f9c8efe5b628686c9a89b1de02ea2dea8c1618f4295cb049c2621f468a'
p=load_project(source);g=p.geometry
check('N2c_source',g,18)
members={m for e in p.beam_edges for m in g.members_using_edge(e)}
handles=tuple([g.handle('face',i) for i in sorted(g.faces)]+[g.handle('edge',i) for i in p.beam_edges]+[g.handle('sheet',i) for i in sorted(g.sheets)]+[g.handle('member',i) for i in sorted(members)])
closure=extract_model_closure(g,handles,include_structural_closure=True,include_features=False)
assert len(closure.working_model.faces)==561
check('N2c_closure',closure.working_model,18)
assert not env_setup.foreign_modules()
(out/'admission.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows))