"""Read-only census of actual sector60 stage outputs and retained model state."""
import collections,dataclasses,json,os,pathlib,runpy,sys,struct
import numpy as np
ROOT=pathlib.Path('C:/ANYworkspaceWork/anyfem-aip-startup-20261029');W=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'resume'));import env_setup
original=env_setup.repo_root
env_setup.repo_root=lambda name: W if name=='ANYgeometry' else (ROOT/'int/ANYmesh' if name=='ANYmesh' else original(name))
env_setup.setup()
from anygeometry import batch_intersections as bi
from anygeometry._planning_qualification import _check_model_record_state,QualificationDecline
out=pathlib.Path(os.environ['TASK_EVIDENCE_DIRECTORY']);records=collections.Counter();extras=collections.Counter();warm=collections.Counter();calls=collections.Counter();retained=[];rebuild=collections.Counter();failures=[];warm_nonempty=collections.Counter()
def census(value):
    pending=[value];seen=set()
    while pending:
        item=pending.pop()
        if id(item) in seen:continue
        seen.add(id(item));t=type(item)
        if t is dict:pending.extend(item.values());pending.extend(item)
        elif t in (tuple,list):pending.extend(item)
        elif dataclasses.is_dataclass(item) and not isinstance(item,type):
            name=t.__module__+'.'+t.__qualname__;records[name]+=1;fields=dataclasses.fields(item)
            namespace=getattr(item,'__dict__',{})
            for key in set(namespace)-{f.name for f in fields}:extras[name+'.'+key]+=1
            try:
                fresh=t(**{f.name:getattr(item,f.name) for f in fields if f.init})
                for f in fields:
                    if not f.init:continue
                    old,new=getattr(item,f.name),getattr(fresh,f.name)
                    if type(old) is np.ndarray:
                        assert type(new) is np.ndarray and old.dtype==new.dtype and old.shape==new.shape and old.tobytes()==new.tobytes(),f.name
                    elif type(old) is float:
                        assert type(new) is float and struct.pack('!d',old)==struct.pack('!d',new),f.name
                rebuild[name]+=1
            except Exception as exc:
                failures.append(dict(record=name,error=type(exc).__name__+': '+str(exc),face=getattr(item,'face_id',None)))
            for f in fields:
                v=getattr(item,f.name)
                if f.metadata.get('definition') is False and v is not None:
                    warm[name+'.'+f.name]+=1
                    if hasattr(v,'__len__') and len(v):warm_nonempty[name+'.'+f.name]+=1
                pending.append(v)
original_pair=bi._pair_paths
original_face=bi.arrange_material
def pair(*args,**kwargs):
    value=original_pair(*args,**kwargs);calls['pair']+=1;census(value)
    if not retained:
        try:_check_model_record_state(vars(args[2]));reason=None
        except QualificationDecline as exc:reason=str(exc)
        retained.append(dict(first_pair_ids=[args[0].face_id,args[1].face_id],post_pair_model_decline=reason))
    return value
def face(*args,**kwargs):
    value=original_face(*args,**kwargs);calls['face']+=1;census(value);return value
bi._pair_paths=pair;bi.arrange_material=face
os.environ['ANYGEOMETRY_PLANNING_MODE']='exact';os.environ['ANYGEOMETRY_PLANNING_WORKERS']='0';os.environ.pop('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING',None)
sys.argv=[str(ROOT/'redesign/s0f_apply_probe.py'),str(ROOT/'redesign/s0b_apply.py'),'--subset','elbow_sector60']
runpy.run_path(sys.argv[0],run_name='__main__')
result=json.loads((out/'s0b.json').read_text());assert result['plan_checksum']=='1078d6bf9c5b9c33f84e0d68954fbf89f52e677368b23c5f7641b8c9323df153' and result['applied']
report=dict(calls=dict(calls),records=dict(records),extras=dict(extras),warm=dict(warm),retained=retained,identity=result['plan_checksum'],constructor_rebuild=dict(rebuild),constructor_failures=failures,warm_nonempty=dict(warm_nonempty))
(out/'census.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(calls=dict(calls),record_classes=len(records),constructor_failures=len(failures),first_failures=failures[:5],warm_nonempty=dict(warm_nonempty))))
