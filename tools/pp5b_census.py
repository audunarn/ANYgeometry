"""PP5b real census and exact constructor-recomputation experiment; no planning."""
import collections, dataclasses, hashlib, json, os, pathlib, sys
import numpy as np
REPO=pathlib.Path(__file__).resolve().parents[1]; ROOT=REPO.parents[1]
sys.path.insert(0,str(ROOT/'resume'))
import env_setup
original=env_setup.repo_root
env_setup.repo_root=lambda name: REPO if name=='ANYgeometry' else (ROOT/'int/ANYmesh' if name=='ANYmesh' else original(name))
env_setup.setup()
from anyfem.io import load_project
from anygeometry.closure import extract_model_closure

def array_record(array):
    return dict(dtype=array.dtype.str,shape=list(array.shape),c_contiguous=bool(array.flags.c_contiguous),writeable=bool(array.flags.writeable),finite=bool(np.all(np.isfinite(array))),bytes_hex=array.tobytes(order='C').hex())

def census(model):
    pending=[(vars(model),'model')];seen=set();counts=collections.Counter();mismatches=[];matches=0;classes={}
    while pending:
        value,path=pending.pop()
        if id(value) in seen: continue
        seen.add(id(value))
        if type(value) is dict:
            pending.extend((v,path+'['+repr(k)+']') for k,v in value.items())
        elif type(value) in (tuple,list,set,frozenset): pending.extend((v,path+'[]') for v in value)
        elif not isinstance(value,type) and dataclasses.is_dataclass(value):
            fields=dataclasses.fields(value);extras=set(getattr(value,'__dict__',{}))-{f.name for f in fields}
            if extras:
                name=type(value).__module__+'.'+type(value).__qualname__
                classes[name]=sorted(extras)
                reconstructed=type(value)(**{f.name:getattr(value,f.name) for f in fields if f.init})
                for key in sorted(extras):
                    counts[(name,key)]+=1
                    left,right=getattr(value,key),getattr(reconstructed,key)
                    if type(left) is not np.ndarray or type(right) is not np.ndarray: raise TypeError((name,key,type(left),type(right)))
                    a,b=array_record(left),array_record(right)
                    equality=all(a[k]==b[k] for k in ('dtype','shape','c_contiguous','finite','bytes_hex')) and (a['writeable'] or not b['writeable'])
                    if equality: matches+=1
                    else:
                        declared_changes=[]
                        for f in fields:
                            va,vb=getattr(value,f.name),getattr(reconstructed,f.name)
                            if type(va) is np.ndarray and type(vb) is np.ndarray and array_record(va)!=array_record(vb): declared_changes.append(f.name)
                        mismatches.append(dict(path=path,record_class=name,attribute=key,actual=a,recomputed=b,declared_array_changes=declared_changes))
            pending.extend((getattr(value,f.name),path+'.'+f.name) for f in fields)
    return dict(faces=len(model.faces),classes=classes,census=[dict(record_class=k[0],attribute=k[1],count=v) for k,v in sorted(counts.items())],matches=matches,mismatches=mismatches)

source=pathlib.Path('C:/ANYworkspaceWork/AIP Gen2/ANYfem-local-assessment/hydrodynamics/AIP-Gen2-full-tripod-no-tower.anyfem')
p=load_project(source)
record=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),subset_source=census(p.geometry))
source=ROOT/'model-n2-20261030/AIP-Gen2-full-tripod-no-tower.MODEL-N2c-rotated-arms.SIMPLIFIED-EXERCISE.anyfem'
record['N2c_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
assert record['N2c_sha256']=='969cc5f9c8efe5b628686c9a89b1de02ea2dea8c1618f4295cb049c2621f468a'
p=load_project(source);g=p.geometry
record['N2c_source']=census(g)
members={m for e in p.beam_edges for m in g.members_using_edge(e)}
handles=tuple([g.handle('face',i) for i in sorted(g.faces)]+[g.handle('edge',i) for i in p.beam_edges]+[g.handle('sheet',i) for i in sorted(g.sheets)]+[g.handle('member',i) for i in sorted(members)])
closure=extract_model_closure(g,handles,include_structural_closure=True,include_features=False)
assert len(closure.working_model.faces)==561
record['N2c_closure']=census(closure.working_model)
assert not env_setup.foreign_modules()
path=pathlib.Path(os.environ['TASK_EVIDENCE_DIRECTORY'])/'census.json'
path.write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({k:dict(faces=v['faces'],census=v['census'],matches=v['matches'],mismatches=len(v['mismatches'])) for k,v in record.items() if type(v)is dict}))