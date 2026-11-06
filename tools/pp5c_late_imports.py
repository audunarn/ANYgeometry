"""Diagnose late imports on saved sector pair tasks; no acceptance bypass claim."""
import os,pathlib,sys,runpy,json
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
W=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src'))
child=runpy.run_path(str(W/'src/anygeometry/_planning_child.py'),run_name='pp5_child_diagnostic')
child['prepare']()
a=sys.modules['anygeometry._planning_executor'];module=sys.modules['anygeometry._planning_stages'];ledger_api=sys.modules['anygeometry._planning_ledger']
base=W/'.pp1-evidence/pp5c-sector60-parallel1/pool/pair'
rows=[]
for folder in sorted(base.glob('worker-*')):
    request=a.read_packet(folder/'request.json','assignment');context=module.WorkerModel(request['model'])
    for ordinal,task in request['tasks'][:1]:
        before=set(sys.modules)  # Import census only; NOT an admitted kernel execution.
        module.stage_task(context,task)
        added=sorted(set(sys.modules)-before)
        if added:rows.append(dict(worker=folder.name,ordinal=ordinal,added=added))
    context.verify()
out=pathlib.Path(os.environ['TASK_EVIDENCE_DIRECTORY']);(out/'late-imports.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
