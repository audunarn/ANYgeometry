"""Bounded sector60/r7 serial or owned-stage control; never whole-model planning."""
import hashlib,json,os,pathlib,sys,time
import psutil
W=pathlib.Path(__file__).resolve().parents[1];ROOT=W.parents[1]
case,mode,*references=sys.argv[1:]
assert case in ('elbow_sector60','elbow_r7') and mode in ('serial','parallel')
os.environ['ANYGEOMETRY_PLANNING_MODE']='exact';os.environ['ANYGEOMETRY_PLANNING_WORKERS']='0';os.environ.pop('ANYGEOMETRY_SEAM_POLICY',None)
os.environ['ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING']='1' if mode=='parallel' else '0'
sys.path.insert(0,str(ROOT/'resume'));import env_setup
original=env_setup.repo_root;env_setup.repo_root=lambda name: W if name=='ANYgeometry' else (ROOT/'int/ANYmesh' if name=='ANYmesh' else original(name));env_setup.setup()
from anyfem.io import load_project
from anygeometry import batch_intersections as bi
from anygeometry.serialization import from_dict,to_dict
from anygeometry._planning_host import StageControl
subsets={}
for p in sorted((ROOT/'redesign/runs').glob('R11*-ladder-*/ladder_subsets.json')):subsets.update(json.loads(p.read_text()))
faces=subsets[case];assert len(faces)==(97 if case=='elbow_sector60' else 240)
source=pathlib.Path('C:/ANYworkspaceWork/AIP Gen2/ANYfem-local-assessment/hydrodynamics/AIP-Gen2-full-tripod-no-tower.anyfem')
source_sha=hashlib.sha256(source.read_bytes()).hexdigest();assert source_sha=='6943e22988b7d1767f77c9cf49ad4c017cfc82cc1cdf5d465605fb4140299dc0'
g=load_project(source).geometry;doc=to_dict(g)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
initial=digest(doc);out=pathlib.Path(os.environ['TASK_EVIDENCE_DIRECTORY']);policy=bi.IntersectionBatchPolicy();ci_seconds=[]
report=dict(case=case,requested_mode=mode,faces=len(faces),source_sha256=source_sha,model_document_sha256=initial,source_model_id=str(g.model_id),load_before=dict(python_process_count=sum(1 for p in psutil.process_iter(['name'],ad_value='') if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe','python','python3')),cpu_percent=psutil.cpu_percent(interval=.1),logical_cpus=os.cpu_count()))
control=None
if mode=='parallel':
    reference=json.loads(pathlib.Path(references[0]).read_text());assert reference['case']==case and reference['model_document_sha256']==initial
    def observed(plan):
        start=time.perf_counter();shadow=from_dict(doc);bi.apply_intersections(shadow,plan,policy=policy)
        result=(plan.content_checksum,to_dict(shadow)['checksum']['value']);ci_seconds.append(time.perf_counter()-start);return result
    control=StageControl(out/'pool',expected_ci=(reference['plan_checksum'],reference['apply_checksum']),observed_ci=observed,workers=8)
try:
    start=time.perf_counter();plan=bi.plan_intersections(g,tuple(g.handle('face',f) for f in faces),policy=policy,_parallel_control=control)
    report['plan_seconds']=time.perf_counter()-start;report['plan_checksum']=plan.content_checksum
    report['source_unchanged']=digest(to_dict(g))==initial;assert report['source_unchanged']
    report['mode']=control.mode if control else 'serial';report['decline_reason']=control.decline_reason if control else None
    if control:assert control.mode=='parallel',control.decline_reason
    start=time.perf_counter();bi.apply_intersections(g,plan,policy=policy);report['apply_seconds']=time.perf_counter()-start
    report['apply_checksum']=to_dict(g)['checksum']['value'];report['applied']=True
    report['arrangements']=len(plan.arrangements);report['face_contacts']=len(plan.face_contacts)
    expected={'elbow_sector60':'1078d6bf9c5b9c33f84e0d68954fbf89f52e677368b23c5f7641b8c9323df153','elbow_r7':'160166d64deb7b576e6c18bc5fa4dcea5e394b5fc1de896890629b183b1e0e70'}
    assert plan.content_checksum==expected[case]
    if mode=='parallel':assert report['apply_checksum']==reference['apply_checksum']
    assert not env_setup.foreign_modules()
    report['status']='passed'
except Exception as exc:
    report.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
finally:
    report['ci_apply_seconds']=ci_seconds
    if control:
        report.update(ci_verified=control.ci_verified,decline_reason=control.decline_reason)
        (out/'stage-reports.json').write_text(json.dumps(control.reports,indent=2),encoding='utf-8')
        report['stages']=[]
        for stage in control.reports:
            times=[sum(row['task_seconds']) for row in stage['worker_reports']]
            report['stages'].append(dict(stage=stage['stage'],workers=stage['actual_workers'],tasks=sum(row['task_count'] for row in stage['worker_reports']),samples=len(stage['sampling']['indices']),worker_task_seconds=times,max_over_mean=max(times)/(sum(times)/len(times)),worker_startup_seconds=[row['startup_seconds'] for row in stage['worker_reports']],worker_boundary_seconds=[sum(row['boundary_seconds']) for row in stage['worker_reports']],host_timing=stage['host_timing']))
    (out/'reference.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
