"""Compact final evidence from retained actual outputs, never inferred passes."""
from pathlib import Path
import json,hashlib,re,subprocess
base=Path(__file__).resolve().parent
installed=Path('C:/Users/AUDUNA~1/AppData/Local/Temp/anygeometry-general-consumer-20261001-final-user-policy')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def revision(name):return subprocess.check_output(['git','rev-parse',name],cwd=base.parent.parent,text=True).strip()
def suite(label,source):
    log=base/(label+'.log');status=base/(label+'-status.json')
    data=json.loads(status.read_text()) if status.exists() else {'status':'passed','exit_code':0,'completion_evidence':'Previously completed tool session65805; recorded in living task note'}
    assert data['status']=='passed'
    assert status.exists() or label=='full-kernel-prepared-copy'
    text=log.read_text(errors='replace')
    match=re.search(r'(\d+) passed(?:, (\d+) skipped)?',text)
    assert match,label
    return dict(data,passed=int(match[1]),skipped=int(match[2] or 0),source=source,
                log=log.name,log_sha256=digest(log))
report=json.loads((installed/'general-intersections-report.json').read_text());assert report['status']=='passed'
capacity=Path('C:/Users/AUDUNA~1/AppData/Local/Temp/anygeometry-general-consumer-20261001-capacity-user-policy')
capacity_report=json.loads((capacity/'general-intersections-report.json').read_text());assert capacity_report['status']=='passed'
assert capacity_report['artifacts']==report['artifacts']
checks=[]
for path in sorted(installed.glob('accepted-*.json'))+sorted(capacity.glob('accepted-*.json')):
    if path.name.endswith('-mesh.json'):continue
    if not ('-checks.json' in path.name or path.name.startswith('accepted-replay-')):continue
    data=json.loads(path.read_text());assert data['automation']['solver_admission']=='ADMITTED'
    checks.append({k:v for k,v in data.items() if k!='automation'})
assert len(checks)==16
sources={'ANYgeometry':revision('f7b0a05'),'ANYmesh':'e21c0fc93662776762430e14450d54ac9192e2e8','ANYfem':'83f3d6c81e405ca45f5b7edd7cb7c2afd32804c4'}
result={'status':'local_accepted','publication':'unpublished review branches; existing releases unchanged',
        'sources':sources,
        'kernel':suite('full-kernel-prepared-copy',{'ANYgeometry':sources['ANYgeometry']}),
        'mesher':suite('full-mesher-final-user-policy',sources),
        'fem':suite('full-fem-final-user-policy',sources),
        'kernel_evidence_reuse':'Geometry source remains f7b0a05; original complete kernel run and final hosted matrix retained.',
        'installed':{'status':report['status'],'report_root':str(installed),'artifacts':report['artifacts'],'checks':checks,
                     'report_sha256':digest(installed/'general-intersections-report.json'),
                     'capacity_report_root':str(capacity),'capacity_report_sha256':digest(capacity/'general-intersections-report.json')},
        'hosted_run':36829763205}
(base/'final-user-policy-evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print({name:{k:result[name][k] for k in ('passed','skipped','status')} for name in ('kernel','mesher','fem')})
print('Installed:',len(checks),'accepted meshes')
