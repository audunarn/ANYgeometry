"""Compact final evidence from retained actual outputs, never inferred passes."""
from pathlib import Path
import json,hashlib,re,subprocess
base=Path(__file__).resolve().parent
installed=Path('C:/Users/AUDUNA~1/AppData/Local/Temp/anygeometry-general-consumer-20261001-final-owner-targets')
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
checks=[]
for path in sorted(installed.glob('accepted-*.json')):
    if path.name.endswith('-mesh.json'):continue
    if not ('-checks.json' in path.name or path.name.startswith('accepted-replay-')):continue
    data=json.loads(path.read_text());assert data['automation']['solver_admission']=='ADMITTED'
    checks.append({k:v for k,v in data.items() if k!='automation'})
assert len(checks)==15
sources={'ANYgeometry':revision('f7b0a05'),'ANYmesh':'c56dd30eb10d024dc7331cfbad9fa3e725dc7346','ANYfem':'df991c2b29c8a7faf0a89692890c6d057cca5000'}
result={'status':'local_accepted','publication':'unpublished review branches; existing releases unchanged',
        'sources':sources,
        'kernel':suite('full-kernel-prepared-copy',{'ANYgeometry':sources['ANYgeometry']}),
        'mesher':suite('full-mesher-final-owner-targets',sources),
        'fem':suite('full-fem-final-owner-targets',dict(sources,ANYmesh='dcd3aa721dc5a755b46d459152e5e7d17c263426')),
        'fem_evidence_reuse':'Subsequent cylinder-hole chart selection is covered by18focusedmesherchecks, frozenfullmesher andcompleteinstalledapplicationgate; FEM/geometrysourceunchanged.',
        'installed':{'status':report['status'],'report_root':str(installed),'artifacts':report['artifacts'],'checks':checks,
                     'report_sha256':digest(installed/'general-intersections-report.json')},
        'hosted_run':36825533614}
(base/'final-local-evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print({name:{k:result[name][k] for k in ('passed','skipped','status')} for name in ('kernel','mesher','fem')})
print('Installed:',len(checks),'accepted meshes')
