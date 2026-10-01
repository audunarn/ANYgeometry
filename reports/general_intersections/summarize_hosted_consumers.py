"""Compact retained evidence; complete artifacts remain attached to the run."""
import json
import sys
from pathlib import Path

base=Path(__file__).parent
run_id=int(sys.argv[1]) if len(sys.argv)>1 else 36795873823
run=base/f'hosted-run-{run_id}'
status_path=base/f'hosted-{run_id}-status.json'
status=json.loads(status_path.read_text()) if status_path.exists() else None
summary={'run_id':run_id,'url':f'https://github.com/audunarn/ANYgeometry/actions/runs/{run_id}',
         'overall_status': ('passed' if status['conclusion']=='success' else status['conclusion']) if status else 'unavailable',
         'failed_jobs': [item['name'] for item in status['jobs'] if item['conclusion']!='success'] if status else [],
         'installed_general_consumers':[]}
for platform in sorted(run.iterdir()):
    report_path=next(platform.rglob('general-intersections-report.json'))
    report=json.loads(report_path.read_text())
    root=report_path.parent
    sources=json.loads(next(platform.rglob('source-commits.json')).read_text())
    cases=[]
    for path in sorted(root.glob('accepted-replay-*.json')):
        case=json.loads(path.read_text())
        assert case['automation']['status']=='ready' and case['automation']['solver_admission']=='ADMITTED'
        assert case['source_unchanged'] and case['source_save_load'] and case['mesh_codec_roundtrip']
        cases.append({'case':'replay','target_size':case['target_size'],'nodes':case['nodes'],
                      'triangles':case['triangles'],'quads':case['quads'],'solver_admission':'ADMITTED',
                      'source_unchanged':True,'save_load':True,'lifecycle':case['lifecycle']})
    for path in sorted(root.glob('accepted-cylinders-*-checks.json')):
        case=json.loads(path.read_text())
        assert case['automation']['status']=='ready' and case['automation']['solver_admission']=='ADMITTED'
        cases.append({key:value for key,value in case.items() if key!='automation'})
    for path in sorted(root.glob('accepted-material-*-checks.json')):
        case=json.loads(path.read_text())
        assert case['automation']['status']=='ready' and case['automation']['solver_admission']=='ADMITTED'
        assert case['source_unchanged'] and case['save_load']
        cases.append({key:value for key,value in case.items() if key!='automation'})
    expected=len(report['cases'])+(2 if 'growing' in report['cases'] else 0)+(2 if 'cylinder-holes' in report['cases'] else 0)
    if report['status']=='passed':
        assert len(cases)==expected
    failed_commands=[{'args':item['args'],'log':item['log'],'timed_out':item.get('timed_out',False),
                      'exit_code':item.get('exit_code')} for item in report['commands']
                     if item.get('timed_out') or item.get('exit_code',0)!=0]
    summary['installed_general_consumers'].append({'platform':platform.name,
        'machine':report['machine'],'status':report['status'],'sources':sources,
        'artifacts':report['artifacts'],'cases':cases,'expected_meshes':expected,
        'unaccepted_meshes':expected-len(cases),'failed_commands':failed_commands})
(base/f'installed-qualification-{run_id}.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print([{ 'platform':item['platform'],'status':item['status'],'cases':len(item['cases'])}
       for item in summary['installed_general_consumers']])
