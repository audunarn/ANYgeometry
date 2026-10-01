"""Record observed delivery state without promoting pending consumer gates."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
def load(name):
    return json.loads((root/name).read_text(encoding='utf-8-sig'))
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name, value):
    (root/name).write_bytes((json.dumps(value, indent=2, sort_keys=True)+'\n').encode())
def compact(status):
    return {k: v for k, v in status.items() if k != 'jobs'} | {
        'jobs': [{k: v for k, v in job.items() if k != 'steps'} for job in status['jobs']]}

q = load('qualification.json')
review = load('independent_review.json')
ledger = json.loads((root.parent.parent/'docs/release/anygeometry-0.4.5-ledger.json').read_text())
assert digest(root/'qualification.json').upper() == ledger['qualification']['evidence_sha256']
assert digest(root/'independent_review.json').upper() == ledger['qualification']['independent_review_sha256']
assert review['status'] == 'passed' and q['accepted_terminal'] == 'ACCEPTED_ANYGEOMETRY_0_4_5_RELEASE'
summary_path = root.parent/'general_intersections/installed-qualification-36834513207.json'
summary = json.loads(summary_path.read_text())
assert digest(summary_path).upper() == q['installed_evidence']['sha256']
status = {owner: compact(load(owner+'-ci.json')) for owner in ('geometry','mesher','fem','publish')}
assert status['geometry']['conclusion'] == 'success'
receipt = load('pypi-receipt.json')
assert receipt['version'] == '0.4.5' and receipt['status'] == 'verified'
assert status['publish']['status'] == 'completed' and status['publish']['conclusion'] == 'success'
assert receipt['artifacts'] == q['artifacts']
for owner, record in status.items():
    write(owner+'-ci-summary.json', record)
logs = {}
for name in ('fem-linux-313-failure.log','fem-main-603-baseline-failed.log',
             'mesher-windows-311-failure.log','mesher-pin-contract.log','authority-verification.json'):
    path = root/name
    assert path.is_file()
    logs[name] = {'bytes':path.stat().st_size,'sha256':digest(path),'availability':'retained locally; hosted source run links in this record'}
delivery = {
    'schema':'anygeometry.general-intersection-delivery-v1',
    'status':'geometry published; consumer owner gates reported separately',
    'geometry_version':'0.4.5',
    'geometry_artifact_source':q['artifact_source'],
    'geometry_ledger_tag_commit':'254cdc33ec1ae4b07c57b62791eb461103e7ff90',
    'geometry_release_ledger_main':'51843d14998993e43ffcfc57a4d2672cda452989',
    'integration_commits':q['integration_commits'],
    'qualified_consumer_inputs':summary['installed_general_consumers'][0]['sources'],
    'ci_candidate_artifacts':q['ci_candidate_artifacts'],
    'published_artifacts':q['artifacts'],
    'installed_platform_artifacts':[{k:p[k] for k in ('platform','machine','sources','artifacts','status')} for p in summary['installed_general_consumers']],
    'hosted_status':status,
    'accepted_installed_native_meshes':48,
    'windows_skew_seconds':632.4844599,
    'command_timeout_seconds':900,
    'sector_probe':{'sectors':260,'shared_panel_interfaces':260,'material_area':'2*pi','accepted_platforms':['Windows','Linux','macOS Apple silicon'],'historical_direct_scope_unchanged':True},
    'pypi':receipt,
    'evidence_binding':{'qualification_sha256':digest(root/'qualification.json'),'independent_review_sha256':digest(root/'independent_review.json'),'installed_summary_sha256':digest(summary_path)},
    'preserved_logs':logs,
    'fem_ci_disposition':q['fem_ci_disposition'],
    'mesher_ci_disposition':q['mesher_ci_disposition'],
    'limitations':q['limitations'],
    'publication_scope':'ANYgeometry only; no mesher/FEM publication, Qt default switch or Tk removal',
}
merged = root/'mesher-main-merge.json'
if merged.is_file():
    merge = load('mesher-main-merge.json')
    assert merge['merged'] is True
    assert status['mesher']['status'] == 'completed' and status['mesher']['conclusion'] == 'success'
    delivery['integration_commits']['ANYmesh_main'] = merge['sha']
    delivery['mesher_ci_disposition'] = 'Completed owner matrix passed; reviewed integration merged to main.'
write('delivery.json', delivery)
print(json.dumps({'status':delivery['status'],'installed_meshes':48,'source':q['artifact_source']['commit'],'owner_status':{k:v['conclusion'] or v['status'] for k,v in status.items()}}))
