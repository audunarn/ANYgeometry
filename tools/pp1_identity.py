"""Original subset observers, pinned to this worktree. Never a whole-model plan."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import types

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parents[1]
case = sys.argv[1]
assert case in ('elbow_sector60', 'elbow_r7', 'face200')
for key in tuple(os.environ):
    if key.startswith('RESUME_'):
        del os.environ[key]
os.environ.update(ANYGEOMETRY_PLANNING_MODE='exact', ANYGEOMETRY_PLANNING_WORKERS='0')
os.environ.pop('ANYGEOMETRY_SEAM_POLICY', None)
sys.path.insert(0, str(ROOT / 'resume'))
import env_setup
original_root = env_setup.repo_root
env_setup.repo_root = lambda name: REPO if name == 'ANYgeometry' else (
    ROOT / 'int/ANYmesh' if name == 'ANYmesh' else original_root(name))
origins = env_setup.setup()
import anygeometry
assert Path(anygeometry.__file__).resolve().is_relative_to(REPO)
out = Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
(out / 'runtime.json').write_text(json.dumps(dict(case=case, origins=origins,
    mode=os.environ['ANYGEOMETRY_PLANNING_MODE'], workers=os.environ['ANYGEOMETRY_PLANNING_WORKERS'],
    seam_override=os.environ.get('ANYGEOMETRY_SEAM_POLICY')), indent=2))
if case == 'face200':
    sys.modules['run_gui'] = types.ModuleType('run_gui')
    sys.argv = [str(ROOT / 'resume/face200_check.py')]
    runpy.run_path(sys.argv[0], run_name='__main__')
    result = json.loads((out / 'face200-summary.json').read_text())
    stable = {k: v for k, v in result.items() if k not in ('origins', 'arrangement_seconds', 'seconds')}
    digest = hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == '2ab517de9f4bc32af5f0aada85b234084282f982c313a73224f5bedda400c51f', digest
    assert result['area'] == 4.218529952235817
    identity = dict(canonical_sha256=digest, area=result['area'])
else:
    sys.argv = [str(ROOT / 'redesign/s0f_apply_probe.py'), str(ROOT / 'redesign/s0b_apply.py'), '--subset', case]
    runpy.run_path(sys.argv[0], run_name='__main__')
    result = json.loads((out / 's0b.json').read_text())
    expected = dict(elbow_sector60='1078d6bf9c5b9c33f84e0d68954fbf89f52e677368b23c5f7641b8c9323df153',
                    elbow_r7='160166d64deb7b576e6c18bc5fa4dcea5e394b5fc1de896890629b183b1e0e70')[case]
    assert result.get('applied') and result.get('plan_checksum') == expected, result.get('error')
    identity = dict(plan_checksum=expected, applied=True, faces=result['faces'])
assert not env_setup.foreign_modules(), env_setup.foreign_modules()
assert 'anygeometry._planning_purity' not in sys.modules, 'off path imported new module'
(out / 'identity.json').write_text(json.dumps(identity, indent=2))
print(json.dumps(identity))
