"""Record observed stores; no cache clears or numerical planning."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from anygeometry import branch_algebra, branch_curves
from anygeometry import _planning_purity as p

modules = tuple(m for name, m in sorted(sys.modules.items())
                if name == 'anygeometry' or name.startswith('anygeometry.'))
report = p.inventory(modules)
result = asdict(report)
result['modules'] = [m.__name__ for m in modules]
for label, check in [('purity', lambda: p.require_pure(report)),
                     ('code', lambda: p.python_token(modules))]:
    try:
        check()
    except p.PurityDecline as exc:
        result[label] = dict(status='declined', reason=str(exc))
    else:
        raise AssertionError('unexpected numerical admission')
out = Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
(out / 'registry.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(dict(modules=len(modules), stores=len(report.stores), unknowns=len(report.unknowns),
                     aliases=len(report.aliases), truncated=report.truncated,
                     purity=result['purity'], code=result['code'])))
