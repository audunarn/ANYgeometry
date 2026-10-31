"""Real subprocess startup refusal against actual two-thread BLAS pools."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import scipy.linalg
from threadpoolctl import threadpool_info, threadpool_limits
from anygeometry._planning_child import prepare
from anygeometry._planning_qualification import QualificationDecline

with threadpool_limits(limits=2):
    counts=[p['num_threads'] for p in threadpool_info()]
    assert counts and all(value==2 for value in counts)
    try:
        prepare()
    except QualificationDecline as exc:
        assert 'thread count' in str(exc)
        print(json.dumps(dict(actual_native_threads=counts,refused=True,canaries_started=False)))
    else:
        raise AssertionError('wrong-thread real startup was admitted')
