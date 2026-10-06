import json, os, sys
from pathlib import Path
import numpy, pytest, anygeometry
Path(os.environ['PROOF_RUNTIME']).write_text(json.dumps(dict(python=sys.version, executable=sys.executable, numpy=numpy.__version__, pytest=pytest.__version__, geometry_origin=anygeometry.__file__, threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})))
assert Path(anygeometry.__file__).resolve().is_relative_to(Path.cwd()/'src')
raise SystemExit(pytest.main(['tests/test_polynomial_extrusion_support.py','-q','-k','not public and not aggregate and not explicit_sampled','--junitxml='+os.environ['PROOF_XML']]))
