"""Offline 0.4.3 build/Twine/installed-wheel qualification; requires a resource lease.

Artifacts stay in a fresh external TEMP child. Only its verified venv child is
removed. No global package installation, network, or publication is performed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    temp_root = Path(tempfile.gettempdir()).resolve()
    root = Path(tempfile.mkdtemp(prefix="anygeometry-043-qualification-", dir=temp_root)).resolve()
    if root.parent != temp_root or root.is_relative_to(repository):
        raise RuntimeError("qualification root must be a fresh external TEMP child")
    environment = root / "venv"
    artifacts = root / "dist"
    print("QUALIFICATION_ROOT", root, flush=True)
    report: dict[str, object] = {
        "repository": str(repository), "root": str(root), "version": "0.4.3",
        "tools": {name: importlib.metadata.version(name)
                  for name in ("build", "wheel", "twine", "numpy", "shapely", "setuptools")},
        "result": "FAIL", "venv_removed": False,
    }
    print("TOOL_VERSIONS", json.dumps(report["tools"], sort_keys=True), flush=True)
    clean_env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        clean_env.pop(key, None)
    clean_env["PYTHONNOUSERSITE"] = "1"
    clean_env["PIP_NO_INDEX"] = "1"
    clean_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    def run(args, *, cwd=repository, input_text=None):
        print("COMMAND", args, flush=True)
        result = subprocess.run(args, cwd=cwd, env=clean_env, input=input_text,
                                text=True, capture_output=True)
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        result.check_returncode()
        return result.stdout

    try:
        run([sys.executable, "-B", "-m", "build", "--no-isolation", "--outdir", str(artifacts)])
        wheels, sources = sorted(artifacts.glob("*.whl")), sorted(artifacts.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sources) != 1 or "0.4.3" not in wheels[0].name or "0.4.3" not in sources[0].name:
            raise RuntimeError("expected exactly the new 0.4.3 wheel and source archive")
        report["artifacts"] = [
            {"path": str(p), "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in (*wheels, *sources)
        ]
        run([sys.executable, "-B", "-m", "twine", "check", "--strict", *(str(p) for p in (*wheels, *sources))])
        if environment.exists() or environment.resolve().parent != root:
            raise RuntimeError("venv child must be absent before creation")
        # System NumPy is reused offline, but kernel origin MUST be in this venv.
        venv.EnvBuilder(with_pip=True, system_site_packages=True, symlinks=False).create(environment)
        executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run([str(executable), "-B", "-m", "pip", "install", "--no-index", "--no-deps",
             "--ignore-installed", str(wheels[0])], cwd=root)
        clean_env["QUALIFICATION_ENV"] = str(environment)
        clean_env["QUALIFICATION_REPO"] = str(repository)
        smoke = '''
import importlib.abc, os, sys
from pathlib import Path
import anygeometry
from anygeometry.serialization import VERSION
environment = Path(os.environ["QUALIFICATION_ENV"]).resolve()
repository = Path(os.environ["QUALIFICATION_REPO"]).resolve()
module = Path(anygeometry.__file__).resolve()
for path in (Path(sys.executable).resolve(), Path(sys.prefix).resolve(), module):
    assert path.is_relative_to(environment), (path, environment)
    assert not path.is_relative_to(repository), path
assert anygeometry.__version__ == "0.4.3"
assert VERSION == 4
assert (module.parent / "py.typed").is_file()
print("EXECUTABLE", sys.executable)
print("PREFIX", sys.prefix)
print("MODULE", module)
print("VERSION", anygeometry.__version__, "SCHEMA", VERSION, "PY_TYPED", True)
g = anygeometry.GeometryModel()
a = g.add_plate(g.add_points(((0,0,0),(2,0,0),(2,1,0),(0,1,0))))
b = g.add_plate(g.add_points(((1,0,0),(3,0,0),(3,1,0),(1,1,0))))
plan = anygeometry.plan_coplanar_fragmentation(g, (a,b), ownership_policy="first_selected")
result = anygeometry.apply_coplanar_fragmentation(g, plan)
assert len(g.faces) == 3 and result.overlap_area == 1
assert g.validate_topology() == ()
# A fresh module import denial verifies the optional-extra error path without
# changing ambient installed dependencies.
for key in tuple(sys.modules):
    if key == "shapely" or key.startswith("shapely."):
        del sys.modules[key]
class NoShapely(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "shapely" or fullname.startswith("shapely."):
            raise ImportError("isolated optional-absence probe")
sys.meta_path.insert(0, NoShapely())
from anygeometry.overlaps import _shapely
try:
    _shapely()
except anygeometry.GeometryError as error:
    assert "planar" in str(error)
else:
    raise AssertionError("missing planar extra did not fail closed")
print("OVERLAP_AND_OPTIONAL_ABSENCE_SMOKE_PASS")
'''
        report["smoke"] = run([str(executable), "-B", "-"], cwd=root, input_text=smoke)
        report["cli"] = run([str(executable), "-B", "-m", "anygeometry", "--version"], cwd=root).strip()
        if report["cli"] != "0.4.3":
            raise RuntimeError("unexpected installed CLI version")
        report["result"] = "PASS"
    finally:
        if environment.exists():
            resolved = environment.resolve()
            if resolved.parent != root or resolved.name != "venv" or environment.is_symlink():
                raise RuntimeError("refusing cleanup outside the exact venv child")
            shutil.rmtree(resolved)
        report["venv_removed"] = not environment.exists()
        (root / "qualification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("QUALIFICATION_REPORT", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
