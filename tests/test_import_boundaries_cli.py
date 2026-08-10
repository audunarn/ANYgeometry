"""Dependency direction, typed packaging, public imports, and module CLI."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import anygeometry
from anygeometry import EntityRef, GeometryModel
from anygeometry.__main__ import main


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
PACKAGE = SOURCE / "anygeometry"
FORBIDDEN = {
    "anyfem",
    "anymesher",
    "anysolver",
    "anystruct",
    "anytk3d",
    "gmsh",
    "meshio",
    "scipy",
    "tkinter",
}


def test_public_owner_exports_use_one_geometry_and_reference_type() -> None:
    geometry = GeometryModel()
    vertex = geometry.add_point(0.0, 0.0, 0.0)

    assert anygeometry.GeometryModel is GeometryModel
    assert anygeometry.EntityRef is EntityRef
    assert geometry.entity_ref("vertex", vertex).__class__ is EntityRef
    assert anygeometry.__version__ == "0.1.0"
    assert set(anygeometry.__all__) >= {
        "GeometryModel",
        "EntityRef",
        "Plane",
        "Cylinder",
        "Cone",
        "intersect_faces",
        "read_geometry",
        "write_geometry",
    }


def test_source_import_graph_has_no_forbidden_or_undeclared_dependency() -> None:
    external_imports: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                external_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                external_imports.add(node.module.split(".", 1)[0])

    assert not (external_imports & FORBIDDEN)
    assert external_imports <= set(sys.stdlib_module_names) | {"numpy", "shapely"}


def test_core_and_optional_dependencies_match_release_metadata() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert project["dependencies"] == ["numpy>=1.26"]
    assert project["optional-dependencies"]["planar"] == ["shapely>=2.0"]
    assert project["scripts"] == {"anygeometry": "anygeometry.__main__:main"}
    assert (PACKAGE / "py.typed").is_file()
    assert project["readme"] == "README.md"


def test_fresh_import_does_not_load_consumers_gui_mesh_or_solver(tmp_path) -> None:
    script = (
        "import json,sys;"
        f"sys.path.insert(0,{str(SOURCE)!r});"
        "import anygeometry;"
        f"forbidden={sorted(FORBIDDEN)!r};"
        "print(json.dumps(sorted(name for name in forbidden if name in sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_cli_writes_and_inspects_example(tmp_path, capsys) -> None:
    target = tmp_path / "example.anygeometry.json"

    assert main(["--write-example", str(target)]) == 0
    assert target.is_file()
    assert "Wrote" in capsys.readouterr().out

    assert main([str(target), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["topology"] == "valid"
    assert summary["faces"] == 6
    assert summary["groups"]["shell"] == 6
    assert summary["groups"]["longitudinal_stiffeners"] == 4


def test_module_main_and_version_work_from_outside_checkout(tmp_path) -> None:
    environment = os.environ.copy()
    command = (
        "import runpy,sys;"
        f"sys.path.insert(0,{str(SOURCE)!r});"
        "sys.argv=['anygeometry','--version'];"
        "runpy.run_module('anygeometry',run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", command],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == anygeometry.__version__
