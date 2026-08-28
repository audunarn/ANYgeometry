"""Dependency direction, typed packaging, public imports, and module CLI."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest

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
    assert anygeometry.__version__ == "0.4.1"
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
    assert project["version"] == "0.4.1"
    assert "Development Status :: 3 - Alpha" in project["classifiers"]


def test_manual_release_workflow_builds_without_production_publication() -> None:
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )

    assert workflow.startswith(
        "name: Publish to PyPI\n\non:\n  workflow_dispatch:\n"
    )
    assert "release:" not in workflow
    assert "sha256sum *.whl *.tar.gz > SHA256SUMS" in workflow
    assert "repository-url:" not in workflow
    assert "gh-action-pypi-publish" not in workflow
    assert "timeout-minutes:" not in workflow
    assert 'version != "0.4.1"' in workflow
    assert "python -m twine check --strict dist/*.whl dist/*.tar.gz" in workflow
    assert "ANYgeometry-${{ steps.contract.outputs.version }}-release-bundle" in workflow


RELEASE_VERIFIER = ROOT / "tools" / "verify_release_authority.py"
RELEASE_DISTRIBUTION = "ANYgeometry"
RELEASE_NORMALIZED = "anygeometry"
RELEASE_VERSION = "0.4.1"
RELEASE_TAG = f"v{RELEASE_VERSION}"
RELEASE_TERMINAL = "ACCEPTED_ANYGEOMETRY_0_4_1_RELEASE"
RELEASE_WHEEL = f"{RELEASE_NORMALIZED}-{RELEASE_VERSION}-py3-none-any.whl"
RELEASE_SDIST = f"{RELEASE_NORMALIZED}-{RELEASE_VERSION}.tar.gz"
RELEASE_LEDGER = (
    Path("docs/release")
    / f"{RELEASE_NORMALIZED}-{RELEASE_VERSION}-ledger.json"
)
CHECKOUT_ACTION = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_ACTION = "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
PUBLISH_ACTION = (
    "pypa/gh-action-pypi-publish@"
    "dc37677b2e1c63e2034f94d8a5b11f265b73ba33"
)


def _neutral_test_git_environment() -> dict[str, str]:
    """Give authority fixtures a clean Git surface under an outer guard."""

    environment = os.environ.copy()
    exact_names = {
        "GIT_ATTR_NOSYSTEM",
        "GIT_ATTR_SOURCE",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_SYSTEM",
        "GIT_EXTERNAL_DIFF",
        "GIT_GRAFT_FILE",
        "GIT_NO_LAZY_FETCH",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_OPTIONAL_LOCKS",
        "GIT_REPLACE_REF_BASE",
    }
    for name in tuple(environment):
        if (
            name in exact_names
            or name.startswith("GIT_CONFIG_KEY_")
            or name.startswith("GIT_CONFIG_VALUE_")
        ):
            environment.pop(name, None)
    return environment


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release Authority Test",
            "-c",
            "user.email=release-authority@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        env=_neutral_test_git_environment(),
    )
    return completed.stdout.strip()


def _release_metadata(
    distribution: str = RELEASE_DISTRIBUTION,
) -> bytes:
    return (
        "Metadata-Version: 2.1\n"
        f"Name: {distribution}\n"
        f"Version: {RELEASE_VERSION}\n\n"
    ).encode("utf-8")


def _write_release_wheel(
    path: Path,
    payload: bytes,
    *,
    distribution: str = RELEASE_DISTRIBUTION,
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{RELEASE_NORMALIZED}/__init__.py", payload)
        archive.writestr(
            f"{RELEASE_NORMALIZED}-{RELEASE_VERSION}.dist-info/METADATA",
            _release_metadata(distribution),
        )


def _write_release_sdist(path: Path) -> None:
    metadata = _release_metadata()
    info = tarfile.TarInfo(
        f"{RELEASE_NORMALIZED}-{RELEASE_VERSION}/PKG-INFO"
    )
    info.size = len(metadata)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(metadata))


def _write_release_checksums(assets: Path) -> None:
    text = "".join(
        f"{hashlib.sha256((assets / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted((RELEASE_WHEEL, RELEASE_SDIST))
    )
    (assets / "SHA256SUMS").write_text(
        text,
        encoding="ascii",
        newline="\n",
    )


def _run_release_verifier(
    tmp_path: Path,
    mutation: str = "",
) -> subprocess.CompletedProcess[str]:
    repository = tmp_path / "repository"
    remote = tmp_path / "origin.git"
    assets = tmp_path / "release-assets"
    repository.mkdir(parents=True)
    remote.mkdir()
    assets.mkdir()
    _git(repository, "init", "--quiet")
    _git(remote, "init", "--bare", "--quiet")
    (repository / "source.txt").write_text(
        "frozen artifact source\n",
        encoding="utf-8",
    )
    source_paths = ["source.txt"]
    if mutation == "textconv-diff-driver":
        (repository / ".gitattributes").write_text(
            "* diff=release-bypass\n",
            encoding="utf-8",
        )
        source_paths.append(".gitattributes")
    _git(repository, "add", *source_paths)
    _git(repository, "commit", "--quiet", "-m", "freeze artifact source")
    source_commit = _git(repository, "rev-parse", "HEAD")
    source_tree = _git(repository, "rev-parse", "HEAD^{tree}")
    _git(repository, "branch", "-M", "main")
    _git(repository, "remote", "add", "origin", str(remote))
    _git(repository, "push", "--quiet", "-u", "origin", "main")

    attribute_source_commit = ""
    if mutation == "git-attr-source":
        _git(repository, "checkout", "--quiet", "-b", "attack-attributes")
        (repository / ".gitattributes").write_text(
            "* diff=release-bypass\n",
            encoding="utf-8",
        )
        _git(repository, "add", ".gitattributes")
        _git(repository, "commit", "--quiet", "-m", "attacker attributes")
        attribute_source_commit = _git(repository, "rev-parse", "HEAD")
        _git(repository, "checkout", "--quiet", "main")

    _write_release_wheel(assets / RELEASE_WHEEL, b"accepted build\n")
    if mutation == "wrong-metadata":
        _write_release_wheel(
            assets / RELEASE_WHEEL,
            b"accepted build\n",
            distribution="DifferentDistribution",
        )
    _write_release_sdist(assets / RELEASE_SDIST)
    artifact_rows = []
    for name in sorted((RELEASE_WHEEL, RELEASE_SDIST)):
        raw = (assets / name).read_bytes()
        artifact_rows.append(
            {
                "bytes": len(raw),
                "filename": name,
                "sha256": hashlib.sha256(raw).hexdigest().upper(),
            }
        )
    ledger = {
        "artifact_source": {
            "commit": source_commit,
            "tree": source_tree,
        },
        "artifacts": artifact_rows,
        "distribution": RELEASE_DISTRIBUTION,
        "publication_authorized": True,
        "qualification": {
            "accepted_terminal": RELEASE_TERMINAL,
            "evidence_sha256": "A" * 64,
            "independent_review_sha256": "B" * 64,
        },
        "schema": "anyecosystem.release-ledger-v1",
        "tag": RELEASE_TAG,
        "version": RELEASE_VERSION,
    }
    if mutation == "wrong-byte-count":
        ledger["artifacts"][0]["bytes"] += 1
    elif mutation == "wrong-terminal":
        ledger["qualification"]["accepted_terminal"] = "REJECTED_RELEASE"
    elif mutation == "evidence-hash":
        ledger["qualification"]["evidence_sha256"] = "0" * 64
    elif mutation == "review-hash":
        ledger["qualification"]["independent_review_sha256"] = "A" * 64
    elif mutation == "noncanonical-tag-ref":
        ledger["tag"] = f"{RELEASE_TAG}^{{commit}}"
    if mutation == "wrong-source":
        ledger["artifact_source"]["tree"] = "0" * 40

    target = repository / RELEASE_LEDGER
    target.parent.mkdir(parents=True)
    if mutation == "noncanonical":
        target.write_text(json.dumps(ledger), encoding="utf-8")
    else:
        target.write_text(
            json.dumps(ledger, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    _git(repository, "add", RELEASE_LEDGER.as_posix())
    if mutation == "extra-child-path":
        (repository / "unexpected.txt").write_text(
            "not ledger-only\n",
            encoding="utf-8",
        )
        _git(repository, "add", "unexpected.txt")
    _git(
        repository,
        "commit",
        "--quiet",
        "-m",
        "docs: authorize release artifacts",
    )
    _git(repository, "tag", RELEASE_TAG)
    if mutation != "unmerged-tag-child":
        _git(repository, "push", "--quiet", "origin", "HEAD:main")

    git_directory = Path(_git(repository, "rev-parse", "--git-dir"))
    if not git_directory.is_absolute():
        git_directory = repository / git_directory
    git_info = git_directory / "info"
    git_info.mkdir(exist_ok=True)
    if mutation == "moved-tag-ref":
        _git(repository, "tag", "--force", RELEASE_TAG, source_commit)
    elif mutation == "missing-tag-ref":
        _git(repository, "tag", "--delete", RELEASE_TAG)
    elif mutation == "replacement-ref":
        _git(
            repository,
            "replace",
            source_commit,
            _git(repository, "rev-parse", "HEAD"),
        )
    elif mutation == "graft-file":
        (git_info / "grafts").write_text(
            _git(repository, "rev-parse", "HEAD") + "\n",
            encoding="ascii",
        )
    elif mutation == "info-attributes":
        (git_info / "attributes").write_text(
            "* diff=release-bypass\n",
            encoding="utf-8",
        )

    _write_release_checksums(assets)
    invoked_tag = (
        f"{RELEASE_TAG}^{{commit}}"
        if mutation == "noncanonical-tag-ref"
        else RELEASE_TAG
    )
    verifier_environment = _neutral_test_git_environment()
    attacker_marker = tmp_path / "attacker.marker"
    attacker = tmp_path / "attacker.py"
    attacker.write_text(
        "from pathlib import Path\n"
        f"Path({str(attacker_marker)!r}).write_text("
        "'invoked\\n', encoding='utf-8')\n"
        "raise SystemExit(97)\n",
        encoding="utf-8",
    )
    attacker_command = shlex.join((sys.executable, str(attacker)))
    external_attributes = tmp_path / "external.attributes"
    external_attributes.write_text(
        "* diff=release-bypass\n",
        encoding="utf-8",
    )
    external_config = tmp_path / "external.gitconfig"
    external_config.write_text("", encoding="utf-8")
    _git(
        repository,
        "config",
        "--file",
        str(external_config),
        "core.attributesFile",
        str(external_attributes),
    )
    _git(
        repository,
        "config",
        "--file",
        str(external_config),
        "diff.external",
        attacker_command,
    )
    _git(
        repository,
        "config",
        "--file",
        str(external_config),
        "diff.release-bypass.textconv",
        attacker_command,
    )
    assert (
        _git(
            repository,
            "config",
            "--file",
            str(external_config),
            "--get",
            "diff.external",
        )
        == attacker_command
    )
    if mutation == "global-attributes-config":
        verifier_environment["GIT_CONFIG_GLOBAL"] = str(external_config)
    elif mutation == "system-attributes-config":
        verifier_environment["GIT_CONFIG_SYSTEM"] = str(external_config)
    elif mutation == "core-attributes-config":
        _git(
            repository,
            "config",
            "core.attributesFile",
            str(external_attributes),
        )
        _git(
            repository,
            "config",
            "diff.release-bypass.textconv",
            attacker_command,
        )
    elif mutation == "environment-external-diff":
        verifier_environment["GIT_EXTERNAL_DIFF"] = attacker_command
    elif mutation == "local-external-diff":
        _git(repository, "config", "diff.external", attacker_command)
    elif mutation == "textconv-diff-driver":
        _git(
            repository,
            "config",
            "diff.release-bypass.textconv",
            attacker_command,
        )
    elif mutation == "git-attr-source":
        verifier_environment["GIT_ATTR_SOURCE"] = attribute_source_commit
        _git(
            repository,
            "config",
            "diff.release-bypass.textconv",
            attacker_command,
        )
    if mutation == "paired-replacement":
        _write_release_wheel(
            assets / RELEASE_WHEEL,
            b"replacement build\n",
        )
        _write_release_checksums(assets)
    elif mutation == "checksum":
        (assets / "SHA256SUMS").write_text(
            "0" * 64
            + f"  {RELEASE_WHEEL}\n"
            + hashlib.sha256((assets / RELEASE_SDIST).read_bytes()).hexdigest()
            + f"  {RELEASE_SDIST}\n",
            encoding="ascii",
            newline="\n",
        )
    elif mutation == "extra-asset":
        (assets / "unregistered.txt").write_text(
            "extra\n",
            encoding="utf-8",
        )
    elif mutation == "tag":
        invoked_tag = "v0.4.0"

    return subprocess.run(
        [
            sys.executable,
            str(RELEASE_VERIFIER),
            "--repository-root",
            str(repository),
            "--ledger",
            RELEASE_LEDGER.as_posix(),
            "--assets",
            str(assets),
            "--output",
            str(tmp_path / "dist"),
            "--tag",
            invoked_tag,
            "--protected-ref",
            "refs/remotes/origin/main",
            "--expected-terminal",
            RELEASE_TERMINAL,
            "--distribution",
            RELEASE_DISTRIBUTION,
            "--version",
            RELEASE_VERSION,
            "--artifact",
            RELEASE_WHEEL,
            "--artifact",
            RELEASE_SDIST,
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        env=verifier_environment,
    )


def test_production_release_uses_immutable_ledger_authority() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "publish-release-assets.yml"
    ).read_text(encoding="utf-8")
    assert "types: [published]" in workflow
    assert "github.event.release.prerelease == false" in workflow
    assert "ref: ${{ github.event.release.tag_name }}" in workflow
    assert "fetch-depth: 0" in workflow
    assert "--protected-ref refs/remotes/origin/main" in workflow
    assert "--expected-terminal " + RELEASE_TERMINAL in workflow
    assert CHECKOUT_ACTION in workflow
    assert SETUP_ACTION in workflow
    assert PUBLISH_ACTION in workflow
    assert "@release/v1" not in workflow
    assert 'gh release download "$RELEASE_TAG"' in workflow
    assert "--pattern" not in workflow
    assert "tools/verify_release_authority.py" in workflow
    assert RELEASE_LEDGER.as_posix() in workflow
    assert "--artifact " + RELEASE_WHEEL in workflow
    assert "--artifact " + RELEASE_SDIST in workflow
    assert "python -m build" not in workflow
    assert "id-token: write" in workflow


def test_release_authority_accepts_exact_ledger_bound_artifacts(
    tmp_path: Path,
) -> None:
    completed = _run_release_verifier(tmp_path)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "paired-replacement",
        "checksum",
        "extra-asset",
        "tag",
        "wrong-source",
        "unmerged-tag-child",
        "wrong-terminal",
        "evidence-hash",
        "review-hash",
        "wrong-byte-count",
        "wrong-metadata",
        "extra-child-path",
        "noncanonical",
        "moved-tag-ref",
        "missing-tag-ref",
        "noncanonical-tag-ref",
        "replacement-ref",
        "graft-file",
        "info-attributes",
    ],
)
def test_release_authority_rejects_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    completed = _run_release_verifier(tmp_path / mutation, mutation)
    assert completed.returncode != 0, mutation
    expected_errors = {
        "graft-file": "Git grafts are forbidden",
        "info-attributes": "Git info attributes are forbidden",
        "missing-tag-ref": "release tag ref does not resolve to a commit",
        "moved-tag-ref": "release tag ref does not identify the ledger HEAD",
        "noncanonical-tag-ref": "release tag is not canonical",
        "replacement-ref": "Git replacement objects are forbidden",
    }
    if mutation in expected_errors:
        assert expected_errors[mutation] in completed.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "core-attributes-config",
        "environment-external-diff",
        "git-attr-source",
        "global-attributes-config",
        "local-external-diff",
        "system-attributes-config",
        "textconv-diff-driver",
    ],
)
def test_release_authority_neutralizes_external_git_configuration(
    tmp_path: Path,
    mutation: str,
) -> None:
    case = tmp_path / mutation
    completed = _run_release_verifier(case, mutation)

    assert completed.returncode == 0, completed.stderr
    assert not (case / "attacker.marker").exists()


def test_paired_asset_and_checksum_replacement_is_not_authority(
    tmp_path: Path,
) -> None:
    completed = _run_release_verifier(tmp_path, "paired-replacement")
    assert completed.returncode != 0
    assert "committed authority" in completed.stderr


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
