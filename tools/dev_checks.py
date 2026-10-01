"""Focused development checks; never a substitute for supported-release qualification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SCOPES = {
    "smoke": ["test_coordinates.py", "test_kernel_identity_structural.py",
              "test_mesher_model_contract.py",
              "test_import_boundaries_cli.py::test_public_owner_exports_use_one_geometry_and_reference_type",
              "test_import_boundaries_cli.py::test_source_import_graph_has_no_forbidden_or_undeclared_dependency",
              "test_import_boundaries_cli.py::test_fresh_import_does_not_load_consumers_gui_mesh_or_solver",
              "test_import_boundaries_cli.py::test_cli_writes_and_inspects_example",
              "test_import_boundaries_cli.py::test_module_main_and_version_work_from_outside_checkout"],
    "coordinates": ["test_coordinates.py", "test_automation_contract.py"],
    "serialization": ["test_serialization.py", "test_serialization_v3.py",
                      "test_serialization_gap_closure.py", "test_exact_intersection_curves.py",
                      "test_serialization_quadric.py"],
    "intersections": ["test_intersections.py", "test_intersection_workflow.py",
                      "test_batch_intersections.py", "test_cylinder_atlas_contract.py",
                      "test_material_arrangement.py", "test_trimmed_charts.py",
                      "test_preparation_binding.py", "test_curved_intersection_matrix.py",
                      "test_exact_intersection_curves.py", "test_cylinder_curve_events.py",
                      "test_quadric_curve_events.py", "test_boundary_curve_connect.py",
                      "test_pair_policy_generalization.py", "test_curved_split_contract.py",
                      "test_cone_material_arrangement.py", "test_quadric_intersection_curve.py",
                      "test_quadric_supports.py", "test_cone_intersections_engine.py",
                      "test_extruded_surface.py", "test_extruded_support_recognition.py",
                      "test_extruded_intersections.py", "test_extruded_public_routing.py"],
    "features": ["test_features_editing.py", "test_feature_topology_ownership.py",
                 "test_transform_pattern_api.py", "test_unknown_features.py"],
}
# Only narrow, well-understood modules get automatic focused coverage.
# Shared geometry, numerical primitives and all unfamiliar paths run the full suite.
MODULE_SCOPES = {"src/anygeometry/coordinates.py": "coordinates",
                 "src/anygeometry/serialization.py": "serialization"}
DEVELOPMENT_JOBS = {
    "Classify changes", "Build and verify wheel", "Development gate",
    "Development kernel (ubuntu-latest, Python 3.13)",
    "Development kernel (windows-latest, Python 3.13)",
    "Installed wheel (ubuntu-latest)", "Installed wheel (macos-15)",
    "Installed mesher development check",
}


def documentation_only(path: str) -> bool:
    if path == "docs/LICENSE.md":  # packaged license: installation contract
        return False
    return path.endswith(".md") and (
        "/" not in path or path.startswith(("docs/", "reports/")))


def select(paths: list[str], root: Path = ROOT) -> dict:
    """Fail closed for missing tests, shared fixtures and unclassified changes."""
    selected = set()
    reasons = []
    for path in paths:
        path = path.replace("\\", "/")
        if documentation_only(path):
            continue
        if path in MODULE_SCOPES:
            selected.update("tests/" + name for name in SCOPES[MODULE_SCOPES[path]])
        elif path.startswith("tests/test_") and path.endswith(".py") and (root / path).is_file():
            selected.add(path)
        else:
            reasons.append("Full suite required by " + path)
    if reasons:
        return {"scope": "full", "tests": ["tests"], "reasons": reasons}
    if selected:
        return {"scope": "focused", "tests": sorted(selected), "reasons": []}
    return {"scope": "documentation", "tests": [], "reasons": ["No runtime changes detected"]}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def changed_paths(base: str) -> list[str]:
    # Merge-base changes plus staged/unstaged and new files, including deleted tests.
    paths = []
    for args in (("diff", "--name-only", "-z", base + "...HEAD"),
                 ("diff", "--name-only", "-z", "HEAD"),
                 ("ls-files", "--others", "--exclude-standard", "-z")):
        paths.extend(p for p in git(*args).split("\0") if p)
    return sorted(set(paths))


def reusable_evidence(run: dict, jobs: list[dict], *, pr: int, base: str,
                      delta: list[str]) -> bool:
    """Only a complete successful development run can cover unchanged runtime inputs."""
    return (
        run.get("status") == "completed" and run.get("conclusion") == "success"
        and run.get("event") == "pull_request"
        and any(p.get("number") == pr and p.get("base", {}).get("sha") == base
                for p in run.get("pull_requests", []))
        and len(jobs) == len(DEVELOPMENT_JOBS)
        and {j.get("name") for j in jobs} == DEVELOPMENT_JOBS
        and all(j.get("status") == "completed" and j.get("conclusion") == "success" for j in jobs)
        and all(documentation_only(p) for p in delta)
    )


def ci_reuse() -> dict | None:
    """Bounded, read-only lookup. Missing/inaccessible evidence means run all checks."""
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    pr, base = os.environ.get("PR_NUMBER"), os.environ.get("BASE_SHA")
    if not all((token, repo, pr, base)):
        return None

    def api(path: str) -> dict:
        request = urllib.request.Request("https://api.github.com/repos/" + repo + path,
            headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    try:
        runs = api("/actions/workflows/development.yml/runs?status=success&event=pull_request&per_page=5")
        for run in runs.get("workflow_runs", []):
            if not any(p.get("number") == int(pr) for p in run.get("pull_requests", [])):
                continue
            head = run.get("head_sha", "")
            if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
                continue
            # HEAD is the actual checkout (including base merge), not the mutable
            # head field in run.pull_requests. Base/source changes cannot be hidden.
            delta = [p for p in git("diff", "--name-only", "-z", head, "HEAD").split("\0") if p]
            if not all(documentation_only(p) for p in delta):
                continue
            jobs = api(f"/actions/runs/{run['id']}/jobs?per_page=100").get("jobs", [])
            if reusable_evidence(run, jobs, pr=int(pr), base=base, delta=delta):
                return {"run": run["id"], "head_sha": head, "base_sha": base,
                        "delta": delta, "jobs": sorted(DEVELOPMENT_JOBS)}
    except (OSError, ValueError, subprocess.CalledProcessError, KeyError):
        # API failure, unavailable prior commit or malformed evidence: no reuse.
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["auto", "full", *SCOPES], default="auto")
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--plan", action="store_true", help="Print selection without running")
    parser.add_argument("--report", type=Path, help="Save selection, duration and exit status")
    parser.add_argument("--paths-file", type=Path, help="JSON path list supplied by CI")
    parser.add_argument("--ci-reuse", action="store_true", help="Reuse complete successful PR evidence for prose-only deltas")
    args = parser.parse_args(argv)
    try:
        paths = (json.loads(args.paths_file.read_text(encoding="utf-8")) if args.paths_file
                 else changed_paths(args.base) if args.scope == "auto" else [])
        if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
            raise ValueError("changed paths must be a JSON array of strings")
        plan = select(paths) if args.scope == "auto" else {
            "scope": args.scope, "tests": ["tests"] if args.scope == "full" else
            ["tests/" + name for name in SCOPES[args.scope]], "reasons": ["Explicit scope"]}
        if args.ci_reuse and args.scope == "auto" and plan["tests"]:
            evidence = ci_reuse()
            if evidence:
                plan = {"scope": "documentation", "tests": [],
                        "reasons": ["Runtime inputs match complete successful development evidence"],
                        "reused_evidence": evidence}
        if any(not (ROOT / path.split("::", 1)[0]).exists() for path in plan["tests"]):
            plan = {"scope": "full", "tests": ["tests"], "reasons": ["Selected test unavailable"]}
        plan.update(changed_paths=paths, evidence="development only", python=sys.version,
                    command=[sys.executable, "-m", "pytest", *plan["tests"], "--durations=20"]
                    if plan["tests"] else ["git", "diff", "--check"])
        print(json.dumps(plan, indent=2), flush=True)
        if args.plan:
            code = 0
        else:
            start = time.monotonic()
            code = subprocess.run(plan["command"], cwd=ROOT).returncode
            plan.update(exit_code=code, duration_seconds=time.monotonic() - start)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        return code
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Cannot determine/run development checks: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
