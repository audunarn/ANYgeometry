"""Focused development checks; never a substitute for supported-release qualification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCOPES = {
    "smoke": ["test_coordinates.py", "test_kernel_identity_structural.py",
              "test_mesher_model_contract.py", "test_import_boundaries_cli.py"],
    "coordinates": ["test_coordinates.py", "test_automation_contract.py"],
    "serialization": ["test_serialization.py", "test_serialization_v3.py",
                      "test_serialization_gap_closure.py", "test_exact_intersection_curves.py"],
    "intersections": ["test_intersections.py", "test_intersection_workflow.py",
                      "test_material_arrangement.py", "test_trimmed_charts.py",
                      "test_preparation_binding.py", "test_curved_intersection_matrix.py",
                      "test_exact_intersection_curves.py", "test_cylinder_curve_events.py",
                      "test_quadric_curve_events.py", "test_boundary_curve_connect.py",
                      "test_pair_policy_generalization.py", "test_curved_split_contract.py"],
    "features": ["test_features_editing.py", "test_feature_topology_ownership.py",
                 "test_transform_pattern_api.py", "test_unknown_features.py"],
}
# Only narrow, well-understood modules get automatic focused coverage.
# Shared geometry, numerical primitives and all unfamiliar paths run the full suite.
MODULE_SCOPES = {"src/anygeometry/coordinates.py": "coordinates",
                 "src/anygeometry/serialization.py": "serialization"}


def documentation_only(path: str) -> bool:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["auto", "full", *SCOPES], default="auto")
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--plan", action="store_true", help="Print selection without running")
    parser.add_argument("--report", type=Path, help="Save selection, duration and exit status")
    parser.add_argument("--paths-file", type=Path, help="JSON path list supplied by CI")
    args = parser.parse_args(argv)
    try:
        paths = (json.loads(args.paths_file.read_text(encoding="utf-8")) if args.paths_file
                 else changed_paths(args.base) if args.scope == "auto" else [])
        if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
            raise ValueError("changed paths must be a JSON array of strings")
        plan = select(paths) if args.scope == "auto" else {
            "scope": args.scope, "tests": ["tests"] if args.scope == "full" else
            ["tests/" + name for name in SCOPES[args.scope]], "reasons": ["Explicit scope"]}
        if any(not (ROOT / path).exists() for path in plan["tests"]):
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
