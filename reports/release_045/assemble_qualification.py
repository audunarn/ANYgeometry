"""Close the release record only from complete, bound acceptance evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--geometry-main", required=True)
parser.add_argument("--mesher-main", required=True)
parser.add_argument("--fem-main", required=True)
parser.add_argument("--fem-integration", required=True)
parser.add_argument("--mesher-integration", required=True)
args = parser.parse_args()

def load(name):
    return json.loads((root / name).read_text(encoding="utf-8-sig"))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()

def git(owner, *arguments):
    return subprocess.check_output(["git", "-C", "C:/Github/" + owner, *arguments], text=True).strip()

statuses = {owner: load(owner + "-ci.json") for owner in ("geometry", "mesher", "fem")}
heads = {"geometry": "dbd784d81f743c60cdfccae02b93ecd38e22767c",
         "mesher": "edfaa28d86251c360ee87de8aa257cd576357cec",
         "fem": "f9a5e53d89d754004b2d8ebd88b833fd9d5f2a7f"}
for owner, status in statuses.items():
    assert status["headSha"] == heads[owner], owner
    if owner == "mesher":
        assert not any(job["conclusion"] == "failure" for job in status["jobs"])
        continue
    if owner == "fem":
        assert any(job["conclusion"] == "failure" for job in status["jobs"])
        continue
    assert status["status"] == "completed" and status["conclusion"] == "success", owner
    assert status["jobs"] and all(job["conclusion"] == "success" for job in status["jobs"]), owner
assert len(statuses["geometry"]["jobs"]) == 32
summary_path = root.parent / "general_intersections/installed-qualification-36834513207.json"
summary = json.loads(summary_path.read_text())
assert summary["overall_status"] == "passed" and len(summary["installed_general_consumers"]) == 3
payload = load("payload-review.json")
assert payload["status"] == "passed"
candidate_hash = payload["ci_candidate_artifacts"][0]["sha256"]
for platform in summary["installed_general_consumers"]:
    assert platform["status"] == "passed" and platform["unaccepted_meshes"] == 0
    assert len(platform["cases"]) == 16 and platform["expected_meshes"] == 16
    assert platform["artifacts"]["geometry"]["sha256"] == candidate_hash
    assert platform["sources"]["ANYmesh"] == "e21c0fc93662776762430e14450d54ac9192e2e8"
    assert platform["sources"]["ANYfem"] == "83f3d6c81e405ca45f5b7edd7cb7c2afd32804c4"
assert git("ANYgeometry", "rev-parse", args.geometry_main + "^{tree}") == git("ANYgeometry", "rev-parse", "dbd784d^{tree}")
for owner, main, qualified in (("ANYmesh", args.mesher_integration, "e21c0fc"), ("ANYfem", args.fem_integration, "83f3d6c")):
    assert git(owner, "rev-parse", main + ":src") == git(owner, "rev-parse", qualified + ":src")
assert load("installed-wheel-review.json")["status"] == "passed"
assert load("installed-batch-review.json")["status"] == "passed"
review = {"schema": "anygeometry.release-independent-review-v1", "status": "passed",
          "review_method": "separate post-download static/payload/checksum/metadata/Twine and external-installed-wheel assessment",
          "artifact_review": payload,
          "external_wheel_review_sha256": digest(root / "installed-wheel-review.json"),
          "external_batch_review": load("installed-batch-review.json"),
          "scientific_assessment": {"evidence": str(summary_path.relative_to(root.parent.parent)),
              "sha256": digest(summary_path), "accepted_meshes": 48,
              "independence": "analytic material areas, implicit support residuals/branch completeness and complete shared-node/lifecycle oracles; checksums are identity evidence only",
              "limits": "Known development fixtures; no blind-validation, universal-input, Intel-native-consumer or Qt scientific-acceptance claim"}}
(root / "independent_review.json").write_text(json.dumps(review, indent=2) + "\n")
source = {"commit": args.geometry_main, "tree": git("ANYgeometry", "rev-parse", args.geometry_main + "^{tree}")}
qualification = {"accepted_terminal": "ACCEPTED_ANYGEOMETRY_0_4_5_RELEASE", "artifact_source": source,
    "publication_authorized": True, "authority": "User explicitly requested merge and normal publication on 2026-10-01",
    "artifacts": payload["artifacts"], "ci_candidate_artifacts": payload["ci_candidate_artifacts"],
    "integration_commits": {"ANYgeometry_main": args.geometry_main, "ANYmesh_main": args.mesher_main,
        "ANYmesh_review_integration": args.mesher_integration,
        "ANYfem_main_base": args.fem_main, "ANYfem_unmerged_integration": args.fem_integration},
    "hosted_runs": {owner: {key: value for key,value in status.items() if key != "jobs"} |
        {"passed_jobs": sum(job["conclusion"] == "success" for job in status["jobs"]),
         "total_jobs": len(status["jobs"]),
         "failed_jobs": [job["name"] for job in status["jobs"] if job["conclusion"] == "failure"]}
        for owner,status in statuses.items()},
    "installed_evidence": {"path": str(summary_path.relative_to(root.parent.parent)), "sha256": digest(summary_path), "platform_meshes": 48},
    "preserved_failure": "../general_intersections/installed-qualification-36825533614.json",
    "budgets_and_tolerances": "900 seconds per consumer command and scientific tolerances unchanged; user-authorized 15-degree default admission floor, explicit stricter policies and opt-in direct recovery retained",
    "mesher_ci_disposition": "Owner matrix is pending; no full hosted mesher-suite acceptance or main merge is inferred. Geometry publication is bound to its completed32-job gate and48 installed native meshes.",
    "fem_ci_disposition": "Unaccepted scientific gate: Linux/Python3.13 has1041passed/130skipped/3capacity-workflow errors under frozen ANYsolver d04199; complete raw log retained. PR10 remains unmerged. This is not acceptance of broader FEM/Qt physics, nor an established pre-change introduction baseline.",
    "limitations": ["Intel macOS: kernel/wheel only", "Qt parity/scientific acceptance remains ANYfem-owned; full FEM CI has three plane-stress convergence errors", "FEM integration PR10 remains unmerged pending its scientific gate", "No ANYfem or mesher package publication", "Positive-area overlap needs explicit ownership", "Ambiguous feature-role changes/nonexact attachment remaps refuse atomically", "CAD export not activated"]}
(root / "qualification.json").write_text(json.dumps(qualification, indent=2) + "\n")
ledger = {"schema": "anyecosystem.release-ledger-v1", "distribution": "ANYgeometry", "version": "0.4.5", "tag": "v0.4.5",
    "artifact_source": source, "artifacts": [{**row, "sha256": row["sha256"].upper()} for row in payload["artifacts"]],
    "publication_authorized": True, "qualification": {"accepted_terminal": qualification["accepted_terminal"],
        "evidence_sha256": digest(root / "qualification.json"), "independent_review_sha256": digest(root / "independent_review.json")}}
(root / "ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
print(json.dumps({"terminal": qualification["accepted_terminal"], "integration_commits": qualification["integration_commits"], "artifacts": ledger["artifacts"]}))
