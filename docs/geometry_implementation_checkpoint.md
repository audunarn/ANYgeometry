# Workflow 2 geometry implementation checkpoint

State: **verified against assigned development criteria**, root independent acceptance **under review**. Source frozen 2026-10-27 14:32:20 UTC. No commits, pushes, branch changes, provider/meshing runs or primary-checkout edits performed.

Repository: `C:/ANYworkspaceWork/workflow2-20261027/ANYgeometry`, baseline `72a7e011`. Primary `C:/Github/ANYgeometry` remained off limits. Root owns `docs/solid_volume_contract.md` and `docs/WORKFLOW_2_PROJECT.md`; neither edited by this worker.

## Delivered slices and criteria

1. Native production model API for closed boxes/strictly convex polygonal normal prisms/signed offset thin prisms. Immutable owner-bound persistent region UUIDs reference ordinary planar source faces and shared straight edges. Snapshot has the exact v1 handoff fields, immutable owned coordinate storage, outward loops, analytic volume and source SHA-256. Acceptance: independent oriented triangle-volume sums, all outward normals, directed row and actual shared source-edge closure, triangle/rectangle/pentagon/heptagon, rotated profiles, both winding and extrusion signs; rejection of zero/oblique/concave/nonplanar/self-crossing/repeated/collinear/nonfinite input.
2. Transaction/revision, source mutation refusal, explicit whole-face groups, role-preserving regeneration, ordinary feature-history incremental/full replay/suppression, topology/design restoration, clone rebinding and strict audit integrated in actual owner APIs. Acceptance: nested failed transaction clears additions and restores group edits; invalid source mutation/removal and conflicting regeneration fail atomically; snapshots stay detached and stale/altered/foreign reuse fails; source edge substitution cannot create a geometric-only closure.
3. Persistent document schema **8** only for region-bearing documents, per-region schema **1**. Root review explicitly required dedicated top-level versioning. Existing volume-free schemas and encoding stay unchanged; schema5/6/7 with a region field are rejected even with recomputed valid checksum. Acceptance: owner-preserving round-trip/checksum, clone owner rebinding, strict malformed ownership/orientation/group/schema refusal, public required_version agreement and existing NURBS/quadric codec regressions.

Implementation files: `src/anygeometry/solid_regions.py` (new); `model.py`, `features.py`, `serialization.py`, `strict_audit.py`, `__init__.py`; `tests/test_solid_regions.py` (new); `tools/dev_checks.py`. Public documentation: `docs/solid_regions.md`.

## Exact check execution and evidence

Interpreter: `C:/ANYworkspaceWork/workflow2-20261027/runtime/Scripts/python.exe`. Every run used explicit owned repository cwd, `PYTHONPATH=C:/ANYworkspaceWork/workflow2-20261027/ANYgeometry/src`, one BLAS/OpenMP thread, bounded Windows process-tree supervision and a unique evidence directory under `C:/ANYworkspaceWork/workflow2-20261027/evidence`. Each directory retains full command/source hashes/effective runtime in `identity.json`, precise process-wall/exit/cleanup in `result.json`, full stdout/stderr, and pytest XML. `git diff --check` passed.

Complete chronological commands/counts/accounting: `C:/ANYworkspaceWork/workflow2-20261027/evidence/geometry-worker-check-summary.json`. Cumulative worker check process-wall **67.736 seconds /600**, remaining **532.264 seconds**. Every owned run terminated with zero owned processes confirmed. No timeout.

Key successful commands (all invoked as interpreter `-m pytest -q` plus retained `--junitxml` argument):

- `tests/test_solid_regions.py`: **35 passed**, `geometry-repeat-vertex-final-e12449b972`, 1.922 s. Subsequent final schema helper run also includes this complete final test file.
- `tests/test_solid_regions.py tests/test_nurbs_definitions.py tests/test_serialization_quadric.py`: **105 passed**, `geometry-required-version-final-bba7fd63ea`, 3.219 s, final source identity.
- `tests/test_solid_regions.py tests/test_serialization.py tests/test_serialization_v3.py tests/test_serialization_gap_closure.py tests/test_serialization_quadric.py tests/test_features_editing.py tests/test_feature_topology_ownership.py tests/test_automation_contract.py`: **171 passed**, `geometry-schema8-final-32c698153c`, 5.016 s. Later edits only add repeated-vertex refusal and required_version schema8 recognition, independently checked above.
- `tests/test_solid_regions.py tests/test_features_editing.py tests/test_detached_feature_staging.py tests/test_feature_topology_ownership.py tests/test_feature_snapshot_isolation.py tests/test_serialization.py tests/test_serialization_v3.py tests/test_serialization_gap_closure.py tests/test_identity_topology.py tests/test_model_kernel_update.py tests/test_strict_audit.py tests/test_structural_lifecycle.py`: **219 passed**, `geometry-final-focused-17c5b4e3af`, 11.547 s. This run preceded source-edge closure/schema8 hardening; subsequent focused impact checks are retained.
- `tests/test_solid_regions.py tests/test_automation_contract.py`: **52 passed**, `geometry-closure-final-c352f0cc04`, 2.312 s.

Preserved failed evidence:

- First new tests: 9 passed/14 failed; fixed production corners call (`corners=None` for neutral faces). `geometry-focused-1-731648cb99`.
- Second: 19 passed/4 failed; fixed atomic complete-prism regeneration instead of sequential intermediate point edits, preserved feature region lookup through staging owner changes, and test float comparisons. `geometry-focused-2-e8f88da1a8`.
- Third: 21 passed/2 failed; fixed one float comparison and added authoritative Plane supports needed by strict audit. Diagnostic `geometry-audit-diagnostic-1c15454db8` identified unsupported missing-support findings; passing fourth focused run retained.
- Broad applicable baseline attempt: **220 passed /28 failed**, `geometry-baseline-1-e883dbea39`, 30.500 s. All failures are disposable release-authority Git fixtures in `test_import_boundaries_cli.py`, at their local `git push -u origin main` preparation. These are sandbox fixture failures, not established inherent baseline defects. Root independently reproduced the environment issue and owns the escalated fixture/full applicable verification. This worker did not alter global Git settings or claim those omitted checks passed.

## Remaining gates and limitations

Root retains final acceptance, installed downstream consumer integration and independent review; this worker cannot certify its own implementation. Source review is under way. Root will run applicable broader gates with the fixture environment repaired. No full repository suite, volume-provider campaign, qualification/performance sweep, installed wheel acceptance, publication or release gate was run by this worker. Focused development passes are not a volume-mesher qualification claim.

Envelope remains a single strictly convex normal prism per snapshot, native planar faces/straight shared edges, SI metres, no holes/curves/concavity/multiple components/interior interfaces. Existing model edge/plane representability limits remain. Regeneration changes in ordered topology/orientation fail with explicit conflict; no implicit topology rebind or selection transfer. Original signed construction parameters are retained in provenance; later generic source edits are separately bound by snapshot content/signature/revision.
