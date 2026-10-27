# Workflow 2 project record

## 2026-10-27 — Implemented core, provider feasibility unresolved

Current state: PARTIAL_IMPLEMENTATION_BLOCKED_PROVIDER_FEASIBILITY.
Native geometry/lifecycle/schema8, neutral volume/admission, guarded optional
adapter, focused profiles and candidate-owner CI definitions are implemented.
Automatic meshing remains disabled by a production guard; no bypass was added.

Core evidence: 225 affected geometry checks passed; subsequent final replay
checks passed. Linux controlled mesh/process/installed-consumer source slice:
72 passed. Installed exact core owner/mesher wheels outside source trees:
108 passed. Clean core-only installation imports without provider, PyVista/VTK,
solver or ANYsolid. Source and artifact identities are in the external evidence
index; worker checkpoints retain every intermediate failure.

Independent source review found a real full-replay blocked-prism retention bug.
Root reproduced it through public feature removal (reordering alone selected the
incremental route and did not reproduce it), repaired the activation timing and
added regression coverage. Original owner continuation was unavailable due to
an agent-thread-limit error, so root made the narrowly coupled repair. A separate
non-author Read reviewer accepted it; the adjacent typed report boundary was
also tightened. Review is same-family; no human/different-family claim is made.

Provider: official0.4.2 import requires undeclared PyVista. User approved a local
import-only correction; all native bytes were verified unchanged. Corrected
Windows/Linux wheels import without PyVista/VTK. Native box execution then timed
out at120s Windows,30s Windows optimizer-isolation diagnostic,120s Linux; owned
cleanup was verified for all. No raw mesh accepted and no physical limits changed.
The three native attempts are retained; further runs stopped. Thin/quality/
convergence, real generated-mesh physics, full release/native/platform/performance
gates remain unmet. Those are blockers, not waived or relabeled expected failures.

Handoff: sibling ANYmesh/docs/WORKFLOW_2_HANDOFF.md and volume_qualification.json.
No source edits in ANYsolid/ANYsolver or other primary sibling checkouts; their
Workflow1 NO_GO remains unchanged. No merge/push/publication performed.

## 2026-10-27 — Start

State: W2.0 in progress; implementation/qualification/release not complete.
User authorized the replacement Workflow 2 plan. Writes are confined to isolated
ANYgeometry and ANYmesh worktrees; sibling repositories remain read-only.

Workspace: C:/ANYworkspaceWork/workflow2-20261027

Baselines: ANYgeometry 72a7e011fe3777a6aa8107cae388e3a36a917b39;
ANYmesh 0415256ee5e0a0a6088465f58bec7df3de3f2e3e;
read-only ANYsolid 06f3276d9207376316502aca651c3455db984104.
Primary dirty checkouts (129/57 status entries respectively) were preserved.
Branches: codex/workflow-2-solid-regions / codex/workflow-2-volume-meshing.

One architecture/integration owner coordinates separate geometry and meshing
implementation ownership. Claude subscription/Mistral quota exhaustion is cached;
OpenAI workers are the available fallback, with review-family limitations explicit.
No worker may merge, push, change branches, edit siblings or weaken acceptance.

Slices and acceptance:
1. W2.0: frozen small handoff, baselines, pinned provider feasibility.
2. W2.1: atomic closed prisms, orientation, identity, stale state, serialization.
3. W2.2: neutral immutable meshes and independent negative validators.
4. W2.3: bounded optional provider, complete classified boundaries, failure cleanup.
5. W2.4: installed ANYsolid P1/P2 all-solid numerical interoperability.
6. W2.5: applicable full/installed/platform gates, non-author review and handoff.

Open risks: approximate TetWild boundaries can cross source face partitions;
native region transaction/feature integration; overlapping tetrahedra despite
positive Jacobians; historical CI geometry pins; actual optional wheel support.
Provider feasibility is tested before promotion. No fallback is implied.

Evidence root: ../evidence outside the source trees, with unique attempt folders.
Full records must include command, input/source hashes, environment/effective
threads, precise process-wall duration, exit/timeout and cleanup status.
Existing scientific tolerances and Workflow 1 NO_GO evidence remain unchanged.
