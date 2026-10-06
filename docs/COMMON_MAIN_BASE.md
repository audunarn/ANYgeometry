# Common geometry main base — 2026-10-06

The parallel handoff and preserved native-reference continuation now share a
common main base. Parallel implementation `3ba1bf4c2db40605bbabb95c6b2a5104f479769b`
was reconciled with continuation `740944fbd9801f264e84a1bcd32ed4b57103e1bb`;
the integrated package source is committed at
`3c876d1539dc454f9c3908dae7a743a14f2908b8`.

The public component partition, closure transport and typed chart-query
contracts are described in [PARALLEL_COMPONENT_HANDOFF.md](PARALLEL_COMPONENT_HANDOFF.md).
Native-reference commits retain their support, material, attachment, ancestry
and preparation-epoch guards. Polynomial Coons/Bezier recognition remains
**support only**: it does not certify material, reference or mesh permission.

The separately authorised 60-process-second integration tier recorded:

| Check | Result | Process-wall seconds |
|---|---|---:|
| Initial integrated focused suite | 92 passed; 1 error-text mismatch | 10.12959150003735 |
| Repaired cached-apply paths | 5 passed | 2.104968799976632 |
| Initial installed wheel build/install/spawn | Passed | 4.861700400011614 |
| Initial installed origin/source/runtime identity | Passed | 0.3510878999950364 |
| Clean committed main focused suite | 93 passed | 10.73368089995347 |
| Clean-main installed wheel build/install/spawn | Passed | 4.859947400167584 |
| Clean-main installed origin/source/runtime identity | Passed | 0.31382609996944666 |
| Development runner and six runtime fixtures | 23 passed | 2.540620799991302 |

Total: **35.895423800102435 s**; remaining: **24.104576199897565 s**.
The original failed attempt, commands, runtime receipts and full output remain
under `reports/main-integration-20261006`. These are focused checks, not a full
platform or consumer qualification campaign. The three previously pending
Coons/Bezier public guards passed in the clean-main suite.

The original local integration wheel is
`reports/main-integration-20261006/installed-02/wheels/anygeometry-0.4.5-py3-none-any.whl`,
SHA-256 `aa18ac5815dffd2ccb3171a83f78141024930f52ac7689ab8910969ca5e346e5`.
It was built from the committed package source above. Verification
matched all 100 packaged source files against the recorded source manifest.
Windows/Python 3.14.2 spawned two transported components from outside every
checkout, importing the owner from the external environment's site-packages.
The environment deliberately reused system NumPy 2.4.6; effective OpenBLAS
thread count was one. This is not clean dependency-resolution evidence.

Hosted PR28 run `37537868022` subsequently failed its full kernels: Linux
2,737 passed/86 failed and Windows 2,733 passed/90 failed. Its candidate wheel
SHA-256 is `f90aa4b5b1607a8371557cb16bbaa36509c77b907feea4c75757e2cf1946d525`.
Linux/macOS installed-wheel and the existing installed-mesher smoke checks
passed. These results do not override the kernel failures.

The return-value shadowing defect in attachment splitting was repaired and
passed 15 focused regressions. The broader check exposed a second mismatch:
prepared point-reference scopes expected obsolete edge-only lineage. Their
source repair now requires the exact owner-issued attachment and ordered edge
provenance. Independent source review found no remaining defect in that delta;
its new regressions remain unexecuted locally. Windows checkout now preserves
the frozen builder's exact pinned LF bytes rather than changing the oracle hash.

The broader attempt retained 32 passes/61 failures, with a timeout and failed
process-tree termination attempt; it is unaccepted evidence. Cumulative local
process time is **58.13752980006393 s** of the authorised 60 seconds, leaving
**1.8624701999360695 s** held. No further local checks are launched. The repaired
candidate requires fresh hosted verification and a matching artifact; neither
original wheel above binds its changed package source. Release artifacts remain
unchanged. Full failed logs and runtime/JUnit receipts are retained under
`.local-evidence/github-pr28/run-37537868022`, with hashes in the integration record.

CI and the development runner now supply the required evidence paths and child
thread limits. Kernel jobs retain runtime receipts even after failure.
Independent source review found no functional defect in this wiring.

The measured 8.9–12.2% geometry-preparation improvement in
[PARALLEL_PREPARATION_EVIDENCE.md](PARALLEL_PREPARATION_EVIDENCE.md) binds the
original parallel candidate and its comparison fixtures. No new performance
claim is made for the integrated native-reference package.

The published 0.4.5 artifacts and ledger are unchanged. Both consumed ten-shot
meshing families remain closed, and the old 0.2813071-second proof reserve
remains held. Mixed-model linear/quadratic meshing is still unaccepted. The
ANYmesher split-repair candidate needs final source review and authorised
verification; no mesher execution, application default or package publication
is enabled by this integration.

Historical generated artifacts remain at their original paths, with an exact
local exclusion inventory under `.local-evidence/main-cleanup-20261006`.
Preserved dirty source/evidence was integrated explicitly; unrelated work was
not discarded. GitHub review/CI and the exact final delivery commit are recorded
in the living [continuation record](../reports/general_intersections/quadric_generalization_study.md).

The repaired-source PR28 run `37542063533` completed with Windows **2,836
passed**, Linux **2,835 passed/1 failed**, and all five wheel/classification/
installed-consumer jobs passed. The remaining Linux failure was a test oracle
requiring every certified station to be `bounded`, although the public contract
returns `exact` when both independent residual bounds are zero. Independent
source review confirmed this distinction. The test now checks that exact/bounded
classification against both bounds, retains each unchanged tolerance and
requires a bounded split descendant. Production proof code is unchanged.

This run's development wheel SHA-256 is
`c333af3c5cea5b2534f750eceeb30ee934a6d23d5cf8248d57eae55c0fb963b2`.
All 109 packaged source files match reviewed commit
`1ef18deca6bb1c80c07fc273b07cc8890a9737f8`, after only Windows archive CRLF-to-LF
conversion. Full status, logs, runtime/JUnit receipts, source comparison and a
67-file evidence hash inventory are retained in
`.local-evidence/github-pr28/run-37542063533`. The failed Linux gate remains
unaccepted; the test-only correction requires the normal hosted PR gates before
GitHub main is merged. No further local execution authority was consumed.
