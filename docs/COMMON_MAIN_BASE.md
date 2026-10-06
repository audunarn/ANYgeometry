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

The authoritative local development wheel is
`reports/main-integration-20261006/installed-02/wheels/anygeometry-0.4.5-py3-none-any.whl`,
SHA-256 `aa18ac5815dffd2ccb3171a83f78141024930f52ac7689ab8910969ca5e346e5`.
It was built from the committed package source above; subsequent delivery
changes touch CI, developer tooling, tests and documentation only. Verification
matched all 100 packaged source files against the recorded source manifest.
Windows/Python 3.14.2 spawned two transported components from outside every
checkout, importing the owner from the external environment's site-packages.
The environment deliberately reused system NumPy 2.4.6; effective OpenBLAS
thread count was one. This is not clean dependency-resolution evidence.

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
