# Common-main hosted closure — 2026-10-07

[PR28](https://github.com/audunarn/ANYgeometry/pull/28) merged at
`b1e710571bc8c4bb4039ce82377c29ffdb47f248` after all eight jobs in
[Development run 37544536573](https://github.com/audunarn/ANYgeometry/actions/runs/37544536573)
passed. Local main and the preserved continuation branch were fast-forwarded to
that merge. Its tree exactly equals reviewed head
`5d028a8eb963ebdd39405f7bbbc0c907c3337584` and CI checkout
`d7e4d59c1a3d05972ffadddcb0261026dd11a699`:
`3202fa282c963afcab191fc38ed096305be1b443`.

| Check | Result |
|---|---|
| Windows / Python 3.13 full kernel | 2,836 passed; zero failures, errors or skips |
| Linux / Python 3.13 full kernel | 2,836 passed; zero failures, errors or skips |
| Classifier, wheel build and aggregate Development gate | Passed |
| Isolated installed wheel, Linux and macOS 15 Apple silicon | Passed |
| Installed released-mesher development smoke | Passed |

The kernel JUnit durations are 1,050.570 seconds on Windows and 1,091.302 on Linux;
these are pytest durations, not process-budget accounting. This is the existing
development gate, not new full-platform or large-model meshing qualification.

## Exact development artifact

Wheel: `anygeometry-0.4.5-py3-none-any.whl`, SHA-256
`5adbcfdd3a6df9d897160cd1f8217379c8bf157e711182a68ba260f03fa23956`.
Source distribution SHA-256:
`753b195c4cd1b5381d262323259d2a4b0114d8f3ea67ebf1a6ed5e047d43ed29`.

The package source tree is `8b8317aa31f070165ed1599a2facc382febff6e9`, unchanged
from `1ef18deca6bb1c80c07fc273b07cc8890a9737f8`. All 109 packaged source files in
this wheel are byte-identical to the c333 wheel from run37542063533. That prior
source comparison normalized only Windows git-archive CRLF to LF. The installed
reports bind this new wheel hash and CI source; before/after origin and artifact
checks passed outside source checkouts. This wheel is not a published release.

## Preserved receipts

Complete status, all job logs, artifact metadata, JUnit, effective runtime,
installed-origin reports, source identity and merge receipt remain under
`C:/Github/ANYgeometry/.local-evidence/github-pr28/run-37544536573`.
The immutable inventory contains 65 files; the inventory itself is separately
hashed below. Earlier failed runs37537868022 and37542063533 remain preserved and
unaccepted. Historical local receipts in acceptance.json remain unchanged.

| Receipt | SHA-256 |
|---|---|
| status-final.json | c21df0b31c99824b5c127787589038114eca6d74da6666beadcb038b95279156 |
| full-run.log | 3f1e32c5e7b253f26f6c255a3dc893a87d26542bd36d88756d3747464adc75ca |
| source-identity.json | 253a0f30443be6f42857c4801c65084d597972f56dabe13ddd4692b8d0241460 |
| evidence-hashes.json | d647c269881081ca3804393eed54dee40dcb4871c1b948808290bfb2f0cee620 |
| Linux kernel-results.xml | 77fcf005347e135a2d2e4d87cdce8cbacd706bea8c7101e2be9a60b4100be520 |
| Windows kernel-results.xml | 8de41693640fa8fea375c6aca6e43d3908fb7a479edac62686cc4e244509e75c |

## Remaining boundaries

Published 0.4.5 artifacts, version, tags and release ledger are unchanged.
Large mixed-model linear/quadratic meshing remains unaccepted. Both consumed
ten-shot mesher families remain closed. Local geometry's fresh 60-second tier
used 58.13752980006393 process-seconds; remaining 1.8624701999360695 seconds and
the older 0.2813071-second proof reserve remain held. No extra local checks or
manual qualification dispatches were made.

Polynomial Coons/Bezier receipts remain support-only, with no material/reference/
mesh permission. Original 8.9–12.2% parallel-preparation timing evidence binds
the original 3ba candidate only, not this integrated package.

ANYmesher source repair candidate
`aac9303e8dbda9bcd48c5e14caddaeae472128d9` has independent source review, but all
44 new tests remain unexecuted pending separate unit-check authority. Its owner
main is `7182cb34d6388f96d1ed07b5b6bf33c801997d26`. This geometry delivery grants
no native campaign, mesher publication, application default or ANYfem release.

This follow-up changes documentation only. The verified package, tests and
workflows remain identical; no repeated runtime gate is needed for this record.
