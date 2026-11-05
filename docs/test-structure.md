# Test structure and release evidence

The default automatic Development workflow runs the explicit `unit` profile on
Ubuntu/Python 3.13 under a ten-minute job cap. Its named Development gate means
that small development feedback passed. It does not establish release acceptance.
The curated source-backed primitive predicate cases have not been measured as a
complete cohort locally; hosted CI supplies the actual timing evidence.

Use the isolated runner for deliberate selections:

```console
python tools/ci_isolated_tests.py --profile unit
python tools/ci_isolated_tests.py --profile integration
python tools/ci_isolated_tests.py --profile qualification
python tools/ci_isolated_tests.py --profile full
```

`fast` aliases `unit`. Unit contains five explicitly listed modules: primitive
predicates/tolerances and four small CI metadata/selection tool modules.
Qualification contains the explicit certified hydrostatics and planner modules
listed in `tools/ci_test_profiles.py`. Every other test module, including newly
added modules, defaults to integration. These three tiers are disjoint and their
union is the full collected inventory, including every parameter case. Slow marks
and elapsed-time thresholds do not determine tier membership or remove tests.

The manual Selected test profile workflow exposes unit on one shard and integration/qualification on sixteen
Ubuntu/Python 3.13 shards and verifies their complete selected JUnit coverage.
These development selections do not exercise the required Windows qualified
runtime. Qualification is the third tier's name; only **full** selects all tiers.
The existing Tests workflow remains the complete multi-runtime release route and
runs explicit `--profile full`. It preserves every original scientific, qualified
positive, exact-shard, runtime, installed-wheel and consumer gate. The previous
Development integration/build/installed-wheel/mesher jobs are retained together
in the reusable Integration workflow and are required by Tests, including their
original gate. Artifact names are separated from the caller to avoid collisions.
No full release acceptance requirement has been replaced by a development result.

`profile-modules.json` binds the source module inventory, hashes, declarations and
requested selection before pytest imports test modules. Partial profiles use the
existing directory allowlist to avoid importing other test modules. After actual
collection, `profile-receipt.json` records exact node IDs and their tier partition;
full receipts are exhaustive over the full collected node inventory. Partial
receipts explicitly describe partial development scope. Receipts have no release
acceptance claim. The original shard report, exact collection checks, fresh
core/oracle interpreters, and one fresh H interpreter per authoritative positive
remain unchanged. The plan binds the receipt hashes.

Explicit full refuses positional selectors, `-k`, `-m`, inherited
`PYTEST_ADDOPTS`, and inherited profile/tier environment variables. Old runner
calls that omit `--profile` and use existing selectors retain focused behavior;
their receipts say `legacy-focused` and `partial-development-only`. Omitting both
profile and selectors preserves the complete runner selection. A new evidence
directory is required for each run.

Scheduling uses the maximum historical Windows, retained passed JUnit and
separately labeled completed progress-protocol costs. Raw retained seconds and
source SHA-256 provenance remain in the duration table. Progress elapsed is not
JUnit elapsed or process wall. Cancelled/unfinished cases receive no fabricated
completed duration; they remain censored and unknown. `schedule-receipt.json`
reports estimated test costs plus an explicit 180-second planning reserve for
setup, collection, interpreter startup and artifacts against the unchanged
1800-second envelope. The reserve is policy, not a measured future bound. Neither
a favorable estimate nor balanced assignment proves that a future job or an
unobserved single case fits. The driver-only deadline starts later than the
stronger whole-job deadline. The retained observation refresh redistributes
known heavy cases without changing scientific tests, tolerances or runtime code.

Validation for this change is pure selection/manifest/workflow testing and the
existing synthetic miniature driver protocol, run under the root coordinator's
bounded harness. Local full collection, numerical/model suites, actual hosted
workflow execution and release acceptance are separate outstanding checks.
