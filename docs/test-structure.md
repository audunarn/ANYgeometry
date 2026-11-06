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

`fast` aliases `unit`. Unit contains seven explicitly listed modules: primitive
predicates/tolerances and six small CI metadata/selection tool modules.
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

The general sixteen-shard lane now opts into fixture-aware assignment. At actual
pytest collection, resolved FixtureDef scopes in each item's fixture information
admit modules using module/class-scoped fixtures. Each admitted module stays in
one shard; fixture-free modules still balance individual nodes. This preserves
fixture lifetime without forcing a large fixture-free module into one job. The
node policy remains the API default and the qualified Windows 3.14/128-shard
workflow, numerical runtime pins and fresh-host positive route remain unchanged.

The shard report and collection receipt record the admitted module list. The
schedule, profile, plan, child configs, phase manifests and progress carry the
same grouping policy/list; existing report/config/receipt SHA-256 bindings cover
that list. Every execution phase checks its actual present scoped-fixture modules
against the admitted-list intersection. Mixed, unknown or returned mismatching
policies/lists refuse; the verifier also rejects splitting an admitted module.
The grouping environment is cleared before execution, so nested tooling fixtures
do not inherit the outer job's assignment policy.

The root adopted seven engineering-proof modules into qualification using retained real-case evidence:
`test_bounded_nurbs_clipping`, `test_local_sweep_competition`,
`test_local_contact_field`, `test_local_contact_transport`, and
`test_local_contact_stars`, and
`nurbs_surface_intersection/test_generic`, and `test_bounded_nurbs`. Their test bodies and inclusion in full qualification
are unchanged. The retained integration kernel now explicitly runs
only the integration profile through the real isolated driver in five shards. Its
original forty-minute job cap and the stronger 1800-second driver budget remain
unchanged. Four provisional shards exceeded the driver envelope including reserve;
five were the minimum tested count below it. This is an estimate, not measured fit.
Full remains the
exhaustive unit/integration/qualification union; required qualified positives,
wheels, consumers and release gates remain required. This removes duplicate full
scientific selection from the legacy integration route, not tests from full.

Offline scheduling uses preserved node IDs plus explicitly provisional new pure
cases, with four historical fixture-consumer modules as provisional admission.
This does not substitute for authoritative fixture metadata from future actual
collection. Largest whole-module and admitted-group costs, worst projected shard,
setup reserve and unknown costs are reported. A favorable estimate proves no
measured fit, future timing bound or savings. No cost cap, timeout increase or
automatic refusal of unmeasured cases was introduced.
