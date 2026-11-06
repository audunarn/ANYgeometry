# PP5c Part 1 verification handoff

Implemented constructor-state admission and transport. Verified against Part 1
criteria; independent review and release are pending. Part 2 remains outstanding.
Base b93f2715eefb66723352b64da7118ff711acf2bb on
claude/pp5-stage-partition-20261031. No push or remote operation.

## Final checks

Evidence is under .pp1-evidence/. PP5C-PART1-MANIFEST.json binds ten final receipts
to the actual current source and command-file hashes: zero mismatches; every receipt
records unchanged inputs, exit 0 and verified cleanup with zero owned processes.
PP5C-COMMANDS.json records all 32 supervised commands, including retained failures,
with exact argv, cwd, exit/status and cleanup. Full stdout/stderr remain per receipt.
The canonical hash in PP5C-CONTRACT.md is of the committed fixture captured by the
first admission run, not a later freshly UUID-assigned closure. No golden was changed.

| Final receipt | Passed | Failed |
|---|---:|---:|
| pp5c-accepted-unit | 219 | 0 |
| pp5c-accepted-real | 4 | 0 |
| pp5c-accepted-t2b | 170 | 0 |
| pp5c-accepted-ring | 7 | 0 |
| pp5c-accepted-seam | 14 | 0 |
| pp5c-accepted-focused | 57 | 0 |
| Total | 471 | 0 |

Focused tests retain three known NumPy overflow warnings. Some pytest processes
report a pyreadline3 finalizer WinError after completion; receipts retain this stderr.
It is not represented as a test failure or concealed as empty stderr.

OFF identities (pp5c-off-sector60, pp5c-off-r7, pp5c-off-face200) pass:
sector60 1078d6bf9c5b9c33f84e0d68954fbf89f52e677368b23c5f7641b8c9323df153;
r7 160166d64deb7b576e6c18bc5fa4dcea5e394b5fc1de896890629b183b1e0e70;
face200 2ab517de9f4bc32af5f0aada85b234084282f982c313a73224f5bedda400c51f,
area 4.218529952235817. Both subsets applied successfully. These exact existing
checksum comparisons support ordinary-output identity on these cases only.
No new purity module was imported by these OFF paths.

pp5c-admission-final verifies source unchanged plus exact constructor-state packet
roundtrip for source 1083 faces/540 derived arrays (985821 payload bytes), N2c source
561/18 (900300 bytes), and N2c closure 561/18 (842545 bytes). Closure alone permits the
existing absent-to-empty extension transition. N2c planning/apply was not run.
The numerical argument and accepted raw-native-view risk are in PP5C-CONTRACT.md.
Binary differences are zero at the checked boundary; no cross-product error bound
or universal alias-freedom claim is made.

## Retained red/failure evidence

pp5c-constructor-red-02: 6 failed / 13 passed on base, pristine constructor state
refused. First red fixture's warm arc cache is separately retained in constructor-red.
constructor-green-01: 6 failed / 59 passed, fixed owner-count assumption and accidental
test-held reference; green-02 then 65 passed. alias-extra-red: 2 failed, then added
weak-reference and tracked-array checks. final-unit: 212 passed / 3 failed because
full application imports contaminated ledger-test closure; ledger-isolation 194 passed,
then captured real fixture separated unit and real-input checks. real-admission first
failed only the pre-existing extension transition assertion; second passed. metadata-red
2 failed, missing-extra-red 1 failed, field-preflight-red 1 failed; each retained with
the pre-fix source and followed by checks. No test/gate/tolerance or certified pin was
weakened. Earlier passing groups are superseded by accepted-* receipts, not counted twice.

## Changes and commands

Source: _planning_constructor.py:1 (exact schema/recomputation, bounded alias audit,
manifest restore); _planning_qualification.py:_check_model_record_state delegates;
_planning_executor.py:freeze_model/thaw_model bind and check the manifest.
Tests: tests/test_planning_constructor_state.py:1 (25 cases);
tests/data/pp5c-cylinder-input.json:1 (one actual Cylinder-face closure).
Tools: tools/pp5c_real_admission.py:1. Docs: PP5C-CONTRACT.md:1 and this file:1.
Exact line map and artifact hashes are in .pp1-evidence/PP5C-PART1-CHANGES.json.

Every check used this wrapper in the named worktree:
& 'C:\Python\Python314\python.exe' -B tools/pp1_check.py <label> <arguments>
Full concrete child argv are recorded verbatim in PP5C-COMMANDS.json; wrapper labels
are the entry's label, and arguments are child argv after '-m' for pytest or after '-u'
for scripts, omitting wrapper-added '-p no:cacheprovider --basetemp ...'.
Audit command:
& 'C:\Python\Python314\python.exe' -B .pp1-evidence/pp5c_part1_audit.py
Git commands: git diff --check; git diff --stat; git branch --show-current;
git add -- src/anygeometry/_planning_constructor.py src/anygeometry/_planning_executor.py src/anygeometry/_planning_qualification.py tests/test_planning_constructor_state.py tests/data/pp5c-cylinder-input.json tools/pp5c_real_admission.py docs/PP5C-CONTRACT.md docs/PP5C-PART1-EVIDENCE.md
git commit -m 'Qualify bounded Cylinder constructor state and transport'
git rev-parse HEAD; git status --short

## NOT CHECKED

Part 2 real stages, A1-A7, serial/parallel equality, timing and balance: next slice.
Full N2c planning: not authorized. Universal alias-freedom: explicitly not claimed.
Opus/forseti review, integration, acceptance and publication: coordinator-owned, pending.
No performance saving is inferred from passing functional checks.
