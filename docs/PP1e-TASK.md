# PP1e fifth release: bounded store classification

Parent d58b8ff1; same pp1 worktree/branch. No remote or delegation. Planned
measurements are distinct from the owner's three-failed-attempt cap.

Plan / acceptance:
1. Parent-red tests for structural Field/typing/enum/sentinel metadata, direct
   class-annotation dictionaries, exact constant containers and require-empty
   runtime descendants. Keep unknown-object/default-memo negatives intact.
2. Add bounded static-content encodings and object-role decisions. No task-input
   adapter expansion, arbitrary-object traversal or path protection wildcard.
3. Preflight the full observed closure; measure hash/reset or exact refusal costs.
   Any unreachable-store exclusion requires the specified whole-reference trace.
   No such exclusion is made merely from a canary, source name or lack of hits.
4. After the final edit run PP1 tests, 191/57 existing groups and the three ordinary
   serial identity gates. Commit candidate and append exact evidence and blockers.

Open execution boundary: PP1e names an N2c 8-worker trace and 118-second cap, but
does not name a scratch folder for that whole-model trace. The standing exception
requires that folder. Worktree writes do not by themselves supply a whole-model
scratch exception. No whole-model trace or reachability exclusion will occur on
that incomplete run specification. Classification can proceed without exclusions.

The task permits an exact blocker report. In particular native/backend runtime
globals must not be cleared simply to obtain admission. Nonempty require-empty
rows remain refusals. Finite static digests do not solve F7 loaded-code identity,
F8 reference provenance, DET-1 value-equal keys, or PP2/PP6 integration.

## Delivered classification and exact blockers

The original 413 rows now have 388 `pinned-definition`, 24 `require-empty`, and
one `decline` decision. No row is excluded as unreachable, and no path wildcard
protects descendants. Every original path is mapped in
`.pp1-evidence/pp1e-measure-01/original-413-decisions.json`.

| Original kind/category | Count | Current decision |
|---|---:|---|
| Field | 254 | Structural digest |
| Annotation-closure dictionary | 66 | 65 structural; 1 declined |
| Runtime descendant | 24 | Require empty; 22 are nonempty |
| Constant container | 17 | Structural digest |
| Module annotation set | 12 | Structural digest |
| GenericAlias | 8 | Structural digest |
| ABCMeta class symbol | 8 | Process-local symbol identity, pinned source |
| _NoValueType | 6 | Named sentinel plus instance state |
| Union | 5 | Structural digest |
| ConnectionIntent | 3 | Structural member state |
| AuditScope | 2 | Structural member state |
| _MISSING_TYPE | 1 | Named sentinel plus instance state |
| AuditEvidenceQuality | 1 | Structural member state |
| FeatureTopologyRole | 1 | Structural member state |
| object | 1 | Exact transactions._MISSING singleton only |
| Orientation | 1 | Structural member state |
| SafeUUID | 1 | Structural member state |
| SheetTopologyPolicy | 1 | Explicit static record schema |
| TolerancePolicy | 1 | Explicit static record schema |

The full observed ledger has 1,395 stores: 1,039 protected, 293 require-empty,
41 reset roots, 21 covered descendants, one declined. The former
`ReferenceLink.__dataclass_fields__` refusal is gone. The remaining declined row is
`anygeometry.exact_curves._BranchConstants.__annotate_func__.cell[0]`, containing
an unsupported `ForwardRef`. No reference is evaluated to manufacture a digest.

Admission first refuses the nonempty
`anygeometry.backend._RUNTIME_PYTHON_GLOBALS` (16 entries). It is the live
`bspline_evaluate.__globals__` mapping (`backend.py:191`), not a disposable numerical
memo. Its builtins mapping, captured NumPy function globals/attributes, finfo cache,
dispatcher metadata and two other runtime object dictionaries are also present.
The 22 nonempty rows and sizes are in `nonempty-require-empty.json`. Clearing these
bindings to satisfy an empty gate would not be a valid reset. Changing their
qualification policy or excluding them needs explicit evidence/owner disposition.

## Static metadata boundary

Field and dataclass-parameter native slots are encoded structurally. Metadata
mapping proxies must have a plain dict backing (checked through gc referents).
Supported typing aliases encode their actual structure without evaluation. Enum
member dictionaries, property accessors/docs, exact Fraction numerator/denominator,
and a finite list of frozen default-record schemas are covered. This does not add
task-input codecs or an arbitrary-instance adapter. Unknown objects still decline.
Static-record schemas are TolerancePolicy, AuditMetrics,
IntersectionQualificationPolicy, FrozenMetadata, structural.ParameterRange and
SheetTopologyPolicy; their actual fields are encoded, including typed float bits.

Annotation dictionaries must be the actual class annotation closure dictionary,
or the exact native Python typing factory's bound `checked_types`/`types` closure
(code-object identity and globals checked). Only those dictionary objects receive
roles. Two source-reviewed constant container trees record concrete container IDs;
callables/classes and their defaults are not traversed as constant descendants.
Runtime require-empty decisions are exact named paths tied to the pinned runtime.

Protected-content digests are rechecked at both task boundaries by the existing
gate. SHA256 collision resistance and trusted process-local class/function symbol
identities remain engineering assumptions. F7 is still open. Native loaded-image
purity is not proved. The full gate actually refuses, so these changes establish
no production admission or universal determinism claim.

One candidate bug was caught before delivery: temporary structural tuples could
reuse object IDs in the digest alias table. The failing last-field mutation test
is retained. Keeping temporary structures alive until encoding ends fixes that
specific issue. A subsequent sentinel-state probe already passed; it required no
code repair despite its evidence label containing `red`.

## Verification and costs

19 new cases fail on exact parent d58b8ff1 modules. The first nine were red on the
untouched parent before implementation; eight later cases were red before their
implementation; the temporary-ID defect was candidate-red before repair. Final
parent replay binds all 19 cases, including the supplementary sentinel-state check.
It is an exact-module replay with unchanged dependencies, not a parent checkout.

Final PP1 group: 83 passed (19 new +64 prior). Required 191 group re-run in bounded
chunks: 170 +7 +14 passed. Additional group: 57 passed, 3 retained overflow warnings.
All are after the final implementation edit; total **331 passed**. No prior test
was changed, deleted or weakened in this release. Numerical sources/tolerances and
task-input adapters are unchanged. Independent review is not performed here.

F5 measurement: 700 modules, 644 files; complete hash capture/verification costs
0.28055310000490863 s and 0.28552060000947677 s (sum 0.5660737000143854 s).
Ledger construction: 0.28575870000349823 s. Full-census/digest refusal:
0.40128530000220053 s. Actual full-task reset cost is NOT measured because admission
refuses before reset. All 65 watched reset/runtime root identities or cache-info
witnesses remain unchanged, and the full ledger records zero resets.

A separate 1,000-entry one-dictionary fixture resets to zero in
0.000019699989934451878 s, census 0.0009172999998554587 s. This is not production
reset overhead. Load: aggregate CPU 8.5% before /10.5% after, 32 logical CPUs,
BLAS [1,1], clock resolution 1e-7 s. These are single observations, not performance
bounds; scheduling/I/O uncertainty is unbounded here. The clock resolution alone
does not establish measurement accuracy. No hash caching or speedup is claimed.

Status: **BLOCKED** pending the explicit runtime-store disposition and remaining
ForwardRef schema. Whole-model reachability trace was not run and no store was
excluded; the named scratch location is missing from that run's authorization.
Next release should resolve those boundaries and requalify real startup/reset.
PP2, F7/F8, DET-1 equivalence and independent acceptance remain open. No push.

Ordinary identity gates after the final implementation edit: sector60 97 faces,
checksum `1078d6bf9c5b9c33f84e0d68954fbf89f52e677368b23c5f7641b8c9323df153`;
r7 240 faces, checksum
`160166d64deb7b576e6c18bc5fa4dcea5e394b5fc1de896890629b183b1e0e70`.
Both applied, 110/360 vertex-on-face checks respectively, zero over tolerance.
face200 canonical checksum
`2ab517de9f4bc32af5f0aada85b234084282f982c313a73224f5bedda400c51f`,
area exactly 4.218529952235817, one cell. These finite projection checks are the
bit-identity evidence; no unseen-model or whole-model equivalence is claimed.
