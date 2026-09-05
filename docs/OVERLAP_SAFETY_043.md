# ANYgeometry 0.4.3 overlap safety

This is an intentional safety tightening of the 0.4.x API. Geometry documents
remain schema 4; automation remains protocol 1; licensing remains MPL-2.0.
Names remain labels, while model-bound handles remain identity.

## Explicit ownership

```python
from anygeometry import (
    OverlapOwnershipPolicy, plan_coplanar_fragmentation,
    apply_coplanar_fragmentation,
)

plan = plan_coplanar_fragmentation(
    geometry, ordered_face_ids,
    ownership_policy=OverlapOwnershipPolicy.FIRST_SELECTED,
)
# Present plan.effects, plan.expected_fragment_counts and plan.blockers.
result = apply_coplanar_fragmentation(geometry, plan)
```

The direct `fragment_coplanar_overlaps` convenience API requires the same explicit
`ownership_policy`. Omitting it raises before mutation. The policy means the
first selected face owns each common cell: it supplies metadata and inherits
only its own groups/tags. A completely covered later Sheet can disappear.
The preview lists each source's metadata/groups/tags, owner Sheet/Part names,
post-edit FaceUse counts, removed Sheets, conflicting metadata, and common cells.
Boundary group/tag disposition and expected descendant counts are included too.
Selecting the policy acknowledges these consequences; it does not resolve
material/section conflicts by combining their values.

Planning is read-only and allocates no live IDs. Plans are ephemeral and bound
to UUID/revision, with a canonical SHA-256 digest checked against a fresh live
preview. A stale, tampered, wrong-model, or dependency-blocked plan cannot edit.
A successful apply is one outer transaction. Reapplying is stale. Failed commit
restores live state, but provisional allocator high-water marks remain monotonic.

Plans and result mappings are deeply immutable. Copy a mapping explicitly if
an application needs its own editable presentation data. Do not use plan-local
cell positions as persistent identifiers; use returned model handles/references.

New `geometry.fragment.overlaps` feature records require
`parameters={"ownership_policy": "first_selected"}`. Previously saved records
without that field replay with their historical first-selected semantics. This
does not automatically authorize a newly authored feature without the policy.
CONNECT uses its explicit connection policy and retains owner Sheets for shared
material; it must not be replaced downstream with first-selected fragmentation.

## Conservative boundaries

`find_coplanar_overlaps` still returns a tuple of `FaceOverlap` records when it
can provide a complete answer. Fixed sampled chords no longer establish curved
overlap or disjointness. Curved candidates go through `query_intersection`;
only complete certificates can dismiss disjoint or zero-area intersections.
There is no general certified curved-region area contract yet: such candidates
raise `OverlapQualificationError`, even when positive-area coincidence is known.
Its `candidate_pairs` are canonical model-bound handles and `diagnostics` are
immutable tuples. No partial success tuple is returned when any pair is unresolved.
An explicit curved tolerance that differs from the predicate's tolerance also
fails closed rather than being silently ignored.

Fragmentation still requires straight planar boundaries and the Shapely extra.
Member/attachment boundary dependencies requiring remapping are blocked, as are
Sheet removals with attachment/Junction dependencies. No coordinate-inferred
connections, new curve approximations, or material merging are introduced.

Face IDs supplied to Sheet creation or fragmentation must be existing positive
integer IDs, not booleans, strings, floating-point numbers, or wrong-kind refs.
Explicit fragmentation tolerance is an absolute model-unit length; omission uses
the model tolerance policy and the participating extent.

## Qualification and release

Focused evidence and remaining qualification are recorded separately; changing
the package version does not mean that 0.4.3 has been published. Existing 0.4.2
release authority, ledgers, and artifact hashes remain historical and must not
be repurposed. Publication requires new explicit authorization.

## Local qualification evidence

Base commit: `f2df459eebf160fe66dbaf47154699207d6c852b`; local qualification
was performed before committing. All tests below use repository source through pytest configuration,
not the machine's older ambient ANYgeometry 0.4.0 installation.

Final resource request `21a6706008744ec0b2f821c3f67c369a`, each command once,
stopping on nonzero:

1. `python -B -m pytest tests/test_overlap_safety.py -q`: **43 passed, 0.34s**.
2. `python -B -m pytest -q`: **601 passed, 140.30s**, Python 3.13.
3. `python -B tools/qualify_overlap_release.py`: **PASS**. Offline build,
   strict Twine checks, external-venv origin assertions, version/schema/typing,
   overlap application, CLI, and simulated Shapely import-absence checks passed.

The venv reused system NumPy offline but loaded ANYgeometry exclusively from
its own installed wheel. Executable, prefix, and module paths all resolved
beneath that exact venv and outside the repository. Only the verified temporary
venv child was removed; artifacts and `qualification.json` remain at:

```text
C:\Users\AudunArnesenNyhus\AppData\Local\Temp\anygeometry-043-qualification-__f_uyxt
```

Artifacts under its `dist` child:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| anygeometry-0.4.3-py3-none-any.whl | 361299 | e9e2ffe1230f66047fc64fa1da1ff5335b5d4cbe424d7fb6273121cd4a72be18 |
| anygeometry-0.4.3.tar.gz | 449310 | 32eb156363024e47630e9055db5993e6bcd0e354db34a219e5fff8ce6a8321e1 |

Tool versions: build 1.5.0, wheel 0.45.1, Twine 6.2.0, setuptools 80.9.0,
NumPy 2.4.3, Shapely 2.1.2. Build/smoke wall times and peak resources were not
measured separately. No timing or memory speedup is claimed. The locality
regression proves equal affected-edge and structural-validation work counts
with and without twelve unrelated remote plates, then checks full validation.

Preserved qualification history:

- An early broad curved-matrix invocation was interrupted after taking longer
  than anticipated; it provides no completion evidence.
- Request `c224a526149040d78d89de67fd403527`: earlier full suite passed
  595 tests in 139.64s, before final hostile/rollback coverage was added.
- Request `05801494a94647b5a9c8f25abe0db7cc`: 598 passed, 1 failed in
  138.48s. A warmed-cache rollback assertion exposed cross-kind cache
  invalidation. Packaging was correctly skipped. The correction and direct
  Face/Arc same-integer-ID regressions are included in the final passing run.
- Every approved run released its resource lock; no qualification process
  remains. No failed command was retried under its consumed request.

Remaining gate at the local qualification checkpoint: Python 3.11–3.14 hosted CI and its genuinely Shapely-absent
environment on the final committed tree. This requires separately authorized
commit/push/workflow activity. The locally simulated absent-extra check is not
represented as that hosted gate. The final committed tree's GitHub Tests check
records the hosted verdict. No release or PyPI publication was performed during
local qualification.
The existing `.idea/vcs.xml` remains unchanged (SHA-256
`D21341BDF42ABDA70DCA0099C442ED782906259FCF1882BE37467E858E78BF30`);
protected historical artifacts were not cleaned or uploaded.
