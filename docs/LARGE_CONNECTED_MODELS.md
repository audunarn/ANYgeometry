# Large connected model development checks

The fixtures in `tools/general_intersections/large_connected_fixtures.py` use
public geometry authoring APIs. They are development evidence, not a new release
or a blanket claim that every surface pairing is supported. Nonparallel cubic
wall pairs remain outside this stage. Schema and package version are unchanged.

| Builder | Authored operands | Independent expectations |
|---|---:|---|
| `connected_strip(n)` | n, even | Adjacent unit panels, alternating beam/shell stiffeners; exact straight joint intervals and material areas |
| `connected_hub(n)` | n + 1 | One host with n incident stiffeners; n²/4 canonical crossing vertices |
| `connected_mixed(n)` | n, multiple of 10 | Connected floor bays, polynomial walls, eight cylinder/cone panels per bay; analytic quadric areas and independently integrated wall areas |
| `concave_boundary_junctions()` | 7 | Concave host with circular hole, interior-ended and boundary members, crossing shells and high-valence vertices |

Generated panels count individually. Mixed bays cycle through a cubic wall with
an oblique cylinder, a parabolic wall with a cylinder, and a cubic wall with a
cone. Expected exact edge families include BezierQuadricCurve and
QuadricIntersectionCurve. Curve incidence is checked; this fixture alone is not
an independent oracle for completeness of every analytic branch. Existing exact
curve regressions remain necessary.

## Running proportionately

Routine regression coverage is `tests/test_large_connected_contract.py`. It covers
small representatives, translated bays, operand-order invariance, material,
authored Sheet/FaceUse preservation, boundary-member attachment coordinates and
planning budget/cancellation behavior. Shared numerical code changes still require
the repository's normal full development gate.

Run a growing fixture separately, using the intended Python environment:

```powershell
python tools/general_intersections/check_large_connected.py --family strip --count 100 --report reports/strip100.json --deadline-seconds 120
python tools/general_intersections/check_large_connected.py --family hub --count 100 --report reports/hub100.json --deadline-seconds 120
python tools/general_intersections/check_large_connected.py --family mixed --count 100 --report reports/mixed100.json --deadline-seconds 120 --export reports/mixed100
```

`--profile path.prof` records planning call counts and a profile. Profiled times
must not be compared as if they were unprofiled benchmarks. The runner records
source paths/hashes, stage times, topology counts and process peak memory. It
rejects optimized Python, so disabled assertions cannot produce accepted output.
Exported authored/prepared documents and the expectation manifest are written
only after all checks pass.

For the 1,000-operand mixed case, the current development execution uses an
840-second geometry cancellation deadline inside a 900-second external process
ceiling. The external ceiling must also cover document round-trips and export;
the runner's callback alone does not interrupt those operations. Preserve failures
and investigate the focused reproducer before repeating a large failed case.

## Consumer acceptance

ANYmesher must use the exact candidate artifact and these portable inputs through
the public preparation/chart APIs. Acceptance requires a `ready` automatic mesh,
all shared edge node sequences present in incident faces/members, retained source
associations, unchanged authored geometry and the existing linear/high-order
quality gates. An inspection result or successful chart query is insufficient.
The full-scale linear and smaller mixed quadratic results remain separate from
geometry preparation evidence.

`tools/general_intersections/check_large_connected_mesh.py` provides the public
consumer probe with `--family`, `--count`, `--order`, `--target-size` and `--report`.
Its `--max-seconds` bounds mesher recovery; use an external process ceiling for
the complete preparation-plus-meshing run. Record exact geometry/mesher commits
and wheel hashes alongside its import origins. Source imports are development
integration evidence and must not be labelled clean installed-package checks.

The living capability record, including retained failures and exact evidence
locations, is `reports/general_intersections/quadric_generalization_study.md`.

## Certified corner finalization

`set_prepared_face_corners(model, {face_id: corner_indices})` lets a consumer
finalize mapped corner labels without discarding a valid material-preparation
proof. It requires a current complete receipt, exact built-in Plane, Cylinder,
Cone or ExtrudedSurface supports, and no separate parameterization on edited
faces. The owner stages the complete update and checks document equality except
for the requested corner fields and revision/checksum. Invalid batches leave the
model unchanged. Existing transactions and stale proofs are rejected.

It preserves only the preparation proof; old plans remain stale. Ordinary edits
still invalidate proof. As with batch application, change hooks run before the
new receipt is published and see a fail-closed query; callers query after return.
This operation does not admit a mesh or relax mapping and element-quality gates.
