# PP5 first release: blocked at real-model admission

Base: 5c3fc82d7ca155aeb34a913a5241e3bdb9e222bb.
Branch: claude/pp5-stage-partition-20261031.
Status: inspection and diagnostic tools only; no production implementation.

## Intended slices and acceptance

1. Establish real input admission and scale measurements; preserve every existing guard.
2. Add red tests for real pair/face transport, retained worker models, ordinal reduction,
   largest-first scheduling, terminal deadlines and infrastructure refusal.
3. Implement explicit stage seams and qualified transport, preserving ordinary serial
   reductions and apply. PP3 owns error attribution/cooperative cancellation/budgets;
   PP4 owns memory/loaded-host admission. No frame inspection or O(n^2) replacement sweep.
4. Run final PP1/planning/T2b/ring/seam/57 groups, OFF identities and serial-versus-eight
   worker sector60/r7 controls with load, timing and balance receipts.

Slice 1 exposed an admission prerequisite. Slices 2-4 are NOT implemented or verified.
No whole-model planning or meshing was executed. N2c was loaded/encoded/counted only.

## Observed blocker

freeze_model -> check_types -> _check_model_record_state refuses the original source
model used by BOTH subset observers with QualificationDecline:
`non-field model record instance state`.
The pristine Cylinder constructor creates `_circumferential = np.cross(axis, radial)`
(surfaces.py:133), but this is not a dataclass field. The input guard at
_planning_qualification.py:415-416 rejects it before model transport. There are 540
such record extras in the subset source and 18 in the N2c source/closure.
The six model-level cache collections are empty: this is not a warm top-level cache.
A pristine isolated Cylinder reproduces the same refusal.

This is NOT fixed by configurable size/count limits. The task assigns admission work
to PP4 and forbids weakening gates. A2 authorizes configurable resource ceilings, not
an implicit exemption for omitted record state. Do not strip/redecode the original
host object, ignore extra fields, or mark the gate green because the JSON matches.

## Checked semantic counterexample

For a cylinder, p(u,v) = o + r[cos(theta)d + sin(theta)c] + vh a, where
c is the stored circumferential vector. Holding all declared fields fixed and changing
c by delta changes p by r sin(theta) delta in real arithmetic. The test sets the stored
y component from 1 to 2 with unit axis/radial directions, r=1, u=1/4, v=0. Evaluation
returns y=1 then y=2 while definition_checksum is identical. The existing gate refuses
both the pristine and altered instances. This is an observed binary regression and an
algebraic dependence argument, not a new numerical certificate or an error bound for
arbitrary axes. No tolerance, rounding-error budget or source pin was changed.

Needed next release: an exact reviewed constructor-state adapter (or explicit authority
to implement it), preserving unknown-field refusal and detecting mutation, dtype/shape,
nonfinite values, signed-zero differences and alias hazards. Reconstruction must be
shown compatible with original H evaluation; c=fl(cross(a,d)) must not be assumed
bit-identical after repeated normalization without checking. PP4 integration or a
narrow schema qualification is preferable to an unguarded PP5 bypass. This resolves
only the first observed prerequisite; other stage/transport issues remain untested.

## Final input-size observations

The original subset reference uses the entire 1083-face source model with 97 or 240
operand faces; it does not pass a physically reduced model. N2c source SHA256 is
969cc5f9c8efe5b628686c9a89b1de02ea2dea8c1618f4295cb049c2621f468a.

| Input | Faces | JSON document bytes | Prospective model payload bytes | Record-walk objects | Serialized type-walk objects |
|---|---:|---:|---:|---:|---:|
| sector60/r7 source | 1083 | 846909 | 847034 | 58793 | 26107 |
| N2c source | 561 | 895509 | 895634 | 63983 | 31318 |
| N2c structural closure | 561 | 837833 | 837897 | 86090 | 26469 |

Payload limit=4000000 bytes; record/type walks=100000 objects; all above fit. Record
walk counts follow identity-deduplicated declared model state, not the full interpreter
heap. The type walk here is the serialized document only; the full model gate declines
before reaching it. Wrapper sizes are prospective, not accepted transport packets.
The 3-face diagnostic task limit still excludes every case; the 100 s planner limit
was inspected, not exercised. Configurability, outer request/result-envelope growth,
real stages and deadline behavior were not implemented or certified.

## Evidence and checks

- `.pp1-evidence/pp5-input-final/sizes.json`: final size/admission measurements;
  result.json passed, input files unchanged, owned cleanup verified zero.
- `pp5-admission-contract-red`: 3 failed, 1 passed on unchanged production base;
  two subset input contracts and pristine Cylinder admission fail for the stated reason.
  The semantic counterexample passes. This explicit diagnostic test file is outside
  normal tests/ discovery and remains deliberately red, not skipped or xfailed.
- `pp5-preserved-admission`: 46 passed, 0 failed across qualification/review/transport
  schema tests. Existing rejection tests are untouched.
- First preflight mislabelled the original project's closure as N2c. Retained as
  preliminary evidence only, with exact probe-v1.py. Second preflight binds the actual
  N2c file and 561-face closure; retained probe-v2.py. Final adds exact type-walk and
  wrapper counts. These are three successful diagnostic refinements, not failed
  implementation attempts. There is one deliberately red pytest invocation.
- Python's known pyreadline3 finalizer emitted WinError 0xc0000008 after pytest;
  receipt exit codes distinguish the expected red test run from the 46-pass run.

Production src/ and existing tests/ have zero diff against the base. No new ordinary
output identity claim is made: the three golden numerical gates were not rerun because
there is no production change and admission blocks PP5. Historical PP2 evidence is
not relabelled as PP5 execution. No timing/speed/balance claim is made.

NOT CHECKED: production stages, decode-once compatibility with per-task reset, new
codec/cache/alias semantics, A2 configurable limits, A3 deadlines, A4 infrastructure
fallback, A5 integrity/thread fixes, A6/A7 scheduling/timing, real worker ordering,
serial-v-parallel plan/apply equality, full final groups, golden identities, eight or
sixteen workers, PP3/PP4 hooks in code, independent Opus/forseti review and whole N2c
planning. Reason: preserve admission boundary and stop this release at the demonstrated
prerequisite. No worker was launched, no push/remote operation, no source pin refreshed.