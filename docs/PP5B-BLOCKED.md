# PP5b / PP5 release 2: blocked on unrestricted outside-alias admission

Base 5036559b872f58508292ebd19afb85aceceb81c9; branch
claude/pp5-stage-partition-20261031. This is an evidence-only handoff, not an
implemented or accepted Part 1 adapter. Part 2 was not started.

## Intended order and acceptance

1. Census real extra state; write baseline acceptance/refusal tests.
2. Admit exact registered derived state only with all release predicates proven,
   including byte equality and absence of aliases inside AND outside the model.
3. Commit qualified Part 1 and run the required groups/identities before claiming it.
4. Implement PP5 stages and A1-A7, then serial/parallel subset equality and measurements.

Step 1 completed. Step 2 has an unresolved alias-domain prerequisite. No production
source, existing test, tolerance, source pin, or admission rule was changed.

## Real census and constructor recomputation

| Input | Exact class | Extra | Count | Byte-equal fresh recomputations | Mismatches |
|---|---|---|---:|---:|---:|
| sector60/r7 shared 1083-face source | anygeometry.surfaces.Cylinder | _circumferential | 540 | 540 | 0 |
| N2c source (561 faces) | same | same | 18 | 18 | 0 |
| N2c structural closure (561 faces) | same | same | 18 | 18 | 0 |

No other extra-bearing record class was found in the declared-state traversal.
Each reconstruction called the actual class constructor with its own init fields.
Comparison checked dtype, shape, C-contiguity, finite-value flag and raw C-order
bytes, with signed zero distinguished. It also required a nonwriteable source's
reconstructed extra to remain nonwriteable. Results are a census over these inputs,
not a claim about every Cylinder or all aliases. All measured extras were finite.
N2c file is pinned to SHA256
969cc5f9c8efe5b628686c9a89b1de02ea2dea8c1618f4295cb049c2621f468a.
No N2c planning, apply or meshing ran; only load, closure extraction and census.

## Exact outside-alias counterexample

The release requires the extra to be "unaliased (no sharing with other records or
outside)". A model graph walk cannot account for arbitrary external views. The
passing test constructs an ordinary live Cylinder and a valid ctypes from_address
view of its own live _circumferential array, followed by np.ctypeslib.as_array.
This accesses only the test's own allocated memory, with its owner kept alive.

The external NumPy view shares memory (np.shares_memory is True), but:
- the owner's sys.getrefcount is unchanged;
- dtype, shape, C-contiguity, OWNDATA, writeability, base=None, finite values and
  every byte of the owner are unchanged, and fresh recomputation still matches;
- the external ndarray is not GC tracked and is absent from gc.get_objects();
- definition_checksum is unchanged.

Writing 2 into the external view's y component changes evaluate(1/4,0).y from 1
into 2 while the definition checksum remains unchanged. Thus ownership flags,
refcounts, model traversal, GC-array census and recomputation equality do NOT prove
the complete outside-alias predicate. This is a demonstrated limitation of those
candidate checks, not a universal claim that every conceivable provenance mechanism
is impossible. No complete external-memory provenance mechanism is supplied by the
current gate; PP4 host/native admission remains outside this task's implementation.

Real arithmetic argument: p=o+r(cos(theta)d+sin(theta)c)+vh*a; varying c by delta
changes p by r*sin(theta)*delta. The unit-axis regression checks the exact observed
binary y values 1 and 2. There is no new numerical certificate, tolerance relaxation
or general floating-point error bound. The mismatch does not rely on a near miss.

A new release must either establish a trusted ownership/provenance boundary covering
external aliases, or explicitly bound the alias contract and accept the remaining
opaque/native-view risk. Do not silently reinterpret "outside" as only references
reachable from the model. Merely copying or recomputing a worker's record does not
prove that the original host record has no such aliases; no original state was stripped.
Only an explicit changed contract can permit that weaker claim. Ordinary Python/NumPy
references and intra-model aliases would still require checked refusals.

## Tests and failures

Baseline tools/pp5b_admission_contract_red.py is explicitly invoked and outside normal
tests/ discovery. Final baseline: 1 failed, 12 passed. Positive pristine constructor
admission fails at the existing q._check_model_record_state:416. Eleven negative cases
remain refused: edited extra, edited fields, shape, dtype, nonfinite, other-record alias,
outside reference, outside view, signed zero, unknown extra and unregistered class.
These pass because the base already refuses all extra state, not because a new adapter
was implemented. The twelfth pass is the outside raw-view counterexample.

First invocation: 2 failed, 11 passed. In addition to the intended baseline failure,
the edited-fields setup tried writing a read-only radial array. Exact source is retained
in pp5b-contract-red/test-attempt1.py. Repair replaced the field in the test with a
new finite vector using object.__setattr__, without regenerating the extra; its refusal
then passed. No production fix, skipped test or softened assertion. One setup repair;
no further attempts. Known pyreadline3 finalizer WinError appeared after pytest; receipt
exit codes distinguish the deliberately failing baseline from successful census.

## Evidence and not checked

- .pp1-evidence/pp5b-census/census.json and result.json: successful census, zero recomputation
  mismatches, runtime source inputs unchanged, zero owned processes after cleanup.
- pp5b-contract-red and pp5b-contract-red-02 retain complete stdout/stderr and receipts.
- PP5B-MANIFEST.json and PP5B-COMMANDS.json bind artifacts and exact child commands.
- Existing production src/ and tests/ are unchanged against the base; this is source
  identity, not fresh execution of the golden numerical cases. Prior 46-pass admission
  evidence remains prior-release evidence, not a new PP5b run.

NOT CHECKED: green Part 1 admission/freeze_model for the real inputs, independent review,
full PP1/planning/T2b/ring/seam/57 groups, OFF golden identities, Part 2 integration,
A1-A7 fixes, serial/parallel equality, timing/load/balance, eight/sixteen workers,
PP3/PP4 code interfaces, whole N2c planning and publication. Reason: stop at the
unproven mandatory alias predicate without shipping a misleading partial acceptance.
Implemented: diagnostic code only. Tested: census and red contracts/counterexample.
Reviewed: self-inspection only. Pushed: no. No external worker or remote operation.