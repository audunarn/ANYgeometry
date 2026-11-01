# PP5c bounded constructor-state qualification

PP5 release 3, Part 1. Base b93f2715eefb66723352b64da7118ff711acf2bb;
branch claude/pp5-stage-partition-20261031. Independent review is pending.
This adds admission/transport only. Real pair/face stage integration is Part 2.

## Plan and acceptance

1. Qualify the real census's exact Cylinder schema, bind its derived state into
   transport and verify constructor reconstruction, with red/green refusal tests.
2. After the last Part 1 source edit, run PP1/planning, real-process, T2b, ring, seam
   and focused 57 tests; original sector60/r7/face200 OFF identities; original source
   and N2c closure admission/roundtrip. Commit Part 1 separately.
3. Then implement the real pair/face stage interfaces and A1-A7. Keep original serial
   reductions, typed refusals, sampled serial checks and mandatory CI. Record model
   retention/reset, transport, errors/deadlines and scheduling boundaries explicitly.
4. Verify sector60 then r7 serial/parallel equality, timing/load/balance and final
   required groups. No whole N2c planning here. PP3/PP4 retain their assigned ownership.

## Bounded domain and explicit residual risk

RAW NATIVE VIEWS (ctypes/buffer protocol over the same memory) are an ACCEPTED
RESIDUAL RISK. This is risk-accepted qualification, NOT a purity or aliasing proof.
The bounded domain is traversed model state, exact owner/base-reference counts,
NumPy ownership/base flags, weak-reference checks and a GC array census including
ndarray subclasses. The registered record class comes from the real census:

| Input | Class | Attribute | Count | Fresh-constructor byte matches |
|---|---|---|---:|---:|
| sector60/r7 shared source, 1083 faces | anygeometry.surfaces.Cylinder | _circumferential | 540 | 540 |
| N2c source, 561 faces | same | same | 18 | 18 |
| N2c structural closure, 561 faces | same | same | 18 | 18 |

Mitigations are (i) worker reconstruction through record constructors with fresh
arrays, (ii) sampled serial verification and (iii) the CI control. This adapter
supplies (i); it does not claim that direct freeze/thaw has itself performed (ii)
or (iii). PP2 execute retains mandatory CI and sampling; real plan/apply CI is
still an integration obligation. A host raw-memory alias is not transported.
The named documented-limitation test proves that a raw-address view can remain
undetected until it changes bytes. That passing test is NOT a no-alias pass claim.

## Exact adapter and transport

The complete Cylinder instance key set must equal its dataclass fields plus
_circumferential. Missing, extra, subclass and other derived-record schemas refuse.
The adapter calls the same exact Cylinder constructor using the record's own init
fields, after exact passive-field validation prevents unknown __array__/__float__
callbacks. Derived ndarray dtype, shape, C-contiguity, finiteness and all C-order bytes
must match. Signed zeros are compared by bytes. Every declared ndarray and float
must also match the fresh constructor bitwise; renormalization drift refuses.

Owner/base counts are compared through the same helper to a fresh unexported
constructor record. A fixed literal count is inappropriate under CPython 3.14's
borrowed-local execution. Ordinary outside references, views, weak references and
other-record/field aliases refuse. np.cross on this runtime returns a full-buffer
view backed by its single owning ndarray; that exact finite contiguous base pattern
is checked. Extra base chains, offsets/striding and extra owner references refuse.
GC-tracked subclass views are inspected through np.ndarray.view's base C method,
avoiding external __array_function__ overrides. Bounded shares_memory queries refuse
when too hard. The model-record walk keeps the existing 100000-object ceiling.
Unknown extra state and warm dataclass caches retain their previous refusals.

freeze_model audits original state before JSON can omit it. A constructor_state
manifest binds path, exact class/attribute, dtype, shape, contiguity, byte SHA256 and
writeability. Plain models without derived state retain the original wrapper shape.
The ordinary geometry serializer/schema is unchanged. The decoder runs from_dict's
record constructors, audits reconstructed values against the manifest, preserves a
source derived array's readonly flag on the fresh receiver, and verifies complete
manifest coverage. A constructor-bearing packet without the manifest refuses.
Metadata types are exact: integer/float equality cannot impersonate bool/int fields.
Checksums detect corruption; they are not authenticity proofs or native-memory proofs.

The existing PP2 extension rule still permits an absent source extension dictionary
to decode as the constructor-owned empty dictionary. The real closure roundtrip check
allows exactly this documented presence transition and compares all other packet
bytes exactly. It does not omit constructor state or weaken the JSON/value checks.

## Argument and limits

For each admitted record, stored bytes equal freshly computed bytes for every
supported field and the derived vector. Thus the tested binary-state difference
at this boundary is zero. No new bound on the floating-point error of the geometric
cross product is claimed; ordinary constructor arithmetic is retained unchanged.
Hash collision resistance is an engineering assumption, not a mathematical identity
proof. Full unchanged-output claims require the recorded numerical identity gates.
External mutation between checks and undetected raw views remain residual risks;
source revalidation, sampled serial execution and mandatory CI remain required.

A small real face-1 closure fixture replaces full ANYfem loading inside unit tests.
Origin file SHA256: 6943e22988b7d1767f77c9cf49ad4c017cfc82cc1cdf5d465605fb4140299dc0.
Canonical fixture document SHA256:
cfb258b3d624530ff70a8e57f02340a043dbd1729e999730b37340d76a4c5382.
It has one face and is transport-test data, not a new golden numerical result.
Full original-source and pinned N2c admission are separately receipted processes.
This avoids contaminating the baseline ledger tests with the ANYfem application
import closure. The initial combined failures remain in evidence, not discarded.

## Retained failures and corrections

- First red fixture used legacy arcs and populated _arc_cache; retained, then replaced
  with actual cold reference data without stripping caches. Valid baseline remained red.
- First candidate assumed a fixed reference count and the positive test accidentally
  held a copied array reference. Fixed using the fresh-owner control and identity-only
  before-state evidence. No alias allowance was widened to make an alias pass.
- Added red cases for weakrefs, tracked subclass views, exact transport metadata types
  missing registered derived state and pre-constructor foreign-field callbacks; each preserved and repaired without test skips.
- The first real admission assertion compared the extension presence flag too strictly;
  corrected to the already-authorized absent-to-empty rule, all other packet bytes exact.
- A temporary reporting variable shadowed the derived-record count with a packet body;
  that verbose log is retained on disk, not copied to the outbox. Final reporting is compact.
- Earlier PP5b prose said base=None. The actual test asserted flags unchanged, not base=None.
  Here np.cross yields OWNDATA=False with a full-buffer owning base. The original raw-view
  counterexample remains valid; the present adapter explicitly checks this base pattern.

No certified source pin, canary/golden, scientific tolerance or existing test changed.
No remote/push, whole-model planning or independent review is implied by this contract.
Part 2 scheduling, global budget/cancellation, loaded-host/memory admission, real plan
CI, timing and release claims require their own implementation and evidence.