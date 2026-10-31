# PP1i review fixes: plan and acceptance

Ninth release; parent 41c20b0a; PP1 worktree only. No PP2 writes, remote or whole
model. Tests first, then fixes, then PP1/191/57 and the three unchanged identities.

1. Stratified sampler with actual worker/position/class metadata and plan-digest
   offsets. Verify every worker's last task and each worker/class; deterministic
   late samples and recorded selection. Preserve the legacy single-stream API
   and its tests, but require the new metadata in mandatory production CI mode.
2. Structural guard for the owned annotation dictionary and exact identities
   for all seven trusted stores. Retain runtime risk, with corrected provenance.
3. Optional mandatory CI, opt-in selector versus numerical-setting distinction,
   accurate exception contract, optional shared adjacent boundary, real verifier
   settings/thread negatives and live-census rebinding/new-store tests.
4. Exact evidence/commit handoff; independent acceptance remains pending.

No new numerical certificate, tolerance change, guard refresh or universal
purity claim. Batch transient mutation remains an explicitly accepted risk.

The new run_qualified task_layout records (worker id, execution position, pair or
face) for EVERY ordinal. Positions must uniquely cover 0..Nw-1 per worker. The
sampler always chooses the last task in each worker and the last task of each
class present in that worker. It fills to at least min(Nw,max(2,ceil(Nw/20))) per
worker, using the late half rotated by SHA256(plan_digest + ':' + worker_id),
then earlier positions if needed. The digest, offset and selected ordinals are
returned for audit. The finite coverage proof is direct construction, with no
floating-point arithmetic/error. No probability or unsampled-accuracy bound is
claimed. A supplied layout with bad schema or incomplete coverage declines.

Legacy sample_indices/count-only calls retain their old behavior solely for
compatibility with the existing diagnostic tests. They do NOT provide worker
coverage. require_ci=True is the production/PP6 entry contract: BOTH plan/apply
controls AND dispatch metadata/digest are mandatory before worker startup. The
executor must supply actual scheduling metadata; inferred round-robin is not
used. The CI control remains unexecuted when both arguments are absent and the
mandatory flag is false. Its verification result is explicitly returned.

Five namespace stores must be the actual module dictionaries of builtins and
numpy._core.{fromnumeric,shape_base,einsumfunc,_asarray}. finfo must be the exact
np.finfo cache owned by the captured class. The annotation dictionary must be the
exact _BranchConstants annotate-function closure. That last dictionary protects
ordered keys, exact ForwardRef class, identity, expression, flags and opaque
reference identities structurally; it never evaluates annotations or traverses
the referred globals. The six runtime dictionaries remain opaque content risks.
Versions/binaries/source files identify provenance, not mutable dictionary bytes.

The opt-in environment variable selects host dispatch only and is excluded from
new numerical settings captures. Actual mode/seam/workers/native-thread settings
remain pinned. Old saved references are not rewritten or refreshed: their stale
settings schema must be recaptured by a separately authorised admission flow.
The absent-selector baseline then env=1 path is tested with real capture/compare
logic and isolated image-discovery fixtures, not a new production process.

S4: QualificationDecline follows serial fallback after retirement. ValueError
and other genuine task errors keep their original object/type and propagate with
no serial replay. BaseException cancellation clears staging and propagates.
PP3 must supply the promised serial error attribution; this slice does not.
