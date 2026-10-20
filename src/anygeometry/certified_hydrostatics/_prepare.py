"""Preparation and ``check_preparation`` (design sections 2, 6 and 7).

``prepare_certified_hydrostatics`` reads the model once under a consistent-read guard (no open
transaction; the revision recorded before and after collecting must be equal), builds the exact
snapshot, decides the six admission checks and returns an immutable ``PreparedCertifiedHydrostatics``.
``check_preparation`` rebuilds the snapshot from the model it is given under the same guard and
compares the canonical digest and every identity field. ``GeometryModel`` is not thread-safe: no other
thread may mutate the model during these calls (a documented precondition, not detected beyond the
revision guard)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..serialization import _canonical_payload, _serialized_model_state
from ._admission import admit
from ._errors import CertifiedHydrostaticsRefusal, RefusalCategory, refuse
from ._identity import ImplementationIdentity, implementation_identity, sha256_chunks
from ._snapshot import ExactSnapshot, collect_snapshot
from ._work import Meter, WorkBudget

__all__ = ["CertifiedHydrostaticsPolicy", "PreparedCertifiedHydrostatics", "check_preparation",
           "prepare_certified_hydrostatics"]


@dataclass(frozen=True)
class CertifiedHydrostaticsPolicy:
    """Preparation policy recorded in the proof (limits the later proof parsing needs, design section 7)."""

    convention_version: str = "1"
    max_proof_bytes: int = 64 * 1024 * 1024
    max_integer_digits: int = 100_000

    def as_dict(self) -> dict:
        return {"convention_version": self.convention_version, "max_proof_bytes": self.max_proof_bytes,
                "max_integer_digits": self.max_integer_digits}


@dataclass(frozen=True)
class PreparedCertifiedHydrostatics:
    """An admitted, immutable snapshot of a face set; queries answer for the snapshot, not the live model."""

    face_ids: tuple
    units: str
    policy: CertifiedHydrostaticsPolicy
    model_id: str
    revision: int
    document_checksum: str
    snapshot_digest: str
    implementation: ImplementationIdentity = field(repr=False)
    snapshot: ExactSnapshot = field(repr=False, compare=False)
    work: dict = field(compare=False)

    def artifact_identity(self) -> dict:
        """The proof's ``artifact`` fields."""
        return self.implementation.as_dict()

    def model_identity(self) -> dict:
        """The proof's ``model`` fields."""
        return {"model_id": self.model_id, "revision": self.revision, "document_checksum": self.document_checksum,
                "face_ids": list(self.face_ids), "snapshot_digest": self.snapshot_digest}

    def proof_identity(self) -> dict:
        return {"artifact": self.artifact_identity(), "model": self.model_identity()}


def _document_checksum(model, meter: Meter) -> str:
    document = _serialized_model_state(model)
    digest = sha256_chunks(_canonical_payload(document), meter)
    if digest != document["checksum"]["value"]:
        raise RuntimeError("the geometry document checksum is not reproducible")   # an internal defect
    return digest


def _transaction_open(model) -> bool:
    return getattr(model, "_transaction_journal", None) is not None


def _consistent_read(model, faces: Iterable[int], meter: Meter):
    """``(snapshot, model id, revision, document checksum)`` read between two equal revisions."""
    if _transaction_open(model):
        raise refuse(RefusalCategory.CONCURRENT_MUTATION, "a model transaction is open")
    revision = model.revision
    snapshot = collect_snapshot(model, faces, meter)
    checksum = _document_checksum(model, meter)
    if _transaction_open(model) or model.revision != revision:
        raise refuse(RefusalCategory.CONCURRENT_MUTATION, "the model changed while it was being read")
    return snapshot, str(model.model_id), int(revision), checksum


def prepare_certified_hydrostatics(model, faces: Iterable[int], *, units: str = "m",
                                   policy: CertifiedHydrostaticsPolicy | None = None,
                                   budget: WorkBudget | None = None,
                                   cancellation_check: Callable[[], bool] | None = None
                                   ) -> PreparedCertifiedHydrostatics:
    """Snapshot ``faces`` of ``model`` exactly and admit them as the oriented boundary of a polyhedral solid.

    Raises ``CertifiedHydrostaticsRefusal`` (category, phase, work counts) when any admission check
    fails, the budget/deadline/bit cap/cancellation stops it, or the model changes during the read."""
    if not isinstance(units, str) or not units:
        raise TypeError("units must be a non-empty string")
    policy = CertifiedHydrostaticsPolicy() if policy is None else policy
    if not isinstance(policy, CertifiedHydrostaticsPolicy):
        raise TypeError("policy must be a CertifiedHydrostaticsPolicy or None")
    meter = Meter(budget, cancellation_check)
    faces = list(faces)
    request = {"faces": [f if isinstance(f, int) else repr(f) for f in faces], "units": units}
    identity = {"model_id": str(model.model_id)}
    try:
        implementation = implementation_identity(meter)
        snapshot, model_id, revision, checksum = _consistent_read(model, faces, meter)
        identity.update(revision=revision, document_checksum=checksum)
        digest = snapshot.digest(meter)
        admit(snapshot, meter)
    except CertifiedHydrostaticsRefusal as refusal:
        raise refusal.complete(phase="prepare", request=request, identity=identity,
                               work=meter.budget.counts())
    return PreparedCertifiedHydrostatics(
        face_ids=snapshot.face_ids, units=units, policy=policy, model_id=model_id, revision=revision,
        document_checksum=checksum, snapshot_digest=digest, implementation=implementation, snapshot=snapshot,
        work=meter.budget.counts())


def check_preparation(model, prepared: PreparedCertifiedHydrostatics, *, budget: WorkBudget | None = None,
                      cancellation_check: Callable[[], bool] | None = None) -> dict:
    """Rebuild the snapshot of ``prepared.face_ids`` from ``model`` and compare it with ``prepared``.

    Returns the work counts when the model is still the prepared one; raises ``STALE_PREPARATION`` when
    the snapshot digest or any identity field (model id, revision, document checksum, artifact) differs."""
    if not isinstance(prepared, PreparedCertifiedHydrostatics):
        raise TypeError("prepared must be a PreparedCertifiedHydrostatics")
    meter = Meter(budget, cancellation_check)
    request = {"faces": list(prepared.face_ids), "units": prepared.units}
    identity = prepared.model_identity()
    try:
        implementation = implementation_identity(meter)
        snapshot, model_id, revision, checksum = _consistent_read(model, prepared.face_ids, meter)
        digest = snapshot.digest(meter)
        differences = [name for name, now, before in (
            ("snapshot_digest", digest, prepared.snapshot_digest), ("model_id", model_id, prepared.model_id),
            ("revision", revision, prepared.revision), ("document_checksum", checksum, prepared.document_checksum),
            ("artifact", implementation.digest, prepared.implementation.digest)) if now != before]
        if differences:
            raise refuse(RefusalCategory.STALE_PREPARATION, "the model or artifact differs from the preparation: "
                         + ", ".join(differences))
    except CertifiedHydrostaticsRefusal as refusal:
        raise refusal.complete(phase="check", request=request, identity=identity, work=meter.budget.counts())
    return meter.budget.counts()
