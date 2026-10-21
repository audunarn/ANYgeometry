"""Preparation and ``check_preparation`` (design sections 2, 6 and 7).

``prepare_certified_hydrostatics`` reads the model once under a consistent-read guard (no open
transaction; the revision recorded before and after collecting must be equal), builds the exact
snapshot, decides the six admission checks and returns an immutable ``PreparedCertifiedHydrostatics``.
``check_preparation`` rebuilds the snapshot from the model it is given under the same guard and
compares the canonical digest and every identity field. ``GeometryModel`` is not thread-safe: no other
thread may mutate the model during these calls (a documented precondition, not detected beyond the
revision guard)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable

import numpy as np

from ..errors import GeometryError
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
    """Preparation policy recorded in the proof (limits the later proof parsing needs, design section 7).

    The curved-class fields (CH3 design D-CH3-2 and D-CH3-7) are used only by the curved admission and queries;
    ``as_dict`` records them only for another ``convention_version`` than ``"1"``, so planar proofs, digests and
    defaults are unchanged. ``delta_hint = 2**-delta_hint_bits * (1 + max |coord|)`` (world units, max-norm) and
    ``delta_chart = 2**-delta_chart_bits`` (knot units) are the tolerances of the vertex and domain hints."""

    convention_version: str = "1"
    max_proof_bytes: int = 64 * 1024 * 1024
    max_integer_digits: int = 100_000
    delta_hint_bits: int = 40
    delta_chart_bits: int = 40
    max_bits: int = 4096
    max_isolation_bits: int = 4096
    max_taylor_order: int = 16
    max_pieces_per_integral: int = 1024
    max_refinement_rounds: int = 8
    default_enclosure_digits: int = 12

    _CURVED = ("delta_hint_bits", "delta_chart_bits", "max_bits", "max_isolation_bits", "max_taylor_order",
               "max_pieces_per_integral", "max_refinement_rounds", "default_enclosure_digits")

    def __post_init__(self) -> None:
        for name in self._CURVED:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def as_dict(self) -> dict:
        planar = {"convention_version": self.convention_version, "max_proof_bytes": self.max_proof_bytes,
                  "max_integer_digits": self.max_integer_digits}
        if self.convention_version == "1":
            return planar
        return {**planar, **{name: getattr(self, name) for name in self._CURVED}}

    def kernel_limits(self):
        """The CH3 kernel limits of this policy (``max_bits`` belongs to the work budget, see ``new_budget``)."""
        from ._ch3_poly import KernelLimits
        return KernelLimits(self.max_isolation_bits, self.max_taylor_order, self.max_pieces_per_integral,
                            self.max_refinement_rounds)

    def new_budget(self, max_operations: int | None = None, deadline: float | None = None) -> WorkBudget:
        """A work budget carrying this policy's ``max_bits``."""
        return WorkBudget(max_operations, self.max_bits, deadline)


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


def _meter_elements(document, meter: Meter) -> None:
    """One unit per scalar/array element of the encoded document (polled once per container)."""
    pending = [document]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            children = list(node.values())
        elif isinstance(node, (list, tuple)):
            children = list(node)
        else:
            continue
        meter.unit("document_element", len(children) + 1)
        pending.extend(child for child in children if isinstance(child, (dict, list, tuple)))


# every record collection the library encoder walks (serialization._serialized_model_state), plus the feature history
_MODEL_COLLECTIONS = ("vertices", "edges", "faces", "parts", "sheets", "face_uses", "coedges", "members",
                      "member_edge_uses", "attachments", "junctions", "groups", "tags", "features")


def _document_checksum(model, meter: Meter) -> str:
    """The model document's checksum. The library encoding is one bulk operation (design amendment 6b): one
    ``document_entity`` unit per model entity is charged before it starts, and a checkpoint follows it."""
    for name in _MODEL_COLLECTIONS:
        try:
            count = len(getattr(model, name))
        except (AttributeError, TypeError):
            count = 0
        for _ in range(count):
            meter.unit("document_entity")
    meter.unit("document_encode")
    try:
        document = _serialized_model_state(model)
        payload = _canonical_payload(document)
    except (GeometryError, ValueError, RecursionError, TypeError) as error:
        broken = any(not np.all(np.isfinite(vertex.position)) for vertex in model.vertices.values())
        raise refuse(RefusalCategory.NONFINITE_INPUT if broken else RefusalCategory.INVALID_QUERY,
                     "the model document cannot be encoded: " + str(error), error)
    # 6b: the encoding's size includes free-form content (units, CRS, metadata, names) that entity counts do not
    # bound; it is charged per 64 KiB immediately after the encoding, before anything else uses it
    meter.unit("document_bytes", len(payload) // 65536 + 1)
    _meter_elements(document, meter)         # the encoder itself is one non-preemptible call
    digest = sha256_chunks(payload, meter)
    if digest != document["checksum"]["value"]:
        raise RuntimeError("the geometry document checksum is not reproducible")   # an internal defect
    return digest


def _transaction_open(model) -> bool:
    return getattr(model, "_transaction_journal", None) is not None


def _consistent_read(model, faces: Iterable[int], meter: Meter, policy=None):
    """``(snapshot, model id, revision, document checksum)`` read between two equal revisions."""
    if _transaction_open(model):
        raise refuse(RefusalCategory.CONCURRENT_MUTATION, "a model transaction is open")
    revision = model.revision
    snapshot = collect_snapshot(model, faces, meter, policy)
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
    requested: list = []
    try:
        for face in faces:                   # one unit per requested id: an endless iterable still stops
            meter.unit("face_request")
            requested.append(face)
    except CertifiedHydrostaticsRefusal as refusal:
        raise refusal.complete(phase="prepare", request={"units": units}, work=meter.budget.counts())
    faces = requested
    request = {"faces": [f if isinstance(f, int) else repr(f) for f in faces], "units": units}
    identity = {"model_id": str(model.model_id)}
    try:
        implementation = implementation_identity(meter)
        snapshot, model_id, revision, checksum = _consistent_read(model, faces, meter, policy)
        identity.update(revision=revision, document_checksum=checksum)
        digest = snapshot.digest(meter)
        if snapshot.curved is not None and policy.convention_version == "1":
            policy = replace(policy, convention_version="2")      # curved faces: convention 2 (design 3.4)
        admit(snapshot, meter, policy)
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
        if prepared.snapshot.digest(meter) != prepared.snapshot_digest:
            raise refuse(RefusalCategory.STALE_PREPARATION, "the stored snapshot does not match its digest")
        snapshot, model_id, revision, checksum = _consistent_read(model, prepared.face_ids, meter,
                                                                  prepared.policy)
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
