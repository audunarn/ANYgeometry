"""Stable, fail-closed result types for strict geometry audits.

This module contains reporting policy and orchestration primitives, not model
knowledge or geometric predicates.  A model integration supplies explicit
identity/revision getters plus side-effect-free checker callables.  Checkers
can therefore be developed independently and can use either a full-model or
changed-region spatial query without introducing an import cycle.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
import hashlib
import json
import math
from numbers import Real
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

__all__ = [
    "AuditCheck",
    "AuditCode",
    "AuditCollector",
    "AuditContext",
    "AuditEntity",
    "AuditIssue",
    "AuditMetrics",
    "AuditPolicy",
    "AuditReport",
    "AuditScope",
    "AuditSeverity",
    "AuditWitness",
    "BroadPhaseDiagnostics",
    "HandleLike",
    "run_audit",
]


Vector3: TypeAlias = tuple[float, float, float]
ModelT = TypeVar("ModelT")


def _text(value: object, *, name: str) -> str:
    if isinstance(value, Enum):
        value = value.value
    result = str(value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _non_negative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _point3(value: Sequence[Real]) -> Vector3:
    if len(value) != 3:
        raise ValueError("an audit witness point must have three components")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError("audit witness coordinates must be finite")
    return result  # type: ignore[return-value]


class AuditSeverity(IntEnum):
    """Severity is ordered so policy thresholds remain explicit."""

    INFO = 10
    WARNING = 20
    ERROR = 30
    BLOCKER = 40


class AuditCode(str, Enum):
    """Stable machine-readable strict-audit issue codes."""

    VERTEX_COINCIDENCE = "vertex_coincidence"
    VERTEX_EDGE_T_JUNCTION = "vertex_edge_t_junction"
    EDGE_CROSSING = "edge_crossing"
    EDGE_DUPLICATE = "edge_duplicate"
    EDGE_REVERSED_DUPLICATE = "edge_reversed_duplicate"
    EDGE_COLLINEAR_OVERLAP = "edge_collinear_overlap"
    MEMBER_MEMBER_CROSSING = "member_member_crossing"
    MEMBER_MEMBER_OVERLAP = "member_member_overlap"
    JUNCTION_INCONSISTENT = "junction_inconsistent"
    MEMBER_FACE_EMBEDDED = "member_face_embedded"
    MEMBER_FACE_BOUNDARY_COINCIDENT = "member_face_boundary_coincident"
    MEMBER_FACE_CROSSING = "member_face_crossing"
    ATTACHMENT_INCONSISTENT = "attachment_inconsistent"
    FACE_FACE_CROSSING = "face_face_crossing"
    FACE_COPLANAR_OVERLAP = "face_coplanar_overlap"
    FACE_CONTAINMENT = "face_containment"
    FACE_COINCIDENT = "face_coincident"
    SHEET_ORIENTATION = "sheet_orientation"
    SHEET_NON_MANIFOLD = "sheet_non_manifold"
    NONCONFORMAL_INTERFACE = "nonconformal_interface"
    SLIVER = "sliver"
    UNRESOLVED_LINEAGE = "unresolved_lineage"
    UNOWNED_STRUCTURAL_USE = "unowned_structural_use"
    MULTIPLY_OWNED_STRUCTURAL_USE = "multiply_owned_structural_use"
    INTENTIONAL_COINCIDENCE = "intentional_coincidence"
    UNCLASSIFIED_CANDIDATE = "unclassified_candidate"
    UNVERIFIED_CLASSIFICATION = "unverified_classification"
    SPATIAL_INDEX_INCONSISTENT = "spatial_index_inconsistent"
    CHECK_FAILED = "check_failed"


class AuditScope(str, Enum):
    FULL_MODEL = "full_model"
    CHANGED_REGION = "changed_region"


@runtime_checkable
class HandleLike(Protocol):
    """Structural protocol used to decouple reports from model handle classes."""

    @property
    def model_id(self) -> object: ...

    @property
    def kind(self) -> object: ...

    @property
    def id(self) -> int: ...


@dataclass(frozen=True, slots=True, order=True)
class AuditEntity:
    """Canonical model-bound entity locator stored in audit reports."""

    model_id: str
    kind: str
    id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, name="model_id"))
        object.__setattr__(self, "kind", _text(self.kind, name="entity kind"))
        _non_negative_integer(self.id, name="entity id")

    @classmethod
    def from_handle(cls, handle: HandleLike) -> "AuditEntity":
        """Normalize any public handle satisfying ``HandleLike``."""

        try:
            return cls(str(handle.model_id), _text(handle.kind, name="kind"), handle.id)
        except AttributeError as exc:
            raise TypeError("handle must expose model_id, kind, and id") from exc

    @classmethod
    def from_key(cls, model_id: object, key: tuple[str, int]) -> "AuditEntity":
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("entity key must be a (kind, integer_id) tuple")
        return cls(str(model_id), key[0], key[1])

    def to_dict(self) -> dict[str, object]:
        return {"model_id": self.model_id, "kind": self.kind, "id": self.id}


@dataclass(frozen=True, slots=True, order=True)
class AuditWitness:
    """A labelled, finite 3D point supporting an audit finding."""

    label: str
    point: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "point", _point3(self.point))

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "point": list(self.point)}


def _canonical_detail(value: object) -> str:
    """Convert a JSON-compatible detail to stable compact text."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("audit detail values must be finite JSON values") from exc


@dataclass(frozen=True, slots=True)
class AuditIssue:
    """One immutable, canonically ordered audit finding."""

    code: AuditCode
    severity: AuditSeverity
    message: str
    entities: tuple[AuditEntity, ...] = ()
    witnesses: tuple[AuditWitness, ...] = ()
    classification: str | None = None
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        code = self.code if isinstance(self.code, AuditCode) else AuditCode(self.code)
        severity = (
            self.severity
            if isinstance(self.severity, AuditSeverity)
            else AuditSeverity(self.severity)
        )
        message = str(self.message)
        if not message:
            raise ValueError("an audit issue message must not be empty")
        entities = tuple(sorted(self.entities))
        if any(not isinstance(entity, AuditEntity) for entity in entities):
            raise TypeError("entities must contain AuditEntity values")
        witnesses = tuple(sorted(self.witnesses))
        if any(not isinstance(witness, AuditWitness) for witness in witnesses):
            raise TypeError("witnesses must contain AuditWitness values")
        classification = (
            None
            if self.classification is None
            else _text(self.classification, name="classification")
        )
        normalized_details: list[tuple[str, str]] = []
        seen: set[str] = set()
        for key, value in self.details:
            detail_key = _text(key, name="detail key")
            if detail_key in seen:
                raise ValueError(f"duplicate audit detail key: {detail_key!r}")
            seen.add(detail_key)
            if not isinstance(value, str):
                raise TypeError("canonical audit detail values must be strings")
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                # Direct construction remains ergonomic for a plain string;
                # ``create`` already supplies canonical JSON text.
                decoded = value
            normalized_details.append((detail_key, _canonical_detail(decoded)))
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "witnesses", witnesses)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "details", tuple(sorted(normalized_details)))

    @classmethod
    def create(
        cls,
        code: AuditCode,
        severity: AuditSeverity,
        message: str,
        *,
        entities: Iterable[AuditEntity] = (),
        witnesses: Iterable[AuditWitness] = (),
        classification: object | None = None,
        details: Mapping[str, object] | None = None,
    ) -> "AuditIssue":
        canonical_details = ()
        if details:
            canonical_details = tuple(
                (str(key), _canonical_detail(value)) for key, value in details.items()
            )
        return cls(
            code=code,
            severity=severity,
            message=message,
            entities=tuple(entities),
            witnesses=tuple(witnesses),
            classification=(
                None
                if classification is None
                else _text(classification, name="classification")
            ),
            details=canonical_details,
        )

    @property
    def sort_key(self) -> tuple[object, ...]:
        """Stable ordering with the most severe issues first."""

        return (
            -int(self.severity),
            self.code.value,
            self.entities,
            self.classification or "",
            self.witnesses,
            self.message,
            self.details,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "severity": self.severity.name.lower(),
            "message": self.message,
            "entities": [entity.to_dict() for entity in self.entities],
            "witnesses": [witness.to_dict() for witness in self.witnesses],
            "classification": self.classification,
            "details": {key: json.loads(value) for key, value in self.details},
        }


_FAIL_CLOSED_CODES = frozenset(
    {
        AuditCode.UNCLASSIFIED_CANDIDATE,
        AuditCode.UNVERIFIED_CLASSIFICATION,
        AuditCode.SPATIAL_INDEX_INCONSISTENT,
        AuditCode.CHECK_FAILED,
    }
)


@dataclass(frozen=True, slots=True)
class AuditPolicy:
    """Explicit thresholds for clean results and certified handoff.

    Exemptions can downgrade known domain findings but can never exempt a
    fail-closed code.  Under the strict default, warnings leave a report
    operationally clean but prevent certification.
    """

    name: str = "strict"
    clean_threshold: AuditSeverity = AuditSeverity.ERROR
    certification_threshold: AuditSeverity = AuditSeverity.WARNING
    exempt_codes: frozenset[AuditCode] = frozenset()
    require_full_model_for_certification: bool = True
    require_verified_for_certification: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, name="policy name"))
        clean = (
            self.clean_threshold
            if isinstance(self.clean_threshold, AuditSeverity)
            else AuditSeverity(self.clean_threshold)
        )
        certification = (
            self.certification_threshold
            if isinstance(self.certification_threshold, AuditSeverity)
            else AuditSeverity(self.certification_threshold)
        )
        exemptions = frozenset(
            code if isinstance(code, AuditCode) else AuditCode(code)
            for code in self.exempt_codes
        )
        forbidden = exemptions & _FAIL_CLOSED_CODES
        if forbidden:
            names = ", ".join(sorted(code.value for code in forbidden))
            raise ValueError(f"fail-closed audit codes cannot be exempted: {names}")
        object.__setattr__(self, "clean_threshold", clean)
        object.__setattr__(self, "certification_threshold", certification)
        object.__setattr__(self, "exempt_codes", exemptions)

    @classmethod
    def strict(cls) -> "AuditPolicy":
        return cls()

    def blocks_clean(self, issue: AuditIssue) -> bool:
        if issue.code in _FAIL_CLOSED_CODES:
            return True
        return (
            issue.code not in self.exempt_codes
            and issue.severity >= self.clean_threshold
        )

    def blocks_certification(self, issue: AuditIssue) -> bool:
        if issue.code in _FAIL_CLOSED_CODES:
            return True
        return (
            issue.code not in self.exempt_codes
            and issue.severity >= self.certification_threshold
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "clean_threshold": self.clean_threshold.name.lower(),
            "certification_threshold": self.certification_threshold.name.lower(),
            "exempt_codes": sorted(code.value for code in self.exempt_codes),
            "require_full_model_for_certification": (
                self.require_full_model_for_certification
            ),
            "require_verified_for_certification": (
                self.require_verified_for_certification
            ),
        }


@runtime_checkable
class BroadPhaseDiagnostics(Protocol):
    """The subset of spatial query counters consumed by audit collectors."""

    @property
    def candidate_count(self) -> int: ...

    @property
    def node_visits(self) -> int: ...

    @property
    def leaf_tests(self) -> int: ...


@dataclass(frozen=True, slots=True)
class AuditMetrics:
    candidate_count: int = 0
    narrow_phase_tests: int = 0
    classified_count: int = 0
    unclassified_count: int = 0
    index_node_visits: int = 0
    index_leaf_tests: int = 0
    index_updates: int = 0

    def __post_init__(self) -> None:
        for name in (
            "candidate_count",
            "narrow_phase_tests",
            "classified_count",
            "unclassified_count",
            "index_node_visits",
            "index_leaf_tests",
            "index_updates",
        ):
            _non_negative_integer(getattr(self, name), name=name)
        if self.classified_count + self.unclassified_count > self.candidate_count:
            raise ValueError(
                "classified and unclassified counts cannot exceed candidates"
            )

    @property
    def classification_complete(self) -> bool:
        return (
            self.unclassified_count == 0
            and self.classified_count == self.candidate_count
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "candidate_count": self.candidate_count,
            "narrow_phase_tests": self.narrow_phase_tests,
            "classified_count": self.classified_count,
            "unclassified_count": self.unclassified_count,
            "index_node_visits": self.index_node_visits,
            "index_leaf_tests": self.index_leaf_tests,
            "index_updates": self.index_updates,
        }


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Canonical audit output with computed clean/certifiable semantics."""

    model_id: str
    revision: int
    scope: AuditScope
    policy: AuditPolicy
    issues: tuple[AuditIssue, ...] = ()
    metrics: AuditMetrics = AuditMetrics()
    completed: bool = True
    verified: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, name="model_id"))
        _non_negative_integer(self.revision, name="revision")
        scope = self.scope if isinstance(self.scope, AuditScope) else AuditScope(self.scope)
        if not isinstance(self.policy, AuditPolicy):
            raise TypeError("policy must be an AuditPolicy")
        if not isinstance(self.metrics, AuditMetrics):
            raise TypeError("metrics must be AuditMetrics")
        if not isinstance(self.completed, bool) or not isinstance(self.verified, bool):
            raise TypeError("completed and verified must be booleans")
        issues = tuple(self.issues)
        if any(not isinstance(issue, AuditIssue) for issue in issues):
            raise TypeError("issues must contain AuditIssue values")
        # Exact duplicate findings usually arise from examining a pair in both
        # directions.  Canonical de-duplication makes report ordering stable.
        unique = set(issues)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "issues", tuple(sorted(unique, key=lambda i: i.sort_key)))

    @property
    def clean(self) -> bool:
        return (
            self.completed
            and self.verified
            and self.metrics.unclassified_count == 0
            and all(not self.policy.blocks_clean(issue) for issue in self.issues)
        )

    @property
    def certifiable(self) -> bool:
        if not self.clean or not self.metrics.classification_complete:
            return False
        if self.policy.require_full_model_for_certification:
            if self.scope is not AuditScope.FULL_MODEL:
                return False
        if self.policy.require_verified_for_certification and not self.verified:
            return False
        return all(
            not self.policy.blocks_certification(issue) for issue in self.issues
        )

    @property
    def issue_counts(self) -> dict[str, int]:
        counts = Counter(issue.code.value for issue in self.issues)
        return dict(sorted(counts.items()))

    @property
    def checksum(self) -> str:
        payload = json.dumps(
            self.to_dict(include_checksum=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "model_id": self.model_id,
            "revision": self.revision,
            "scope": self.scope.value,
            "policy": self.policy.to_dict(),
            "completed": self.completed,
            "verified": self.verified,
            "clean": self.clean,
            "certifiable": self.certifiable,
            "metrics": self.metrics.to_dict(),
            "issue_counts": self.issue_counts,
            "issues": [issue.to_dict() for issue in self.issues],
        }
        if include_checksum:
            result["checksum"] = self.checksum
        return result


@dataclass(frozen=True, slots=True)
class AuditContext:
    """Stable context passed to independent model audit checker hooks."""

    model_id: str
    revision: int
    scope: AuditScope
    changed_entities: tuple[AuditEntity, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, name="model_id"))
        _non_negative_integer(self.revision, name="revision")
        scope = self.scope if isinstance(self.scope, AuditScope) else AuditScope(self.scope)
        changed = tuple(sorted(set(self.changed_entities)))
        if any(entity.model_id != self.model_id for entity in changed):
            raise ValueError("changed entities must belong to the audited model")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "changed_entities", changed)


@runtime_checkable
class AuditCheck(Protocol):
    """A side-effect-free checker callable supplied by model integration."""

    def __call__(
        self, model: object, context: AuditContext, collector: "AuditCollector"
    ) -> None: ...


class AuditCollector:
    """Mutable checker sink that emits one immutable ``AuditReport``."""

    def __init__(
        self,
        model_id: object,
        revision: int,
        *,
        scope: AuditScope = AuditScope.FULL_MODEL,
        policy: AuditPolicy | None = None,
        index_updates: int = 0,
    ) -> None:
        self.model_id = _text(model_id, name="model_id")
        self.revision = _non_negative_integer(revision, name="revision")
        self.scope = scope if isinstance(scope, AuditScope) else AuditScope(scope)
        self.policy = AuditPolicy.strict() if policy is None else policy
        if not isinstance(self.policy, AuditPolicy):
            raise TypeError("policy must be an AuditPolicy")
        self._issues: list[AuditIssue] = []
        self._candidate_count = 0
        self._narrow_phase_tests = 0
        self._classified_count = 0
        self._unclassified_count = 0
        self._index_node_visits = 0
        self._index_leaf_tests = 0
        self._index_updates = _non_negative_integer(
            index_updates, name="index_updates"
        )
        self._completed = True
        self._verified = True
        self._finished = False

    def _ensure_open(self) -> None:
        if self._finished:
            raise RuntimeError("the audit collector has already been finished")

    def add(self, issue: AuditIssue) -> AuditIssue:
        self._ensure_open()
        if not isinstance(issue, AuditIssue):
            raise TypeError("issue must be an AuditIssue")
        for entity in issue.entities:
            if entity.model_id != self.model_id:
                raise ValueError("audit issue entity belongs to a different model")
        self._issues.append(issue)
        if issue.code is AuditCode.UNVERIFIED_CLASSIFICATION:
            self._verified = False
        return issue

    def issue(
        self,
        code: AuditCode,
        severity: AuditSeverity,
        message: str,
        **kwargs: object,
    ) -> AuditIssue:
        allowed = {"entities", "witnesses", "classification", "details"}
        unexpected = set(kwargs) - allowed
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise TypeError(f"unexpected audit issue arguments: {names}")
        issue = AuditIssue.create(
            code,
            severity,
            message,
            entities=kwargs.get("entities", ()),  # type: ignore[arg-type]
            witnesses=kwargs.get("witnesses", ()),  # type: ignore[arg-type]
            classification=kwargs.get("classification"),
            details=kwargs.get("details"),  # type: ignore[arg-type]
        )
        return self.add(issue)

    def record_broad_phase(self, diagnostics: BroadPhaseDiagnostics) -> None:
        """Accumulate counters from ``spatial.QueryDiagnostics`` by protocol."""

        self._ensure_open()
        try:
            candidates = diagnostics.candidate_count
            node_visits = diagnostics.node_visits
            leaf_tests = diagnostics.leaf_tests
        except AttributeError as exc:
            raise TypeError("diagnostics do not satisfy BroadPhaseDiagnostics") from exc
        self._candidate_count += _non_negative_integer(
            candidates, name="candidate_count"
        )
        self._index_node_visits += _non_negative_integer(
            node_visits, name="node_visits"
        )
        self._index_leaf_tests += _non_negative_integer(
            leaf_tests, name="leaf_tests"
        )

    def record_candidates(self, count: int) -> None:
        self._ensure_open()
        self._candidate_count += _non_negative_integer(count, name="count")

    def record_narrow_phase(self, count: int = 1) -> None:
        self._ensure_open()
        self._narrow_phase_tests += _non_negative_integer(count, name="count")

    def record_classification(
        self,
        *,
        classified: bool,
        verified: bool = True,
        count: int = 1,
    ) -> None:
        self._ensure_open()
        if not isinstance(classified, bool) or not isinstance(verified, bool):
            raise TypeError("classified and verified must be booleans")
        value = _non_negative_integer(count, name="count")
        if classified:
            self._classified_count += value
        else:
            self._unclassified_count += value
        if not verified:
            self._verified = False

    def mark_incomplete(self) -> None:
        self._ensure_open()
        self._completed = False

    def mark_unverified(self) -> None:
        self._ensure_open()
        self._verified = False

    def finish(self) -> AuditReport:
        self._ensure_open()
        accounted = self._classified_count + self._unclassified_count
        if accounted < self._candidate_count:
            self._unclassified_count += self._candidate_count - accounted
        elif accounted > self._candidate_count:
            raise ValueError(
                "classification accounting exceeds broad-phase candidates"
            )
        if self._unclassified_count and not any(
            issue.code is AuditCode.UNCLASSIFIED_CANDIDATE
            for issue in self._issues
        ):
            self._issues.append(
                AuditIssue.create(
                    AuditCode.UNCLASSIFIED_CANDIDATE,
                    AuditSeverity.BLOCKER,
                    "one or more broad-phase candidates were not classified",
                    details={"count": self._unclassified_count},
                )
            )
        if not self._verified and not any(
            issue.code is AuditCode.UNVERIFIED_CLASSIFICATION
            for issue in self._issues
        ):
            self._issues.append(
                AuditIssue.create(
                    AuditCode.UNVERIFIED_CLASSIFICATION,
                    AuditSeverity.BLOCKER,
                    "one or more classifications were not geometrically verified",
                )
            )
        metrics = AuditMetrics(
            candidate_count=self._candidate_count,
            narrow_phase_tests=self._narrow_phase_tests,
            classified_count=self._classified_count,
            unclassified_count=self._unclassified_count,
            index_node_visits=self._index_node_visits,
            index_leaf_tests=self._index_leaf_tests,
            index_updates=self._index_updates,
        )
        self._finished = True
        return AuditReport(
            model_id=self.model_id,
            revision=self.revision,
            scope=self.scope,
            policy=self.policy,
            issues=tuple(self._issues),
            metrics=metrics,
            completed=self._completed,
            verified=self._verified,
        )


def _check_name(check: object) -> str:
    explicit = getattr(check, "audit_name", None)
    if explicit:
        return str(explicit)
    name = getattr(check, "__qualname__", None) or getattr(check, "__name__", None)
    if name:
        return str(name)
    return type(check).__qualname__


def run_audit(
    model: ModelT,
    checks: Iterable[
        Callable[[ModelT, AuditContext, AuditCollector], None]
    ],
    *,
    model_id_getter: Callable[[ModelT], object],
    revision_getter: Callable[[ModelT], int],
    scope: AuditScope = AuditScope.FULL_MODEL,
    policy: AuditPolicy | None = None,
    changed_entities: Iterable[AuditEntity] = (),
    index_updates: int = 0,
    catch_exceptions: bool = True,
) -> AuditReport:
    """Run independent checker hooks and return a fail-closed report.

    Identity and revision access are required callables instead of assumed
    model internals.  A checker exception marks the report incomplete and adds
    a stable blocker (unless ``catch_exceptions=False`` requests propagation).
    """

    if not callable(model_id_getter) or not callable(revision_getter):
        raise TypeError("model identity and revision getters must be callable")
    model_id = _text(model_id_getter(model), name="model_id")
    revision = _non_negative_integer(revision_getter(model), name="revision")
    context = AuditContext(
        model_id=model_id,
        revision=revision,
        scope=scope,
        changed_entities=tuple(changed_entities),
    )
    collector = AuditCollector(
        model_id,
        revision,
        scope=context.scope,
        policy=policy,
        index_updates=index_updates,
    )
    for check in checks:
        if not callable(check):
            raise TypeError("checks must contain callables")
        try:
            check(model, context, collector)
        except Exception as exc:
            if not catch_exceptions:
                raise
            collector.mark_incomplete()
            collector.issue(
                AuditCode.CHECK_FAILED,
                AuditSeverity.BLOCKER,
                "an audit checker failed before qualification completed",
                details={
                    "check": _check_name(check),
                    "exception_type": type(exc).__qualname__,
                },
            )
    return collector.finish()
