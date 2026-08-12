"""Focused tests for stable strict-audit results and integration hooks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from uuid import UUID

import pytest

from anygeometry.audit import (
    AuditCode,
    AuditCollector,
    AuditEntity,
    AuditIssue,
    AuditMetrics,
    AuditPolicy,
    AuditReport,
    AuditScope,
    AuditSeverity,
    AuditWitness,
    run_audit,
)
from anygeometry.spatial import QueryDiagnostics


MODEL_ID = "32d0d562-82fb-4e00-98d6-709ff879194e"


@dataclass(frozen=True)
class ExampleHandle:
    model_id: UUID
    kind: str
    id: int


def issue(
    code: AuditCode,
    severity: AuditSeverity,
    entity_id: int,
) -> AuditIssue:
    return AuditIssue.create(
        code,
        severity,
        f"stable message for {code.value}",
        entities=(AuditEntity(MODEL_ID, "edge", entity_id),),
        witnesses=(AuditWitness("intersection", (1.0, 2.0, 3.0)),),
        classification="cross",
        details={"distance": 0.0, "component": [2, 1]},
    )


def test_handle_normalization_and_issue_order_are_stable() -> None:
    handle = ExampleHandle(UUID(MODEL_ID), "face", 8)
    assert AuditEntity.from_handle(handle) == AuditEntity(MODEL_ID, "face", 8)
    first = issue(AuditCode.EDGE_CROSSING, AuditSeverity.ERROR, 20)
    second = issue(AuditCode.SLIVER, AuditSeverity.WARNING, 1)
    report_a = AuditReport(
        MODEL_ID,
        7,
        AuditScope.FULL_MODEL,
        AuditPolicy.strict(),
        issues=(second, first, first),
    )
    report_b = AuditReport(
        MODEL_ID,
        7,
        AuditScope.FULL_MODEL,
        AuditPolicy.strict(),
        issues=(first, second),
    )
    assert report_a.issues == (first, second)
    assert report_a.to_dict() == report_b.to_dict()
    assert report_a.checksum == report_b.checksum
    assert json.dumps(report_a.to_dict(), sort_keys=True, allow_nan=False)


def test_clean_and_certifiable_have_distinct_strict_semantics() -> None:
    policy = AuditPolicy.strict()
    info = issue(AuditCode.INTENTIONAL_COINCIDENCE, AuditSeverity.INFO, 1)
    warning = issue(AuditCode.SLIVER, AuditSeverity.WARNING, 2)
    error = issue(AuditCode.EDGE_CROSSING, AuditSeverity.ERROR, 3)

    informational = AuditReport(
        MODEL_ID, 1, AuditScope.FULL_MODEL, policy, issues=(info,)
    )
    assert informational.clean
    assert informational.certifiable

    warned = AuditReport(
        MODEL_ID, 1, AuditScope.FULL_MODEL, policy, issues=(warning,)
    )
    assert warned.clean
    assert not warned.certifiable

    failed = AuditReport(
        MODEL_ID, 1, AuditScope.FULL_MODEL, policy, issues=(error,)
    )
    assert not failed.clean
    assert not failed.certifiable

    local = AuditReport(MODEL_ID, 1, AuditScope.CHANGED_REGION, policy)
    assert local.clean
    assert not local.certifiable


def test_fail_closed_codes_cannot_be_exempted() -> None:
    with pytest.raises(ValueError, match="cannot be exempted"):
        AuditPolicy(exempt_codes=frozenset({AuditCode.UNCLASSIFIED_CANDIDATE}))

    allowed = AuditPolicy(exempt_codes=frozenset({AuditCode.EDGE_CROSSING}))
    report = AuditReport(
        MODEL_ID,
        2,
        AuditScope.FULL_MODEL,
        allowed,
        issues=(issue(AuditCode.EDGE_CROSSING, AuditSeverity.BLOCKER, 8),),
    )
    assert report.clean
    assert report.certifiable


def test_collector_accounts_spatial_candidates_and_fails_unclassified_closed() -> None:
    collector = AuditCollector(MODEL_ID, 4)
    collector.record_broad_phase(
        QueryDiagnostics(
            region_count=1,
            node_visits=17,
            branch_visits=8,
            leaf_tests=5,
            raw_candidate_hits=3,
            candidate_count=3,
        )
    )
    collector.record_narrow_phase(2)
    collector.record_classification(classified=True, count=2)
    report = collector.finish()
    assert report.metrics == AuditMetrics(
        candidate_count=3,
        narrow_phase_tests=2,
        classified_count=2,
        unclassified_count=1,
        index_node_visits=17,
        index_leaf_tests=5,
    )
    assert report.issues[0].code is AuditCode.UNCLASSIFIED_CANDIDATE
    assert not report.clean
    assert not report.certifiable


def test_verified_full_classification_can_be_certified() -> None:
    collector = AuditCollector(MODEL_ID, 4, index_updates=2)
    collector.record_candidates(2)
    collector.record_narrow_phase(2)
    collector.record_classification(classified=True, verified=True, count=2)
    report = collector.finish()
    assert report.clean
    assert report.certifiable
    assert report.metrics.classification_complete
    assert report.metrics.index_updates == 2
    with pytest.raises(RuntimeError, match="already been finished"):
        collector.finish()


def test_unverified_classification_is_fail_closed() -> None:
    collector = AuditCollector(MODEL_ID, 4)
    collector.record_candidates(1)
    collector.record_classification(classified=True, verified=False)
    report = collector.finish()
    assert report.metrics.classification_complete
    assert report.issues[0].code is AuditCode.UNVERIFIED_CLASSIFICATION
    assert not report.clean
    assert not report.certifiable


@dataclass
class ExampleModel:
    uid: str
    version: int


def test_run_audit_uses_explicit_hooks_and_catches_checker_failure() -> None:
    model = ExampleModel(MODEL_ID, 12)

    def successful_check(
        source: ExampleModel, context: object, collector: AuditCollector
    ) -> None:
        assert source is model
        assert getattr(context, "revision") == 12
        collector.record_candidates(1)
        collector.record_narrow_phase()
        collector.record_classification(classified=True)

    def failed_check(
        source: ExampleModel, context: object, collector: AuditCollector
    ) -> None:
        raise ArithmeticError("backend-specific text is intentionally omitted")

    report = run_audit(
        model,
        (successful_check, failed_check),
        model_id_getter=lambda source: source.uid,
        revision_getter=lambda source: source.version,
    )
    assert report.model_id == MODEL_ID
    assert report.revision == 12
    assert not report.completed
    assert not report.clean
    assert report.issues[0].code is AuditCode.CHECK_FAILED
    details = dict(report.issues[0].details)
    assert "ArithmeticError" in details["exception_type"]
    assert "backend-specific text" not in str(report.to_dict())


def test_run_audit_can_propagate_checker_errors_for_development() -> None:
    model = ExampleModel(MODEL_ID, 1)

    def broken(*args: object) -> None:
        raise LookupError("test")

    with pytest.raises(LookupError, match="test"):
        run_audit(
            model,
            (broken,),
            model_id_getter=lambda source: source.uid,
            revision_getter=lambda source: source.version,
            catch_exceptions=False,
        )
