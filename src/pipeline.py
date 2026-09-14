"""Orchestration: load a loan tape, run every detector, collect results.

The pipeline owns the error boundary. A detector that raises is logged and
skipped for that loan; the loan still gets a result and the run still finishes.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .detectors import DETECTORS, WARNING_RULES, Detector, WarningRule
from .loader import LoadReport, load_loans
from .models import Anomaly, Loan, LoanResult, Severity

TapeDetector = Callable[[Sequence[Loan]], dict[int, list[Anomaly]]]

logger = logging.getLogger(__name__)


def detect_duplicate_loan_ids(loans: Sequence[Loan]) -> dict[int, list[Anomaly]]:
    """Tape-level check: flag loans whose ID appears more than once."""
    counts: dict[int, int] = Counter(loan.loan_id for loan in loans)
    duplicates = {lid for lid, count in counts.items() if count > 1}
    if not duplicates:
        return {}
    results: dict[int, list[Anomaly]] = {}
    for lid in duplicates:
        results[lid] = [
            Anomaly(
                code="DUPLICATE_LOAN_ID",
                severity=Severity.MEDIUM,
                reason=f"Loan ID {lid} appears {counts[lid]} times in the tape",
                detector="detect_duplicate_loan_ids",
            )
        ]
    return results


TAPE_DETECTORS: tuple[TapeDetector, ...] = (detect_duplicate_loan_ids,)


@dataclass(slots=True)
class PipelineResult:
    """Everything a run produced: per-loan results plus run-level statistics."""

    results: list[LoanResult] = field(default_factory=list)
    load_report: LoadReport | None = None
    detector_errors: list[tuple[int, str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def flagged(self) -> list[LoanResult]:
        return [r for r in self.results if r.is_flagged]

    @property
    def clean(self) -> list[LoanResult]:
        return [r for r in self.results if not r.is_flagged]

    @property
    def with_warnings(self) -> list[LoanResult]:
        """Loans carrying a data-completeness caveat, flagged or not."""
        return [r for r in self.results if r.data_warnings]

    def severity_counts(self) -> dict[str, int]:
        """How many loans sit at each top severity level."""
        counts = Counter(str(r.max_severity) for r in self.results)
        return {level.name: counts.get(level.name, 0) for level in reversed(Severity)}

    def anomaly_counts(self) -> dict[str, int]:
        """How many times each anomaly code fired, most frequent first."""
        counts = Counter(a.code for r in self.results for a in r.anomalies)
        return dict(counts.most_common())

    def summary_lines(self) -> list[str]:
        """Human-readable run summary for the console and the logs."""
        flagged = len(self.flagged)
        share = (flagged / self.total * 100) if self.total else 0.0
        lines = [
            f"Loans analysed:      {self.total}",
            f"Flagged:             {flagged} ({share:.1f}%)",
            f"Normal:              {len(self.clean)}",
            "",
            "By highest severity:",
        ]
        lines += [
            f"  {name:<9} {count}"
            for name, count in self.severity_counts().items() if count
        ]
        lines += ["", "By anomaly type:"]
        lines += [f"  {code:<28} {count}" for code, count in self.anomaly_counts().items()]
        warned = len(self.with_warnings)
        if warned:
            lines += [
                "",
                f"Data warnings:       {warned} loan(s) with incomplete source data "
                f"(not counted as anomalies)",
            ]
        if self.load_report and self.load_report.failures:
            lines += ["", f"Unreadable rows:     {len(self.load_report.failures)}"]
        if self.detector_errors:
            lines += [f"Detector errors:     {len(self.detector_errors)}"]
        return lines


def run_detectors(
    loan: Loan,
    detectors: Sequence[Detector] = DETECTORS,
    warning_rules: Sequence[WarningRule] = WARNING_RULES,
) -> tuple[LoanResult, list[tuple[int, str, str]]]:
    """Run every rule over one loan, isolating failures to the rule that failed."""
    result = LoanResult(loan_id=loan.loan_id)
    errors: list[tuple[int, str, str]] = []

    def guarded(rule, kind: str):
        try:
            return rule(loan) or []
        except Exception as exc:
            name = getattr(rule, "__name__", repr(rule))
            logger.exception("Loan %s: %s %s failed", loan.loan_id, kind, name)
            errors.append((loan.loan_id, name, str(exc)))
            result.errors.append(f"{name} failed: {exc}")
            return []

    for detector in detectors:
        found: Iterable[Anomaly] = guarded(detector, "detector")
        result.anomalies.extend(found)

    for rule in warning_rules:
        result.data_warnings.extend(guarded(rule, "warning rule"))

    return result, errors


def analyse(
    loans: Iterable[Loan],
    detectors: Sequence[Detector] = DETECTORS,
    warning_rules: Sequence[WarningRule] = WARNING_RULES,
    tape_detectors: Sequence[TapeDetector] = TAPE_DETECTORS,
) -> PipelineResult:
    """Run the detector suite over already-loaded loans."""
    loan_list = list(loans)
    outcome = PipelineResult()
    for loan in loan_list:
        result, errors = run_detectors(loan, detectors, warning_rules)
        outcome.results.append(result)
        outcome.detector_errors.extend(errors)

    # Tape-level detectors (cross-loan checks like duplicate IDs)
    results_by_id = {r.loan_id: r for r in outcome.results}
    for tape_detector in tape_detectors:
        try:
            findings = tape_detector(loan_list)
        except Exception as exc:
            name = getattr(tape_detector, "__name__", repr(tape_detector))
            logger.exception("Tape detector %s failed: %s", name, exc)
            continue
        for loan_id, anomalies in findings.items():
            if loan_id in results_by_id:
                results_by_id[loan_id].anomalies.extend(anomalies)

    return outcome


def run_pipeline(
    path: str | Path,
    sheet: str | int = 0,
    detectors: Sequence[Detector] = DETECTORS,
    warning_rules: Sequence[WarningRule] = WARNING_RULES,
) -> PipelineResult:
    """Load a loan tape and run the full detection suite over it."""
    loans, load_report = load_loans(path, sheet=sheet)
    outcome = analyse(loans, detectors, warning_rules)
    outcome.load_report = load_report

    logger.info(
        "Detection complete: %d/%d loans flagged, %d detector errors",
        len(outcome.flagged), outcome.total, len(outcome.detector_errors),
    )
    return outcome
