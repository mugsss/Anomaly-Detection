"""Reporting: turn pipeline results into a CSV report and a console summary.

The CSV has one row per loan, flagged or not, so the report doubles as a
coverage record: every loan ID that went in comes out with a verdict.

`data_warnings` is deliberately separate from the anomaly columns. A truncated
payment history says our view of the loan is incomplete, not that the loan is
bad, so it annotates the row without flagging it. The JSON report carries the
structured evidence behind each anomaly, which the CSV omits to stay scannable.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from .models import LoanResult
from .pipeline import PipelineResult

logger = logging.getLogger(__name__)

CSV_COLUMNS = (
    "loan_id",
    "status",
    "max_severity",
    "anomaly_count",
    "anomaly_codes",
    "reasons",
    "data_warnings",
)

#: Separator between reasons in the single-cell `reasons` column. Chosen so the
#: cell stays readable in Excel and never collides with text in a reason.
REASON_SEPARATOR = " | "


def result_to_row(result: LoanResult) -> dict[str, Any]:
    """Flatten one loan result into a CSV row."""
    anomalies = result.sorted_anomalies()
    return {
        "loan_id": result.loan_id,
        "status": result.status,
        "max_severity": str(result.max_severity),
        "anomaly_count": len(anomalies),
        "anomaly_codes": ";".join(a.code for a in anomalies),
        "reasons": REASON_SEPARATOR.join(
            f"[{a.severity!s}] {a.reason}" for a in anomalies
        ),
        "data_warnings": REASON_SEPARATOR.join(result.data_warnings),
    }


def write_csv(outcome: PipelineResult, path: str | Path) -> Path:
    """Write the per-loan CSV report. Flagged loans first, worst severity first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(
        outcome.results,
        key=lambda r: (-int(r.max_severity), -len(r.anomalies), r.loan_id),
    )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in ordered:
            writer.writerow(result_to_row(result))

    logger.info("Wrote %d rows to %s", len(ordered), path)
    return path


def write_json(outcome: PipelineResult, path: str | Path) -> Path:
    """Write the full result set as JSON, keeping structured evidence intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": {
            "loans_analysed": outcome.total,
            "flagged": len(outcome.flagged),
            "normal": len(outcome.clean),
            "with_data_warnings": len(outcome.with_warnings),
            "by_severity": outcome.severity_counts(),
            "by_anomaly": outcome.anomaly_counts(),
        },
        "loans": [
            {
                "loan_id": r.loan_id,
                "status": r.status,
                "max_severity": str(r.max_severity),
                "anomalies": [
                    {
                        "code": a.code,
                        "severity": str(a.severity),
                        "reason": a.reason,
                        "detector": a.detector,
                        "evidence": a.evidence,
                    }
                    for a in r.sorted_anomalies()
                ],
                "data_warnings": r.data_warnings,
                "errors": r.errors,
            }
            for r in outcome.results
        ],
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote JSON report to %s", path)
    return path


def format_console_summary(outcome: PipelineResult, max_reason_chars: int = 110) -> str:
    """Build the console summary: run statistics then every flagged loan."""
    lines = ["", "=" * 78, "LOAN TAPE DATA QUALITY REPORT", "=" * 78, ""]
    lines += outcome.summary_lines()

    flagged = sorted(
        outcome.flagged, key=lambda r: (-int(r.max_severity), -len(r.anomalies), r.loan_id)
    )
    if flagged:
        lines += ["", "-" * 78, f"FLAGGED LOANS ({len(flagged)})", "-" * 78]
        for result in flagged:
            lines.append(f"\n{result.loan_id}  [{result.max_severity}]")
            for anomaly in result.sorted_anomalies():
                reason = anomaly.reason
                if len(reason) > max_reason_chars:
                    reason = reason[: max_reason_chars - 3] + "..."
                lines.append(f"    - {str(anomaly.severity):<8} {anomaly.code}: {reason}")
            for warning in result.data_warnings:
                lines.append(f"    ! {'WARNING':<8} {warning}")

    warned = sorted(r.loan_id for r in outcome.with_warnings if not r.is_flagged)
    if warned:
        lines += [
            "", "-" * 78,
            f"NORMAL LOANS WITH DATA WARNINGS ({len(warned)})",
            "-" * 78,
            "  " + ", ".join(str(x) for x in warned),
            f"  {outcome.with_warnings[0].data_warnings[0]}",
        ]

    normal = sorted(r.loan_id for r in outcome.clean)
    lines += [
        "", "-" * 78,
        f"NORMAL LOANS ({len(normal)}) - no anomalies detected",
        "-" * 78,
        "  " + ", ".join(str(x) for x in normal),
        "",
    ]
    return "\n".join(lines)
