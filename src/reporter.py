"""Reporting: turn pipeline results into a JSON report and a console summary.

The JSON report has one row per loan, flagged or not, so the report doubles as a
coverage record: every loan ID that went in comes out with a verdict.

`data_warnings` is deliberately separate from the anomaly columns. A truncated
payment history says our view of the loan is incomplete, not that the loan is
bad, so it annotates the row without flagging it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .models import LoanResult
from .pipeline import PipelineResult

logger = logging.getLogger(__name__)

def result_to_row(result: LoanResult) -> dict[str, Any]:
    """Serialize one loan result into a structured JSON object."""
    anomalies = result.sorted_anomalies()
    return {
        "loan_id": result.loan_id,
        "status": result.status,
        "max_severity": str(result.max_severity),
        "anomaly_count": len(anomalies),
        "anomalies": [
            {
                "code": a.code,
                "severity": str(a.severity),
                "detector": a.detector,
                "reason": a.reason,
            }
            for a in anomalies
        ],
        "data_warnings": list(result.data_warnings),
    }


def write_json(outcome: PipelineResult, path: str | Path) -> Path:
    """Write the per-loan JSON report, flagged loans first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(
        outcome.results,
        key=lambda r: (-int(r.max_severity), -len(r.anomalies), r.loan_id),
    )

    rows = [result_to_row(result) for result in ordered]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")

    logger.info("Wrote %d rows to %s", len(ordered), path)
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
