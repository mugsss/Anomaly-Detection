"""Ingestion: read a loan tape from Excel and turn rows into `Loan` objects.

Design rules that matter here:
  * A bad row never kills the run. Every row is parsed inside a try/except and
    failures are recorded in `LoadReport.failures`.
  * Missing columns are tolerated. Column lookup is normalised (case- and
    punctuation-insensitive) and returns None when a column is absent.
  * Payment strings truncated by Excel's 32767-character cell limit are
    repaired to the last complete record instead of being discarded.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .models import Loan, Payment

logger = logging.getLogger(__name__)

#: Excel's hard limit on characters in a single cell. Payment histories longer
#: than this are silently cut off by the exporting system.
EXCEL_CELL_LIMIT = 32767

#: Date formats seen in the wild, tried in order. ISO first since the loan-level
#: columns use it; dd/mm/yyyy is what the nested payment dicts use.
DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d/%m/%y",
    "%d-%m-%Y",
    "%m/%d/%Y",
)

#: Source column -> Loan attribute. Anything not listed lands in `Loan.extra`.
COLUMN_MAP: dict[str, str] = {
    "loan id": "loan_id",
    "borrower id": "borrower_id",
    "credit score": "credit_score",
    "loan amount": "loan_amount",
    "interest rate": "interest_rate",
    "loan term": "loan_term",
    "loan type": "loan_type",
    "loan status": "loan_status",
    "borrower type": "borrower_type",
    "purpose": "purpose",
    "disbursal date": "disbursal_date",
    "expected repayment date": "expected_repayment_date",
    "repayment date": "repayment_date",
    "days late": "days_late",
    "monthly payment": "monthly_payment",
    "outstanding principal": "outstanding_principal",
    "repaid principal": "repaid_principal",
    "outstanding interest": "outstanding_interest",
    "repaid interest": "repaid_interest",
    "arrears": "arrears",
    "employment status": "employment_status",
    "occupation": "occupation",
    "months at current employer": "months_at_employer",
    "years working total": "years_working_total",
    "borrower income": "borrower_income",
    "family income": "family_income",
    "borrower liabilities": "borrower_liabilities",
    "family liabilities": "family_liabilities",
    "dti": "dti",
    "birth year": "birth_year",
    "annual revenue": "annual_revenue",
    "number of employees": "number_of_employees",
    "company type": "company_type",
}

DATE_FIELDS = {"disbursal_date", "expected_repayment_date", "repayment_date"}
FLOAT_FIELDS = {
    "loan_amount", "interest_rate", "monthly_payment", "outstanding_principal",
    "repaid_principal", "outstanding_interest", "repaid_interest", "arrears",
    "months_at_employer", "years_working_total", "borrower_income",
    "family_income", "borrower_liabilities", "family_liabilities", "dti",
    "annual_revenue",
}
INT_FIELDS = {"loan_term", "days_late", "birth_year", "number_of_employees"}


@dataclass(slots=True)
class LoadReport:
    """What happened during ingestion, for logging and for the final summary."""

    rows_seen: int = 0
    loans_loaded: int = 0
    truncated_payments: int = 0
    payments_parsed: int = 0
    failures: list[tuple[Any, str]] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.loans_loaded}/{self.rows_seen} rows loaded, "
            f"{self.payments_parsed} payments parsed, "
            f"{self.truncated_payments} truncated payment histories, "
            f"{len(self.failures)} unreadable rows"
        )


def _normalise(name: Any) -> str:
    """Fold a column name so 'Loan ID', 'loan_id' and 'LOAN  ID' all match."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).strip().lower()).strip()


def parse_date(value: Any) -> date | None:
    """Parse a date from any of the formats this tape mixes. None if unusable.

    Partial or malformed values return None rather than raising: a missing date
    is a data-quality signal for the detectors, not a reason to drop the row.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null", "-"}:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:  # last resort: let pandas try its heuristics
        parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
        return None if pd.isna(parsed) else parsed.date()
    except Exception:  # pragma: no cover - pandas is defensive already
        return None


def _to_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace("%", "").replace(",", "")
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return None if number is None else int(round(number))


def _to_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def repair_truncated_payload(raw: str) -> tuple[str, bool]:
    """Make a cut-off list-of-dicts string parseable.

    Excel truncates long cells at 32767 characters, which leaves the payment
    history ending mid-record. The last complete record always ends at the last
    '}', so cut there and close the list. Returns (payload, was_repaired).
    """
    text = raw.rstrip()
    if text.endswith("]"):
        return text, False
    last_close = text.rfind("}")
    if last_close == -1:
        return text, False
    return text[: last_close + 1] + "]", True


def parse_payments(
    raw: Any, loan_id: int | None = None
) -> tuple[list[Payment], bool, list[str]]:
    """Parse the stringified list of payment dicts.

    Returns (payments, truncated, warnings). Never raises: an unparseable
    history yields an empty list plus a warning so the loan still reaches the
    detectors.
    """
    warnings: list[str] = []
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return [], False, ["payments column is empty"]

    text = str(raw).strip()
    if not text or text in {"[]", "nan"}:
        return [], False, [] if text == "[]" else ["payments column is empty"]

    at_cell_limit = len(text) >= EXCEL_CELL_LIMIT
    records: list[Any] | None = None
    truncated = False

    try:
        records = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        repaired, was_repaired = repair_truncated_payload(text)
        if was_repaired:
            try:
                records = ast.literal_eval(repaired)
                truncated = True
            except (ValueError, SyntaxError) as exc:
                warnings.append(f"payment history unparseable even after repair: {exc}")
        else:
            warnings.append("payment history unparseable and not repairable")

    if records is None:
        return [], at_cell_limit or truncated, warnings
    if isinstance(records, dict):  # a single payment, not a list
        records = [records]
    if not isinstance(records, list):
        return [], truncated, ["payment history is not a list"]

    payments: list[Payment] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            warnings.append(f"payment #{index} is not a mapping, skipped")
            continue
        try:
            payments.append(
                Payment(
                    loan_id=_to_int(record.get("Loan ID")),
                    due_date=parse_date(record.get("Payment date")),
                    paid_date=parse_date(record.get("Repayment date")),
                    payment_type=(_to_str(record.get("Type")) or "unknown").lower(),
                    state=(_to_str(record.get("State")) or "unknown").lower(),
                    amount=_to_float(record.get("Amount")) or 0.0,
                    pending_amount=_to_float(record.get("Pending amount")) or 0.0,
                )
            )
        except Exception as exc:  # a single malformed record must not kill the row
            warnings.append(f"payment #{index} skipped: {exc}")

    if truncated or at_cell_limit:
        truncated = True
        warnings.append(
            f"payment history truncated at Excel's {EXCEL_CELL_LIMIT}-character "
            f"cell limit; parsed {len(payments)} complete records"
        )

    return payments, truncated, warnings


def row_to_loan(row: dict[str, Any], columns: dict[str, str], row_number: int) -> Loan:
    """Build a `Loan` from one source row. Raises only if the loan ID is unusable."""
    def get(field_name: str) -> Any:
        source = columns.get(field_name)
        return row.get(source) if source else None

    loan_id = _to_int(get("loan_id"))
    if loan_id is None:
        raise ValueError("missing or unreadable Loan ID")

    values: dict[str, Any] = {"loan_id": loan_id, "source_row": row_number}
    for field_name in COLUMN_MAP.values():
        if field_name == "loan_id":
            continue
        raw = get(field_name)
        if field_name in DATE_FIELDS:
            values[field_name] = parse_date(raw)
        elif field_name in FLOAT_FIELDS:
            values[field_name] = _to_float(raw)
        elif field_name in INT_FIELDS:
            values[field_name] = _to_int(raw)
        else:
            values[field_name] = _to_str(raw)

    payments_column = columns.get("payments")
    payments, truncated, warnings = parse_payments(
        row.get(payments_column) if payments_column else None, loan_id
    )

    mapped_sources = set(columns.values())
    extra = {k: v for k, v in row.items() if k not in mapped_sources and pd.notna(v)}

    loan = Loan(**values, payments=payments, payments_truncated=truncated, extra=extra)
    loan.parse_warnings.extend(warnings)
    return loan


def _resolve_columns(raw_headers: list[str]) -> dict[str, str]:
    """Map our field names onto whatever the file actually calls its columns."""
    by_normalised = {_normalise(c): c for c in raw_headers}
    resolved = {
        field_name: by_normalised[source]
        for source, field_name in COLUMN_MAP.items()
        if source in by_normalised
    }
    if "payments" in by_normalised:
        resolved["payments"] = by_normalised["payments"]
    return resolved


def _open_worksheet(path: Path, sheet: str | int = 0):
    """Open an Excel worksheet in read-only streaming mode.

    Returns (workbook, worksheet, headers). The caller must close the workbook
    when finished. Read-only mode never loads the entire file into memory, so
    this scales to arbitrarily large tapes.
    """
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    if isinstance(sheet, int):
        ws = wb.worksheets[sheet]
    else:
        ws = wb[sheet]
    rows = ws.iter_rows()
    header_cells = next(rows)
    headers = [str(c.value) if c.value is not None else "" for c in header_cells]
    return wb, ws, rows, headers


def _iter_rows_streaming(
    path: Path, sheet: str | int = 0
) -> Iterator[tuple[dict[str, Any], int, dict[str, str], list[str]]]:
    """Yield (row_dict, row_number, column_map, raw_headers) per data row.

    Uses openpyxl's read-only mode: only one row is in memory at a time.
    """
    wb, _ws, rows, headers = _open_worksheet(path, sheet)
    columns = _resolve_columns(headers)
    try:
        for position, row_cells in enumerate(rows, start=2):
            record = {
                headers[i]: cell.value
                for i, cell in enumerate(row_cells)
                if i < len(headers)
            }
            yield record, position, columns, headers
    finally:
        wb.close()


def iter_loans(path: str | Path, sheet: str | int = 0) -> Iterator[Loan]:
    """Yield loans one at a time, streaming from disk.

    Only one row is in memory at a time, so this scales to tapes of any size.
    Rows that fail to parse are logged and skipped.
    """
    for record, position, columns, _headers in _iter_rows_streaming(Path(path), sheet):
        try:
            yield row_to_loan(record, columns, position)
        except Exception as exc:
            logger.error("Row %d skipped: %s", position, exc)


def load_loans(
    path: str | Path, sheet: str | int = 0
) -> tuple[list[Loan], LoadReport]:
    """Read a loan tape and return the loans plus an ingestion report."""
    path = Path(path)
    report = LoadReport()
    logger.info("Reading loan tape from %s", path)

    loans: list[Loan] = []
    columns: dict[str, str] = {}

    for record, position, columns, headers in _iter_rows_streaming(path, sheet):
        if report.rows_seen == 0:
            resolved_fields = set(columns)
            expected = set(COLUMN_MAP.values()) | {"payments"}
            report.missing_columns = sorted(expected - resolved_fields)
            if report.missing_columns:
                logger.warning(
                    "Loan tape is missing %d expected columns: %s",
                    len(report.missing_columns), ", ".join(report.missing_columns),
                )

        report.rows_seen += 1
        try:
            loan = row_to_loan(record, columns, position)
        except Exception as exc:
            report.failures.append((record.get(columns.get("loan_id", ""), "?"), str(exc)))
            logger.error("Row %d skipped: %s", position, exc)
            continue

        loans.append(loan)
        report.loans_loaded += 1
        report.payments_parsed += len(loan.payments)
        report.truncated_payments += int(loan.payments_truncated)
        for warning in loan.parse_warnings:
            logger.debug("Loan %s: %s", loan.loan_id, warning)

    logger.info("Ingestion complete: %s", report.summary())
    return loans, report
