"""Core domain types for the loan-tape data quality pipeline.

Everything downstream (loader, detectors, reporter) speaks in these types, so a
detector never touches a pandas row or a raw Excel cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum
from typing import Any


class Severity(IntEnum):
    """Ordered severity levels. Higher value means more serious."""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Payment:
    """A single scheduled or settled payment inside a loan's payment history.

    `due_date` is the scheduled due date ("Payment date" in the source file) and
    `paid_date` is when the money actually moved ("Repayment date"). `paid_date`
    is None for anything not yet settled.
    """

    loan_id: int | None
    due_date: date | None
    paid_date: date | None
    payment_type: str
    state: str
    amount: float
    pending_amount: float

    @property
    def delay_days(self) -> int | None:
        """Days between the scheduled due date and the actual repayment."""
        if self.due_date is None or self.paid_date is None:
            return None
        return (self.paid_date - self.due_date).days

    @property
    def is_settled(self) -> bool:
        return self.paid_date is not None

    @property
    def settlement_date(self) -> date | None:
        """Best available date for cash-flow purposes."""
        return self.paid_date or self.due_date


@dataclass(slots=True)
class Loan:
    """One row of the loan tape, with its payment history parsed out.

    Fields are optional by design: a lender export may be missing columns, and
    the pipeline must still carry the row through detection rather than drop it.
    `extra` keeps every remaining source column so future detectors can reach
    fields this model does not name explicitly.
    """

    loan_id: int
    borrower_id: str | None = None
    credit_score: str | None = None
    loan_amount: float | None = None
    interest_rate: float | None = None
    loan_term: int | None = None
    loan_type: str | None = None
    loan_status: str | None = None
    borrower_type: str | None = None
    purpose: str | None = None
    disbursal_date: date | None = None
    expected_repayment_date: date | None = None
    repayment_date: date | None = None
    days_late: int | None = None
    monthly_payment: float | None = None
    outstanding_principal: float | None = None
    repaid_principal: float | None = None
    outstanding_interest: float | None = None
    repaid_interest: float | None = None
    arrears: float | None = None
    # Borrower profile
    employment_status: str | None = None
    occupation: str | None = None
    months_at_employer: float | None = None
    years_working_total: float | None = None
    borrower_income: float | None = None
    family_income: float | None = None
    borrower_liabilities: float | None = None
    family_liabilities: float | None = None
    dti: float | None = None
    birth_year: int | None = None
    # Business borrower fields
    annual_revenue: float | None = None
    number_of_employees: int | None = None
    company_type: str | None = None
    # Payment history
    payments: list[Payment] = field(default_factory=list)
    payments_truncated: bool = False
    # Everything else from the source row, plus parse diagnostics
    extra: dict[str, Any] = field(default_factory=dict)
    parse_warnings: list[str] = field(default_factory=list)
    source_row: int | None = None

    @property
    def total_interest(self) -> float | None:
        """Repaid plus still-outstanding interest, when either is known."""
        parts = [v for v in (self.repaid_interest, self.outstanding_interest) if v is not None]
        return sum(parts) if parts else None

    def payments_of_type(self, *types: str) -> list[Payment]:
        wanted = {t.lower() for t in types}
        return [p for p in self.payments if p.payment_type.lower() in wanted]


@dataclass(frozen=True, slots=True)
class Anomaly:
    """A single reason a loan was flagged.

    `code` is the stable machine-readable identifier, `reason` the human
    sentence that lands in the report.
    """

    code: str
    severity: Severity
    reason: str
    detector: str = ""


@dataclass(slots=True)
class LoanResult:
    """Detection outcome for one loan.

    `anomalies` are findings about the loan itself and decide whether it is
    flagged. `data_warnings` are notes about the completeness of the source
    record: they explain which checks could not run, and never flag a loan.
    """

    loan_id: int
    anomalies: list[Anomaly] = field(default_factory=list)
    data_warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_flagged(self) -> bool:
        return bool(self.anomalies)

    @property
    def status(self) -> str:
        return "FLAGGED" if self.is_flagged else "NORMAL"

    @property
    def max_severity(self) -> Severity:
        return max((a.severity for a in self.anomalies), default=Severity.NONE)

    def sorted_anomalies(self) -> list[Anomaly]:
        """Most serious first, stable within a severity level."""
        return sorted(self.anomalies, key=lambda a: -int(a.severity))
