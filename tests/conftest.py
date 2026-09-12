"""Shared fixtures. The real loan tape is loaded once per session."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.loader import load_loans
from src.models import Loan, Payment

TAPE = Path(__file__).resolve().parents[1] / "loans.xlsx"

# Reference loans called out in the assignment brief.
CLEAN_LOANS = (32271989, 99981632)
IRREGULAR_LOAN = 37216892


@pytest.fixture(scope="session")
def tape_path() -> Path:
    if not TAPE.exists():
        pytest.skip(f"loan tape not available at {TAPE}")
    return TAPE


@pytest.fixture(scope="session")
def loaded(tape_path):
    return load_loans(tape_path)


@pytest.fixture(scope="session")
def loans(loaded):
    return loaded[0]


@pytest.fixture(scope="session")
def load_report(loaded):
    return loaded[1]


@pytest.fixture(scope="session")
def by_id(loans) -> dict[int, Loan]:
    return {loan.loan_id: loan for loan in loans}


def make_payment(
    due: str | None = "2024-01-15",
    paid: str | None = "2024-01-15",
    payment_type: str = "principal",
    state: str = "paid on time",
    amount: float = 100.0,
    pending: float = 0.0,
    loan_id: int = 1,
) -> Payment:
    """Build a payment from ISO date strings, for synthetic test loans."""
    return Payment(
        loan_id=loan_id,
        due_date=date.fromisoformat(due) if due else None,
        paid_date=date.fromisoformat(paid) if paid else None,
        payment_type=payment_type,
        state=state,
        amount=amount,
        pending_amount=pending,
    )


def make_loan(**overrides) -> Loan:
    """A minimal, internally consistent loan that no detector flags."""
    defaults = dict(
        loan_id=1,
        loan_amount=1200.0,
        interest_rate=12.0,
        loan_term=12,
        loan_type="instalment",
        loan_status="granted",
        monthly_payment=106.62,
        disbursal_date=date(2024, 1, 1),
        outstanding_principal=0.0,
        repaid_principal=1200.0,
        outstanding_interest=0.0,
        repaid_interest=78.0,
        days_late=0,
        employment_status="employed full time",
        occupation="Cook",
        months_at_employer=24.0,
        borrower_income=1000.0,
        family_income=2000.0,
    )
    defaults.update(overrides)
    return Loan(**defaults)
