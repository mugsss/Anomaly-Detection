"""Anomaly detection rules.

Every detector is a pure function `(Loan) -> list[Anomaly]`. It never raises,
never mutates the loan, and returns an empty list when the rule does not apply
or the inputs it needs are missing. Adding a rule means writing one function and
appending it to `DETECTORS`.

Warning rules follow the same shape but return `list[str]` and live in
`WARNING_RULES`. They describe the completeness of the source record rather than
the health of the loan, so they never flag it. Truncated payment histories are
the current example: the export cut them off, which is a robustness concern for
the pipeline and a caveat on the report, not a finding about the borrower.

Thresholds live in the constants at the top of this module. They were calibrated
against the sample tape so that the two reference clean loans (32271989 and
99981632) produce no findings while the seeded anomalies are caught.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date

from .models import Anomaly, Loan, Payment, Severity

logger = logging.getLogger(__name__)

Detector = Callable[[Loan], list[Anomaly]]
WarningRule = Callable[[Loan], list[str]]

# --- Thresholds -------------------------------------------------------------

#: A scheduled payment settled this many days late counts as a default.
DEFAULT_DELAY_DAYS = 90

#: Payment states that mean "overdue and still unpaid".
OVERDUE_STATES = {"pending late"}

#: Total interest below this fraction of the expected interest is treated as a
#: broken interest calculation rather than a cheap loan. The sample tape splits
#: cleanly here: anomalies sit below 0.01, the lowest healthy loan at 0.21.
MIN_INTEREST_RATIO = 0.05

#: Absolute divergence (percentage points) between realized XIRR and the stated
#: nominal rate that counts as a mismatch.
XIRR_TOLERANCE_PP = 5.0

#: An annualized IRR computed over an effective horizon shorter than this is
#: numerically unstable, so the XIRR rule stands down. Expressed in years of
#: cash-flow weighted average life.
MIN_WEIGHTED_AVERAGE_LIFE_YEARS = 0.2

#: Reported against any loan whose payment history the export cut short.
TRUNCATION_WARNING = (
    "payment history truncated at source export limit; "
    "XIRR check skipped"
)

#: Payment types that represent interest income.
INTEREST_TYPES = ("interest", "overdue interest")

#: Loan-level monetary fields that must never be negative.
NON_NEGATIVE_FIELDS = (
    "loan_amount", "interest_rate", "loan_term", "monthly_payment",
    "outstanding_principal", "repaid_principal", "outstanding_interest",
    "repaid_interest", "days_late", "arrears",
)

#: Placeholder/dummy values that should not appear in required fields.
PLACEHOLDER_VALUES = frozenset({
    "no data", "99999", "n/a", "tbd", "null", "0000-00-00",
})

#: Required fields checked for placeholder values.
PLACEHOLDER_CHECK_FIELDS = {
    "loan_id": "Loan ID",
    "borrower_id": "Borrower ID",
    "loan_amount": "Loan amount",
    "interest_rate": "Interest rate",
    "loan_term": "Loan term",
    "disbursal_date": "Disbursal date",
}

#: Allowed categorical value sets.
CATEGORICAL_ALLOWED: dict[str, frozenset[str]] = {
    "credit_score": frozenset({"a", "b", "c", "d"}),
    "borrower_type": frozenset({"individual", "business"}),
    "loan_type": frozenset({"instalment", "deferred annuity"}),
    "loan_status": frozenset({"granted", "repaid", "terminated"}),
    "employment_status": frozenset({
        "employed full time", "self employed", "unemployed",
    }),
}

#: Human labels for categorical fields (for reporting).
CATEGORICAL_LABELS: dict[str, str] = {
    "credit_score": "Credit score",
    "borrower_type": "Borrower type",
    "loan_type": "Loan type",
    "loan_status": "Loan status",
    "employment_status": "Employment status",
}


# --- Shared helpers ---------------------------------------------------------


def expected_total_interest(loan: Loan) -> float | None:
    """Rough interest a loan of this size, rate and term should generate.

    Uses the average-outstanding-balance approximation (half the principal over
    the life of the loan). It is deliberately coarse: the rule it feeds only
    asks whether interest is near zero, not whether it is exact.
    """
    if not loan.loan_amount or loan.interest_rate is None or not loan.loan_term:
        return None
    return loan.loan_amount * (loan.interest_rate / 100.0) * (loan.loan_term / 12.0) / 2.0


def observed_total_interest(loan: Loan) -> tuple[float, str] | None:
    """Interest actually booked, from the summary columns or the payments.

    Returns (amount, source). Some rows leave both interest columns empty; the
    payment history still carries the truth, so fall back to it rather than
    skipping the loan.
    """
    from_columns = loan.total_interest
    if from_columns is not None:
        return from_columns, "Outstanding interest + Repaid interest"
    if loan.payments:
        return sum(p.amount for p in loan.payments_of_type(*INTEREST_TYPES)), "payment history"
    return None


def build_cash_flows(loan: Loan) -> list[tuple[date, float]]:
    """Borrower-perspective cash flows: disbursal out, every payment in.

    Each payment is placed on the date the money actually moved, falling back to
    the scheduled due date for anything not yet settled. All payment types count,
    including contract fees, because they are part of what the borrower pays.
    """
    if loan.disbursal_date is None or not loan.loan_amount:
        return []

    early_types = {"full early repayment", "partial early repayment"}
    paid_early_dates: set[date] = set()
    for p in loan.payments:
        if p.paid_date is not None and p.payment_type in early_types:
            paid_early_dates.add(p.due_date or p.paid_date)

    flows: list[tuple[date, float]] = [(loan.disbursal_date, -float(loan.loan_amount))]
    for payment in loan.payments:
        if not payment.amount:
            continue
        if payment.paid_date is not None:
            flows.append((payment.paid_date, float(payment.amount)))
        elif payment.state == "pending late":
            continue
        elif payment.payment_type in early_types and payment.due_date in paid_early_dates:
            continue
        elif payment.due_date is not None:
            flows.append((payment.due_date, float(payment.amount)))
    return flows


def weighted_average_life(loan: Loan, flows: list[tuple[date, float]]) -> float:
    """Inflow-weighted average time to repayment, in years."""
    if loan.disbursal_date is None:
        return 0.0
    total = weighted = 0.0
    for when, amount in flows:
        if amount <= 0:
            continue
        total += amount
        weighted += amount * max((when - loan.disbursal_date).days, 0)
    return (weighted / total / 365.0) if total else 0.0


def compute_xirr(flows: list[tuple[date, float]]) -> float | None:
    """Internal rate of return for dated cash flows, as a decimal fraction.

    Uses pyxirr when available, otherwise falls back to bisection.
    """
    if len(flows) < 2:
        return None
    if not (any(a < 0 for _, a in flows) and any(a > 0 for _, a in flows)):
        return None
    try:
        from pyxirr import xirr

        return xirr(flows)
    except ImportError:
        return _xirr_bisect(flows)


def _xirr_bisect(flows: list[tuple[date, float]], tolerance: float = 1e-7) -> float | None:
    """Bisection XIRR when pyxirr is not installed."""
    start = min(when for when, _ in flows)

    def npv(rate: float) -> float:
        total = 0.0
        for when, amount in flows:
            years = (when - start).days / 365.0
            total += amount / ((1.0 + rate) ** years)
        return total

    low, high = -0.9999, 100.0
    try:
        npv_low, npv_high = npv(low), npv(high)
    except (OverflowError, ZeroDivisionError):
        return None
    if npv_low * npv_high > 0:
        return None
    for _ in range(200):
        mid = (low + high) / 2.0
        try:
            value = npv(mid)
        except (OverflowError, ZeroDivisionError):
            return None
        if abs(value) < tolerance:
            return mid
        if npv_low * value <= 0:
            high = mid
        else:
            low, npv_low = mid, value
    return (low + high) / 2.0


# --- Rule 1: payment default ------------------------------------------------


def detect_payment_default(loan: Loan) -> list[Anomaly]:
    """90+ day gaps between scheduled and actual payment dates, or unpaid arrears.

    Payments settled *before* their due date are ignored: an early repayment
    settles the whole remaining schedule at once and shows up as a large
    negative delay on every future instalment, which is healthy, not late.
    """
    anomalies: list[Anomaly] = []

    worst: Payment | None = None
    late_count = 0
    for payment in loan.payments:
        delay = payment.delay_days
        if delay is None or delay < DEFAULT_DELAY_DAYS:
            continue
        late_count += 1
        if worst is None or delay > (worst.delay_days or 0):
            worst = payment
    if worst is not None:
        anomalies.append(
            Anomaly(
                code="PAYMENT_DEFAULT",
                severity=Severity.CRITICAL,
                reason=(
                    f"{late_count} payment(s) settled {DEFAULT_DELAY_DAYS}+ days after "
                    f"the scheduled date; worst is a {worst.payment_type} instalment due "
                    f"{worst.due_date} and repaid {worst.paid_date}, "
                    f"{worst.delay_days} days late"
                ),
                detector="detect_payment_default",
            )
        )

    overdue = [p for p in loan.payments if p.state in OVERDUE_STATES]
    if overdue:
        outstanding = sum(p.pending_amount or p.amount for p in overdue)
        earliest = min((p.due_date for p in overdue if p.due_date), default=None)
        anomalies.append(
            Anomaly(
                code="PAYMENT_OVERDUE_UNPAID",
                severity=Severity.CRITICAL,
                reason=(
                    f"{len(overdue)} payment(s) are in state 'pending late' with no "
                    f"repayment date, {outstanding:.2f} still outstanding"
                    + (f", oldest due {earliest}" if earliest else "")
                    + (f"; loan-level days late is {loan.days_late}" if loan.days_late else "")
                ),
                detector="detect_payment_default",
            )
        )

    return anomalies


# --- Warning rule: truncated payment history --------------------------------


def warn_truncated_payments(loan: Loan) -> list[str]:
    """Note a payment history cut off by the source export's cell limit.

    This is a warning, not an anomaly. The loan itself may be perfectly healthy;
    what is damaged is our view of it. It matters because the cash-flow rules
    (XIRR, amortization) stand down without a complete payment history, so the
    report has to say that those checks did not run.
    """
    if not loan.payments_truncated:
        return []
    return [TRUNCATION_WARNING]


# --- Rule 3: interest calculation -------------------------------------------


def detect_interest_anomaly(loan: Loan) -> list[Anomaly]:
    """Interest booked is near zero for a loan that carries a real rate."""
    expected = expected_total_interest(loan)
    observed = observed_total_interest(loan)
    if expected is None or observed is None or expected <= 0:
        return []
    if not loan.interest_rate:
        return []

    amount, source = observed
    ratio = amount / expected
    if ratio >= MIN_INTEREST_RATIO:
        return []

    return [
        Anomaly(
            code="INTEREST_NEAR_ZERO",
            severity=Severity.HIGH,
            reason=(
                f"total interest of {amount:.2f} is effectively zero for a "
                f"{loan.loan_amount:.0f} loan at {loan.interest_rate:.0f}% over "
                f"{loan.loan_term} months, which should generate roughly "
                f"{expected:.2f} ({source})"
            ),
            detector="detect_interest_anomaly",
        )
    ]


# --- Rule 4: XIRR vs stated rate --------------------------------------------


def detect_xirr_mismatch(loan: Loan) -> list[Anomaly]:
    """Realized XIRR diverges from the stated nominal rate.

    Stands down when the cash flows cannot support an annualized IRR:
      * truncated payment history (incomplete cash flows);
      * effective horizon under `MIN_WEIGHTED_AVERAGE_LIFE_YEARS` (a loan repaid
        almost immediately annualizes to an arbitrary rate);
      * no sign change or no convergence.

    Deferred annuity loans are checked one-sided. Their structure realizes an
    annualized IRR well below the nominal rate by design (the sample tape shows
    a consistent ratio near 0.55), so only an XIRR *above* the stated rate is
    indicative of a broken interest calculation.
    """
    if loan.payments_truncated or loan.interest_rate is None:
        return []

    flows = build_cash_flows(loan)
    if len(flows) < 3:
        return []

    life = weighted_average_life(loan, flows)
    if life < MIN_WEIGHTED_AVERAGE_LIFE_YEARS:
        logger.debug(
            "Loan %s: XIRR skipped, weighted average life %.2fy is too short to annualize",
            loan.loan_id, life,
        )
        return []

    rate = compute_xirr(flows)
    if rate is None:
        logger.debug("Loan %s: XIRR did not converge", loan.loan_id)
        return []

    realized_pct = rate * 100.0
    divergence = realized_pct - loan.interest_rate
    is_deferred = (loan.loan_type or "").lower() == "deferred annuity"

    if is_deferred and divergence <= XIRR_TOLERANCE_PP:
        return []
    if abs(divergence) <= XIRR_TOLERANCE_PP:
        return []

    direction = "above" if divergence > 0 else "below"
    return [
        Anomaly(
            code="XIRR_RATE_MISMATCH",
            severity=Severity.HIGH,
            reason=(
                f"realized XIRR of {realized_pct:.1f}% is {abs(divergence):.1f} percentage "
                f"points {direction} the stated interest rate of {loan.interest_rate:.0f}% "
                f"(tolerance {XIRR_TOLERANCE_PP:.0f}pp, computed from "
                f"{len(flows) - 1} payment cash flows)"
            ),
            detector="detect_xirr_mismatch",
        )
    ]


# --- Rule 5: employment contradiction ---------------------------------------


def detect_employment_contradiction(loan: Loan) -> list[Anomaly]:
    """Borrower is recorded as unemployed but also carries a current occupation."""
    status = (loan.employment_status or "").strip().lower()
    occupation = (loan.occupation or "").strip()
    if status != "unemployed" or not occupation:
        return []

    return [
        Anomaly(
            code="EMPLOYMENT_CONTRADICTION",
            severity=Severity.MEDIUM,
            reason=(
                f"employment status is 'unemployed' but an occupation is recorded "
                f"({occupation!r}) with {loan.months_at_employer:.0f} months at the "
                f"current employer"
                if loan.months_at_employer is not None
                else "employment status is 'unemployed' but an occupation is "
                f"recorded ({occupation!r})"
            ),
            detector="detect_employment_contradiction",
        )
    ]


# --- Rule 6: income inconsistency -------------------------------------------


def detect_income_inconsistency(loan: Loan) -> list[Anomaly]:
    """Borrower income exceeds household income, which cannot happen.

    Family income is the household total and includes the borrower, so it is a
    hard upper bound on the borrower's own income.
    """
    borrower, family = loan.borrower_income, loan.family_income
    if borrower is None or family is None:
        return []
    if borrower <= family + 0.01:
        return []

    return [
        Anomaly(
            code="INCOME_INCONSISTENCY",
            severity=Severity.MEDIUM,
            reason=(
                f"borrower income ({borrower:.2f}) exceeds total family income "
                f"({family:.2f}) by {borrower - family:.2f}, which is impossible "
                f"since family income includes the borrower"
            ),
            detector="detect_income_inconsistency",
        )
    ]


# --- Rule 7: loan status ----------------------------------------------------


def detect_loan_status_anomaly(loan: Loan) -> list[Anomaly]:
    """Loan status contradicts the balances and arrears it carries."""
    anomalies: list[Anomaly] = []
    status = (loan.loan_status or "").strip().lower()

    if status == "terminated":
        anomalies.append(
            Anomaly(
                code="LOAN_TERMINATED",
                severity=Severity.CRITICAL,
                reason=(
                    f"loan agreement was terminated with {loan.outstanding_principal:.2f} "
                    f"principal still outstanding and {loan.days_late} days late"
                    if loan.outstanding_principal is not None
                    else "loan agreement was terminated"
                ),
                detector="detect_loan_status_anomaly",
            )
        )

    if status == "repaid" and (loan.outstanding_principal or 0) > 0.01:
        anomalies.append(
            Anomaly(
                code="STATUS_BALANCE_CONTRADICTION",
                severity=Severity.HIGH,
                reason=(
                    f"loan status is 'repaid' but {loan.outstanding_principal:.2f} of "
                    f"principal is still outstanding"
                ),
                detector="detect_loan_status_anomaly",
            )
        )

    if status != "terminated" and (loan.days_late or 0) >= DEFAULT_DELAY_DAYS:
        anomalies.append(
            Anomaly(
                code="STATUS_ARREARS_CONTRADICTION",
                severity=Severity.CRITICAL,
                reason=(
                    f"loan is {loan.days_late} days late but still carries status "
                    f"{loan.loan_status!r}"
                ),
                detector="detect_loan_status_anomaly",
            )
        )

    return anomalies


# --- Rule 8: structural data validity ----------------------------------------


def detect_data_validity(loan: Loan) -> list[Anomaly]:
    """Structurally impossible or missing values in the loan record."""
    anomalies: list[Anomaly] = []

    negatives = {
        name: value
        for name in NON_NEGATIVE_FIELDS
        if (value := getattr(loan, name, None)) is not None and value < 0
    }
    if negatives:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(negatives.items()))
        anomalies.append(
            Anomaly(
                code="NEGATIVE_VALUE",
                severity=Severity.HIGH,
                reason=f"negative value in field(s) that cannot be negative: {detail}",
                detector="detect_data_validity",
            )
        )

    required = {
        "Disbursal date": loan.disbursal_date,
        "Loan amount": loan.loan_amount,
        "Interest rate": loan.interest_rate,
        "Loan term": loan.loan_term,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        anomalies.append(
            Anomaly(
                code="MISSING_REQUIRED_FIELD",
                severity=Severity.MEDIUM,
                reason=f"required field(s) missing or unparseable: {', '.join(missing)}",
                detector="detect_data_validity",
            )
        )

    if (
        loan.disbursal_date
        and loan.expected_repayment_date
        and loan.expected_repayment_date < loan.disbursal_date
    ):
        anomalies.append(
            Anomaly(
                code="DATE_ORDER_INVALID",
                severity=Severity.MEDIUM,
                reason=(
                    f"expected repayment date {loan.expected_repayment_date} precedes "
                    f"the disbursal date {loan.disbursal_date}"
                ),
                detector="detect_data_validity",
            )
        )

    foreign = {p.loan_id for p in loan.payments if p.loan_id not in (None, loan.loan_id)}
    if foreign:
        anomalies.append(
            Anomaly(
                code="FOREIGN_PAYMENT_RECORD",
                severity=Severity.MEDIUM,
                reason=(
                    f"payment history contains records belonging to other loan(s): "
                    f"{', '.join(str(x) for x in sorted(foreign))}"
                ),
                detector="detect_data_validity",
            )
        )

    # --- Placeholder/dummy values in required fields ---
    placeholders_found: list[str] = []
    for attr, label in PLACEHOLDER_CHECK_FIELDS.items():
        raw = getattr(loan, attr, None)
        if raw is not None and str(raw).strip().lower() in PLACEHOLDER_VALUES:
            placeholders_found.append(f"{label}={raw!r}")
    if placeholders_found:
        anomalies.append(
            Anomaly(
                code="PLACEHOLDER_VALUE",
                severity=Severity.MEDIUM,
                reason=(
                    f"placeholder/dummy value in required field(s): "
                    f"{', '.join(placeholders_found)}"
                ),
                detector="detect_data_validity",
            )
        )

    # --- Outstanding principal > loan amount by more than 1% ---
    if (
        loan.outstanding_principal is not None
        and loan.loan_amount is not None
        and loan.loan_amount > 0
        and loan.outstanding_principal > loan.loan_amount * 1.01
    ):
        anomalies.append(
            Anomaly(
                code="OUTSTANDING_EXCEEDS_PRINCIPAL",
                severity=Severity.MEDIUM,
                reason=(
                    f"outstanding principal ({loan.outstanding_principal:.2f}) exceeds "
                    f"loan amount ({loan.loan_amount:.2f}) by more than 1%"
                ),
                detector="detect_data_validity",
            )
        )

    # --- Zero interest rate ---
    if loan.interest_rate is not None and loan.interest_rate == 0:
        anomalies.append(
            Anomaly(
                code="ZERO_INTEREST_RATE",
                severity=Severity.HIGH,
                reason="interest rate is zero",
                detector="detect_data_validity",
            )
        )

    # --- Zero loan amount ---
    if loan.loan_amount is not None and loan.loan_amount == 0:
        anomalies.append(
            Anomaly(
                code="ZERO_LOAN_AMOUNT",
                severity=Severity.HIGH,
                reason="loan amount is zero",
                detector="detect_data_validity",
            )
        )

    # --- Future disbursal with existing payments or repaid status ---
    if loan.disbursal_date is not None and loan.disbursal_date > date.today():
        status = (loan.loan_status or "").strip().lower()
        has_payments = bool(loan.payments)
        if has_payments or status == "repaid":
            anomalies.append(
                Anomaly(
                    code="FUTURE_DISBURSAL",
                    severity=Severity.MEDIUM,
                    reason=(
                        f"disbursal date {loan.disbursal_date} is in the future but "
                        f"{'loan is already repaid' if status == 'repaid' else 'payments exist'}"
                    ),
                    detector="detect_data_validity",
                )
            )

    # --- DTI = 0 with positive income on a granted loan ---
    status_lower = (loan.loan_status or "").strip().lower()
    if (
        status_lower == "granted"
        and loan.borrower_income is not None
        and loan.borrower_income > 0
        and loan.dti is not None
        and loan.dti == 0
    ):
        anomalies.append(
            Anomaly(
                code="ZERO_DTI",
                severity=Severity.MEDIUM,
                reason=(
                    f"DTI is zero on a granted loan with borrower income "
                    f"of {loan.borrower_income:.2f}"
                ),
                detector="detect_data_validity",
            )
        )

    # --- Duplicate payment records ---
    seen: dict[tuple, int] = {}
    for p in loan.payments:
        if p.payment_type.lower() in ("overdue interest",):
            continue
        key = (p.due_date, p.paid_date, p.payment_type, p.amount)
        seen[key] = seen.get(key, 0) + 1
    dup_keys = {k: v for k, v in seen.items() if v > 1}
    if dup_keys:
        detail = "; ".join(
            f"{v}x ({k[0]}, {k[1]}, {k[2]}, {k[3]:.2f})" for k, v in dup_keys.items()
        )
        anomalies.append(
            Anomaly(
                code="DUPLICATE_PAYMENT_RECORD",
                severity=Severity.MEDIUM,
                reason=f"duplicate payment record(s) in history: {detail}",
                detector="detect_data_validity",
            )
        )

    # --- Payment scheduled before disbursal ---
    if loan.disbursal_date is not None:
        before_disbursal = [
            p for p in loan.payments
            if p.due_date is not None and p.due_date < loan.disbursal_date
        ]
        if before_disbursal:
            earliest = min(p.due_date for p in before_disbursal)
            anomalies.append(
                Anomaly(
                    code="PAYMENT_BEFORE_DISBURSAL",
                    severity=Severity.MEDIUM,
                    reason=(
                        f"{len(before_disbursal)} payment(s) scheduled before "
                        f"disbursal date {loan.disbursal_date}, "
                        f"earliest on {earliest}"
                    ),
                    detector="detect_data_validity",
                )
            )

    # --- Age > 90 at origination (data-entry error) ---
    age = _age_at_origination(loan)
    if age is not None and age > 90:
        anomalies.append(
            Anomaly(
                code="AGE_DATA_ENTRY_ERROR",
                severity=Severity.MEDIUM,
                reason=(
                    f"borrower age at origination was {age}, likely a data-entry error"
                ),
                detector="detect_data_validity",
            )
        )

    return anomalies


# --- Rule 10: age at origination -----------------------------------------------


def _age_at_origination(loan: Loan) -> int | None:
    """Calculate borrower age at disbursal from birth year."""
    if loan.birth_year is None or loan.disbursal_date is None:
        return None
    return loan.disbursal_date.year - loan.birth_year


def detect_age_at_origination(loan: Loan) -> list[Anomaly]:
    """Flag underage borrowers (legal issue)."""
    age = _age_at_origination(loan)
    if age is None:
        return []
    if age < 18:
        return [
            Anomaly(
                code="UNDERAGE_BORROWER",
                severity=Severity.HIGH,
                reason=(
                    f"borrower was {age} years old at origination "
                    f"(born {loan.birth_year}, disbursed {loan.disbursal_date})"
                ),
                detector="detect_age_at_origination",
            )
        ]
    return []


# --- Rule 12: monthly payment vs income ----------------------------------------


def detect_monthly_payment_vs_income(loan: Loan) -> list[Anomaly]:
    """Flag when the monthly payment exceeds borrower income."""
    if (
        loan.borrower_income is None
        or loan.borrower_income <= 0
        or loan.monthly_payment is None
    ):
        return []
    if loan.monthly_payment > loan.borrower_income:
        return [
            Anomaly(
                code="PAYMENT_EXCEEDS_INCOME",
                severity=Severity.MEDIUM,
                reason=(
                    f"monthly payment ({loan.monthly_payment:.2f}) exceeds borrower "
                    f"income ({loan.borrower_income:.2f})"
                ),
                detector="detect_monthly_payment_vs_income",
            )
        ]
    return []


# --- Rule 13: field completeness -----------------------------------------------


def detect_field_completeness(loan: Loan) -> list[Anomaly]:
    """Flag missing fields based on borrower type."""
    btype = (loan.borrower_type or "").strip().lower()
    if not btype:
        return []

    missing: list[str] = []
    if btype == "business":
        if loan.annual_revenue is None:
            missing.append("Annual revenue")
        if loan.number_of_employees is None:
            missing.append("Number of employees")
        if not (loan.company_type or "").strip():
            missing.append("Company type")
    elif btype == "individual":
        if loan.birth_year is None:
            missing.append("Birth year")
        if not (loan.employment_status or "").strip():
            missing.append("Employment status")

    if not missing:
        return []
    return [
        Anomaly(
            code="FIELD_COMPLETENESS",
            severity=Severity.LOW,
            reason=(
                f"missing field(s) for {btype} borrower: {', '.join(missing)}"
            ),
            detector="detect_field_completeness",
        )
    ]


# --- Rule 14: categorical value validation -------------------------------------


def detect_categorical_values(loan: Loan) -> list[Anomaly]:
    """Validate categorical fields against allowed value sets."""
    anomalies: list[Anomaly] = []
    for attr, allowed in CATEGORICAL_ALLOWED.items():
        raw = getattr(loan, attr, None)
        if raw is None:
            continue
        normalised = str(raw).strip().lower()
        if not normalised:
            continue
        if normalised not in allowed:
            label = CATEGORICAL_LABELS[attr]
            anomalies.append(
                Anomaly(
                    code="INVALID_CATEGORICAL_VALUE",
                    severity=Severity.MEDIUM,
                    reason=(
                        f"{label} value {raw!r} is not in the allowed set "
                        f"{{{', '.join(sorted(allowed))}}}"
                    ),
                    detector="detect_categorical_values",
                )
            )
    return anomalies


#: The rule set the pipeline runs, in report order. Append to extend.
DETECTORS: tuple[Detector, ...] = (
    detect_payment_default,
    detect_interest_anomaly,
    detect_xirr_mismatch,
    detect_employment_contradiction,
    detect_income_inconsistency,
    detect_loan_status_anomaly,
    detect_data_validity,
    detect_age_at_origination,
    detect_monthly_payment_vs_income,
    detect_field_completeness,
    detect_categorical_values,
)

#: Source-completeness rules. These annotate a loan; they never flag it.
WARNING_RULES: tuple[WarningRule, ...] = (warn_truncated_payments,)
