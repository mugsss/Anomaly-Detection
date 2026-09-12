"""Detector tests.

Two layers:
  * unit tests on synthetic loans, which pin each rule's boundary behaviour;
  * regression tests against the real tape, which assert the exact set of loans
    each rule fires on. Those sets are the contract: the two reference clean
    loans must never appear in any of them.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.detectors import (
    AMORTIZATION_TOLERANCE,
    DEFAULT_DELAY_DAYS,
    DETECTORS,
    MIN_INTEREST_RATIO,
    TRUNCATION_WARNING,
    WARNING_RULES,
    XIRR_TOLERANCE_PP,
    _xirr_bisect,
    annuity_payment,
    build_cash_flows,
    compute_xirr,
    detect_amortization_mismatch,
    detect_data_validity,
    detect_employment_contradiction,
    detect_income_inconsistency,
    detect_interest_anomaly,
    detect_loan_status_anomaly,
    detect_payment_default,
    detect_xirr_mismatch,
    warn_truncated_payments,
    weighted_average_life,
)
from src.models import Severity
from tests.conftest import CLEAN_LOANS, IRREGULAR_LOAN, make_loan, make_payment

# Loans the EDA identified for each rule. These are the regression contract.
DEFAULT_LOANS = {79811839, 96579687, 65318525, 61752881, 31397492, 37216892}
TRUNCATED_LOANS = {
    56721442, 51117775, 72457171, 35612451, 64933882, 13314659, 69683127,
    33817222, 43986823, 25372336, 94146169, 26165342, 95814762,
}
# 17611322 and 58271697 book near-zero interest; the other three book exactly
# zero and have empty interest columns, so they are only visible through the
# payment history.
INTEREST_LOANS = {17611322, 58271697, 14146974, 35294697, 46313736}
EMPLOYMENT_LOANS = {37216892, 31397492, 78579969, 13314659}
INCOME_LOANS = {53926762}
STATUS_LOANS = {37216892}


def fired_on(detector, loans) -> set[int]:
    return {loan.loan_id for loan in loans if detector(loan)}


# --- Rule 1: payment default ------------------------------------------------


class TestPaymentDefault:
    def test_flags_delay_at_threshold(self):
        loan = make_loan(payments=[make_payment("2024-01-01", "2024-03-31")])  # 90 days
        found = detect_payment_default(loan)
        assert len(found) == 1
        assert found[0].code == "PAYMENT_DEFAULT"
        assert found[0].severity is Severity.CRITICAL
        assert found[0].evidence["max_delay_days"] == DEFAULT_DELAY_DAYS

    def test_ignores_delay_below_threshold(self):
        loan = make_loan(payments=[make_payment("2024-01-01", "2024-03-30")])  # 89 days
        assert detect_payment_default(loan) == []

    def test_ignores_early_repayment(self):
        """Early settlement shows as a large negative delay and is healthy."""
        loan = make_loan(payments=[make_payment("2026-01-01", "2024-01-05")])
        assert detect_payment_default(loan) == []

    def test_ignores_unsettled_payments_without_a_late_state(self):
        loan = make_loan(payments=[make_payment("2020-01-01", None, state="pending")])
        assert detect_payment_default(loan) == []

    def test_flags_pending_late_with_no_repayment_date(self):
        loan = make_loan(
            days_late=271,
            payments=[make_payment("2024-04-26", None, state="pending late", pending=500.0)],
        )
        codes = {a.code for a in detect_payment_default(loan)}
        assert codes == {"PAYMENT_OVERDUE_UNPAID"}

    def test_reason_names_the_worst_payment(self):
        loan = make_loan(
            payments=[
                make_payment("2024-01-01", "2024-05-01"),   # 121 days
                make_payment("2024-02-01", "2025-02-01"),   # 366 days
            ]
        )
        found = detect_payment_default(loan)[0]
        assert found.evidence["max_delay_days"] == 366
        assert found.evidence["late_payments"] == 2
        assert "2025-02-01" in found.reason

    def test_overdue_interest_paid_same_day_is_not_late(self):
        loan = make_loan(
            payments=[make_payment("2024-01-01", "2024-01-01", payment_type="overdue interest")]
        )
        assert detect_payment_default(loan) == []

    def test_fires_on_exactly_the_expected_tape_loans(self, loans):
        assert fired_on(detect_payment_default, loans) == DEFAULT_LOANS


# --- Warning rule: truncated payment history --------------------------------


class TestTruncationWarning:
    def test_warns_on_a_truncated_loan(self):
        loan = make_loan(payments_truncated=True, payments=[make_payment()])
        assert warn_truncated_payments(loan) == [TRUNCATION_WARNING]

    def test_says_which_checks_were_skipped(self):
        assert "XIRR" in TRUNCATION_WARNING and "amortization" in TRUNCATION_WARNING

    def test_ignores_intact_loan(self):
        assert warn_truncated_payments(make_loan()) == []

    def test_is_not_an_anomaly_detector(self):
        """Truncation is a robustness concern, so it must not flag a loan."""
        assert warn_truncated_payments not in DETECTORS
        loan = make_loan(payments_truncated=True, payments=[make_payment()])
        assert [a for detector in DETECTORS for a in detector(loan)] == []

    def test_fires_on_exactly_the_expected_tape_loans(self, loans):
        assert fired_on(warn_truncated_payments, loans) == TRUNCATED_LOANS


# --- Rule 3: interest calculation -------------------------------------------


class TestInterestAnomaly:
    def test_flags_near_zero_interest(self):
        loan = make_loan(
            loan_amount=129.0, interest_rate=33.0, loan_term=12,
            repaid_interest=0.01, outstanding_interest=0.0,
        )
        found = detect_interest_anomaly(loan)
        assert len(found) == 1
        assert found[0].severity is Severity.HIGH
        assert found[0].evidence["ratio"] < MIN_INTEREST_RATIO

    def test_accepts_healthy_interest(self):
        assert detect_interest_anomaly(make_loan()) == []

    def test_falls_back_to_payment_history_when_columns_are_empty(self):
        loan = make_loan(
            outstanding_interest=None, repaid_interest=None,
            payments=[make_payment(payment_type="principal", amount=1200.0)],
        )
        found = detect_interest_anomaly(loan)
        assert len(found) == 1
        assert found[0].evidence["source"] == "payment history"

    def test_payment_history_interest_counts_towards_the_total(self):
        loan = make_loan(
            outstanding_interest=None, repaid_interest=None,
            payments=[make_payment(payment_type="interest", amount=78.0)],
        )
        assert detect_interest_anomaly(loan) == []

    def test_skips_when_rate_or_amount_is_missing(self):
        assert detect_interest_anomaly(make_loan(interest_rate=None)) == []
        assert detect_interest_anomaly(make_loan(loan_amount=None)) == []

    def test_zero_rate_loan_is_not_flagged(self):
        loan = make_loan(interest_rate=0.0, repaid_interest=0.0, outstanding_interest=0.0)
        assert detect_interest_anomaly(loan) == []

    def test_fires_on_exactly_the_expected_tape_loans(self, loans):
        assert fired_on(detect_interest_anomaly, loans) == INTEREST_LOANS


# --- Rule 4: XIRR -----------------------------------------------------------


class TestXirr:
    def test_bisection_matches_a_known_rate(self):
        flows = [(date(2024, 1, 1), -1000.0), (date(2025, 1, 1), 1100.0)]
        assert _xirr_bisect(flows) == pytest.approx(0.10, abs=1e-3)

    def test_compute_xirr_agrees_with_the_fallback(self):
        flows = [(date(2024, 1, 1), -1000.0), (date(2025, 1, 1), 1200.0)]
        assert compute_xirr(flows) == pytest.approx(_xirr_bisect(flows), abs=1e-4)

    def test_returns_none_without_a_sign_change(self):
        assert compute_xirr([(date(2024, 1, 1), 100.0), (date(2025, 1, 1), 50.0)]) is None

    def test_cash_flows_start_with_the_disbursal(self):
        loan = make_loan(payments=[make_payment("2024-06-01", "2024-06-01", amount=50.0)])
        flows = build_cash_flows(loan)
        assert flows[0] == (date(2024, 1, 1), -1200.0)
        assert flows[1] == (date(2024, 6, 1), 50.0)

    def test_unsettled_payments_use_the_scheduled_date(self):
        loan = make_loan(payments=[make_payment("2024-06-01", None, state="pending")])
        assert build_cash_flows(loan)[1][0] == date(2024, 6, 1)

    def test_skips_truncated_loans(self):
        loan = make_loan(
            payments_truncated=True,
            payments=[make_payment("2024-06-01", "2024-06-01", amount=5000.0)],
        )
        assert detect_xirr_mismatch(loan) == []

    def test_skips_when_the_horizon_is_too_short_to_annualise(self):
        """A loan repaid days after disbursal annualises to an arbitrary rate."""
        loan = make_loan(
            payments=[
                make_payment("2024-01-10", "2024-01-10", amount=1250.0),
                make_payment("2024-01-11", "2024-01-11", amount=10.0),
                make_payment("2024-01-12", "2024-01-12", amount=10.0),
            ]
        )
        assert weighted_average_life(loan, build_cash_flows(loan)) < 0.2
        assert detect_xirr_mismatch(loan) == []

    def test_flags_a_rate_far_above_the_stated_one(self):
        loan = make_loan(
            interest_rate=10.0,
            payments=[
                make_payment("2024-07-01", "2024-07-01", amount=900.0),
                make_payment("2025-01-01", "2025-01-01", amount=900.0),
                make_payment("2025-07-01", "2025-07-01", amount=200.0),
            ],
        )
        found = detect_xirr_mismatch(loan)
        assert len(found) == 1
        assert found[0].severity is Severity.HIGH
        assert abs(found[0].evidence["divergence_pp"]) > XIRR_TOLERANCE_PP

    def test_deferred_annuity_is_only_flagged_when_the_rate_runs_high(self):
        """Deferred annuity realises below its nominal rate by design."""
        low = make_loan(
            loan_type="deferred annuity", interest_rate=50.0,
            payments=[
                make_payment("2024-07-01", "2024-07-01", amount=400.0),
                make_payment("2025-01-01", "2025-01-01", amount=500.0),
                make_payment("2025-07-01", "2025-07-01", amount=400.0),
            ],
        )
        assert detect_xirr_mismatch(low) == []

        high = make_loan(
            loan_type="deferred annuity", interest_rate=5.0,
            payments=[
                make_payment("2024-07-01", "2024-07-01", amount=800.0),
                make_payment("2025-01-01", "2025-01-01", amount=800.0),
                make_payment("2025-07-01", "2025-07-01", amount=300.0),
            ],
        )
        assert detect_xirr_mismatch(high)

    def test_clean_reference_loan_rate_is_within_tolerance(self, by_id):
        loan = by_id[32271989]
        realized = compute_xirr(build_cash_flows(loan)) * 100
        assert abs(realized - loan.interest_rate) <= XIRR_TOLERANCE_PP

    def test_flags_the_irregular_reference_loan(self, by_id):
        assert detect_xirr_mismatch(by_id[IRREGULAR_LOAN])

    def test_never_fires_on_a_clean_reference_loan(self, by_id):
        for loan_id in CLEAN_LOANS:
            assert detect_xirr_mismatch(by_id[loan_id]) == []

    def test_does_not_flag_most_of_the_book(self, loans):
        """A rate rule that fires on half the tape is not a rate rule."""
        fired = fired_on(detect_xirr_mismatch, loans)
        assert len(fired) / len(loans) < 0.30


# --- Rule 5: employment -----------------------------------------------------


class TestEmploymentContradiction:
    def test_flags_unemployed_with_an_occupation(self):
        loan = make_loan(employment_status="unemployed", occupation="Saleswoman",
                         months_at_employer=0.0)
        found = detect_employment_contradiction(loan)
        assert len(found) == 1
        assert found[0].severity is Severity.MEDIUM
        assert "Saleswoman" in found[0].reason

    def test_unemployed_without_an_occupation_is_consistent(self):
        assert detect_employment_contradiction(
            make_loan(employment_status="unemployed", occupation=None)
        ) == []

    def test_blank_occupation_is_not_a_contradiction(self):
        assert detect_employment_contradiction(
            make_loan(employment_status="unemployed", occupation="   ")
        ) == []

    def test_employed_borrower_is_not_flagged(self):
        assert detect_employment_contradiction(make_loan()) == []

    def test_status_matching_is_case_insensitive(self):
        assert detect_employment_contradiction(
            make_loan(employment_status="Unemployed", occupation="Cook")
        )

    def test_missing_months_at_employer_does_not_break_the_reason(self):
        loan = make_loan(employment_status="unemployed", occupation="Cook",
                         months_at_employer=None)
        assert detect_employment_contradiction(loan)[0].reason

    def test_fires_on_exactly_the_expected_tape_loans(self, loans):
        assert fired_on(detect_employment_contradiction, loans) == EMPLOYMENT_LOANS


# --- Rule 6: income ---------------------------------------------------------


class TestIncomeInconsistency:
    def test_flags_borrower_income_above_family_income(self):
        loan = make_loan(borrower_income=1041.98, family_income=870.98)
        found = detect_income_inconsistency(loan)
        assert len(found) == 1
        assert found[0].evidence["difference"] == pytest.approx(171.0)

    def test_equal_incomes_are_valid(self):
        assert detect_income_inconsistency(
            make_loan(borrower_income=900.0, family_income=900.0)
        ) == []

    def test_sole_earner_within_rounding_is_valid(self):
        assert detect_income_inconsistency(
            make_loan(borrower_income=900.005, family_income=900.0)
        ) == []

    def test_skips_when_either_income_is_missing(self):
        assert detect_income_inconsistency(make_loan(borrower_income=None)) == []
        assert detect_income_inconsistency(make_loan(family_income=None)) == []

    def test_fires_on_exactly_the_expected_tape_loans(self, loans):
        assert fired_on(detect_income_inconsistency, loans) == INCOME_LOANS


# --- Rule 7: loan status ----------------------------------------------------


class TestLoanStatusAnomaly:
    def test_terminated_loan_is_critical(self):
        loan = make_loan(loan_status="terminated", outstanding_principal=3666.87, days_late=271)
        found = detect_loan_status_anomaly(loan)
        codes = {a.code for a in found}
        assert "LOAN_TERMINATED" in codes
        assert max(a.severity for a in found) is Severity.CRITICAL

    def test_repaid_loan_with_a_balance_is_contradictory(self):
        loan = make_loan(loan_status="repaid", outstanding_principal=500.0)
        codes = {a.code for a in detect_loan_status_anomaly(loan)}
        assert codes == {"STATUS_BALANCE_CONTRADICTION"}

    def test_repaid_and_settled_is_fine(self):
        assert detect_loan_status_anomaly(
            make_loan(loan_status="repaid", outstanding_principal=0.0)
        ) == []

    def test_granted_loan_deep_in_arrears_is_contradictory(self):
        loan = make_loan(loan_status="granted", days_late=120)
        codes = {a.code for a in detect_loan_status_anomaly(loan)}
        assert codes == {"STATUS_ARREARS_CONTRADICTION"}

    def test_terminated_loan_is_not_double_reported_for_arrears(self):
        loan = make_loan(loan_status="terminated", days_late=271)
        codes = {a.code for a in detect_loan_status_anomaly(loan)}
        assert "STATUS_ARREARS_CONTRADICTION" not in codes

    def test_healthy_loan_is_not_flagged(self):
        assert detect_loan_status_anomaly(make_loan()) == []

    def test_fires_on_exactly_the_expected_tape_loans(self, loans):
        assert fired_on(detect_loan_status_anomaly, loans) == STATUS_LOANS


# --- Rule 8: amortization ---------------------------------------------------


class TestAmortizationMismatch:
    def test_annuity_formula_matches_a_worked_example(self):
        # 10000 at 12% nominal over 12 months is a well known 888.49
        assert annuity_payment(10000, 12, 12) == pytest.approx(888.49, abs=0.01)

    def test_zero_rate_amortises_linearly(self):
        assert annuity_payment(1200, 0, 12) == pytest.approx(100.0)

    def test_flags_a_payment_far_from_the_formula(self):
        loan = make_loan(monthly_payment=400.0, payments=[
            make_payment(f"2024-{m:02d}-01", None, payment_type="principal") for m in range(1, 13)
        ])
        found = detect_amortization_mismatch(loan)
        assert len(found) == 1
        assert found[0].severity is Severity.HIGH
        assert abs(found[0].evidence["divergence_pct"]) > AMORTIZATION_TOLERANCE * 100

    def test_accepts_a_payment_that_matches_the_formula(self):
        loan = make_loan(payments=[
            make_payment(f"2024-{m:02d}-01", None, payment_type="principal") for m in range(1, 13)
        ])
        assert detect_amortization_mismatch(loan) == []

    def test_contract_fee_is_added_to_the_expected_payment(self):
        """Monthly payment includes the fee, so the fee must be in the expectation."""
        schedule = [
            make_payment(f"2024-{m:02d}-01", None, payment_type="principal")
            for m in range(1, 13)
        ] + [
            make_payment(f"2024-{m:02d}-01", None, payment_type="contract fee repayment",
                         amount=20.0)
            for m in range(1, 13)
        ]
        loan = make_loan(monthly_payment=126.62, payments=schedule)  # 106.62 annuity + 20 fee
        assert detect_amortization_mismatch(loan) == []

    def test_skips_deferred_annuity_loans(self):
        loan = make_loan(loan_type="deferred annuity", monthly_payment=400.0)
        assert detect_amortization_mismatch(loan) == []

    def test_skips_truncated_loans(self):
        assert detect_amortization_mismatch(
            make_loan(monthly_payment=400.0, payments_truncated=True)
        ) == []

    def test_stands_down_when_the_schedule_outruns_the_stated_term(self):
        """A restructured loan's stated term is not its amortization period."""
        loan = make_loan(loan_term=7, monthly_payment=343.76, payments=[
            make_payment(f"{2024 + m // 12}-{m % 12 + 1:02d}-01", None,
                         payment_type="principal")
            for m in range(36)
        ])
        assert detect_amortization_mismatch(loan) == []

    def test_skips_when_inputs_are_missing(self):
        assert detect_amortization_mismatch(make_loan(monthly_payment=None)) == []
        assert detect_amortization_mismatch(make_loan(loan_term=None)) == []

    def test_does_not_fire_on_the_tape(self, loans):
        """Every instalment loan whose term is corroborated matches the formula."""
        assert fired_on(detect_amortization_mismatch, loans) == set()


# --- Rule 9: structural validity --------------------------------------------


class TestDataValidity:
    def test_flags_negative_outstanding_principal(self):
        found = detect_data_validity(make_loan(outstanding_principal=-50.0))
        assert [a.code for a in found] == ["NEGATIVE_VALUE"]
        assert "outstanding_principal=-50.0" in found[0].reason

    def test_flags_missing_required_fields(self):
        found = detect_data_validity(make_loan(loan_amount=None, disbursal_date=None))
        codes = {a.code for a in found}
        assert "MISSING_REQUIRED_FIELD" in codes

    def test_flags_repayment_date_before_disbursal(self):
        loan = make_loan(
            disbursal_date=date(2024, 6, 1), expected_repayment_date=date(2023, 1, 1)
        )
        assert "DATE_ORDER_INVALID" in {a.code for a in detect_data_validity(loan)}

    def test_flags_payment_records_from_another_loan(self):
        loan = make_loan(payments=[make_payment(loan_id=999)])
        found = detect_data_validity(loan)
        assert [a.code for a in found] == ["FOREIGN_PAYMENT_RECORD"]

    def test_healthy_loan_is_not_flagged(self):
        assert detect_data_validity(make_loan()) == []

    def test_tape_is_structurally_valid(self, loans):
        assert fired_on(detect_data_validity, loans) == set()


# --- Cross-cutting guarantees ------------------------------------------------


class TestTruncatedLoansAreNotFlagged:
    """A truncated export must not make an otherwise healthy loan look bad."""

    def test_truncation_alone_produces_no_anomaly(self, by_id):
        # 13314659 is excluded: it carries a real employment contradiction too.
        for loan_id in TRUNCATED_LOANS - {13314659}:
            found = [a for detector in DETECTORS for a in detector(by_id[loan_id])]
            assert found == [], f"{loan_id} was flagged: {[a.code for a in found]}"

    def test_truncated_loan_with_a_real_issue_is_still_flagged(self, by_id):
        loan = by_id[13314659]
        found = [a for detector in DETECTORS for a in detector(loan)]
        assert [a.code for a in found] == ["EMPLOYMENT_CONTRADICTION"]
        assert warn_truncated_payments(loan) == [TRUNCATION_WARNING]


class TestReferenceLoans:
    @pytest.mark.parametrize("loan_id", CLEAN_LOANS)
    def test_clean_reference_loans_raise_no_anomaly_at_all(self, by_id, loan_id):
        found = [a for detector in DETECTORS for a in detector(by_id[loan_id])]
        assert found == [], f"{loan_id} was flagged: {[a.code for a in found]}"

    def test_irregular_reference_loan_is_critical(self, by_id):
        found = [a for detector in DETECTORS for a in detector(by_id[IRREGULAR_LOAN])]
        assert found
        assert max(a.severity for a in found) is Severity.CRITICAL


class TestDetectorContract:
    """Properties every detector must satisfy, enforced across the whole suite."""

    @pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.__name__)
    def test_survives_an_almost_empty_loan(self, detector):
        from src.models import Loan

        assert isinstance(detector(Loan(loan_id=1)), list)

    @pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.__name__)
    def test_returns_anomalies_with_a_reason_and_a_detector_name(self, detector, loans):
        for loan in loans:
            for anomaly in detector(loan):
                assert anomaly.code and anomaly.reason
                assert anomaly.detector == detector.__name__
                assert anomaly.severity > Severity.NONE

    @pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.__name__)
    def test_does_not_mutate_the_loan(self, detector, by_id):
        loan = by_id[IRREGULAR_LOAN]
        before = (len(loan.payments), loan.loan_status, loan.payments_truncated)
        detector(loan)
        assert (len(loan.payments), loan.loan_status, loan.payments_truncated) == before

    @pytest.mark.parametrize("rule", WARNING_RULES, ids=lambda r: r.__name__)
    def test_warning_rules_return_plain_strings(self, rule, loans):
        for loan in loans:
            produced = rule(loan)
            assert isinstance(produced, list)
            assert all(isinstance(w, str) and w.strip() for w in produced)

    def test_anomaly_codes_are_unique_to_one_detector(self, loans):
        owners: dict[str, str] = {}
        for detector in DETECTORS:
            for loan in loans:
                for anomaly in detector(loan):
                    owners.setdefault(anomaly.code, detector.__name__)
                    assert owners[anomaly.code] == detector.__name__
