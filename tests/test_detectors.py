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
    detect_age_at_origination,
    detect_amortization_mismatch,
    detect_categorical_values,
    detect_data_validity,
    detect_employment_contradiction,
    detect_field_completeness,
    detect_income_inconsistency,
    detect_interest_anomaly,
    detect_loan_status_anomaly,
    detect_monthly_payment_vs_income,
    detect_payment_default,
    detect_schedule_integrity,
    detect_xirr_mismatch,
    warn_truncated_payments,
    weighted_average_life,
)
from src.models import Severity
from src.pipeline import detect_duplicate_loan_ids
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


# --- Schedule integrity -------------------------------------------------------


class TestScheduleIntegrity:
    def test_gap_over_45_days_is_flagged(self):
        payments = [
            make_payment("2024-01-01", "2024-01-01"),
            make_payment("2024-03-15", "2024-03-15"),  # 74 days gap
        ]
        found = detect_schedule_integrity(make_loan(payments=payments))
        codes = [a.code for a in found]
        assert "SCHEDULE_GAP" in codes
        assert all(a.severity is Severity.MEDIUM for a in found if a.code == "SCHEDULE_GAP")

    def test_gap_within_45_days_is_fine(self):
        payments = [
            make_payment("2024-01-01", "2024-01-01"),
            make_payment("2024-02-01", "2024-02-01"),  # 31 days
        ]
        found = detect_schedule_integrity(make_loan(payments=payments))
        assert not any(a.code == "SCHEDULE_GAP" for a in found)

    def test_spike_over_5x_median_is_flagged(self):
        payments = [
            make_payment("2024-01-01", "2024-01-01", amount=100.0),
            make_payment("2024-02-01", "2024-02-01", amount=100.0),
            make_payment("2024-03-01", "2024-03-01", amount=100.0),
            make_payment("2024-04-01", "2024-04-01", amount=600.0),  # 6x median
        ]
        found = detect_schedule_integrity(make_loan(payments=payments))
        codes = [a.code for a in found]
        assert "PAYMENT_AMOUNT_SPIKE" in codes
        assert all(
            a.severity is Severity.MEDIUM for a in found if a.code == "PAYMENT_AMOUNT_SPIKE"
        )

    def test_spike_at_or_below_5x_is_fine(self):
        payments = [
            make_payment("2024-01-01", "2024-01-01", amount=100.0),
            make_payment("2024-02-01", "2024-02-01", amount=100.0),
            make_payment("2024-03-01", "2024-03-01", amount=500.0),  # exactly 5x
        ]
        found = detect_schedule_integrity(make_loan(payments=payments))
        assert not any(a.code == "PAYMENT_AMOUNT_SPIKE" for a in found)

    def test_truncated_loan_is_skipped(self):
        payments = [
            make_payment("2024-01-01", "2024-01-01"),
            make_payment("2024-06-01", "2024-06-01"),  # huge gap, but truncated
        ]
        found = detect_schedule_integrity(make_loan(payments=payments, payments_truncated=True))
        assert found == []

    def test_overdue_interest_excluded_from_gap_check(self):
        payments = [
            make_payment("2024-01-01", "2024-01-01"),
            make_payment("2024-06-01", "2024-06-01", payment_type="overdue interest"),
            make_payment("2024-02-01", "2024-02-01"),
        ]
        found = detect_schedule_integrity(make_loan(payments=payments))
        assert not any(a.code == "SCHEDULE_GAP" for a in found)

    def test_early_repayments_excluded_from_spike_check(self):
        payments = [
            make_payment("2024-01-01", "2024-01-01", amount=100.0),
            make_payment("2024-02-01", "2024-02-01", amount=100.0),
            make_payment("2024-03-01", "2024-03-01", amount=100.0),
            make_payment(
                "2024-04-01", "2024-04-01", amount=5000.0,
                payment_type="full early repayment",
            ),
        ]
        found = detect_schedule_integrity(make_loan(payments=payments))
        assert not any(a.code == "PAYMENT_AMOUNT_SPIKE" for a in found)


# --- Age at origination ------------------------------------------------------


class TestAgeAtOrigination:
    def test_underage_borrower_is_high(self):
        loan = make_loan(birth_year=2010, disbursal_date=date(2024, 6, 1))
        found = detect_age_at_origination(loan)
        assert len(found) == 1
        assert found[0].code == "UNDERAGE_BORROWER"
        assert found[0].severity is Severity.HIGH

    def test_normal_adult_is_fine(self):
        loan = make_loan(birth_year=1990, disbursal_date=date(2024, 1, 1))
        assert detect_age_at_origination(loan) == []

    def test_age_over_90_not_flagged_here(self):
        loan = make_loan(birth_year=1920, disbursal_date=date(2024, 1, 1))
        assert detect_age_at_origination(loan) == []

    def test_null_birth_year_is_skipped(self):
        loan = make_loan(birth_year=None)
        assert detect_age_at_origination(loan) == []

    def test_age_exactly_18_is_fine(self):
        loan = make_loan(birth_year=2006, disbursal_date=date(2024, 1, 1))
        assert detect_age_at_origination(loan) == []


# --- Monthly payment vs income -----------------------------------------------


class TestMonthlyPaymentVsIncome:
    def test_payment_exceeds_income_is_flagged(self):
        loan = make_loan(monthly_payment=1500.0, borrower_income=1000.0)
        found = detect_monthly_payment_vs_income(loan)
        assert len(found) == 1
        assert found[0].code == "PAYMENT_EXCEEDS_INCOME"
        assert found[0].severity is Severity.MEDIUM

    def test_payment_within_income_is_fine(self):
        loan = make_loan(monthly_payment=500.0, borrower_income=1000.0)
        assert detect_monthly_payment_vs_income(loan) == []

    def test_zero_income_is_skipped(self):
        loan = make_loan(monthly_payment=500.0, borrower_income=0.0)
        assert detect_monthly_payment_vs_income(loan) == []

    def test_null_income_is_skipped(self):
        loan = make_loan(monthly_payment=500.0, borrower_income=None)
        assert detect_monthly_payment_vs_income(loan) == []


# --- Expanded data validity (Rule 8) -----------------------------------------


class TestExpandedDataValidity:
    def test_placeholder_value_is_flagged(self):
        loan = make_loan(borrower_id="N/A")
        found = detect_data_validity(loan)
        assert any(a.code == "PLACEHOLDER_VALUE" for a in found)

    def test_outstanding_exceeds_principal_by_over_1_percent(self):
        loan = make_loan(loan_amount=1000.0, outstanding_principal=1020.0)
        found = detect_data_validity(loan)
        assert any(a.code == "OUTSTANDING_EXCEEDS_PRINCIPAL" for a in found)

    def test_outstanding_within_1_percent_is_fine(self):
        loan = make_loan(loan_amount=1000.0, outstanding_principal=1005.0)
        found = detect_data_validity(loan)
        assert not any(a.code == "OUTSTANDING_EXCEEDS_PRINCIPAL" for a in found)

    def test_zero_interest_rate_is_high(self):
        loan = make_loan(interest_rate=0)
        found = detect_data_validity(loan)
        zero_rate = [a for a in found if a.code == "ZERO_INTEREST_RATE"]
        assert len(zero_rate) == 1
        assert zero_rate[0].severity is Severity.HIGH

    def test_zero_loan_amount_is_high(self):
        loan = make_loan(loan_amount=0)
        found = detect_data_validity(loan)
        zero_amount = [a for a in found if a.code == "ZERO_LOAN_AMOUNT"]
        assert len(zero_amount) == 1
        assert zero_amount[0].severity is Severity.HIGH

    def test_future_disbursal_with_payments_is_flagged(self):
        loan = make_loan(
            disbursal_date=date(2030, 1, 1),
            payments=[make_payment("2030-01-15", "2030-01-15")],
        )
        found = detect_data_validity(loan)
        assert any(a.code == "FUTURE_DISBURSAL" for a in found)

    def test_future_disbursal_repaid_is_flagged(self):
        loan = make_loan(disbursal_date=date(2030, 1, 1), loan_status="repaid")
        found = detect_data_validity(loan)
        assert any(a.code == "FUTURE_DISBURSAL" for a in found)

    def test_future_disbursal_no_payments_not_repaid_is_fine(self):
        loan = make_loan(
            disbursal_date=date(2030, 1, 1), loan_status="granted", payments=[],
        )
        found = detect_data_validity(loan)
        assert not any(a.code == "FUTURE_DISBURSAL" for a in found)

    def test_zero_dti_with_income_on_granted_loan_is_flagged(self):
        loan = make_loan(
            dti=0.0, borrower_income=1000.0, loan_status="granted",
        )
        found = detect_data_validity(loan)
        assert any(a.code == "ZERO_DTI" for a in found)

    def test_nonzero_dti_is_fine(self):
        loan = make_loan(dti=25.0, borrower_income=1000.0, loan_status="granted")
        found = detect_data_validity(loan)
        assert not any(a.code == "ZERO_DTI" for a in found)

    def test_unexpected_arrears_on_healthy_loan_is_flagged(self):
        loan = make_loan(arrears=50.0, days_late=10, loan_status="granted")
        found = detect_data_validity(loan)
        assert any(a.code == "UNEXPECTED_ARREARS" for a in found)

    def test_arrears_on_terminated_loan_is_fine(self):
        loan = make_loan(arrears=50.0, days_late=10, loan_status="terminated")
        found = detect_data_validity(loan)
        assert not any(a.code == "UNEXPECTED_ARREARS" for a in found)

    def test_arrears_on_delinquent_loan_is_fine(self):
        loan = make_loan(arrears=50.0, days_late=95, loan_status="granted")
        found = detect_data_validity(loan)
        assert not any(a.code == "UNEXPECTED_ARREARS" for a in found)

    def test_age_over_90_is_data_entry_error(self):
        loan = make_loan(birth_year=1920, disbursal_date=date(2024, 1, 1))
        found = detect_data_validity(loan)
        assert any(a.code == "AGE_DATA_ENTRY_ERROR" for a in found)
        age_finding = [a for a in found if a.code == "AGE_DATA_ENTRY_ERROR"][0]
        assert age_finding.severity is Severity.MEDIUM

    def test_normal_age_no_data_entry_error(self):
        loan = make_loan(birth_year=1990, disbursal_date=date(2024, 1, 1))
        found = detect_data_validity(loan)
        assert not any(a.code == "AGE_DATA_ENTRY_ERROR" for a in found)


# --- Duplicate Loan IDs (tape-level) -----------------------------------------


class TestDuplicateLoanIds:
    def test_duplicate_ids_are_flagged(self):
        from src.models import Loan
        loans = [Loan(loan_id=1), Loan(loan_id=1), Loan(loan_id=2)]
        findings = detect_duplicate_loan_ids(loans)
        assert 1 in findings
        assert 2 not in findings
        assert findings[1][0].code == "DUPLICATE_LOAN_ID"

    def test_unique_ids_produce_no_findings(self):
        from src.models import Loan
        loans = [Loan(loan_id=1), Loan(loan_id=2), Loan(loan_id=3)]
        assert detect_duplicate_loan_ids(loans) == {}


# --- Field completeness ------------------------------------------------------


class TestFieldCompleteness:
    def test_missing_business_fields_are_flagged(self):
        loan = make_loan(
            borrower_type="business",
            annual_revenue=None,
            number_of_employees=None,
            company_type=None,
        )
        found = detect_field_completeness(loan)
        assert len(found) == 1
        assert found[0].code == "FIELD_COMPLETENESS"
        assert found[0].severity is Severity.LOW
        assert "Annual revenue" in found[0].reason

    def test_complete_business_fields_are_fine(self):
        loan = make_loan(
            borrower_type="business",
            annual_revenue=500000.0,
            number_of_employees=10,
            company_type="LLC",
        )
        assert detect_field_completeness(loan) == []

    def test_missing_individual_fields_are_flagged(self):
        loan = make_loan(
            borrower_type="individual", birth_year=None, employment_status=None,
        )
        found = detect_field_completeness(loan)
        assert len(found) == 1
        assert found[0].severity is Severity.LOW
        assert "Birth year" in found[0].reason

    def test_complete_individual_fields_are_fine(self):
        loan = make_loan(
            borrower_type="individual", birth_year=1990,
            employment_status="employed full time",
        )
        assert detect_field_completeness(loan) == []

    def test_unknown_borrower_type_is_skipped(self):
        loan = make_loan(borrower_type=None)
        assert detect_field_completeness(loan) == []


# --- Categorical value validation ---------------------------------------------


class TestCategoricalValues:
    def test_invalid_credit_score_is_flagged(self):
        loan = make_loan(credit_score="X")
        found = detect_categorical_values(loan)
        assert len(found) == 1
        assert found[0].code == "INVALID_CATEGORICAL_VALUE"
        assert found[0].severity is Severity.MEDIUM

    def test_valid_credit_scores_are_fine(self):
        for score in ("A", "B", "C", "D"):
            assert detect_categorical_values(make_loan(credit_score=score)) == []

    def test_invalid_loan_type_is_flagged(self):
        loan = make_loan(loan_type="balloon")
        found = detect_categorical_values(loan)
        assert any(a.code == "INVALID_CATEGORICAL_VALUE" for a in found)

    def test_valid_loan_types_are_fine(self):
        for lt in ("instalment", "deferred annuity"):
            assert detect_categorical_values(make_loan(loan_type=lt)) == []

    def test_null_value_is_not_flagged(self):
        loan = make_loan(credit_score=None, employment_status=None)
        assert detect_categorical_values(loan) == []

    def test_valid_employment_statuses_are_fine(self):
        for status in ("employed full time", "self employed", "unemployed"):
            assert detect_categorical_values(make_loan(employment_status=status)) == []


# --- Reference loan regression for new detectors -----------------------------


class TestNewDetectorsOnReferenceLoanRegression:
    @pytest.mark.parametrize("loan_id", CLEAN_LOANS)
    def test_schedule_integrity_clean(self, by_id, loan_id):
        assert detect_schedule_integrity(by_id[loan_id]) == []

    @pytest.mark.parametrize("loan_id", CLEAN_LOANS)
    def test_age_at_origination_clean(self, by_id, loan_id):
        assert detect_age_at_origination(by_id[loan_id]) == []

    @pytest.mark.parametrize("loan_id", CLEAN_LOANS)
    def test_monthly_payment_vs_income_clean(self, by_id, loan_id):
        assert detect_monthly_payment_vs_income(by_id[loan_id]) == []

    @pytest.mark.parametrize("loan_id", CLEAN_LOANS)
    def test_field_completeness_clean(self, by_id, loan_id):
        assert detect_field_completeness(by_id[loan_id]) == []

    @pytest.mark.parametrize("loan_id", CLEAN_LOANS)
    def test_categorical_values_clean(self, by_id, loan_id):
        assert detect_categorical_values(by_id[loan_id]) == []
