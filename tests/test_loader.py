"""Ingestion tests: parsing, truncation repair, and tolerance of bad input."""

from __future__ import annotations

from datetime import date

import pytest

from src.loader import (
    EXCEL_CELL_LIMIT,
    _resolve_columns,
    parse_date,
    parse_payments,
    repair_truncated_payload,
    row_to_loan,
)
from tests.conftest import CLEAN_LOANS, IRREGULAR_LOAN

TRUNCATED_LOAN_IDS = {
    56721442, 51117775, 72457171, 35612451, 64933882, 13314659, 69683127,
    33817222, 43986823, 25372336, 94146169, 26165342, 95814762,
}


class TestParseDate:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2022-05-12T00:00:00.000", date(2022, 5, 12)),
            ("2022-05-12", date(2022, 5, 12)),
            ("2022-01-10 00:00:00", date(2022, 1, 10)),
            ("26/04/2024", date(2024, 4, 26)),
            ("01/12/2024", date(2024, 12, 1)),  # day first, not month first
        ],
    )
    def test_parses_known_formats(self, raw, expected):
        assert parse_date(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "nan", "not a date", "2022-13-45"])
    def test_returns_none_for_unusable_values(self, raw):
        assert parse_date(raw) is None

    def test_accepts_datetime_objects(self):
        import datetime as dt

        assert parse_date(dt.datetime(2023, 3, 4, 12, 0)) == date(2023, 3, 4)


class TestTruncationRepair:
    def test_repairs_payload_cut_mid_record(self):
        raw = "[{'a': 1}, {'a': 2}, {'a': 3, 'b': 'half"
        payload, repaired = repair_truncated_payload(raw)
        assert repaired is True
        assert payload == "[{'a': 1}, {'a': 2}]"

    def test_leaves_complete_payload_alone(self):
        raw = "[{'a': 1}]"
        assert repair_truncated_payload(raw) == (raw, False)

    def test_handles_payload_with_no_complete_record(self):
        payload, repaired = repair_truncated_payload("[{'a': 1")
        assert repaired is False


class TestParsePayments:
    def test_parses_well_formed_history(self):
        raw = (
            "[{'Loan ID': '99', 'Payment date': '26/04/2024', "
            "'Repayment date': '30/04/2024', 'Type': 'principal', "
            "'State': 'paid with delay', 'Amount': 10.5, 'Pending amount': 0}]"
        )
        payments, truncated, warnings = parse_payments(raw)
        assert truncated is False and warnings == []
        assert len(payments) == 1
        payment = payments[0]
        assert payment.loan_id == 99
        assert payment.due_date == date(2024, 4, 26)
        assert payment.paid_date == date(2024, 4, 30)
        assert payment.delay_days == 4
        assert payment.amount == 10.5

    def test_recovers_records_from_truncated_history(self):
        raw = (
            "[{'Loan ID': '1', 'Payment date': '01/01/2024', 'Repayment date': None, "
            "'Type': 'principal', 'State': 'pending', 'Amount': 5, 'Pending amount': 5}, "
            "{'Loan ID': '1', 'Payment date': '01/02/2024', 'Repayment dat"
        )
        payments, truncated, warnings = parse_payments(raw)
        assert truncated is True
        assert len(payments) == 1
        assert any("truncated" in w for w in warnings)

    def test_null_repayment_date_becomes_none(self):
        raw = (
            "[{'Loan ID': '1', 'Payment date': '01/01/2024', 'Repayment date': None, "
            "'Type': 'interest', 'State': 'pending', 'Amount': 5, 'Pending amount': 5}]"
        )
        payments, _, _ = parse_payments(raw)
        assert payments[0].paid_date is None
        assert payments[0].delay_days is None
        assert payments[0].is_settled is False

    @pytest.mark.parametrize("raw", [None, "", "nan"])
    def test_empty_history_is_not_an_error(self, raw):
        payments, truncated, _ = parse_payments(raw)
        assert payments == [] and truncated is False

    def test_garbage_history_does_not_raise(self):
        payments, _, warnings = parse_payments("this is not a list at all")
        assert payments == [] and warnings

    def test_skips_non_mapping_records_but_keeps_the_rest(self):
        raw = (
            "['junk', {'Loan ID': '1', 'Payment date': '01/01/2024', "
            "'Repayment date': None, 'Type': 'interest', 'State': 'pending', "
            "'Amount': 5, 'Pending amount': 5}]"
        )
        payments, _, warnings = parse_payments(raw)
        assert len(payments) == 1
        assert any("not a mapping" in w for w in warnings)

    def test_missing_keys_fall_back_to_defaults(self):
        payments, _, _ = parse_payments("[{'Amount': 3}]")
        assert payments[0].payment_type == "unknown"
        assert payments[0].state == "unknown"
        assert payments[0].due_date is None


class TestColumnResolution:
    def test_matches_columns_regardless_of_case_and_punctuation(self):
        resolved = _resolve_columns(["Loan_ID", "loan amount", "INTEREST RATE", "payments"])
        assert resolved["loan_id"] == "Loan_ID"
        assert resolved["loan_amount"] == "loan amount"
        assert resolved["interest_rate"] == "INTEREST RATE"

    def test_missing_columns_are_simply_absent(self):
        resolved = _resolve_columns(["Loan ID"])
        assert "monthly_payment" not in resolved

    def test_row_without_loan_id_is_rejected(self):
        with pytest.raises(ValueError, match="Loan ID"):
            row_to_loan({"Loan amount": 100}, {"loan_amount": "Loan amount"}, 2)

    def test_row_with_only_a_loan_id_still_loads(self):
        loan = row_to_loan({"Loan ID": 7}, {"loan_id": "Loan ID"}, 2)
        assert loan.loan_id == 7
        assert loan.loan_amount is None
        assert loan.payments == []


class TestRealTape:
    def test_loads_every_row(self, loans, load_report):
        assert len(loans) == 72
        assert load_report.rows_seen == 72
        assert load_report.failures == []

    def test_no_expected_columns_are_missing(self, load_report):
        assert load_report.missing_columns == []

    def test_loan_ids_are_unique(self, loans):
        ids = [loan.loan_id for loan in loans]
        assert len(set(ids)) == len(ids)

    def test_identifies_exactly_the_truncated_histories(self, loans, load_report):
        truncated = {loan.loan_id for loan in loans if loan.payments_truncated}
        assert truncated == TRUNCATED_LOAN_IDS
        assert load_report.truncated_payments == 13

    def test_every_loan_has_payments(self, loans):
        assert all(loan.payments for loan in loans)

    def test_truncated_loans_still_recover_payments(self, by_id):
        for loan_id in TRUNCATED_LOAN_IDS:
            assert len(by_id[loan_id].payments) > 0

    @pytest.mark.parametrize("loan_id", [*CLEAN_LOANS, IRREGULAR_LOAN])
    def test_reference_loans_parse_completely(self, by_id, loan_id):
        loan = by_id[loan_id]
        assert loan.payments_truncated is False
        assert loan.parse_warnings == []
        assert loan.loan_amount and loan.interest_rate and loan.disbursal_date

    def test_irregular_reference_loan_details(self, by_id):
        loan = by_id[IRREGULAR_LOAN]
        assert loan.loan_status == "terminated"
        assert loan.days_late == 271
        assert sum(1 for p in loan.payments if p.state == "pending late") == 7

    def test_clean_reference_loan_has_only_minor_delays(self, by_id):
        loan = by_id[99981632]
        delays = [p.delay_days for p in loan.payments if p.delay_days is not None]
        assert max(delays) < 90

    def test_payment_records_belong_to_their_loan(self, loans):
        for loan in loans:
            foreign = {p.loan_id for p in loan.payments if p.loan_id is not None}
            assert foreign <= {loan.loan_id}

    def test_cell_limit_constant_matches_observed_truncation(self, tape_path):
        import pandas as pd

        lengths = pd.read_excel(tape_path)["payments"].astype(str).str.len()
        assert lengths.max() == EXCEL_CELL_LIMIT
