"""Pipeline and reporter tests: orchestration, error isolation, and output."""

from __future__ import annotations

import csv
import json

import pytest

from src.detectors import DETECTORS
from src.models import Anomaly, Loan, LoanResult, Severity
from src.pipeline import PipelineResult, analyse, run_detectors, run_pipeline
from src.reporter import (
    CSV_COLUMNS,
    format_console_summary,
    result_to_row,
    write_csv,
    write_json,
)
from tests.conftest import CLEAN_LOANS, IRREGULAR_LOAN, make_loan

TRUNCATED_LOANS = {
    56721442, 51117775, 72457171, 35612451, 64933882, 13314659, 69683127,
    33817222, 43986823, 25372336, 94146169, 26165342, 95814762,
}


@pytest.fixture(scope="module")
def outcome(tape_path):
    return run_pipeline(tape_path)


class TestErrorIsolation:
    def test_a_failing_detector_does_not_stop_the_others(self):
        def explodes(loan):
            raise RuntimeError("boom")

        def finds_something(loan):
            return [Anomaly("OK", Severity.LOW, "found it")]

        result, errors = run_detectors(make_loan(), [explodes, finds_something])
        assert [a.code for a in result.anomalies] == ["OK"]
        assert len(errors) == 1 and "boom" in errors[0][2]
        assert result.errors

    def test_a_failing_detector_does_not_stop_the_run(self):
        def explodes(loan):
            raise ValueError("nope")

        loans = [make_loan(loan_id=i) for i in range(5)]
        result = analyse(loans, [explodes])
        assert result.total == 5
        assert len(result.detector_errors) == 5

    def test_a_failing_warning_rule_does_not_stop_the_run(self):
        def explodes(loan):
            raise RuntimeError("warn boom")

        result, errors = run_detectors(make_loan(), DETECTORS, [explodes])
        assert result.data_warnings == []
        assert len(errors) == 1 and "warn boom" in errors[0][2]

    def test_a_detector_returning_none_is_tolerated(self):
        result, errors = run_detectors(make_loan(), [lambda loan: None])
        assert result.anomalies == [] and errors == []

    def test_a_loan_with_no_data_still_produces_a_result(self):
        result = analyse([Loan(loan_id=1)], DETECTORS)
        assert result.total == 1


class TestPipelineResult:
    def test_counts_split_flagged_and_clean(self):
        result = PipelineResult(results=[
            LoanResult(1, [Anomaly("A", Severity.HIGH, "r")]),
            LoanResult(2),
        ])
        assert result.total == 2
        assert [r.loan_id for r in result.flagged] == [1]
        assert [r.loan_id for r in result.clean] == [2]

    def test_severity_counts_cover_every_level(self):
        result = PipelineResult(results=[LoanResult(1, [Anomaly("A", Severity.HIGH, "r")])])
        counts = result.severity_counts()
        assert set(counts) == {s.name for s in Severity}
        assert counts["HIGH"] == 1 and counts["NONE"] == 0

    def test_anomaly_counts_are_ordered_by_frequency(self):
        result = PipelineResult(results=[
            LoanResult(1, [Anomaly("A", Severity.LOW, "r"), Anomaly("B", Severity.LOW, "r")]),
            LoanResult(2, [Anomaly("B", Severity.LOW, "r")]),
        ])
        assert list(result.anomaly_counts()) == ["B", "A"]

    def test_max_severity_is_the_worst_anomaly(self):
        result = LoanResult(1, [
            Anomaly("A", Severity.LOW, "r"), Anomaly("B", Severity.CRITICAL, "r"),
        ])
        assert result.max_severity is Severity.CRITICAL
        assert [a.code for a in result.sorted_anomalies()] == ["B", "A"]

    def test_clean_result_reports_none(self):
        assert LoanResult(1).max_severity is Severity.NONE
        assert LoanResult(1).status == "NORMAL"


class TestRealTapeRun:
    def test_every_loan_gets_a_result(self, outcome):
        assert outcome.total == 72
        assert len({r.loan_id for r in outcome.results}) == 72

    def test_no_detector_errors(self, outcome):
        assert outcome.detector_errors == []

    def test_no_rows_were_dropped(self, outcome):
        assert outcome.load_report.failures == []

    @pytest.mark.parametrize("loan_id", CLEAN_LOANS)
    def test_reference_clean_loans_are_normal(self, outcome, loan_id):
        result = next(r for r in outcome.results if r.loan_id == loan_id)
        assert result.status == "NORMAL"
        assert result.max_severity is Severity.NONE

    def test_reference_irregular_loan_is_critical(self, outcome):
        result = next(r for r in outcome.results if r.loan_id == IRREGULAR_LOAN)
        assert result.status == "FLAGGED"
        assert result.max_severity is Severity.CRITICAL

    def test_flagged_share_matches_the_brief(self, outcome):
        """The brief says roughly 30% of the tape is anomalous."""
        assert 0.20 <= len(outcome.flagged) / outcome.total <= 0.40

    def test_truncation_alone_does_not_flag_a_loan(self, outcome):
        """Truncation is an export artifact, so it belongs in warnings only."""
        by_id = {r.loan_id: r for r in outcome.results}
        for loan_id in TRUNCATED_LOANS - {13314659}:
            result = by_id[loan_id]
            assert result.status == "NORMAL"
            assert result.max_severity is Severity.NONE
            assert result.anomalies == []
            assert result.data_warnings

    def test_truncated_loan_with_a_real_issue_stays_flagged(self, outcome):
        result = next(r for r in outcome.results if r.loan_id == 13314659)
        assert result.status == "FLAGGED"
        assert [a.code for a in result.anomalies] == ["EMPLOYMENT_CONTRADICTION"]
        assert result.data_warnings

    def test_every_truncated_loan_carries_a_warning(self, outcome):
        warned = {r.loan_id for r in outcome.with_warnings}
        assert warned == TRUNCATED_LOANS

    def test_loans_with_complete_data_carry_no_warning(self, outcome):
        for result in outcome.results:
            if result.loan_id not in TRUNCATED_LOANS:
                assert result.data_warnings == []

    def test_every_flagged_loan_states_a_reason(self, outcome):
        for result in outcome.flagged:
            assert all(a.reason.strip() for a in result.anomalies)


class TestReporter:
    def test_csv_has_one_row_per_loan(self, outcome, tmp_path):
        path = write_csv(outcome, tmp_path / "report.csv")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        assert len(rows) == outcome.total
        assert list(rows[0]) == list(CSV_COLUMNS)

    def test_csv_lists_clean_loans_as_normal(self, outcome, tmp_path):
        path = write_csv(outcome, tmp_path / "report.csv")
        rows = {int(r["loan_id"]): r for r in csv.DictReader(path.open(encoding="utf-8"))}
        for loan_id in CLEAN_LOANS:
            assert rows[loan_id]["status"] == "NORMAL"
            assert rows[loan_id]["anomaly_count"] == "0"
            assert rows[loan_id]["reasons"] == ""

    def test_csv_orders_worst_severity_first(self, outcome, tmp_path):
        path = write_csv(outcome, tmp_path / "report.csv")
        severities = [
            Severity[r["max_severity"]]
            for r in csv.DictReader(path.open(encoding="utf-8"))
        ]
        assert severities == sorted(severities, reverse=True)

    def test_csv_drops_the_evidence_and_error_columns(self, outcome, tmp_path):
        """The CSV is the scannable summary; structured evidence lives in the JSON."""
        path = write_csv(outcome, tmp_path / "report.csv")
        header = next(csv.reader(path.open(encoding="utf-8")))
        assert "evidence" not in header and "detector_errors" not in header
        assert header == list(CSV_COLUMNS)

    def test_csv_carries_the_data_warning_without_flagging(self, outcome, tmp_path):
        path = write_csv(outcome, tmp_path / "report.csv")
        rows = {int(r["loan_id"]): r for r in csv.DictReader(path.open(encoding="utf-8"))}
        truncated_only = rows[25372336]
        assert truncated_only["status"] == "NORMAL"
        assert truncated_only["max_severity"] == "NONE"
        assert truncated_only["anomaly_count"] == "0"
        assert truncated_only["anomaly_codes"] == ""
        assert "truncated at source export limit" in truncated_only["data_warnings"]

    def test_csv_leaves_data_warnings_empty_for_complete_loans(self, outcome, tmp_path):
        path = write_csv(outcome, tmp_path / "report.csv")
        rows = {int(r["loan_id"]): r for r in csv.DictReader(path.open(encoding="utf-8"))}
        for loan_id in CLEAN_LOANS:
            assert rows[loan_id]["data_warnings"] == ""

    def test_csv_row_for_a_truncated_loan_with_a_real_anomaly(self, outcome, tmp_path):
        path = write_csv(outcome, tmp_path / "report.csv")
        rows = {int(r["loan_id"]): r for r in csv.DictReader(path.open(encoding="utf-8"))}
        row = rows[13314659]
        assert row["status"] == "FLAGGED"
        assert row["anomaly_codes"] == "EMPLOYMENT_CONTRADICTION"
        assert row["anomaly_count"] == "1"
        assert "truncated at source export limit" in row["data_warnings"]

    def test_json_keeps_the_structured_evidence(self, outcome, tmp_path):
        path = write_json(outcome, tmp_path / "report.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        flagged = next(
            entry for entry in payload["loans"] if entry["loan_id"] == IRREGULAR_LOAN
        )
        assert all(a["evidence"] for a in flagged["anomalies"])
        assert payload["summary"]["with_data_warnings"] == 13

    def test_row_carries_severity_names_not_numbers(self):
        result = LoanResult(1, [Anomaly("A", Severity.HIGH, "r")])
        row = result_to_row(result)
        assert row["max_severity"] == "HIGH"
        assert row["reasons"].startswith("[HIGH]")

    def test_json_report_round_trips(self, outcome, tmp_path):
        path = write_json(outcome, tmp_path / "report.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["summary"]["loans_analysed"] == outcome.total
        assert len(payload["loans"]) == outcome.total

    def test_console_summary_names_flagged_and_normal_loans(self, outcome):
        text = format_console_summary(outcome)
        assert f"Loans analysed:      {outcome.total}" in text
        assert str(IRREGULAR_LOAN) in text
        for loan_id in CLEAN_LOANS:
            assert str(loan_id) in text

    def test_output_directory_is_created_on_demand(self, outcome, tmp_path):
        path = write_csv(outcome, tmp_path / "nested" / "deep" / "report.csv")
        assert path.exists()


class TestCommandLine:
    def test_entry_point_writes_a_report(self, tape_path, tmp_path, capsys):
        from src.__main__ import main

        target = tmp_path / "report.csv"
        assert main([str(tape_path), "-o", str(target), "--quiet"]) == 0
        assert target.exists()

    def test_missing_file_exits_non_zero(self, tmp_path):
        from src.__main__ import main

        assert main([str(tmp_path / "nope.xlsx")]) == 2
