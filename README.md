# Loan-Tape Data Quality Pipeline

Data quality pipeline that ingests loan tape Excel files, detects anomalies in borrower and payment data, and produces a structured report. Built for the Exaloan platform engineering take-home.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src loans.xlsx                  # writes output/report.csv
python -m src loans.xlsx --json output/report.json  # also writes JSON
pytest                                    # 176 tests
```

## Results

| Metric | Count |
|--------|-------|
| Loans analysed | 72 |
| Flagged (anomalies detected) | 20 (27.8%) |
| Normal (no anomalies) | 52 |
| Data warnings (truncated export, not flagged) | 13 |

Reference loans from the assignment brief:
- **32271989**: NORMAL, zero anomalies, no warnings
- **99981632**: NORMAL, zero anomalies, no warnings
- **37216892**: FLAGGED/CRITICAL with 4 anomalies (overdue payments, terminated status, XIRR mismatch, employment contradiction)

## Detection Methodology

The pipeline runs eight independent detectors over each loan. Every detector is a pure function that takes a `Loan` object and returns a list of `Anomaly` findings, or an empty list if nothing is wrong. Detectors never raise exceptions, never mutate the loan, and never depend on each other, so adding or removing a rule is a one-line change to the registry.

Truncated payment histories (13 loans hit Excel's 32767-character cell limit) are handled separately as data warnings, not anomalies. They annotate the report with which checks were skipped, without flagging the loan.

### Rule 1: Payment Default (CRITICAL)

Flags any loan where a payment was settled 90 or more days after its scheduled date, or where payments remain in state `pending late` with no repayment date.

The 90-day threshold follows standard lending practice (and is called out in the assignment). Early repayments that show as negative delays are ignored. For loans like 37216892, the `pending late` state with no settlement date is a separate signal from the delay calculation: the borrower has stopped paying entirely.

**Catches**: 79811839 (613 days), 96579687 (302), 65318525 (237), 61752881 (228), 31397492 (140), 37216892 (7 pending-late payments, 271 days late).

### Rule 2: Interest Calculation (HIGH)

Compares total booked interest (repaid + outstanding) against a rough expected value: `principal * rate * term / 2`. A ratio below 0.05 means the loan collected effectively zero interest despite carrying a real rate.

When the summary interest columns are NULL (three deferred-annuity loans), the detector falls back to summing interest-typed payments from the history. This caught 14146974, 35294697, and 46313736, which book exactly 0.00 interest at 25%, 25%, and 19% respectively.

**Catches**: 17611322 (0.01 on 129 at 33%), 58271697 (0.14 on 367 at 21%), 14146974, 35294697, 46313736 (all zero via payment fallback).

### Rule 3: XIRR vs Stated Rate (HIGH)

Builds borrower-perspective cash flows (disbursal out, every payment in, including contract fees) and computes the internal rate of return using pyxirr. Flags when the realized rate diverges from the stated nominal rate by more than 5 percentage points.

Three guards prevent false positives:

1. **Truncated loans are skipped** since incomplete cash flows produce meaningless rates.
2. **Deferred annuity loans are checked one-sided.** Their structure realizes well below the nominal rate by design (the sample tape shows a consistent ratio near 0.55x), so only a rate *above* the stated one is evidence of a broken calculation. This is why 26961823 (43%) and 82251944 (50%) are correctly not flagged.
3. **Short-horizon loans are skipped.** When the weighted-average life is under ~2.5 months, annualizing produces unstable results. This excludes restructured loan 99981632, whose payment history spans two concatenated schedules.

**Catches**: 17 loans total, including all 5 default loans (whose irregular cash flows distort the IRR), the 5 near-zero-interest loans, and 7 others with material rate divergence.

### Rule 4: Employment Contradiction (MEDIUM)

Flags borrowers recorded as "unemployed" who also carry a non-empty occupation field. All four cases in the tape have `months_at_employer = 0`, reinforcing the contradiction.

**Catches**: 37216892 ("Skyriaus vadovas/shift leader"), 31397492 ("Pardaveja/Saleswoman"), 78579969 ("Suvirintojas/locksmith"), 13314659 ("Aukletojos padejeja/Tutor's helper").

### Rule 5: Income Inconsistency (MEDIUM)

Flags any loan where the borrower's income exceeds the family income. Family income is the household total and includes the borrower, so it is a hard upper bound.

**Catches**: 53926762 (borrower 1041.98, family 870.98, difference 171.00).

### Rule 6: Loan Status (CRITICAL)

Cross-checks the loan status against balances and arrears. A terminated loan is always reported. A "repaid" loan with outstanding principal, or a non-terminated loan 90+ days late, are flagged as contradictions.

**Catches**: 37216892 (terminated with 3666.87 outstanding principal and 271 days late).

### Rule 7: Amortization Formula (HIGH)

For instalment loans, compares the stated monthly payment against the standard annuity formula `P * r(1+r)^n / ((1+r)^n - 1)`, adding the median contract fee (since `Monthly payment` in this tape is the borrower's full instalment). Deferred annuity loans are excluded since the formula does not describe their payment structure.

The rule also requires the stated term to be corroborated by the payment schedule. When the schedule runs well past the stated term (a restructured loan), the formula would be tested against the wrong `n`, so the rule stands down. This is what prevents clean reference loan 99981632 from being flagged: its stated term is 7 months but its history spans 78.

**Catches**: No loans in this tape (the three candidates are already caught by other rules, and their schedule/term mismatch triggers the stand-down).

### Rule 8: Structural Data Validity (HIGH/MEDIUM)

Safety net for fields that other lender tapes commonly get wrong: negative monetary values, missing required fields (disbursal date, amount, rate, term), repayment date before disbursal, and payment records that carry a different loan ID. The current tape is clean on all of these, but the rule exists so the pipeline does not silently pass a broken export.

**Catches**: None in this tape.

## Scaling Beyond 72 Loans

The current implementation loads the full tape into memory with pandas and processes loans sequentially. This already handles thousands of loans without issues: pandas reads a 10,000-row Excel file in seconds, each detector runs in microseconds per loan, and the per-loan memory footprint (a `Loan` dataclass with its parsed payments) is small. 

Three properties of the codebase make scaling straightforward:

### Per-loan isolation

Every detector is a pure function that takes one `Loan` and returns findings. There is no cross-loan state, no shared counters, no global accumulators. Each loan can be processed independently, which means:

- **Parallel detection.** Distributing loans across a `ProcessPoolExecutor` (or across Fargate tasks in Part 2) requires no synchronization. The XIRR solver is the most CPU-intensive rule, and it benefits directly from additional cores.
- **Error containment.** A bad row or a crashing detector is caught per-loan. The pipeline logs the error and continues to the next loan. This is tested: `test_a_failing_detector_does_not_stop_the_run` asserts that one broken detector does not prevent the remaining 71 loans from being processed.

### Streaming-ready loader

The `iter_loans()` generator in `loader.py` already yields one loan at a time. For larger files, replacing the pandas backend with `openpyxl`'s read-only mode (`load_workbook(read_only=True)`) would stream rows without loading the full sheet into memory. The detector pipeline does not care where the `Loan` object came from.

For tapes arriving as CSV or Parquet (common at higher volumes), the same interface works: swap the reader, keep the detectors.

### Additive reporting

The reporter accumulates results incrementally. At higher volumes, writing to a database (PostgreSQL/RDS in the Part 2 architecture) instead of a CSV gives consumers filtering by severity, anomaly type, and date range without loading the full report.

## AI Usage

This project was built with AI-assisted development (Claude Code). AI was used for:

- **Exploratory data analysis**: profiling the dataset, identifying the 13 truncated payment histories, discovering the date format mismatch between loan-level columns (ISO) and payment dicts (dd/mm/yyyy), and calibrating detection thresholds against the reference loans.
- **Code generation**: writing the detector functions, loader, pipeline orchestration, reporter, and test suite. Every function was generated with explicit requirements (the detector interface contract, error isolation, the pure-function constraint) and verified against the known clean and dirty loans before moving to the next piece.
- **Threshold tuning**: the XIRR rule's one-sided deferred-annuity treatment, the amortization rule's term-corroboration guard, and the interest rule's payment-history fallback were all developed iteratively by running the candidate rule against all 72 loans and adjusting until the two clean references produced zero findings while the seeded anomalies were caught.

No LLM runs inside the pipeline at runtime. All detection is deterministic: rule-based checks with fixed thresholds and no probabilistic components. This means every run on the same input produces the same output, every finding can be traced to a specific field comparison, and there is no need for LLM evaluation or fallback logic in production.

## Project Structure

```
src/
    __init__.py
    __main__.py       # CLI entry point: python -m src [loans.xlsx]
    models.py         # Loan, Payment, Anomaly, LoanResult, Severity
    loader.py         # Excel ingestion, payment parsing, truncation repair
    detectors.py      # 8 detection rules + 1 warning rule
    pipeline.py       # orchestration: load -> detect -> report
    reporter.py       # CSV and JSON output
tests/
    conftest.py       # shared fixtures, synthetic loan builders
    test_loader.py    # 41 tests: parsing, truncation, column resolution
    test_detectors.py # 99 tests: per-rule unit + regression on the full tape
    test_pipeline.py  # 36 tests: orchestration, error isolation, reporter output
output/
    report.csv        # per-loan summary (7 columns, 72 rows)
    report.json       # structured report with evidence dicts
requirements.txt
ruff.toml             # lint config (ruff check passes clean)
```

## Design Decisions

**Pure-function detectors.** Each rule is a standalone function with no side effects, no shared state, and no dependency on other rules. This makes them independently testable, trivially parallelizable, and safe to add or remove without regression risk.

**Warnings vs anomalies.** Truncated payment histories are a property of the export, not the loan. Treating them as anomalies inflated the flagged count to 44% and mixed export-quality noise with real findings. Moving them to a separate `data_warnings` channel gives a cleaner signal (20 flagged = 27.8%) while still documenting which checks were skipped.

**Contract-fee-aware amortization.** The tape's `Monthly payment` field is the borrower's total instalment, including the contract fee. Without accounting for the fee, every loan looks 5-20% understated against the annuity formula, and clean reference loan 99981632 gets flagged. Adding the median contract fee to the expected payment eliminates systematic bias.

**One-sided XIRR for deferred annuity.** Deferred annuity loans realize roughly 55% of their nominal rate by design (interest accrues during a grace period but payments are deferred). Checking both directions would flag every deferred annuity loan in the tape. Checking only the upside catches the four loans with genuinely broken interest calculations while leaving the fourteen healthy ones alone.

## What I Would Improve With More Time

- **Term vs schedule mismatch as its own detector.** 20 loans have a stated term that disagrees with their payment schedule by more than one month. The amortization rule currently stands down for these (to avoid false positives), but the mismatch itself is worth reporting as a data quality finding.
- **Payment schedule gap detection.** Look for missing monthly instalments in the payment history (months with no scheduled principal payment), which could indicate deleted records or export errors.
- **Configurable thresholds.** Move the magic numbers (90-day default, 5pp XIRR tolerance, 30% amortization tolerance, 0.05 interest floor) into a YAML or TOML config file so they can be tuned per lender without code changes.
- **Parquet/CSV input support.** The loader currently only handles Excel via openpyxl. Adding a file-type dispatcher would let the same pipeline handle the formats lenders actually use at scale.
