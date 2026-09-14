# Loan-Tape Data Quality Pipeline

Data quality pipeline that ingests loan tape Excel files, detects anomalies in borrower and payment data, and produces a structured report. Built for the Exaloan platform engineering take-home.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src loans.xlsx                  # writes output/report.json
pytest                                    # runs the full test suite
```

**Results:** 72 loans analysed, 20 flagged (27.8%), 52 normal, 13 data warnings (truncated exports). Both clean reference loans (32271989, 99981632) produce zero findings. Loan 37216892 is flagged CRITICAL with four anomalies.

## Approach and Design Decisions

Each detector is a pure function: takes a `Loan`, returns a list of `Anomaly` findings or an empty list. No shared state, no side effects, no dependency on other detectors. Adding or removing a rule is a one-line change to the registry. This makes them independently testable, trivially parallelizable, and safe to extend.

Truncated payment histories (13 loans hit Excel's 32767-character cell limit) are handled as **data warnings**, not anomalies. A loan marked NORMAL with a data warning means "no anomaly detected, but data quality prevented complete verification" -- the cash-flow checks (XIRR, amortization) could not run, so their absence is noted rather than silently omitted.

The pipeline never crashes on bad data. Each loan is processed in isolation with per-loan and per-detector error catching, so one broken row or one failing rule does not stop the rest.

## Detection Methodology

Thirteen independent detectors, each documented in `src/detectors.py`:

| # | Rule | Severity | What it catches |
|---|------|----------|-----------------|
| 1 | Payment default (90+ days or pending late) | CRITICAL | 6 loans with overdue/stalled payments |
| 2 | Near-zero interest vs stated rate | HIGH | 5 loans booking effectively zero interest |
| 3 | XIRR vs stated rate mismatch (>5pp) | HIGH | 17 loans with realized rate divergence |
| 4 | Employment contradiction | MEDIUM | 4 "unemployed" borrowers with occupations |
| 5 | Income > family income | MEDIUM | 1 impossible income relationship |
| 6 | Loan status cross-check | CRITICAL | 1 terminated loan (37216892) |
| 7 | Amortization formula mismatch | HIGH | 0 in this tape |
| 8 | Structural data validity | HIGH/MEDIUM | 0 in this tape |
| 9 | Schedule integrity (gaps, spikes) | MEDIUM | 0 in this tape |
| 10 | Underage borrower at origination | HIGH | 0 in this tape |
| 11 | Monthly payment exceeds income | MEDIUM | 0 in this tape |
| 12 | Field completeness by borrower type | LOW | 0 in this tape |
| 13 | Categorical value validation | MEDIUM | 0 in this tape |

Detectors 7 through 13 catch zero loans in this tape. They are production safety nets for future lender exports -- different originators produce different data quality problems, and having these checks ready means new issues are caught on arrival rather than after manual review.

Key calibration decisions: XIRR uses one-sided checking for deferred annuity loans (their structure realizes ~55% of the nominal rate by design). The amortization rule stands down when the payment schedule runs well past the stated term (restructured loans). The interest detector falls back to summing payment-history entries when summary columns are NULL.

## Assumptions

- **90-day default threshold** follows standard lending practice; payments settled more than 90 days late indicate a real default, not administrative delay.
- **Deferred annuity loans** structurally realize below their nominal interest rate due to their interest-only-then-principal payment structure. The XIRR check is one-sided: only a realized rate *above* the stated rate is anomalous.
- **Payment data truncation** is an Excel export artifact (32767-character cell limit), not a loan anomaly. Truncated loans are marked NORMAL with a data warning.
- **Monthly payment field includes contract fees**, so the amortization check adds the median contract fee to the expected annuity payment before comparing.

## Scaling Beyond 72 Loans

The loader uses openpyxl's read-only streaming mode, not `pd.read_excel`. Only one row is in memory at a time during ingestion, so the file size is bounded by disk, not RAM. `iter_loans()` yields `Loan` objects lazily from disk without ever materializing the full tape. This already handles tens of thousands of loans on a single machine.

At a million rows, three things would change:

1. **Chunked detect-and-write.** Instead of collecting all loans into a list, process in chunks of ~1,000: load a chunk via `iter_loans()`, run detectors, write results to the JSON report (or a database sink), then free the chunk. Peak memory stays proportional to chunk size, not tape size. Tape-level detectors like duplicate-ID detection would run in a separate pass using only the loan ID column, not the full loan objects.

2. **Parallel detection.** Each detector is a pure function with no shared state, so detection parallelizes trivially. A `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor` can distribute chunks across cores. On AWS, the same property lets Fargate tasks each process a slice of the tape independently, with results merged into a shared database.

3. **Format change.** Excel's row limits (1,048,576) and cell-size limits (32,767 characters) make it a poor container at scale. The same loader pattern works with Parquet or CSV, which have no such limits and support true streaming reads. The `row_to_loan` function is format-agnostic -- it takes a `dict[str, Any]`, so swapping the reader is a one-file change in `loader.py`.

## What I Would Improve With More Time

- **Cross-tape balance continuity.** Compare opening balances in a new tape against closing balances in the previous one to catch gaps or double-counted periods.
- **Loan count reconciliation.** Track expected vs actual loan counts per lender across submissions. A sudden drop or spike signals a broken export.
- **Duplicate borrower detection.** Fuzzy matching on borrower name, ID, and address across lenders to flag the same individual appearing under different identifiers.
- **LTV and collateral parsing.** Extract loan-to-value ratios and collateral descriptions when present, flag loans where the LTV exceeds lender policy thresholds.
- **Statistical baseline per lender.** Build rolling distributions of flag rates, interest spreads, and default frequencies per lender. Flag submissions that deviate significantly from that lender's historical norm.
- **Configurable thresholds.** Move magic numbers (90-day default, 5pp XIRR tolerance) into a per-lender config file so operations can tune without code changes.

## AI Usage

An LLM (Claude) assisted with code generation, test writing, and threshold calibration during development. No LLM runs at inference time: every detector is deterministic Python with no model calls, API dependencies, or probabilistic outputs. The pipeline produces the same results on every run.

## Project Structure

```
src/                        # Part 1: detection pipeline
    models.py               # Loan, Payment, Anomaly, LoanResult, Severity
    loader.py               # Excel ingestion, payment parsing, truncation repair
    detectors.py            # 13 detection rules + 1 warning rule
    pipeline.py             # load -> detect -> report orchestration
    reporter.py             # JSON output and console summary
tests/                      # pytest suite
infra/                      # Part 2: AWS architecture
    diagram.py              # diagrams-as-code source
    diagram.png             # architecture diagram
    cdk/                    # AWS CDK stack (passes cdk synth)
.github/workflows/
    ci.yml                  # CI/CD pipeline (GitHub Actions)
docker/
    run_pipeline.py         # Fargate entrypoint
Dockerfile                  # Part 1 pipeline container
AI_DISCLOSURE.md            # AI usage disclosure
```
