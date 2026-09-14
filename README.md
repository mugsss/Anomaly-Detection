# Loan-Tape Data Quality Pipeline

Data quality pipeline that ingests loan tape Excel files, detects anomalies in borrower and payment data, and produces a structured report. Built for the Exaloan platform engineering take-home.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src loans.xlsx                  # writes output/report.json
pytest                                    # runs the full test suite
```

**Results:** 72 loans analysed, 17 flagged (23.6%), 55 normal, 13 data warnings (truncated exports). Both clean reference loans (32271989, 99981632) produce zero findings. Loan 37216892 is flagged CRITICAL with four anomalies.

## Approach and Design Decisions

Each detector is a pure function: takes a `Loan`, returns a list of `Anomaly` findings or an empty list. No shared state, no side effects, no dependency on other detectors. Adding or removing a rule is a one-line change to the registry. This makes them independently testable, trivially parallelizable, and safe to extend.

Truncated payment histories (13 loans hit Excel's 32767-character cell limit) are handled as **data warnings**, not anomalies. A loan marked NORMAL with a data warning means "no anomaly detected, but data quality prevented complete verification" -- the cash-flow checks (XIRR) could not run, so their absence is noted rather than silently omitted.

The pipeline never crashes on bad data. Each loan is processed in isolation with per-loan and per-detector error catching, so one broken row or one failing rule does not stop the rest.

## Detection Methodology

Eleven independent detectors, each documented in `src/detectors.py`:

| # | Rule | Severity | What it catches |
|---|------|----------|-----------------|
| 1 | Payment default (90+ days or pending late) | CRITICAL | 6 loans with overdue/stalled payments |
| 2 | Near-zero interest vs stated rate | HIGH | 5 loans booking effectively zero interest |
| 3 | XIRR vs stated rate mismatch (>5pp) | HIGH | 11 loans with realized rate divergence |
| 4 | Employment contradiction | MEDIUM | 4 "unemployed" borrowers with occupations |
| 5 | Income > family income | MEDIUM | 1 impossible income relationship |
| 6 | Loan status cross-check | CRITICAL | 1 terminated loan (37216892) |
| 7 | Structural data validity (negatives, missing fields, duplicates, pre-disbursal payments) | HIGH/MEDIUM | 0 in this tape |
| 8 | Underage borrower at origination | HIGH | 0 in this tape |
| 9 | Monthly payment exceeds income | MEDIUM | 0 in this tape |
| 10 | Field completeness by borrower type | LOW | 0 in this tape |
| 11 | Categorical value validation | MEDIUM | 0 in this tape |

Detectors 7 through 11 catch zero loans in this tape. They are production safety nets for future lender exports -- different originators produce different data quality problems, and having these checks ready means new issues are caught on arrival rather than after manual review.


## Assumptions

- **90-day default threshold** follows standard lending practice; payments settled more than 90 days late indicate a real default, not administrative delay.
- **Deferred annuity loans** structurally realize below their nominal interest rate due to their interest-only-then-principal payment structure. The XIRR check is one-sided: only a realized rate *above* the stated rate is anomalous.
- **Payment data truncation** is an Excel export artifact (32767-character cell limit), not a loan anomaly. Truncated loans are marked NORMAL with a data warning.

## Scaling Beyond 72 Loans

The loader uses openpyxl's read-only streaming mode (not `pd.read_excel`), so only one row is in memory at a time. `iter_loans()` yields `Loan` objects lazily from disk. This already handles tens of thousands of loans on a single machine.

At a million rows: chunk into batches of ~1,000 (detect and write per chunk, then free), parallelize across cores via `ProcessPoolExecutor` (each detector is a pure function with no shared state), and swap Excel for Parquet/CSV (no row or cell-size limits). The `row_to_loan` function takes a `dict[str, Any]`, so changing the reader is a one-file change in `loader.py`.


## AI Usage

**Part 1:** Claude (Anthropic) for EDA, code generation (loader, detectors, pipeline, tests), and threshold calibration. **Part 2:** CDK stack generation, CI/CD pipeline definition, Dockerfile, Fargate wrapper, and the diagrams-as-code diagram.


### What I personally verified and changed

**Detection correctness:** Validated every detector against the three reference loans. Both clean loans produce zero findings; the irregular loan is caught as CRITICAL.

**XIRR bug fix:** Found and fixed a bug where pending late payments and stale early repayment records inflated XIRR. Excluding cash flows that never arrived removed 3 false positives (20 flagged down to 17).

**Truncation design:** Decided to separate truncated histories into data warnings rather than anomalies, since those loans might be healthy but unverifiable.

**Architecture decisions:** I chose the service boundaries based on the workload characteristics: Lambda for the fast validation gate (fail in seconds before paying Fargate cold-start cost), Fargate for the detection pipeline (no timeout or memory limits for large tapes), and ArangoDB for results persistence (the pipeline output is document-shaped with variable nested anomalies, and ArangoDB's graph capability supports cross-lender and cross-borrower queries at scale).

## Project Structure

```
src/                        # Part 1: detection pipeline
    models.py               # Loan, Payment, Anomaly, LoanResult, Severity
    loader.py               # Excel ingestion, payment parsing, truncation repair
    detectors.py            # 11 detection rules + 1 warning rule
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
```
