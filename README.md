# Loan-Tape Data Quality Pipeline

Data quality pipeline that ingests loan tape Excel files, detects anomalies in borrower and payment data, and produces a structured report. Built for the Exaloan platform engineering take-home.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src loans.xlsx                  # writes output/report.csv + report.json
pytest                                    # 176 tests
```

**Results:** 72 loans analysed, 20 flagged (27.8%), 52 normal, 13 data warnings (truncated exports). Both clean reference loans (32271989, 99981632) produce zero findings. Loan 37216892 is flagged CRITICAL with four anomalies.

## Approach and Design Decisions

Each detector is a pure function: takes a `Loan`, returns a list of `Anomaly` findings or an empty list. No shared state, no side effects, no dependency on other detectors. Adding or removing a rule is a one-line change to the registry. This makes them independently testable, trivially parallelizable, and safe to extend.

Truncated payment histories (13 loans hit Excel's 32767-character cell limit) are handled as data warnings, not anomalies. Mixing export-quality noise with real findings inflated the flagged count to 44%. Separating them gives a cleaner signal while still documenting which checks were skipped.

The pipeline never crashes on bad data. Each loan is processed in isolation with per-loan error catching, so one broken row does not stop the remaining 71 from being processed.

## Detection Methodology

Eight independent detectors, each documented in `src/detectors.py`:

| # | Rule | Severity | What it catches |
|---|------|----------|-----------------|
| 1 | Payment default (90+ days or pending late) | CRITICAL | 6 loans with overdue/stalled payments |
| 2 | Near-zero interest vs stated rate | HIGH | 5 loans booking effectively zero interest |
| 3 | XIRR vs stated rate mismatch (>5pp) | HIGH | 17 loans with realized rate divergence |
| 4 | Employment contradiction | MEDIUM | 4 "unemployed" borrowers with occupations |
| 5 | Income > family income | MEDIUM | 1 impossible income relationship |
| 6 | Loan status cross-check | CRITICAL | 1 terminated loan (37216892) |
| 7 | Amortization formula mismatch | HIGH | 0 in this tape (guard prevents false positives) |
| 8 | Structural data validity | HIGH/MEDIUM | 0 in this tape (safety net for other exports) |

Key calibration decisions: XIRR uses one-sided checking for deferred annuity loans (their structure realizes ~55% of the nominal rate by design). The amortization rule stands down when the payment schedule runs well past the stated term (restructured loans). The interest detector falls back to summing payment-history entries when summary columns are NULL.

## Architecture (Part 2)

### Why Step Functions + Fargate, Not Just Lambda

The Part 1 pipeline loads the full tape into pandas, parses nested payment dicts, and solves XIRR per loan. For tapes with thousands of rows, this easily exceeds Lambda's 15-minute timeout and 10 GB memory ceiling. Fargate has no fixed limits on either, and it charges per second of actual compute with no idle capacity between runs.

Step Functions orchestrates the three-step flow (validate, run pipeline, persist to ArangoDB) with built-in retry, backoff, and catch chains. Each step's failure path routes to a DLQ and an SNS notification so a stuck onboarding is never silent. This is simpler and more observable than wiring retry logic into application code or chaining Lambdas through SQS.

Lambda handles the two lightweight bookend steps (format validation and ArangoDB persistence) where cold start latency is acceptable and execution time is under 30 seconds.

### Flow

S3 upload (`raw-uploads/{lender_id}/`) triggers an EventBridge rule, which starts the Step Functions state machine. Step 1 (Lambda) validates the file. Step 2 (Fargate) runs the Part 1 pipeline and writes `processed/{lender_id}/{run_id}/report.json`. Step 3 (Lambda) reads the report and upserts results to ArangoDB via its HTTP API. On success, SNS notifies the operations team. On failure after retries, the file reference goes to the DLQ.

S3 lifecycle moves raw uploads to Glacier after 90 days (financial records, never deleted). Old object versions transition after 30 days. CloudWatch retains logs for one month with alarms on state machine failures.

The CDK stack (`infra/cdk/`) synthesizes cleanly with `cdk synth` and models this full flow. The GitHub Actions workflow (`.github/workflows/ci.yml`) runs lint, test, CDK synth, Docker build, and ECR push.

## Scaling Beyond 72 Loans

The current implementation handles thousands of loans without issues: pandas reads a 10,000-row Excel file in seconds, and each detector runs in microseconds per loan. Three properties make further scaling straightforward: per-loan isolation (no cross-loan state, so detection parallelizes trivially across cores or Fargate tasks), a streaming-ready loader (`iter_loans()` yields one loan at a time), and additive reporting (swap the CSV writer for a database sink at higher volumes).

## Production Roadmap

With more time and access to multiple lender tapes:

- **Cross-tape balance continuity.** Compare opening balances in a new tape against closing balances in the previous one to catch gaps or double-counted periods.
- **Loan count reconciliation.** Track expected vs actual loan counts per lender across submissions. A sudden drop or spike signals a broken export.
- **Duplicate borrower detection.** Fuzzy matching on borrower name, ID, and address across lenders to flag the same individual appearing under different identifiers.
- **LTV and collateral parsing.** Extract loan-to-value ratios and collateral descriptions when present, flag loans where the LTV exceeds lender policy thresholds.
- **Statistical baseline per lender.** Build rolling distributions of flag rates, interest spreads, and default frequencies per lender. Flag submissions that deviate significantly from that lender's historical norm.
- **Configurable thresholds.** Move magic numbers (90-day default, 5pp XIRR tolerance) into a per-lender config file so operations can tune without code changes.

## Project Structure

```
src/                        # Part 1: detection pipeline
    models.py               # Loan, Payment, Anomaly, LoanResult, Severity
    loader.py               # Excel ingestion, payment parsing, truncation repair
    detectors.py            # 8 detection rules + 1 warning rule
    pipeline.py             # load -> detect -> report orchestration
    reporter.py             # CSV and JSON output
tests/                      # 176 tests (pytest)
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
