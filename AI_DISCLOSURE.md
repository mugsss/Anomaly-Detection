# AI Disclosure

## How AI Was Used

This project was built with Claude Code (Anthropic's CLI agent) as a development partner throughout both parts. The collaboration was iterative, not generative: I described the requirements, reviewed every output, and redirected when the result did not match the data or the assignment's intent.

**Part 1 (pipeline):** AI assisted with exploratory data analysis (profiling the 72 loans, identifying the 13 truncated payment histories, discovering the date format split between loan-level ISO strings and payment-dict dd/mm/yyyy). It generated the detector functions, loader, reporter, and test suite to my specifications. Each detector was developed by describing the rule, generating a candidate, running it against all 72 loans, and adjusting until the two clean reference loans (32271989, 99981632) produced zero findings while the seeded anomalies were caught.

**Part 2 (infrastructure):** AI drafted the CDK stack, Dockerfile, Fargate entrypoint, Lambda handlers, GitHub Actions workflow, and the architecture diagram source. I specified the architecture flow (S3 event to Step Functions to three-step state machine) and the constraints (environment-agnostic synth, no ArangoDB provisioning, per-lender namespacing).

## What I Personally Verified and Changed

- **Detection thresholds.** The 90-day default threshold, 5pp XIRR tolerance, 30% amortization tolerance, and 0.05 interest floor were all calibrated by running candidates against the full tape and checking results loan by loan.
- **XIRR guards.** The one-sided treatment for deferred annuity loans (flag only if realized rate exceeds stated rate) and the short-horizon exclusion were my design decisions after observing that two-sided checks flagged every deferred annuity loan in the tape.
- **Deferred annuity handling.** The interest detector's fallback to summing payment-history interest (when summary columns are NULL) was added after I noticed three deferred annuity loans with zero booked interest were slipping through.
- **Clean loan validation.** Every rule change was tested against both reference loans. The amortization rule's term-corroboration guard exists specifically because 99981632 was being falsely flagged (its stated term is 7 months but its payment history spans 78).
- **CDK synth.** Ran `cdk synth` locally and resolved three issues: environment-agnostic AZ handling, ECR repository reference for auto IAM grants, and the deprecated `logRetention` parameter migration.

## How I Validated AI Output

Every detector was verified against the full 72-loan tape. I compared flagged loans to my manual EDA findings and confirmed that:
- The 20 flagged loans matched the anomalies I identified during data exploration.
- The 52 normal loans included the two clean references with zero findings.
- The 13 data warnings correctly identified truncated exports without over-flagging.

For infrastructure, `cdk synth` validates the CDK stack, `ruff check .` confirms code style, and 176 pytest tests cover the detection logic. The Fargate entrypoint and Lambda handlers were reviewed for correct S3 key conventions and error handling.

I am comfortable discussing any part of this codebase in detail, including the tradeoffs behind each detector's threshold and the architecture decisions in Part 2.
