# AI Disclosure

## How AI Was Used

This project was built with Claude Code (Anthropic's CLI agent) as a 
development partner throughout both parts. The collaboration was iterative: 
I described the requirements, reviewed every output, and redirected when 
the result did not match the data or the assignment's intent.

**Part 1 (pipeline):** AI assisted with exploratory data analysis (profiling 
the 72 loans, identifying the 13 truncated payment histories, discovering 
the date format split between loan-level ISO strings and payment-dict 
dd/mm/yyyy). It generated the detector functions, loader, reporter, and 
test suite to my specifications. Each detector was developed by describing 
the rule, generating a candidate, running it against all 72 loans, and 
adjusting until the two clean reference loans (32271989, 99981632) produced 
zero findings while the seeded anomalies were caught.

**Part 2 (infrastructure):** AI drafted the CDK stack, Dockerfile, Fargate 
entrypoint, Lambda handlers, GitHub Actions workflow, and the architecture 
diagram source. I specified the architecture flow (S3 event to Step 
Functions to three-step state machine).

## What I Personally Verified and Changed

- **Detection thresholds.** The 90-day default threshold, 5pp XIRR 
  tolerance, and 0.05 interest floor were calibrated by running candidates 
  against the full tape and checking results loan by loan.
- **XIRR bug fix.** Found that pending late payments and stale early 
  repayment records were being included in XIRR cash flows, inflating 
  the computed rate. Fixing this removed 3 false positives (20 flagged 
  down to 17).
- **Interest fallback.** The payment-history fallback (when summary 
  columns are NULL) was added after noticing three loans with zero 
  booked interest were slipping through the column-based check.
- **Truncation design.** Decided to separate truncated histories into 
  data warnings rather than anomalies, since those loans might be 
  healthy but unverifiable.
