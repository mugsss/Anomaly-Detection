#!/usr/bin/env python3
"""CDK entry point for the loan-tape onboarding pipeline (Part 2).

Run `cdk synth` from this directory (infra/cdk) with the dependencies in
requirements.txt installed. Nothing here is deployed as part of the
assignment; only `cdk synth` is expected to pass.

The stack's environment is left agnostic (no fixed account/region) unless
`-c account=... -c region=...` is passed on the CDK CLI. This is deliberate:
pinning a concrete account here would make CDK treat availability-zone
lookups for the VPC as real AWS API calls, which fails without credentials.
Left agnostic, CDK resolves those with placeholder values and `cdk synth`
runs fully offline. A real deployment supplies the actual account and
region via context or `CDK_DEFAULT_ACCOUNT`/`CDK_DEFAULT_REGION`.
"""

import aws_cdk as cdk
from loan_pipeline_stack.loan_pipeline_stack import LoanPipelineStack

app = cdk.App()

account = app.node.try_get_context("account")
region = app.node.try_get_context("region")

LoanPipelineStack(
    app,
    "ExaloanLoanPipelineStack",
    description="Loan-tape onboarding",
    env=cdk.Environment(account=account, region=region) if account or region else None,
)

app.synth()
