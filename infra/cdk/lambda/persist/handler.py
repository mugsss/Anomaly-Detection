"""Step 3 of the onboarding state machine: persist pipeline results to ArangoDB.

Triggered after the Fargate step (loan_pipeline_task.py) writes its report to
processed/{lender_id}/{run_id}/report.json. Reads that report from S3 and
upserts one document per loan into ArangoDB, so downstream scoring and
reporting can query current results without touching S3 directly.

ArangoDB connection details come from Secrets Manager, not environment
variables, since this Lambda only needs read access to one secret and no
plaintext credential should live in the Lambda configuration or in this
repository. Exaloan's ArangoDB cluster already exists; this function only
writes to it.
"""

from __future__ import annotations

import json
import logging
import os

import boto3
import urllib3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
secrets = boto3.client("secretsmanager")
http = urllib3.PoolManager()

ARANGO_SECRET_ARN = os.environ["ARANGO_SECRET_ARN"]
ARANGO_COLLECTION = os.environ.get("ARANGO_COLLECTION", "loan_anomaly_results")


def handler(event: dict, context) -> dict:
    """Read the pipeline's report from S3 and upsert it into ArangoDB.

    Expects `bucket`, `key` (the report.json written by the Fargate step),
    `lender_id`, and `run_id` in the event, as produced by the Step Functions
    state machine after the ECS task completes.
    """
    bucket = event["bucket"]
    key = event["key"]
    lender_id = event["lender_id"]
    run_id = event["run_id"]

    logger.info("Persisting results for lender=%s run=%s from s3://%s/%s",
                lender_id, run_id, bucket, key)

    report = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    loans = report.get("loans", [])

    credentials = _get_arango_credentials()
    written = _upsert_documents(credentials, lender_id, run_id, loans)

    logger.info("Wrote %d/%d loan results to ArangoDB", written, len(loans))
    return {
        "lender_id": lender_id,
        "run_id": run_id,
        "loans_persisted": written,
        "loans_flagged": sum(1 for loan in loans if loan.get("status") == "FLAGGED"),
    }


def _get_arango_credentials() -> dict:
    secret = secrets.get_secret_value(SecretId=ARANGO_SECRET_ARN)
    return json.loads(secret["SecretString"])


def _upsert_documents(credentials: dict, lender_id: str, run_id: str, loans: list[dict]) -> int:
    """Upsert one document per loan via ArangoDB's HTTP API.

    Uses the document API's `overwriteMode=update` so re-running a tape
    (a lender re-uploads a corrected file) updates existing records instead
    of duplicating them. A single AQL upsert-per-batch call would be more
    efficient at real volume; this is deliberately the simple version for a
    representative sample.
    """
    base_url = credentials["url"].rstrip("/")
    db = credentials["database"]
    auth_headers = {
        "Authorization": f"Bearer {credentials['token']}",
        "Content-Type": "application/json",
    }

    written = 0
    for loan in loans:
        document = {
            **loan,
            "_key": f"{lender_id}_{loan['loan_id']}_{run_id}",
            "lender_id": lender_id,
            "run_id": run_id,
        }
        response = http.request(
            "POST",
            f"{base_url}/_db/{db}/_api/document/{ARANGO_COLLECTION}?overwriteMode=update",
            body=json.dumps(document).encode("utf-8"),
            headers=auth_headers,
        )
        if response.status >= 300:
            logger.error(
                "Failed to persist loan %s: HTTP %d %s",
                loan.get("loan_id"), response.status, response.data[:500],
            )
            continue
        written += 1

    return written
