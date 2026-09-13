"""Fargate entrypoint: run the Part 1 pipeline against an S3-hosted loan tape.

Step 2 of the onboarding state machine (see infra/cdk/loan_pipeline_stack.py)
overrides three environment variables on this container: INPUT_BUCKET,
INPUT_KEY (the validated raw upload), and LENDER_ID. This script downloads
that object, runs the existing `src` pipeline against it unchanged, and
uploads the JSON report to OUTPUT_BUCKET/OUTPUT_PREFIX/{lender_id}/{run_id}/
report.json for the persist Lambda to pick up.

Kept as a thin wrapper on purpose: all detection logic stays in src/, so the
same code path this container runs is the one Part 1's test suite exercises.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline  # noqa: E402
from src.reporter import write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("run_pipeline")


def main() -> int:
    input_bucket = os.environ["INPUT_BUCKET"]
    input_key = os.environ["INPUT_KEY"]
    lender_id = os.environ["LENDER_ID"]
    output_bucket = os.environ.get("OUTPUT_BUCKET", input_bucket)
    output_prefix = os.environ.get("OUTPUT_PREFIX", "processed/").rstrip("/")
    run_id = os.environ.get("RUN_ID", uuid.uuid4().hex[:12])

    s3 = boto3.client("s3")
    work_dir = Path("/tmp/loan-tape")
    work_dir.mkdir(parents=True, exist_ok=True)
    local_tape = work_dir / Path(input_key).name
    local_report = work_dir / "report.json"

    logger.info("Downloading s3://%s/%s", input_bucket, input_key)
    s3.download_file(input_bucket, input_key, str(local_tape))

    outcome = run_pipeline(local_tape)
    write_json(outcome, local_report)

    output_key = f"{output_prefix}/{lender_id}/{run_id}/report.json"
    logger.info("Uploading report to s3://%s/%s", output_bucket, output_key)
    s3.upload_file(str(local_report), output_bucket, output_key)

    logger.info(
        "Done: %d/%d loans flagged for lender=%s run=%s",
        len(outcome.flagged), outcome.total, lender_id, run_id,
    )

    # Printed to CloudWatch Logs. EcsRunTask does not return this to the
    # state machine directly; a production version would pass an
    # `activity token` (Step Functions callback pattern) instead, which is
    # a scope call-out for a future iteration, not part of this sample.
    print(json.dumps({
        "bucket": output_bucket,
        "key": output_key,
        "lender_id": lender_id,
        "run_id": run_id,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
