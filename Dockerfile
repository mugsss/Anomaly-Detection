# Part 1 pipeline, packaged for the Fargate step of the onboarding state
# machine (infra/cdk/loan_pipeline_stack.py). Built and pushed to ECR by
# .github/workflows/ci.yml.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first so this layer is cached across builds that only touch
# source code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "boto3>=1.34"

COPY src/ ./src/
COPY docker/run_pipeline.py ./docker/run_pipeline.py

# Runs as an unprivileged user; the container only needs to read/write S3
# and call ArangoDB, never local files outside /tmp.
RUN useradd --create-home --shell /usr/sbin/nologin pipeline
USER pipeline

ENTRYPOINT ["python", "docker/run_pipeline.py"]
