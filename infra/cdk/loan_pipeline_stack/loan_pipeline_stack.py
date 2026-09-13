"""Representative CDK stack for the loan-tape onboarding pipeline.

Models the flow described in the assignment: a lender drops a loan tape into
S3, an S3 event drives a Step Functions state machine through three steps
(validate, run the Part 1 pipeline on Fargate, persist to ArangoDB), and the
raw file plus results are retained with lifecycle rules for cost control.

This is a representative sample, not full coverage: one ingestion trigger,
one storage resource, and one compute resource running the Part 1 pipeline,
per the assignment's scope. It does not provision ArangoDB (Exaloan already
runs that); the persist Lambda reaches it over the network via a Secrets
Manager credential.
"""

from __future__ import annotations

from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sqs as sqs
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct

# Placeholder repository name for the Part 1 pipeline image. Real onboarding
# points this at the ECR repository the CI pipeline pushes to (see
# .github/workflows/ci.yml). Referencing it via ecr.Repository rather
# than a raw registry URL lets CDK grant the task execution role pull
# permissions automatically instead of requiring a hand-written policy.
PIPELINE_ECR_REPOSITORY_NAME = "loan-pipeline"
PIPELINE_IMAGE_TAG = "latest"

# S3 prefixes, not separate buckets. One versioned bucket keeps a single
# IAM/lifecycle surface while still giving each stage of the flow (and each
# lender within a stage) its own namespace: raw-uploads/{lender_id}/{file},
# processed/{lender_id}/{run_id}/report.json, archive/{lender_id}/{file}
# (objects moved here by lifecycle after 90 days, see below).
RAW_PREFIX = "raw-uploads/"
PROCESSED_PREFIX = "processed/"
ARCHIVE_PREFIX = "archive/"


class LoanPipelineStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------------------------------------------------------------
        # Storage: one bucket, three prefixes, per-lender namespacing
        # inside each. EventBridge notifications are what let Step
        # Functions react to an upload without a Lambda in between.
        # ---------------------------------------------------------------
        tape_bucket = s3.Bucket(
            self,
            "LoanTapeBucket",
            bucket_name=None,  # let CDK generate a unique name per environment
            versioned=True,
            event_bridge_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                # Raw exports are the system of record for what a lender
                # sent us. Keep them hot for 90 days for reprocessing and
                # debugging, then move to Glacier. They are never deleted:
                # a loan tape is a financial record.
                s3.LifecycleRule(
                    id="ArchiveRawUploadsAfter90Days",
                    prefix=RAW_PREFIX,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90),
                        )
                    ],
                ),
                s3.LifecycleRule(
                    id="ExpireOldVersions",
                    noncurrent_version_transitions=[
                        s3.NoncurrentVersionTransition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(30),
                        )
                    ],
                ),
            ],
        )

        # ---------------------------------------------------------------
        # Notifications: one topic, subscribed by whoever owns onboarding
        # operations. Step Functions publishes to it on both success and
        # failure so a stuck lender onboarding is never silent.
        # ---------------------------------------------------------------
        notifications_topic = sns.Topic(
            self,
            "OnboardingNotifications",
            topic_name="loan-tape-onboarding-notifications",
            display_name="Loan tape onboarding: completions and failures",
        )

        # Dead-letter queue for files that fail validation or processing
        # after retries are exhausted. Holds a small JSON pointer to the
        # unprocessable object, not the file itself, so an operator can
        # inspect and re-drive it by hand.
        unprocessable_dlq = sqs.Queue(
            self,
            "UnprocessableTapesDlq",
            queue_name="loan-tape-unprocessable-dlq",
            retention_period=Duration.days(14),
            encryption=sqs.QueueEncryption.SQS_MANAGED,
        )

        # ---------------------------------------------------------------
        # Secrets: the persist step's only credential. Exaloan's ArangoDB
        # cluster already exists; this stack references, not creates, it.
        # ---------------------------------------------------------------
        arango_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "ArangoCredentials", secret_name="exaloan/arangodb/onboarding-writer"
        )

        # ---------------------------------------------------------------
        # Step 1: validate. A fast Lambda gate so a malformed export fails
        # in seconds, before a Fargate task ever spins up.
        # ---------------------------------------------------------------
        validate_fn = _lambda.Function(
            self,
            "ValidateTapeFunction",
            function_name="loan-tape-validate",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=_lambda.Code.from_asset("lambda/validate"),
            timeout=Duration.seconds(30),
            memory_size=256,
            log_group=logs.LogGroup(
                self,
                "ValidateTapeLogGroup",
                retention=logs.RetentionDays.ONE_MONTH,
                removal_policy=RemovalPolicy.DESTROY,
            ),
        )
        tape_bucket.grant_read(validate_fn, f"{RAW_PREFIX}*")

        # ---------------------------------------------------------------
        # Step 2: run the Part 1 pipeline in Fargate. Chosen over a second
        # Lambda because the detection run (pandas, openpyxl, XIRR solving
        # over the full tape) is memory- and CPU-heavier and longer-running
        # than fits comfortably in Lambda's limits once tapes reach
        # thousands of rows, and it needs no idle capacity between runs.
        # ---------------------------------------------------------------
        # Explicit VPC with fixed availability zones. ecs.Cluster would
        # otherwise create a default VPC that looks up AZs for the target
        # account/region, which requires real AWS credentials at synth
        # time. Naming the AZs directly keeps `cdk synth` fully offline.
        # No NAT gateway: this is a representative sample, and the Fargate
        # task only needs outbound access to S3 and ArangoDB, so a public
        # subnet with an internet gateway is enough. Production would use
        # private subnets with a NAT gateway or VPC endpoints for S3.
        vpc = ec2.Vpc(
            self,
            "PipelineVpc",
            ip_addresses=ec2.IpAddresses.cidr("10.0.0.0/16"),
            availability_zones=["eu-central-1a", "eu-central-1b"],
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                )
            ],
        )
        cluster = ecs.Cluster(
            self, "PipelineCluster", cluster_name="loan-pipeline-cluster", vpc=vpc
        )

        task_definition = ecs.FargateTaskDefinition(
            self,
            "PipelineTaskDefinition",
            cpu=1024,
            memory_limit_mib=2048,
        )
        pipeline_repository = ecr.Repository.from_repository_name(
            self, "PipelineRepository", PIPELINE_ECR_REPOSITORY_NAME
        )
        container = task_definition.add_container(
            "PipelineContainer",
            image=ecs.ContainerImage.from_ecr_repository(
                pipeline_repository, PIPELINE_IMAGE_TAG
            ),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="loan-pipeline",
                log_retention=logs.RetentionDays.ONE_MONTH,
            ),
            environment={
                "OUTPUT_PREFIX": PROCESSED_PREFIX,
                "INPUT_BUCKET": tape_bucket.bucket_name,
                "OUTPUT_BUCKET": tape_bucket.bucket_name,
            },
        )
        tape_bucket.grant_read(task_definition.task_role, f"{RAW_PREFIX}*")
        tape_bucket.grant_write(task_definition.task_role, f"{PROCESSED_PREFIX}*")

        # ---------------------------------------------------------------
        # Step 3: persist to ArangoDB.
        # ---------------------------------------------------------------
        persist_fn = _lambda.Function(
            self,
            "PersistResultsFunction",
            function_name="loan-tape-persist",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=_lambda.Code.from_asset("lambda/persist"),
            timeout=Duration.minutes(2),
            memory_size=512,
            log_group=logs.LogGroup(
                self,
                "PersistResultsLogGroup",
                retention=logs.RetentionDays.ONE_MONTH,
                removal_policy=RemovalPolicy.DESTROY,
            ),
            environment={"ARANGO_SECRET_ARN": arango_secret.secret_arn},
        )
        tape_bucket.grant_read(persist_fn, f"{PROCESSED_PREFIX}*")
        arango_secret.grant_read(persist_fn)

        # ---------------------------------------------------------------
        # State machine: validate -> run pipeline -> persist, each with
        # retry-with-backoff, all three sharing one failure path that
        # notifies SNS and records the object in the DLQ.
        # ---------------------------------------------------------------
        notify_failure = tasks.SnsPublish(
            self,
            "NotifyFailure",
            topic=notifications_topic,
            message=sfn.TaskInput.from_json_path_at("$"),
            subject="Loan tape onboarding failed",
        )
        send_to_dlq = tasks.SqsSendMessage(
            self,
            "RecordUnprocessableTape",
            queue=unprocessable_dlq,
            message_body=sfn.TaskInput.from_json_path_at("$"),
        )
        failure_chain = send_to_dlq.next(notify_failure).next(
            sfn.Fail(self, "OnboardingFailed", cause="Loan tape could not be processed")
        )

        def _with_retry_and_catch(task: sfn.IChainable) -> sfn.IChainable:
            if isinstance(task, sfn.TaskStateBase):
                task.add_retry(
                    errors=["States.ALL"],
                    interval=Duration.seconds(15),
                    max_attempts=3,
                    backoff_rate=2.0,
                )
                task.add_catch(failure_chain, errors=["States.ALL"], result_path="$.error")
            return task

        validate_task = tasks.LambdaInvoke(
            self,
            "ValidateTape",
            lambda_function=validate_fn,
            payload_response_only=True,
            result_path="$.validation",
        )
        _with_retry_and_catch(validate_task)

        run_pipeline_task = tasks.EcsRunTask(
            self,
            "RunLoanPipeline",
            cluster=cluster,
            task_definition=task_definition,
            launch_target=tasks.EcsFargateLaunchTarget(
                platform_version=ecs.FargatePlatformVersion.LATEST
            ),
            # Public subnet, no NAT gateway (see the VPC comment above), so
            # the task needs a public IP to reach ECR, S3, and ArangoDB.
            assign_public_ip=True,
            subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            container_overrides=[
                tasks.ContainerOverride(
                    container_definition=container,
                    environment=[
                        tasks.TaskEnvironmentVariable(
                            name="INPUT_KEY", value=sfn.JsonPath.string_at("$.validation.key")
                        ),
                        tasks.TaskEnvironmentVariable(
                            name="LENDER_ID",
                            value=sfn.JsonPath.string_at("$.validation.lender_id"),
                        ),
                        tasks.TaskEnvironmentVariable(
                            name="RUN_ID", value=sfn.JsonPath.string_at("$$.Execution.Name")
                        ),
                    ],
                )
            ],
            result_path="$.pipeline_run",
        )
        _with_retry_and_catch(run_pipeline_task)

        # EcsRunTask does not surface the container's application output
        # back into the state (that needs the run-a-job / task-token
        # callback pattern, a further iteration beyond this representative
        # sample). Instead, the state machine derives the report location
        # itself from the same {lender_id}/{execution name} convention the
        # container writes to (see docker/run_pipeline.py), so persist does
        # not depend on anything the Fargate step returns.
        persist_task = tasks.LambdaInvoke(
            self,
            "PersistResults",
            lambda_function=persist_fn,
            payload=sfn.TaskInput.from_object(
                {
                    "bucket": tape_bucket.bucket_name,
                    "key": sfn.JsonPath.format(
                        f"{PROCESSED_PREFIX}{{}}/{{}}/report.json",
                        sfn.JsonPath.string_at("$.validation.lender_id"),
                        sfn.JsonPath.string_at("$$.Execution.Name"),
                    ),
                    "lender_id": sfn.JsonPath.string_at("$.validation.lender_id"),
                    "run_id": sfn.JsonPath.string_at("$$.Execution.Name"),
                }
            ),
            payload_response_only=True,
            result_path="$.persisted",
        )
        _with_retry_and_catch(persist_task)

        notify_success = tasks.SnsPublish(
            self,
            "NotifyCompletion",
            topic=notifications_topic,
            message=sfn.TaskInput.from_json_path_at("$"),
            subject="Loan tape onboarding completed",
        )

        definition = validate_task.next(run_pipeline_task).next(persist_task).next(notify_success)

        state_machine_log_group = logs.LogGroup(
            self,
            "StateMachineLogGroup",
            log_group_name="/aws/vendedlogs/states/loan-tape-onboarding",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )
        state_machine = sfn.StateMachine(
            self,
            "OnboardingStateMachine",
            state_machine_name="loan-tape-onboarding",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=Duration.minutes(30),
            logs=sfn.LogOptions(
                destination=state_machine_log_group, level=sfn.LogLevel.ALL
            ),
        )

        # ---------------------------------------------------------------
        # Trigger: S3 event -> EventBridge -> Step Functions. No Lambda
        # glue code is needed for the trigger itself; only object-created
        # events under raw-uploads/ start an execution.
        # ---------------------------------------------------------------
        upload_rule = events.Rule(
            self,
            "RawUploadRule",
            rule_name="loan-tape-raw-upload",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {"name": [tape_bucket.bucket_name]},
                    "object": {"key": [{"prefix": RAW_PREFIX}]},
                },
            ),
        )
        upload_rule.add_target(
            targets.SfnStateMachine(
                state_machine,
                role=iam.Role(
                    self,
                    "EventBridgeToStepFunctionsRole",
                    assumed_by=iam.ServicePrincipal("events.amazonaws.com"),
                ),
            )
        )
