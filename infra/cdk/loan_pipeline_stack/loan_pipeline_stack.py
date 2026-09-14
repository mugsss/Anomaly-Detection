"""Representative CDK stack for the loan-tape onboarding pipeline.

A representative sample, not full coverage: one ingestion trigger (EventBridge),
one storage resource (S3), and one compute resource (Fargate) running the Part 1
pipeline, per the assignment's scope.
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
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct

PIPELINE_ECR_REPOSITORY_NAME = "loan-pipeline"
PIPELINE_IMAGE_TAG = "latest"

RAW_PREFIX = "raw-uploads/"
PROCESSED_PREFIX = "processed/"


class LoanPipelineStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------------------------------------------------------------
        # Storage: one versioned S3 bucket with lifecycle rules.
        # Per-lender namespacing via prefixes: raw-uploads/{lender_id}/,
        # processed/{lender_id}/{run_id}/report.json.
        # ---------------------------------------------------------------
        tape_bucket = s3.Bucket(
            self,
            "LoanTapeBucket",
            versioned=True,
            event_bridge_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
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
            ],
        )

        # ---------------------------------------------------------------
        # Compute: ECS Fargate task running the Part 1 pipeline image.
        # Chosen over Lambda because detection (pandas, openpyxl, XIRR)
        # is memory- and CPU-heavier than Lambda's limits once tapes
        # reach thousands of rows.
        # ---------------------------------------------------------------
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
                "INPUT_BUCKET": tape_bucket.bucket_name,
                "OUTPUT_BUCKET": tape_bucket.bucket_name,
                "OUTPUT_PREFIX": PROCESSED_PREFIX,
            },
        )
        tape_bucket.grant_read(task_definition.task_role, f"{RAW_PREFIX}*")
        tape_bucket.grant_write(task_definition.task_role, f"{PROCESSED_PREFIX}*")

        # ---------------------------------------------------------------
        # State machine: a single-step workflow that runs the Fargate task.
        # Step Functions provides retry-with-backoff and execution history
        # out of the box.
        # ---------------------------------------------------------------
        run_pipeline_task = tasks.EcsRunTask(
            self,
            "RunLoanPipeline",
            cluster=cluster,
            task_definition=task_definition,
            launch_target=tasks.EcsFargateLaunchTarget(
                platform_version=ecs.FargatePlatformVersion.LATEST
            ),
            assign_public_ip=True,
            subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            container_overrides=[
                tasks.ContainerOverride(
                    container_definition=container,
                    environment=[
                        tasks.TaskEnvironmentVariable(
                            name="INPUT_KEY", value=sfn.JsonPath.string_at("$.detail.object.key")
                        ),
                        tasks.TaskEnvironmentVariable(
                            name="RUN_ID", value=sfn.JsonPath.string_at("$$.Execution.Name")
                        ),
                    ],
                )
            ],
        )
        run_pipeline_task.add_retry(
            errors=["States.ALL"],
            interval=Duration.seconds(15),
            max_attempts=3,
            backoff_rate=2.0,
        )

        state_machine = sfn.StateMachine(
            self,
            "OnboardingStateMachine",
            state_machine_name="loan-tape-onboarding",
            definition_body=sfn.DefinitionBody.from_chainable(run_pipeline_task),
            timeout=Duration.minutes(30),
        )

        # ---------------------------------------------------------------
        # Trigger: S3 ObjectCreated -> EventBridge -> Step Functions.
        # Only files under raw-uploads/ start an execution.
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
