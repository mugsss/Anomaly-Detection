"""Architecture diagram source for the loan-tape onboarding pipeline (Part 2).

Diagrams-as-code (the `diagrams` package, https://diagrams.mingrammer.com),
so the picture is generated from and stays in sync with this file rather
than drifting from a hand-drawn image. Run:

    pip install diagrams
    # requires the Graphviz `dot` binary on PATH (brew install graphviz,
    # or conda install -c conda-forge graphviz)
    python infra/diagram.py

to regenerate infra/diagram.png.

This mirrors the representative CDK stack in infra/cdk/: one S3 bucket
namespaced by prefix and by lender, a Step Functions state machine with
three steps, and the shared monitoring/notification/failure-handling
surface around it.
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import ECS, Fargate, Lambda
from diagrams.aws.integration import SF, SNS, SQS
from diagrams.aws.management import Cloudwatch, CloudwatchAlarm
from diagrams.aws.security import SecretsManager
from diagrams.aws.storage import S3, S3Glacier
from diagrams.generic.database import SQL as GenericDatabase
from PIL import Image, ImageDraw, ImageFont

GRAPH_ATTR = {
    "fontsize": "22",
    "fontname": "Helvetica-Bold",
    "labelloc": "t",
    "pad": "0.5",
    "splines": "spline",
    "nodesep": "0.7",
    "ranksep": "1.0",
    "concentrate": "false",
}
NODE_ATTR = {"fontsize": "12"}

# Short annotation notes rendered as a text footer below the diagram, rather
# than as graphviz clusters. An empty cluster placed next to a tall,
# multi-rank cluster (the state machine) gets stretched by graphviz to match
# that rank's height, which produced an oddly tall, mostly-blank box; a
# plain text footer keeps the notes readable without fighting graphviz's
# layout engine for something that is not really a node in the flow.
NOTES = [
    (
        "Failure handling",
        "Each Step Functions task retries 3x with exponential backoff (15s base). "
        "After retries are exhausted, the failed object's pointer is written to an "
        "SQS dead-letter queue and SNS notifies operators, so a bad file is never "
        "silently dropped.",
    ),
    (
        "Monitoring",
        "A CloudWatch dashboard tracks loans processed, flag rate, error rate, and Fargate task "
        "duration per run. A CloudWatch alarm on state-machine failures feeds the same SNS topic "
        "used for success notifications.",
    ),
    (
        "Cost",
        "Fargate bills per second with no idle capacity between onboarding runs. The S3 lifecycle "
        "rule transitions raw uploads to Glacier after 90 days. Each Lambda invocation (validate, "
        "persist) costs a fraction of a cent.",
    ),
]

with Diagram(
    "Exaloan Loan-Tape Onboarding Pipeline",
    filename="infra/diagram",
    show=False,
    direction="LR",
    graph_attr=GRAPH_ATTR,
    node_attr=NODE_ATTR,
    outformat="png",
):
    with Cluster("Ingestion  (per-lender namespacing: <prefix>/{lender_id}/...)"):
        raw = S3("raw-uploads/\n(hot, versioned)")
        archive = S3Glacier("archive/\n(Glacier after 90d)")
        raw >> Edge(style="dashed", label="lifecycle\n90 days") >> archive

    with Cluster("Onboarding State Machine (Step Functions)"):
        state_machine = SF("Standard workflow\nretry w/ backoff\nper step")

        with Cluster("Step 1: Validate"):
            validate = Lambda("Validate\nformat + columns")

        with Cluster("Step 2: Run Pipeline"):
            cluster_icon = ECS("Fargate cluster")
            pipeline_task = Fargate("Part 1 pipeline\n(detect anomalies)")
            cluster_icon >> pipeline_task

        with Cluster("Step 3: Persist"):
            persist = Lambda("Persist results")

        state_machine >> validate >> pipeline_task >> persist

    processed = S3("processed/\n(reports)")
    pipeline_task >> Edge(label="report.json") >> processed
    processed >> Edge(style="dashed", label="read") >> persist

    with Cluster("Results store"):
        arango_secret = SecretsManager("ArangoDB\ncredentials")
        arango = GenericDatabase("ArangoDB\n(existing cluster)")
        arango_secret >> Edge(label="GetSecretValue") >> persist
        persist >> Edge(label="upsert per loan") >> arango

    with Cluster("Notifications & DLQ"):
        dlq = SQS("Unprocessable\ntapes DLQ")
        notifications = SNS("Onboarding\nnotifications")
        dlq >> Edge(color="firebrick") >> notifications

    raw >> Edge(
        label="ObjectCreated\n(EventBridge)", color="darkgreen", style="bold"
    ) >> state_machine
    validate - Edge(style="dotted", color="firebrick", label="on failure, after retries") - dlq
    pipeline_task - Edge(style="dotted", color="firebrick") - dlq
    persist - Edge(style="dotted", color="firebrick") - dlq
    persist >> Edge(color="darkgreen", label="on success") >> notifications

    with Cluster("Monitoring"):
        dashboard = Cloudwatch(
            "Dashboard:\nloans processed,\nflag rate, error rate,\nFargate duration"
        )
        alarm = CloudwatchAlarm("Alarm on\nstate-machine failure")
        dashboard >> alarm

    validate >> Edge(style="dashed", color="gray50") >> dashboard
    pipeline_task >> Edge(style="dashed", color="gray50") >> dashboard
    persist >> Edge(style="dashed", color="gray50") >> dashboard
    alarm >> Edge(color="firebrick") >> notifications


def _add_notes_footer(png_path: str) -> None:
    """Composite the short annotation notes below the rendered diagram."""
    diagram = Image.open(png_path).convert("RGB")

    margin, line_gap, section_gap = 40, 6, 22
    heading_font, body_font = _load_fonts()
    body_width_chars = 150

    footer_lines: list[tuple[str, ImageFont.FreeTypeFont]] = []
    for heading, body in NOTES:
        footer_lines.append((heading, heading_font))
        for wrapped in _wrap(body, body_width_chars):
            footer_lines.append((wrapped, body_font))
        footer_lines.append(("", body_font))  # spacing between sections

    line_height = body_font.size + line_gap
    footer_height = margin * 2 + line_height * len(footer_lines) + section_gap

    canvas = Image.new("RGB", (diagram.width, diagram.height + footer_height), "white")
    canvas.paste(diagram, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.line(
        [(margin, diagram.height + 10), (diagram.width - margin, diagram.height + 10)],
        fill="lightgray",
        width=2,
    )

    y = diagram.height + section_gap
    for text, font in footer_lines:
        draw.text((margin, y), text, fill="black", font=font)
        y += line_height

    canvas.save(png_path)


def _load_fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, 18, index=1), ImageFont.truetype(path, 15)
        except OSError:
            continue
    return ImageFont.load_default(), ImageFont.load_default()


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


_add_notes_footer("infra/diagram.png")
