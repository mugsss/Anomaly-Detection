"""Command-line entry point: `python -m src [tape.xlsx]`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import run_pipeline
from .reporter import format_console_summary, write_csv, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Detect data-quality anomalies in a loan tape.",
    )
    parser.add_argument("tape", nargs="?", default="loans.xlsx", help="path to the Excel loan tape")
    parser.add_argument("-o", "--output", default="output/report.csv", help="CSV report path")
    parser.add_argument("--json", dest="json_output", default=None, help="also write a JSON report")
    parser.add_argument("--sheet", default=0, help="worksheet name or index")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--quiet", action="store_true", help="suppress the console summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    tape = Path(args.tape)
    if not tape.exists():
        logging.error("Loan tape not found: %s", tape)
        return 2

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    outcome = run_pipeline(tape, sheet=sheet)

    write_csv(outcome, args.output)
    if args.json_output:
        write_json(outcome, args.json_output)

    if not args.quiet:
        print(format_console_summary(outcome))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
