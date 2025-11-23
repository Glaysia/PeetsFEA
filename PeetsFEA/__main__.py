"""
Command line entry point for PeetsFEA.

Currently supports running the legacy litzpcb NSGA-II optimizer via:

    python -m PeetsFEA litzpcb [--scratch-root PATH] [--cleanup] [--show-plot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .peetspareto.litzpcb import run_litzpcb_nsga2


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="PeetsFEA", description="PeetsFEA command line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    litzpcb_parser = subparsers.add_parser(
        "litzpcb", help="Run the legacy litzpcb NSGA-II optimizer and print the Pareto CSV path."
    )
    litzpcb_parser.add_argument(
        "--scratch-root",
        type=Path,
        default=None,
        help="Root directory for Pareto scratch data (default: ~/.peetsfea/pareto).",
    )
    litzpcb_parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="Remove the scratch directory after the run completes.",
    )
    litzpcb_parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the Pareto scatter plot when the run finishes.",
    )
    litzpcb_parser.set_defaults(func=_run_litzpcb_command)

    return parser.parse_args(argv)


def _run_litzpcb_command(args: argparse.Namespace) -> int:
    scratch_root = args.scratch_root.expanduser().resolve() if args.scratch_root else None
    try:
        result = run_litzpcb_nsga2(
            show_plot=args.show_plot,
            scratch_root=scratch_root,
            cleanup=args.cleanup,
        )
    except FileNotFoundError as exc:
        print(f"[litzpcb] missing LightGBM artifacts: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - safety net for CLI usage
        print(f"[litzpcb] failed: {exc}", file=sys.stderr)
        return 1

    pareto_path = result.pareto_path.resolve()
    print(pareto_path)
    if args.cleanup and not pareto_path.exists():
        print("[litzpcb] cleanup enabled; Pareto scratch directory was removed.", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
