"""
Command line entry point for PeetsFEA.

Subcommands:

- litzpcb: run the legacy EVDD_litz_PCB_v2 NSGA-II wrapper
- pcbpcb: run the PCB-on-PCB NSGA-II example pipeline with default settings
- ansys: run the EVDD_PCB_PCB AEDT simulation using the built-in defaults

    python -m PeetsFEA litzpcb [--scratch-root PATH] [--cleanup] [--show-plot]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .peetspareto.litzpcb import run_litzpcb_nsga2
from .peetspareto.pcbpcb.config import ModelArtifactSelectionError, ParetoRunConfig
from .peetspareto.pcbpcb.runtime import default_objectives, run_pcbpcb_nsga2


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

    pcbpcb_parser = subparsers.add_parser(
        "pcbpcb",
        help="Run the PCB-on-PCB NSGA-II optimizer with notebook-equivalent defaults.",
    )
    pcbpcb_parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Directory for Pareto CSV/provenance artifacts (default: data/pareto).",
    )
    pcbpcb_parser.set_defaults(func=_run_pcbpcb_command)

    ansys_parser = subparsers.add_parser(
        "ansys",
        help="Run the EVDD_PCB_PCB AEDT simulation using the bundled defaults or random parameters.",
    )
    ansys_parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the built-in sample parameters to run (0-based, ignored when --random is set).",
    )
    ansys_parser.add_argument(
        "--random",
        action="store_true",
        help="Use the simulation's internal random parameter generation instead of the bundled samples.",
    )
    ansys_parser.set_defaults(func=_run_ansys_command)

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


def _run_pcbpcb_command(args: argparse.Namespace) -> int:
    export_dir = args.export_dir.expanduser() if args.export_dir else None
    try:
        config = _build_pcbpcb_config(export_dir)
        result = run_pcbpcb_nsga2(config=config)
    except ModelArtifactSelectionError as exc:
        print(f"[pcbpcb] {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[pcbpcb] missing LightGBM artifacts: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - safety net for CLI usage
        print(f"[pcbpcb] failed: {exc}", file=sys.stderr)
        return 1

    csv_path = result.config.export.csv_path().resolve()
    prov_path = result.config.export.provenance_path().resolve()
    print(csv_path)
    if prov_path.exists():
        print(prov_path, file=sys.stderr)
    return 0


def _build_pcbpcb_config(export_dir: Path | None) -> ParetoRunConfig:
    config = ParetoRunConfig(objectives=default_objectives())
    if export_dir is None:
        return config
    export_config = replace(config.export, directory=export_dir)
    return replace(config, export=export_config)


def _run_ansys_command(args: argparse.Namespace) -> int:
    try:
        from .fea.EVDD_PCB_PCB import SimulationInputParameters, run_simulation
    except Exception as exc:  # pragma: no cover - AEDT import heavy
        print(f"[ansys] failed to import EVDD_PCB_PCB module: {exc}", file=sys.stderr)
        return 2

    if args.random:
        selected = None
        use_random = True
    else:
        samples = _default_sim_parameters(SimulationInputParameters)
        if args.index < 0 or args.index >= len(samples):
            print(f"[ansys] index {args.index} out of range (0-{len(samples)-1})", file=sys.stderr)
            return 2
        selected = samples[args.index]
        use_random = False

    try:
        project_dir = run_simulation(run_simulation=True, use_random=use_random, input_parameters=selected)
    except Exception as exc:  # pragma: no cover - AEDT runtime
        print(f"[ansys] simulation failed: {exc}", file=sys.stderr)
        return 1

    if project_dir:
        print(project_dir)
    else:
        print("[ansys] simulation completed (project directory not returned).", file=sys.stderr)
    return 0


def _default_sim_parameters(cls):
    return [
        cls(
            freq=310.0,
            input_voltage=200.0,
            w1=32.0,
            l1_leg=13.8,
            l1_top=0.8,
            l2=23.7,
            h1=2.52,
            l1_center=6.0,
            Tx_turns=3.0,
            Tx_width=1.9,
            Tx_height=0.175,
            Tx_space_x=3.5,
            Tx_space_y=1.9,
            Tx_preg=0.19,
            Rx_width=9.6,
            Rx_height=0.035,
            Rx_space_x=2.4,
            Rx_space_y=4.2,
            Rx_preg=0.26,
            g2=1.75,
            Tx_layer_space_x=0.5,
            Tx_layer_space_y=2.1,
            Tx_current=5.0,
            Rx_current=6,
        ),
        cls(
            freq=310.0,
            input_voltage=200.0,
            w1=32.0,
            l1_leg=13.8,
            l1_top=0.8,
            l2=23.7,
            h1=2.52,
            l1_center=6.0,
            Tx_turns=3.0,
            Tx_width=1.9,
            Tx_height=0.175,
            Tx_space_x=3.5,
            Tx_space_y=1.9,
            Tx_preg=0.19,
            Rx_width=9.6,
            Rx_height=0.035,
            Rx_space_x=2.4,
            Rx_space_y=4.2,
            Rx_preg=0.26,
            g2=1.75,
            Tx_layer_space_x=0.5,
            Tx_layer_space_y=2.1,
            Tx_current=5.0,
            Rx_current=12,
        ),
        cls(
            freq=310.0,
            input_voltage=200.0,
            w1=32.0,
            l1_leg=13.8,
            l1_top=0.8,
            l2=23.7,
            h1=2.52,
            l1_center=6.0,
            Tx_turns=3.0,
            Tx_width=1.9,
            Tx_height=0.175,
            Tx_space_x=3.5,
            Tx_space_y=1.9,
            Tx_preg=0.19,
            Rx_width=9.6,
            Rx_height=0.035,
            Rx_space_x=2.4,
            Rx_space_y=4.2,
            Rx_preg=0.26,
            g2=1.75,
            Tx_layer_space_x=0.5,
            Tx_layer_space_y=2.1,
            Tx_current=5.0,
            Rx_current=18,
        ),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
