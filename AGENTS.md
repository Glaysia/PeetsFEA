# AGENTS

## Purpose
PeetsFEA delivers two coordinated capabilities: (1) data-driven transformer optimization using a pretrained LightGBM model and multi-objective search, and (2) Ansys Electronics Desktop simulation pipelines via `pyaedt==0.22.0`. The agents below align to these responsibilities so development remains focused and auditable.

## Roles

### Product Lead & Domain Specialist
- Owns transformer design requirements and validates assumptions in `tmp/characteristics_defaults.toml`.
- Defines optimization objectives, constraints, and acceptance criteria for both ML-driven and simulated outputs.

### Optimization Engineer (`peetspareto`)
- Packages the LightGBM `.pkl` artifact and exposes a clean API for bulk inference.
- Designs the NSGA-II (or comparable) pipeline that expands parameter grids, evaluates models, and emits Pareto-front CSVs.
- Ensures outputs carry forward provenance (input parameters, prediction metadata, and objective scores).

### Simulation Integration Engineer (`peetsansys`)
- Wraps `pyaedt==0.22.0` workflows to build and solve transformer models from sanitized parameter dictionaries.
- Implements result extraction helpers (field data, loss metrics, thermal results) that downstream consumers can serialize.
- Maintains compatibility with AEDT desktop/Automation environments and documents prerequisites.

### Data & Tooling Engineer
- Manages dependency pinning, packaging metadata, and artifact storage (pkl, templates, calibration data).
- Builds internal validators to ensure parameter translations stay consistent across the ML and simulation layers.

### QA & Release Engineer
- Adds regression and validation tests (unit, smoke, integration) covering both `peetspareto` and `peetsansys`.
- Oversees CI/CD automation, release notes, and verifiable version bumps.

## Collaboration Notes
- Keep shared schemas in sync: transformers' parameter definitions must be mirrored in code, docs, and artifacts.
- Favor reproducible pipelines (seed control, deterministic sampling) when generating candidate populations.
- Record environment details (AEDT version, OS, Python version) so simulations can be rerun without drift.
- Before closing any work item, log the outcome in the `PEETSPARETO_PCBPCB_PLAN.md` 결과보고 section so parallel agents can track progress.

## DO NOT EDIT codes in legacy_codes
