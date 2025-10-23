# DEV_PLAN

## Project Goals
- Deliver a Python package `PeetsFEA` exposing `peetspareto` and `peetsansys` modules.
- Bundle a pretrained LightGBM `.pkl` to generate transformer performance predictions for multi-objective optimization.
- Integrate `pyaedt==0.21.2` to simulate individual transformer designs.

## Milestone 1 — Repository Bootstrap
- Create `pyproject.toml` (PEP 621) with core metadata, Python version floor (>=3.9), and dependency pins (`pyaedt==0.21.2`, `numpy`, `pandas`, `joblib`, `pymoo` or alternative MOO library).
- Scaffold package layout:
  - `PeetsFEA/__init__.py`
  - `PeetsFEA/peetspareto/__init__.py` and functional modules
  - `PeetsFEA/peetsansys/__init__.py` and AEDT helpers
  - `PeetsFEA/data/` for the LightGBM artifact and schema definitions.
- Add base README, contribution guide, and packaging scripts (build, publish placeholders).

## Milestone 2 — Data Assets & Schema Contracts
- Finalize parameter schema derived from `tmp/characteristics_defaults.toml`; codify conversions to the model’s expected input features.
- Store the LightGBM `.pkl` artifact under `PeetsFEA/data/model/transformer_predictor.pkl` with loading utilities.
- Document provenance: training dataset summary, target metrics, and constraints for model updates.
- Add validation routines ensuring TOML input matches model feature ordering and data types.

## Milestone 3 — `peetspareto` Optimization Workflow
- Implement parameter expansion: sample/expand candidate transformers from seed defaults, respecting physical constraints.
- Run batched model inference to score candidates; include vectorized inference and optional multiprocessing.
- Integrate NSGA-II (e.g., via `pymoo.algorithms.nsga2.NSGA2`) to compute Pareto-optimal fronts over objectives (efficiency, losses, cost proxies, etc.).
- Export Pareto front results to CSV with full parameter payload and objective summaries.
- Provide a public API:
  - `generate_pareto_front(config_path: Path | str, output_csv: Path) -> None`
  - Lower-level utilities for interactive use.
- Cover with unit tests (schema conversion, inference) and functional tests (small optimization run).

## Milestone 4 — `peetsansys` Simulation Pipeline
- Build adapters translating a single transformer parameter dict into AEDT model construction steps.
- Encapsulate AEDT session management (launch, project setup, design cleanup) with context managers.
- Implement core workflows:
  - Geometry/material assignment
  - Excitation setup matching the electrical specs
  - Solver execution and result extraction (EM fields, thermal behavior, KPI metrics).
- Provide deterministic output data structures and optional exports (CSV/JSON).
- Add smoke tests gated behind environment variables to avoid requiring AEDT during routine CI runs.

## Milestone 5 — Integration, QA, and Delivery
- Create shared utilities (logging, error handling, configuration loading).
- Write documentation (API reference, end-to-end tutorial, troubleshooting for AEDT).
- Configure automated tests (pytest) and pre-commit hooks (formatting, linting).
- Define release checklist: artifact verification, changelog update, tag creation, package publish to an internal index.

## Open Questions / Follow-ups
- Confirm how the LightGBM features map to raw transformer parameters (need feature list and preprocessing steps).
- Decide on additional objectives/constraints (thermal performance, cost) and their normalization.
- Determine licensing and distribution strategy for the bundled `.pkl` and AEDT integration scripts.
