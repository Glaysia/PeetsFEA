# peetspareto.pcbpcb Collaboration Plan

## Scope & Intent
- Rebuild `legacy_codes/EVDD_PCB_PCB/NSGA-II.ipynb` as a production-ready module under `PeetsFEA/peetspareto/pcbpcb`.
- Ensure every public API, data class, and optimizer component is fully type-hinted, documented, and regression-testable.
- Preserve notebook logic (LightGBM inference + NSGA-II multi-objective search) while adding provenance tracking, deterministic runs, and clean separation of concerns.
- Provide a zero-config `run_pcbpcb_nsga2()` entry point that mirrors the notebook’s behavior without requiring external settings files.
- Hold off on creating git commits for this effort until the team agrees the refactor is ready to land.

## Target Module Layout (proposed)
- `PeetsFEA/peetspareto/pcbpcb/__init__.py` — re-exports high-level entry points.
- `config.py` — runtime configuration loaders, environment knobs, and validation helpers.
- `schemas.py` — typed parameter/story schemas (pydantic or dataclasses) mirroring transformer characteristics and model inputs.
- `model.py` — LightGBM artifact loader, bulk inference service, prediction metadata handling.
- `optimizer.py` — NSGA-II (or equivalent) pipeline, candidate evaluators, constraint hooks, RNG controls.
- `io.py` — Pareto CSV/Parquet writers, provenance bundles, logging utilities.
> Keep `legacy_codes/*` untouched; all new work lives under the package tree.

## Shared Conventions
- Stick to Python ≥3.9, type hints everywhere, and mypy-friendly constructs.
- Seed all randomness (NumPy, random, LightGBM, NSGA-II library) to maintain reproducibility.
- Ship the notebook’s `vars1` defaults directly in code so the optimizer runs out-of-the-box; treat TOML/Product-Lead overrides as a later enhancement and call out any schema drift once those inputs exist.
- Every result artifact (CSV, JSON) must embed config hashes, artifact versions, and environment snippets (OS, Python, LightGBM).
- Avoid hidden state: prefer pure functions or explicit service objects; document any caching (e.g., model singleton).

## Workstreams & Suggested Owners
### Agent A — Schema & Data Contracts
- Extract parameter definitions from the notebook and domain notes, then formalize them in `schemas.py`.
- Build validators ensuring inputs cover numeric ranges, enum domains, and derived constraints (e.g., PCB copper thickness vs layer count).
- Codify the notebook’s `vars1` bounds/scales as first-class constants and supply fixtures so QA can run without external config; TOML/override plumbing can follow later.

### Agent B — Model & Inference Service
- Package the LightGBM `.pkl` and expose `PCBPCBModel` with `predict_batch(parameters: Sequence[TransformerSpec]) -> list[Prediction]`.
- Ensure feature ordering, scaling, and categorical encodings match the legacy notebook; add unit tests comparing against sampled notebook outputs.
- Emit prediction metadata (timestamp, model version, checksum) for downstream provenance use.

### Agent C — Optimization & Results
- Recreate the NSGA-II workflow: population seeding from validated specs, fitness evaluation via the model service, Pareto front extraction.
- Make evaluation pluggable so future simulation scores (from `peetsansys`) can be injected without refactors.
- Implement exporters writing Pareto fronts + run context to `data/pareto/*.csv` (path overridable), and surface a high-level `generate_pareto_front(...)` API.
- Add smoke/integration tests that run a trimmed population (e.g., 8–16 candidates, 2 generations) to keep CI fast.

> Agents should coordinate on shared utilities (logging, enums) before merging to avoid duplicated helpers.

## Execution Sequence
1. **Notebook Traceability** — catalog every step from the legacy notebook; record any assumptions or hard-coded values in this file or design docs.
2. **Schema Finalization** — Agent A delivers typed specs + validators backed by baked-in defaults so Agents B/C can execute the pipeline with zero extra inputs.
3. **Model Service** — Agent B lands artifact loader & prediction API, along with fixtures for deterministic comparisons.
4. **Optimizer & IO** — Agent C layers NSGA-II orchestration plus result writers, wiring in Agent B’s service and Agent A’s schema.
5. **Integration Testing** — run a shared smoke test harness; log environment data (Python/pymoo/LightGBM versions) per Collaboration Notes in `AGENTS.md`.
6. **Documentation & Handoff** — add module README/docstrings, update `AGENTS.md` or other docs with new touchpoints, ensure QA owns regression coverage.

## Deliverables Checklist
- ✅ Typed schema module with validation and regression tests.
- ✅ LightGBM service class with cached artifact loading, deterministic batch inference, and metadata-rich outputs.
- ✅ NSGA-II runner with a documented API entry point, seeded RNG, progress logging, and Pareto CSV export.
- ✅ Provenance report template (JSON/YAML) capturing config hashes, artifact versions, and runtime environment.
- ✅ Unit + smoke tests covering schema parsing, inference parity, optimizer loop, and IO.
- ☐ `run_pcbpcb_nsga2()` convenience wrapper that executes with notebook-equivalent defaults and no external configuration files; TOML-driven overrides remain optional future work.

## Agent Checklist & Status
### Agent A — Schema & Data Contracts
- [x] Ported notebook `vars1` bounds/scales into `schemas.VariableSpec`/`DESIGN_FIELD_ORDER` constants (`PeetsFEA/peetspareto/pcbpcb/schemas.py`).
- [x] Embed a notebook-equivalent default `DesignVector`/dict directly in code so QA fixtures no longer depend on TOML.
- [x] Provide helper(s) that turn raw integer vectors into validated specs (zero-config factory for the forthcoming `CandidateEncoder`).

### Agent B — Model & Inference Service
- [x] `PCBPCBModel` eagerly loads LightGBM artifacts and exposes batch inference + provenance (`PeetsFEA/peetspareto/pcbpcb/model.py`).
- [x] Regression test comparing predictions to the legacy notebook pipeline (`tests/pcbpcb/test_model.py`).
- [ ] Surface user-friendly messaging in `run_pcbpcb_nsga2` when the legacy model directory is missing or corrupt.

### Agent C — Optimization & Results
- [x] Implemented `ParetoOptimizer`, `run_pareto`, `generate_pareto_front`, and IO/provenance helpers plus integration tests.
- [x] Build a `CandidateEncoder` that reproduces the integer search space (bounds, resolution, decode logic) and cover it with unit tests.
- [x] Provide a default `ObjectiveAggregator` mirroring the notebook’s Volume + total_loss objectives.
- [x] Finish `run_pcbpcb_nsga2()`: assemble encoder + model + objectives, run the optimizer, and emit Pareto CSV/provenance artifacts.

## 결과보고 (Results Report)
- **Agent A — Schema & Data Contracts**  
  - _Status_: 2024‑03‑17 — Embedded the notebook defaults directly in `schemas.DEFAULT_DESIGN_VECTOR`, added `decision_vector_to_design(...)`, and updated config/tests to run zero-config (`PeetsFEA/peetspareto/pcbpcb/schemas.py`, `config.py`, `tests/peetspareto/pcbpcb/*`). Blocker: legacy data contains fractional `Rx_width` (15.7 mm) while `VariableSpec` enforces a 1 mm step, so integer decision vectors cannot represent those points exactly—need guidance on relaxing that step vs. accepting quantization before CandidateEncoder parity tests.
  - _Artifacts_: `PeetsFEA/peetspareto/pcbpcb/schemas.py`, `PeetsFEA/peetspareto/pcbpcb/config.py`, `tests/peetspareto/pcbpcb/test_schemas.py`, `tests/peetspareto/pcbpcb/test_config.py`
- **Agent B — Model & Inference Service**  
  - _Status_: 2024-11-12 — `PCBPCBModel` now bootstraps from the legacy `legacy_codes/EVDD_PCB_PCB/model` folder by default, surfaces friendly `ModelArtifactSelectionError` guidance when artifacts are missing/corrupt, and passes new regression tests covering zero-config instantiation plus loader failures. Ready for integration with `run_pcbpcb_nsga2`.
  - _Artifacts_: `PeetsFEA/peetspareto/pcbpcb/model.py`, `PeetsFEA/peetspareto/pcbpcb/config.py`, `tests/pcbpcb/test_model.py`
- **Agent C — Optimization & Results**  
  - _Status_: 2024-12-02 — Added `LegacyCandidateEncoder` (grid snapping + repair heuristic), provenance-aware default aggregator (total_loss + geometry volume), and the zero-config `run_pcbpcb_nsga2` wrapper so the legacy NSGA-II workflow now runs end-to-end with bundled defaults and emits Pareto CSV + provenance bundles automatically.
  - _Artifacts_: `PeetsFEA/peetspareto/pcbpcb/runtime.py`, `tests/peetspareto/pcbpcb/test_runtime.py`, `tests/test_pcbpcb_optimizer.py` (tests run via `python -m pytest tests/peetspareto/pcbpcb/test_runtime.py` and `python -m pytest tests/test_pcbpcb_optimizer.py`).

## Coordination Notes
- Communicate via short design notes or ADRs before changing shared schemas.
- Prefer small PRs scoped to a single module; include notebook diff references to prove behavioral parity.
- If any requirement conflicts arise (e.g., optimizer needs new fields), Agent A updates schemas and notifies Agents B/C to rebase.
- QA Engineer should be looped in once APIs stabilize to wire regression suites and CI jobs.
- Document any AEDT/simulation dependencies even if not used yet, so `peetsansys` can consume the optimizer outputs without surprise changes.
