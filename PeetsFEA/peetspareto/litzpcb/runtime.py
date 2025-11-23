"""
Lightweight NSGA-II wrapper that mirrors ``legacy_codes/EVDD_litz_PCB_v2/NSGA-II.ipynb``.

The goal is behavioral parity with the notebook, not a full refactor. Models are
loaded directly from the legacy ``model`` directory, the search space and
objective math are kept intact, and Pareto CSVs are written to a disposable
scratch directory under ``~/.peetsfea/pareto`` before being cleaned up.
"""

from __future__ import annotations

import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

MODEL_DATE = "251112"
MODEL_LABELS: tuple[str, ...] = (
    "Lmt",
    "Llt",
    "copperloss_Tx",
    "copperloss_Rx1",
    "copperloss_Rx2",
    "coreloss",
    "magnetizing_copperloss_Tx",
    "magnetizing_copperloss_Rx1",
    "magnetizing_copperloss_Rx2",
    "B_left",
    "B_center",
    "B_top_left",
)

SCALE = np.array(
    [
        5,
        50,
        1,
        0.1,
        0.1,
        0.1,
        0.01,
        1,
        1,
        0.1,
        0.1,
        0.01,
        0.1,
        0.035,
        0.1,
        0.1,
        0.01,
        0.01,
        0.1,
        0.1,
        0.01,
        1,
    ],
    dtype=float,
)
OFFSET = np.array(
    [
        0,
        100,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    dtype=float,
)

VAR_RANGES: dict[str, Sequence[int]] = {
    "freq_range": [1, 80, 1, 0],
    "input_voltage_range": [1, 8, 1, 0],
    "w1_range": [20, 200, 1, 0],
    "l1_leg_range": [20, 150, 1, 0],
    "l1_top_range": [5, 20, 1, 0],
    "l2_range": [50, 300, 1, 0],
    "h1_range": [10, 300, 1, 0],
    "l1_center_range": [1, 25, 1, 0],
    "Tx_turns_range": [2, 20, 1, 0],
    "Tx_space_x_range": [1, 50, 1, 0],
    "Tx_space_y_range": [1, 50, 1, 0],
    "Tx_preg_range": [5, 30, 1, 0],
    "Rx_width_range": [40, 200, 1, 0],
    "Rx_height_range": [1, 5, 1, 0],
    "Rx_space_x_range": [1, 50, 1, 0],
    "Rx_space_y_range": [1, 50, 1, 0],
    "Rx_preg_range": [5, 30, 1, 0],
    "g2_range": [10, 300, 1, 0],
    "Tx_layer_space_x_range": [2, 50, 1, 0],
    "Tx_layer_space_y_range": [2, 50, 1, 0],
    "wire_diameter_range": [5, 8, 1, 0],
    "strand_number_range": [7, 100, 1, 0],
}

FEATURE_NAMES = (
    "freq",
    "w1",
    "l1_leg",
    "l1_top",
    "l2",
    "h1",
    "l1_center",
    "Tx_turns",
    "Tx_width",
    "Tx_height",
    "Tx_space_x",
    "Tx_space_y",
    "Tx_preg",
    "Rx_width",
    "Rx_height",
    "Rx_space_x",
    "Rx_space_y",
    "Rx_preg",
    "g2",
    "Tx_layer_space_x",
    "Tx_layer_space_y",
    "wire_diameter",
    "strand_number",
    "Tx_current",
    "Rx_current",
)

MU, NGEN = 100, 400
CXPB, MUTPB = 0.7, 0.3
NUM_ITERS = 5

LEGACY_MODEL_ROOT = Path("legacy_codes") / "EVDD_litz_PCB_v2" / "model"


@dataclass(frozen=True)
class LitzPCBResult:
    pareto: pd.DataFrame
    scratch_dir: Path


class ProblemVariables:
    def __init__(self, var_dict: Mapping[str, Sequence[int]]) -> None:
        self.var_dict = var_dict
        self.names = list(var_dict.keys())

    def get_num_of_variables(self) -> int:
        return len(self.var_dict)

    def get_lower_bounds(self) -> np.ndarray:
        return np.array([v[0] for v in self.var_dict.values()], dtype=int)

    def get_upper_bounds(self) -> np.ndarray:
        return np.array([v[1] for v in self.var_dict.values()], dtype=int)


def decode_vars(X_raw: np.ndarray) -> np.ndarray:
    """Integer design vector -> physical units."""
    X = np.atleast_2d(X_raw).astype(float)
    return X * SCALE + OFFSET


def pre_processing_data_copper(X_raw: np.ndarray) -> np.ndarray:
    X = decode_vars(X_raw)
    (
        freq,
        input_voltage,
        w1,
        l1_leg,
        l1_top,
        l2,
        h1,
        l1_center,
        Tx_turns,
        Tx_space_x,
        Tx_space_y,
        Tx_preg,
        Rx_width,
        Rx_height,
        Rx_space_x,
        Rx_space_y,
        Rx_preg,
        g2,
        Tx_layer_space_x,
        Tx_layer_space_y,
        wire_diameter,
        strand_number,
    ) = X.T

    Tx_width = (wire_diameter**2 * strand_number * 2) ** 0.5
    Tx_height = Tx_width

    return np.column_stack(
        (
            freq,
            w1,
            l1_leg,
            l1_top,
            l2,
            h1,
            l1_center,
            Tx_turns,
            Tx_width,
            Tx_height,
            Tx_space_x,
            Tx_space_y,
            Tx_preg,
            Rx_width,
            Rx_height,
            Rx_space_x,
            Rx_space_y,
            Rx_preg,
            g2,
            Tx_layer_space_x,
            Tx_layer_space_y,
            wire_diameter,
            strand_number,
        )
    )


def pre_processing_data_core(
    X_raw: np.ndarray, Tx_current: np.ndarray, magnetizing_current: np.ndarray
) -> np.ndarray:
    X = decode_vars(X_raw)
    (
        freq,
        input_voltage,
        w1,
        l1_leg,
        l1_top,
        l2,
        h1,
        l1_center,
        Tx_turns,
        Tx_space_x,
        Tx_space_y,
        Tx_preg,
        Rx_width,
        Rx_height,
        Rx_space_x,
        Rx_space_y,
        Rx_preg,
        g2,
        Tx_layer_space_x,
        Tx_layer_space_y,
        wire_diameter,
        strand_number,
    ) = X.T

    Tx_width = (wire_diameter**2 * strand_number * 2) ** 0.5
    Tx_height = Tx_width

    return np.column_stack(
        (
            freq,
            w1,
            l1_leg,
            l1_top,
            l2,
            h1,
            l1_center,
            Tx_turns,
            Tx_width,
            Tx_height,
            Tx_space_x,
            Tx_space_y,
            Tx_preg,
            Rx_width,
            Rx_height,
            Rx_space_x,
            Rx_space_y,
            Rx_preg,
            g2,
            Tx_layer_space_x,
            Tx_layer_space_y,
            wire_diameter,
            strand_number,
            magnetizing_current,
        )
    )


def compute_area(X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = decode_vars(X_raw)
    (
        freq,
        input_voltage,
        w1,
        l1_leg,
        l1_top,
        l2,
        h1,
        l1_center,
        Tx_turns,
        Tx_space_x,
        Tx_space_y,
        Tx_preg,
        Rx_width,
        Rx_height,
        Rx_space_x,
        Rx_space_y,
        Rx_preg,
        g2,
        Tx_layer_space_x,
        Tx_layer_space_y,
        wire_diameter,
        strand_number,
    ) = X.T

    Tx_width = (wire_diameter**2 * strand_number * 2) ** 0.5
    Tx_height = Tx_width

    area = (l1_leg + l2 + l1_center / 2) * 2 * (
        w1 + Tx_layer_space_x * 4 + Tx_width * 5 + Tx_space_x + Rx_space_x + Rx_width
    )
    height = l1_top * 2 + h1
    volume = area * height

    Tx = Tx_width * 5 + Tx_layer_space_y * 4 + Tx_space_y
    Rx = Rx_width + Rx_space_y
    max_length = l2 - (Tx + Rx)
    gap = h1 - g2

    coil_height = Tx_height * 2 + Tx_preg + Rx_height * 2 + Rx_preg * 2
    hh = h1 - coil_height
    Tx_Rx = Tx_preg / 2 + Rx_preg

    return area, height, max_length, Tx_Rx, hh, gap, volume


class LitzPCBProblem(Problem):
    def __init__(self, problem_vars: ProblemVariables, models: Mapping[str, Any]) -> None:
        self.vars = problem_vars
        self.models = models
        super().__init__(
            n_var=self.vars.get_num_of_variables(),
            n_obj=2,
            n_constr=11,
            xl=np.array(self.vars.get_lower_bounds()),
            xu=np.array(self.vars.get_upper_bounds()),
            type_var=int,
        )

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:  # noqa: D401
        Lmt_data = pre_processing_data_copper(X)

        tx_val = 4 * math.sqrt(2)
        rx_val = 11 * math.sqrt(2)
        n = Lmt_data.shape[0]

        tx_col = np.full((n, 1), tx_val, dtype=Lmt_data.dtype)
        rx_col = np.full((n, 1), rx_val, dtype=Lmt_data.dtype)

        copper_data = np.hstack((Lmt_data, tx_col, rx_col))

        copperloss_Tx = self.models["copperloss_Tx"].predict(copper_data)
        copperloss_Rx1 = self.models["copperloss_Rx1"].predict(copper_data)
        copperloss_Rx2 = self.models["copperloss_Rx2"].predict(copper_data)

        Lmt = self.models["Lmt"].predict(copper_data)
        Llt = self.models["Llt"].predict(copper_data)

        freq_one = X[:, 0]
        freq = freq_one * 5

        input_voltage_one = X[:, 1]
        input_voltage = input_voltage_one * 50 + 100

        magnetizing_current = (
            input_voltage * math.sqrt(2) / 2 / math.pi / freq / 10**3 / Lmt / 10 ** (-6) / 2
        )

        core_data = pre_processing_data_core(X, tx_col, magnetizing_current)

        coreloss = self.models["coreloss"].predict(core_data)
        magnetizing_copperloss_Tx = self.models["magnetizing_copperloss_Tx"].predict(core_data)
        magnetizing_copperloss_Rx1 = self.models["magnetizing_copperloss_Rx1"].predict(core_data)
        magnetizing_copperloss_Rx2 = self.models["magnetizing_copperloss_Rx2"].predict(core_data)

        area, height, max_length, Tx_Rx, hh, gap, volume = compute_area(X)

        total_loss = (
            copperloss_Tx
            + copperloss_Rx1
            + copperloss_Rx2
            + coreloss
            + magnetizing_copperloss_Tx
            + magnetizing_copperloss_Rx1
            + magnetizing_copperloss_Rx2
        )

        B_center = self.models["B_center"].predict(core_data)
        B_left = self.models["B_left"].predict(core_data)
        B_top_left = self.models["B_top_left"].predict(core_data)

        gLmt = (Lmt - 67) * (Lmt - 73)
        gh = (height - 4.5) * (height - 5)

        gB_center = B_center - 0.3
        gB_left = B_left - 0.3
        gB_top_left = B_top_left - 0.3

        gmax_length = -(max_length - 0.00001)
        ghh = -(hh - 0.00001)
        ggap = -(gap - 0.00001)
        gTx_Rx = -(Tx_Rx - 0.0004)
        ginput_voltage = (input_voltage - 380) * (input_voltage - 420)
        gfreq = (freq - 100) * (freq - 150)

        out["F"] = np.column_stack([volume, total_loss])
        out["G"] = np.column_stack(
            [
                gLmt,
                gh,
                gB_center,
                ghh,
                gmax_length,
                ggap,
                gB_left,
                gB_top_left,
                gTx_Rx,
                ginput_voltage,
                gfreq,
            ]
        )
        out["magnetizing_current"] = magnetizing_current
        out["input_voltage"] = input_voltage
        out["copper_data"] = copper_data
        out["Lmt_data"] = copper_data
        out["core_data"] = core_data
        out["Area"] = area
        out["Volume"] = volume


def load_models(model_root: Path = LEGACY_MODEL_ROOT) -> dict[str, Any]:
    root = Path(model_root).expanduser().resolve()
    model_paths = {label: root / f"{label}_{MODEL_DATE}.pkl" for label in MODEL_LABELS}
    missing = [str(path) for path in model_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing LightGBM artifacts: {missing}")

    models = {}
    for label, path in model_paths.items():
        models[label] = joblib.load(path)
    return models


def run_litzpcb_nsga2(
    show_plot: bool = False,
    *,
    scratch_root: Path | None = None,
    cleanup: bool = True,
) -> LitzPCBResult:
    """
    Execute the legacy NSGA-II loop and return the resulting Pareto DataFrame.

    Args:
        show_plot: When True, renders the Volume vs total_loss scatter at the end.
        scratch_root: Optional override for where temporary CSVs are written.
        cleanup: Remove the scratch directory after the run (kept when False).
    """

    models = load_models()
    problem = LitzPCBProblem(ProblemVariables(VAR_RANGES), models)

    scratch_dir = _prepare_scratch_dir(scratch_root)
    pareto_df = _run_iterations(problem, models, scratch_dir)

    final_path = scratch_dir / "pareto_front_final.csv"
    pareto_df.to_csv(final_path, index=False)

    if show_plot:
        _plot_pareto(pareto_df)

    if cleanup:
        _cleanup_scratch(scratch_dir)

    return LitzPCBResult(pareto=pareto_df, scratch_dir=scratch_dir)


def _run_iterations(problem: LitzPCBProblem, models: Mapping[str, Any], scratch_dir: Path) -> pd.DataFrame:
    pareto_file = scratch_dir / "pareto_front.csv"
    loop_counter_file = scratch_dir / "loop_counter.txt"
    best_values_file = scratch_dir / "best_values.csv"

    pareto_df = pd.DataFrame()
    best_values = pd.DataFrame(columns=["iteration", "max_loss", "min_Volume"])
    loop_counter = 0

    for itr in range(NUM_ITERS):
        seed = np.random.randint(0, 1_000_000)
        res = _run_single_nsga2(seed, problem)
        df = _build_df_from_result(res, models)
        if df is None or df.empty:
            continue

        loop_counter += 1
        current = pd.concat([df, pareto_df], ignore_index=True)
        F = current[["Volume", "total_loss"]].to_numpy()
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
        pareto_df = current.iloc[nds].copy()
        pareto_df["total_loss"] = F[nds, 1]
        pareto_df = pareto_df.sort_values(by="total_loss", ascending=False).reset_index(drop=True)

        max_loss = df["total_loss"].min()
        min_volume = df["Volume"].max()
        best_values = pd.concat(
            [
                best_values,
                pd.DataFrame({"iteration": [loop_counter], "max_loss": [max_loss], "min_Volume": [min_volume]}),
            ],
            ignore_index=True,
        )

        _persist_iteration(
            pareto_df=pareto_df,
            best_values=best_values,
            loop_counter=loop_counter,
            pareto_file=pareto_file,
            best_values_file=best_values_file,
            loop_counter_file=loop_counter_file,
            scratch_dir=scratch_dir,
        )

    return pareto_df


def _run_single_nsga2(seed: int, problem: LitzPCBProblem):
    np.random.seed(seed)
    random.seed(seed)

    algorithm = NSGA2(
        pop_size=MU,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=CXPB, eta=15, repair=RoundingRepair()),
        mutation=PM(eta=20, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    return minimize(
        problem=problem,
        algorithm=algorithm,
        termination=("n_gen", NGEN),
        seed=seed,
        verbose=True,
    )


def _build_df_from_result(res, models: Mapping[str, Any]) -> pd.DataFrame:
    magnetizing_current = res.pop.get("magnetizing_current")
    input_voltage = res.pop.get("input_voltage")
    copper_data = res.pop.get("copper_data")
    core_data = res.pop.get("core_data")
    Lmt_data = res.pop.get("Lmt_data")
    X_design = res.X.astype(int)
    Area = res.pop.get("Area")
    Volume = res.pop.get("Volume")

    copperloss_Tx = models["copperloss_Tx"].predict(copper_data)
    copperloss_Rx1 = models["copperloss_Rx1"].predict(copper_data)
    copperloss_Rx2 = models["copperloss_Rx2"].predict(copper_data)
    Lmt = models["Lmt"].predict(Lmt_data)
    Llt = models["Llt"].predict(copper_data)

    magnetizing_copperloss_Tx = models["magnetizing_copperloss_Tx"].predict(core_data)
    magnetizing_copperloss_Rx1 = models["magnetizing_copperloss_Rx1"].predict(core_data)
    magnetizing_copperloss_Rx2 = models["magnetizing_copperloss_Rx2"].predict(core_data)

    coreloss = models["coreloss"].predict(core_data)
    B_center = models["B_center"].predict(core_data)
    B_top_left = models["B_top_left"].predict(core_data)
    B_left = models["B_left"].predict(core_data)

    total_loss = (
        copperloss_Tx
        + copperloss_Rx1
        + copperloss_Rx2
        + coreloss
        + magnetizing_copperloss_Tx
        + magnetizing_copperloss_Rx1
        + magnetizing_copperloss_Rx2
    )

    df = pd.DataFrame(copper_data, columns=FEATURE_NAMES)
    df = df.assign(
        magnetizing_current=magnetizing_current,
        input_voltage=input_voltage,
        Lmt=Lmt,
        Llt=Llt,
        copperloss_Tx=copperloss_Tx,
        copperloss_Rx1=copperloss_Rx1,
        copperloss_Rx2=copperloss_Rx2,
        coreloss=coreloss,
        magnetizing_copperloss_Tx=magnetizing_copperloss_Tx,
        magnetizing_copperloss_Rx1=magnetizing_copperloss_Rx1,
        magnetizing_copperloss_Rx2=magnetizing_copperloss_Rx2,
        total_loss=total_loss,
        B_center=B_center,
        B_left=B_left,
        Area=Area,
        Volume=Volume,
        # keeping DataFrame schema close to the notebook; X_design unused here
    )
    return df


def _persist_iteration(
    *,
    pareto_df: pd.DataFrame,
    best_values: pd.DataFrame,
    loop_counter: int,
    pareto_file: Path,
    best_values_file: Path,
    loop_counter_file: Path,
    scratch_dir: Path,
) -> None:
    pareto_df.to_csv(pareto_file, index=False)
    best_values.to_csv(best_values_file, index=False)
    loop_counter_file.write_text(str(loop_counter))

    backup_file = scratch_dir / f"pareto_front_backup_{loop_counter}.csv"
    pareto_df.to_csv(backup_file, index=False)


def _prepare_scratch_dir(scratch_root: Path | None) -> Path:
    root = Path(scratch_root) if scratch_root else Path.home() / ".peetsfea" / "pareto"
    root = root.expanduser().resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"litzpcb_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _cleanup_scratch(scratch_dir: Path) -> None:
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir, ignore_errors=True)


def _plot_pareto(df_pareto: pd.DataFrame) -> None:
    f1_vals = (
        ((df_pareto["l1_leg"] + df_pareto["l2"] + df_pareto["l1_center"] / 2) * 2)
        * (df_pareto["w1"] + (df_pareto["Rx_space_x"] + df_pareto["Rx_width"]))
        * (df_pareto["h1"] + df_pareto["l1_top"] * 2)
        * 1e-3
    )
    f2_vals = df_pareto["total_loss"]

    plt.figure()
    plt.scatter(f1_vals, f2_vals)
    plt.xlabel("Volume [cm³]")
    plt.ylabel("total loss [W]")
    plt.title("EVDD transformer design (litzpcb)")
    plt.grid(True)
    plt.show()


__all__ = ["run_litzpcb_nsga2", "LitzPCBResult", "load_models"]
