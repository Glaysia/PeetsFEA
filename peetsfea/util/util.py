import shutil
import time
import os
from typing import Any
from ansys.aedt.core.modules.material import Material
from ansys.aedt.core.modules.material_lib import Materials
from ansys.aedt.core.modeler.cad.polylines import Polyline
import numpy as np
import pandas as pd


import getpass
import socket

from ansys.aedt.core import Desktop, Maxwell3d
from ansys.aedt.core.internal.grpc_plugin_dll_class import AedtPropServer

from ansys.aedt.core.visualization.report.standard import Standard
from ansys.aedt.core.visualization.report.field import Fields

user_host = f"{getpass.getuser()}@{socket.gethostname()}"


def log_simulation(number, state=None, pid=None, filename="log.csv"):
    """
    number: 기록할 숫자 값
    state: None이면 초기 기록, "fail"이면 Error, 그 외는 Finished로 업데이트
    pid: 기록할 프로세스 아이디 값 (인자로 받음)
    filename: 로그 파일명 (기본 'log.csv')

    파일이 없으면 헤더( Number, Status, StartTime, PID )와 함께 생성한 후,
    초기 호출 시 새로운 레코드를 추가하고, state가 전달되면 기존 레코드의 Status를 업데이트합니다.
    """
    lock_timeout = 10  # 락 타임아웃 시간(초)
    import csv
    import portalocker
    from datetime import datetime

    # 파일이 없으면 헤더를 포함하여 생성
    if not os.path.exists(filename):
        with portalocker.Lock(filename, "w", timeout=lock_timeout, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Number", "Status", "StartTime", "PID"])

    # 초기 기록인 경우: state가 None이면 해당 번호의 레코드가 있는지 확인 후 없으면 추가
    if state is None:
        exists = False
        with portalocker.Lock(filename, "r", timeout=lock_timeout, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == str(number):
                    exists = True
                    break
        if not exists:
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with portalocker.Lock(filename, "a", timeout=lock_timeout, newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, "Simulation", start_time, pid])
    else:
        # state가 전달된 경우: 기존 레코드의 상태 업데이트
        new_status = "Error" if state.lower() == "fail" else "Finished"
        with portalocker.Lock(filename, "r+", timeout=lock_timeout, newline="") as f:
            # 파일의 모든 행을 읽고 리스트로 저장
            rows = list(csv.reader(f))
            updated_rows = []
            for row in rows:
                # 헤더나, 해당 번호의 상태가 "Simulation"인 경우만 업데이트
                if row and row[0] == str(number) and row[1] == "Simulation":
                    row[1] = new_status
                updated_rows.append(row)
            # 파일 포인터를 맨 앞으로 돌리고 내용을 덮어씌운 후 파일 내용을 잘라냅니다.
            f.seek(0)
            writer = csv.writer(f)
            writer.writerows(updated_rows)
            f.truncate()


def write_to_csv(filename, pd_data: pd.DataFrame):
    num_retries = 10
    delay = 3

    # check file existence
    file_exists = os.path.isfile(filename)

    if not file_exists:
        pd_data.to_csv(filename, header=True, mode="a")
        return True

    for i in range(num_retries):

        try:
            pd_data.to_csv(filename, header=False, mode="a")
            return True
        except Exception as e:
            print(
                f"An error occurred while writing: {e}. Retrying... ({i+1}/{num_retries})"
            )
            time.sleep(delay)

    print("Failed to read the file after multiple attempts.")
    return None


def extract_data_from_last_line(filename) -> tuple[str, str, str, str, str]:

    with open(filename, "r") as file:
        lines = file.readlines()

    # 공백이 아닌 마지막 줄을 찾기
    last_data_line = ""
    for line in reversed(lines):
        if line.strip():  # 줄이 공백이 아닐 경우
            last_data_line = line
            break

    parts = last_data_line.split("|")
    pass_number = parts[0].strip()
    tetrahedra = parts[1].strip()
    total_energy = parts[2].strip()
    energy_error = parts[3].strip()
    delta_energy = parts[4].strip()

    return pass_number, tetrahedra, total_energy, energy_error, delta_energy


def save_error_log(project_name, error_info) -> None:
    error_folder = "error"
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)
    error_file = os.path.join(error_folder, f"{project_name}_error.txt")
    with open(error_file, "w", encoding="utf-8") as f:
        f.write(error_info)


class abstract_parameter:
    def __init__(self):
        from types import SimpleNamespace as Ranges

        self.num = 0
        self.itr = 0
        self.r = Ranges()
        self.freq_kHz = 130

    def _random_choice(self, X: list[Any]):
        return round(np.random.choice(np.arange(X[0], X[1] + X[2], X[2])), X[3])

    def create_desktop(self, version="2024.2", non_graphical=True):

        # open desktop
        self.desktop = Desktop(
            close_on_exit=False,
            version=version,
            non_graphical=non_graphical
        )
        self.desktop.disable_autosave()

    def create_project(self):
        # project property
        self.num = 0
        self.project_name = f"script{self.num}"
        self.solution_type = "EddyCurrent"

        self.dir_temp = os.getcwd()

        self.dir = os.path.join(self.dir_temp, "script", f"script{self.num}")
        self.dir_project = os.path.join(self.dir, f"{self.project_name}.aedt")

        # delete and make dir
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        if os.path.exists(self.dir) == False:
            os.mkdir(self.dir)

        self.design_name = f"script{self.num}_{self.itr}"

        self.M3D = Maxwell3d(
            project=self.dir_project,
            design=self.design_name,
            solution_type=self.solution_type,
        )
        self.project = self.M3D.oproject

        from typing import Callable

        assert isinstance(self.M3D.odesign, AedtPropServer)
        oDesign: AedtPropServer = self.M3D.odesign

        assert isinstance(oDesign.SetDesignSettings, Callable)
        oDesign.SetDesignSettings(
            [
                "NAME:Design Settings Data",
                "Allow Material Override:=",
                False,
                "Perform Minimal validation:=",
                False,
                "EnabledObjects:=",
                [],
                "PerfectConductorThreshold:=",
                1e30,
                "InsulatorThreshold:=",
                1,
                "SolveFraction:=",
                False,
                "Multiplier:=",
                "1",
                "SkipMeshChecks:=",
                True,
            ],
            [
                "NAME:Model Validation Settings",
                "EntityCheckLevel:=",
                "Strict",
                "IgnoreUnclassifiedObjects:=",
                False,
                "SkipIntersectionChecks:=",
                False,
            ],
        )

    def set_variables(self):
        for k, v in self.r.__dict__.keys():
            self.M3D[k] = f"{self.values[k]}mm"

    def set_analysis(self):
        setup = self.M3D.create_setup(name="Setup1")
        setup.props["MaximumPasses"] = 5
        setup.props["MinimumPasses"] = 2
        setup.props["PercentError"] = 2.5
        setup.props["Frequency"] = f"{self.freq_kHz}kHz"

    @property
    def values(self) -> dict[str, Any]:
        return self.__dict__

    def set_material(self) -> None:
        assert not (self.M3D.materials is None)

        mat = self.M3D.materials.duplicate_material(
            material="ferrite", name="ferrite_simulation"
        )
        assert mat
        self.mat: Material = mat

        self.mat.permeability = 3000
        self.mat.set_power_ferrite_coreloss(
            cm=0.012866, x=1.7893, y=2.52296  # type: ignore
        )  # GP98 material

    def write_data(self):

        self.new_data: pd.DataFrame = pd.concat(
            [v for k, v in self.values.items() if "data" in k], axis=1
        )

        current_dir = os.getcwd()
        csv_file = os.path.join(current_dir, f"output_data.csv")

        if os.path.isfile(csv_file):
            self.new_data.to_csv(csv_file, mode="a", index=False, header=False)
        else:
            self.new_data.to_csv(csv_file, mode="w", index=False, header=True)

    def _create_polyline(self, points, name, coil_width, coil_height, orient=None) -> Polyline:
        assert self.M3D.modeler
        # orient이 주어졌을 때만 kwargs에 담기
        extra = {}
        if orient is not None:
            extra['xsection_orient'] = orient

        polyline_obj = self.M3D.modeler.create_polyline(
            points,
            name=name,
            material="copper",
            xsection_type="Rectangle",
            xsection_width=coil_width,
            xsection_height=coil_height,
            # orient이 None이면 None을 넘기지 않고 생략, 값이 있으면 그걸 넘김
            **extra
        )
        return polyline_obj

    def _create_polyline_litz(self, points, name, coil_width):
        assert self.M3D.modeler

        polyline_obj = self.M3D.modeler.create_polyline(
            points,
            name=name,
            material="copper_Litz_wire",
            xsection_type="Circle",
            xsection_width=coil_width,
            xsection_num_seg=6)

        return polyline_obj

    @staticmethod
    def report_to_df(report: Standard | Fields, name_pair: list[list[str]]) -> pd.Series:
        from ansys.aedt.core.visualization.post.solution_data import SolutionData
        solution_data = report.get_solution_data()
        assert isinstance(solution_data, SolutionData)

        solution_data.enable_pandas_output = True

        units = solution_data.units_data
        values = solution_data._solutions_mag

        assert isinstance(values, pd.DataFrame)

        columns = solution_data.expressions

        column_dict = {k: v for k, v in name_pair}

        columns_renamed = [
            f"({column_dict.get(col, None)})_{col}" for col in columns]

        columns_with_units = []
        for col, renamed in zip(columns, columns_renamed):
            unit = units.get(col, "")

            columns_with_units.append(f"{renamed}_[{unit}]")

        values.columns = columns_with_units
        ret = values.reset_index(drop=True).iloc[0]
        ret = abstract_parameter.simplify_series_keys(ret)
        ret = abstract_parameter.normalize_series_units(ret)
        return ret

    @staticmethod
    def simplify_series_keys(series: pd.Series) -> pd.Series:
        """Return a copy of the Series with compacted keys.

        Convert keys like
            '(Lmt)_Matrix1.L(Tx,Tx)*(Matrix1.CplCoef(Tx,Rx1)^2)_[nH]'
        to
            'Lmt_[nH]'

        This extracts the token inside the leading parentheses and
        preserves the trailing unit suffix (e.g. '_[nH]', '[ohm]').
        Keys that do not match are left unchanged.
        """
        new_index: list[Any] = []
        for key in series.index:
            if isinstance(key, str):
                # Find ")_" and "_[" positions only
                end_token = key.find(")_")
                unit_start = key.rfind("_[")

                if end_token != -1 and unit_start != -1 and unit_start > end_token:
                    # Extract token between '(' and ')_'
                    token = key[1:end_token] if key.startswith(
                        "(") else key[:end_token]
                    # Grab until the closing ']'
                    unit_end = key.find("]", unit_start + 2)
                    unit_suffix = key[unit_start: unit_end +
                                      1] if unit_end != -1 else key[unit_start:]
                    new_index.append(f"{token}{unit_suffix}")
                    continue
            new_index.append(key)

        renamed = series.copy()
        renamed.index = new_index
        return renamed

    @staticmethod
    def normalize_series_units(series: pd.Series) -> pd.Series:
        """Normalize numeric units in the Series and update index names.

        Assumes keys are already simplified like 'Name_[unit]'.
        - Inductance: H, mH, uH, nH, pH  -> uH
        - Resistance: mohm, ohm, kohm    -> ohm
        - Power:      mW, W, kW          -> W
        Values are scaled accordingly; non-matching keys remain unchanged.
        """
        unit_map: dict[str, tuple[str, float]] = {
            # Inductance to uH
            "H": ("uH", 1e6),
            "mH": ("uH", 1e3),
            "uH": ("uH", 1.0),
            "nH": ("uH", 1e-3),
            "pH": ("uH", 1e-6),
            # Resistance to ohm
            "mohm": ("ohm", 1e-3),
            "ohm": ("ohm", 1.0),
            "kohm": ("ohm", 1e3),
            # Power to W
            "mW": ("W", 1e-3),
            "W": ("W", 1.0),
            "kW": ("W", 1e3),
        }

        new_values: list[Any] = []
        new_index: list[Any] = []

        for key, val in series.items():
            if isinstance(key, str):
                # Expected pattern: '<base>_[<unit>]'
                unit_pos = key.rfind("_[")
                end_br = key.rfind("]")
                if unit_pos != -1 and end_br != -1 and end_br > unit_pos:
                    base = key[:unit_pos]
                    unit = key[unit_pos + 2: end_br]
                else:
                    # Fallback: '<base>[<unit>]' (without the underscore)
                    lbr = key.rfind("[")
                    rbr = key.rfind("]")
                    if lbr != -1 and rbr != -1 and rbr > lbr:
                        base = key[:lbr].rstrip("_")
                        unit = key[lbr + 1: rbr]
                    else:
                        new_index.append(key)
                        new_values.append(val)
                        continue

                mapping = unit_map.get(unit)
                if mapping is None:
                    # Unknown unit, keep as is
                    new_index.append(key)
                    new_values.append(val)
                    continue

                target_unit, factor = mapping
                try:
                    scaled_val = float(val) * factor
                except Exception:
                    # Non-numeric; keep value but still rename unit
                    scaled_val = val

                new_key = f"{base}_[{target_unit}]"
                new_index.append(new_key)
                new_values.append(scaled_val)
            else:
                new_index.append(key)
                new_values.append(val)

        return pd.Series(new_values, index=new_index)

    def volumetric_loss(self, assignments: str) -> str:
        oModule = self.M3D.get_module(
            module_name="FieldsReporter"
        )

        oModule.EnterQty("OhmicLoss")
        oModule.EnterVol(assignments)
        oModule.CalcOp("Integrate")
        name = f"P_{assignments}"
        oModule.AddNamedExpression(name, 'Fields')
        return name

    @staticmethod
    def get_magetizing_current(Lmt_uH, freq_kHz, Vin=390):
        import math
        omegaL = 2*math.pi*(freq_kHz*10**3)*Lmt_uH*10**(-6)

        return Vin*math.sqrt(2)/2/math.pi/(freq_kHz*10**(3))/Lmt_uH/10**(-6)/2
