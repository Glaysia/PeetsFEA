import shutil
import time, os
from typing import Any
from ansys.aedt.core.modules.material import Material
from ansys.aedt.core.modules.material_lib import Materials
import numpy as np
import pandas as pd

import getpass, socket

from ansys.aedt.core import Desktop, Maxwell3d
from ansys.aedt.core.internal.grpc_plugin_dll_class import AedtPropServer

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
    import csv, portalocker
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

    def _random_choice(self, X: tuple[Any, ...]):
        return round(np.random.choice(np.arange(X[0], X[1] + X[2], X[2])), X[3])

    def create_desktop(self, version="2024.2", non_graphical=True):

        # open desktop
        self.desktop = Desktop(version=version, non_graphical=non_graphical)
        self.desktop.disable_autosave()

    def create_project(self):
        # project property
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
