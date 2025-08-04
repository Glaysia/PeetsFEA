from dataclasses import dataclass
import fcntl
import sys
from typing import Any

import numpy as np
import pandas as pd
import csv
import os
import math
import time
import shutil
from datetime import datetime

import traceback
import logging
import platform

from ansys.aedt.core import Desktop, Maxwell3d, Icepak
from ansys.aedt.core.application.design import Design
from ansys.aedt.core.modeler.cad.object_3d import Object3d
from ansys.aedt.core.modeler.cad.polylines import Polyline

from peetsfea import util

# geometry parameter class


class Parameter(util.Parameter):
    def __init__(self):

        self.r.l_in = (0, 70, 1, 0)
        self.r.l_mid = (3, 40, 1, 0)
        self.r.l_out = (25, 200, 1, 0)
        self.r.l_side = (0, 70, 1, 0)

        self.r.t_in = (0, 5, 0.1, 1)
        self.r.t_mid = (1, 44.8, 0.3, 1)
        self.r.t_out = (1, 25, 0.3, 1)
        self.r.t_side = (0, 24, 0.3, 1)

        self.r.air_gap = (0, 5, 0.2, 1)

        self.l_in = None
        self.l_mid = None
        self.l_out = None
        self.l_side = None

        self.t_in = None
        self.t_mid = None
        self.t_out = None
        self.t_side = None

        self.air_gap = None


    def set_random_variables(self) -> None:
        for k, v in self.r.__dict__.items():
            self.values[k] = self._random_choice(v)

        while True:  # 제약조건 모두 False여야 루프 종료
            형상제약조건1 = True
            형상제약조건2 = False
            if 형상제약조건1:
                pass

            elif 형상제약조건2:
                pass

            else:
                break


class Sim(Parameter):
    def __init__(self):
        self.project_name = "LCR-TAB"
        self.flag = 1
        self.Proj = 0
        self.num = 0
        self.itr = 0
        self.freq_kHz = 130

        self.computer_name = util.user_host
        self.create_desktop()

    def simulation(self):
        self.start_time = time.time()
        file_path = "simulation_num.txt"

        # 파일이 존재하지 않으면 생성
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as file:
                file.write("1")

        # 읽기/쓰기 모드로 파일 열기
        with open(file_path, "r+", encoding="utf-8") as file:
            # 파일 잠금: LOCK_EX는 배타적 잠금,  블로킹 모드로 실행
            fcntl.flock(file, fcntl.LOCK_EX)

            # 파일에서 값 읽기
            content = int(file.read().strip())
            self.num = content
            self.PROJECT_NAME = f"simulation{content}"
            content += 1

            # 파일 포인터를 처음으로 되돌리고, 파일 내용 초기화 후 새 값 쓰기
            file.seek(0)
            file.truncate()
            file.write(str(content))

        # 파일은 with 블록 종료 시 자동으로 닫히며, 잠금도 해제됨

        util.log_simulation(number=self.num, pid=self.desktop.aedt_process_id)

        self.create_project()
        self.set_random_variables()
        self.set_variables()
        self.set_analysis()
        self.set_material()

        self.eddyloss_project()

        self.coreloss_project()
        self.hfss_project()
        self.icepak_project()
        self.write_data()

    def eddyloss_project(self):
        self.create_core()
        self.create_winding()
        self.assign_mesh()
        self.create_region()

        self.create_exctation()

        self.M3D.analyze()

        self._get_magnetic_report()
        self.get_input_parameter()
        self._get_copper_loss_parameter()

def main():
    sim = Sim()
    for i in range(5000):

    try:
        # 시뮬레이션 코드 실행
        sim.simulation()

main()
