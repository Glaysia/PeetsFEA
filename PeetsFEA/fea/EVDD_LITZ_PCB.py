########################################
# Litz PCB simulation with deterministic input support
########################################
import sys
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

from ansys.aedt.core import Desktop, Maxwell3d, Icepak
from ansys.aedt.core.application.design import Design
from ansys.aedt.core.modeler.cad.object_3d import Object3d
from ansys.aedt.core.modeler.cad.polylines import Polyline

import copy

import fcntl
import portalocker
from dataclasses import dataclass, asdict

PYANSYS_BASE_DIR = os.path.join(os.path.expanduser("~"), ".peetsfea")
PYANSYS_PRJ_DIR = os.path.join(PYANSYS_BASE_DIR, "log")
AEDT_BASE_DIR = os.path.join(PYANSYS_BASE_DIR, "aedt")

for _path in (PYANSYS_PRJ_DIR, AEDT_BASE_DIR):
    os.makedirs(_path, exist_ok=True)


@dataclass
class SimulationInputParameters:
    freq: float
    input_voltage: float
    w1: float
    l1_leg: float
    l1_top: float
    l2: float
    h1: float
    l1_center: float
    Tx_turns: float
    Tx_width: float
    Tx_height: float
    Tx_space_x: float
    Tx_space_y: float
    Tx_preg: float
    Rx_width: float
    Rx_height: float
    Rx_space_x: float
    Rx_space_y: float
    Rx_preg: float
    g2: float
    Tx_layer_space_x: float
    Tx_layer_space_y: float
    wire_diameter: float
    strand_number: float
    Tx_current: float
    Rx_current: float

    @classmethod
    def from_series(cls, series: pd.Series) -> "SimulationInputParameters":
        values = {}
        missing = []
        for field in cls.__dataclass_fields__.values():
            if field.name in series:
                values[field.name] = float(series[field.name])
            else:
                missing.append(field.name)
        if missing:
            raise ValueError(f"Missing fields for SimulationInputParameters: {missing}")
        return cls(**values)


class Parameter:
    def __init__(self):
        self.a = None

    def _random_choice(self, X):
        return round(np.random.choice(np.arange(X[0], X[1] + X[2], X[2])), X[3])

    def get_random_variable(self):
        freq_range = [5, 400, 5, 0]
        input_voltage_range = [100, 500, 50, 0]

        w1_range = [20, 200, 1, 0]
        l1_leg_range = [2, 15, 0.1, 1]
        l1_top_range = [0.5, 2, 0.1, 1]
        l2_range = [5, 30, 0.1, 1]

        h1_range = [0.1, 3, 0.01, 2]

        Tx_turns_range = [2, 20, 1, 0]
        Tx_preg_range = [0.05, 0.3, 0.01, 2]
        Rx_preg_range = [0.05, 0.3, 0.01, 2]
        Rx_height_range = [0.1, 1, 0.1, 1]

        g1_range = [0, 0, 0.01, 2]
        g2_range = [0.1, 3, 0.01, 2]

        l1_center_range = [1, 25, 1, 0]

        Tx_space_x_range = [0.1, 5, 0.1, 1]
        Tx_space_y_range = [0.1, 5, 0.1, 1]
        Rx_space_x_range = [0.1, 5, 0.1, 1]
        Rx_space_y_range = [0.1, 5, 0.1, 1]

        Tx_layer_space_x_range = [0.2, 5, 0.1, 1]
        Tx_layer_space_y_range = [0.2, 5, 0.1, 1]
        Rx_layer_space_x_range = [0.2, 5, 0.1, 1]
        Rx_layer_space_y_range = [0.2, 5, 0.1, 1]

        Rx_width_range = [4, 20, 0.1, 1]

        wire_diameter_range = [0.05, 0.08, 0.01, 2]
        strand_number_range = [7, 100, 2, 0]

        Tx_current_range = [1, 10, 0.1, 0]
        Rx_current_range = [1, 40, 0.1, 0]

        self.freq = self._random_choice(freq_range)
        self.input_voltage = self._random_choice(input_voltage_range)

        self.w1 = self._random_choice(w1_range)
        self.l1_leg = self._random_choice(l1_leg_range)

        self.l1_top = self._random_choice(l1_top_range)
        self.l2 = self._random_choice(l2_range)

        self.h1 = self._random_choice(h1_range)

        self.Tx_turns = self._random_choice(Tx_turns_range)
        self.Tx_preg = self._random_choice(Tx_preg_range)

        self.Rx_space_y = self._random_choice(Rx_space_y_range)
        self.Rx_preg = self._random_choice(Rx_preg_range)

        self.Rx_height = self._random_choice(Rx_height_range)
        self.Rx_space_x = self._random_choice(Rx_space_x_range)

        self.g1 = 0
        self.g2 = self._random_choice(g2_range)
        self.l1_center = self._random_choice(l1_center_range)

        self.Tx_space_x = self._random_choice(Tx_space_x_range)
        self.Tx_space_y = self._random_choice(Tx_space_y_range)
        self.Rx_space_x = self._random_choice(Rx_space_x_range)
        self.Rx_space_y = self._random_choice(Rx_space_y_range)

        self.Tx_layer_space_x = self._random_choice(Tx_layer_space_x_range)
        self.Tx_layer_space_y = self._random_choice(Tx_layer_space_y_range)
        self.Rx_layer_space_x = self._random_choice(Rx_layer_space_x_range)
        self.Rx_layer_space_y = self._random_choice(Rx_layer_space_y_range)

        self.Rx_width = self._random_choice(Rx_width_range)

        self.wire_diameter = self._random_choice(wire_diameter_range)
        self.strand_number = self._random_choice(strand_number_range)

        self.Tx_current = self._random_choice(Tx_current_range)
        self.Rx_current = self._random_choice(Rx_current_range)

        self.Tx_width = round(math.sqrt(self.wire_diameter ** 2 * self.strand_number * 2), 2)
        self.Tx_height = self.Tx_width

        Tx_max = (self.Tx_layer_space_y + self.Tx_width) * math.ceil(self.Tx_turns) + self.Tx_space_y - self.Tx_layer_space_y
        Rx_max = self.Rx_width + self.Rx_space_y

        while True:
            if self.Tx_height * 2 + self.Tx_preg + self.Rx_height * 2 + self.Rx_preg * 2 >= self.h1:
                self.wire_diameter = self._random_choice(wire_diameter_range)
                self.strand_number = self._random_choice(strand_number_range)
                self.Tx_width = round(math.sqrt(self.wire_diameter ** 2 * self.strand_number * 2), 2)
                self.Tx_height = self.Tx_width

                self.Tx_preg = self._random_choice(Tx_preg_range)
                self.Rx_height = self._random_choice(Rx_height_range)
                self.Rx_preg = self._random_choice(Rx_preg_range)
                self.h1 = self._random_choice(h1_range)

            elif Tx_max + Rx_max >= self.l2:
                self.Tx_layer_space_y = self._random_choice(Tx_layer_space_y_range)
                self.wire_diameter = self._random_choice(wire_diameter_range)
                self.strand_number = self._random_choice(strand_number_range)
                self.Tx_width = round(math.sqrt(self.wire_diameter ** 2 * self.strand_number * 2), 2)
                self.Tx_height = self.Tx_width
                self.l2 = self._random_choice(l2_range)
                self.Rx_width = self._random_choice(Rx_width_range)
                self.Rx_space_y = self._random_choice(Rx_space_y_range)

                Tx_max = (self.Tx_layer_space_y + self.Tx_width) * math.ceil(self.Tx_turns) + self.Tx_space_y - self.Tx_layer_space_y
                Rx_max = self.Rx_width + self.Rx_space_y

            elif self.g2 >= self.h1:
                self.g2 = self._random_choice(g2_range)
                self.h1 = self._random_choice(h1_range)

            else:
                break

    def apply_input_parameters(self, params: SimulationInputParameters):
        values = asdict(params)
        for key, value in values.items():
            setattr(self, key, value)
        self.g1 = 0


class Sim(Parameter):
    def __init__(self):
        self.project_name = "script1"
        self.flag = 1
        self.Proj = 0
        self.itr = 0
        self.freq = 105

        self.computer_name = "5950X1"
        self.create_desktop()

    def create_desktop(self):
        self.desktop = Desktop(version="2024.2", non_graphical=False,new_desktop=False, close_on_exit=False)
        self.desktop.disable_autosave()

    def create_project(self):
        self.project_name = f"script{self.num}"
        self.solution_type = "EddyCurrent"

        self.dir_temp = AEDT_BASE_DIR

        self.dir = os.path.join(self.dir_temp, f"script{self.num}")
        self.dir_project = os.path.join(self.dir, f"{self.project_name}.aedt")

        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        self.design_name = f"script{self.num}_{self.itr}"

        self.M3D = Maxwell3d(project=self.dir_project, design=self.design_name, solution_type=self.solution_type)
        self.project = self.M3D.oproject
        oDesign = self.M3D.odesign
        oDesign.SetDesignSettings(
            [
                "NAME:Design Settings Data",
                "Allow Material Override:=", False,
                "Perform Minimal validation:=", False,
                "EnabledObjects:=", [],
                "PerfectConductorThreshold:=", 1e30,
                "InsulatorThreshold:=", 1,
                "SolveFraction:=", False,
                "Multiplier:=", "1",
                "SkipMeshChecks:=", True,
            ],
            [
                "NAME:Model Validation Settings",
                "EntityCheckLevel:=", "Strict",
                "IgnoreUnclassifiedObjects:=", False,
                "SkipIntersectionChecks:=", False,
            ],
        )

    def set_material(self):
        self.mat = self.M3D.materials.duplicate_material(material="ferrite", name="ferrite_simulation")
        self.mat.permeability = 3000
        self.mat.set_power_ferrite_coreloss(cm=0.012866, x=1.7893, y=2.52296)

    def set_variable(self):
        self.M3D["w1"] = f"{self.w1}mm"
        self.M3D["l1_leg"] = f"{self.l1_leg}mm"
        self.M3D["l1_top"] = f"{self.l1_top}mm"
        self.M3D["l2"] = f"{self.l2}mm"
        self.M3D["h1"] = f"{self.h1}mm"
        self.M3D["w1_ratio"] = "1"
        self.M3D["l1_center"] = f"{self.l1_center}mm"
        self.M3D["ratio"] = "1"
        self.M3D["Tx_turns"] = f"{self.Tx_turns}"
        self.M3D["Rx_turns"] = "1"
        self.M3D["Tx_width"] = f"{self.Tx_width}mm"
        self.M3D["Tx_height"] = f"{self.Tx_height}mm"
        self.M3D["Tx_space_x"] = f"{self.Tx_space_x}mm"
        self.M3D["Tx_space_y"] = f"{self.Tx_space_y}mm"
        self.M3D["Tx_preg"] = f"{self.Tx_preg}mm"
        self.M3D["Rx_width"] = f"{self.Rx_width}mm"
        self.M3D["Rx_preg"] = f"{self.Rx_preg}mm"
        self.M3D["Rx_height"] = f"{self.Rx_height}mm"
        self.M3D["Rx_space_x"] = f"{self.Rx_space_x}mm"
        self.M3D["Rx_space_y"] = f"{self.Rx_space_y}mm"
        self.M3D["g1"] = "0"
        self.M3D["g2"] = f"{self.g2}mm"
        self.M3D["Tx_layer_space_x"] = f"{self.Tx_layer_space_x}mm"
        self.M3D["Tx_layer_space_y"] = f"{self.Tx_layer_space_y}mm"
        self.M3D["litz_wire_diameter"] = f"{self.wire_diameter}mm"
        self.M3D["strand_number"] = f"{self.strand_number}mm"

        self.M3D["Tx_current"] = f"{self.Tx_current}A"
        self.M3D["Rx_current"] = f"{self.Rx_current}A"

    def set_analysis(self):
        setup = self.M3D.create_setup(name="Setup1")
        setup.props["MaximumPasses"] = 5
        setup.props["MinimumPasses"] = 2
        setup.props["PercentError"] = 2.5
        setup.props["Frequency"] = f"{self.freq}kHz"

    # Geometry/model creation methods copied from legacy implementation
    def create_core(self):
        origin = ["-(w1)/2*w1_ratio", "-(2*l1_leg+2*l2+l1_center)/2", "-(2*l1_top+h1)/2"]
        dimension = ["(w1)*w1_ratio", "(2*l1_leg+2*l2+l1_center)", "(2*l1_top+h1)"]
        self.core_base = self.M3D.modeler.create_box(position=origin, dimensions_list=dimension, name="core", matname=self.mat)

        origin = ["-(w1)/2*w1_ratio", "l1_center/2", "-(h1)/2"]
        dimension = ["w1", "l2", "h1"]
        self.core_sub1 = self.M3D.modeler.create_box(position=origin, dimensions_list=dimension, name="core_sub1", matname="ferrite")

        origin = ["-(w1)/2*w1_ratio", "-l1_center/2", "-(h1)/2"]
        dimension = ["w1", "-(l2)", "h1"]
        self.core_sub2 = self.M3D.modeler.create_box(position=origin, dimensions_list=dimension, name="core_sub2", matname="ferrite")

        origin = ["-(w1)/2*w1_ratio", "-l1_center/2", "-(h1)/2"]
        dimension = ["w1*w1_ratio", "l1_center", "h1"]
        self.core_sub3 = self.M3D.modeler.create_box(position=origin, dimensions_list=dimension, name="core_sub3", matname="ferrite")

        origin = ["-(w1)/2*w1_ratio", "-(2*l1_leg+2*l2+l1_center)/2", "-(h1)/2"]
        dimension = ["(w1)", "(l1_leg)", "(g1)"]
        self.core_sub_g1 = self.M3D.modeler.create_box(position=origin, dimensions_list=dimension, name="core_sub_g1", matname=self.mat)

        origin = ["-(w1)/2*w1_ratio", "(2*l1_leg+2*l2+l1_center)/2", "-(h1)/2"]
        dimension = ["(w1)", "-(l1_leg)", "(g1)"]
        self.core_sub_g2 = self.M3D.modeler.create_box(position=origin, dimensions_list=dimension, name="core_sub_g2", matname=self.mat)

        origin = ["-(w1)/2*w1_ratio", "-l1_center/2", "-(h1)/2"]
        dimension = ["w1*w1_ratio", "l1_center", "h1"]
        self.core_unite1 = self.M3D.modeler.create_box(position=origin, dimensions_list=dimension, name="core_unite1", matname="ferrite")

        origin = ["-(w1)/2*w1_ratio", "-l1_center/2", "-(h1)/2"]
        dimension = ["w1*w1_ratio", "l1_center", "g2"]
        self.core_unite_sub_g1 = self.M3D.modeler.create_box(position=origin, dimensions_list=dimension, name="core_sub_g3", matname="ferrite")

        blank_list = [self.core_base.name]
        tool_list = [self.core_sub1.name, self.core_sub2.name, self.core_sub3.name, self.core_sub_g1.name, self.core_sub_g2.name]
        self.M3D.modeler.subtract(blank_list=blank_list, tool_list=tool_list, keep_originals=False)

        blank_list = [self.core_unite1.name]
        tool_list = [self.core_unite_sub_g1.name]
        self.M3D.modeler.subtract(blank_list=blank_list, tool_list=tool_list, keep_originals=False)

        self.core_list = [self.core_base, self.core_unite1]
        self.core = self.M3D.modeler.unite(unite_list=self.core_list)
        self.core_base.transparency = 0.6

    def set_wire_material(self):
        self.w_mat = self.M3D.materials.duplicate_material(material_name="copper", new_name="copper_Litz_wire")
        self.w_mat.stacking_type = "Litz Wire"
        self.w_mat.wire_diameter = f"{self.wire_diameter}mm"
        self.w_mat.strand_number = self.strand_number

    def create_winding(self):
        self.terminal = self.Tx_space_x + self.Tx_width + (self.Tx_turns - 1) * (self.Tx_layer_space_x + self.Tx_width) + (self.Rx_space_x + self.Rx_width)
        self.temp = [
            [f"(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)", f"-(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "0mm"],
            [f"(w1*w1_ratio/2 + Rx_space_x + Rx_width/2 + Tx_space_x + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_x + Tx_width))", f"-(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "0mm"],
            [f"-(w1*w1_ratio/2 + Rx_space_x + Rx_width/2 + Tx_space_x + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_x + Tx_width))", f"-(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "0mm"],
            [f"-(w1*w1_ratio/2 + Rx_space_x + Rx_width/2 + Tx_space_x + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_x + Tx_width))", f"(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "0mm"],
            [f"(w1*w1_ratio/2 + Rx_space_x + Rx_width/2 + Tx_space_x + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_x + Tx_width))", f"(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "0mm"],
            [f"(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)", f"(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "0mm"],
        ]

        self.Rx_1 = self._create_polyline(points=self.temp, name="Rx_1", coil_width="Rx_width", coil_height="Rx_height")
        self.M3D.modeler.mirror(objid=self.Rx_1, position=[0, 0, 0], vector=[1, 0, 0])
        Rx_1_move = ["0mm", "0mm", "(Rx_preg/2+Rx_height/2)"]
        self.M3D.modeler.move(objid=self.Rx_1, vector=Rx_1_move)

        self.Rx_2 = self._create_polyline(points=self.temp, name="Rx_2", coil_width="Rx_width", coil_height="Rx_height")
        self.M3D.modeler.mirror(objid=self.Rx_2, position=[0, 0, 0], vector=[1, 0, 0])
        Rx_2_move = ["0mm", "0mm", "-(Rx_preg/2+Rx_height/2)"]
        self.M3D.modeler.move(objid=self.Rx_2, vector=Rx_2_move)

        self.temp_Tx = [
            [f"(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)", f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1)*(Tx_layer_space_y + Tx_width))", "(Rx_preg + Rx_height + Tx_preg + Tx_height)"],
            [f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2)", f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1)*(Tx_layer_space_y + Tx_width))", "(Rx_preg + Rx_height + Tx_preg + Tx_height)"],
            [f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2)", f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1)*(Tx_layer_space_y + Tx_width))", "0"],
        ]

        for i in range(0, math.ceil(self.Tx_turns)):
            self.temp_Tx.append([
                f"-(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}))*(Tx_layer_space_x + Tx_width))",
                f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-{i})*(Tx_layer_space_y + Tx_width))",
                "0",
            ])
            self.temp_Tx.append([
                f"-(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}))*(Tx_layer_space_x + Tx_width))",
                f"(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-{i})*(Tx_layer_space_y + Tx_width))",
                "0",
            ])
            self.temp_Tx.append([
                f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}))*(Tx_layer_space_x + Tx_width))",
                f"(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-{i})*(Tx_layer_space_y + Tx_width))",
                "0",
            ])
            if i == math.ceil(self.Tx_turns) - 1:
                self.temp_Tx.append([
                    f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}))*(Tx_layer_space_x + Tx_width))",
                    f"(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}))*(Tx_layer_space_y + Tx_width))",
                    "(Rx_preg + Rx_height + Tx_preg + Tx_height)",
                ])
                self.temp_Tx.append([
                    f"(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)",
                    f"(l1_center/2 + Tx_space_y + Tx_width/2)",
                    "(Rx_preg + Rx_height + Tx_preg + Tx_height)",
                ])
            else:
                self.temp_Tx.append([
                    f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}))*(Tx_layer_space_x + Tx_width))",
                    f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}+1))*(Tx_layer_space_y + Tx_width))",
                    "0",
                ])

        self.Tx_1 = self._create_polyline_litz(points=self.temp_Tx, name="Tx_1", coil_width="Tx_width")

        self.Rx_1.color = [0, 0, 255]
        self.Rx_1.transparency = 0
        self.Rx_2.color = [0, 0, 255]
        self.Rx_2.transparency = 0
        self.Tx_1.color = [255, 0, 0]
        self.Tx_1.transparency = 0

        origin = ["-(w1)/2*w1_ratio", "-(2*l1_leg+2*l2+l1_center)/2", "-(2*l1_top+h1)/2"]
        dimension = ["(w1)*w1_ratio", "(2*l1_leg+2*l2+l1_center)", "(2*l1_top+h1)"]
        self.air_box = self.M3D.modeler.create_box(position=origin, dimensions_list=dimension, name="air_box", matname="vacuum")

        blank_list = [self.air_box.name]
        tool_list = [self.core_base.name, self.Tx_1.name, self.Rx_1.name, self.Rx_2.name]
        self.M3D.modeler.subtract(blank_list=blank_list, tool_list=tool_list, keep_originals=True)

    def create_exctation(self):
        self.Tx_in = self.M3D.modeler.get_faceid_from_position(position=[
            f"(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)",
            f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1)*(Tx_layer_space_y + Tx_width))",
            "(Rx_preg + Rx_height + Tx_preg + Tx_height)",
        ])
        self.Tx_out = self.M3D.modeler.get_faceid_from_position(position=[
            f"(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)",
            f"(l1_center/2 + Tx_space_y + Tx_width/2)",
            "(Rx_preg + Rx_height + Tx_preg + Tx_height)",
        ])

        self.Rx2_in = self.M3D.modeler.get_faceid_from_position(position=[
            f"-(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)",
            f"-(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))",
            "-(Rx_preg/2+Rx_height/2)",
        ])
        self.Rx2_out = self.M3D.modeler.get_faceid_from_position(position=[
            f"-(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)",
            f"(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))",
            "-(Rx_preg/2+Rx_height/2)",
        ])

        self.Rx1_in = self.M3D.modeler.get_faceid_from_position(position=[
            f"-(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)",
            f"-(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))",
            "(Rx_preg/2+Rx_height/2)",
        ])
        self.Rx1_out = self.M3D.modeler.get_faceid_from_position(position=[
            f"-(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)",
            f"(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))",
            "(Rx_preg/2+Rx_height/2)",
        ])

        self.M3D.assign_coil(self.Tx_in, conductor_number=1, polarity="Positive", name="Tx_in")
        self.M3D.assign_coil(self.Tx_out, conductor_number=1, polarity="Negative", name="Tx_out")

        self.M3D.assign_coil(self.Rx1_in, conductor_number=1, polarity="Positive", name="Rx1_in")
        self.M3D.assign_coil(self.Rx1_out, conductor_number=1, polarity="Negative", name="Rx1_out")

        self.M3D.assign_coil(self.Rx2_in, conductor_number=1, polarity="Positive", name="Rx2_in")
        self.M3D.assign_coil(self.Rx2_out, conductor_number=1, polarity="Negative", name="Rx2_out")

        Tx_winding = self.M3D.assign_winding(coil_terminals=[], winding_type="Current", is_solid=True, current_value="Tx_current", name="Tx")
        Rx_winding = self.M3D.assign_winding(coil_terminals=[], winding_type="Current", is_solid=True, current_value="Rx_current", name="Rx1")
        Rx_winding2 = self.M3D.assign_winding(coil_terminals=[], winding_type="Current", is_solid=True, current_value=0, name="Rx2")

        self.M3D.add_winding_coils(Tx_winding.name, coil_names=["Tx_in", "Tx_out"])
        self.M3D.add_winding_coils(Rx_winding.name, coil_names=["Rx1_in", "Rx1_out"])
        self.M3D.add_winding_coils(Rx_winding2.name, coil_names=["Rx2_in", "Rx2_out"])
        self.M3D.assign_matrix(sources=[Tx_winding.name, Rx_winding.name, Rx_winding2.name], matrix_name="Matrix1")

        self.M3D.parametrics.add(variable="Rx_current", start_point=self.Rx_current / 5, end_point=self.Rx_current, step=self.Rx_current / 5, variation_type="LinearStep", name="Rx_sweep")

    def _create_polyline(self, points, name, coil_width, coil_height, orient=None):
        extra = {}
        if orient is not None:
            extra["xsection_orient"] = orient

        polyline_obj = self.M3D.modeler.create_polyline(
            points,
            name=name,
            material="copper",
            xsection_type="Rectangle",
            xsection_width=coil_width,
            xsection_height=coil_height,
            **extra,
        )
        return polyline_obj

    def _create_polyline_litz(self, points, name, coil_width):
        polyline_obj = self.M3D.modeler.create_polyline(
            points,
            name=name,
            matname="copper_Litz_wire",
            xsection_type="Circle",
            xsection_width=coil_width,
            xsection_num_seg=6,
        )
        return polyline_obj

    def _get_mean_Bfield(self, obj):
        assignment = obj.name
        oModule = self.M3D.ofieldsreporter
        oModule.CalcStack("clear")
        oModule.CopyNamedExprToStack("Mag_B")
        oModule.EnterVol(assignment) if obj.is3d else oModule.EnterSurf(assignment)
        oModule.CalcOp("Mean")
        name = f"B_mean_{assignment}"
        oModule.AddNamedExpression(name, "Fields")
        return name

    def _create_B_field(self):
        self.leg_left = self.M3D.modeler.create_rectangle(orientation="XY", origin=['-(w1)/2', '-(2*l1_leg+2*l2+l1_center)/2', 'g1/2'], sizes=['w1', 'l1_leg'])
        self.leg_left.model = False
        self.leg_left.name = "leg_left"

        self.leg_center = self.M3D.modeler.create_rectangle(orientation="XY", origin=['-(w1)/2', '-l1_center/2', 'g2/2'], sizes=['w1', 'l1_center'])
        self.leg_center.model = False
        self.leg_center.name = "leg_center"

        self.leg_right = self.M3D.modeler.create_rectangle(orientation="XY", origin=['-(w1)/2', '(2*l1_leg+2*l2+l1_center)/2', 'g1/2'], sizes=['w1', '-l1_leg'])
        self.leg_right.model = False
        self.leg_right.name = "leg_right"

        self.leg_top_left = self.M3D.modeler.create_rectangle(orientation="XZ", origin=['-(w1)/2', '-(l1_center+l2)/2', 'h1/2+l1_top'], sizes=['-l1_top', 'w1'])
        self.leg_top_left.model = False
        self.leg_top_left.name = "leg_top_left"

        self.leg_top_right = self.M3D.modeler.create_rectangle(orientation="XZ", origin=['-(w1)/2', '(l1_center+l2)/2', 'h1/2+l1_top'], sizes=['-l1_top', 'w1'])
        self.leg_top_right.model = False
        self.leg_top_right.name = "leg_top_right"

        self.leg_bottom_left = self.M3D.modeler.create_rectangle(orientation="XZ", origin=['-(w1)/2', '-(l1_center+l2)/2', '-(h1/2+l1_top)'], sizes=['l1_top', 'w1'])
        self.leg_bottom_left.model = False
        self.leg_bottom_left.name = "leg_bottom_left"

        self.leg_bottom_right = self.M3D.modeler.create_rectangle(orientation="XZ", origin=['-(w1)/2', '(l1_center+l2)/2', '-(h1/2+l1_top)'], sizes=['l1_top', 'w1'])
        self.leg_bottom_right.model = False
        self.leg_bottom_right.name = "leg_bottom_right"

    def _get_B_field(self):
        parameters2 = [
            [self.core_base, "B_mean", "B_mean_core"],
            [self.leg_left, "B_mean", "B_mean_leg_left"],
            [self.leg_right, "B_mean", "B_mean_leg_right"],
            [self.leg_center, "B_mean", "B_mean_leg_center"],
            [self.leg_top_left, "B_mean", "B_mean_leg_top_left"],
            [self.leg_bottom_left, "B_mean", "B_mean_leg_bottom_left"],
            [self.leg_top_right, "B_mean", "B_mean_leg_top_right"],
            [self.leg_bottom_right, "B_mean", "B_mean_leg_bottom_right"],
        ]

        self.result_expressions = []
        self.name_list = []
        self.report_list = {}
        for obj, expression, name in parameters2:
            if expression == "B_mean":
                self.result_expressions.append(self._get_mean_Bfield(obj))
            self.name_list.append(name)

    def create_region(self):
        region = self.M3D.modeler.create_air_region(z_pos="800", z_neg="800", y_pos="300", y_neg="300", x_pos="0", x_neg="0")
        self.M3D.assign_material(obj=region, mat="vacuum")
        region_face = self.M3D.modeler.get_object_faces("Region")
        self.M3D.assign_radiation(assignment=region_face, radiation="Radiation")

    def assign_mesh(self):
        temp_list = ["Tx_1"]
        skin_depth = f"{math.sqrt(1.7*10**(-8)/math.pi/80/10**3/0.999991/4/math.pi/10**(-7))*10**3}mm"
        self.M3D.mesh.assign_skin_depth(assignment=temp_list, skin_depth=skin_depth, triangulation_max_length="12.2mm")

        air_list = ["air_box"]
        self.M3D.mesh.assign_length_mesh(assignment=air_list, maximum_length="20mm")

    def _close_project(self):
        solution_dir = os.path.join(self.dir, f"script{self.num}.aedtresults")
        aedt_dir = os.path.join(self.dir, f"script{self.num}.aedt")
        if os.path.isdir(aedt_dir):
            shutil.rmtree(aedt_dir)
        self.M3D.close_project()
        self.desktop.release_desktop()

    def _get_magnetic_report(self):
        get_result_list = [
            ["Matrix1.L(Tx,Tx)", "Ltx"],
            ["Matrix1.L(Rx1,Rx1)", "Lrx1"],
            ["Matrix1.L(Rx2,Rx2)", "Lrx2"],
            ["Matrix1.L(Tx,Rx1)", "M1"],
            ["Matrix1.L(Tx,Rx2)", "M2"],
            ["Matrix1.CplCoef(Tx,Rx1)", "k1"],
            ["Matrix1.CplCoef(Tx,Rx2)", "k2"],
            ["Matrix1.L(Tx,Tx)*(Matrix1.CplCoef(Tx,Rx1)^2)", "Lmt"],
            ["Matrix1.L(Rx1,Rx1)*(Matrix1.CplCoef(Tx,Rx1)^2)", "Lmr1"],
            ["Matrix1.L(Rx2,Rx2)*(Matrix1.CplCoef(Tx,Rx2)^2)", "Lmr2"],
            ["Matrix1.L(Tx,Tx)*(1-Matrix1.CplCoef(Tx,Rx1)^2)", "Llt"],
            ["Matrix1.L(Rx1,Rx1)*(1-Matrix1.CplCoef(Tx,Rx1)^2)", "Llr1"],
            ["Matrix1.L(Rx2,Rx2)*(1-Matrix1.CplCoef(Tx,Rx2)^2)", "Llr2"],
            ["Matrix1.R(Tx,Tx)", "Rtx"],
            ["Matrix1.R(Rx1,Rx1)", "Rrx1"],
            ["Matrix1.R(Rx2,Rx2)", "Rrx2"],
        ]

        result_expressions = [item[0] for item in get_result_list]
        report = self.M3D.post.create_report(
            expressions=result_expressions,
            setup_sweep_name=None,
            domain="Sweep",
            variations={"Freq": ["All"]},
            primary_sweep_variable=None,
            secondary_sweep_variable=None,
            report_category=None,
            plot_type="Data Table",
            context=None,
            subdesign_id=None,
            polyline_points=1001,
            plotname="simulation parameter",
        )
        dir_data = self.M3D.post.export_report_to_csv(project_dir=self.dir, plot_name=report.plot_name)

        self.data1 = pd.read_csv(dir_data)
        if "Freq" in self.data1.columns[0]:
            self.data1.drop(columns=self.data1.columns[0], inplace=True)

        factor = {"ph": 1e-6, "nh": 1e-3, "uh": 1, "mh": 1e3, "h": 1e6, "mohm": 1e-3, "ohm": 1, "": 1}
        new_names = {}
        for col in self.data1.columns:
            col_main = col.split(" -", 1)[0].strip()
            unit = ""
            if "[" in col_main and "]" in col_main:
                unit = col_main[col_main.find("[") + 1 : col_main.find("]")].lower()
            if unit in factor:
                self.data1[col] = self.data1[col] * factor[unit]
            base = col_main.split("[")[0].strip()
            new_names[col] = base
        self.data1.rename(columns=new_names, inplace=True)
        self.data1.columns = [
            "Ltx",
            "Lrx1",
            "Lrx2",
            "M1",
            "M2",
            "k1",
            "k2",
            "Lmt",
            "Lmr1",
            "Lmr2",
            "Llt",
            "Llr1",
            "Llr2",
            "Rtx",
            "Rrx1",
            "Rrx2",
        ]
        self.Lmt = self.data1.loc[0, "Lmt"]
        rows = 5
        self.data1 = pd.DataFrame(np.repeat(self.data1.values, rows, axis=0), columns=self.data1.columns)

    def _get_copper_loss_parameter(self):
        Tx = self.M3D.modeler.get_object_from_name(objname="Tx_1")
        Rx1 = self.M3D.modeler.get_object_from_name(objname="Rx_1")
        Rx2 = self.M3D.modeler.get_object_from_name(objname="Rx_2")

        get_result_list = [["PerWindingSolidLoss(Tx)", "copperloss_Tx"], ["PerWindingSolidLoss(Rx1)", "copperloss_Rx1"], ["PerWindingSolidLoss(Rx2)", "copperloss_Rx2"]]
        result_expressions = [item[0] for item in get_result_list]

        report = self.M3D.post.create_report(
            expressions=result_expressions,
            setup_sweep_name=None,
            domain="Sweep",
            variations={"Rx_current": ["All"]},
            primary_sweep_variable=None,
            secondary_sweep_variable=None,
            report_category=None,
            plot_type="Data Table",
            context=None,
            subdesign_id=None,
            polyline_points=1001,
            plotname="copper loss data",
        )
        dir_data = self.M3D.post.export_report_to_csv(project_dir=self.dir, plot_name=report.plot_name)
        self.data2 = pd.read_csv(dir_data)

        for column_name in list(self.data2.columns):
            self.data2[column_name] = abs(self.data2[column_name])
            if (
                "PerWindingSolidLoss(Tx)" not in column_name
                and "PerWindingSolidLoss(Rx1)" not in column_name
                and "PerWindingSolidLoss(Rx2)" not in column_name
                and "Rx_current" not in column_name
            ):
                self.data2 = self.data2.drop(columns=column_name)
            if "[mW]" in column_name:
                self.data2[column_name] = self.data2[column_name] * 1e-3
            elif "[W]" in column_name:
                self.data2[column_name] = self.data2[column_name] * 1e0
            elif "[kW]" in column_name:
                self.data2[column_name] = self.data2[column_name] * 1e3

        self.data2.columns = ["Rx_current_optimetric", "copperloss_Tx", "copperloss_Rx1", "copperloss_Rx2"]

    def get_input_parameter(self):
        input_parameter_array = np.array(
            [
                self.freq,
                self.input_voltage,
                self.w1,
                self.l1_leg,
                self.l1_top,
                self.l2,
                self.h1,
                self.l1_center,
                self.Tx_turns,
                self.Tx_width,
                self.Tx_height,
                self.Tx_space_x,
                self.Tx_space_y,
                self.Tx_preg,
                self.Rx_width,
                self.Rx_height,
                self.Rx_space_x,
                self.Rx_space_y,
                self.Rx_preg,
                self.g2,
                self.Tx_layer_space_x,
                self.Tx_layer_space_y,
                self.wire_diameter,
                self.strand_number,
                self.Tx_current,
            ]
        )
        input_parameter_array = input_parameter_array.reshape(1, len(input_parameter_array))

        input_parameter_columns = [
            "freq",
            "input_voltage",
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
        ]

        self.input_parameter = pd.DataFrame(data=input_parameter_array, columns=input_parameter_columns)
        rows = 5
        self.input_parameter = pd.DataFrame(np.repeat(self.input_parameter.values, rows, axis=0), columns=self.input_parameter.columns)

    def set_coreloss(self):
        self.M3D.set_core_losses(assignment="core", core_loss_on_field=True)
        self._create_B_field()

    def write_data(self):
        self.new_data = pd.concat([self.input_parameter, self.data1.round(4), self.data2.round(4), self.data3.round(4), self.data4.round(4), self.data5.round(5)], axis=1)
        current_dir = os.getcwd()
        csv_file = os.path.join(current_dir, "output_data.csv")
        if os.path.isfile(csv_file):
            self.new_data.to_csv(csv_file, mode="a", index=False, header=False)
        else:
            self.new_data.to_csv(csv_file, mode="w", index=False, header=True)

    def coreloss_project(self):
        self.M3D.duplicate_design(label=f"script1_{self.itr}_coreloss")
        self.M3D.set_active_design(name=f"script1_{self.itr}_coreloss")
        self.M3D.parametrics.delete(name="Rx_sweep")
        to_delete = [exc for exc in self.M3D.design_excitations.values() if exc.name in ["Tx", "Rx1", "Rx2"]]
        for exc in to_delete:
            exc.delete()

        self.magnetizing_current = self.input_voltage * math.sqrt(2) / 2 / math.pi / (self.freq * 10 ** (3)) / self.Lmt / 10 ** (-6)
        self.M3D["magnetizing_current"] = f"{self.magnetizing_current}A"

        Tx_winding = self.M3D.assign_winding(coil_terminals=[], winding_type="Current", is_solid=True, current_value="magnetizing_current", name="Tx")
        Rx_winding = self.M3D.assign_winding(coil_terminals=[], winding_type="Current", is_solid=True, current_value=0, name="Rx1")
        Rx_winding2 = self.M3D.assign_winding(coil_terminals=[], winding_type="Current", is_solid=True, current_value=0, name="Rx2")

        self.M3D.add_winding_coils(Tx_winding.name, coil_names=["Tx_in", "Tx_out"])
        self.M3D.add_winding_coils(Rx_winding.name, coil_names=["Rx1_in", "Rx1_out"])
        self.M3D.add_winding_coils(Rx_winding2.name, coil_names=["Rx2_in", "Rx2_out"])
        self.M3D.assign_matrix(sources=[Tx_winding.name, Rx_winding.name, Rx_winding2.name], matrix_name="Matrix1")

        self.M3D.parametrics.add(variable="magnetizing_current", start_point=self.magnetizing_current / 5, end_point=self.magnetizing_current, step=self.magnetizing_current / 5, variation_type="LinearStep", name="magnetizing_sweep")

        self.M3D.set_core_losses(objects="core", value=True)
        self._create_B_field()
        self.M3D.analyze()

        get_result_list_coreloss = ["Coreloss"]
        report = self.M3D.post.create_report(
            expressions=get_result_list_coreloss,
            setup_sweep_name=None,
            domain="Sweep",
            variations={"magnetizing_current": ["All"]},
            primary_sweep_variable=None,
            secondary_sweep_variable=None,
            report_category=None,
            plot_type="Data Table",
            context=None,
            subdesign_id=None,
            polyline_points=1001,
            plotname="coreloss parameter",
        )
        dir_data = self.M3D.post.export_report_to_csv(project_dir=self.dir, plot_name=report.plot_name)
        self.data3 = pd.read_csv(dir_data)

        for column_name in list(self.data3.columns):
            self.data3[column_name] = abs(self.data3[column_name])
            if "Coreloss" not in column_name and "magnetizing_current" not in column_name:
                self.data3 = self.data3.drop(columns=column_name)
            if "[mW]" in column_name:
                self.data3[column_name] = self.data3[column_name] * 1e-3
            elif "[W]" in column_name:
                self.data3[column_name] = self.data3[column_name] * 1e0
            elif "[kW]" in column_name:
                self.data3[column_name] = self.data3[column_name] * 1e3
        self.data3.columns = ["magnetizing_current_optimetric", "coreloss"]

        parameters2 = [
            [self.core_base, "B_mean", "B_mean_core"],
            [self.leg_left, "B_mean", "B_mean_leg_left"],
            [self.leg_right, "B_mean", "B_mean_leg_right"],
            [self.leg_center, "B_mean", "B_mean_leg_center"],
            [self.leg_top_left, "B_mean", "B_mean_leg_top_left"],
            [self.leg_bottom_left, "B_mean", "B_mean_leg_bottom_left"],
            [self.leg_top_right, "B_mean", "B_mean_leg_top_right"],
            [self.leg_bottom_right, "B_mean", "B_mean_leg_bottom_right"],
        ]

        self.result_expressions = []
        self.name_list = []
        self.report_list = {}
        for obj, expression, name in parameters2:
            if expression == "B_mean":
                self.result_expressions.append(self._get_mean_Bfield(obj))
            self.name_list.append(name)

        report = self.M3D.post.create_report(
            expressions=self.result_expressions,
            setup_sweep_name=None,
            domain="Sweep",
            variations={"Phase": ["0deg"]},
            primary_sweep_variable=None,
            secondary_sweep_variable=None,
            report_category="Fields",
            plot_type="Data Table",
            context=None,
            subdesign_id=None,
            polyline_points=1001,
            plotname="calculator_report",
        )
        export_data = self.M3D.post.export_report_to_csv(project_dir=self.dir, plot_name=report.plot_name)
        self.data4 = pd.read_csv(export_data, skiprows=1, header=None)

        self.data4 = self.data4.iloc[:, [3, 5, 7, 9, 11, 13, 15, 17]]
        self.data4.columns = ["B_core", "B_left", "B_right", "B_center", "B_top_left", "B_bottom_left", "B_top_right", "B_bottom_right"]
        rows = 5
        self.data4 = pd.DataFrame(np.repeat(self.data4.values, rows, axis=0), columns=self.data4.columns)

        data5_result = ["PerWindingSolidLoss(Tx)", "PerWindingSolidLoss(Rx1)", "PerWindingSolidLoss(Rx2)"]
        report = self.M3D.post.create_report(
            expressions=data5_result,
            setup_sweep_name=None,
            domain="Sweep",
            variations={"magnetizing_current": ["All"]},
            primary_sweep_variable=None,
            secondary_sweep_variable=None,
            report_category=None,
            plot_type="Data Table",
            context=None,
            subdesign_id=None,
            polyline_points=1001,
            plotname="magnetizing copper loss data",
        )
        dir_data = self.M3D.post.export_report_to_csv(project_dir=self.dir, plot_name=report.plot_name)
        self.data5 = pd.read_csv(dir_data)

        for column_name in list(self.data5.columns):
            self.data5[column_name] = abs(self.data5[column_name])
            if (
                "PerWindingSolidLoss(Tx)" not in column_name
                and "PerWindingSolidLoss(Rx1)" not in column_name
                and "PerWindingSolidLoss(Rx2)" not in column_name
                and "Rx_current" not in column_name
            ):
                self.data5 = self.data5.drop(columns=column_name)
            if "[mW]" in column_name:
                self.data5[column_name] = self.data5[column_name] * 1e-3
            elif "[W]" in column_name:
                self.data5[column_name] = self.data5[column_name] * 1e0
            elif "[kW]" in column_name:
                self.data5[column_name] = self.data5[column_name] * 1e3

        self.data5.columns = ["magnetizing_copperloss_Tx", "magnetizing_copperloss_Rx1", "magnetizing_copperloss_Rx2"]

    def data_remove(self):
        file_path = os.path.join("script", f"script{self.num}", "simulation parameter.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
        file_path = os.path.join("script", f"script{self.num}", "copper loss data.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
        file_path = os.path.join("script", f"script{self.num}", "coreloss parameter.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
        file_path = os.path.join("script", f"script{self.num}", "calculator_report.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
        file_path = os.path.join("script", f"script{self.num}", "magnetizing copper loss data.csv")
        if os.path.exists(file_path):
            os.remove(file_path)

        self.data1 = []
        self.data2 = []
        self.data3 = []
        self.data4 = []
        self.data5 = []
        self.new_data = []
        self.input_parameter = []

    def simulation(self,run_simulation=True, use_random: bool = True, input_parameters: SimulationInputParameters | None = None):
        self.start_time = time.time()
        file_path = "simulation_num.txt"

        if not use_random and input_parameters is None:
            raise ValueError("input_parameters must be provided when use_random is False.")

        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as file:
                file.write("1")

        with open(file_path, "r+", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            content = int(file.read().strip())
            self.num = content
            self.PROJECT_NAME = f"simulation{content}"
            content += 1
            file.seek(0)
            file.truncate()
            file.write(str(content))

        log_simulation(number=self.num, pid=self.desktop.aedt_process_id)

        self.create_project()
        if use_random:
            self.get_random_variable()
        else:
            self.apply_input_parameters(input_parameters)
        self.set_variable()
        self.set_analysis()
        self.set_material()
        self.set_wire_material()

        self.create_core()
        self.create_winding()
        self.assign_mesh()
        self.create_region()

        self.create_exctation()
        if run_simulation:
            self.M3D.analyze()

            self._get_magnetic_report()
            self._get_copper_loss_parameter()
            self.coreloss_project()
            self.get_input_parameter()
            self.write_data()
        
        self.desktop.release_desktop(close_projects=False,close_on_exit=False)



def loging(msg):
    file_path = "log.txt"
    max_attempts = 5
    attempt = 0
    while attempt < max_attempts:
        try:
            with open(file_path, "a", encoding="utf-8") as file:
                file.write(msg + "\n")
            break
        except Exception:
            attempt += 1
            time.sleep(1)


def safe_open(filename, mode, retries=5, delay=1):
    for i in range(retries):
        try:
            return open(filename, mode, newline="")
        except (IOError, OSError) as e:
            if i == retries - 1:
                raise e
            time.sleep(delay)


def log_simulation(number, state=None, pid=None, filename="log.csv"):
    lock_timeout = 10
    if not os.path.exists(filename):
        with portalocker.Lock(filename, "w", timeout=lock_timeout, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Number", "Status", "StartTime", "PID"])

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
        new_status = "Error" if state.lower() == "fail" else "Finished"
        with portalocker.Lock(filename, "r+", timeout=lock_timeout, newline="") as f:
            rows = list(csv.reader(f))
            updated_rows = []
            for row in rows:
                if row and row[0] == str(number) and row[1] == "Simulation":
                    row[1] = new_status
                updated_rows.append(row)
            f.seek(0)
            writer = csv.writer(f)
            writer.writerows(updated_rows)
            f.truncate()


def save_error_log(project_name, error_info):
    error_folder = "error"
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)
    error_file = os.path.join(error_folder, f"{project_name}_error.txt")
    with open(error_file, "w", encoding="utf-8") as f:
        f.write(error_info)


def run_simulation(run_simulation: bool = True, use_random: bool = True, input_parameters: SimulationInputParameters | None = None):
    sim = Sim()
    project_dir = None

    try:
        
        sim.simulation(run_simulation=run_simulation, use_random=use_random, input_parameters=input_parameters)
        project_dir = getattr(sim, "dir", None)
        end_time = time.time()
        execution_time = end_time - getattr(sim, "start_time", end_time)
        log_simulation(number=sim.num, state="finished")
        loging(f"{sim.PROJECT_NAME} : simulation success!! ({execution_time:.1f} sec)")
    except Exception as e:
        err_info = f"error : {getattr(sim, 'PROJECT_NAME', 'simulation')}(litz)\n"
        err_info += f"{str(e)}\n"
        err_info += traceback.format_exc()
        print(err_info, file=sys.stderr)
        sys.stderr.flush()
        logging.error(err_info, exc_info=True)
        save_error_log(getattr(sim, "PROJECT_NAME", "simulation"), err_info)
        log_simulation(number=getattr(sim, "num", -1), state="fail")
        loging(f"{getattr(sim, 'PROJECT_NAME', 'simulation')} : simulation Failed")
    finally:
        try:
            if hasattr(sim, "desktop"):
                sim.desktop.release_desktop()
        except Exception:
            pass

    return project_dir


if __name__ == "__main__":
    example_params_list = [
        SimulationInputParameters(
            freq=150.0,
            w1=32.0,
            l1_leg=2.0,
            l1_top=1.2000000000000002,
            l2=7.0,
            h1=2.1,
            l1_center=2.0,
            Tx_turns=15.0,
            Tx_width=0.187082869338697,
            Tx_height=0.187082869338697,
            Tx_space_x=0.1,
            Tx_space_y=0.1,
            Tx_preg=0.07,
            Rx_width=4.0,
            Rx_height=0.14,
            Rx_space_x=0.1,
            Rx_space_y=1.0,
            Rx_preg=0.22,
            g2=0.30,
            Tx_layer_space_x=0.2,
            Tx_layer_space_y=0.2,
            wire_diameter=0.05,
            strand_number=7.0,
            Tx_current=5.656854249492381,
            Rx_current=15.556349186104049,
            input_voltage=400.0,
        ),
        SimulationInputParameters(
            freq=150.0,
            w1=32.0,
            l1_leg=2.0,
            l1_top=1.3,
            l2=7.2,
            h1=1.95,
            l1_center=2.0,
            Tx_turns=15.0,
            Tx_width=0.2939387691339813,
            Tx_height=0.2939387691339813,
            Tx_space_x=0.1,
            Tx_space_y=0.1,
            Tx_preg=0.07,
            Rx_width=4.0,
            Rx_height=0.14,
            Rx_space_x=0.1,
            Rx_space_y=0.8,
            Rx_preg=0.22,
            g2=0.29,
            Tx_layer_space_x=0.2,
            Tx_layer_space_y=0.2,
            wire_diameter=0.06,
            strand_number=12.0,
            Tx_current=5.656854249492381,
            Rx_current=15.556349186104049,
            input_voltage=400.0,
        ),
    ]

    selected_idx = np.random.randint(0, len(example_params_list))
    run_simulation(run_simulation=True, use_random=False, input_parameters=example_params_list[selected_idx])
