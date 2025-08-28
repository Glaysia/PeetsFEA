
from ansys.aedt.core.modeler.cad.polylines import Polyline
from evdd_params import Parameter
from ansys.aedt.core import Desktop, Maxwell3d

from peetsfea.util import user_host

import math

import pandas as pd
import numpy as np


class Simul(Parameter):

    def __init__(self, non_graphical=False):
        # super().get_random_variable() # parameter overide
        self.project_name = "evdd_simul_25.8"
        self.flag = 1
        self.Proj = 0
        self.itr = 0
        self.freq_kHz = 105

        self.computer_name = user_host
        self.data: dict[str, pd.Series] = {}
        self.create_desktop(version="2024.2", non_graphical=non_graphical)

    def set_variable(self):

        self.M3D["w1"] = f'{self.w1}mm'
        self.M3D["l1_leg"] = f'{self.l1_leg}mm'
        self.M3D["l1_top"] = f'{self.l1_top}mm'
        self.M3D["l2"] = f'{self.l2}mm'
        self.M3D["h1"] = f'{self.h1}mm'
        self.M3D["w1_ratio"] = f'1'
        self.M3D["l1_center"] = f'{self.l1_center}mm'
        self.M3D["ratio"] = f'1'
        self.M3D["Tx_turns"] = f'{self.Tx_turns}'
        self.M3D["Rx_turns"] = f'{1}'
        self.M3D["Tx_width"] = f'{self.Tx_width}mm'
        self.M3D["Tx_height"] = f'{self.Tx_height}mm'
        self.M3D["Tx_space_x"] = f'{self.Tx_space_x}mm'
        self.M3D["Tx_space_y"] = f'{self.Tx_space_y}mm'
        self.M3D["Tx_preg"] = f'{self.Tx_preg}mm'
        self.M3D["Rx_width"] = f'{self.Rx_width}mm'
        self.M3D["Rx_preg"] = f'{self.Rx_preg}mm'
        self.M3D["Rx_height"] = f'{self.Rx_height}mm'
        self.M3D["Rx_space_x"] = f'{self.Rx_space_x}mm'
        self.M3D["Rx_space_y"] = f'{self.Rx_space_y}mm'
        self.M3D["g1"] = f'{self.g1}mm'
        self.M3D["g2"] = f'{self.g2}mm'
        self.M3D["Tx_layer_space_x"] = f'{self.Tx_layer_space_x}mm'
        self.M3D["Tx_layer_space_y"] = f'{self.Tx_layer_space_y}mm'
        self.M3D["litz_wire_diameter"] = f'{self.wire_diameter}mm'
        self.M3D["strand_number"] = f'{self.strand_number}mm'

    def set_wire_material(self):
        self.w_mat = self.M3D.materials.duplicate_material(  # type: ignore
            material="copper", name="copper_Litz_wire")
        assert self.w_mat

        self.w_mat.stacking_type = "Litz Wire"
        self.w_mat.wire_diameter = f"{self.wire_diameter}mm"
        self.w_mat.strand_number = self.strand_number

    def create_core(self):

        # make core (main part)
        origin = ["-(w1)/2*w1_ratio",
                  "-(2*l1_leg+2*l2+l1_center)/2",  "-(2*l1_top+h1)/2"]

        dimension = ["(w1)*w1_ratio",
                     "(2*l1_leg+2*l2+l1_center)", "(2*l1_top+h1)"]

        self.core_base = self.M3D.modeler.create_box(
            origin=origin,
            sizes=dimension,
            name="core",
            material=self.mat
        )

        origin = ["-(w1)/2*w1_ratio", "l1_center/2", "-(h1)/2"]

        dimension = ["w1", "l2", "h1"]

        self.core_sub1 = self.M3D.modeler.create_box(
            origin=origin,
            sizes=dimension,
            name="core_sub1",
            material="ferrite"
        )

        origin = ["-(w1)/2*w1_ratio", "-l1_center/2", "-(h1)/2"]

        dimension = ["w1", "-(l2)", "h1"]

        self.core_sub2 = self.M3D.modeler.create_box(
            origin=origin,
            sizes=dimension,
            name="core_sub2",
            material="ferrite"
        )

        origin = ["-(w1)/2*w1_ratio", "-l1_center/2", "-(h1)/2"]

        dimension = ["w1", "l1_center", "h1"]

        self.core_sub3 = self.M3D.modeler.create_box(
            origin=origin,
            sizes=dimension,
            name="core_sub3",
            material="ferrite"
        )

        origin = ["-(w1)/2*w1_ratio",
                  "-(2*l1_leg+2*l2+l1_center)/2", "-(h1)/2"]

        dimension = ["(w1)", "(l1_leg)", "(g1)"]

        self.core_sub_g1 = self.M3D.modeler.create_box(
            origin=origin,
            sizes=dimension,
            name="core_sub_g1",
            material=self.mat
        )

        origin = ["-(w1)/2*w1_ratio", "(2*l1_leg+2*l2+l1_center)/2", "-(h1)/2"]

        dimension = ["(w1)", "-(l1_leg)", "(g1)"]

        self.core_sub_g2 = self.M3D.modeler.create_box(
            origin=origin,
            sizes=dimension,
            name="core_sub_g2",
            material=self.mat
        )

        origin = ["-(w1)/2*w1_ratio", "-l1_center/2", "-(h1)/2"]

        dimension = ["w1*w1_ratio", "l1_center", "h1"]

        self.core_unite1 = self.M3D.modeler.create_box(
            origin=origin,
            sizes=dimension,
            name="core_unite1",
            material="ferrite"
        )

        origin = ["-(w1)/2*w1_ratio", "-l1_center/2", "-(h1)/2"]

        dimension = ["w1*w1_ratio", "l1_center", "g2"]

        self.core_unite_sub_g1 = self.M3D.modeler.create_box(
            origin=origin,
            sizes=dimension,
            name="core_sub_g3",
            material="ferrite"
        )

        # subtract core
        blank_list = [self.core_base.name]
        tool_list = [self.core_sub1.name,
                     self.core_sub2.name,
                     self.core_sub3.name,
                     self.core_sub_g1.name,
                     self.core_sub_g2.name]

        self.M3D.modeler.subtract(
            blank_list=blank_list,
            tool_list=tool_list,
            keep_originals=False
        )

        # subtract core
        blank_list = [self.core_unite1.name]
        tool_list = [self.core_unite_sub_g1.name]

        self.M3D.modeler.subtract(
            blank_list=blank_list,
            tool_list=tool_list,
            keep_originals=False
        )

        self.core_list = [self.core_base, self.core_unite1]

        self.core = self.M3D.modeler.unite(unite_list=self.core_list)

        self.core_base.transparency = 0.6

    def create_winding(self):

        self.terminal = (self.Tx_space_x + self.Tx_width + (self.Tx_turns-1) *
                         (self.Tx_layer_space_x+self.Tx_width)+(self.Rx_space_x + self.Rx_width))
        self.temp = [[f"(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)", f"-(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "0mm"],
                     [f"(w1*w1_ratio/2 + Rx_space_x + Rx_width/2 + Tx_space_x + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_x + Tx_width))",
                         f"-(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "0mm"],
                     [f"-(w1*w1_ratio/2 + Rx_space_x + Rx_width/2 + Tx_space_x + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_x + Tx_width))",
                      f"-(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "0mm"],
                     [f"-(w1*w1_ratio/2 + Rx_space_x + Rx_width/2 + Tx_space_x + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_x + Tx_width))",
                      f"(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "0mm"],
                     [f"(w1*w1_ratio/2 + Rx_space_x + Rx_width/2 + Tx_space_x + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_x + Tx_width))",
                         f"(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "0mm"],
                     [f"(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)", f"(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "0mm"],]

        self.Rx_1 = self._create_polyline(
            points=self.temp, name=f"Rx_1", coil_width="Rx_width", coil_height="Rx_height")
        self.M3D.modeler.mirror(assignment=self.Rx_1, origin=[
                                0, 0, 0], vector=[1, 0, 0])
        Rx_1_move = ["0mm", "0mm", "(Rx_preg/2+Rx_height/2)"]
        self.M3D.modeler.move(assignment=self.Rx_1, vector=Rx_1_move)

        self.Rx_2 = self._create_polyline(
            points=self.temp, name=f"Rx_2", coil_width="Rx_width", coil_height="Rx_height")
        self.M3D.modeler.mirror(assignment=self.Rx_2, origin=[
                                0, 0, 0], vector=[1, 0, 0])
        Rx_2_move = ["0mm", "0mm", "-(Rx_preg/2+Rx_height/2)"]
        self.M3D.modeler.move(assignment=self.Rx_2, vector=Rx_2_move)

        self.temp_Tx = [[f"(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)", f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1)*(Tx_layer_space_y + Tx_width))", "(Rx_preg + Rx_height + Tx_preg + Tx_height)"],
                        [f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2)",
                            f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1)*(Tx_layer_space_y + Tx_width))", "(Rx_preg + Rx_height + Tx_preg + Tx_height)"],
                        [f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2)", f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1)*(Tx_layer_space_y + Tx_width))", f"0"]]

        for i in range(0, math.ceil(self.Tx_turns)):
            self.temp_Tx.append([f"-(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}))*(Tx_layer_space_x + Tx_width))",
                                f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-{i})*(Tx_layer_space_y + Tx_width))", f"0"])
            self.temp_Tx.append([f"-(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}))*(Tx_layer_space_x + Tx_width))",
                                f"(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-{i})*(Tx_layer_space_y + Tx_width))", f"0"])
            self.temp_Tx.append([f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}))*(Tx_layer_space_x + Tx_width))",
                                f"(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-{i})*(Tx_layer_space_y + Tx_width))", f"0"])
            if i == math.ceil(self.Tx_turns) - 1:
                self.temp_Tx.append([f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}))*(Tx_layer_space_x + Tx_width))",
                                    f"(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}))*(Tx_layer_space_y + Tx_width))", f"(Rx_preg + Rx_height + Tx_preg + Tx_height)"])
                self.temp_Tx.append([f"(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)",
                                    f"(l1_center/2 + Tx_space_y + Tx_width/2)", f"(Rx_preg + Rx_height + Tx_preg + Tx_height)"])
            else:
                self.temp_Tx.append([f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}))*(Tx_layer_space_x + Tx_width))",
                                    f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1-({i}+1))*(Tx_layer_space_y + Tx_width))", f"0"])

        self.Tx_1 = self._create_polyline_litz(
            points=self.temp_Tx, name=f"Tx_1", coil_width="Tx_width")

        self.Rx_1.color = [0, 0, 255]
        self.Rx_1.transparency = 0
        self.Rx_2.color = [0, 0, 255]
        self.Rx_2.transparency = 0
        self.Tx_1.color = [255, 0, 0]
        self.Tx_1.transparency = 0

        # make core (main part)
        origin = ["-(w1)/2*w1_ratio",
                  "-(2*l1_leg+2*l2+l1_center)/2",  "-(2*l1_top+h1)/2"]

        dimension = ["(w1)*w1_ratio",
                     "(2*l1_leg+2*l2+l1_center)", "(2*l1_top+h1)"]

        self.air_box = self.M3D.modeler.create_box(
            origin=origin,
            sizes=dimension,
            name="air_box",
            material="vacuum"
        )

        blank_list = [self.air_box.name]
        tool_list = [self.core_base.name, self.Tx_1.name,
                     self.Rx_1.name, self.Rx_2.name]

        self.M3D.modeler.subtract(
            blank_list=blank_list,
            tool_list=tool_list,
            keep_originals=True
        )

    def assign_mesh(self):
        temp_list = list()
        temp_list.append(f"Tx_1")
        skin_depth = f"{math.sqrt(1.7*10**(-8)/math.pi/80/10**3/0.999991/4/math.pi/10**(-7))*10**3}mm"

        self.M3D.mesh.assign_skin_depth(
            assignment=temp_list, skin_depth=skin_depth, triangulation_max_length="12.2mm")

        air_list = list()
        air_list.append(f"air_box")
        self.M3D.mesh.assign_length_mesh(
            assignment=air_list, maximum_length="20mm")

    def create_region(self):

        region = self.M3D.modeler.create_air_region(
            z_pos="800", z_neg="800", y_pos="300", y_neg="300", x_pos="0", x_neg="0")
        self.M3D.assign_material(assignment=region, material="vacuum")
        region_face = self.M3D.modeler.get_object_faces("Region")
        region_face
        self.M3D.assign_radiation(
            assignment=region_face, radiation="Radiation")

    def create_excitation(self):
        self.Tx_in = self.M3D.modeler.get_faceid_from_position(
            position=[f"(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)", f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns)}-1)*(Tx_layer_space_y + Tx_width))", "(Rx_preg + Rx_height + Tx_preg + Tx_height)"])
        self.Tx_out = self.M3D.modeler.get_faceid_from_position(
            position=[f"(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)", f"(l1_center/2 + Tx_space_y + Tx_width/2)", f"(Rx_preg + Rx_height + Tx_preg + Tx_height)"])

        self.Rx2_in = self.M3D.modeler.get_faceid_from_position(
            position=[f"-(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)", f"-(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "-(Rx_preg/2+Rx_height/2)"])
        self.Rx2_out = self.M3D.modeler.get_faceid_from_position(
            position=[f"-(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)", f"(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "-(Rx_preg/2+Rx_height/2)"])

        self.Rx1_in = self.M3D.modeler.get_faceid_from_position(
            position=[f"-(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)", f"-(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "(Rx_preg/2+Rx_height/2)"])
        self.Rx1_out = self.M3D.modeler.get_faceid_from_position(
            position=[f"-(w1*w1_ratio/2 + ({self.terminal}mm) + 40mm)", f"(l1_center/2 + Rx_space_y + Rx_width/2 + Tx_space_y + Tx_width + ({self.Tx_turns}-1)*(Tx_layer_space_y + Tx_width))", "(Rx_preg/2+Rx_height/2)"])

        # assign coil terminal
        self.M3D.assign_coil(self.Tx_in, conductors_number=1,
                             polarity="Positive", name="Tx_in")
        self.M3D.assign_coil(self.Tx_out, conductors_number=1,
                             polarity="Negative", name="Tx_out")

        self.M3D.assign_coil(self.Rx1_in, conductors_number=1,
                             polarity="Positive", name="Rx1_in")
        self.M3D.assign_coil(self.Rx1_out, conductors_number=1,
                             polarity="Negative", name="Rx1_out")

        self.M3D.assign_coil(self.Rx2_in, conductors_number=1,
                             polarity="Positive", name="Rx2_in")
        self.M3D.assign_coil(self.Rx2_out, conductors_number=1,
                             polarity="Negative", name="Rx2_out")

        Tx_winding = self.M3D.assign_winding(assignment=[
        ], winding_type="Current", is_solid=True, current=4.1*math.sqrt(2), name="Tx")
        Rx_winding = self.M3D.assign_winding(assignment=[
        ], winding_type="Current", is_solid=True, current=7.35*math.sqrt(2), name="Rx1")
        Rx_winding2 = self.M3D.assign_winding(
            assignment=[], winding_type="Current", is_solid=True, current=0, name="Rx2")

        self.M3D.add_winding_coils(
            Tx_winding.name, coils=["Tx_in", "Tx_out"])
        self.M3D.add_winding_coils(
            Rx_winding.name, coils=["Rx1_in", "Rx1_out"])
        self.M3D.add_winding_coils(
            Rx_winding2.name, coils=["Rx2_in", "Rx2_out"])
        self.M3D.assign_matrix(
            assignment=[Tx_winding.name, Rx_winding.name, Rx_winding2.name], matrix_name="Matrix1")

    def _get_magnetic_report(self):
        assert self.M3D.post

        get_result_list = []
        get_result_list.append(["Matrix1.L(Tx,Tx)", "Ltx"])
        get_result_list.append(["Matrix1.L(Rx1,Rx1)", "Lrx1"])
        get_result_list.append(["Matrix1.L(Rx2,Rx2)", "Lrx2"])
        get_result_list.append(["Matrix1.L(Tx,Rx1)", "M1"])
        get_result_list.append(["Matrix1.L(Tx,Rx2)", "M2"])
        get_result_list.append(["Matrix1.CplCoef(Tx,Rx1)", "k1"])
        get_result_list.append(["Matrix1.CplCoef(Tx,Rx2)", "k2"])
        get_result_list.append(
            ["Matrix1.L(Tx,Tx)*(Matrix1.CplCoef(Tx,Rx1)^2)", "Lmt"])
        get_result_list.append(
            ["Matrix1.L(Rx1,Rx1)*(Matrix1.CplCoef(Tx,Rx1)^2)", "Lmr1"])
        get_result_list.append(
            ["Matrix1.L(Rx2,Rx2)*(Matrix1.CplCoef(Tx,Rx2)^2)", "Lmr2"])
        get_result_list.append(
            ["Matrix1.L(Tx,Tx)*(1-Matrix1.CplCoef(Tx,Rx1)^2)", "Llt"])
        get_result_list.append(
            ["Matrix1.L(Rx1,Rx1)*(1-Matrix1.CplCoef(Tx,Rx1)^2)", "Llr1"])
        get_result_list.append(
            ["Matrix1.L(Rx2,Rx2)*(1-Matrix1.CplCoef(Tx,Rx2)^2)", "Llr2"])
        get_result_list.append(["Matrix1.R(Tx,Tx)", "Rtx"])
        get_result_list.append(["Matrix1.R(Rx1,Rx1)", "Rrx1"])
        get_result_list.append(["Matrix1.R(Rx2,Rx2)", "Rrx2"])

        result_expressions = [item[0] for item in get_result_list]

        report = self.M3D.post.create_report(
            expressions=result_expressions, setup_sweep_name=None, domain='Sweep',
            variations={"Freq": ["All"]}, primary_sweep_variable=None, secondary_sweep_variable=None,
            report_category=None, plot_type='Data Table', context=None, subdesign_id=None,
            polyline_points=1001, plot_name="simulation parameter"
        )
        assert report

        df: pd.Series = self.report_to_df(report, get_result_list)
        self.data['magnetic_report'] = df

    def get_input_parameter(self):
        # ==============================
        # get input parameter
        # ==============================

        # self.magnetizing_current = 390*math.sqrt(2)/2/math.pi/(self.freq*10**(3))/self.Lmt/10**(-6)/2

        input_parameter_array = np.array([
            self.w1, self.l1_leg, self.l1_top, self.l2, self.h1, self.l1_center, self.Tx_turns,
            self.Tx_width, self.Tx_height, self.Tx_space_x, self.Tx_space_y, self.Tx_preg, self.Rx_width, self.Rx_height, self.Rx_space_x, self.Rx_space_y, self.Rx_preg, self.g2,
            self.Tx_layer_space_x, self.Tx_layer_space_y, self.wire_diameter, self.strand_number
        ])
        input_parameter_array = input_parameter_array.reshape(
            1, len(input_parameter_array))

        input_parameter_columns = [
            "w1", "l1_leg", "l1_top", "l2", "h1", "l1_center", "Tx_turns",
            "Tx_width", "Tx_height", "Tx_space_x", "Tx_space_y", "Tx_preg",
            "Rx_width", "Rx_height", "Rx_space_x", "Rx_space_y", "Rx_preg",
            "g2", "Tx_layer_space_x", "Tx_layer_space_y", "wire_diameter",
            "strand_number"
        ]

        # transform pandas data form
        df = pd.DataFrame(
            data=input_parameter_array, columns=input_parameter_columns
        ).iloc[0]

        self.data['input_parameter'] = df

    def _get_copper_loss_report(self):
        assert self.M3D.modeler
        # ==============================
        # get copper loss data
        # ==============================
        Tx = self.M3D.modeler.get_object_from_name(assignment="Tx_1")
        Rx1 = self.M3D.modeler.get_object_from_name(assignment="Rx_1")
        Rx2 = self.M3D.modeler.get_object_from_name(assignment="Rx_2")

        n_Tx_loss = self.volumetric_loss(assignments=Tx.name)
        n_Rx1_loss = self.volumetric_loss(assignments=Rx1.name)
        n_Rx2_loss = self.volumetric_loss(assignments=Rx2.name)

        get_result_list = []
        get_result_list.append([f'P_{Tx.name}', "copperloss_Tx"])
        get_result_list.append([f'P_{Rx1.name}', "copperloss_Rx1"])
        get_result_list.append([f'P_{Rx2.name}', "copperloss_Rx2"])

        result_expressions = [item[0] for item in get_result_list]

        report = self.M3D.post.create_report(expressions=result_expressions, setup_sweep_name=None, domain='Sweep',
                                             variations={"Phase": ["0deg"]}, primary_sweep_variable=None, secondary_sweep_variable=None,
                                             report_category="Fields", plot_type='Data Table', context=None, subdesign_id=None, polyline_points=1001, plot_name="copper loss data")
        df = self.report_to_df(report, get_result_list)
        self.data['copper_loss'] = df

    def _create_B_field(self):
        self.leg_left = self.M3D.modeler.create_rectangle(
            orientation="XY",
            origin=['-(w1)/2', '-(2*l1_leg+2*l2+l1_center)/2', 'g1/2'],
            sizes=['w1', 'l1_leg']
        )
        self.leg_left.model = False
        self.leg_left.name = "leg_left"

        self.leg_center = self.M3D.modeler.create_rectangle(
            orientation="XY",
            origin=['-(w1)/2', '-l1_center/2', 'g2/2'],
            sizes=['w1', 'l1_center']
        )
        self.leg_center.model = False
        self.leg_center.name = "leg_center"

        self.leg_right = self.M3D.modeler.create_rectangle(
            orientation="XY",
            origin=['-(w1)/2', '(2*l1_leg+2*l2+l1_center)/2', 'g1/2'],
            sizes=['w1', '-l1_leg']
        )
        self.leg_right.model = False
        self.leg_right.name = "leg_right"

        self.leg_top_left = self.M3D.modeler.create_rectangle(
            orientation="XZ",
            origin=['-(w1)/2', '-(l1_center+l2)/2', 'h1/2+l1_top'],
            sizes=['-l1_top', 'w1']
        )
        self.leg_top_left.model = False
        self.leg_top_left.name = "leg_top_left"

        self.leg_top_right = self.M3D.modeler.create_rectangle(
            orientation="XZ",
            origin=['-(w1)/2', '(l1_center+l2)/2', 'h1/2+l1_top'],
            sizes=['-l1_top', 'w1']
        )
        self.leg_top_right.model = False
        self.leg_top_right.name = "leg_top_right"

        self.leg_bottom_left = self.M3D.modeler.create_rectangle(
            orientation="XZ",
            origin=['-(w1)/2', '-(l1_center+l2)/2', '-(h1/2+l1_top)'],
            sizes=['l1_top', 'w1']
        )
        self.leg_bottom_left.model = False
        self.leg_bottom_left.name = "leg_bottom_left"

        self.leg_bottom_right = self.M3D.modeler.create_rectangle(
            orientation="XZ",
            origin=['-(w1)/2', '(l1_center+l2)/2', '-(h1/2+l1_top)'],
            sizes=['l1_top', 'w1']
        )
        self.leg_bottom_right.model = False
        self.leg_bottom_right.name = "leg_bottom_right"

    def _get_mean_Bfield(self, obj):

        assignment = obj.name

        oModule = self.M3D.ofieldsreporter
        oModule.CalcStack("clear")
        oModule.CopyNamedExprToStack("Mag_B")
        oModule.EnterVol(
            assignment) if obj.is3d else oModule.EnterSurf(assignment)
        oModule.CalcOp("Mean")
        name = "B_mean_{}".format(assignment)  # Need to check for uniqueness !
        oModule.AddNamedExpression(name, "Fields")

        return name

    def _get_max_Bfield(self, obj):

        assignment = obj.name

        oModule = self.M3D.ofieldsreporter
        oModule.CalcStack("clear")
        oModule.CopyNamedExprToStack("Mag_B")
        oModule.EnterVol(
            assignment) if obj.is3d else oModule.EnterSurf(assignment)
        oModule.CalcOp("Maximum")
        name = "B_max_{}".format(assignment)  # Need to check for uniqueness !
        oModule.AddNamedExpression(name, "Fields")

        return name

    def coreloss_project(self):
        self.M3D.duplicate_design(name=f'script1_{self.itr}_coreloss')
        self.M3D.set_active_design(name=f'script1_{self.itr}_coreloss')

        to_delete = [exc for exc in self.M3D.design_excitations.values() if exc.name in [
            "Tx", "Rx1", "Rx2"]]

        # 미리 수집한 대상에 대해 delete 호출
        for exc in to_delete:
            exc.delete()

        # 코어손실 계산할 자화 전류 인가
        Lmt_uH = self.data['magnetic_report']['Lmt_[uH]']

        magnetizing_current = self.get_magetizing_current(
            self.freq_kHz, Lmt_uH)

        Tx_winding = self.M3D.assign_winding(assignment=[
        ], winding_type="Current", is_solid=True, current=magnetizing_current*math.sqrt(2), name="Tx")
        Rx_winding = self.M3D.assign_winding(
            assignment=[], winding_type="Current", is_solid=True, current=0, name="Rx1")
        Rx_winding2 = self.M3D.assign_winding(
            assignment=[], winding_type="Current", is_solid=True, current=0, name="Rx2")

        self.M3D.add_winding_coils(
            Tx_winding.name, coils=["Tx_in", "Tx_out"])
        self.M3D.add_winding_coils(
            Rx_winding.name, coils=["Rx1_in", "Rx1_out"])
        self.M3D.add_winding_coils(
            Rx_winding2.name, coils=["Rx2_in", "Rx2_out"])
        self.M3D.assign_matrix(
            sources=[Tx_winding.name, Rx_winding.name, Rx_winding2.name], matrix_name="Matrix1")

        # 코어에 코어로스 세팅
        self.M3D.set_core_losses(assignment="core", core_loss_on_field=True)
        self._create_B_field()

        # 시뮬레이션
        self.M3D.analyze()

        # 코어손실 리포트 작성 후 저장 및 데이터로 가져오기
        get_result_list_coreloss = [f'Coreloss']
        report = self.M3D.post.create_report(
            expressions=get_result_list_coreloss, setup_sweep_name=None, domain='Sweep',
            variations={"Freq": ["All"]}, primary_sweep_variable=None,
            secondary_sweep_variable=None,
            report_category=None, plot_type='Data Table', context=None,
            subdesign_id=None, polyline_points=1001, plot_name="m coreloss parameter"
        )
        df1 = self.report_to_df(report, [["Coreloss", "coreloss"]])
        self.data['coreloss'] = df1

        # get B field
        parameters2 = []
        parameters2.append([self.core_base, "B_mean", "B_mean_core"])
        parameters2.append([self.leg_left, "B_mean", "B_mean_leg_left"])
        parameters2.append([self.leg_right, "B_mean", "B_mean_leg_right"])
        parameters2.append([self.leg_center, "B_mean", "B_mean_leg_center"])
        parameters2.append(
            [self.leg_top_left, "B_mean", "B_mean_leg_top_left"])
        parameters2.append(
            [self.leg_bottom_left, "B_mean", "B_mean_leg_bottom_left"])
        parameters2.append(
            [self.leg_top_right, "B_mean", "B_mean_leg_top_right"])
        parameters2.append(
            [self.leg_bottom_right, "B_mean", "B_mean_leg_bottom_right"])

        self.result_expressions = []
        self.name_list = []
        self.report_list = {}
        for obj, expression, name in parameters2:
            if expression == "B_mean":
                self.result_expressions.append(self._get_mean_Bfield(obj))
            self.name_list.append(name)

        report = self.M3D.post.create_report(expressions=self.result_expressions, setup_sweep_name=None, domain='Sweep',
                                             variations={"Phase": ["0deg"]}, primary_sweep_variable=None, secondary_sweep_variable=None,
                                             report_category="Fields", plot_type='Data Table', context=None, subdesign_id=None, polyline_points=1001, plot_name="calculator_report")

        df2 = self.report_to_df(report, [[k, v]for _, k, v in parameters2])
        self.data['B_field'] = df2

        data5_result = [f'P_Tx_1', f'P_Rx_1', f'P_Rx_2']
        report = self.M3D.post.create_report(expressions=data5_result, setup_sweep_name=None, domain='Sweep',
                                             variations={"Phase": ["0deg"]}, primary_sweep_variable=None, secondary_sweep_variable=None,
                                             report_category="Fields", plot_type='Data Table', context=None, subdesign_id=None, polyline_points=1001, plot_name="magnetizing copper loss data")
        df3 = self.report_to_df(report, [[k, k] for k in data5_result])

        self.data['magnetizing_copperloss'] = df3
        print(self.data)

    def simulation(self):
        self.create_project()

        self.set_variable()
        self.set_analysis()
        self.set_material()
        self.set_wire_material()

        self.create_core()
        self.create_winding()
        self.assign_mesh()
        self.create_region()

        self.create_excitation()

        self.M3D.analyze()

        self._get_magnetic_report()
        self.get_input_parameter()
        self._get_copper_loss_report()

        self.coreloss_project()

        self.M3D.release_desktop(close_projects=False, close_desktop=False)


if __name__ == "__main__":
    sim = Simul(non_graphical=False)
    sim.get_random_variable()
    sim.simulation()
