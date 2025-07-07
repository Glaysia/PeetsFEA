from operator import itemgetter
from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any, Iterator, Literal, Sequence, TypedDict
from abc import ABC, abstractmethod

from ansys.aedt.core.maxwell import Maxwell3d
from ansys.aedt.core.modeler.cad.elements_3d import Plane, Point
from ansys.aedt.core.modeler.cad.object_3d import Object3d
from ansys.aedt.core.modeler.cad.polylines import Polyline
from ansys.aedt.core.modeler.modeler_2d import Modeler2D
from ansys.aedt.core.modeler.modeler_3d import Modeler3D

from aedthandler import AedtHandler, AedtInitializationError
from ansys.aedt.core.modules.material_lib import Materials
from ansys.aedt.core.generic.constants import SOLUTIONS
from ansys.aedt.core.modules.material import Material
from ansys.aedt.core.modules.solve_setup import SetupMaxwell
from ansys.aedt.core.modules.solve_sweeps import SetupProps
from pathlib import Path

import numpy as np

import sys


class XEnum(int, Enum):
  EEPlanaPlana2Series = 0
  EIPlanaPlana2Series = 1
  EILitzPlana2Series = 2
  EILitzPlate2Series = 3
  WILLBEADDED = 4


XformerType = Literal[
  XEnum.EEPlanaPlana2Series,
  XEnum.EIPlanaPlana2Series,
  XEnum.EILitzPlana2Series,
  XEnum.EILitzPlate2Series,
  XEnum.WILLBEADDED,
]


@dataclass
class LayoutParams:
  """
  코일의 형상을 담는 파라미터 입니다.
  """
  type: XformerType
  n_xformer: Literal[
      "single", "2series", "3series"
  ]
  core_type: Literal[
      "EE", "EI", "PQ", "UI", "UU"
  ]
  tx_type: Literal["plana", "litz", "plate"]
  rx_type: Literal["plana", "litz", "plate"]

  coil_shape: dict[str, float]


class XformerEntry(TypedDict):
  param: LayoutParams
  coil_keys: tuple[str, ...]


class XformerMakerInterface(ABC):
  def __init__(self, name: str, aedt_dir: str, des_aedt_pid: int) -> None:
    self.xformer_type: XformerType = XEnum.EEPlanaPlana2Series
    self.per: int = 3000
    self.freq_khz: int = 140
    self.is_validated: bool = False

    AedtHandler.initialize(
      project_name=f"{name}_Project", project_path=Path.cwd().joinpath(aedt_dir),
      design_name=f"{name}_Design", sol_type=SOLUTIONS.Maxwell3d.EddyCurrent,
      des_aedt_pid=des_aedt_pid
    )

    if not isinstance(AedtHandler.peets_m3d.modeler, Modeler3D):
      raise AedtInitializationError("[PeetsFEA] AedtHandler is not intialized")

    self.modeler: Modeler3D = AedtHandler.peets_m3d.modeler

    assert self.modeler, "[PeetsFEA] modeler failed"

  def _random_choice(self, X: list[float]) -> float:
    """
    X[0]: 시작값 (inclusive)
    X[1]: 끝값 (inclusive)
    X[2]: 스텝
    X[3]: 반올림할 소수점 자릿수
    """
    start, end, step, digits = X
    choices = np.arange(start, end + step, step)
    value = np.random.choice(choices)
    return round(float(value), int(digits))

  def set_params(self) -> None:
    pass

  @property
  def template(self) -> dict[XformerType, XformerEntry]:
    if not hasattr(self, "_coil_types_template"):
      self._coil_types_template: dict[XformerType, XformerEntry] = {}
      self._coil_types_template[XEnum.EEPlanaPlana2Series] = {
        "param": LayoutParams(
          type=XEnum.EEPlanaPlana2Series, n_xformer="2series",
          core_type="EE", tx_type="plana",
          rx_type="plana", coil_shape={}),
        "coil_keys": (
          "w1", "l1_leg", "l1_top", "l2",
          "h1", "ratio", "Tx_turns", "Tx_height", "Tx_preg",
          "Rx_space_y", "Rx_preg", "Rx_height", "Rx_space_x",
          "g1", "g2", "l1_center", "l2_tap",
          "Tx_space_x", "Tx_space_y", "core_N_w1", "core_P_w1",
          "Tx_layer_space_x", "Tx_layer_space_y", "Rx_layer_space_x",
          "Rx_layer_space_y", "Tx_width", "Rx_width",
        )
      }
      self._coil_types_template[XEnum.EIPlanaPlana2Series] = {
        "param": LayoutParams(
           type=XEnum.EIPlanaPlana2Series, n_xformer="2series",
            core_type="EI", tx_type="plana",
            rx_type="plana", coil_shape={}),
        "coil_keys":
        (
            "w1", "l1_leg", "l1_top", "l2",
            "h1", "Tx_turns", "Tx_height", "Tx_preg",
            "Rx_space_y", "Rx_preg", "Rx_height", "Rx_space_x",
            "g1", "g2", "l1_center", "l2_tap",
            "Tx_space_x", "Tx_space_y", "core_N_w1", "core_P_w1",
            "Tx_layer_space_x", "Tx_layer_space_y", "Rx_layer_space_x",
            "Rx_layer_space_y", "Tx_width", "Rx_width",
          )
      }
      self._coil_types_template[XEnum.EILitzPlana2Series] = {
        "param": LayoutParams(
          type=XEnum.EILitzPlana2Series, n_xformer="2series",
          core_type="EI", tx_type="litz",
          rx_type="plana", coil_shape={}),
        "coil_keys": (
          "w1", "l1_leg", "l1_top", "l2",
          "h1", "ratio", "Tx_turns", "Tx_preg",
          "Rx_space_y", "Rx_preg", "Rx_height", "Rx_space_x",
          "g1", "g2", "l1_center", "l2_tap",
          "Tx_space_x", "Tx_space_y", "core_N_w1", "core_P_w1",
          "Tx_layer_space_x", "Tx_layer_space_y",
          "Rx_width", "wire_diameter", "strand_number",
          "Tx_width", "Tx_height",
        )
      }
      self._coil_types_template[XEnum.EILitzPlate2Series] = {
        "param": LayoutParams(
          type=XEnum.EILitzPlate2Series, n_xformer="2series",
          core_type="EI", tx_type="litz",
          rx_type="plate", coil_shape={}),
        "coil_keys": (
          "w1", "l1_leg", "l1_top", "l2",
          "h1", "ratio", "Tx_turns", "Tx_preg",
          "Rx_space_y", "Rx_preg", "Rx_height", "Rx_space_x",
          "g2", "l1_center",
          "Tx_space_x", "Tx_space_y",
          "Tx_layer_space_x", "Tx_layer_space_y", "Rx_width",
          "wire_diameter", "strand_number", "Tx_width", "Tx_height",
        )
      }

    return self._coil_types_template

  def set_material(self) -> None:
    self.mat: Material | Literal[False] = AedtHandler.peets_m3d.materials.duplicate_material(  # type: ignore
      material="ferrite", name="ferrite_simulation")

    if not isinstance(self.mat, Material):
      raise RuntimeError("[PeetsFEA] duplicate_material failed")
    self.mat.permeability = self.per
    self.bp_point: Sequence[Sequence[float]] = [[0, 0], [0.05, 20.32], [0.06, 35.08], [0.07, 55.3], [
      0.08, 80.4], [0.09, 111.05], [0.1, 156.79], [0.2, 1002.4]]
    self.mat.set_bp_curve_coreloss(  # type: ignore
      points=self.bp_point, frequency=self.freq_khz * 1000)

    # print(f"[PeetsFEA] DEBUG:{self.mat.permeability.value}")  # type: ignore
  @abstractmethod
  def set_variable_byvalue(self, input_values: None | dict[str, Iterator[float | str]]) -> None:
    raise NotImplementedError("이건 인터페이스 클래스입니다. 상속받아서 내부를 작성해주세요")

  @abstractmethod
  def set_variable_byrange(self, input_ranges: None | dict[str, list[float]]) -> None:
    raise NotImplementedError("이건 인터페이스 클래스입니다. 상속받아서 내부를 작성해주세요")

  @abstractmethod
  def validate_variable(self) -> None:
    raise NotImplementedError("이건 인터페이스 클래스입니다. 상속받아서 내부를 작성해주세요")

  @abstractmethod
  def create_core(self) -> None:
    raise NotImplementedError("이건 인터페이스 클래스입니다. 상속받아서 내부를 작성해주세요")

  @abstractmethod
  def create_winding(self) -> None:
    raise NotImplementedError("이건 인터페이스 클래스입니다. 상속받아서 내부를 작성해주세요")

  @abstractmethod
  def create_exctation(self) -> None:
    raise NotImplementedError("이건 인터페이스 클래스입니다. 상속받아서 내부를 작성해주세요")

  def _create_polyline(self, points, name, coil_width, coil_height) -> Polyline:
    polyline_obj: Polyline = self.modeler.create_polyline(
        points,
        name=name,
        material="copper",
        xsection_type="Rectangle",
        xsection_width=coil_width,
        xsection_height=coil_height)

    return polyline_obj

  def _create_box(self, origin, sizes, name=None, material=None, *args, **kwargs) -> Object3d:
    """Create a box.

      Parameters
      ----------
      origin : list
          Anchor point for the box in Cartesian``[x, y, z]`` coordinates.
      sizes : list
          Length of the box edges in Cartesian``[x, y, z]`` coordinates.
      name : str, optional
          Name of the box. The default is ``None``, in which case the
          default name is assigned.
      material : str, optional
          Name of the material.  The default is ``None``, in which case the
          default material is assigned. If the material name supplied is
          invalid, the default material is assigned.

      Returns
      -------
      :class:`ansys.aedt.core.modeler.cad.object_3d.Object3d` or bool
          3D object or ``False`` if it fails.

      References
      ----------
      >>> oEditor.CreateBox

      Examples
      --------
      This example shows how to create a box in HFSS.
      The required parameters are ``position`` that provides the origin of the
      box and ``dimensions_list`` that provide the box sizes.
      The optional parameter ``material`` allows you to set the material name of the box.
      The optional parameter ``name`` allows you to assign a name to the box.

      This method applies to all 3D applications: HFSS, Q3D, Icepak, Maxwell 3D, and
      Mechanical.

      >>> from ansys.aedt.core import hfss
      >>> hfss = Hfss()
      >>> origin = [0, 0, 0]
      >>> dimensions = [10, 5, 20]
      >>> box_object = hfss.modeler.create_box(origin=origin, sizes=dimensions, name="mybox", material="copper")

      """
    ret: Point | Plane | Object3d | Literal[False] = self.modeler.create_box(
      origin, sizes, name, material, *args, **kwargs, **kwargs)
    if not isinstance(ret, Object3d):
      raise RuntimeError(
        f"[PeetsFEA] create_box failed: expected Object3d, got {type(ret).__name__}")

    return ret

  def create_region(self) -> None:

    region: Point | Plane | Object3d | Literal[False] = self.modeler.create_air_region(
      z_pos="800", z_neg="800", y_pos="300", y_neg="300", x_pos="0", x_neg="0")  # type: ignore

    AedtHandler.peets_m3d.assign_material(assignment=region, material="vacuum")
    region_face = self.modeler.get_object_faces("Region")
    # region_face
    AedtHandler.peets_m3d.assign_radiation(
      assignment=region_face, radiation="Radiation")

  @abstractmethod
  def assign_mesh(self) -> None:
    raise NotImplementedError("이건 인터페이스 클래스입니다. 상속받아서 내부를 작성해주세요")

  # def


class Project1_EE_Plana_Plana_2Series(XformerMakerInterface):
  def __init__(self, name: str, aedt_dir: str, des_aedt_pid: int) -> None:
    super().__init__(name, aedt_dir, des_aedt_pid)

    self.v: dict[str, float] = {}
    self.comments: list[str] = []
    self.o3ds: dict[str, Object3d] = {}
    self.xformer_type: XformerType = XEnum.EEPlanaPlana2Series
    self.per: int = 3000
    self.freq_khz: int = 140

  def random_ranges(self) -> dict[str, list[float]]:
    ranges = {}

    ranges["w1"] = [20, 200, 1, 0]
    ranges["l1_leg"] = [2, 10, 0.1, 1]
    ranges["l1_top"] = [0.5, 2, 0.1, 1]
    ranges["l2"] = [5, 20, 0.1, 1]  # under, upper, resolution

    ranges["h1"] = [0.1, 2, 0.01, 2]
    ranges["ratio"] = [0.5, 0.98, 0.01, 2]

    ranges["Tx_turns"] = [14, 14, 1, 0]

    ranges["Tx_height"] = [0.035, 0.175, 0.035, 3]
    ranges["Tx_preg"] = [0.01, 0.1, 0.01, 2]

    ranges["Rx_space_y"] = [0.1, 1, 0.1, 1]
    ranges["Rx_preg"] = [0.01, 0.2, 0.01, 2]

    ranges["Rx_height"] = [0.035, 0.175, 0.035, 3]
    ranges["Rx_space_x"] = [0.05, 1, 0.01, 2]

    ranges["g1"] = [0, 0, 0.01, 2]
    ranges["g2"] = [0, 0.5, 0.01, 2]

    ranges["l1_center"] = [1, 20, 1, 0]
    ranges["l2_tap"] = [0, 0, 1, 0]

    ranges["Tx_space_x"] = [0.1, 5, 0.1, 1]
    ranges["Tx_space_y"] = [0.1, 5, 0.1, 1]
    ranges["Rx_space_x"] = [0.1, 5, 0.1, 1]
    ranges["Rx_space_y"] = [0.1, 5, 0.1, 1]

    ranges["core_N_w1"] = [0, 30, 1, 0]
    ranges["core_P_w1"] = [0, 30, 1, 0]

    ranges["Tx_layer_space_x"] = [0.1, 5, 0.1, 1]
    ranges["Tx_layer_space_y"] = [0.1, 5, 0.1, 1]
    ranges["Rx_layer_space_x"] = [0.1, 5, 0.1, 1]
    ranges["Rx_layer_space_y"] = [0.1, 5, 0.1, 1]

    ranges["Tx_width"] = [0.5, 3, 0.1, 1]
    ranges["Rx_width"] = [4, 20, 0.1, 1]
    return ranges

  def set_variable_byvalue(self, input_values: None | dict[str, Iterator[float | str]]) -> None:
    self.r = {}
    if input_values == None:
      return
    else:
      r = input_values

    values_N = 0
    random_N = 0
    for i in r.keys():
      list_r = list(r[i])
      if values_N == 0:
        values_N = len(list_r)
        random_N = np.random.randint(values_N)

      _ = list_r[random_N]
      if type(_) == float:
        self.v[i] = _
      else:
        self.comments.append(str(_))
        AedtHandler.log(f"상용코어 {_}로 선택됨.")

      self.r[i] = [_, _, 0.01, 2]

  def set_variable_byrange(self, input_ranges: None | dict[str, list[float]]) -> None:

    if input_ranges == None:
      r: dict[str, list[float]] = self.random_ranges()
    else:
      r = input_ranges

    for i in r.keys():
      self.v[i] = self._random_choice(r[i])

    self.r.update(r)

    for k, v in self.v.items():
      AedtHandler.peets_m3d[k] = f"{v}mm"

    AedtHandler.peets_m3d["w1_ratio"] = f'(w1-2*l2)/w1'

  def validate_variable(self) -> None:
    r = self.r
    v = self.v
    Tx_max = max(((self.v["Tx_layer_space_x"] + self.v["Tx_width"]) * math.ceil(self.v["Tx_turns"] / 2) + self.v["Tx_space_x"]),
                 ((self.v["Tx_layer_space_y"] + self.v["Tx_width"]) * math.ceil(self.v["Tx_turns"] / 2) + self.v["Tx_space_y"]))
    Rx_max = max((self.v["Rx_width"] + self.v["Rx_space_x"]),
                 (self.v["Rx_width"] + self.v["Rx_space_y"]))

    start = time.monotonic()  # 시작 시각 기록
    timeout_sec = 500
    while (True):
      AedtHandler.log(str(self.v))
      if time.monotonic() - start > timeout_sec:
        self.is_validated = True
        self.create_core()
        raise RuntimeError(
            f"validate_variable() timed out after {timeout_sec} seconds\nself.v: {self.v}")
      # if self.v["Tx_height"] * 2 + self.v["Tx_preg"] * 2 + self.v["Rx_height"] * 4 + self.v["Rx_preg"] * 4 >= self.v["h1"]:
      #   self.v["Tx_height"] = self._random_choice(r["Tx_height"])
      #   self.v["Tx_preg"] = self._random_choice(r["Tx_preg"])
      #   self.v["Rx_height"] = self._random_choice(r["Rx_height"])
      #   self.v["Rx_preg"] = self._random_choice(r["Rx_preg"])
      #   self.v["h1"] = self._random_choice(r["h1"])
      if v["Rx_width"] > v["l2"]:
        v["Rx_width"] = self._random_choice(
          r["Rx_width"]
        )
      elif (v["w1"] * float(AedtHandler.peets_m3d.get_evaluated_value("w1_ratio")) + 2 * v["Rx_space_x"]) < v["l1_center"]:
        v["Rx_space_x"] = self._random_choice(
          r["Rx_space_x"]
        )

        # elif Tx_max >= self.v["l2"] + self.v["l2_tap"]:
        #   self.v["Tx_layer_space_x"] = self._random_choice(
        #       r["Tx_layer_space_x"])
        #   self.v["Tx_layer_space_y"] = self._random_choice(
        #       r["Tx_layer_space_y"])
        #   self.v["Tx_width"] = self._random_choice(r["Tx_width"])
        #   self.v["l2"] = self._random_choice(r["l2"])
        #   Tx_max = max(((self.v["Tx_layer_space_x"] + self.v["Tx_width"]) * math.ceil(self.v["Tx_turns"] / 2) + self.v["Tx_space_x"]),
        #                ((self.v["Tx_layer_space_y"] + self.v["Tx_width"]) * math.ceil(self.v["Tx_turns"] / 2) + self.v["Tx_space_y"]))

        # elif Rx_max >= self.v["l2"] + self.v["l2_tap"]:
        #   self.v["Rx_layer_space_x"] = self._random_choice(
        #       r["Rx_layer_space_x"])
        #   self.v["Rx_width"] = self._random_choice(r["Rx_width"])
        #   Rx_max = max((self.v["Rx_width"] + self.v["Rx_space_x"]),
        #                (self.v["Rx_width"] + self.v["Rx_space_y"]))

      else:
        break

    del self.r

    for k, v in self.v.items():
      AedtHandler.peets_m3d[k] = f"{v}mm"

    AedtHandler.peets_m3d["w1_ratio"] = f'(w1-2*l2)/w1'
    self.is_validated = True

  def create_core(self) -> None:
    if not hasattr(self, "mat"):
      raise RuntimeError("set_material() must be called before create_core()")

    if not self.is_validated:
      raise RuntimeError(
        "validate_variable() must be called before validate_variable()")

    origin: Sequence[str] = ["-(2*l1_leg+2*l2+2*l2_tap+l1_center)/2",
                             "-(w1)/2", "-(2*l1_top+h1)/2"]
    dimension: Sequence[str] = [
      "(2*l1_leg+2*l2+2*l2_tap+l1_center)", "(w1)", "(2*l1_top+h1)"]
    self.o3ds["core_base"] = self._create_box(
      origin=origin,
      sizes=dimension,
      name="core",
      material=self.mat
    )

    origin = ["l1_center/2", "-(w1)/2", "-(h1)/2"]
    dimension = ["l2+l2_tap", "w1", "h1"]
    self.o3ds["core_sub1"] = self._create_box(
        origin=origin,
        sizes=dimension,
        name="core_sub1",
        material="ferrite"
    )

    origin = ["-l1_center/2", "-(w1)/2", "-(h1)/2"]
    dimension = ["-(l2+l2_tap)", "w1", "h1"]
    self.o3ds["core_sub2"] = self._create_box(
        origin=origin,
        sizes=dimension,
        name="core_sub2",
        material="ferrite"
    )

    # origin = ["-l1_center/2", "-(w1)/2", "-(h1)/2"]
    # dimension = ["l1_center", "w1", "g1"]
    # self.o3ds["core_sub3"] = self._create_box(
    #     origin=origin,
    #     sizes=dimension,
    #     name="core_sub3",
    #     material="ferrite"
    # )

    # origin = ["-(2*l1_leg+2*l2+2*l2_tap+l1_center)/2", "-(w1)/2", "-(h1)/2"] # EI에 유용할듯
    origin = ["-(2*l1_leg+2*l2+2*l2_tap+l1_center)/2", "-(w1)/2", "-(g2)/2"]
    dimension = ["(l1_leg)", "(w1)", "(g2)"]
    self.o3ds["core_sub_g1"] = self._create_box(
        origin=origin,
        sizes=dimension,
        name="core_sub_g1",
        material=self.mat
    )

    origin = ["(2*l1_leg+2*l2+2*l2_tap+l1_center)/2", "-(w1)/2", "-(g2)/2"]
    dimension = ["-(l1_leg)", "(w1)", "(g2)"]
    self.o3ds["core_sub_g2"] = self._create_box(
        origin=origin,
        sizes=dimension,
        name="core_sub_g2",
        material=self.mat
      )

    # origin = ["-l1_center/2", "-(w1)/2", "-(h1)/2"] # 나중에 과제용으로 가운데 일자가 아니게 할 때 사용하자.
    # dimension = ["l1_center", "w1", "h1"]
    # self.o3ds["core_unite1"] = self._create_box(
    #     origin=origin,
    #     sizes=dimension,
    #     name="core_unite1",
    #     material="ferrite"
    # )

    origin = ["-l1_center/2", "-(w1)/2", "-(g2)/2"]
    dimension = ["l1_center", "w1", "g2"]
    self.o3ds["core_unite_sub_g1"] = self._create_box(
        origin=origin,
        sizes=dimension,
        name="core_unite_sub_g1",
        material="ferrite"
    )

    # blank_list = [self.o3ds["core_unite1"].name] # 나중에 과제용으로 가운데 일자가 아니게 할 때 사용하자.
    # tool_list = [self.o3ds["core_unite_sub_g1"].name]
    # self.modeler.subtract(
    #   blank_list=blank_list,
    #   tool_list=tool_list,
    #   keep_originals=True
    # )

    blank_list = [self.o3ds["core_base"].name]
    tool_list = list(map(lambda x: self.o3ds[x].name, [
      "core_sub1",
      "core_sub2",
      # "core_sub3",
      "core_sub_g1",
      "core_sub_g2",
      "core_unite_sub_g1"
    ]))
    self.modeler.subtract(
      blank_list=blank_list,
      tool_list=tool_list,
      keep_originals=False
    )

    self.o3ds["core_base"].transparency = 0.6

    AedtHandler.log("Core 생성 완료")

  def create_winding(self) -> None:
    self.points = [
      ["(w1*w1_ratio/2 + core_P_w1 + 40mm)",
       "-(w1/2 + g1 + Rx_width/2)", "0mm"],
      ["(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)",
       "-(w1/2 + g1 + Rx_width/2)", "0mm"],
      ["-(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)",
       "-(w1/2 + g1 + Rx_width/2)", "0mm"],
      ["-(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)",
       "(w1/2 + g1 + Rx_width/2)", "0mm"],
      ["(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)",
       "(w1/2 + g1 + Rx_width/2)", "0mm"],
      ["(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)", "0", "0mm"],
      ["(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)",
       "0", "-(Rx_height+Rx_preg)"],
      ["(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)",
       "-(w1/2 + g1 + Rx_width/2)", "-(Rx_height+Rx_preg)"],
      ["-(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)",
       "-(w1/2 + g1 + Rx_width/2)", "-(Rx_height+Rx_preg)"],
      ["-(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)",
       "(w1/2 + g1 + Rx_width/2)", "-(Rx_height+Rx_preg)"],
      ["(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)",
       "(w1/2 + g1 + Rx_width/2)", "-(Rx_height+Rx_preg)"],
      ["(w1*w1_ratio/2 + core_P_w1 + 40mm)",
       "(w1/2 + g1 + Rx_width/2)", "-(Rx_height+Rx_preg)"]
    ]
    o3ds = self.o3ds

    o3ds["RX_1"] = self._create_polyline(
      points=self.points, name=f"Rx_1", coil_width="Rx_width", coil_height="Rx_height")
    self.modeler.mirror(
      assignment=self.o3ds["RX_1"],
      origin=[0, 0, 0],
      vector=[1, 0, 0]
    )
    Rx_1_move = ["0mm", "0mm",
                 "2*(Tx_preg+Tx_height+Rx_preg+Rx_height/2+Rx_preg+Rx_height)-h1/2"]
    self.modeler.move(
      assignment=o3ds["RX_1"],
      vector=Rx_1_move
    )

    o3ds["RX_2"] = self._create_polyline(
      points=self.points, name=f"Rx_2", coil_width="Rx_width", coil_height="Rx_height")
    self.modeler.mirror(
      assignment=self.o3ds["RX_2"],
      origin=[0, 0, 0],
      vector=[1, 0, 0]
    )
    Rx_2_move = [
      "0mm", "0mm", "1*(Tx_preg+Tx_height+Rx_preg+Rx_height/2+Rx_preg+Rx_height)-h1/2"]
    self.modeler.move(
      assignment=o3ds["RX_2"],
      vector=Rx_2_move
    )

    turns = self.v['Tx_turns']
    half_turns = math.ceil(turns / 2)

    def from_expression(exp: str) -> float:  # 단위 무조건 mm
      m3d = AedtHandler.peets_m3d
      m3d['peets_tmp'] = exp
      return 1000 * float(f"{m3d.get_evaluated_value('peets_tmp'):.9f}")

    # Tx_total_width_y = turns * (v["Tx_width"] + v["Tx_space_y"])
    # y = (v["w1"] / 2 + Tx_total_width_y)
    total_width = (turns / 2 + 1) * \
        from_expression("Tx_space_y") / 2
    self.points_Tx = [
      ["(w1*w1_ratio/2 + core_P_w1 + 40mm)",
       f"-(w1/2 + {total_width}mm)", "0mm"],
      ["-(l1_center/2 + l2 - Tx_width )",
       f"-(w1/2 + {total_width}mm)", "0mm"],
    ]
    points = self.points_Tx

    def flip(points: list[list[str]] | list[str], axis: str) -> list[str]:
      if isinstance(points[0], str):
        last: list[str] = cast(list[str], points)
      else:
        last: list[str] = cast(list[str], points[-1])

      if not (axis in ('x', 'y', 'z')):
        raise ValueError("axis는 x,y,z만 입력 가능")

      idx = {'x': 0, 'y': 1, 'z': 2}
      idx = idx[axis]

      new_val = -from_expression(last[idx])
      ret = last[:]
      ret[idx] = f"{new_val}mm"
      return ret

    def shrink(points: list[list[str]] | list[str], axis: str, value_exp: str) -> list[str]:
      if isinstance(points[0], str):
        last: list[str] = cast(list[str], points)
      else:
        last: list[str] = cast(list[str], points[-1])

      if not (axis in ('x', 'y', 'z')):
        raise ValueError("axis는 x,y,z만 입력 가능")

      idx = {'x': 0, 'y': 1, 'z': 2}
      idx = idx[axis]

      prev_val = from_expression(last[idx])
      delta = abs(from_expression(value_exp))
      new_val = prev_val - delta if prev_val > 0 else prev_val + delta

      ret = last[:]
      ret[idx] = f"{new_val}mm"

      return ret

    turns = int(self.v["Tx_turns"])
    sign = "-" if turns % 2 == 0 else "+"
    sf = False

    for i in range(turns):
      if sf:
        points.append(shrink(flip(points, 'y'), 'y', 'Tx_space_y/2'))
        points.append(shrink(flip(points, 'x'), 'x', 'Tx_space_x/2'))
      else:
        points.append(flip(points, 'y'))
        points.append(flip(points, 'x'))

      sf = not sf

    points.append(
      shrink(points, 'y', f'{points[-1][1]} {sign} Tx_width * 1.5'))
    # 1turn
    # points.append(flip(points, 'y'))
    # points.append(flip(points, 'x'))
    # points.append(shrink(points, 'y', f'{points[-1][1]} + Tx_width * 1.5'))

    # 2turn
    # points.append(flip(points, 'y'))
    # points.append(flip(points, 'x'))
    # points.append(shrink(flip(points, 'y'), 'y', 'Tx_space_y'))
    # points.append(shrink(flip(points, 'x'), 'x', 'Tx_space_x'))
    # points.append(shrink(points, 'y', f'{points[-1][1]} - Tx_width * 1.5'))

    # 3turn
    # points.append(flip(points, 'y'))
    # points.append(flip(points, 'x'))
    # points.append(shrink(flip(points, 'y'), 'y', 'Tx_space_y'))
    # points.append(shrink(flip(points, 'x'), 'x', 'Tx_space_x'))
    # points.append(flip(points, 'y'))
    # points.append(flip(points, 'x'))
    # points.append(shrink(points, 'y', f'{points[-1][1]} + Tx_width * 1.5'))

    # 4turn
    # points.append(flip(points, 'y'))
    # points.append(flip(points, 'x'))
    # points.append(shrink(flip(points, 'y'), 'y', 'Tx_space_y'))
    # points.append(shrink(flip(points, 'x'), 'x', 'Tx_space_x'))
    # points.append(flip(points, 'y'))
    # points.append(flip(points, 'x'))
    # points.append(shrink(flip(points, 'y'), 'y', 'Tx_space_y'))
    # points.append(shrink(flip(points, 'x'), 'x', 'Tx_space_x'))
    # points.append(shrink(points, 'y', f'{points[-1][1]} - Tx_width * 1.5'))

    # 5turn
    # points.append(flip(points, 'y'))
    # points.append(flip(points, 'x'))
    # points.append(shrink(flip(points, 'y'), 'y', 'Tx_space_y'))
    # points.append(shrink(flip(points, 'x'), 'x', 'Tx_space_x'))
    # points.append(flip(points, 'y'))
    # points.append(flip(points, 'x'))
    # points.append(shrink(flip(points, 'y'), 'y', 'Tx_space_y'))
    # points.append(shrink(flip(points, 'x'), 'x', 'Tx_space_x'))
    # points.append(flip(points, 'y'))
    # points.append(flip(points, 'x'))
    # points.append(shrink(points, 'y', f'{points[-1][1]} + Tx_width * 1.5'))

    points_connect = []
    last_x, last_y, last_z = points[-1]
    points_connect.append(
      [last_x, 0, from_expression(f"{last_z}-(Tx_height/2)")])
    points_connect.append(
      [last_x, 0, from_expression(f"{last_z}+(Tx_height/2)")])

    o3ds['Tx_connect'] = self.modeler.create_polyline(
        points_connect, name=f"Tx_connect", xsection_type="Circle", xsection_width="Tx_width*0.8", xsection_num_seg=12)  # type: ignore

    # points.append(flip(points[-1], 'y'))
    o3ds['Tx_1'] = self._create_polyline(
        points=self.points_Tx, name=f"Tx_1", coil_width="Tx_width", coil_height="Tx_height")

    self.modeler.copy(assignment=o3ds['Tx_1'])
    self.modeler.paste()
    o3ds['Tx_2'] = self.modeler.get_object_from_name(  # type: ignore
      assignment="Tx_2")
    self.modeler.mirror(
      assignment=o3ds['Tx_2'],
      origin=[0, 0, 0], vector=[0, 1, 0]
    )
    self.modeler.move(
      assignment=o3ds['Tx_2'],
      vector=["0mm", "0mm", "-(Tx_preg/2+Tx_height/2)"]
    )

    self.modeler.move(
      assignment=o3ds['Tx_1'],
      vector=["0mm", "0mm", "Tx_preg/2+Tx_height/2"]
    )

    self.modeler.unite(
      assignment=[o3ds['Tx_1'], o3ds["Tx_2"], o3ds["Tx_connect"]])

    o3ds["RX_1"].color = [0, 0, 255]
    o3ds["RX_1"].transparency = 0
    o3ds["RX_2"].color = [0, 0, 255]
    o3ds["RX_2"].transparency = 0
    o3ds['Tx_1'].color = [255, 0, 0]
    o3ds['Tx_1'].transparency = 0

    AedtHandler.log("Winding 생성 완료")

  def create_exctation(self) -> None:
    pass

  def assign_mesh(self) -> None:
    pass


if __name__ == "__main__":

  # print(sys.argv)
  if len(sys.argv) < 2:
    aedt_dir = "../pyaedt_test"
    name = "PeetsFEAdev"
  else:
    parr_idx: str = str(sys.argv[0])[-1]
    name = f"xform_{parr_idx}"
    aedt_dir = f"parrarel{parr_idx}"

  sim = Project1_EE_Plana_Plana_2Series(
    name=name, aedt_dir=aedt_dir, des_aedt_pid=856)
  ccore: list[dict] = [
    {'EXX': 'E10_5.5_5', 'h1': 8.5, 'l1_center': 2.35,
     'l1_leg': 1.175, 'l1_top': 1.25, 'l2': 2.75, 'w1': 4.8},
    {'EXX': 'E13_6_6', 'h1': 8.2, 'l1_center': 3.2,
     'l1_leg': 1.6, 'l1_top': 1.6, 'l2': 3.15, 'w1': 6.4},
    {'EXX': 'E13_7_4', 'h1': 9.0, 'l1_center': 3.7,
     'l1_leg': 1.85, 'l1_top': 2.0, 'l2': 2.6, 'w1': 3.7},
    {'EXX': 'E16_8_5', 'h1': 11.4, 'l1_center': 4.7,
     'l1_leg': 2.35, 'l1_top': 2.5, 'l2': 3.3, 'w1': 4.7},
    {'EXX': 'E19_8_5', 'h1': 11.4, 'l1_center': 4.7,
     'l1_leg': 2.35, 'l1_top': 2.4, 'l2': 4.85, 'w1': 4.7},
    {'EXX': 'E20_10_5', 'h1': 12.6, 'l1_center': 5.2,
     'l1_leg': 2.6, 'l1_top': 3.7, 'l2': 5.15, 'w1': 5.3},
    {'EXX': 'E20_10_6', 'h1': 14.0, 'l1_center': 5.9,
     'l1_leg': 2.95, 'l1_top': 3.2, 'l2': 4.1, 'w1': 5.9},
    {'EXX': 'E25_10_6', 'h1': 12.8, 'l1_center': 6.35,
     'l1_leg': 3.175, 'l1_top': 3.25, 'l2': 6.35, 'w1': 6.35},
    {'EXX': 'E25_13_7', 'h1': 17.4, 'l1_center': 7.5,
     'l1_leg': 3.75, 'l1_top': 4.1, 'l2': 5.0, 'w1': 7.5},
    {'EXX': 'E30_15_7', 'h1': 19.4, 'l1_center': 7.2,
     'l1_leg': 3.6, 'l1_top': 5.3, 'l2': 8.2, 'w1': 7.3},
    {'EXX': 'E32_6_20', 'h1': 6.36, 'l1_center': 6.35,
     'l1_leg': 3.175, 'l1_top': 3.17, 'l2': 9.525, 'w1': 20.32},
    {'EXX': 'E34_14_9', 'h1': 19.6, 'l1_center': 9.3,
     'l1_leg': 4.65, 'l1_top': 4.3, 'l2': 7.85, 'w1': 9.3},
    {'EXX': 'E35_18_10', 'h1': 25.0, 'l1_center': 10.0,
     'l1_leg': 5.0, 'l1_top': 5.0, 'l2': 7.5, 'w1': 10.0},
    {'EXX': 'E38_8_25', 'h1': 8.9, 'l1_center': 7.6,
     'l1_leg': 3.8, 'l1_top': 3.81, 'l2': 11.45, 'w1': 25.4},
    {'EXX': 'E41_17_12', 'h1': 20.8, 'l1_center': 12.45,
     'l1_leg': 6.225, 'l1_top': 6.2, 'l2': 7.85, 'w1': 12.4},
    {'EXX': 'E42_21_15', 'h1': 29.6, 'l1_center': 12.2,
     'l1_leg': 6.1, 'l1_top': 6.2, 'l2': 9.3, 'w1': 15.2},
    {'EXX': 'E42_21_20', 'h1': 29.6, 'l1_center': 12.2,
     'l1_leg': 6.1, 'l1_top': 6.2, 'l2': 9.3, 'w1': 20.0},
    {'EXX': 'E42_33_20', 'h1': 52.0, 'l1_center': 12.2,
     'l1_leg': 6.1, 'l1_top': 6.8, 'l2': 8.8, 'w1': 20.0},
    {'EXX': 'E43_10_28', 'h1': 10.8, 'l1_center': 8.1,
     'l1_leg': 4.05, 'l1_top': 4.1, 'l2': 13.5, 'w1': 27.9},
    {'EXX': 'E47_20_16', 'h1': 24.2, 'l1_center': 15.6,
     'l1_leg': 7.8, 'l1_top': 7.5, 'l2': 7.85, 'w1': 15.6},
    {'EXX': 'E55_28_21', 'h1': 37.0, 'l1_center': 17.2,
     'l1_leg': 8.6, 'l1_top': 9.0, 'l2': 10.9, 'w1': 21.0},
    {'EXX': 'E56_24_19', 'h1': 29.2, 'l1_center': 18.8,
     'l1_leg': 9.4, 'l1_top': 9.0, 'l2': 9.25, 'w1': 18.8},
    {'EXX': 'E58_11_38', 'h1': 13.0, 'l1_center': 8.1,
     'l1_leg': 4.05, 'l1_top': 4.0, 'l2': 21.1, 'w1': 38.1},
    {'EXX': 'E64_10_50', 'h1': 10.2, 'l1_center': 10.2,
     'l1_leg': 5.1, 'l1_top': 5.1, 'l2': 21.8, 'w1': 50.8},
    {'EXX': 'E65_32_27', 'h1': 44.4, 'l1_center': 20.0,
     'l1_leg': 10.0, 'l1_top': 10.6, 'l2': 12.5, 'w1': 27.4},
    {'EXX': 'E71_33_32', 'h1': 43.8, 'l1_center': 22.0,
     'l1_leg': 11.0, 'l1_top': 11.3, 'l2': 13.25, 'w1': 32.0},
    {'EXX': 'E80_38_20', 'h1': 56.4, 'l1_center': 19.8,
     'l1_leg': 9.9, 'l1_top': 9.9, 'l2': 20.2, 'w1': 19.8}
  ]

  values: dict[str, Iterator[float | str]] = {}
  ranges: dict[str, list[float]] = {}

  values["EXX"] = map(itemgetter("EXX"), ccore)
  values["w1"] = map(itemgetter("w1"), ccore)
  values["l1_leg"] = map(itemgetter("l1_leg"), ccore)
  values["l1_top"] = map(itemgetter("l1_top"), ccore)
  values["l2"] = map(itemgetter("l2"), ccore)
  values["h1"] = map(itemgetter("h1"), ccore)
  values["l1_center"] = map(itemgetter("l1_center"), ccore)

  ranges["l2_tap"] = [0, 0, 1, 0]
  ranges["ratio"] = [0.5, 0.50, 0.01, 2]

  ranges["Tx_turns"] = [14, 14, 1, 0]
  ranges["Tx_height"] = [0.035, 0.175, 0.035, 3]
  ranges["Tx_preg"] = [0.01, 0.1, 0.01, 2]
  ranges["Rx_space_y"] = [0.1, 1, 0.1, 1]

  ranges["Rx_preg"] = [0.01, 0.2, 0.01, 2]
  ranges["Rx_height"] = [0.035, 0.175, 0.035, 3]

  ranges["g1"] = [0.1, 0.1, 0.01, 2]
  ranges["g2"] = [0, 0.5, 0.01, 2]

  ranges["Tx_space_x"] = [0.1, 5, 0.1, 1]
  ranges["Tx_space_y"] = [0.1, 5, 0.1, 1]
  ranges["Rx_space_x"] = [0.1, 5, 0.1, 1]
  ranges["Rx_space_y"] = [0.1, 5, 0.1, 1]

  ranges["core_N_w1"] = [0, 30, 1, 0]
  ranges["core_P_w1"] = [0, 30, 1, 0]

  ranges["Tx_layer_space_x"] = [0.1, 5, 0.1, 1]
  ranges["Tx_layer_space_y"] = [0.1, 5, 0.1, 1]
  ranges["Rx_layer_space_x"] = [0.1, 5, 0.1, 1]
  ranges["Rx_layer_space_y"] = [0.1, 5, 0.1, 1]

  ranges["Tx_width"] = [0.5, 3, 0.1, 1]
  ranges["Rx_width"] = [1, 20, 0.1, 1]

  sim.set_variable_byvalue(input_values=values)
  sim.set_variable_byrange(input_ranges=ranges)
  sim.set_material()
  sim.validate_variable()
  sim.create_core()
  sim.create_winding()
  # print(sim.template[XEnum.EEPlanaPlana2Series]["coil_keys"])
  # x.set_material()
  # AedtHandler.initialize(
  #   project_name="AIPDProject", project_path=Path.cwd().joinpath("../pyaedt_test"),
  #   design_name="AIPDDesign", sol_type=SOLUTIONS.Maxwell3d.EddyCurrent
  # )
  # AedtHandler.peets_aedt.close_desktop()
