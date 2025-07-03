from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Literal, Sequence, TypedDict
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

    for i in r.keys():
      list_r = list(r[i])
      values_N = len(list_r)
      random_N = np.random.randint(values_N)
      _ = list_r[random_N]
      if type(_) == float:
        self.v[i] = _
      else:
        self.comments.append(str(_))

      self.r[i] = [_, _, 0.01, 2]

  def set_variable_byrange(self, input_ranges: None | dict[str, list[float]]) -> None:

    if input_ranges == None:
      r: dict[str, list[float]] = self.random_ranges()
    else:
      r = input_ranges

    for i in self.template[self.xformer_type]["coil_keys"]:
      self.v[i] = self._random_choice(r[i])

    self.r.update(r)

    for k, v in self.v.items():
      AedtHandler.peets_m3d[k] = f"{v}mm"

    AedtHandler.peets_m3d["w1_ratio"] = f'(w1-2*l2)/w1'

  def validate_variable(self) -> None:
    r = self.r

    Tx_max = max(((self.v["Tx_layer_space_x"] + self.v["Tx_width"]) * math.ceil(self.v["Tx_turns"] / 2) + self.v["Tx_space_x"]),
                 ((self.v["Tx_layer_space_y"] + self.v["Tx_width"]) * math.ceil(self.v["Tx_turns"] / 2) + self.v["Tx_space_y"]))
    Rx_max = max((self.v["Rx_width"] + self.v["Rx_space_x"]),
                 (self.v["Rx_width"] + self.v["Rx_space_y"]))

    start = time.monotonic()  # 시작 시각 기록
    timeout_sec = 5
    while (True):
      if time.monotonic() - start > timeout_sec:
        raise RuntimeError(
            f"validate_variable() timed out after {timeout_sec} seconds")
      if self.v["Tx_height"] * 2 + self.v["Tx_preg"] * 2 + self.v["Rx_height"] * 4 + self.v["Rx_preg"] * 4 >= self.v["h1"]:
        self.v["Tx_height"] = self._random_choice(r["Tx_height"])
        self.v["Tx_preg"] = self._random_choice(r["Tx_preg"])
        self.v["Rx_height"] = self._random_choice(r["Rx_height"])
        self.v["Rx_preg"] = self._random_choice(r["Rx_preg"])
        self.v["h1"] = self._random_choice(r["h1"])

      elif Tx_max >= self.v["l2"] + self.v["l2_tap"]:
        self.v["Tx_layer_space_x"] = self._random_choice(
            r["Tx_layer_space_x"])
        self.v["Tx_layer_space_y"] = self._random_choice(
            r["Tx_layer_space_y"])
        self.v["Tx_width"] = self._random_choice(r["Tx_width"])
        self.v["l2"] = self._random_choice(r["l2"])
        Tx_max = max(((self.v["Tx_layer_space_x"] + self.v["Tx_width"]) * math.ceil(self.v["Tx_turns"] / 2) + self.v["Tx_space_x"]),
                     ((self.v["Tx_layer_space_y"] + self.v["Tx_width"]) * math.ceil(self.v["Tx_turns"] / 2) + self.v["Tx_space_y"]))

      elif Rx_max >= self.v["l2"] + self.v["l2_tap"]:
        self.v["Rx_layer_space_x"] = self._random_choice(
            r["Rx_layer_space_x"])
        self.v["Rx_width"] = self._random_choice(r["Rx_width"])
        Rx_max = max((self.v["Rx_width"] + self.v["Rx_space_x"]),
                     (self.v["Rx_width"] + self.v["Rx_space_y"]))

      else:
        break

    del self.r
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

    origin = ["-l1_center/2", "-(w1)/2", "-(h1)/2"]
    dimension = ["l1_center", "w1", "h1"]
    self.o3ds["core_sub3"] = self._create_box(
        origin=origin,
        sizes=dimension,
        name="core_sub3",
        material="ferrite"
    )

    origin = ["-(2*l1_leg+2*l2+2*l2_tap+l1_center)/2", "-(w1)/2", "-(h1)/2"]
    dimension = ["(l1_leg)", "(w1)", "(g1)"]
    self.o3ds["core_sub_g1"] = self._create_box(
        origin=origin,
        sizes=dimension,
        name="core_sub_g1",
        material=self.mat
    )

    # origin = ["(2*l1_leg+2*l2+2*l2_tap+l1_center)/2", "-(w1)/2", "-(h1)/2"]
    # dimension = ["-(l1_leg)", "(w1)", "(g1)"]
    # self.o3ds["core_sub_g2"] = self._create_box(
    #     origin=origin,
    #     sizes=dimension,
    #     name="core_sub_g2",
    #     material=self.mat
    #   )

    # origin = ["-l1_center/2", "-(w1)/2*w1_ratio", "-(h1)/2"]
    # dimension = ["l1_center", "w1*w1_ratio", "h1"]
    # self.o3ds["core_unite1"] =
  def create_winding(self) -> None:
    pass

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
    name=name, aedt_dir=aedt_dir, des_aedt_pid=1)
  sim.set_variable(None)
  sim.set_material()
  sim.create_core()
  # print(sim.template[XEnum.EEPlanaPlana2Series]["coil_keys"])
  # x.set_material()
  # AedtHandler.initialize(
  #   project_name="AIPDProject", project_path=Path.cwd().joinpath("../pyaedt_test"),
  #   design_name="AIPDDesign", sol_type=SOLUTIONS.Maxwell3d.EddyCurrent
  # )
  # AedtHandler.peets_aedt.close_desktop()
