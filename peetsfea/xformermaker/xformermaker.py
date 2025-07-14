import sys
from operator import itemgetter
from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any, Iterator, Literal, Sequence, TypedDict, cast
from abc import ABC, abstractmethod

from ansys.aedt.core.maxwell import Maxwell3d
from ansys.aedt.core.modeler.cad.elements_3d import Plane, Point
from ansys.aedt.core.modeler.cad.object_3d import Object3d
from ansys.aedt.core.modeler.cad.polylines import Polyline
from ansys.aedt.core.modeler.modeler_2d import Modeler2D
from ansys.aedt.core.modeler.modeler_3d import Modeler3D

from peetsfea.aedthandler import AedtHandler, AedtInitializationError
from ansys.aedt.core.modules.material_lib import Materials
from ansys.aedt.core.generic.constants import SOLUTIONS
from ansys.aedt.core.modules.material import Material
from ansys.aedt.core.modules.solve_setup import SetupMaxwell
from ansys.aedt.core.modules.solve_sweeps import SetupProps
from pathlib import Path

import numpy as np


# np.random.seed(123)  # E38_8_25
# np.random.seed(1)  # E20_10_5
# np.random.seed(2)  # E25_13_7
# np.random.seed(3)  # E32_6_20
# np.random.seed((seed := 3474842696))
np.random.seed((seed := int(time.time_ns() % (2**32))))
# 3474842696 권선 실패


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
  def __init__(self, name: str, aedt_dir: str, new_desktop: bool) -> None:
    self.xformer_type: XformerType = XEnum.EEPlanaPlana2Series
    self.per: int = 3000
    self.freq_khz: int = 140
    self.is_validated: bool = False
    self.data = {}
    self.o3ds: dict[str, Object3d] = {}
    AedtHandler.initialize(
      project_name=f"{name}_Project", project_path=Path.cwd().joinpath(aedt_dir),
      design_name=f"{name}_Design", sol_type=SOLUTIONS.Maxwell3d.EddyCurrent,
      new_desktop=new_desktop
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

  def set_analysis(self, freq_kHz):
    setup = AedtHandler.peets_m3d.create_setup(setupname="Setup1")
    setup.props["MaximumPasses"] = 10
    setup.props["MinimumPasses"] = 2
    setup.props["PercentError"] = 5
    setup.props["Frequency"] = f'{freq_kHz}kHz'

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
      z_pos="800", z_neg="800", y_pos="0", y_neg="0", x_pos="300", x_neg="300")  # type: ignore

    AedtHandler.peets_m3d.assign_material(assignment=region, material="vacuum")
    region_face = self.modeler.get_object_faces("Region")
    # region_face
    AedtHandler.peets_m3d.assign_radiation(
      assignment=region_face, radiation="Radiation")

  @abstractmethod
  def assign_mesh(self) -> None:
    raise NotImplementedError("이건 인터페이스 클래스입니다. 상속받아서 내부를 작성해주세요")

  def export_report_to_csv(self, plot_name):
    import tempfile
    ret: str
    with tempfile.TemporaryDirectory(prefix="peetsfea_") as tmpdir:
      assert isinstance(AedtHandler.peets_m3d.post, PostProcessorMaxwell)
      csv_path = AedtHandler.peets_m3d.post.export_report_to_csv(
          project_dir=tmpdir,
          plot_name=plot_name
      )
      assert csv_path, "export_report_to_csv failed"
      ret = csv_path
      print(csv_path)  # shows full path in tmpdir
      # load the CSV

      self.data[plot_name] = pd.read_csv(csv_path)

  def delete_all(self):
    AedtHandler.peets_m3d.delete_design()
