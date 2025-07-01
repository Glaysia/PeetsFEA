from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypedDict
from aedthandler import AedtHandler
from ansys.aedt.core.generic.constants import SOLUTIONS
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
  def __init__(self, name: str, aedt_dir: str) -> None:
    self.xformer_type: XformerType = XEnum.EEPlanaPlana2Series

    AedtHandler.initialize(
      project_name=f"{name}_Project", project_path=Path.cwd().joinpath(aedt_dir),
      design_name=f"{name}_Design", sol_type=SOLUTIONS.Maxwell3d.EddyCurrent
    )
    pass

  def _random_choice(self, X: tuple[float, float, float, int]) -> float:
    """
    X[0]: 시작값 (inclusive)
    X[1]: 끝값 (inclusive)
    X[2]: 스텝
    X[3]: 반올림할 소수점 자릿수
    """
    start, end, step, digits = X
    choices = np.arange(start, end + step, step)
    value = np.random.choice(choices)
    return round(float(value), digits)

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
  @abstractmethod
  def set_variable(self) -> None:
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

  def _create_polyline(self, points, name, coil_width, coil_height) -> None:
    pass

  def create_region(self) -> None:
  @abstractmethod
  def assign_mesh(self) -> None:
    raise NotImplementedError("이건 인터페이스 클래스입니다. 상속받아서 내부를 작성해주세요")

  # def


if __name__ == "__main__":

  # print(sys.argv)
  if len(sys.argv) < 2:
    aedt_dir = "../pyaedt_test"
    name = "PeetsFEAdev"
  else:
    parr_idx: str = str(sys.argv[0])[-1]
    name = f"xform_{parr_idx}"
    aedt_dir = f"parrarel{parr_idx}"
  x = XformerMakerInterface(name=name, aedt_dir=aedt_dir)

  # AedtHandler.initialize(
  #   project_name="AIPDProject", project_path=Path.cwd().joinpath("../pyaedt_test"),
  #   design_name="AIPDDesign", sol_type=SOLUTIONS.Maxwell3d.EddyCurrent
  # )
  AedtHandler.peets_aedt.close_desktop()
