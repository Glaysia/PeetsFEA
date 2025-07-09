import math
import time
from typing import Iterator, Sequence
from ansys.aedt.core.modeler.cad.object_3d import Object3d
import numpy as np
from peetsfea.xformermaker import XformerMakerInterface, \
  XformerType, XEnum, peets_global_rand_seed
from peetsfea.aedthandler import *


class Project1_EE_Plana_Plana_2Series(XformerMakerInterface):
  def __init__(self, name: str, aedt_dir: str, new_desktop: bool) -> None:
    super().__init__(name, aedt_dir, new_desktop=new_desktop)

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
    v: dict[str, float] = self.v
    self.v['seed'] = peets_global_rand_seed
    AedtHandler.log(f"랜덤시드 {peets_global_rand_seed}")
    Tx_max = max(((self.v["Tx_layer_space_x"] + self.v["Tx_width"]) * math.ceil(self.v["Tx_turns"] / 2) + self.v["Tx_space_x"]),
                 ((self.v["Tx_layer_space_y"] + self.v["Tx_width"]) * math.ceil(self.v["Tx_turns"] / 2) + self.v["Tx_space_y"]))
    Rx_max = max((self.v["Rx_width"] + self.v["Rx_space_x"]),
                 (self.v["Rx_width"] + self.v["Rx_space_y"]))

    start = time.monotonic()  # 시작 시각 기록
    timeout_sec = 500
    w1_ratio = float(AedtHandler.peets_m3d.get_evaluated_value("w1_ratio"))
    while (True):
      # AedtHandler.log(str(self.v))
      print(str(self.v))
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

      def rand(key: str) -> None:
        v[key] = self._random_choice(
          r[key]
        )

      # TX 변수들 :
      turns = int(v["Tx_turns"])
      hTurns = turns // 2
      space_x = v["Tx_space_x"]
      space_y = v["Tx_space_y"]
      width = v["Tx_width"]
      l1_center = v["l1_center"]
      l2 = v["l2"]
      Tx_tap = v["Tx_tap"]
      Tx_total_width_x = (hTurns) * (v["Tx_space_x"]) + v["Tx_width"]
      Tx_total_width_y = (hTurns) * (v["Tx_space_y"]) + v["Tx_width"]

      # TX 불리언 변수들 :

      권선_가닥이_너무두꺼워서_서로_겹치지는_않는가 = not (
        width < min(space_x, space_y)
      )

      권선_X너비가_코일에_들어가는가 = not (
        0.3 * l2 < Tx_total_width_x < 0.97 * l2
      )

      권선_Y너비가_적당한가 = not (
        Tx_total_width_y < 0.97 * Tx_tap
      )

      if turns in (0, 1):
        권선_가닥이_너무두꺼워서_서로_겹치지는_않는가 = False
        권선_X너비가_코일에_들어가는가 = not (
          0.6 * l2 < (v["Tx_width"]) < 0.97 * l2
        )
        권선_Y너비가_적당한가 = False

      bool_list_txcoil = []
      bool_list_txcoil.append(권선_가닥이_너무두꺼워서_서로_겹치지는_않는가)
      bool_list_txcoil.append(권선_X너비가_코일에_들어가는가)
      bool_list_txcoil.append(권선_Y너비가_적당한가)

      # RX 변수들 :
      turns = int(v["Rx_turns"])
      hTurns = turns // 2
      space_x = v["Rx_space_x"]
      space_y = v["Rx_space_y"]
      width = v["Rx_width"]
      l1_center = v["l1_center"]
      l2 = v["l2"]
      Rx_tap = v["Rx_tap"]
      Rx_total_width_x = (hTurns) * (v["Rx_space_x"]) + v["Rx_width"]
      Rx_total_width_y = (hTurns) * (v["Rx_space_y"]) + v["Rx_width"]

      # RX 불리언 변수들 :

      권선_가닥이_너무두꺼워서_서로_겹치지는_않는가 = not (
        width < min(space_x, space_y)
      )

      권선_X너비가_코일에_들어가는가 = not (
        0.3 * l2 < Rx_total_width_x < 0.97 * l2
      )

      권선_Y너비가_적당한가 = not (
        Rx_total_width_y < 0.97 * Rx_tap
      )

      if turns in (0, 1):
        권선_가닥이_너무두꺼워서_서로_겹치지는_않는가 = False
        권선_X너비가_코일에_들어가는가 = not (
          0.6 * l2 < (v["Rx_width"]) < 0.97 * l2
        )
        권선_Y너비가_적당한가 = False

      bool_list_rxcoil = []
      bool_list_rxcoil.append(권선_가닥이_너무두꺼워서_서로_겹치지는_않는가)
      bool_list_rxcoil.append(권선_X너비가_코일에_들어가는가)
      bool_list_rxcoil.append(권선_Y너비가_적당한가)

      # 코어 변수들
      A = 2 * (v["l1_leg"] + v["l2"]) + v['l1_center']
      B = 2 * (v["l2"]) + v['l1_center']
      C = v['l1_center']
      D = v['h1'] * 0.5 + v['l1_top']
      E = v['h1'] / 2
      l1_leg = v["l1_leg"]
      l1_top = v["l1_top"]
      BpA = B / A
      CpA = C / A
      DpA = D / A
      EpA = E / A
      l1lpA = l1_leg / A
      l1tpA = l1_top / A
      w1pA = v['w1'] / A
      l2pA = v['l2'] / A

      # 코어 불리언 변수들
      B적당 = not (0.3 < BpA < 1)
      C적당 = not (0.07 < CpA < 0.6)
      D적당 = not (0.08 < DpA < 1.5)
      E적당 = not (0.02 < EpA < 1.5)
      l1_leg적당 = not (0.02 < l1lpA < 0.5)
      l1_top적당 = not (0.02 < l1tpA < 0.5)
      w1적당 = not (0.1 < w1pA < 3)
      l2적당 = not (0.05 < l2pA < 2)
      # 상용코어의 범위보다 훨씬 넓게 잡음
      #             B/A       C/A       D/A       E/A  l1_leg/A  l1_top/A
      # min    0.664884  0.138699  0.159375  0.079687  0.069349  0.068493
      # max    0.861301  0.335116  0.780952  0.619048  0.167558  0.178744
      #          w1/A      l2/A
      #          0.237013  0.164884
      #          0.793750  0.361301
      코어가_너무기형적이진_않은가 = any(
        (B적당, C적당, D적당, E적당, l1_leg적당, l1_top적당, w1적당, l2적당))
      # 시뮬이 너무 오래 돌 것 같아서 제약조건 추가함

      bool_list_core = []
      # bool_list.append(권선_시작_x좌표가_적당한가)
      # bool_list.append(권선_X_전체너비가_적당한가)
      # bool_list.append(권선_가닥이_너무두꺼워서_서로_겹치지는_않는가)
      bool_list_core.append(코어가_너무기형적이진_않은가)
      # if v["Rx_width"] > v["l2"]:
      #   v["Rx_width"] = self._random_choice(
      #     r["Rx_width"]
      #   )
      # elif (v["w1"] * w1_ratio + 2 * v["Rx_space_x"]) < v["l1_center"]:
      #   v["Rx_space_x"] = self._random_choice(
      #     r["Rx_space_x"]
      #   )
      # else:

      if any(bool_list_core):
        # 코어가 너무 기형적이진 않으냐 <<< 얘네를 분리해라
        rand("l1_leg")
        rand("l2")
        rand("w1")
        rand("l1_center")
        rand("h1")
        rand("l1_top")
      elif any(bool_list_txcoil):
        rand("Tx_space_x")
        rand("Tx_space_y")
        rand("Tx_width")
        rand("Tx_turns")
        rand("Tx_tap")
      elif any(bool_list_rxcoil):
        rand("Rx_space_x")
        rand("Rx_space_y")
        rand("Rx_width")
        rand("Rx_turns")
        rand("Rx_tap")
      else:
        break

    del self.r

    for k, va in self.v.items():
      AedtHandler.peets_m3d[k] = f"{va}mm"

    AedtHandler.peets_m3d["w1_ratio"] = f'(w1-2*l2)/w1'
    self.is_validated = True

  def create_core(self) -> None:
    if not hasattr(self, "mat"):
      raise RuntimeError("set_material() must be called before create_core()")

    if not self.is_validated:
      raise RuntimeError(
        "validate_variable() must be called before validate_variable()")
    from peetsfea.xformermaker.project1 import CoreBuilder4Project1

    builder = CoreBuilder4Project1(
      self.modeler, self._create_box, self.mat, self.o3ds)  # type: ignore
    builder.build()

  def create_winding(self) -> None:
    self.create_winding_new("Tx")
    self.create_winding_new("Rx", False, True)
    self.create_winding_new("Rx", True, True)

  def create_winding_new(self, coil: str, second=None, mirrorX=None):
    from peetsfea.xformermaker.project1 import WindingBuilder4Project1
    builder = WindingBuilder4Project1(
      self.modeler, self._create_polyline, self.v, self.o3ds)
    builder.build(coil, second, mirrorX)

  def create_exctation(self) -> None:
    pass

  def assign_mesh(self) -> None:
    pass


if __name__ == "__main__":
  import os
  import sys
  root = '/home/harry/opt/AnsysEM/v242/Linux64'
  os.environ.setdefault('ANSYSEM_ROOT242', root)

  # print(sys.argv)
  if len(sys.argv) < 2:
    aedt_dir = "../pyaedt_test"
    name = "PeetsFEAdev"
  else:
    parr_idx: str = str(sys.argv[0])[-1]
    name = f"xform_{parr_idx}"
    aedt_dir = f"parrarel{parr_idx}"
  while True:
    try:
      sim = Project1_EE_Plana_Plana_2Series(
        name=name, aedt_dir=aedt_dir, new_desktop=False)
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

      # values["EXX"] = map(itemgetter("EXX"), ccore)
      # values["w1"] = map(itemgetter("w1"), ccore)
      # values["l1_leg"] = map(itemgetter("l1_leg"), ccore)
      # values["l1_top"] = map(itemgetter("l1_top"), ccore)
      # values["l1_center"] = map(itemgetter("l1_center"), ccore)
      # values["l2"] = map(itemgetter("l2"), ccore)
      # values["h1"] = map(itemgetter("h1"), ccore)

      ranges["w1"] = [2, 52, 1.2, 1]
      ranges["l1_leg"] = [1, 12, 1.2, 3]
      ranges["l1_top"] = [1, 12, 1.2, 3]
      ranges["l1_center"] = [2, 25, 1.2, 3]
      ranges["l2"] = [2, 25, 1.2, 3]
      ranges["h1"] = [4, 52, 1.2, 2]

      ranges["l2_tap"] = [0, 0, 1, 0]
      ranges["ratio"] = [0.5, 0.50, 0.01, 2]

      ranges["Tx_turns"] = [2, 14, 1, 0]  # rand TX
      ranges["Tx_tap"] = [2, 35, 1, 0]
      ranges["Tx_height"] = [0.035, 0.175, 0.035, 3]
      ranges["Tx_preg"] = [0.01, 0.1, 0.01, 2]

      ranges["Rx_turns"] = [0, 0, 1, 0]  # rand RX
      ranges["Rx_tap"] = [2, 35, 1, 0]
      ranges["Rx_height"] = [0.035, 0.175, 0.035, 3]
      ranges["Rx_preg"] = [0.01, 0.1, 0.01, 2]

      ranges["g1"] = [0.1, 0.1, 0.01, 2]
      ranges["g2"] = [0, 0.5, 0.01, 2]

      ranges["Tx_space_x"] = [0.1, 7, 0.1, 2]  # rand TX
      ranges["Tx_space_y"] = [0.1, 7, 0.1, 2]  # rand TX
      ranges["Rx_space_x"] = [0.1, 5, 0.1, 1]
      ranges["Rx_space_y"] = [0.1, 5, 0.1, 1]

      ranges["core_N_w1"] = [0, 30, 1, 0]
      ranges["core_P_w1"] = [0, 30, 1, 0]

      ranges["Tx_layer_space_x"] = [0.1, 5, 0.1, 1]
      ranges["Tx_layer_space_y"] = [0.1, 5, 0.1, 1]
      ranges["Rx_layer_space_x"] = [0.1, 5, 0.1, 1]
      ranges["Rx_layer_space_y"] = [0.1, 5, 0.1, 1]

      ranges["Tx_width"] = [0.1, 12, 0.01, 2]  # rand TX
      ranges["Rx_width"] = [1, 20, 0.1, 1]

      sim.set_variable_byvalue(input_values=values)
      sim.set_variable_byrange(input_ranges=ranges)
      sim.set_material()

      try:
        sim.validate_variable()
      except KeyboardInterrupt as e:
        print(e)
        sim.is_validated = True
        sim.create_core()
        exit()

      sim.create_core()
      sim.create_winding()

    except Exception as e:
      print(f"error {e}, retry")
      np.random.seed((seed := int(time.time_ns() % (2**32))))
    else:
      break

  # print(sim.template[XEnum.EEPlanaPlana2Series]["coil_keys"])
  # x.set_material()
  # AedtHandler.initialize(
  #   project_name="AIPDProject", project_path=Path.cwd().joinpath("../pyaedt_test"),
  #   design_name="AIPDDesign", sol_type=SOLUTIONS.Maxwell3d.EddyCurrent
  # )
  # AedtHandler.peets_aedt.close_desktop()
