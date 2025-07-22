from typing import Dict

from ansys.aedt.core.visualization.post.post_3dlayout import PostProcessor3DLayout
from ansys.aedt.core.visualization.post.post_circuit import PostProcessorCircuit
from ansys.aedt.core.visualization.post.post_common_3d import PostProcessor3D
from ansys.aedt.core.visualization.post.post_hfss import PostProcessorHFSS
from ansys.aedt.core.visualization.post.post_icepak import PostProcessorIcepak
from ansys.aedt.core.visualization.report.standard import Standard
from peetsfea.aedthandler import *
from peetsfea.xformermaker import peets_global_rand_seed as global_seed
from peetsfea.xformermaker import XformerMakerInterface, \
  XformerType, XEnum
from ansys.aedt.core.modeler.cad.elements_3d import FacePrimitive, EdgePrimitive
from ansys.aedt.core.modules.boundary.common import BoundaryObject
from ansys.aedt.core.modeler.cad.object_3d import Object3d
from ansys.aedt.core.maxwell import Maxwell3d
from ansys.aedt.core.visualization.report.field import Fields
from ansys.aedt.core.visualization.post.post_maxwell import PostProcessorMaxwell
from pathlib import Path
from typing import Any, ItemsView, Iterator, Literal, Sequence
import time
import math
import sys
import os
import numpy as np
from datetime import datetime
import functools
import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


sim_global: dict[str, XformerMakerInterface | None | bool] = {"sim": None,"debugging":False}


def save_on_exception(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):

    start = time.monotonic()
    try:
      ret = func(*args, **kwargs)
      elapsed: str = f"{time.monotonic() - start:.4f}s"
      sim: XformerMakerInterface | None = sim_global["sim"]

      if isinstance(sim, XformerMakerInterface):
        k: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        v: str = str(func.__name__)
        sim.progress_dict[k] = (v, elapsed)

      return ret

    except Exception as e:
      close(exception=e, progress=func.__name__)
      # 예외 발생시 정보 저장하고 종료함
      raise
  return wrapper


class Project2(XformerMakerInterface):
  def __init__(
    self, name: str, aedt_dir: str,
    new_desktop: bool,
    non_graphical: bool
  ) -> None:
    super().__init__(name, aedt_dir, new_desktop=new_desktop, non_graphical=non_graphical)

    self.v: dict[str, float] = {}
    self.comments: list[str] = []

    self.xformer_type: XformerType = XEnum.EEPlanaPlana2Series
    self.per: int = 3000
    self.freq_khz: int = 140
    self.coils_main: list[str] = []
    global sim_global
    sim_global['sim'] = self
    self.sim_global: Dict[str, XformerMakerInterface | None | bool] = sim_global

  @save_on_exception
  def random_ranges(self) -> dict[str, list[float]]:
    ranges = {}

    ranges["w1"] = [2, 52, 1.2, 1]
    ranges["l1_leg"] = [1, 12, 1.2, 3]
    ranges["l1_top"] = [1, 12, 1.2, 3]
    ranges["l1_center"] = [2, 25, 1.2, 3]
    ranges["l2"] = [2, 25, 1.2, 3]
    ranges["h1"] = [4, 52, 1.2, 2]

    ranges["l2_tap"] = [0, 0, 1, 0]
    ranges["ratio"] = [0.5, 0.50, 0.01, 2]

    # ranges["Tx_turns"] = [2, 14, 1, 0]  # rand TX
    ranges["Tx_turns"] = [14, 14, 1, 0]  # rand TX
    ranges["Tx_tap"] = [2, 35, 1, 0]
    ranges["Tx_height"] = [0.035, 0.175, 0.035, 3]
    ranges["Tx_preg"] = [0.01, 0.1, 0.01, 2]

    # ranges["Rx_turns"] = [0, 2, 1, 0]  # rand RX
    ranges["Rx_turns"] = [2, 2, 1, 0]  # rand RX
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

    return ranges

  @save_on_exception
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

  @save_on_exception
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

  @save_on_exception
  def validate_variable(self):
    try:
      self.validate_variable_new()
    except KeyboardInterrupt as e:
      print(e)
      self.is_validated = True
      self.create_core()
      exit()

    for k, va in self.v.items():
      AedtHandler.peets_m3d[k] = f"{va}mm"

    AedtHandler.peets_m3d["w1_ratio"] = f'(w1-2*l2)/w1'
    self.is_validated = True

  @save_on_exception
  def validate_variable_new(self) -> None:
    self.validate_variable_4Core()
    self.validate_variable_4Tx()
    self.validate_variable_4Rx()

  @save_on_exception
  def validate_variable_4Core(self) -> None:
    r = self.r
    v: dict[str, float] = self.v
    self.v['seed'] = global_seed
    AedtHandler.log(f"랜덤시드validate_variable_4Core {global_seed}")

    start = time.monotonic()  # 시작 시각 기록
    timeout_sec = 10

    while (True):
      # AedtHandler.log(str(self.v))
      print(str(self.v))
      if time.monotonic() - start > timeout_sec:
        self.is_validated = True
        self.create_core()
        raise TimeoutError(
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

      # 코어 변수들
      v["A"] = A = 2 * (v["l1_leg"] + v["l2"]) + v['l1_center']
      v["B"] = B = 2 * (v["l2"]) + v['l1_center']
      v["C"] = C = v['l1_center']
      v["D"] = D = v['h1'] * 0.5 + v['l1_top']
      v["E"] = E = v['h1'] / 2
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

      bool_list_core.append(코어가_너무기형적이진_않은가)

      if any(bool_list_core):
        # 코어가 너무 기형적이진 않으냐 <<< 얘네를 분리해라
        rand("l1_leg")
        rand("l2")
        rand("w1")
        rand("l1_center")
        rand("h1")
        rand("l1_top")

        rand("Rx_space_x")
        rand("Rx_space_y")
        rand("Rx_width")
        rand("Rx_turns")
        rand("Rx_tap")
      else:
        break

    for k, va in self.v.items():
      AedtHandler.peets_m3d[k] = f"{va}mm"

    AedtHandler.peets_m3d["w1_ratio"] = f'(w1-2*l2)/w1'
    self.is_validated = True

  @save_on_exception
  def validate_variable_4Tx(self) -> None:
    r = self.r
    v: dict[str, float] = self.v
    self.v['seed'] = global_seed
    AedtHandler.log(f"랜덤시드validate_variable_4Tx {global_seed}")

    start = time.monotonic()  # 시작 시각 기록
    timeout_sec = 10
    w1_ratio = float(AedtHandler.peets_m3d.get_evaluated_value("w1_ratio"))
    while (True):
      # AedtHandler.log(str(self.v))
      print(str(self.v))
      if time.monotonic() - start > timeout_sec:
        self.is_validated = True
        self.create_core()
        raise TimeoutError(
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

      if any(bool_list_txcoil):
        rand("Tx_space_x")
        rand("Tx_space_y")
        rand("Tx_width")
        rand("Tx_turns")
        rand("Tx_tap")
      else:
        break

    for k, va in self.v.items():
      AedtHandler.peets_m3d[k] = f"{va}mm"

    AedtHandler.peets_m3d["w1_ratio"] = f'(w1-2*l2)/w1'
    self.is_validated = True

  @save_on_exception
  def validate_variable_4Rx(self) -> None:
    r = self.r
    v: dict[str, float] = self.v
    self.v['seed'] = global_seed
    AedtHandler.log(f"랜덤시드validate_variable_4Rx {global_seed}")

    start = time.monotonic()  # 시작 시각 기록
    timeout_sec = 10

    while (True):
      # AedtHandler.log(str(self.v))
      print(str(self.v))
      if time.monotonic() - start > timeout_sec:
        self.is_validated = True
        self.create_core()
        raise TimeoutError(
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

      if any(bool_list_rxcoil):
        rand("Rx_space_x")
        rand("Rx_space_y")
        rand("Rx_width")
        rand("Rx_turns")
        rand("Rx_tap")
      else:
        break

    for k, va in self.v.items():
      AedtHandler.peets_m3d[k] = f"{va}mm"

    AedtHandler.peets_m3d["w1_ratio"] = f'(w1-2*l2)/w1'
    self.is_validated = True

  @save_on_exception
  def create_core(self) -> None:
    if not hasattr(self, "mat"):
      raise AttributeError(
        "set_material() must be called before create_core()")

    if not self.is_validated:
      raise RuntimeError(
        "validate_variable() must be called before validate_variable()")
    from peetsfea.xformermaker.project1 import CoreBuilder4Project1

    builder = CoreBuilder4Project1(
      self.modeler, self._create_box, self.mat, self.o3ds)  # type: ignore
    builder.build()

  @save_on_exception
  def create_winding(self) -> None:
    start = time.monotonic()  # 시작 시각 기록
    timeout_sec = 30
    while True:
      if time.monotonic() - start > timeout_sec:
        raise Exception(
            f"create_winding() timed out after {timeout_sec} seconds")

      try:
        for k, v in self.o3ds.items():
          if "Tx" in k or "Rx" in k:
            if 'delete' in dir(v):
              v.delete()
            self.o3ds[k] = None  # type: ignore

        self.validate_variable_4Tx()
        self.validate_variable_4Rx()
        self.create_winding_new("Tx")
        self.create_winding_new("Rx", False, True)
        self.create_winding_new("Rx", True, True)
      except Exception as e:
        if 'color' in str(e):
          print(f"error while create_winding {e}, retry")

          set_random_seed(e)
        else:
          raise e

      else:
        break

  @save_on_exception
  def create_winding_new(self, coil: str, second=None, mirrorX=None):
    from peetsfea.xformermaker.project1 import WindingBuilder4Project1
    builder = WindingBuilder4Project1(
      self.modeler, self._create_polyline, self.v, self.o3ds)
    coil_name = builder.build(coil, second, mirrorX)
    self.coils_main.append(coil_name)

  @save_on_exception
  def create_excitation(self) -> None:
    # o3ds = self.o3ds
    terminals: dict[str, int] = {}
    faces = {}

    for i in self.coils_main:  # ['Tx_1', 'Rx_False_True_1', 'Rx_True_True_1']
      def get_face_coo(face_id: int):
        face: FacePrimitive | Literal[False] = self.modeler.get_face_by_id(
          face_id)
        assert face, "get_face_by_id"
        center: list[float] = face.center  # type: ignore

        return (face_id, center)

      # [(1,[1.1,1.1,2.2]),(1,[1.1,1.1,2.2]),(1,[1.1,1.1,2.2]),]
      faces[i] = map(get_face_coo, self.modeler.get_object_faces(assignment=i))

    def my_sort(k: str, value: list[tuple[int, list[float]]]):
      reverse = 'Tx' in k
      return sorted(value, key=lambda pair: pair[1][1], reverse=reverse)[:2]

    faces = {k: my_sort(k, v) for k, v in faces.items()}
    M3D: Maxwell3d = AedtHandler.peets_m3d

    (face_a_id, face_a_center), (
      face_b_id, face_b_center) = faces['Tx_1']

    if face_b_center[0] < 0:
      Tx_1_i, Tx_1_o = face_b_id, face_a_id
    else:
      Tx_1_i, Tx_1_o = face_a_id, face_b_id

    (face_a_id, face_a_center), (
      face_b_id, face_b_center) = faces['Rx_False_True_1']

    if face_b_center[0] < 0:
      Rx_1_i, Rx_1_o = face_b_id, face_a_id
    else:
      Rx_1_i, Rx_1_o = face_a_id, face_b_id

    (face_a_id, face_a_center), (
      face_b_id, face_b_center) = faces['Rx_True_True_1']

    if face_b_center[0] < 0:
      Rx_2_i, Rx_2_o = face_b_id, face_a_id
    else:
      Rx_2_i, Rx_2_o = face_a_id, face_b_id

    M3D.assign_coil(
      assignment=Tx_1_i, conductors_number=1,
      polarity='Positive', name='Tx_1_icoil'
    )
    M3D.assign_coil(
      assignment=Tx_1_o, conductors_number=1,
      polarity='Negative', name='Tx_1_ocoil'
    )

    M3D.assign_coil(
      assignment=Rx_1_i, conductors_number=1,
      polarity='Positive', name='Rx_1_icoil'
    )
    M3D.assign_coil(
      assignment=Rx_1_o, conductors_number=1,
      polarity='Negative', name='Rx_1_ocoil'
    )

    M3D.assign_coil(
      assignment=Rx_2_i, conductors_number=1,
      polarity='Positive', name='Rx_2_icoil'
    )
    M3D.assign_coil(
      assignment=Rx_2_o, conductors_number=1,
      polarity='Negative', name='Rx_2_ocoil'
    )

    Tx1_winding = M3D.assign_winding(
      assignment=[], winding_type="Current", is_solid=True, current=4.2, name="Tx1"  # type: ignore
    )
    Rx1_winding = M3D.assign_winding(
      assignment=[], winding_type="Current", is_solid=True, current=12.6, name="Rx1"  # type: ignore
    )
    Rx2_winding = M3D.assign_winding(
      assignment=[], winding_type="Current", is_solid=True, current=0.0, name="Rx2"  # type: ignore
    )
    assert Tx1_winding, "assign_winding failed Tx1_winding"
    assert Rx1_winding, "assign_winding failed Rx1_winding"
    assert Rx2_winding, "assign_winding failed Rx2_winding"

    M3D.add_winding_coils(
        assignment=Tx1_winding.name,
        coils=['Tx_1_icoil', 'Tx_1_ocoil'],
      )
    M3D.add_winding_coils(
        assignment=Rx1_winding.name,
        coils=['Rx_1_icoil', 'Rx_1_ocoil'],
      )
    M3D.add_winding_coils(
        assignment=Rx2_winding.name,
        coils=['Rx_2_icoil', 'Rx_2_ocoil'],
      )

    M3D.assign_matrix(
      assignment=[Tx1_winding.name, Rx1_winding.name, Rx2_winding.name],
      matrix_name="Matrix1"
    )

  @save_on_exception
  def assign_mesh(self) -> None:
    temp_list = list()
    temp_list.append(f"Tx_1")
    temp_list.append(f"Rx_True_True_1")
    temp_list.append(f"Rx_False_True_1")
    skindepth = f"{math.sqrt(1.7*10**(-8)/math.pi/140/10**3/0.999991/4/math.pi/10**(-7))*10**3}mm"
    mesh = AedtHandler.peets_m3d.mesh
    if mesh is None:
      raise AedtInitializationError

    mesh.assign_skin_depth(
      assignment=temp_list, skin_depth=skindepth,
      triangulation_max_length="12.2mm"
    )

  @save_on_exception
  def validate_design(self) -> None:
    v = AedtHandler.peets_m3d.validate_simple()
    assert bool(v), "validate_simple failed"

  @save_on_exception
  def analyze_all(self) -> None:
    setting = {
      'cores': 1, 'tasks': 1,
      'gpus': 1, 'use_auto_settings': False
    }
    self.setting = setting

    AedtHandler.peets_m3d.analyze_setup(
      **setting
    )
    AedtHandler.peets_m3d.analyze(
      **setting
    )

  @save_on_exception
  def _get_magnetic_report(self) -> bool:
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

    result_expressions = [item[0] for item in get_result_list]

    import tempfile
    from ansys.aedt.core.visualization.report.standard import Standard
    assert AedtHandler.peets_m3d.post, "post 안됨"

    report = AedtHandler.peets_m3d.post.create_report(
        expressions=result_expressions,
        setup_sweep_name=None,
        domain='Sweep',
        variations=None,
        primary_sweep_variable=None,
        secondary_sweep_variable=None,
        report_category=None,
        plot_type='Data Table',
        context=None,
        subdesign_id=None,
        polyline_points=1001,
        plot_name="simulation parameter"
    )  # type: ignore
    report: Standard = report
    assert report, "report 실패"
    assert isinstance(report, Standard), "report 실패"

    # export CSV into the temp folder
    csv_path = self.export_report_to_csv(
      plot_name=report.plot_name
    )

    print(self.data)
    data1 = self.data[report.plot_name]

    for itr, (column_name) in enumerate(data1.columns):

      data1[column_name] = abs(data1[column_name])

      if itr == 0:  # delete "Freq [kHz]" columns
        data1 = data1.drop(columns=column_name)
        continue

      if "[pH]" in column_name:  # consider error case
        return False
      elif "[nH]" in column_name:
        data1[column_name] = data1[column_name] * 1e-3
      elif "[uH]" in column_name:
        data1[column_name] = data1[column_name] * 1e+0
      elif "[mH]" in column_name:
        data1[column_name] = data1[column_name] * 1e+3
      elif "[H]" in column_name:  # consider error case
        return False

    data1.columns = ["Ltx", "Lrx1", "Lrx2", "M1", "M2",
                     "k1", "k2", "Lmt", "Lmr1", "Lmr2", "Llt", "Llr1", "Llr2"]

    self.Lmt = data1.iloc[0, 7]

    return True

  @save_on_exception
  def get_input_parameter(self):
    import pandas as pd
    self.input_parameter = pd.DataFrame.from_dict([self.v])  # type: ignore

  @save_on_exception
  def _get_copper_loss_parameter(self):

    # ==============================
    # get copper loss data
    # ==============================
    coils = self.coils_main[:]
  @save_on_exception
  def __create_B_field(self):
    M3D = AedtHandler.peets_m3d

    leg_left = M3D.modeler.create_rectangle(
        orientation="XY",
        origin=["B/2", "-(w1)/2", "g1/2"],
        sizes=["(A-B)/2", "w1"],
        name="leg_left"
    )

    leg_left.model = False

    leg_center = M3D.modeler.create_rectangle(
        orientation="XY",
        origin=["-C/2", "-(w1)/2", "g1/2"],
        sizes=["C", "w1"],
        name="leg_center"
    )
    leg_center.model = False

    leg_right = M3D.modeler.create_rectangle(
        orientation="XY",
        origin=["-B/2", "-(w1)/2", "g1/2"],
        sizes=["-(A-B)/2", "w1"],
        name="leg_right"
    )
    leg_right.model = False

    leg_top_left = M3D.modeler.create_rectangle(
        orientation="YZ",
        origin=['-(B-l2)/2', '-(w1)/2', 'h1/2+l1_top'],
        sizes=['w1', '-(D-E)']
    )
    leg_top_left.model = False
    leg_top_left.name = "leg_top_left"

    leg_bottom_left = M3D.modeler.create_rectangle(
        orientation="YZ",
        origin=['-(B-l2)/2', '-(w1)/2', '-(h1/2+l1_top)'],
        sizes=['w1', '(D-E)']
    )
    leg_bottom_left.model = False
    leg_bottom_left.name = "leg_bottom_left"

    leg_top_right = M3D.modeler.create_rectangle(
        orientation="YZ",
        origin=['(B-l2)/2', '-(w1)/2', 'h1/2+l1_top'],
        sizes=['w1', '-(D-E)']
    )
    leg_top_right.model = False
    leg_top_right.name = "leg_top_right"

    leg_bottom_right = M3D.modeler.create_rectangle(
        orientation="YZ",
        origin=['(B-l2)/2', '-(w1)/2', '-(h1/2+l1_top)'],
        sizes=['w1', '(D-E)']
    )
    leg_bottom_right.model = False
    leg_bottom_right.name = "leg_bottom_right"

    l = self.o3ds["B_fields"] = []
    l.append(leg_left)
    l.append(leg_center)
    l.append(leg_right)

    l.append(leg_top_left)
    l.append(leg_bottom_left)
    l.append(leg_top_right)
    l.append(leg_bottom_right)

  @save_on_exception
  def coreloss_project(self):
    M3D: Maxwell3d = AedtHandler.peets_m3d
    M3D.duplicate_design("script_coreloss")
    M3D.set_active_design("script_coreloss")
    to_delete = [
      exc for exc in M3D.design_excitations if exc.name in self.coils_main]

    for exc in to_delete:
      exc.delete()

      self.magnetizing_current = 390 * \
          math.sqrt(2) / 2 / math.pi / 140000 / self.Lmt / 10**(-6)


def close(exception: Exception | None = None, progress: str = "") -> None:
  global sim_global
  import uuid
  import json
  sim: XformerMakerInterface | None = sim_global["sim"]
  if isinstance(sim, XformerMakerInterface):
    sim.progress = progress
    sim.end_time_pretty = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sim.end_time = str(time.monotonic_ns())

  base_tmp = Path("./tmp")
  base_tmp.mkdir(exist_ok=True)

  data = getattr(sim, "__dict__", {})
  data['exception'] = exception

  unique_name = f"sim_dict_{uuid.uuid4().__str__()}.json"
  json_path = os.path.join(base_tmp, unique_name)
  with open(json_path, "w") as f:
    json.dump(data, f, default=lambda o: repr(o), indent=2)

  print("saved JSON to", json_path)

  if not (exception == None or debugging):
    # 예외가 발생하지 않았으면 끝내지 않음
    # 디버깅시엔 끝내지 않음
    exit()


def set_random_seed(exception: Exception | None = None, seed=0, fixed=False, is_manual: dict = {}):
  global global_seed
  if len(is_manual) != 0:
    fixed = True

  if fixed:
    is_manual[0] = True
    print(is_manual)
    if seed != 0:
      global_seed = seed
  else:
    global_seed = int(time.time_ns() % (2**32))

  np.random.seed(global_seed)
  import tempfile
  close(exception=None, progress='set_random_seed')


@save_on_exception
def project2_start() -> None:
  global sim_global
  root = '/home/harry/opt/AnsysEM/v242/Linux64'

  os.environ.setdefault('ANSYSEM_ROOT242', root)

  # print(sys.argv)
  if len(sys.argv) < 2:
    aedt_dir = "../pyaedt_test"
    name = "PeetsFEAdev"
    non_graphical = False
  else:
    parr_idx: str = str(sys.argv[0])[-1]
    name = f"xform_{parr_idx}"
    aedt_dir = f"parrarel{parr_idx}"
    non_graphical = True

  set_random_seed(None, 3019364285, True)
  # set_random_seed(None, time.time_ns() % (2**32), True)
  sim = Project2(
    name=name,
    aedt_dir=aedt_dir,
    new_desktop=False,
    non_graphical=non_graphical
  )
  values: dict[str, Iterator[float | str]] = {}
  ranges: dict[str, list[float]] = {}

  ranges.update(sim.random_ranges())

  sim.set_variable_byvalue(input_values=values)
  sim.set_variable_byrange(input_ranges=ranges)
  sim.set_material()
  sim.set_analysis(sim.freq_khz)
  sim.validate_variable()

  sim.create_core()
  sim.create_winding()
  sim.assign_mesh()
  sim.create_region()
  sim.create_excitation()
  sim.validate_design()
  sim.analyze_all()

  close()


if __name__ == '__main__':
  project2_start()
