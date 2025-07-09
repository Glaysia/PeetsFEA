from typing import Callable, Literal, Sequence, cast
from ansys.aedt.core.modeler.cad.object_3d import Object3d
from ansys.aedt.core.modeler.modeler_3d import Modeler3D

from peetsfea.aedthandler import AedtHandler


class WindingBuilder:
  """Create winding geometry."""

  def __init__(
      self,
      modeler: Modeler3D,
      create_polyline: Callable[..., Object3d],
      variables: dict[str, float],
      o3ds: dict[str, Object3d],
  ) -> None:
    self.modeler = modeler
    self._create_polyline = create_polyline
    self.v = variables
    self.o3ds = o3ds

  def build(self, coil: str, second=None, mirrorX=None):
    if second != None or mirrorX != None:
      coil_name = f"{coil}_{second}_{mirrorX}"
    else:
      coil_name = coil

    AedtHandler.peets_m3d[f"{coil}_hTurns"] = int(self.v[f"{coil}_turns"]) // 2

    START_X = f"(-({coil}_hTurns * {coil}_space_x + {coil}_width + l1_center/2 - 0.5*{coil}_width))"
    START_Y = f"({coil}_hTurns * {coil}_space_y + {coil}_width + w1/2 - 0.5*{coil}_width)"
    START_Z = f"0"
    points = [
      ["l1_center+100mm", START_Y, START_Z],
      [START_X, START_Y, START_Z]
    ]

    def from_expression(exp: str) -> float:  # 단위 무조건 mm
      m3d = AedtHandler.peets_m3d
      m3d['peets_tmp'] = exp
      return 1000 * float(f"{m3d.get_evaluated_value('peets_tmp'):.9f}")

    def abs_exp(exp: str) -> str:
      value = from_expression(exp)
      sign = is_positive = value >= 0

      if is_positive:
        if exp[2] == '-':
          abs_value = exp[2:-1]
        else:
          abs_value = exp
      else:
        abs_value = exp[2:-1]

      return abs_value

    def flip(point: list[str], axis: Literal['x', 'y', 'z']) -> list[str]:
      idx = {'x': 0, 'y': 1, 'z': 2}
      idx = idx[axis]
      new_point = point[:]
      old_value = point[idx]
      abs_value = abs_exp(old_value)
      is_positive = from_expression(old_value) > 0
      new_value = f"(-{abs_value})" if is_positive else abs_value
      new_point[idx] = new_value
      return new_point

    def shrink(point: list[str], axis: Literal['x', 'y', 'z'], exp: str) -> list[str]:
      idx = {'x': 0, 'y': 1, 'z': 2}
      idx = idx[axis]
      new_point = point[:]
      value = from_expression(point[idx])
      old_value = point[idx]
      sign = is_positive = value >= 0
      sign_str: Literal[''] | Literal['-'] = "" if sign else "-"
      abs_value = abs_exp(old_value)

      new_value = f"({sign_str}({abs_value}-{exp}))"
      new_point[idx] = new_value

      return new_point

    turn = int(self.v[f'{coil}_turns'])
    is_turn_odd = (turn % 2 == 1)

    for _ in range(turn):
      if _ % 2 == 0:
        last = points[-1]
        points.append(flip(last, 'y'))

        last = points[-1]
        points.append(flip(last, 'x'))
      else:
        last = points[-1]
        points.append(flip(shrink(last, 'y', f"{coil}_space_{'y'}"), 'y'))

        last = points[-1]
        points.append(flip(shrink(last, 'x', f"{coil}_space_{'x'}"), 'x'))

    last = points[-1]
    sign = "+" if is_turn_odd else "-"
    msign = "-" if is_turn_odd else "+"
    points.append(
      shrink(last, 'y', f'{abs_exp(last[1])} - 1.5*{coil}_width'))

    points_connect = []
    last_x, last_y, last_z = points[-1]
    points_connect.append(
      [last_x, 0, from_expression(f"{last_z}-({coil}_height/2)")])
    points_connect.append(
      [last_x, 0, from_expression(f"{last_z}+({coil}_height/2)")])

    o3ds = self.o3ds

    o3ds[f'{coil_name}_connect'] = self.modeler.create_polyline(
        points_connect, name=f"{coil_name}_connect", xsection_type="Circle", xsection_width=f"{coil}_width*0.8", xsection_num_seg=12)  # type: ignore

    o3ds[f'{coil_name}_1'] = self._create_polyline(
        points=points, name=f"{coil_name}_1", coil_width=f"{coil}_width", coil_height=f"{coil}_height")

    self.modeler.copy(assignment=o3ds[f'{coil_name}_1'])
    self.modeler.paste()
    o3ds[f'{coil_name}_2'] = self.modeler.get_object_from_name(  # type: ignore
      assignment=f"{coil_name}_2")
    self.modeler.mirror(
      assignment=o3ds[f'{coil_name}_2'],
      origin=[0, 0, 0], vector=[0, 1, 0]
    )
    self.modeler.move(
      assignment=o3ds[f'{coil_name}_2'],
      vector=["0mm", "0mm", f"-({coil}_preg/2+{coil}_height/2)"]
    )

    self.modeler.move(
      assignment=o3ds[f'{coil_name}_1'],
      vector=["0mm", "0mm", f"{coil}_preg/2+{coil}_height/2"]
    )

    self.modeler.unite(
      assignment=[o3ds[f'{coil_name}_1'], o3ds[f"{coil_name}_2"], o3ds[f"{coil_name}_connect"]])

    o3ds[f'{coil_name}_1'].color = [255, 0, 0]
    o3ds[f'{coil_name}_1'].transparency = 0

    if mirrorX:
      self.modeler.mirror(
        assignment=o3ds[f'{coil_name}_1'],
        origin=[0, 0, 0], vector=[1, 0, 0]
      )

    if second != None and second == False:
      self.modeler.move(
        assignment=o3ds[f'{coil_name}_1'],
        vector=["0mm", "0mm", f"g2+{coil}_height/2"]
      )

    if second != None and second == True:
      self.modeler.move(
        assignment=o3ds[f'{coil_name}_1'],
        vector=["0mm", "0mm", f"-g2-{coil}_height/2"]
      )

    AedtHandler.log(f"Winding{coil_name} 생성 완료")
