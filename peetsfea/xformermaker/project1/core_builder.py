
from typing import Callable, Sequence
from ansys.aedt.core.modeler.cad.object_3d import Object3d
from ansys.aedt.core.modeler.modeler_3d import Modeler3D
from ansys.aedt.core.modules.material import Material

from peetsfea.aedthandler import AedtHandler


class CoreBuilder:
  """Create transformer core geometry."""

  def __init__(
      self,
      modeler: Modeler3D,
      create_box: Callable[..., Object3d],
      material: Material,
      o3ds: dict[str, Object3d],
  ) -> None:
    self.modeler = modeler
    self._create_box = create_box
    self.mat = material
    self.o3ds = o3ds

  def build(self) -> None:
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

    origin = ["-l1_center/2", "-(w1)/2", "-(g2)/2"]
    dimension = ["l1_center", "w1", "g2"]
    self.o3ds["core_unite_sub_g1"] = self._create_box(
        origin=origin,
        sizes=dimension,
        name="core_unite_sub_g1",
        material="ferrite"
    )

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
