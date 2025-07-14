from typing import Any, Callable
from ansys.aedt.core import Desktop, Maxwell3d
from ansys.aedt.core.modules.boundary.common import BoundaryObject
from ansys.aedt.core.modules.boundary.maxwell_boundary import MaxwellParameters
from ansys.aedt.core.modules.solve_setup import SetupMaxwell
from ansys.aedt.core.modules.solve_sweeps import SetupProps
from ansys.aedt.core.modeler.cad.object_3d import Object3d

from functools import wraps


from typing import ParamSpec, TypeVar, Callable
import functools

from ansys.aedt.core.modeler.modeler_3d import Modeler3D

P = ParamSpec("P")   # 함수의 파라미터 타입 시그니처(P.args, P.kwargs)를 담는 타입 변수
R = TypeVar("R")     # 함수의 리턴 타입을 담는 타입 변수


def assert_true(fn: Callable[P, R]) -> Callable[P, R]:
  @functools.wraps(fn)
  def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
    result = fn(*args, **kwargs)
    assert result is not False, f"{fn.__name__}이 False를 반환했습니다"
    return result
  return wrapper


class PeetsModeler(Modeler3D):
  pass


class PeetsMaxwell3d(Maxwell3d):
  @assert_true  # type: ignore
  def create_setup(  # type: ignore
    self, name: str = "MySetupAuto",
    setup_type: str | None = None,
    *args: Any, **kwargs: Any
  ) -> SetupMaxwell:
    ret: SetupMaxwell | Any = super(  # type: ignore
    ).create_setup(name=name, setup_type=setup_type, *args, **kwargs)  # type: ignore
    assert isinstance(ret, SetupMaxwell), "create_setup failed"

    return ret

  @assert_true  # type: ignore
  def assign_material(self, assignment: Object3d, material: str) -> bool:  # type: ignore
    return super().assign_material(assignment, material)  # type: ignore

  @assert_true
  def assign_radiation(self, assignment: str | list[str] | list[int], radiation: str | None = None) -> BoundaryObject | MaxwellParameters:
    return super().assign_radiation(assignment, radiation)  # type: ignore

  @assert_true
  def assign_winding(
    self,
    assignment=None, winding_type="Current",
    is_solid=True, current=1, resistance=0,
    inductance=0, voltage=0,
    parallel_branches=1,
    phase=0, name=None
  ) -> BoundaryObject | MaxwellParameters:
    ret = super().assign_winding(assignment, winding_type, is_solid, current,
                                 resistance, inductance, voltage, parallel_branches, phase, name)
    assert isinstance
    return


if __name__ == '__main__':
  # a = PeetsMaxwell3d()
  # a.create_setup()
  # a.assign_material()
  pass
