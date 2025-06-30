import logging
from typing import Any, Sequence, cast

# from Coil import Coil
# from Peets import CoilDesignParam
from ansys.aedt.core import Desktop, Maxwell3d                        # type: ignore
from ansys.aedt.core.generic.constants import SOLUTIONS               # type: ignore
from ansys.aedt.core.modules.solve_setup import SetupMaxwell          # type: ignore
from ansys.aedt.core.modules.solve_sweeps import SetupProps           # type: ignore
from ansys.aedt.core.modules.material import Material                 # type: ignore
from ansys.aedt.core.modeler.cad.object_3d import Object3d
from ansys.aedt.core.generic.general_methods import active_sessions   # type: ignore
from ansys.aedt.core.internal.grpc_plugin_dll_class import AedtPropServer
from pathlib import Path
import shutil


class AedtInitializationError(Exception):
  """Raised when AEDT Desktop instance creation fails."""
  pass


class PeetsLogSetter:
  def __init__(self, pid: int) -> None:
    self.pid: int = pid
    self._old_formatters: dict[logging.Handler, logging.Formatter] = {}

  def setlogger(self, pid: int) -> None:
    """
    with문 없이 앞으로 로거가 pid를 표시합니다.
    """
    self.pid: int = pid
    logger: logging.Logger = logging.getLogger("Global")
    for handler in logger.handlers:
      if isinstance(handler, logging.StreamHandler):
        # 기존 formatter 저장
        self._old_formatters[handler] = cast(
          logging.Formatter, handler.formatter)
        # PID prefix formatter 설정
        new_fmt: str = f"[PID {self.pid}] PyAEDT %(levelname)s: %(message)s"
        handler.setFormatter(logging.Formatter(new_fmt))

  def reset_logger(self) -> None:
    """
    pyaedt 로거를 초기화합니다.
    """
    logger: logging.Logger = logging.getLogger("Global")
    for handler in logger.handlers:
      if isinstance(handler, logging.StreamHandler):
        formatter = logging.Formatter("PyAEDT %(levelname)s: %(message)s")
        handler.setFormatter(formatter)

    self._old_formatters.clear()

  def __enter__(self) -> None:
    """
    with문으로 pyaedt 로거에 pid를 표시해줍니다. with문 밖으로 나가면 pyaedt 로거가 초기화됩니다.
    """
    self.setlogger(self.pid)

    return None

  def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # type: ignore
    self.reset_logger()
    return False


class MyMaxwell3d(Maxwell3d):
  def __init__(self, *args, **kargs) -> None:  # type: ignore
    super().__init__(*args, **kargs)  # type: ignore

    orig_create_box = self.modeler.create_box  # type: ignore
    orig_create_polyline = self.modeler.create_polyline  # type: ignore

    def create_box_with_error(*args, **kwargs) -> Object3d:  # type: ignore
        # 실제 박스 생성 시도
      result: Object3d = orig_create_box(*args, **kwargs)  # type: ignore
      if not result:
          # 실패했으면 명확한 예외
        raise RuntimeError("create_box() failed: returned falsy value")
      return result  # type: ignore

    def create_polyline_with_error(*args, **kwargs) -> Object3d:  # type: ignore
        # 실제 폴리라인 생성 시도
      result: Object3d = orig_create_polyline(*args, **kwargs)  # type: ignore
      if not result:
          # 실패했으면 명확한 예외
        raise RuntimeError("create_polyline() failed: returned falsy value")
      return result  # type: ignore

    self.modeler.create_box = create_box_with_error  # type: ignore
    self.modeler.create_polyline = create_polyline_with_error  # type: ignore


class AedtHandler:
  def __init__(self) -> None:
    AedtHandler.peets_aedt_pid: int = -1
    AedtHandler.peets_aedt: Desktop
    AedtHandler.peets_m3d: Maxwell3d
    AedtHandler.oproj: AedtPropServer

    AedtHandler.freq_hz: int
    # AedtHandler.input_coil: CoilDesignParam
    AedtHandler.project_name: str
    AedtHandler.project_path: Path
    AedtHandler.design_name: str
    AedtHandler.solution_type: str

    # AedtHandler.coil: Coil

  @classmethod
  def close_all_aedt(cls) -> int:
    """
    모든 aedt를 종료합니다.
    종료한 aedt의 수를 반환합니다.
    """
    pids: Sequence[int] = cls.get_aedt_pids()
    for pid in pids:
      with PeetsLogSetter(pid):
        Desktop(new_desktop=False, aedt_process_id=pid).close_desktop()

    return len(pids)

  @classmethod
  def get_aedt_pid(cls) -> int:
    """
    peets의 aedt pid를 반환합니다.
    """
    pid: Any = getattr(cls.peets_aedt, "aedt_process_id", None)
    if type(pid) != int:
      raise AedtInitializationError(
          f"AEDT 초기화 실패"
      )
    else:
      cls.peets_aedt_pid = pid
      return pid

  @classmethod
  def open_aedt(cls) -> int:
    """
    *주의 만약 여러개가 켜져있다면 하나의 aedt를 빼고 모두 종료합니다.
    peets의 aedt의 pid를 반환합니다.
    """
    pids: Sequence[int] = cls.get_aedt_pids()
    if len(pids) > 0:
      cls.peets_aedt_pid = pids[0]
      for pid in pids:
        with PeetsLogSetter(pid):
          if pid == cls.peets_aedt_pid:
            aedt = Desktop(
              non_graphical=False, new_desktop=False,
              close_on_exit=False, student_version=False, aedt_process_id=pid)
            cls.peets_aedt = aedt
            PeetsLogSetter(pid).setlogger(pid)
            return pid
          else:
            Desktop(new_desktop=False, aedt_process_id=pid).close_desktop()
      return cls.peets_aedt_pid
    else:
      aedt = Desktop(
        new_desktop=True, close_on_exit=False, student_version=False,
      )
      cls.peets_aedt = aedt
      pid: int = cls.get_aedt_pid()
      PeetsLogSetter(pid).setlogger(pid)
      return pid

  @classmethod
  def get_aedt_pids(cls) -> Sequence[int]:
    """
    켜져있는 모든 aedt의 pid 리스트를 반환합니다.
    """
    sessions: dict[int, int] = cast(dict[int, int], active_sessions())
    return list(sessions.keys())

  @classmethod
  def save_project(cls) -> bool:
    return (cls.peets_m3d.save_project(  # type: ignore[reportUnknownMemberType]
      str(cls.project_path.joinpath(f"{cls.project_name}.aedt")))
    )

  @classmethod
  def open_project(cls) -> bool:
    """
    프로젝트를 열고, 저장 여부를 반환합니다.
    """
    required = ["project_name", "design_name", "solution_type"]
    missing = [name for name in required if not hasattr(cls, name)]
    if missing:
        # use built-in AttributeError for missing class attributes
      raise AttributeError(
          f"Missing required class attribute(s): {', '.join(missing)}" +
          f"\nset_project_properties를 하고 실행해야 합니다."
      )

    cls.oproj: AedtPropServer = cls.peets_aedt.active_project(  # type: ignore
      name=cls.project_name)

    try:
      cls.peets_m3d = MyMaxwell3d(
        project=cls.project_name,
        design=cls.design_name,
        solution_type=cls.solution_type,
        new_desktop=False, close_on_exit=False,
        student_version=False, aedt_process_id=cls.peets_aedt_pid
      )

    except Exception as e:
      print(f"에러 발생{e}")
      cls.close_all_aedt()
      cls.open_project()

    cls.peets_m3d.autosave_disable()

    ret: bool = cls.save_project()

    return ret

  @classmethod
  def check_directory(cls) -> None:
    if cls.project_path.exists():
      cls.close_all_aedt()
      shutil.rmtree(cls.project_path, ignore_errors=True)
      cls.project_path.mkdir(parents=True)
      print("종료된 aedt의 수: ", cls.close_all_aedt())

  @classmethod
  def set_project_properties(
    cls, project_name: str = "AIPDProject",
    project_path: Path = Path.home().joinpath("peets_aipd").absolute(),
    design_name: str = "AIPDDesign",
    sol_type: str = SOLUTIONS.Maxwell3d.EddyCurrent,
  ) -> None:
    """
    프로젝트 이름, 출력 위치 등을 설정합니다.
    * 주의 모든 aedt가 종료될 수 있습니다.
    """
    cls.project_name: str = project_name
    cls.project_path: Path = Path(project_path)
    cls.design_name: str = design_name
    cls.solution_type: str = sol_type

  @classmethod
  def reset_desktop(cls) -> None:
    """
    대상 프로젝트를 제외하고 모두 지웁니다.
    """
    # print(cls.peets_aedt.project_list())
    project_list: Sequence[str] = cast(
      list[str], cls.peets_aedt.project_list())

    i: int = 0
    for proj in project_list:
      cls.oproj: AedtPropServer = cls.peets_aedt.active_project()  # type: ignore
      if proj != cls.oproj.GetName():  # type: ignore
        i += int(cls.peets_m3d.delete_project(proj))  # type: ignore

    print(f"{i} 개의 프로젝트를 지웠습니다.")

  @classmethod
  def reset_project(cls) -> None:
    """
    대상 디자인을 제외하고 모두 지웁니다.
    """

    _: list[Any] = cls.peets_aedt.design_list()  # type: ignore
    design_list = cast(Sequence[str], _)

    i: int = 0

    for desi in design_list:
      if desi != cls.design_name:
        i += int(cls.peets_m3d.delete_design(desi))  # type: ignore

    print(f"{i} 개의 디자인을 지웠습니다.")

  @classmethod
  def initialize(
    cls, project_name: str = "AIPDProject",
    project_path: Path = Path.home().joinpath("peets_aipd").absolute(),
    design_name: str = "AIPDDesign",
    sol_type: str = SOLUTIONS.Maxwell3d.EddyCurrent,
  ) -> None:
    """
    aedt를 초기화 합니다.
    """
    cls.set_project_properties(
        project_name, project_path,
        design_name, sol_type
    )
  # AedtHandler.check_directory()
    # print(dir(AedtHandler))
    cls.tmp = cls.project_name
    cls.project_name = "tmp"
    cls.open_aedt()

    print(f"프로젝트 저장 여부:{cls.open_project()}")
    cls.reset_desktop()

    cls.project_name = cls.tmp
    cls.reset_desktop()
    cls.reset_project()
    cls.save_project()

  # @classmethod
  # def set_input(cls, input_coil: CoilDesignParam) -> None:
  #   cls.input_coil: CoilDesignParam = input_coil


AedtHandler()
if __name__ == '__main__':

  AedtHandler.initialize(
    project_name="AIPDProject", project_path=Path.cwd().joinpath("../test"),
    design_name="AIPDDesign", sol_type=SOLUTIONS.Maxwell3d.EddyCurrent
  )
  AedtHandler.peets_aedt.close_desktop()
  # c1 = Coil()
  # coil_shape: dict[str, float] = {k: 1.0 for k in Coil.get_template()[
  #     CoilTypeEnum.EIPlanaPlana2Series]["coil_keys"]}

  # print(coil_shape)

  # c1.set_coil(
  #   coil_type=CoilTypeEnum.EIPlanaPlana2Series,
  #   coil_shape=coil_shape
  # )

  # AedtHandler.set_variable(c1)
  # AedtHandler.set_analysis()
