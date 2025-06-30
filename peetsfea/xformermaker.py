from peetsfea import AedtHandler
from ansys.aedt.core.generic.constants import SOLUTIONS
from pathlib import Path
class XformerMaker:
  pass

if __name__ == "__main__":
  AedtHandler.initialize(
    project_name="AIPDProject", project_path=Path.cwd().joinpath("../pyaedt_test"),
    design_name="AIPDDesign", sol_type=SOLUTIONS.Maxwell3d.EddyCurrent
  )
  # AedtHandler.peets_aedt.close_desktop()
