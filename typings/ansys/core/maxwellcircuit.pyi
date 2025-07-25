"""
This type stub file was generated by pyright.
"""

from ansys.aedt.core.application.analysis_maxwell_circuit import AnalysisMaxwellCircuit
from ansys.aedt.core.generic.general_methods import pyaedt_function_handler

"""This module contains the ``MaxwellCircuit`` class."""
class MaxwellCircuit(AnalysisMaxwellCircuit):
    """Provide the Maxwell Circuit application interface.

    Parameters
    ----------
    project : str, optional
        Name of the project to select or the full path to the project
        or AEDTZ archive to open. The default is ``None``, in which
        case an attempt is made to get an active project. If no
        projects are present, an empty project is created.
    design : str, optional
        Name of the design to select. The default is ``None``, in
        which case an attempt is made to get an active design. If no
        designs are present, an empty design is created.
    version : str, int, float, optional
        Version of AEDT to use. The default is ``None``. If ``None``,
        the active setup is used or the latest installed version is
        used.
        Examples of input values are ``251``, ``25.1``, ``2025.1``, ``"2025.1"``.
    non_graphical : bool, optional
        Whether to launch AEDT in non-graphical mode. The default
        is ``False``, in which case AEDT is launched in graphical mode.
        This parameter is ignored when a script is launched within AEDT.
    new_desktop : bool, optional
        Whether to launch an instance of AEDT in a new thread, even if
        another instance of the ``specified_version`` is active on the
        machine. The default is ``False``.
    close_on_exit : bool, optional
        Whether to release AEDT on exit. The default is ``False``.
    student_version : bool, optional
        Whether to open the AEDT student version. The default is ``False``.
    machine : str, optional
        Machine name to connect the oDesktop session to. This works only in 2022 R2
        and later. The remote server must be up and running with the command
        `"ansysedt.exe -grpcsrv portnum"`. If the machine is `"localhost"`, the server
        is also started if not present.
    port : int, optional
        Port number on which to start the oDesktop communication on an already existing server.
        This parameter is ignored when creating a new server. It works only in 2022 R2 or
        later. The remote server must be up and running with the command `"ansysedt.exe -grpcsrv portnum"`.
    aedt_process_id : int, optional
        Process ID for the instance of AEDT to point PyAEDT at. The default is
        ``None``. This parameter is only used when ``new_desktop = False``.
    remove_lock : bool, optional
        Whether to remove lock to project before opening it or not.
        The default is ``False``, which means to not unlock
        the existing project if needed and raise an exception.

    Examples
    --------
    Create an instance of Maxwell Circuit and connect to an existing
    Maxwell circuit design or create a new Maxwell circuit design if one does
    not exist.

    >>> from ansys.aedt.core import MaxwellCircuit
    >>> app = MaxwellCircuit()

    Create an instance of Maxwell Circuit and link to a project named
    ``"projectname"``. If this project does not exist, create one with
    this name.

    >>> app = MaxwellCircuit(projectname)

    Create an instance of Maxwell Circuit and link to a design named
    ``"designname"`` in a project named ``"projectname"``.

    >>> app = MaxwellCircuit(projectname, designame)

    Create an instance of Maxwell Circuit and open the specified
    project, which is named ``"myfile.aedt"``.

    >>> app = MaxwellCircuit("myfile.aedt")
    """
    @pyaedt_function_handler(designname="design", projectname="project", specified_version="version", setup_name="setup", new_desktop_session="new_desktop")
    def __init__(self, project=..., design=..., solution_type=..., version=..., non_graphical=..., new_desktop=..., close_on_exit=..., student_version=..., machine=..., port=..., aedt_process_id=..., remove_lock=...) -> None:
        """Constructor."""
        ...
    
    @pyaedt_function_handler(file_to_import="input_file")
    def create_schematic_from_netlist(self, input_file): # -> Literal[True]:
        """Create a circuit schematic from an HSpice net list.

        Supported currently are:

        * R
        * L
        * C
        * Diodes

        Parameters
        ----------
        input_file : str
            Full path to the HSpice file.

        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.

        """
        ...
    
    @pyaedt_function_handler(file_to_export="output_file")
    def export_netlist_from_schematic(self, output_file): # -> Literal[False]:
        """Create netlist from schematic circuit.

        Parameters
        ----------
        output_file : str or :class:`pathlib.Path`
            File path to export the netlist to.

        Returns
        -------
        str
            Netlist file path when successful, ``False`` when failed.

        """
        ...
    


