"""
This type stub file was generated by pyright.
"""

from ansys.aedt.core.application.analysis_twin_builder import AnalysisTwinBuilder
from ansys.aedt.core.generic.general_methods import pyaedt_function_handler

"""This module contains the ``TwinBuilder`` class."""
class TwinBuilder(AnalysisTwinBuilder):
    """Provides the Twin Builder application interface.

    Parameters
    ----------
    project : str, optional
        Name of the project to select or the full path to the project
        or AEDTZ archive to open.  The default is ``None``, in which
        case an attempt is made to get an active project. If no
        projects are present, an empty project is created.
    design : str, optional
        Name of the design to select. The default is ``None``, in
        which case an attempt is made to get an active design. If no
        designs are present, an empty design is created.
    solution_type : str, optional
        Solution type to apply to the design. The default is
        ``None``, in which case the default type is applied.
    setup : str, optional
        Name of the setup to use as the nominal. The default is
        ``None``, in which case the active setup is used or
        nothing is used.
    version : str, int, float, optional
        Version of AEDT to use. The default is ``None``, in which
        case the active setup or latest installed version is
        used.
        Examples of input values are ``251``, ``25.1``, ``2025.1``, ``"2025.1"``.
    non_graphical : bool, optional
        Whether to launch AEDT in non-graphical mode. The default
        is ``False``, in which case AEDT is launched in graphical mode.
        This parameter is ignored when a script is launched within AEDT.
    new_desktop : bool, optional
        Whether to launch an instance of AEDT in a new thread, even if
        another instance of the ``specified_version`` is active on the
        machine.  The default is ``False``.
    close_on_exit : bool, optional
        Whether to release AEDT on exit. The default is ``False``.
    student_version : bool, optional
        Whether to open the AEDT student version. The default is ``False``.
    machine : str, optional
        Machine name to connect the oDesktop session to. This works only in 2022 R2
        or later. The remote server must be up and running with the command
        `"ansysedt.exe -grpcsrv portnum"`. If the machine is `"localhost"`,
        the server also starts if not present.
    port : int, optional
        Port number on which to start the oDesktop communication on an already existing server.
        This parameter is ignored when creating a new server. It works only in 2022 R2 or later.
        The remote server must be up and running with command `"ansysedt.exe -grpcsrv portnum"`.
    aedt_process_id : int, optional
        Process ID for the instance of AEDT to point PyAEDT at. The default is
        ``None``. This parameter is only used when ``new_desktop = False``.
    remove_lock : bool, optional
        Whether to remove lock to project before opening it or not.
        The default is ``False``, which means to not unlock
        the existing project if needed and raise an exception.

    Examples
    --------
    Create an instance of Twin Builder and connect to an existing
    Maxwell design or create a new Maxwell design if one does not
    exist.

    >>> from ansys.aedt.core import TwinBuilder
    >>> app = TwinBuilder()

    Create a instance of Twin Builder and link to a project named
    ``"projectname"``. If this project does not exist, create one with
    this name.

    >>> app = TwinBuilder(projectname)

    Create an instance of Twin Builder and link to a design named
    ``"designname"`` in a project named ``"projectname"``.

    >>> app = TwinBuilder(projectname, designame)

    Create an instance of Twin Builder and open the specified
    project, which is named ``"myfile.aedt"``.

    >>> app = TwinBuilder("myfile.aedt")
    """
    @pyaedt_function_handler(designname="design", projectname="project", specified_version="version", setup_name="setup", new_desktop_session="new_desktop")
    def __init__(self, project=..., design=..., solution_type=..., setup=..., version=..., non_graphical=..., new_desktop=..., close_on_exit=..., student_version=..., machine=..., port=..., aedt_process_id=..., remove_lock=...) -> None:
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
        * Bjts
        * Discrete components with syntax ``Uxxx net1 net2 ... netn modname``

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
    
    @pyaedt_function_handler()
    def set_end_time(self, expression): # -> Literal[True]:
        """Set the end time.

        Parameters
        ----------
        expression :


        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.

        References
        ----------
        >>> oDesign.ChangeProperty
        """
        ...
    
    @pyaedt_function_handler()
    def set_hmin(self, expression): # -> Literal[True]:
        """Set hmin.

        Parameters
        ----------
        expression :


        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.

        References
        ----------
        >>> oDesign.ChangeProperty
        """
        ...
    
    @pyaedt_function_handler()
    def set_hmax(self, expression): # -> Literal[True]:
        """Set hmax.

        Parameters
        ----------
        expression :


        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.

        References
        ----------
        >>> oDesign.ChangeProperty
        """
        ...
    
    @pyaedt_function_handler(var_str="variable")
    def set_sim_setup_parameter(self, variable, expression, analysis_name=...): # -> Literal[True]:
        """Set simulation setup parameters.

        Parameters
        ----------
        variable : string
            Name of the variable.
        expression :

        analysis_name : str, optional
            Name of the analysis. The default is ``"TR"``.

        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.

        References
        ----------
        >>> oDesign.ChangeProperty
        """
        ...
    
    @pyaedt_function_handler()
    def create_subsheet(self, name, design_name): # -> bool:
        """Create a subsheet from a parent design.

        If the parent design does not exist, it will add at top level.
        Nested subsheets are currently not supported.

        Parameters
        ----------
        name : str
            Name of the subsheet.
        design_name : str
            Name of the parent design.

        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.

        Examples
        --------
        >>> from ansys.aedt.core import TwinBuilder
        >>> tb = TwinBuilder(version="2025.1")
        >>> tb.create_subsheet("subsheet", "parentdesign")
        """
        ...
    
    @pyaedt_function_handler(setup_name="setup", sweep_name="sweep")
    def add_q3d_dynamic_component(self, source_project, source_design_name, setup, sweep, coupling_matrix_name, model_depth=..., maximum_order=..., error_tolerance=..., z_ref=..., state_space_dynamic_link_type=..., component_name=..., save_project=...):
        """Add a Q2D or Q3D dynamic component to a Twin Builder design.

        Parameters
        ----------
        source_project : str
            Source project name or project path.
        source_design_name : str
            Source design name.
        setup : str
            Setup name.
        sweep : str
            Sweep name.
        coupling_matrix_name : str
            Coupling matrix name.
        model_depth : str, optional
            2D model depth specified as value with unit.
            To be provided if design is Q2D.
            The default value is ``1meter``
        maximum_order : float, optional
            The Maximum Order value specifies the highest order state space
            system that you can choose while fitting to represent the system behavior.
            A lower order may lead to less accuracy but faster simulation.
            The default value is ``10000``.
        error_tolerance : float, optional
            Error Tolerance sets the error tolerance for S-Matrix fitting.
            The default value is ``0.005``.
        z_ref : float, optional
            Sets the value of the Z (ref) in ohms.
            The default value is ``50``.
        state_space_dynamic_link_type : str, optional
            Q3D state space dynamic link type.
            Possible options are:
                - ``S`` for S parameters link type.
                - ``RLGC`` for RLGC Parameters link type.
                - ``EQ`` for Equivalent Circuit.
            The default value is ``RLGC``.
        component_name : str, optional
            Component name.
            If not provided a generic name with prefix "SimpQ3DData" will be given.
        save_project : bool, optional
            Save project after use.
            The default value is ``True``.

        Returns
        -------
        :class:`ansys.aedt.core.modeler.circuits.object_3d_circuit.CircuitComponent` or bool
            Circuit component object if successful or ``False`` if fails.

        References
        ----------
        >>> oComponentManager.AddDynamicNPortData

        Examples
        --------

        Create an instance of Twin Builder.

        >>> from ansys.aedt.core import TwinBuilder
        >>> tb = TwinBuilder()

        Add a Q2D dynamic link component.

        >>> tb.add_q3d_dynamic_component(
        ...     "Q2D_ArmouredCableExample", "2D_Extractor_Cable", "MySetupAuto", "sweep1", "Original", "100mm"
        ... )
        >>> tb.release_desktop()
        """
        ...
    
    @pyaedt_function_handler()
    def add_excitation_model(self, project, design, use_default_values=..., setup=..., start=..., stop=..., export_uniform_points=..., export_uniform_points_step=..., excitations=...): # -> Literal[False]:
        """Use the excitation component to assign output quantities

        This works in a Twin Builder design to a windings in a Maxwell design.
        This method works only with AEDT 2025 R1 and later.

        Parameters
        ----------
        project : str
            Name or path to the project to provide.
        design : str
            Name of the design to import the excitations from.
        use_default_values : bool, optional
            Whether to use the default values for the start and stop times for the chosen TR setup.
            The default value is ``True``.
        setup : str, optional
            Name of the Twinbuilder setup.
            If not provided, the default value is the first setup in the design.
        start : str, optional
            Start time provided as value + units.
            The default value is ``None``.
            If not provided and ``use_default_values=True``, the value is chosen from the TR setup.
        stop : float, optional
            Stop time provided as value + units.
            The default value is ``None``.
            If not provided and ``use_default_values=True``, the value is chosen from the TR setup.
        export_uniform_points : bool, optional
            Whether Twin Builder is to perform linear interpolation to uniformly space out time and data points.
            The interpolation is based on the step size provided. The default is ``False``.
        export_uniform_points_step : float, optional
            Step size to use for the uniform interpolation.
            The default value is ``1E-5``.
        excitations : dict, optional
            List of excitations to extract from the Maxwell design.
            It is a dictionary where the keys are the excitation names and value a list
            containing respectively:
            - The excitation value to assign to the winding, provided as a string.
            - A boolean whether to enable the component or not.
            - The excitation type. Possible options are ``Current`` or ``Voltage``.
            - A boolean to enable the pin. If ``True`` the pin will be used to make connection on the schematic
            and the excitation value will be zeroed, since the expectation is that the value is provided
            through schematic connections.
            To know which excitations will be extracted from the Maxwell design use
            ``app.excitations_by_type["Winding Group"]`` where ``app`` is the Maxwell instance.
            If not provided, the method automatically retrieves the excitations from the Maxwell Design
            and sets the default excitation settings.

        Returns
        -------
        :class:`pyaedt.modeler.cad.object3dcircuit.CircuitComponent` or bool
            Circuit component object if successful or ``False`` if fails.

        References
        ----------
        >>> oComponentManager.AddExcitationModel

        Example
        -------
        >>> from ansys.aedt.core import TwinBuilder
        >>> tb = TwinBuilder(specified_version="2025.1")
        >>> maxwell_app = tb.desktop_class[[project_name, "my_maxwell_design"]]
        >>> excitations = {}
        >>> for e in maxwell_app.excitations_by_type["Winding Group"]:
        ...     excitations[e.name] = ["20", True, e.props["Type"], False]
        >>> comp = tb.add_excitation_model(project=project_name, design="my_maxwell_design", excitations=excitations)
        >>> tb.release_desktop(False, False)
        """
        ...
    


