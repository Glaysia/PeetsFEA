"""
This type stub file was generated by pyright.
"""

from ansys.aedt.core.generic.general_methods import pyaedt_function_handler
from ansys.aedt.core.internal.checks import min_aedt_version

class ComponentArray:
    """Manages object attributes for a 3D component array.

    Parameters
    ----------
    app : :class:`ansys.aedt.core.Hfss`
        HFSS PyAEDT object.
    name : str, optional
        Array name. The default is ``None``, in which case a random name is assigned.

    Examples
    --------
    Basic usage demonstrated with an HFSS design with an existing array:

    >>> from ansys.aedt.core import Hfss
    >>> aedtapp = Hfss(project="Array.aedt")
    >>> array_names = aedtapp.component_array_names[0]
    >>> array = aedtapp.component_array[array_names[0]]
    """
    def __init__(self, app, name=...) -> None:
        ...
    
    @pyaedt_function_handler()
    def __getitem__(self, key): # -> Literal[False]:
        """Get cell object corresponding to a key (row, column).

        Parameters
        ----------
        key : tuple(int,int)
            Row and column associated to the cell.

        Returns
        -------
        :class:`ansys.aedt.core.modeler.cad.component_array.CellArray`
        """
        ...
    
    @property
    def component_names(self):
        """List of component names."""
        ...
    
    @property
    def cells(self): # -> list[list[None]] | None:
        """List of :class:`ansys.aedt.core.modeler.cad.component_array.CellArray` objects."""
        ...
    
    @property
    def name(self): # -> str:
        """Name of the array."""
        ...
    
    @name.setter
    def name(self, array_name): # -> None:
        ...
    
    @property
    def properties(self): # -> dict[Any, Any] | Literal[False]:
        """Ordered dictionary of the properties of the component array."""
        ...
    
    @property
    def post_processing_cells(self): # -> dict[Any, Any]:
        """Dictionary of each component's postprocessing cells."""
        ...
    
    @post_processing_cells.setter
    def post_processing_cells(self, val): # -> None:
        ...
    
    @property
    def visible(self):
        """Flag indicating if the array is visible."""
        ...
    
    @visible.setter
    def visible(self, val): # -> None:
        ...
    
    @property
    def show_cell_number(self):
        """Flag indicating if the array cell number is shown."""
        ...
    
    @show_cell_number.setter
    def show_cell_number(self, val): # -> None:
        ...
    
    @property
    def render_choices(self): # -> list[Any]:
        """List of rendered name choices."""
        ...
    
    @property
    def render(self):
        """Array rendering."""
        ...
    
    @render.setter
    def render(self, val): # -> None:
        ...
    
    @property
    def render_id(self): # -> int:
        """Array rendering ID."""
        ...
    
    @property
    def a_vector_choices(self): # -> list[Any]:
        """List of name choices for vector A."""
        ...
    
    @property
    def b_vector_choices(self): # -> list[Any]:
        """List of name choices for vector B."""
        ...
    
    @property
    def a_vector_name(self):
        """Name of vector A."""
        ...
    
    @a_vector_name.setter
    def a_vector_name(self, val): # -> None:
        ...
    
    @property
    def b_vector_name(self):
        """Name of vector B."""
        ...
    
    @b_vector_name.setter
    def b_vector_name(self, val): # -> None:
        ...
    
    @property
    def a_size(self): # -> int:
        """Number of cells in the vector A direction."""
        ...
    
    @a_size.setter
    def a_size(self, val): # -> None:
        ...
    
    @property
    def b_size(self): # -> int:
        """Number of cells in the vector B direction."""
        ...
    
    @b_size.setter
    def b_size(self, val): # -> None:
        ...
    
    @property
    def a_length(self):
        """Length of the array in A direction."""
        ...
    
    @property
    def b_length(self):
        """Length of the array in B direction."""
        ...
    
    @property
    def padding_cells(self): # -> int:
        """Number of padding cells."""
        ...
    
    @padding_cells.setter
    def padding_cells(self, val): # -> None:
        ...
    
    @property
    def coordinate_system(self): # -> str:
        """Coordinate system name."""
        ...
    
    @coordinate_system.setter
    def coordinate_system(self, name): # -> None:
        ...
    
    @pyaedt_function_handler()
    def update_properties(self): # -> dict[Any, Any] | Literal[False]:
        """Update component array properties.

        Returns
        -------
        dict
           Ordered dictionary of the properties of the component array.
        """
        ...
    
    @pyaedt_function_handler()
    def delete(self): # -> None:
        """Delete the component array.

        References
        ----------
        >>> oModule.DeleteArray

        """
        ...
    
    @pyaedt_function_handler(array_path="output_file")
    @min_aedt_version("2024.1")
    def export_array_info(self, output_file=...): # -> str:
        """Export array information to a CSV file.

        Returns
        -------
        str
           Path of the CSV file.

        References
        ----------
        >>> oModule.ExportArray
        """
        ...
    
    @pyaedt_function_handler(csv_file="input_file")
    def parse_array_info_from_csv(self, input_file): # -> dict[Any, Any] | Literal[False]:
        """Parse component array information from the CSV file.

        Parameters
        ----------
        input_file : str
             Name of the CSV file.

        Returns
        -------
        dict
           Ordered dictionary of the properties of the component array.

        Examples
        --------
        >>> from ansys.aedt.core import Hfss
        >>> aedtapp = Hfss(project="Array.aedt")
        >>> array_names = aedtapp.component_array_names[0]
        >>> array = aedtapp.component_array[array_names[0]]
        >>> array_csv = array.export_array_info()
        >>> array_info = array.array_info_parser(array_csv)
        """
        ...
    
    @pyaedt_function_handler()
    def edit_array(self): # -> Literal[True]:
        """Edit component array.

        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.

        References
        ----------
        >>> oModule.EditArray
        """
        ...
    
    @pyaedt_function_handler()
    def get_cell(self, row, col): # -> Literal[False]:
        """Get cell object corresponding to a row and column.

        Returns
        -------
        :class:`ansys.aedt.core.modeler.cad.component_array.CellArray`

        """
        ...
    
    @pyaedt_function_handler()
    def lattice_vector(self): # -> list[Any]:
        """Get model lattice vector.

        Returns
        -------
        list
            List of starting point coordinates paired with ending point coordinates.

        References
        ----------
        >>> oModule.GetLatticeVectors()

        """
        ...
    
    @pyaedt_function_handler()
    def get_component_objects(self): # -> dict[Any, Any]:
        """Get 3D component center.

        Returns
        -------
        dict
            Dictionary of the center position and part name for all 3D components.

        """
        ...
    
    @pyaedt_function_handler()
    def get_cell_position(self): # -> list[list[None]]:
        """Get cell position.

        Returns
        -------
        list
            List of the center position and part name for all cells.

        """
        ...
    


class CellArray:
    """Manages object attributes for a 3D component and a user-defined model.

    Parameters
    ----------
    row : int
        Row index of the cell.
    col : int
        Column index of the cell.
    array_props : dict
        Dictionary containing the properties of the array.
    component_names : list
        List of component names in the array.
    array_obj : class:`ansys.aedt.core.modeler.cad.component_array.ComponentArray`
        Instance of the array containing the cell.

    """
    def __init__(self, row, col, array_props, component_names, array_obj) -> None:
        ...
    
    @property
    def rotation(self):
        """Rotation value of the cell object."""
        ...
    
    @rotation.setter
    def rotation(self, val): # -> None:
        ...
    
    @property
    def component(self): # -> None:
        """Component name of the cell object."""
        ...
    
    @component.setter
    def component(self, val): # -> None:
        ...
    
    @property
    def is_active(self): # -> bool:
        """Flag indicating if the cell object is active or passive."""
        ...
    
    @is_active.setter
    def is_active(self, val): # -> None:
        ...
    


