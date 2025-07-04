"""
This type stub file was generated by pyright.
"""

from ansys.aedt.core.generic.general_methods import pyaedt_function_handler
from ansys.aedt.core.modeler.cad.elements_3d import BinaryTreeNode
from ansys.aedt.core.modules.boundary.common import BoundaryCommon

class MaxwellParameters(BoundaryCommon, BinaryTreeNode):
    """Manages parameters data and execution.

    Parameters
    ----------
    app : :class:`ansys.aedt.core.maxwell.Maxwell3d`, :class:`ansys.aedt.core.maxwell.Maxwell2d`
        Either ``Maxwell3d`` or ``Maxwell2d`` application.
    name : str
        Name of the boundary.
    props : dict, optional
        Properties of the boundary.
    boundarytype : str, optional
        Type of the boundary.

    Examples
    --------

    Create a matrix in Maxwell3D return a ``ansys.aedt.core.modules.boundary.common.BoundaryObject``

    >>> from ansys.aedt.core import Maxwell2d
    >>> maxwell_2d = Maxwell2d()
    >>> coil1 = maxwell_2d.modeler.create_rectangle([8.5, 1.5, 0], [8, 3], True, "Coil_1", "vacuum")
    >>> coil2 = maxwell_2d.modeler.create_rectangle([8.5, 1.5, 0], [8, 3], True, "Coil_2", "vacuum")
    >>> maxwell_2d.assign_matrix(["Coil_1", "Coil_2"])
    """
    def __init__(self, app, name, props=..., boundarytype=...) -> None:
        ...
    
    @property
    def reduced_matrices(self): # -> list[Any] | dict[Any, Any] | None:
        """List of reduced matrix groups for the parent matrix.

        Returns
        -------
        dict
            Dictionary of reduced matrices where the key is the name of the parent matrix
            and the values are in a list of reduced matrix groups.
        """
        ...
    
    @property
    def props(self): # -> BoundaryProps | dict[Any, Any]:
        """Maxwell parameter data.

        Returns
        -------
        :class:BoundaryProps
        """
        ...
    
    @property
    def name(self): # -> str | Any:
        """Boundary Name."""
        ...
    
    @name.setter
    def name(self, value): # -> None:
        ...
    
    @pyaedt_function_handler()
    def create(self): # -> bool:
        """Create a boundary.

        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.

        """
        ...
    
    @pyaedt_function_handler()
    def update(self): # -> bool:
        """Update the boundary.

        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.

        """
        ...
    
    @pyaedt_function_handler()
    def join_series(self, sources, matrix_name=..., join_name=...): # -> tuple[Literal[False], Literal[False]] | tuple[str | Any, str | Any]:
        """Create matrix reduction by joining sources in series.

        Parameters
        ----------
        sources : list
            Sources to be included in matrix reduction.
        matrix_name :  str, optional
            Name of the string to create.
        join_name : str, optional
            Name of the Join operation.

        Returns
        -------
        (str, str)
            Matrix name and Joint name.

        """
        ...
    
    @pyaedt_function_handler()
    def join_parallel(self, sources, matrix_name=..., join_name=...): # -> tuple[Literal[False], Literal[False]] | tuple[str | Any, str | Any]:
        """Create matrix reduction by joining sources in parallel.

        Parameters
        ----------
        sources : list
            Sources to be included in matrix reduction.
        matrix_name :  str, optional
            Name of the matrix to create.
        join_name : str, optional
            Name of the Join operation.

        Returns
        -------
        (str, str)
            Matrix name and Joint name.

        """
        ...
    


class MaxwellMatrix:
    def __init__(self, app, parent_name, reduced_name) -> None:
        ...
    
    @property
    def sources(self): # -> dict[Any, Any] | None:
        """List of matrix sources."""
        ...
    
    @pyaedt_function_handler()
    def update(self, old_source, source_type, new_source=..., new_excitations=...): # -> bool:
        """Update the reduced matrix.

        Parameters
        ----------
        old_source : str
            Original name of the source to update.
        source_type : str
            Source type, which can be ``Series`` or ``Parallel``.
        new_source : str, optional
            New name of the source to update.
            The default value is the old source name.
        new_excitations : str, optional
            List of excitations to include in the matrix reduction.
            The default values are excitations included in the source to update.

        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.
        """
        ...
    
    @pyaedt_function_handler()
    def delete(self, source): # -> bool:
        """Delete a specified source in a reduced matrix.

        Parameters
        ----------
        source : string
            Name of the source to delete.

        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.
        """
        ...
    


