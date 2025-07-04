"""
This type stub file was generated by pyright.
"""

from ansys.aedt.core.generic.general_methods import pyaedt_function_handler
from ansys.aedt.core.internal.checks import graphics_required

logger = ...
ZONE_LETTERS = ...
class BuildingsPrep:
    """Contains all basic functions needed to generate buildings stl files."""
    def __init__(self, cad_path) -> None:
        ...
    
    @staticmethod
    @pyaedt_function_handler()
    @graphics_required
    def create_building_roof(all_pos):
        """Generate a filled in polygon from outline.

        Includes concave and convex shapes.

        Parameters
        ----------
        all_pos : list

        Returns
        -------
        :class:`pyvista.PolygonData`
        """
        ...
    
    @pyaedt_function_handler()
    @graphics_required
    def generate_buildings(self, center_lat_lon, terrain_mesh, max_radius=...): # -> dict[str, None] | dict[str, Any]:
        """Generate the buildings stl file.

        Parameters
        ----------
        center_lat_lon : list
            Latitude and longitude.
        terrain_mesh : :class:`pyvista.PolygonData`
            Terrain mesh.
        max_radius : float, int
            Radius around latitude and longitude.

        Returns
        -------
        dict
            Info of generated stl file.
        """
        ...
    


class RoadPrep:
    """Contains all basic functions needed to generate road stl files."""
    def __init__(self, cad_path) -> None:
        ...
    
    @pyaedt_function_handler()
    @graphics_required
    def create_roads(self, center_lat_lon, terrain_mesh, max_radius=..., z_offset=..., road_step=..., road_width=...): # -> dict[str, Any]:
        """Generate the road stl file.

        Parameters
        ----------
        center_lat_lon : list
            Latitude and longitude.
        terrain_mesh : :class:`pyvista.PolygonData`
            Terrain mesh.
        max_radius : float, int
            Radius around latitude and longitude.
        z_offset : float, optional
            Elevation offset of the road.
        road_step : float, optional
            Road computation steps in meters.
        road_width : float, optional
            Road width in meter.

        Returns
        -------
        dict
            Info of generated stl file.
        """
        ...
    


class TerrainPrep:
    """Contains all basic functions needed for creating a terrain stl mesh."""
    def __init__(self, cad_path=...) -> None:
        ...
    
    @pyaedt_function_handler()
    @graphics_required
    def get_terrain(self, center_lat_lon, max_radius=..., grid_size=..., buffer_percent=...): # -> dict[str, Any]:
        """Generate the terrain stl file.

        Parameters
        ----------
        center_lat_lon : list
            Latitude and longitude.
        max_radius : float, int
            Radius around latitude and longitude.
        grid_size : float, optional
            Grid size in meters.
        buffer_percent : float, optional
            Buffer extra size over the radius.


        Returns
        -------
        dict
            Info of generated stl file.
        """
        ...
    
    @staticmethod
    @pyaedt_function_handler()
    def get_elevation(center_lat_lon, max_radius=..., grid_size=...): # -> tuple[_Array[tuple[int, int], float64], _Array[tuple[int, int, int], float64], _Array[tuple[int, int, int], float64]]:
        """Get Elevation map.

        Parameters
        ----------
        center_lat_lon : list
            Latitude and longitude.
        max_radius : float, int
            Radius around latitude and longitude.
        grid_size : float, optional
            Grid size in meters.

        Returns
        -------
        tuple
        """
        ...
    


@pyaedt_function_handler()
def convert_latlon_to_utm(latitude: float, longitude: float, zone_letter: str = ..., zone_number: int = ...) -> tuple:
    """Convert latitude and longitude to UTM (Universal Transverse Mercator) coordinates.

    Parameters
    ----------
    latitude : float
        Latitude value in decimal degrees.
    longitude : float
        Longitude value in decimal degrees.
    zone_letter: str, optional
        UTM zone designators. The default is ``None``. The valid zone letters are ``"CDEFGHJKLMNPQRSTUVWXX"``
    zone_number: int, optional
        Global map numbers of a UTM zone numbers map. The default is ``None``.

    Notes
    -----
    .. [1] https://mapref.org/UTM-ProjectionSystem.html

    Returns
    -------
    tuple
        Tuple containing UTM East coordinate, North coordinate, zone letter, and zone number.
    """
    ...

@pyaedt_function_handler()
def convert_utm_to_latlon(east: int, north: float, zone_number: int, zone_letter: str = ..., northern: bool = ...) -> tuple:
    """Convert UTM (Universal Transverse Mercator) coordinates to latitude and longitude.

    Parameters
    ----------
    east : int
        East value of UTM coordinates.
    north : int
        North value of UTM coordinates.
    zone_number: int, optional
        Global map numbers of a UTM zone numbers map.
    zone_letter: str, optional
        UTM zone designators. The default is ``None``. The valid zone letters are ``"CDEFGHJKLMNPQRSTUVWXX"``
    northern: bool, optional
        Indicates whether the UTM coordinates are in the Northern Hemisphere. If ``True``, the coordinates
        are treated as being north of the equator, if ``False``, they are treated as being south of the equator.
        The default is ``None`` in which case the method determines the hemisphere using ``zone_letter``.

    Notes
    -----
    .. [1] https://mapref.org/UTM-ProjectionSystem.html

    Returns
    -------
    tuple
        Tuple containing latitude and longitude.
    """
    ...

