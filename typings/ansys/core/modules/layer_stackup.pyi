"""
This type stub file was generated by pyright.
"""

from ansys.aedt.core.generic.general_methods import pyaedt_function_handler

"""
This module contains these classes: `Layer` and `Layers`.

This module provides all layer stackup functionalities for the Circuit and HFSS 3D Layout tools.
"""
class Layer:
    """Manages the stackup layer for the Circuit and HFSS 3D Layout tools.

    Parameters
    ----------
    app : :class:`ansys.aedt.core.modules.layer_stackup.Layers`
    layertype : string, optional
        The default is ``"signal"``.
    negative : bool, optional
        Whether the geometry on the layer is cut away
        from the layer. The default is ``False``.

    Examples
    --------
    >>> from ansys.aedt.core import Hfss3dLayout
    >>> app = Hfss3dLayout()
    >>> layers = app.modeler.layers["Top"]
    """
    def __init__(self, app, layertype=..., negative=...) -> None:
        ...
    
    @property
    def color(self): # -> list[int] | list[Any]:
        """Return or set the property of the active layer. Color it is list of rgb values (0,255).

        Returns
        -------
        list

        """
        ...
    
    @color.setter
    def color(self, val): # -> None:
        ...
    
    @property
    def transparency(self): # -> int:
        """Get/Set the property to the active layer.

        Returns
        -------
        int
        """
        ...
    
    @transparency.setter
    def transparency(self, val): # -> None:
        ...
    
    @property
    def is_visible(self): # -> bool:
        """Get/Set the active layer visibility.

        Returns
        -------
        bool
        """
        ...
    
    @is_visible.setter
    def is_visible(self, val): # -> None:
        ...
    
    @property
    def is_visible_shape(self): # -> bool:
        """Get/Set the active layer shape visibility.

        Returns
        -------
        bool
        """
        ...
    
    @is_visible_shape.setter
    def is_visible_shape(self, val): # -> None:
        ...
    
    @property
    def is_visible_path(self): # -> bool:
        """Get/Set the active layer paths visibility.

        Returns
        -------
        bool
        """
        ...
    
    @is_visible_path.setter
    def is_visible_path(self, val): # -> None:
        ...
    
    @property
    def is_visible_pad(self): # -> bool:
        """Get/Set the active layer pad visibility.

        Returns
        -------
        bool
        """
        ...
    
    @is_visible_pad.setter
    def is_visible_pad(self, val): # -> None:
        ...
    
    @property
    def is_visible_hole(self): # -> bool:
        """Get/Set the active layer hole visibility.

        Returns
        -------
        bool
        """
        ...
    
    @is_visible_hole.setter
    def is_visible_hole(self, val): # -> None:
        ...
    
    @property
    def is_visible_component(self): # -> bool:
        """Get/Set the active layer component visibility.

        Returns
        -------
        bool
        """
        ...
    
    @is_visible_component.setter
    def is_visible_component(self, val): # -> None:
        ...
    
    @property
    def is_mesh_background(self): # -> bool:
        """Get/Set the active layer mesh backgraound.

        Returns
        -------
        bool
        """
        ...
    
    @is_mesh_background.setter
    def is_mesh_background(self, val): # -> None:
        ...
    
    @property
    def is_mesh_overlay(self): # -> bool:
        """Get/Set the active layer mesh overlay.

        Returns
        -------
        bool
        """
        ...
    
    @is_mesh_overlay.setter
    def is_mesh_overlay(self, val): # -> None:
        ...
    
    @property
    def locked(self): # -> bool:
        """Get/Set the active layer lock flag.

        Returns
        -------
        bool
        """
        ...
    
    @locked.setter
    def locked(self, val): # -> None:
        ...
    
    @property
    def top_bottom(self): # -> str:
        """Get/Set the active layer top bottom alignment.

        Returns
        -------
        str
        """
        ...
    
    @top_bottom.setter
    def top_bottom(self, val): # -> None:
        ...
    
    @property
    def pattern(self): # -> int:
        """Get/Set the active layer pattern.

        Returns
        -------
        float
        """
        ...
    
    @pattern.setter
    def pattern(self, val): # -> None:
        ...
    
    @property
    def draw_override(self): # -> int:
        """Get/Set the active layer draw override value.

        Returns
        -------
        float
        """
        ...
    
    @draw_override.setter
    def draw_override(self, val): # -> None:
        ...
    
    @property
    def thickness(self): # -> int | str:
        """Get/Set the active layer thickness value.

        Returns
        -------
        float
        """
        ...
    
    @thickness.setter
    def thickness(self, val): # -> None:
        ...
    
    @property
    def upper_elevation(self):
        """Get the active layer upper elevation value with units.

        Returns
        -------
        float
        """
        ...
    
    @property
    def thickness_units(self): # -> str:
        """Get the active layer thickness units value.

        Returns
        -------
        str
        """
        ...
    
    @property
    def lower_elevation(self): # -> int:
        """Get/Set the active layer lower elevation.

        Returns
        -------
        float
        """
        ...
    
    @lower_elevation.setter
    def lower_elevation(self, val): # -> None:
        ...
    
    @property
    def roughness(self): # -> int:
        """Return or set the active layer roughness (with units).

        Returns
        -------
        str
        """
        ...
    
    @roughness.setter
    def roughness(self, value): # -> None:
        ...
    
    @property
    def bottom_roughness(self): # -> int:
        """Return or set the active layer bottom roughness (with units).

        Returns
        -------
        str
        """
        ...
    
    @bottom_roughness.setter
    def bottom_roughness(self, value): # -> None:
        ...
    
    @property
    def top_roughness(self): # -> int:
        """Get/Set the active layer top roughness (with units).

        Returns
        -------
        str
        """
        ...
    
    @top_roughness.setter
    def top_roughness(self, val): # -> None:
        ...
    
    @property
    def side_roughness(self): # -> int:
        """Get/Set the active layer side roughness (with units).

        Returns
        -------
        str
        """
        ...
    
    @side_roughness.setter
    def side_roughness(self, val): # -> None:
        ...
    
    @property
    def material(self): # -> str:
        """Get/Set the active layer material name.

        Returns
        -------
        str
        """
        ...
    
    @material.setter
    def material(self, val): # -> None:
        ...
    
    @property
    def fill_material(self): # -> str:
        """Get/Set the active layer filling material.

        Returns
        -------
        str
        """
        ...
    
    @fill_material.setter
    def fill_material(self, val): # -> None:
        ...
    
    @property
    def index(self): # -> int:
        """Get/Set the active layer index.

        Returns
        -------
        int
        """
        ...
    
    @index.setter
    def index(self, val): # -> None:
        ...
    
    @property
    def is_negative(self): # -> bool:
        """Get/Set the active layer negative flag. When `True` the layer will be negative.

        Returns
        -------
        bool
        """
        ...
    
    @is_negative.setter
    def is_negative(self, val): # -> None:
        ...
    
    @property
    def use_etch(self): # -> bool:
        """Get/Set the active layer etiching flag. When `True` the layer will use etch.

        Returns
        -------
        bool
        """
        ...
    
    @use_etch.setter
    def use_etch(self, val): # -> None:
        ...
    
    @property
    def etch(self): # -> int:
        """Get/Set the active layer etch value.

        Returns
        -------
        float
        """
        ...
    
    @etch.setter
    def etch(self, val): # -> None:
        ...
    
    @property
    def user(self): # -> bool:
        """Get/Set the active layer user flag.

        Returns
        -------
        bool
        """
        ...
    
    @user.setter
    def user(self, val): # -> None:
        ...
    
    @property
    def top_roughness_model(self): # -> str:
        """Get/Set the active layer top roughness model.

        Returns
        -------
        str
        """
        ...
    
    @top_roughness_model.setter
    def top_roughness_model(self, val): # -> None:
        ...
    
    @property
    def top_nodule_radius(self): # -> float:
        """Get/Set the active layer top roughness radius.

        Returns
        -------
        float
        """
        ...
    
    @top_nodule_radius.setter
    def top_nodule_radius(self, val): # -> None:
        ...
    
    @property
    def top_huray_ratio(self): # -> float:
        """Get/Set the active layer top roughness ratio.

        Returns
        -------
        float
        """
        ...
    
    @top_huray_ratio.setter
    def top_huray_ratio(self, val): # -> None:
        ...
    
    @property
    def bottom_roughness_model(self): # -> str:
        """Get/Set the active layer bottom roughness model.

        Returns
        -------
        str
        """
        ...
    
    @bottom_roughness_model.setter
    def bottom_roughness_model(self, val): # -> None:
        ...
    
    @property
    def bottom_nodule_radius(self): # -> float:
        """Get/Set the active layer bottom roughness radius.

        Returns
        -------
        float
        """
        ...
    
    @bottom_nodule_radius.setter
    def bottom_nodule_radius(self, val): # -> None:
        ...
    
    @property
    def bottom_huray_ratio(self): # -> float:
        """Get/Set the active layer bottom roughness ratio.

        Returns
        -------
        float
        """
        ...
    
    @bottom_huray_ratio.setter
    def bottom_huray_ratio(self, val): # -> None:
        ...
    
    @property
    def side_model(self): # -> str:
        """Get/Set the active layer side roughness model.

        Returns
        -------
        str
        """
        ...
    
    @side_model.setter
    def side_model(self, val): # -> None:
        ...
    
    @property
    def side_nodule_radius(self): # -> float:
        """Get/Set the active layer side roughness radius.

        Returns
        -------
        float
        """
        ...
    
    @side_nodule_radius.setter
    def side_nodule_radius(self, val): # -> None:
        ...
    
    @property
    def side_huray_ratio(self): # -> float:
        """Get/Set the active layer bottom roughness ratio.

        Returns
        -------
        float
        """
        ...
    
    @side_huray_ratio.setter
    def side_huray_ratio(self, val): # -> None:
        ...
    
    @property
    def usp(self): # -> bool:
        """Get/Set the active layer usp flag.

        Returns
        -------
        bool
        """
        ...
    
    @usp.setter
    def usp(self, val): # -> None:
        ...
    
    @property
    def hfss_solver_settings(self): # -> dict[str, Any]:
        """Get/Set the active layer hfss solver settings.

        Returns
        -------
        dict
        """
        ...
    
    @hfss_solver_settings.setter
    def hfss_solver_settings(self, val): # -> None:
        ...
    
    @property
    def planar_em_solver_settings(self): # -> dict[str, bool]:
        """Get/Set the active layer PlanarEm solver settings.

        Returns
        -------
        dict
        """
        ...
    
    @planar_em_solver_settings.setter
    def planar_em_solver_settings(self, val): # -> None:
        ...
    
    @property
    def zones(self): # -> list[Any]:
        """Get/Set the active layer zoness.

        Returns
        -------
        list
        """
        ...
    
    @zones.setter
    def zones(self, val): # -> None:
        ...
    
    @property
    def oeditor(self):
        """Oeditor Module."""
        ...
    
    @property
    def visflag(self): # -> int:
        """Visibility flag for objects on the layer."""
        ...
    
    @pyaedt_function_handler()
    def set_layer_color(self, r, g, b): # -> Literal[True]:
        """Update the color of the layer.

        Parameters
        ----------
        r : int
            Red color value.
        g : int
            Green color value.
        b :  int
            Blue color value.

        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.

        References
        ----------
        >>> oEditor.ChangeLayer
        """
        ...
    
    @pyaedt_function_handler()
    def create_stackup_layer(self): # -> Literal[True]:
        """Create a stackup layer.

        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.

        References
        ----------
        >>> oEditor.AddStackupLayer
        """
        ...
    
    @pyaedt_function_handler()
    def update_stackup_layer(self): # -> Literal[True]:
        """Update the stackup layer.

        .. note::
           This method is valid for release 2021 R1 and later.
           This method works only for signal and dielectric layers.

        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.

        References
        ----------
        >>> oEditor.ChangeLayer
        """
        ...
    
    @pyaedt_function_handler()
    def remove_stackup_layer(self): # -> bool:
        """Remove the stackup layer.

        Returns
        -------
        bool
            ``True`` when successful, ``False`` when failed.

        References
        ----------
        >>> oEditor.RemoveLayer
        """
        ...
    


class Layers:
    """Manages stackup for the Circuit and HFSS 3D Layout tools.

    Parameters
    ----------
    modeler : :class:`ansys.aedt.core.modeler.modeler_pcb.Modeler3DLayout`

    roughnessunits : str, optional
       Units for the roughness of layers. The default is ``"um"``.

    Examples
    --------
    >>> from ansys.aedt.core import Hfss3dLayout
    >>> app = Hfss3dLayout()
    >>> layers = app.modeler.layers
    """
    def __init__(self, modeler, roughnessunits=...) -> None:
        ...
    
    @property
    def oeditor(self):
        """Editor.

        References
        ----------
        >>> oEditor = oDesign.SetActiveEditor("Layout")
        """
        ...
    
    @property
    def zones(self): # -> list[Any]:
        """List of all available zones.

        Returns
        -------
        list
        """
        ...
    
    @property
    def LengthUnit(self):
        """Length units."""
        ...
    
    @property
    def all_layers(self): # -> list[Any]:
        """All stackup layers.

        Returns
        -------
        list
           List of stackup layers.

        References
        ----------
        >>> oEditor.GetStackupLayerNames()
        """
        ...
    
    @property
    def drawing_layers(self): # -> list[Any]:
        """All drawing layers.

        Returns
        -------
        list
           List of drawing layers.

        References
        ----------
        >>> oEditor.GetAllLayerNames()
        """
        ...
    
    @property
    def stackup_layers(self): # -> list[Any]:
        """All stackup layers.

        Returns
        -------
        List of :class:`ansys.aedt.core.modules.layer_stackup.Layer`
           List of stackup layers.

        References
        ----------
        >>> oEditor.GetAllLayerNames()
        """
        ...
    
    @property
    def all_signal_layers(self): # -> list[Any]:
        """All signal layers.

        Returns
        -------
        List of :class:`ansys.aedt.core.modules.layer_stackup.Layer`
            List of signal layers.
        """
        ...
    
    @property
    def signals(self): # -> dict[Any, Any]:
        """All signal layers.

        Returns
        -------
        dict[str, :class:`ansys.aedt.core.modules.layer_stackup.Layer`]
           Conductor layers.

        References
        ----------
        >>> oEditor.GetAllLayerNames()
        """
        ...
    
    @property
    def dielectrics(self): # -> dict[Any, Any]:
        """All dielectric layers.

        Returns
        -------
        dict[str, :class:`ansys.aedt.core.modules.layer_stackup.Layer`]
           Dielectric layers.

        References
        ----------
        >>> oEditor.GetAllLayerNames()
        """
        ...
    
    @property
    def drawings(self): # -> dict[Any, Any]:
        """All stackup layers.

        Returns
        -------
        dict[str, :class:`ansys.aedt.core.modules.layer_stackup.Layer`]
           Drawing layers.

        References
        ----------
        >>> oEditor.GetAllLayerNames()
        """
        ...
    
    @property
    def all_diel_layers(self): # -> list[Any]:
        """All dielectric layers.

        Returns
        -------
        List of :class:`ansys.aedt.core.modules.layer_stackup.Layer`
            List of dielectric layers.
        """
        ...
    
    @pyaedt_function_handler()
    def layer_id(self, name): # -> None:
        """Retrieve a layer ID.

        Parameters
        ----------
        name : str
            Name of the layer.

        Returns
        -------
        :class:`ansys.aedt.core.modules.layer_stackup.Layer`
            Layer objecy if the layer name exists.
        """
        ...
    
    @property
    def layers(self): # -> dict[Any, Any]:
        """Refresh all layers in the current stackup.

        Returns
        -------
         dict[int, :class:`ansys.aedt.core.modules.layer_stackup.Layer`]
            Number of layers in the current stackup.
        """
        ...
    
    @pyaedt_function_handler(layername="layer", layertype="layer_type")
    def add_layer(self, layer, layer_type=..., thickness=..., elevation=..., material=..., isnegative=...):
        """Add a layer.

        Parameters
        ----------
        layer : str
            Name of the layer.
        layer_type : str, optional
            Type of the layer. The default is ``"signal"``.
        thickness : str, optional
            Thickness with units. The default is ``"0mm"``.
        elevation : str, optional
            Elevation with units. The default is ``"0mm"``.
        material : str, optional
            Name of the material. The default is ``"copper"``.
        isnegative : bool, optional
            If ``True``, the geometry on the layer is cut away from the layer. The default is ``False``.

        Returns
        -------
        :class:`ansys.aedt.core.modules.layer_stackup.Layer`
            Layer object.
        """
        ...
    
    @pyaedt_function_handler()
    def change_stackup_type(self, mode=..., number_zones=...): # -> bool:
        """Change the stackup type between Multizone, Overlap and Laminate.

        Parameters
        ----------
        mode : str, optional
            Stackup type. Default is `"Multizone"`. Options are `"Overlap"` and `"Laminate"`.
        number_zones : int, optional
            Number of zones of multizone. By default all layers will be enabled in all zones.

        Returns
        -------
        bool
            `True` if successful.
        """
        ...
    


