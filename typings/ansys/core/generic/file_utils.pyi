"""
This type stub file was generated by pyright.
"""

from pathlib import Path
from typing import Dict, List, TextIO, Union
from ansys.aedt.core.generic.general_methods import pyaedt_function_handler

is_linux = ...
is_windows = ...
@pyaedt_function_handler(path_in="input_dir")
def normalize_path(input_dir: Union[str, Path], sep: str = ...) -> str:
    """Normalize path separators.

    Parameters
    ----------
    input_dir : str or :class:`pathlib.Path`
        Path to normalize.
    sep : str, optional
        Separator.

    Returns
    -------
    str
        Path normalized to new separator.
    """
    ...

@pyaedt_function_handler(path="input_file")
def get_filename_without_extension(input_file: Union[str, Path]) -> str:
    """Get the filename without its extension.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
        Path of the file.

    Returns
    -------
    str
       Name of the file without extension.
    """
    ...

@pyaedt_function_handler(project_path="input_file")
def is_project_locked(input_file: Union[str, Path]) -> bool:
    """Check if the AEDT project lock file exists.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
        Path for the AEDT project.

    Returns
    -------
    bool
        ``True`` when successful, ``False`` when failed.
    """
    ...

@pyaedt_function_handler(project_path="input_file")
def remove_project_lock(input_file: Union[str, Path]) -> bool:
    """Check if the AEDT project exists and try to remove the lock file.

    .. note::
       This operation is risky because the file could be opened in another AEDT instance.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
        Path for the AEDT project.

    Returns
    -------
    bool
        ``True`` when successful, ``False`` when failed.
    """
    ...

@pyaedt_function_handler()
def check_and_download_file(remote_path: Union[str, Path], overwrite: bool = ...) -> str:
    """Check if a file is remote. Download it or return the path.

    Parameters
    ----------
    remote_path : str or :class:`pathlib.Path`
        Path to the remote file.
    overwrite : bool, optional
        Whether to overwrite the file if it already exists locally.
        The default is ``True``.

    Returns
    -------
    str
        Path to the remote file.
    """
    ...

@pyaedt_function_handler()
def check_if_path_exists(path: Union[str, Path]) -> bool:
    """Check whether a path exists on a local or on a remote machine (for remote sessions only).

    Parameters
    ----------
    path : str or :class:`pathlib.Path`
        Local or remote path to check.

    Returns
    -------
    bool
        ``True`` when exist, ``False`` when fails.
    """
    ...

@pyaedt_function_handler()
def check_and_download_folder(local_path: Union[str, Path], remote_path: Union[str, Path], overwrite: bool = ...) -> str:
    """Download remote folder.

    Parameters
    ----------
    local_path : str or :class:`pathlib.Path`
        Local path to save the folder to.
    remote_path : str or :class:`pathlib.Path`
        Path to the remote folder.
    overwrite : bool, optional
        Whether to overwrite the folder if it already exists locally.
        The default is ``True``.

    Returns
    -------
    str
        Path to the local folder if downloaded, otherwise the remote path.
    """
    ...

@pyaedt_function_handler(rootname="root_name")
def generate_unique_name(root_name: str, suffix: str = ..., n: int = ...) -> str:
    """Generate a new name given a root name and optional suffix.

    Parameters
    ----------
    root_name : str
        Root name to add random characters to.
    suffix : string, optional
        Suffix to add. The default is ``''``.
    n : int, optional
        Number of random characters to add to the name. The default value is ``6``.

    Returns
    -------
    str
        Newly generated name.
    """
    ...

@pyaedt_function_handler(rootname="root_name")
def generate_unique_folder_name(root_name: str = ..., folder_name: str = ...) -> str:
    """Generate a new AEDT folder name given a root name.

    Parameters
    ----------
    root_name : str, optional
        Root name for the new folder. The default is ``None``.
    folder_name : str, optional
        Name for the new AEDT folder if one must be created.

    Returns
    -------
    str
        Newly generated name.
    """
    ...

@pyaedt_function_handler(rootname="root_name")
def generate_unique_project_name(root_name: str = ..., folder_name: str = ..., project_name: str = ..., project_format: str = ...): # -> str:
    """Generate a new AEDT project name given a root name.

    Parameters
    ----------
    root_name : str, optional
        Root name where the new project is to be created.
    folder_name : str, optional
        Name of the folder to create. The default is ``None``, in which case a random folder
        is created. Use ``""`` if you do not want to create a subfolder.
    project_name : str, optional
        Name for the project. The default is ``None``, in which case a random project is
        created. If a project with this name already exists, a new suffix is added.
    project_format : str, optional
        Project format. The default is ``"aedt"``. Options are ``"aedt"`` and ``"aedb"``.

    Returns
    -------
    str
        Newly generated name.
    """
    ...

@pyaedt_function_handler(startpath="path", filepattern="file_pattern")
def recursive_glob(path: Union[str, Path], file_pattern: str): # -> list[Any] | list[str]:
    """Get a list of files matching a pattern, searching recursively from a start path.

    Parameters
    ----------
    path : str or :class:`pathlib.Path`
        Starting path.
    file_pattern : str
        File pattern to match.

    Returns
    -------
    list
        List of files matching the given pattern.
    """
    ...

@pyaedt_function_handler()
def open_file(file_path: Union[str, Path], file_options: str = ..., encoding: str = ..., override_existing: bool = ...) -> Union[TextIO, None]:
    """Open a file and return the object.

    Parameters
    ----------
    file_path : str or :class:`pathlib.Path`
        Full absolute path to the file (either local or remote).
    file_options : str, optional
        Options for opening the file.
    encoding : str, optional
        Name of the encoding used to decode or encode the file.
        The default is ``None``, which means a platform-dependent encoding is used. You can
        specify any encoding supported by Python.
    override_existing : bool, optional
        Whether to override an existing file if opening a file in write mode on a remote
        machine. The default is ``True``.

    Returns
    -------
    Union[TextIO, None]
        Opened file object or ``None`` if the file or folder does not exist.
    """
    ...

@pyaedt_function_handler(fn="input_file")
def read_json(input_file: Union[str, Path]) -> dict:
    """Load a JSON file to a dictionary.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
        Full path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON file as a dictionary.
    """
    ...

@pyaedt_function_handler(file_path="input_file")
def read_toml(input_file: Union[str, Path]) -> dict:
    """Read a TOML file and return as a dictionary.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
        Full path to the TOML file.

    Returns
    -------
    dict
        Parsed TOML file as a dictionary.
    """
    ...

@pyaedt_function_handler(file_name="input_file")
def read_csv(input_file: Union[str, Path], encoding: str = ...) -> list:
    """Read information from a CSV file and return a list.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
            Full path and name for the CSV file.
    encoding : str, optional
            File encoding for the CSV file. The default is ``"utf-8"``.

    Returns
    -------
    list
        Content of the CSV file.
    """
    ...

@pyaedt_function_handler(filename="input_file")
def read_csv_pandas(input_file: Union[str, Path], encoding: str = ...): # -> DataFrame | None:
    """Read information from a CSV file and return a list.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
            Full path and name for the CSV file.
    encoding : str, optional
            File encoding for the CSV file. The default is ``"utf-8"``.

    Returns
    -------
    :class:`pandas.DataFrame`
        CSV file content.
    """
    ...

@pyaedt_function_handler(output="output_file", quotechar="quote_char")
def write_csv(output_file: str, list_data: list, delimiter: str = ..., quote_char: str = ..., quoting: int = ...) -> bool:
    """Write data to a CSV .

    Parameters
    ----------
    output_file : str
        Full path and name of the file to write the data to.
    list_data : list
        Data to be written to the specified output file.
    delimiter : str
        Delimiter. The default value is ``"|"``.
    quote_char : str
        Quote character. The default value is ``"|"``
    quoting : int
        Quoting character. The default value is ``"csv.QUOTE_MINIMAL"``.
        It can take one any of the following module constants:

        - ``"csv.QUOTE_MINIMAL"`` means only when required, for example, when a
            field contains either the quote char or the delimiter
        - ``"csv.QUOTE_ALL"`` means that quotes are always placed around fields.
        - ``"csv.QUOTE_NONNUMERIC"`` means that quotes are always placed around
            fields which do not parse as integers or floating point
            numbers.
        - ``"csv.QUOTE_NONE"`` means that quotes are never placed around fields.

    Return
    ------
    bool
        ``True`` when successful, ``False`` when failed.
    """
    ...

@pyaedt_function_handler(file_name="input_file")
def read_tab(input_file: Union[str, Path]) -> list:
    """Read information from a TAB file and return a list.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
        Full path and name for the TAB file.

    Returns
    -------
    list
        TAB file content.
    """
    ...

@pyaedt_function_handler(file_name="input_file")
def read_xlsx(input_file: Union[str, Path]): # -> DataFrame | list[Any]:
    """Read information from an XLSX file and return a list.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
        Full path and name for the XLSX file.

    Returns
    -------
    list
        XLSX file content.
    """
    ...

@pyaedt_function_handler()
def read_component_file(input_file: Union[str, Path]) -> dict:
    """Read the component file and extract variables.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
        Path to the component file.

    Returns
    -------
    dict
        Dictionary of variables in the component file.
    """
    ...

@pyaedt_function_handler(file_name="input_file")
def parse_excitation_file(input_file: Union[str, Path], is_time_domain: bool = ..., x_scale: float = ..., y_scale: float = ..., impedance: float = ..., data_format: str = ..., encoding: str = ..., out_mag: str = ..., window: str = ...) -> Union[tuple, bool]:
    """Parse a csv file and convert data in list that can be applied to Hfss and Hfss3dLayout sources.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
        Full name of the input file.
    is_time_domain : bool, optional
        Either if the input data is Time based or Frequency Based. Frequency based data are Mag/Phase (deg).
    x_scale : float, optional
        Scaling factor for x axis.
    y_scale : float, optional
        Scaling factor for y axis.
    data_format : str, optional
        Either `"Power"`, `"Current"` or `"Voltage"`.
    impedance : float, optional
        Excitation impedance. Default is `50`.
    encoding : str, optional
        Csv file encoding.
    out_mag : str, optional
        Output magnitude format. It can be `"Voltage"` or `"Power"` depending on Hfss solution.
    window : str, optional
        Fft window. Options are ``"hamming"``, ``"hanning"``, ``"blackman"``, ``"bartlett"`` or ``None``.

    Returns
    -------
    tuple or bool
        Frequency, magnitude and phase.
    """
    ...

@pyaedt_function_handler(file_path="input_file", unit="units", control_path="output_file")
def tech_to_control_file(input_file: Union[str, Path], units: str = ..., output_file: Union[str, Path] = ...): # -> str:
    """Convert a TECH file to an XML file for use in a GDS or DXF import.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
        Full path to the TECH file.
    units : str, optional
        Tech units. If specified in tech file this parameter will not be used. Default is ``"nm"``.
    output_file : str or :class:`pathlib.Path`, optional
        Path for outputting the XML file.

    Returns
    -------
    str
        Output file path.
    """
    ...

@pyaedt_function_handler()
def get_dxf_layers(input_file: Union[str, Path]) -> List[str]:
    """Read a DXF file and return all layer names.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
        Full path to the DXF file.

    Returns
    -------
    list
        List of layers in the DXF file.
    """
    ...

@pyaedt_function_handler(file_path="input_file")
def read_configuration_file(input_file: Union[str, Path]) -> Union[Dict, List]:
    """Parse a file and return the information in a list or dictionary.

    Parameters
    ----------
    input_file : str or :class:`pathlib.Path`
        Full path to the file. Supported formats are ``"csv"``, ``"json"``, ``"tab"``, ``"toml"``, and ``"xlsx"``.

    Returns
    -------
    Union[Dict, List]
        Dictionary if configuration file is ``"toml"`` or ``"json"``, List is ``"csv"``, ``"tab"`` or ``"xlsx"``.
    """
    ...

@pyaedt_function_handler(dict_in="input_data", full_path="output_file")
def write_configuration_file(input_data: dict, output_file: Union[str, Path]) -> bool:
    """Create a configuration file in JSON or TOML format from a dictionary.

    Parameters
    ----------
    input_data : dict
        Dictionary to write the file to.
    output_file : str or :class:`pathlib.Path`
        Full path to the file, including its extension.

    Returns
    -------
    bool
        ``True`` when successful, ``False`` when failed.
    """
    ...

@pyaedt_function_handler(time_vals="time_values", value="data_values")
def compute_fft(time_values, data_values, window=...) -> Union[tuple, bool]:
    """Compute FFT of input transient data.

    Parameters
    ----------
    time_values : `pandas.Series`
        Time points corresponding to the x-axis of the input transient data.
    data_values : `pandas.Series`
        Points corresponding to the y-axis.
    window : str, optional
        Fft window. Options are "hamming", "hanning", "blackman", "bartlett".

    Returns
    -------
    tuple or bool
        Frequency and values.
    """
    ...

@pyaedt_function_handler()
def available_license_feature(feature: str = ..., input_dir: Union[str, Path] = ..., port: int = ..., name: str = ...) -> int:
    """Check available license feature.

    .. warning::

        Do not execute this function with untrusted function argument, environment
        variables or pyaedt global settings.
        See the :ref:`security guide<ref_security_consideration>` for details.

    The method retrieves the port and name values from the ``ANSYSLMD_LICENSE_FILE`` environment variable if available.
    If not, the default values are applied.

    Parameters
    ----------
    feature : str
        Feature increment name. The default is the ``"electronics_desktop"``.
    input_dir: str or :class:`pathlib.Path`, optional
        AEDT installation path. The default is ``None``, in which case the first identified AEDT
        installation from :func:`ansys.aedt.core.internal.aedt_versions.installed_versions`
        method is taken.
    port : int, optional
        Server port number. The default is ``1055``.
    name : str, optional
        License server name. The default is ``"127.0.0.1"``.

    Returns
    -------
    int
        Number of available license features, ``False`` when license server is down.
    """
    ...

