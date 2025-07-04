"""
This type stub file was generated by pyright.
"""

from enum import IntEnum, unique

RAD2DEG = ...
DEG2RAD = ...
HOUR2SEC = ...
MIN2SEC = ...
SEC2MIN = ...
SEC2HOUR = ...
INV2PI = ...
V2PI = ...
METER2IN = ...
METER2MILES = ...
MILS2METER = ...
SpeedOfLight = ...
def db20(x, inverse=...): # -> float:
    """Convert db20 to decimal and vice versa."""
    ...

def db10(x, inverse=...): # -> float:
    """Convert db10 to decimal and vice versa."""
    ...

def dbw(x, inverse=...): # -> float:
    """Convert W to decimal and vice versa."""
    ...

def dbm(x, inverse=...): # -> float:
    """Convert W to decimal and vice versa."""
    ...

def fah2kel(val, inverse=...):
    """Convert a temperature from Fahrenheit to Kelvin.

    Parameters
    ----------
    val : float
        Temperature value in Fahrenheit.
    inverse : bool, optional
        The default is ``True``.

    Returns
    -------
    float
        Temperature value converted to Kelvin.

    """
    ...

def cel2kel(val, inverse=...):
    """Convert a temperature from Celsius to Kelvin.

    Parameters
    ----------
    val : float
        Temperature value in Celsius.
    inverse : bool, optional
        The default is ``True``.

    Returns
    -------
    float
        Temperature value converted to Kelvin.

    """
    ...

def unit_system(units): # -> str | Literal[False]:
    """Retrieve the name of the unit system associated with a unit string.

    Parameters
    ----------
    units : str
        Units for retrieving the associated unit system name.

    Returns
    -------
    str
        Key from the ``AEDT_units`` when successful. For example, ``"AngularSpeed"``.
    ``False`` when the units specified are not defined in AEDT units.

    """
    ...

def unit_converter(values, unit_system=..., input_units=..., output_units=...): # -> list[Any]:
    """Convert unit in specified unit system.

    Parameters
    ----------
    values : float, list
        Values to convert.
    unit_system : str
        Unit system. Default is `"Length"`.
    input_units : str
        Input units. Default is `"meter"`.
    output_units : str
        Output units. Default is `"mm"`.

    Returns
    -------
    float, list
        Converted value.
    """
    ...

def scale_units(scale_to_unit): # -> float:
    """Find the scale_to_unit into main system unit.

    Parameters
    ----------
    scale_to_unit : str
        Unit to Scale.

    Returns
    -------
    float
        Return the scaling factor if any.
    """
    ...

def validate_enum_class_value(cls, value): # -> Literal[False]:
    """Check whether the value for the class ``enumeration-class`` is valid.

    Parameters
    ----------
    cls : class
        Enumeration-style class with integer members in range(0, N) where cls.Invalid equals N-1.
    value : int
        Value to check.

    Returns
    -------
    bool
        ``True`` when the value is valid for the ``enumeration-class``, ``False`` otherwise.
    """
    ...

AEDT_UNITS = ...
SI_UNITS = ...
UNIT_SYSTEM_OPERATIONS = ...
class INFINITE_SPHERE_TYPE:
    """INFINITE_SPHERE_TYPE Enumerator class."""
    ...


class FILLET:
    """FilletType Enumerator class."""
    ...


class AXIS:
    """CoordinateSystemAxis Enumerator class.

    This static class defines integer constants corresponding to the
    Cartesian axes: X, Y, and Z. Attributes can be
    assigned to the `orientation` keyword argument for
    geometry creation methods of
    the :class:`ansys.aedt.core.modeler.modeler_3d.Modeler3D` and
    :class:`ansys.aedt.core.modeler.modeler_2d.Modeler2D` classes.

    Attributes
    ----------
    X : int
        Axis index corresponding to the X axis (typically 0).
    Y : int
        Axis index corresponding to the Y axis (typically 1).
    Z : int
        Axis index corresponding to the Z axis (typically 2).

    Examples
    --------
    >>> from ansys.aedt.core.generic import constants
    >>> from ansys.aedt.core import Hfss
    >>> hfss = Hfss()
    >>> cylinder1 = hfss.modeler.create_cylinder(
    ...     orientation=constants.AXIS.Z,
    ...     origin=[0, 0, 0],
    ...     radius="0.5mm",
    ...     height="3cm",
    ... )

    """
    ...


class PLANE:
    """CoordinateSystemPlane Enumerator class.

    This static class defines integer constants corresponding to the
    Y-Z, Z-X, and X-Y planes of the current coordinate system.

    Attributes
    ----------
    YZ : int
        Y-Z plane (typically 0).
    ZX : int
        Z-X plane (typically 1).
    XY : int
        X-Y plane (typically 2).

    """
    ...


class GRAVITY:
    """GravityDirection Enumerator class.

    This static class defines integer constants corresponding to the
    positive direction of gravity force.

    Attributes
    ----------
    XNeg : int
        Positive gravity force is in the -X direction.
    YNeg : int
        Positive gravity force is in the -Y direction.
    ZNeg : int
        Positive gravity force is in the -Z direction.
    XPos : int
        Positive gravity force is in the +X direction.
    YPos : int
        Positive gravity force is in the +Y direction.
    ZPos : int
        Positive gravity force is in the +Z direction.

    """
    ...


class VIEW:
    """View Enumerator class.

    This static class defines integer constants corresponding to the
    Y-Z, Z-X, and X-Y planes of the current coordinate system.

    Attributes
    ----------
    XY : int
        Set the view along the Z-Axis (view of the XY plane).
    YZ : int
        Set the view along the X-Axis (view of the YZ plane).
    ZX : int
        Set the view along the Y-Axis (view of the ZX plane).
    ISO : int
        Isometric view from the (x=1, y=1, z=1) direction.
    """
    ...


class GLOBALCS:
    """GlobalCS Enumerator class."""
    ...


class MATRIXOPERATIONSQ3D:
    """Matrix Reduction types."""
    ...


class MATRIXOPERATIONSQ2D:
    """Matrix Reduction types."""
    ...


class CATEGORIESQ3D:
    """Plot Categories for Q2d and Q3d."""
    class Q2D:
        ...
    
    
    class Q3D:
        ...
    
    


class CSMODE:
    """COORDINATE SYSTEM MODE Enumerator class."""
    ...


class SEGMENTTYPE:
    """CROSSSECTION Enumerator class."""
    ...


class CROSSSECTION:
    """CROSSSECTION Enumerator class."""
    ...


class SWEEPDRAFT:
    """SweepDraftType Enumerator class."""
    ...


class FlipChipOrientation:
    """Chip orientation enumerator class."""
    ...


class SolverType:
    """Provides solver type classes."""
    ...


class CutoutSubdesignType:
    ...


class RadiationBoxType:
    ...


class SweepType:
    ...


class BasisOrder:
    """Enumeration-class for HFSS basis order settings.

    Warning: the value ``single`` has been renamed to ``Single`` for consistency. Please update references to
    ``single``.
    """
    ...


class NodeType:
    """Type of node for source creation."""
    ...


class SourceType:
    """Type of excitation enumerator."""
    ...


class SOLUTIONS:
    """Provides the names of default solution types."""
    class Hfss:
        """Provides HFSS solution types."""
        ...
    
    
    class Maxwell3d:
        """Provides Maxwell 3D solution types."""
        ...
    
    
    class Maxwell2d:
        """Provides Maxwell 2D solution types."""
        ...
    
    
    class Icepak:
        """Provides Icepak solution types."""
        ...
    
    
    class Circuit:
        """Provides Circuit solution types."""
        ...
    
    
    class Mechanical:
        """Provides Mechanical solution types."""
        ...
    
    


class SETUPS:
    """Provides constants for the default setup types."""
    ...


CSS4_COLORS = ...
class LineStyle:
    """Provides trace line style constants."""
    ...


class TraceType:
    """Provides trace type constants."""
    ...


class SymbolStyle:
    """Provides symbol style constants."""
    ...


@unique
class EnumUnits(IntEnum):
    hz = ...
    khz = ...
    mhz = ...
    ghz = ...
    thz = ...
    rps = ...
    dbFrequency = ...
    mohm = ...
    ohm = ...
    kohm = ...
    megohm = ...
    gohm = ...
    dbResistance = ...
    fsie = ...
    psie = ...
    nsie = ...
    usie = ...
    msie = ...
    sie = ...
    dbConductance = ...
    fh = ...
    ph = ...
    nh = ...
    uh = ...
    mh = ...
    h = ...
    dbInductance = ...
    ff = ...
    pf = ...
    nf = ...
    uf = ...
    mf = ...
    farad = ...
    dbCapacitance = ...
    nm = ...
    um = ...
    mm = ...
    meter = ...
    cm = ...
    ft = ...
    inch = ...
    mil = ...
    uin = ...
    dbLength = ...
    fs = ...
    ps = ...
    ns = ...
    us = ...
    ms = ...
    s = ...
    dbTime = ...
    deg = ...
    rad = ...
    dbAngle = ...
    fw = ...
    pw = ...
    nw = ...
    uw = ...
    mw = ...
    w = ...
    dbm = ...
    dbw = ...
    dbPower = ...
    fv = ...
    pv = ...
    nv = ...
    uv = ...
    mv = ...
    v = ...
    kv = ...
    dbv = ...
    dbVoltage = ...
    fa = ...
    pa = ...
    na = ...
    ua = ...
    ma = ...
    a = ...
    dbCurrent = ...
    kel = ...
    cel = ...
    fah = ...
    dbTemperature = ...
    dbchz = ...
    dbNoiseSpectrum = ...
    oz_obsolete = ...
    dbWeight_obsolete = ...
    L = ...
    ml = ...
    dbVolume = ...
    gauss = ...
    ugauss = ...
    tesla = ...
    utesla = ...
    dbMagInduction = ...
    oersted = ...
    a_per_meter = ...
    dbMagFieldStrength = ...
    fnewton = ...
    pnewton = ...
    nnewton = ...
    unewton = ...
    mnewton = ...
    newton = ...
    knewton = ...
    megnewton = ...
    gnewton = ...
    lbForce = ...
    dbForce = ...
    fnewtonmeter = ...
    pnewtonmeter = ...
    nnewtonmeter = ...
    unewtonmeter = ...
    mnewtonmeter = ...
    newtonmeter = ...
    knewtonmeter = ...
    megnewtonmeter = ...
    gnewtonmeter = ...
    OzIn = ...
    LbIn = ...
    ftlb = ...
    GmCm = ...
    KgCm = ...
    KgMeter = ...
    dbTorque = ...
    mmps = ...
    cmps = ...
    mps = ...
    mph = ...
    fps = ...
    fpm = ...
    miph = ...
    dbSpeed = ...
    rpm = ...
    degps = ...
    degpm = ...
    degph = ...
    radps = ...
    radpm = ...
    radph = ...
    dbAngularSpeed = ...
    weber = ...
    dbFlux = ...
    vpm = ...
    ugram = ...
    mgram = ...
    gram = ...
    kgram = ...
    oz = ...
    lb = ...
    dbMass = ...
    km = ...
    megv = ...
    ka = ...
    mtesla = ...
    mgauss = ...
    kgauss = ...
    ktesla = ...
    ka_per_meter = ...
    koersted = ...
    kw = ...
    horsepower = ...
    btu_per_hr = ...
    km_per_hr = ...
    km_per_min = ...
    km_per_sec = ...
    mi_per_min = ...
    mi_per_sec = ...
    in_per_sec = ...
    n_per_msq = ...
    kn_per_msq = ...
    megn_per_msq = ...
    gn_per_msq = ...
    psi = ...
    kpsi = ...
    megpsi = ...
    gpsi = ...
    upascal = ...
    mpascal = ...
    cpascal = ...
    dpascal = ...
    pascal = ...
    hpascal = ...
    kpascal = ...
    megpascal = ...
    gpascal = ...
    inps2 = ...
    mmps2 = ...
    cmps2 = ...
    mps2 = ...
    aigb2 = ...
    aigb1 = ...
    nmol = ...
    umol = ...
    mmol = ...
    mol = ...
    kmol = ...
    degsec = ...
    degmin = ...
    perdeg = ...
    perrad = ...
    vpervperdeg = ...
    vpervperrad = ...
    degpers2 = ...
    pers2 = ...
    radpers2 = ...
    dmsperrad = ...
    nmsperrad = ...
    kmsperrad = ...
    jerk_degpers2 = ...
    jerk_radpers2 = ...
    kgm2pers = ...
    newtonmetersec = ...
    AngRps = ...
    AngSpers = ...
    newtonmeterperdeg = ...
    newtonmeterperrad = ...
    kgm2perrad2 = ...
    nms2perrad2 = ...
    in2 = ...
    ft2 = ...
    um2 = ...
    mm2 = ...
    cm2 = ...
    m2 = ...
    km2 = ...
    percm2 = ...
    perm2 = ...
    m2perhour = ...
    m2permin = ...
    m2pers = ...
    m2perJs = ...
    m2perw = ...
    Am2perkW = ...
    Am2perW = ...
    m2perkV = ...
    m2perV = ...
    Am2perWKel = ...
    m2perVKel = ...
    bigB1 = ...
    bigB2 = ...
    Fpercm2 = ...
    nFperm2 = ...
    uFperm2 = ...
    mFperm2 = ...
    Fperm2 = ...
    pFperVm2 = ...
    nFperVm2 = ...
    uFperVm2 = ...
    mFperVm2 = ...
    FperVm2 = ...
    pFperm = ...
    nFperm = ...
    uFperm = ...
    mFperm = ...
    Fperm = ...
    nFperCel = ...
    uFperCel = ...
    FperCel = ...
    pFperFah = ...
    nFperFah = ...
    uFperFah = ...
    mFperFah = ...
    FperFah = ...
    mFperCel = ...
    pFperCel = ...
    pFperKel = ...
    nFperKel = ...
    uFperKel = ...
    mFperKel = ...
    FperKel = ...
    As = ...
    Ah = ...
    nC = ...
    uC = ...
    mC = ...
    C = ...
    kC = ...
    inperlbf = ...
    cmperN = ...
    mperN = ...
    mho = ...
    inverseohm = ...
    apv = ...
    ksie = ...
    megsie = ...
    uSperm = ...
    mSperm = ...
    cSperm = ...
    Sperm = ...
    kSperm = ...
    Apermin = ...
    Aperhour = ...
    uApers = ...
    mApers = ...
    Apers = ...
    kApers = ...
    uApercm2 = ...
    mApercm2 = ...
    Apercm2 = ...
    uAperm2 = ...
    mAperm2 = ...
    Aperm2 = ...
    Apermm2 = ...
    ApermA = ...
    mAperA = ...
    AperA = ...
    uAmperV = ...
    mAmperV = ...
    AmperV = ...
    AperC = ...
    AperAhour = ...
    AperAs = ...
    uAperWperm2 = ...
    mAperWperm2 = ...
    AperWperm2 = ...
    kAperWperm2 = ...
    uAperm = ...
    mAperm = ...
    Aperm = ...
    kAperm = ...
    AperKel2half = ...
    AperKel3 = ...
    AperCelDiff3 = ...
    AperKelDiff3 = ...
    mA2s = ...
    A2s = ...
    uAperCel = ...
    mAperCel = ...
    AperCel = ...
    mAperFah = ...
    AperFah = ...
    uAperKel = ...
    mAperKel = ...
    AperKel = ...
    kAperKel = ...
    mNsperm = ...
    cNsperm = ...
    dNsperm = ...
    Nsperm = ...
    kNsperm = ...
    gpcm3 = ...
    gpl = ...
    kgpl = ...
    kgpdm3 = ...
    kgpm3 = ...
    vpcm = ...
    nCperm2 = ...
    uCperm2 = ...
    mCperm2 = ...
    Cperm2 = ...
    Whour = ...
    kWhour = ...
    eV = ...
    erg = ...
    Ws = ...
    uJ = ...
    mJ = ...
    J = ...
    kJ = ...
    megJ = ...
    GJ = ...
    cm3perPa = ...
    m3perPa = ...
    cm3perPas = ...
    m3perPas = ...
    Nsperm5 = ...
    Pasperm3 = ...
    maxwell = ...
    vh = ...
    vs = ...
    dyne = ...
    kpond = ...
    persec = ...
    lmperm2 = ...
    lmpercm2 = ...
    Wperm2 = ...
    Wpercm2 = ...
    lmperin2 = ...
    lx = ...
    klx = ...
    meglx = ...
    pHperm = ...
    nHperm = ...
    uHperm = ...
    mHperm = ...
    Hperm = ...
    kgperm4 = ...
    Ns2perm5 = ...
    Pas2perm3 = ...
    IrradWpercm2 = ...
    Wperin2 = ...
    uWperm2 = ...
    mWperm2 = ...
    IrradWperm2 = ...
    kWperm2 = ...
    megWperm2 = ...
    inpers3 = ...
    nmpers3 = ...
    umpers3 = ...
    mmpers3 = ...
    cmpers3 = ...
    mpers3 = ...
    yd = ...
    mileUS = ...
    ltyr = ...
    mileNaut = ...
    fm = ...
    pm = ...
    dm = ...
    mileTerr = ...
    m2perV2 = ...
    percm = ...
    permm = ...
    perum = ...
    perkm = ...
    perin = ...
    VperVperm = ...
    VperVperin = ...
    perm = ...
    umperV = ...
    mmperV = ...
    cmperV = ...
    dmperV = ...
    mperV = ...
    kmperV = ...
    umperVhalf = ...
    mmperVhalf = ...
    cmperVhalf = ...
    dmperVhalf = ...
    mperVhalf = ...
    kmperVhalf = ...
    gm2pers3 = ...
    mlm = ...
    lm = ...
    klm = ...
    meglm = ...
    mCd = ...
    Cd = ...
    kCd = ...
    megCd = ...
    GCd = ...
    AperVs = ...
    AperWb = ...
    gpers = ...
    kgpers = ...
    molperdm3 = ...
    molpercm3 = ...
    molperl = ...
    molperm3 = ...
    uJpermol = ...
    mJpermol = ...
    Jpermol = ...
    kJpermol = ...
    megJpermol = ...
    gJpermol = ...
    umolpers = ...
    mmolpers = ...
    cmolpers = ...
    molpers = ...
    kmolpers = ...
    Paspermol = ...
    lbin2 = ...
    lbft2 = ...
    kgm2 = ...
    gmpers = ...
    kgmpers = ...
    percent = ...
    percentperm = ...
    percentperhour = ...
    percentperday = ...
    pers = ...
    permin = ...
    perhour = ...
    perday = ...
    percentpers = ...
    VsperA = ...
    WbperA = ...
    megw = ...
    gw = ...
    mbar = ...
    bar = ...
    mmh2o = ...
    mmhg = ...
    techAtm = ...
    torr = ...
    stAtm = ...
    psipermin = ...
    statmpermin = ...
    techatmpermin = ...
    mmH2Opermin = ...
    torrpermin = ...
    Paperhour = ...
    mbarperhour = ...
    barperhour = ...
    psiperhour = ...
    statmperhour = ...
    techatmperhour = ...
    mbarpers = ...
    barpers = ...
    mmH2Operhour = ...
    torrperhour = ...
    psipers = ...
    statmpers = ...
    techatmpers = ...
    mmH2Opers = ...
    torrpers = ...
    Papermin = ...
    mbarpermin = ...
    barpermin = ...
    Papers = ...
    perbar = ...
    permbar = ...
    perpsi = ...
    perstatm = ...
    pertechatm = ...
    permmHg = ...
    permmH2O = ...
    VperVperPa = ...
    perPa = ...
    bel = ...
    permegW = ...
    permW = ...
    perkW = ...
    pergW = ...
    perJs = ...
    perW = ...
    perOhmAs = ...
    perOhmAh = ...
    perOhmC = ...
    perOhmmin = ...
    perOhmhour = ...
    perOhms = ...
    uohm = ...
    OhmperAs = ...
    OhmperAhour = ...
    nOhmperC = ...
    uOhmperC = ...
    mOhmperC = ...
    OhmperC = ...
    kOhmperC = ...
    megOhmperC = ...
    gOhmperC = ...
    Ohmperum = ...
    uOhmperm = ...
    mOhmperm = ...
    Ohmperm = ...
    kOhmperm = ...
    megOhmperm = ...
    OhmperCel = ...
    mOhmperKel = ...
    OhmperKel = ...
    kOhmperKel = ...
    Ohmmm2permm = ...
    Ohmum = ...
    Ohmcm = ...
    Ohmm = ...
    mJperKelkg = ...
    JperKelkg = ...
    kJperKelkg = ...
    Npercm = ...
    lbfperin = ...
    Nperm = ...
    kNperm = ...
    SufCDAsperm2 = ...
    SufCDnCperm2 = ...
    SufCDuCperm2 = ...
    SufCDmCperm2 = ...
    SufCDCperm2 = ...
    cm2perVs = ...
    m2perVs = ...
    cm2pV2s = ...
    m2pV2s = ...
    mkel = ...
    ckel = ...
    dkel = ...
    kelm2pw = ...
    celm2pw = ...
    perCel = ...
    perFah = ...
    percentperKel = ...
    percentperCel = ...
    percentperFah = ...
    perKel = ...
    perCel2 = ...
    perFah2 = ...
    perKel2 = ...
    celdiff = ...
    mkeldiff = ...
    keldiff = ...
    WsperKel = ...
    JperKel = ...
    mWperCel = ...
    WperCel = ...
    kWperCel = ...
    mWperKel = ...
    WperKel = ...
    kWperKel = ...
    mWperKelm = ...
    WperKelm = ...
    wpcm2kel = ...
    wpm2kel = ...
    mWperKel4 = ...
    WperKel4 = ...
    kWperKel4 = ...
    Wpercm2Kel4 = ...
    Wperm2Kel4 = ...
    KelsperJ = ...
    KelperW = ...
    min = ...
    hour = ...
    day = ...
    sperdeg = ...
    sperrev = ...
    msperrad = ...
    sperrad = ...
    s2perdeg2 = ...
    s2perrad2 = ...
    cnewtonmeter = ...
    mAperV = ...
    AperV = ...
    kAperV = ...
    mAperV2 = ...
    AperV2 = ...
    inpers2 = ...
    cmpers2 = ...
    dmpers2 = ...
    mpers2 = ...
    VelSatumperV = ...
    VelSatmmperV = ...
    VelSatcmperV = ...
    VelSatmperV = ...
    umperV2 = ...
    mmperV2 = ...
    cmperV2 = ...
    mperV2 = ...
    Nsperm2 = ...
    cpoise = ...
    poise = ...
    uPas = ...
    mPas = ...
    cPas = ...
    dPas = ...
    Pas = ...
    hPas = ...
    kPas = ...
    VisFricmNsperm = ...
    VisFricCNsperm = ...
    VisFricNsperm = ...
    VisFrickNsperm = ...
    gv = ...
    mVperm2pers2 = ...
    Vperm2pers2 = ...
    Vpermin = ...
    Vperhour = ...
    mVpers = ...
    Vpers = ...
    kVpers = ...
    permV = ...
    perkV = ...
    perV = ...
    perV2 = ...
    mV3 = ...
    V3 = ...
    VpermV = ...
    mVperV = ...
    VperV = ...
    mVpermpers3 = ...
    Vpermpers3 = ...
    uVm = ...
    mVm = ...
    Vm = ...
    kVm = ...
    pVpercell = ...
    nVpercell = ...
    uVpercell = ...
    mVpercell = ...
    Vpercell = ...
    kVpercell = ...
    megVpercell = ...
    gVpercell = ...
    Vpermhalf = ...
    mVperPahalf = ...
    VperPahalf = ...
    Vhalf = ...
    perVhalf = ...
    uVperCel10 = ...
    mVperCel10 = ...
    VperCel10 = ...
    uVperKel10 = ...
    mVperKel10 = ...
    VperKel10 = ...
    uVperCel11 = ...
    mVperCel11 = ...
    VperCel11 = ...
    uVperKel11 = ...
    mVperKel11 = ...
    VperKel11 = ...
    uVperCel12 = ...
    mVperCel12 = ...
    VperCel12 = ...
    uVperKel12 = ...
    mVperKel12 = ...
    VperKel12 = ...
    uVperCel13 = ...
    mVperCel13 = ...
    VperCel13 = ...
    uVperKel13 = ...
    mVperKel13 = ...
    VperKel13 = ...
    uVperCel14 = ...
    mVperCel14 = ...
    VperCel14 = ...
    uVperKel14 = ...
    mVperKel14 = ...
    VperKel14 = ...
    uVperCel15 = ...
    mVperCel15 = ...
    VperCel15 = ...
    uVperKel15 = ...
    mVperKel15 = ...
    VperKel15 = ...
    uVperCel2 = ...
    mVperCel2 = ...
    VperCel2 = ...
    uVperKel2 = ...
    mVperKel2 = ...
    VperKel2 = ...
    uVperCel3 = ...
    mVperCel3 = ...
    VperCel3 = ...
    uVperKel3 = ...
    mVperKel3 = ...
    VperKel3 = ...
    uVperCel4 = ...
    mVperCel4 = ...
    VperCel4 = ...
    uVperKel4 = ...
    mVperKel4 = ...
    VperKel4 = ...
    uVperCel5 = ...
    mVperCel5 = ...
    VperCel5 = ...
    uVperKel5 = ...
    mVperKel5 = ...
    VperKel5 = ...
    uVperCel6 = ...
    mVperCel6 = ...
    VperCel6 = ...
    uVperKel6 = ...
    mVperKel6 = ...
    VperKel6 = ...
    uVperCel7 = ...
    mVperCel7 = ...
    VperCel7 = ...
    uVperKel7 = ...
    mVperKel7 = ...
    VperKel7 = ...
    uVperCel8 = ...
    mVperCel8 = ...
    VperCel8 = ...
    uVperKel8 = ...
    mVperKel8 = ...
    VperKel8 = ...
    uVperCel9 = ...
    mVperCel9 = ...
    VperCel9 = ...
    uVperKel9 = ...
    mVperKel9 = ...
    VperKel9 = ...
    uVperCel = ...
    mVperCel = ...
    VperCel = ...
    uVperKel = ...
    mVperKel = ...
    VperKel = ...
    mm3 = ...
    m3 = ...
    galUK = ...
    cup = ...
    galUS = ...
    percm3 = ...
    perm3 = ...
    VolFConcm3perPas = ...
    VolFConm3perPas = ...
    m3persPahalf = ...
    m3permin = ...
    m3perhour = ...
    cm3pers = ...
    m3pers = ...
    ltrpermin = ...
    cm3pers2 = ...
    m3pers2 = ...
    mton = ...
    Wirein2 = ...
    Wireft2 = ...
    Wireum2 = ...
    Wiremm2 = ...
    Wirecm2 = ...
    Wirem2 = ...
    JPerM3 = ...
    kJPerM3 = ...
    cm3 = ...
    inch3 = ...
    foot3 = ...
    yard3 = ...
    at = ...
    uat = ...
    nat = ...
    mat = ...
    kat = ...
    Fraction = ...
    fah_diff = ...
    ckel_diff = ...
    dkel_diff = ...
    ft2pers = ...
    cm2pers = ...
    kgpermol = ...
    gpermol = ...
    kgperms = ...
    lbmperfts = ...
    slugperfts = ...
    celin2pw = ...
    celmm2pw = ...
    btupspft2 = ...
    btuphrpft2 = ...
    ergpspcm2 = ...
    micron2 = ...
    mil2 = ...
    btuPerFahSec = ...
    btuPerRankSec = ...
    btuPerFahHr = ...
    btuPerRankHr = ...
    btuPerFahFtSec = ...
    btuPerRankFtSec = ...
    btuPerFahFtHr = ...
    btuPerRankFtHr = ...
    calpersmCel = ...
    calpersmKel = ...
    ergperscmKel = ...
    wPerCelM = ...
    lbmPerFt3 = ...
    slugPerFt3 = ...
    btuPerFahSecFt2 = ...
    btuPerFahHrFt2 = ...
    btuPerRankSecFt2 = ...
    btuPerRankHrFt2 = ...
    wPerCelM2 = ...
    copperOzPerFt2 = ...
    lbmPerSec = ...
    lbmPerMin = ...
    btuPerSec = ...
    ergPerSec = ...
    IrradWPerMm2 = ...
    IrradMet = ...
    btuPerSecFt3 = ...
    btuPerHrFt3 = ...
    ergPerSecCm3 = ...
    wPerM3 = ...
    lbfPerFt2 = ...
    celPerW = ...
    fahSecPerBtu = ...
    btuPerLbmFah = ...
    btuPerLbmRank = ...
    calPerGKel = ...
    calPerGCel = ...
    ergPerGKel = ...
    JPerCelKg = ...
    kcalPerKgKel = ...
    kcalPerKgCel = ...
    rank = ...
    rankdiff = ...
    m2PerSec3 = ...
    ft2PerSec3 = ...
    m2PerSec2 = ...
    ft2PerSec2 = ...
    dissPerSec = ...
    perRank = ...
    percentperRank = ...
    ft3PerMin = ...
    ft3PerSec = ...
    cfm = ...
    pressWaterInches = ...
    q = ...
    bps = ...
    kbps = ...
    mbps = ...
    gbps = ...
    kgpersm2 = ...
    lbmperminft2 = ...
    gperscm2 = ...
    Wperm2perCel = ...
    Wperin2perCel = ...
    Wpermm2perCel = ...
    dBperm = ...
    dBpercm = ...
    dBperdm = ...
    dBperkm = ...
    dBperft = ...
    dBpermi = ...
    Nppercm = ...
    Npperdm = ...
    Npperft = ...
    Npperkm = ...
    Npperm = ...
    Nppermi = ...


@unique
class AllowedMarkers(IntEnum):
    Octahedron = ...
    Tetrahedron = ...
    Sphere = ...
    Box = ...
    Arrow = ...


