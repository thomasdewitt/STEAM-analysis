# Data paths and shared constants for turbulon-analysis
from pathlib import Path

# SAM TWPICE configuration
SAM_TWPICE_DATA_DIR = '/Volumes/BLUE/TWPICE'
SAM_TWPICE_TIMESTEPS = ['0000000150', '0000001800', '0000003450']
SAM_TWPICE_VARIABLES = ['QV', 'QT', 'QC', 'QI', 'MSE', 'TABS', 'U', 'V', 'W', 'PP']

# RCEMIP shared data directory
RCEMIP_DATA_DIR = '/Volumes/BLUE/RCEMIP'

# SAM RCEMIP (SAM_CRM) configuration
SAM_RCEMIP_LARGE_TIMESTEPS = ['0000660600', '0000680400', '0000700200', '0000720000']
SAM_RCEMIP_SMALL_TIMESTEPS = ['0001512000', '0001728000', '0001944000', '0002160000']
SAM_RCEMIP_LARGE_VARIABLES = ['hus', 'ta', 'ua', 'va', 'wa', 'pa', 'clw', 'cli', 'plw', 'pli',
                               'hur', 'QV', 'tntr', 'tntrs', 'tntrl']
SAM_RCEMIP_SMALL_VARIABLES = ['U', 'V', 'W', 'PP', 'QRAD', 'TABS', 'QV', 'QN', 'QP',
                               'LQRAD', 'SQRAD']

# CM1 RCEMIP configuration
CM1_LARGE_TIMESTEPS = ['hour1860', 'hour2040', 'hour2220', 'hour2400']
CM1_SMALL_TIMESTEPS = ['hour0840', 'hour0960', 'hour1080', 'hour1200']
CM1_VARIABLES = ['hus', 'ta', 'ua', 'va', 'wa', 'pa', 'clw', 'cli', 'plw', 'pli',
                 'hur', 'tntr', 'tntrs', 'tntrl']

# STEAM configuration
STEAM_DATA_DIR = str(Path(__file__).resolve().parent / 'STEAM' / 'data')
STEAM_FILE_PATTERN = 'steam_*.nc'
STEAM_3D_VARIABLES = ['h', 'qt', 'T', 'qv', 'qc', 'qi', 'p']

# Dropsonde configuration
DROPSONDE_FILE = 'data/dropsonde_data.nc'
DROPSONDE_VARIABLES = ['q', 'ta', 'theta', 'rh', 'u', 'v', 'p', 'wspd', 'wdir']

# Variable mapping: SAM name -> STEAM name
VARIABLE_MAP_SAM_TO_STEAM = {
    'QV': 'qv',
    'QT': 'qt',
    'QC': 'qc',
    'QI': 'qi',
    'TABS': 'T',
    'MSE': 'h',
}
VARIABLE_MAP_STEAM_TO_SAM = {v: k for k, v in VARIABLE_MAP_SAM_TO_STEAM.items()}


# Variable mapping: SAM name -> dropsonde name
VARIABLE_MAP_SAM_TO_DROPSONDE = {
    'QV': 'q',
    'TABS': 'ta',
    'U': 'u',
    'V': 'v',
    'PP': 'p',
}
VARIABLE_MAP_DROPSONDE_TO_SAM = {v: k for k, v in VARIABLE_MAP_SAM_TO_DROPSONDE.items()}


def resolve_variable_pair(variable):
    """Given either a SAM or STEAM variable name, return (sam_name, steam_name)."""
    if variable in VARIABLE_MAP_SAM_TO_STEAM:
        return variable, VARIABLE_MAP_SAM_TO_STEAM[variable]
    if variable in VARIABLE_MAP_STEAM_TO_SAM:
        return VARIABLE_MAP_STEAM_TO_SAM[variable], variable
    raise ValueError(f"No cross-dataset mapping for '{variable}'")


def resolve_to_sam(variable):
    """Map any dataset's variable name to its SAM equivalent."""
    if variable in SAM_TWPICE_VARIABLES:
        return variable
    if variable in VARIABLE_MAP_STEAM_TO_SAM:
        return VARIABLE_MAP_STEAM_TO_SAM[variable]
    if variable in VARIABLE_MAP_DROPSONDE_TO_SAM:
        return VARIABLE_MAP_DROPSONDE_TO_SAM[variable]
    raise ValueError(f"Cannot map '{variable}' to SAM variable")


# Display units and conversion factors (multiply raw data by factor to get display units)
# SAM stores mixing ratios in kg/kg, temperature in K, MSE in J/kg
# STEAM stores same conventions
VARIABLE_UNITS = {
    'QV':   ('g/kg',  1),
    'QT':   ('g/kg',  1),
    'QC':   ('g/kg',  1),
    'QI':   ('g/kg',  1),
    'TABS': ('K',     1.0),
    'MSE':  ('K',     1.0),
    'U':    ('m/s',   1.0),
    'V':    ('m/s',   1.0),
    'W':    ('m/s',   1.0),
    'PP':   ('Pa',    1.0),
    'qv':   ('g/kg',  1e3),
    'qt':   ('g/kg',  1e3),
    'qc':   ('g/kg',  1e3),
    'qi':   ('g/kg',  1e3),
    'T':    ('K',     1.0),
    'h':    ('K',     1.0 / 1004.0),
    'p':    ('Pa',    1.0),
    # Dropsonde variables
    'q':     ('g/kg',  1e3),
    'ta':    ('K',     1.0),
    'theta': ('K',     1.0),
    'rh':    ('',      1.0),
    'wspd':  ('m/s',   1.0),
    'wdir':  ('deg',   1.0),
    # RCEMIP CF-convention variables (SAM_CRM large and CM1)
    'hus':   ('g/kg',  1e3),   # specific humidity kg/kg -> g/kg
    'ua':    ('m/s',   1.0),
    'va':    ('m/s',   1.0),
    'wa':    ('m/s',   1.0),
    'pa':    ('Pa',    1.0),
    'clw':   ('g/kg',  1e3),   # cloud liquid water kg/kg -> g/kg
    'cli':   ('g/kg',  1e3),   # cloud ice kg/kg -> g/kg
    'plw':   ('g/kg',  1e3),   # precip liquid kg/kg -> g/kg
    'pli':   ('g/kg',  1e3),   # precip ice kg/kg -> g/kg
    'hur':   ('',      1.0),   # relative humidity (fraction)
    'tntr':  ('K/s',   1.0),   # total radiative T tendency
    'tntrs': ('K/s',   1.0),   # SW radiative T tendency
    'tntrl': ('K/s',   1.0),   # LW radiative T tendency
}


def get_unit_label(variable):
    """Return display unit string for a variable."""
    return VARIABLE_UNITS.get(variable, ('', 1.0))[0]


def get_unit_factor(variable):
    """Return multiplicative factor to convert raw data to display units."""
    return VARIABLE_UNITS.get(variable, ('', 1.0))[1]

# Shared analysis constants
MAX_HEIGHT = 12000   # meters
MIN_HEIGHT = 100     # meters
HORIZ_STRIDE = 256    # subsample every nth horizontal point
