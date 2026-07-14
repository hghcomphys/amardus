import numpy as np
from numpy.typing import NDArray
from typing import Literal, NamedTuple, TypeAlias

FLOAT: TypeAlias = np.float64
INDEX: TypeAlias = np.int32
Array: TypeAlias = NDArray[FLOAT]
IndexArray: TypeAlias = NDArray[INDEX]
ORDER: Literal["C", "F"] = "C"


global default_unit


class MetalUnit(NamedTuple):
    boltzman = 8.617343e-5  # eV/K
    coulomb = 14.399645478  # eV/(e^2 * A)
    picoseconds: float = 98.2269 
    bar: float = 6.2415e-7
    grcm3: float = 0.602214
    kelvin: float = 1.0
    electronvolt: float = 1.0
    angstrom: float = 1.0


class Parameters(NamedTuple):
    time_step: float
    temperature: float
    pressure: float
    cutoff_radius: float
    skin_radius: float
    thermostat_coupling_time: float
    barostat_coupling_time: float
    compressibility: float
    neighbor_max_atoms: int
    ewald_alpha: float
    ewald_kmax: int
    boltzman: float
    coulomb: float


Unit = MetalUnit
default_unit = Unit()


def set_default_unit(unit: Unit) -> None:
    global default_unit
    default_unit = unit


def get_default_unit() -> Unit:
    global default_unit
    return default_unit
