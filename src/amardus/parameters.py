import numpy as np
from numpy.typing import NDArray
from typing import Literal, NamedTuple, TypeAlias

FLOAT: TypeAlias = np.float64
INDEX: TypeAlias = np.int32
Array: TypeAlias = NDArray[FLOAT]
IndexArray: TypeAlias = NDArray[INDEX]
ORDER: Literal["C", "F"] = "C"


class Unit(NamedTuple):  
    boltzman = 8.617343e-5   # eV/K
    coulomb = 14.399645478   # eV/(e^2 * A)
    picoseconds: float = 1.0e3 / 1.018051e1
    bar: float = 1.0 / 1.602e6
    grcm3: float = 1.0 / 1.66054
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
