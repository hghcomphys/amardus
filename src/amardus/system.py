from __future__ import annotations
from numba import njit
from typing import NamedTuple
from amardus.atoms import Atoms
from amardus.neighbors import CellList, NeighborList
from amardus.parameters import Parameters, Array, FLOAT
import numpy as np


class MDSystem(NamedTuple):
    atoms: Atoms
    params: Parameters
    cells: CellList
    neighbors: NeighborList
    energy: Array
    virial: Array

    @classmethod
    def build(cls, atoms: Atoms, params: Parameters) -> MDSystem:
        num_atoms = len(atoms.position)

        return cls(
            atoms=atoms, 
            params=params, 
            cells=CellList.build(params, atoms.box, num_atoms), 
            neighbors=NeighborList.build(params, num_atoms), 
            energy = np.empty(1, dtype=FLOAT),
            virial = np.empty(1, dtype=FLOAT),
        )


# @njit
def get_kinetic_energy(sys: MDSystem) -> FLOAT:
    v = sys.atoms.velocity
    m = sys.atoms.mass.reshape(-1, 1)
    return FLOAT(0.5) * np.sum(m * v * v)


# @njit
def get_temperature(sys: MDSystem) -> FLOAT:
    natoms = len(sys.atoms)
    KB = sys.params.boltzman
    return FLOAT(2.0) / (FLOAT(3) * natoms * KB) * get_kinetic_energy(sys)


# @njit
def get_pressure(sys: MDSystem) -> float:
    atoms = sys.atoms
    virial = sys.virial
    vol = np.linalg.det(atoms.box)
    natoms = len(atoms)
    KB = sys.params.boltzman
    temp = get_temperature(sys)
    pres = (natoms * KB * temp - virial[0] / FLOAT(3.0)) / vol
    return pres


