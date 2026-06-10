from __future__ import annotations
import numpy as np
from numba import njit
from typing import NamedTuple, TypeAlias
from amardus.atoms import Atoms
from amardus.neighbors import CellList, NeighborList
from amardus.parameters import Parameters, Array, FLOAT


ForceField: TypeAlias = tuple


class MDSystem(NamedTuple):
    atoms: Atoms
    params: Parameters
    cells: CellList
    neighbors: NeighborList
    energy: Array
    virial: Array
    forcefield: ForceField

    @classmethod
    def create(
        cls,
        atoms: Atoms,
        params: Parameters,
        forcefield: ForceField,
    ) -> MDSystem:
        num_atoms = len(atoms.position)

        return cls(
            atoms=atoms,
            params=params,
            cells=CellList.create(params, atoms.box, num_atoms),
            neighbors=NeighborList.create(params, num_atoms),
            energy=np.empty(1, dtype=FLOAT),
            virial=np.empty(1, dtype=FLOAT),
            forcefield=forcefield,
        )


@njit
def get_kinetic_energy(sys: MDSystem) -> FLOAT:
    m, v = sys.atoms.mass.reshape(-1, 1), sys.atoms.velocity
    return FLOAT(0.5) * np.sum(m * v * v)


@njit
def get_temperature(sys: MDSystem) -> FLOAT:
    m, v = sys.atoms.mass.reshape(-1, 1), sys.atoms.velocity
    KB = sys.params.boltzman
    return np.mean(m * v * v) / KB


@njit
def get_pressure(sys: MDSystem) -> float:
    m, v = sys.atoms.mass.reshape(-1, 1), sys.atoms.velocity
    vol = np.linalg.det(sys.atoms.box)
    virial = sys.virial
    return (np.sum(m * v * v) - virial[0]) / (FLOAT(3.0) * vol)
