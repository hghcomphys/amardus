from __future__ import annotations
import numpy as np
from numba import njit
from typing import NamedTuple, TypeAlias
from amardus.atoms import Atoms
from amardus.neighbors import CellList, NeighborList
from amardus.parameters import get_default_unit, Array, FLOAT


KB = get_default_unit().boltzman


class System(NamedTuple):
    atoms: Atoms
    cells: CellList
    neighbors: NeighborList
    energy: Array
    virial: Array

    @classmethod
    def create(
        cls,
        atoms: Atoms,
        cell_length: float,
        neighbor_max_atoms: int,
    ) -> System:
        num_atoms = len(atoms)

        return cls(
            atoms=atoms,
            cells = CellList.create(atoms.box, num_atoms, cell_length),
            neighbors=NeighborList.create(num_atoms, neighbor_max_atoms),
            energy=np.zeros(1, dtype=FLOAT),
            virial=np.zeros(1, dtype=FLOAT),
        )


@njit
def get_kinetic_energy(sys: System) -> FLOAT:
    m, v = sys.atoms.masses.reshape(-1, 1), sys.atoms.velocities
    return FLOAT(0.5) * np.sum(m * v * v)


@njit
def get_temperature(sys: System) -> FLOAT:
    m, v = sys.atoms.masses.reshape(-1, 1), sys.atoms.velocities
    return np.mean(m * v * v) / KB


@njit
def get_pressure(sys: System) -> float:
    m, v = sys.atoms.masses.reshape(-1, 1), sys.atoms.velocities
    vol = np.linalg.det(sys.atoms.box)
    virial = sys.virial
    return (np.sum(m * v * v) - virial[0]) / (FLOAT(3.0) * vol)
