from __future__ import annotations
import numpy as np
from numba import njit
from typing import NamedTuple
from amardus.atoms import Atoms
from amardus.neighbors import CellList, NeighborList
from amardus.parameters import get_default_unit, Array, FLOAT

KB = get_default_unit().boltzman


class SystemParams(NamedTuple):
    neighbor_cutoff: float
    neighbor_skin: float
    neighbor_max_atoms: int


class System(NamedTuple):
    atoms: Atoms
    cells: CellList
    neighbors: NeighborList
    energy: Array
    virial: Array
    params: SystemParams

    @classmethod
    def create(
        cls,
        atoms: Atoms,
        neighbor_cutoff: float,
        neighbor_skin: float,
        neighbor_max_atoms: int,
    ) -> System:
        num_atoms = len(atoms)

        return cls(
            atoms=atoms,
            cells=CellList.create(atoms.box, num_atoms, cell_length=neighbor_cutoff),
            neighbors=NeighborList.create(num_atoms, neighbor_max_atoms),
            energy=np.empty(1, dtype=FLOAT),
            virial=np.empty(1, dtype=FLOAT),
            params=SystemParams(neighbor_cutoff, neighbor_skin, neighbor_max_atoms),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_atoms={len(self.atoms)})" 


@njit
def get_kinetic_energy(sys: System) -> FLOAT:
    mass, vel = sys.atoms.masses.reshape(-1, 1), sys.atoms.velocities
    return FLOAT(0.5) * np.sum(mass * vel * vel)


@njit
def get_temperature(sys: System) -> FLOAT:
    mass, vel = sys.atoms.masses.reshape(-1, 1), sys.atoms.velocities
    return np.mean(mass * vel * vel) / KB


@njit
def get_pressure(sys: System) -> float:
    mass, vel = sys.atoms.masses.reshape(-1, 1), sys.atoms.velocities
    vol = np.linalg.det(sys.atoms.box)
    virial = sys.virial
    return (np.sum(mass * vel * vel) - virial[0]) / (FLOAT(3.0) * vol)
