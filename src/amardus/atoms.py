from __future__ import annotations
import numpy as np
from typing import NamedTuple
from amardus.parameters import Array, FLOAT, ORDER, IndexArray, INDEX
from ase import Atoms as AseAtoms


class Atoms(NamedTuple):
    masses: Array
    positions: Array
    velocities: Array
    forces: Array
    charges: Array
    box: Array
    positions_old: Array
    atom_types: IndexArray 

    @classmethod
    def from_ase(cls, atoms: AseAtoms, atom_types_map: dict[str, int]) -> Atoms:

        charges = atoms.get_initial_charges()
        masses = atoms.get_masses()
        positions = np.asarray(atoms.get_positions(), dtype=FLOAT, order=ORDER)
        velocities = np.asarray(atoms.get_velocities(), dtype=FLOAT, order=ORDER)
        forces = np.empty(positions.shape, dtype=FLOAT, order=ORDER)
        box = np.asarray(atoms.cell.array, dtype=FLOAT)
        atom_types = np.array([atom_types_map[s] for s in atoms.get_chemical_symbols()], dtype=INDEX)
        
        return cls(
            masses=masses,
            positions=positions,
            velocities=velocities,
            forces=forces,
            charges=charges,
            box=box,
            positions_old=positions.copy(),
            atom_types=atom_types,
        )

    def __len__(self):
        return self.positions.shape[0]


