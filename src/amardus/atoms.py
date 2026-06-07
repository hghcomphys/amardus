from __future__ import annotations
import numpy as np
from typing import NamedTuple
from amardus.parameters import Array, FLOAT, ORDER
from ase import Atoms as AseAtoms


class Atoms(NamedTuple):
    mass: Array
    position: Array
    velocity: Array
    force: Array
    charge: Array
    box: Array
    position_old: Array

    @classmethod
    def from_ase(cls, atoms: AseAtoms) -> Atoms:
        charge = atoms.get_initial_charges()
        mass = atoms.get_masses()
        position = np.asarray(atoms.get_positions(), dtype=FLOAT, order=ORDER)
        velocity = np.asarray(atoms.get_velocities(), dtype=FLOAT, order=ORDER)
        force = np.empty(position.shape, dtype=FLOAT, order=ORDER)
        box = np.asarray(atoms.cell.array, dtype=FLOAT)
        return cls(
            mass=mass,
            position=position,
            velocity=velocity,
            force=force,
            charge=charge,
            box=box,
            position_old=position.copy(),
        )

    def __len__(self):
        return len(self.position)


