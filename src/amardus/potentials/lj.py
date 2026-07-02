from __future__ import annotations
import numpy as np
from numba import njit
from typing import NamedTuple
from dataclasses import dataclass, field
from amardus.system import System
from amardus.neighbors import apply_pbc
from amardus.parameters import FLOAT, Array, get_default_unit
from amardus.potentials.interface import ForceFieldInterface

KE = get_default_unit().coulomb


@dataclass
class LennardJonesPotential(ForceFieldInterface):
    params: LennardJonesParams = field(repr=False)
    cutoff_radius: float

    def compute_forces(self, sys: System) -> None:
        sys.atoms.forces[:] = FLOAT(0.0)
        sys.energy[0] = FLOAT(0.0)
        sys.virial[0] = FLOAT(0.0)

        _compute_lj_forces(sys, self.params, self.cutoff_radius)


class LennardJonesParams(NamedTuple):
    epsilon: Array
    sigma: Array

    @classmethod
    def from_list(
        cls,
        epsilon: list[float],
        sigma: list[float],
    ) -> LennardJonesParams:

        epsilon = np.array(epsilon, dtype=FLOAT)
        sigma = np.array(sigma, dtype=FLOAT)

        epsilon_matrix = np.sqrt(epsilon[:, None] * epsilon[None, :])
        sigma_matrix = 0.5 * (sigma[:, None] + sigma[None, :])

        return cls(epsilon=epsilon_matrix, sigma=sigma_matrix)
    

@njit
def _compute_lj_forces(
    sys: System,
    params: LennardJonesParams,
    cutoff_radius: float,
) -> None:
    box = sys.atoms.box
    energy = sys.energy
    virial = sys.virial
    neighbors = sys.neighbors
    forces = sys.atoms.forces
    pos = sys.atoms.positions
    atom_types = sys.atoms.atom_types
    cutoff2 = cutoff_radius**2
    length = box[0, 0], box[1, 1], box[2, 2]
    num_atoms = len(pos)

    epsilon, sigma = params.epsilon, params.sigma
    sigma3 = sigma * sigma * sigma
    sigma6 = sigma3 * sigma3
    sigma12 = sigma6 * sigma6
    e24s6 = FLOAT(24.0) * epsilon * sigma6
    e48s12 = FLOAT(48.0) * epsilon * sigma12
    e4s6 = FLOAT(4.0) * epsilon * sigma6
    e4s12 = FLOAT(4.0) * epsilon * sigma12

    for ni in range(num_atoms):
        ai = atom_types[ni] 

        for j in range(neighbors.num_neighbors_per_atom[ni]):
            nj = neighbors.neighbor_index[ni, j]
            aj = atom_types[nj]

            if ni < nj:
                dx = apply_pbc(pos[nj, 0] - pos[ni, 0], length[0])
                dy = apply_pbc(pos[nj, 1] - pos[ni, 1], length[1])
                dz = apply_pbc(pos[nj, 2] - pos[ni, 2], length[2])
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > cutoff2:
                    continue

                r2inv = FLOAT(1.0) / r2
                r4inv = r2inv * r2inv
                r6inv = r2inv * r4inv
                r8inv = r4inv * r4inv
                r12inv = r4inv * r8inv
                r14inv = r6inv * r8inv
                fij = e24s6[ai][aj] * r8inv - e48s12[ai][aj] * r14inv
                fij_x, fij_y, fij_z = fij * dx, fij * dy, fij * dz

                forces[ni, 0] += fij_x
                forces[ni, 1] += fij_y
                forces[ni, 2] += fij_z
                forces[nj, 0] -= fij_x
                forces[nj, 1] -= fij_y
                forces[nj, 2] -= fij_z
                energy[0] += e4s12[ai][aj] * r12inv - e4s6[ai][aj] * r6inv
                virial[0] += dx * fij_x + dy * fij_y + dz * fij_z

