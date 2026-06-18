from __future__ import annotations
import math
import numpy as np
from numba import njit
from typing import NamedTuple, Protocol
from amardus.system import System
from amardus.neighbors import apply_pbc
from amardus.parameters import FLOAT, Array, get_default_unit
from dataclasses import dataclass

KE = get_default_unit().coulomb


# @njit
# def compute_coul_forces(sys: System) -> None:
#     compute_ewald_real_forces(sys)
#     compute_ewald_reciprocal_forces(sys)
#     compute_ewald_self_energy(sys)


class ForceFieldInterface(Protocol):

    def compute_forces(self, sys: System) -> None: ...


@dataclass
class LennardJonesForceField(ForceFieldInterface):
    params: LennardJonesParams
    cutoff_radius: float

    def compute_forces(self, sys: System) -> None:
        sys.atoms.forces[:] = FLOAT(0.0)
        sys.energy[0] = FLOAT(0.0)
        sys.virial[0] = FLOAT(0.0)

        compute_lj_forces(sys, self.params, self.cutoff_radius)


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
def compute_lj_forces(
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


@njit
def compute_ewald_real_forces(
    sys: System,
    cutoff_radius: float,
    alpha: float,
) -> None:
    atoms = sys.atoms
    neighbors = sys.neighbors
    energy, virial = sys.energy, sys.virial
    box = atoms.box
    forces = atoms.forces
    pos = atoms.positions
    charge = atoms.charges
    cutoff2 = cutoff_radius**2
    length = box[0, 0], box[1, 1], box[2, 2]
    num_atoms = pos.shape[0] 

    for ni in range(num_atoms):
        for j in range(neighbors.num_neighbors_per_atom[ni]):
            nj = neighbors.neighbor_index[ni, j]

            if ni < nj:
                dx = apply_pbc(pos[nj, 0] - pos[ni, 0], length[0])
                dy = apply_pbc(pos[nj, 1] - pos[ni, 1], length[1])
                dz = apply_pbc(pos[nj, 2] - pos[ni, 2], length[2])
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > cutoff2:
                    continue

                r = math.sqrt(r2)
                r_inv = FLOAT(1.0) / r

                erfc_term = math.erfc(alpha * r) * r_inv
                coef_1 = KE * charge[ni] * charge[nj]
                coef_2 = FLOAT(2) * alpha / math.sqrt(math.pi)

                fij = (
                    -coef_1
                    * (erfc_term + coef_2 * math.exp(-((alpha * r) ** 2)))
                    * r_inv**2
                )
                fij_x, fij_y, fij_z = fij * dx, fij * dy, fij * dz

                forces[ni, 0] += fij_x
                forces[ni, 1] += fij_y
                forces[ni, 2] += fij_z
                forces[nj, 0] -= fij_x
                forces[nj, 1] -= fij_y
                forces[nj, 2] -= fij_z
                energy[0] += coef_1 * erfc_term
                virial[0] += dx * fij_x + dy * fij_y + dz * fij_z


@njit
def compute_ewald_reciprocal_forces(
    sys: System,
    alpha: float,
    kmax: int,
) -> None:

    atoms = sys.atoms
    energy = sys.energy
    pos = atoms.positions
    charge = atoms.charges
    forces = atoms.forces
    box = atoms.box
    vol = np.linalg.det(box)
    length = box[0, 0], box[1, 1], box[2, 2]
    num_atoms = pos.shape[0] 

    energy_sum = FLOAT(0)
    for nx in range(-kmax, kmax + 1):
        for ny in range(-kmax, kmax + 1):
            for nz in range(-kmax, kmax + 1):

                if nx == 0 and ny == 0 and nz == 0:
                    continue

                kvec = np.array(
                    [
                        2 * math.pi * nx / length[0],
                        2 * math.pi * ny / length[1],
                        2 * math.pi * nz / length[2],
                    ]
                )
                k2 = (kvec**2).sum()
                prefactor = (
                    (4.0 * math.pi / k2) * math.exp(-k2 / (4.0 * alpha**2)) / vol
                )

                # structure factor: S(k)
                Sk_real = 0.0
                Sk_imag = 0.0
                for ni in range(num_atoms):
                    kr = (
                        kvec[0] * pos[ni, 0]
                        + kvec[1] * pos[ni, 1]
                        + kvec[2] * pos[ni, 2]
                    )
                    Sk_real += charge[ni] * math.cos(kr)
                    Sk_imag += charge[ni] * math.sin(kr)
                energy_sum += prefactor * (Sk_real**2 + Sk_imag**2)

                # forces
                for ni in range(num_atoms):
                    kr_i = (
                        kvec[0] * pos[ni, 0]
                        + kvec[1] * pos[ni, 1]
                        + kvec[2] * pos[ni, 2]
                    )
                    force_factor = (
                        2.0
                        * prefactor
                        * charge[ni]
                        * (Sk_imag * math.cos(kr_i) - Sk_real * math.sin(kr_i))
                    )
                    forces[ni] += force_factor * kvec

        energy[0] += 0.5 * energy_sum


@njit
def compute_ewald_self_energy(sys: System, alpha: float) -> None:
    atoms = sys.atoms
    energy = sys.energy
    charge = atoms.charges
    energy[0] += -alpha / math.sqrt(math.pi) * np.sum(charge**2)
