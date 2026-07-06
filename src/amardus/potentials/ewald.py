from __future__ import annotations
import math
import numpy as np
from numba import njit
from amardus.system import System
from amardus.neighbors import calculate_distance2
from amardus.parameters import FLOAT, get_default_unit

KE = get_default_unit().coulomb


# @njit
# def compute_coul_forces(sys: System) -> None:
#     _compute_ewald_real_forces(sys)
#     _compute_ewald_reciprocal_forces(sys)
#     _compute_ewald_self_energy(sys)


@njit
def _compute_ewald_real_forces(
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
    num_atoms = pos.shape[0] 

    for ni in range(num_atoms):

        for j in range(neighbors.num_neighbors_per_atom[ni]):
            nj = neighbors.neighbor_index[ni, j]

            r2, (dx, dy, dz) = calculate_distance2(pos[ni], pos[nj], box)
            if ni < nj:
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
def _compute_ewald_reciprocal_forces(
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
def _compute_ewald_self_energy(sys: System, alpha: float) -> None:
    atoms = sys.atoms
    energy = sys.energy
    charge = atoms.charges
    energy[0] += -alpha / math.sqrt(math.pi) * np.sum(charge**2)

