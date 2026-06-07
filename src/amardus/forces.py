import math
import numpy as np
from numba import njit
from amardus.system import MDSystem
from amardus.neighbors import apply_pbc
from amardus.parameters import FLOAT


@njit
def compute_forces(sys: MDSystem) -> None:
    sys.atoms.force[:] = FLOAT(0.0)
    sys.energy[0] = FLOAT(0.0)
    sys.virial[0] = FLOAT(0.0)

    compute_lj_forces(sys)
    # compute_coul_forces(sys)


@njit
def compute_coul_forces(
    sys: MDSystem
) -> None:
    compute_ewald_real_forces(sys)
    compute_ewald_reciprocal_forces(sys)
    compute_ewald_self_energy(sys)


@njit
def compute_lj_forces(sys: MDSystem) -> None:
    box = sys.atoms.box
    energy = sys.energy
    virial = sys.virial
    neighbors = sys.neighbors
    force = sys.atoms.force
    position = sys.atoms.position
    cutoff2 = sys.params.cutoff_radius**2
    length = box[0, 0], box[1, 1], box[2, 2]
    num_atoms = len(position)

    epsilon = FLOAT(1.032e-2)
    sigma = FLOAT(3.405)
    sigma3 = sigma * sigma * sigma
    sigma6 = sigma3 * sigma3
    sigma12 = sigma6 * sigma6
    e24s6 = FLOAT(24.0) * epsilon * sigma6
    e48s12 = FLOAT(48.0) * epsilon * sigma12
    e4s6 = FLOAT(4.0) * epsilon * sigma6
    e4s12 = FLOAT(4.0) * epsilon * sigma12


    for ni in range(num_atoms):
        for j in range(neighbors.num_neighbors_per_atom[ni]):
            nj = neighbors.neighbor_index[ni, j]

            if ni < nj:
                dx = apply_pbc(position[nj, 0] - position[ni, 0], length[0])
                dy = apply_pbc(position[nj, 1] - position[ni, 1], length[1])
                dz = apply_pbc(position[nj, 2] - position[ni, 2], length[2])
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > cutoff2:
                    continue

                r2inv = FLOAT(1.0) / r2
                r4inv = r2inv * r2inv
                r6inv = r2inv * r4inv
                r8inv = r4inv * r4inv
                r12inv = r4inv * r8inv
                r14inv = r6inv * r8inv
                fij = e24s6 * r8inv - e48s12 * r14inv
                fij_x, fij_y, fij_z = fij * dx, fij * dy, fij * dz

                force[ni, 0] += fij_x
                force[ni, 1] += fij_y
                force[ni, 2] += fij_z
                force[nj, 0] -= fij_x
                force[nj, 1] -= fij_y
                force[nj, 2] -= fij_z
                energy[0] += e4s12 * r12inv - e4s6 * r6inv
                virial[0] += dx * fij_x + dy * fij_y + dz * fij_z


@njit
def compute_ewald_real_forces(sys: MDSystem) -> None:
    atoms = sys.atoms
    params = sys.params
    neighbors = sys.neighbors
    energy, virial = sys.energy, sys.virial
    box = atoms.box
    force = atoms.force
    position = atoms.position
    charge = atoms.charge
    cutoff2 = params.cutoff_radius **2
    length = box[0, 0], box[1, 1], box[2, 2]
    alpha = params.ewald_alpha
    num_atoms = len(position)
    KE = params.coulomb

    for ni in range(num_atoms):
        for j in range(neighbors.num_neighbors_per_atom[ni]):
            nj = neighbors.neighbor_index[ni, j]

            if ni < nj:
                dx = apply_pbc(position[nj, 0] - position[ni, 0], length[0])
                dy = apply_pbc(position[nj, 1] - position[ni, 1], length[1])
                dz = apply_pbc(position[nj, 2] - position[ni, 2], length[2])
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

                force[ni, 0] += fij_x
                force[ni, 1] += fij_y
                force[ni, 2] += fij_z
                force[nj, 0] -= fij_x
                force[nj, 1] -= fij_y
                force[nj, 2] -= fij_z
                energy[0] += coef_1 * erfc_term
                virial[0] += dx * fij_x + dy * fij_y + dz * fij_z


@njit
def compute_ewald_reciprocal_forces(
    sys: MDSystem
) -> None:

    atoms, params = sys.atoms, sys.params
    energy = sys.energy
    position = atoms.position
    charge = atoms.charge
    force = atoms.force
    box = atoms.box
    vol = np.linalg.det(box)
    length = box[0, 0], box[1, 1], box[2, 2]
    kmax = params.ewald_kmax
    alpha = params.ewald_alpha
    num_atoms = len(position)

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
                        kvec[0] * position[ni, 0]
                        + kvec[1] * position[ni, 1]
                        + kvec[2] * position[ni, 2]
                    )
                    Sk_real += charge[ni] * math.cos(kr)
                    Sk_imag += charge[ni] * math.sin(kr)
                energy_sum += prefactor * (Sk_real**2 + Sk_imag**2)

                # forces
                for ni in range(num_atoms):
                    kr_i = (
                        kvec[0] * position[ni, 0]
                        + kvec[1] * position[ni, 1]
                        + kvec[2] * position[ni, 2]
                    )
                    force_factor = (
                        2.0
                        * prefactor
                        * charge[ni]
                        * (Sk_imag * math.cos(kr_i) - Sk_real * math.sin(kr_i))
                    )
                    force[ni] += force_factor * kvec

        energy[0] += 0.5 * energy_sum


@njit
def compute_ewald_self_energy(
    sys: MDSystem
) -> None:
    atoms, params = sys.atoms, sys.params
    energy = sys.energy
    charge = atoms.charge
    alpha = params.ewald_alpha
    energy[0] += -alpha / math.sqrt(math.pi) * np.sum(charge**2)

