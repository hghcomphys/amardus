# MD engine from scratch
# Author: Hossein Ghorbanfekr, May 2026
# https://github.com/hghcomphys

import math
import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import NamedTuple, TextIO

FLOAT = np.float64
Array = NDArray[FLOAT]
ORDER = "C"

K_B = 8.617343e-5  # eV/K
K_e = 14.399645478  # eV/(e^2 * A)

femtosecond = 1.0 / 1.018051e1
bar = 1.0 / 1.602e6
grcm3 = 1.0 / 1.66054


class Parameters(NamedTuple):
    time_step: float
    temperature: float
    pressure: float
    cutoff_radius: float
    skin_radius: float
    thermostat_coupling_time: float
    barostat_coupling_time: float
    compressibility: float
    neighbor_max_atoms: int
    ewald_alpha: float 
    ewald_kmax: int


class Particles(NamedTuple):
    mass: Array
    position: Array
    velocity: Array
    force: Array
    charge: Array
    box: Array
    position_old: Array


class CellList(NamedTuple):
    num_atoms_per_cell: NDArray
    atom_index: NDArray
    sizes: tuple[int, int, int]


class NeighborList(NamedTuple):
    num_neighbors_per_atom: NDArray
    neighbor_index: NDArray


def simulate(
    params: Parameters,
    particles: Particles,
    steps: int = 1,
    log_frequency: int = 100,
    filename: str = "out.xyz",
) -> None:
    box = particles.box
    num_atoms = len(particles.position)

    cells, neighbors = initialize_neighbors(params, box, num_atoms)
    update_neighbors(particles, params, cells, neighbors)

    energy = np.empty(1, dtype=FLOAT)
    virial = np.empty(1, dtype=FLOAT)
    compute_forces(particles, params, neighbors, energy, virial)
    with open(filename, "w") as file:
        # Simulate
        print(
            f"{'Step':8}"
            f" {'Temp[K]':10}"
            f" {'Pres[bar]':10}"
            f" {'PotE[eV]':10}"
            f" {'KinE[eV]':10}"
            f" {'TotE[eV]':10}"
            f" {'UpdateNb':10}"
            f" {'AvgNb':10}"
            f" {'AvgCell':10}"
            f" {'L[Å]':10}"
            f" {'Dens[gr/cm3]':10}"
        )
        num_neighbor_updates = 1
        for step in range(steps):
            if step % log_frequency == 0:

                ke = get_kinetic_energy(particles)
                pe = energy[0]
                temp = get_temperature(particles)
                pres = get_pressure(particles, virial)
                vol = np.linalg.det(particles.box)
                dens = np.sum(particles.mass) / vol
                print(
                    f"{step:<8}"
                    f" {temp:<10.4f}"
                    f" {pres / bar:<10.4f}"
                    f" {pe:<10.4f}"
                    f" {ke:<10.4f}"
                    f" {ke + pe:<10.4f}"
                    f" {num_neighbor_updates:<10}"
                    f" {neighbors.num_neighbors_per_atom.mean():<10.2f}"
                    f" {cells.num_atoms_per_cell.mean():<10.2f}"
                    f" {math.pow(vol, 1/3):<10.4f}"
                    f" {dens / grcm3:<10.6f}"
                )
                # save_structure(particles, file)

            # Next time step (update r, v, and F)
            verlet_integration_position(particles, params)

            if check_if_neighbor_need_update(particles, params):
                # cells = initialize_cells(params, particles.box)
                update_neighbors(particles, params, cells, neighbors)
                num_neighbor_updates += 1

            compute_forces(particles, params, neighbors, energy, virial)
            verlet_integration_velocity(particles, params)

            # thermostat(particles, params)
            # barostat(particles, params, virial)


@njit
def compute_forces(
    particles: Particles,
    params: Parameters,
    neighbors: NeighborList,
    energy: Array,
    virial: Array,
) -> None:

    particles.force[:] = FLOAT(0.0)
    energy[0] = FLOAT(0.0)
    virial[0] = FLOAT(0.0)

    compute_lj_forces(particles, params, neighbors, energy, virial)
    # compute_coul_forces(particles, params, neighbors, energy, virial)


@njit
def compute_coul_forces(
    particles: Particles,
    params: Parameters,
    neighbors: NeighborList,
    energy: Array,
    virial: Array,
) -> None:
    compute_ewald_real_forces(particles, params, neighbors, energy, virial)
    compute_ewald_reciprocal_forces(particles, params, energy, virial)
    compute_ewald_self_energy(particles, params, energy)


@njit
def compute_lj_forces(
    particles: Particles,
    params: Parameters,
    neighbors: NeighborList,
    energy: Array,
    virial: Array,
) -> None:

    epsilon = FLOAT(1.032e-2)
    sigma = FLOAT(3.405)
    sigma3 = sigma * sigma * sigma
    sigma6 = sigma3 * sigma3
    sigma12 = sigma6 * sigma6
    e24s6 = FLOAT(24.0) * epsilon * sigma6
    e48s12 = FLOAT(48.0) * epsilon * sigma12
    e4s6 = FLOAT(4.0) * epsilon * sigma6
    e4s12 = FLOAT(4.0) * epsilon * sigma12

    box = particles.box
    force = particles.force
    position = particles.position
    cutoff2 = params.cutoff_radius * params.cutoff_radius
    length = box[0, 0], box[1, 1], box[2, 2]
    num_atoms = len(position)

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
def compute_ewald_real_forces(
    particles: Particles,
    params: Parameters,
    neighbors: NeighborList,
    energy: Array,
    virial: Array,
) -> None:

    box = particles.box
    force = particles.force
    position = particles.position
    charge = particles.charge
    cutoff2 = params.cutoff_radius * params.cutoff_radius
    length = box[0, 0], box[1, 1], box[2, 2]
    alpha = params.ewald_alpha 
    num_atoms = len(position)

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
                coef_1 = K_e * charge[ni] * charge[nj]
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
    particles: Particles,
    params: Parameters,
    energy: Array,
    virial: Array,
) -> None:

    position = particles.position
    charge = particles.charge
    force = particles.force
    box = particles.box
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
    particles: Particles,
    params: Parameters,
    energy: Array,
) -> None:
    charge = particles.charge
    alpha = params.ewald_alpha 
    energy[0] += -alpha / math.sqrt(math.pi) * np.sum(charge**2)


@njit
def update_neighbors(
    particles: Particles,
    params: Parameters,
    cells: CellList,
    neighbors: NeighborList,
) -> None:

    neighbor_cutoff2 = (params.cutoff_radius + params.skin_radius) ** 2
    box, position = particles.box, particles.position
    length = box[0, 0], box[1, 1], box[2, 2]
    num_atoms = len(position)
    nc = cells.sizes

    # Cell list
    cells.num_atoms_per_cell[:] = 0
    for n in range(num_atoms):
        ic0 = math.floor(position[n, 0] * nc[0] / length[0])
        ic1 = math.floor(position[n, 1] * nc[1] / length[1])
        ic2 = math.floor(position[n, 2] * nc[2] / length[2])
        idx = index3d((ic0, ic1, ic2), nc)
        cells.atom_index[idx, cells.num_atoms_per_cell[idx]] = n
        cells.num_atoms_per_cell[idx] += 1

    # Neighbor list
    neighbors.num_neighbors_per_atom[:] = 0
    for ic0 in range(nc[0]):
        for ic1 in range(nc[1]):
            for ic2 in range(nc[2]):
                idx_i = index3d((ic0, ic1, ic2), nc)

                for i in range(cells.num_atoms_per_cell[idx_i]):
                    ni = cells.atom_index[idx_i, i]

                    for kc0 in range(ic0 - 1, ic0 + 2):
                        if (kc0 == -1) or (kc0 == nc[0]):
                            jc0 = (kc0 + nc[0]) % nc[0]
                        else:
                            jc0 = kc0

                        for kc1 in range(ic1 - 1, ic1 + 2):
                            if (kc1 == -1) or (kc1 == nc[1]):
                                jc1 = (kc1 + nc[1]) % nc[1]
                            else:
                                jc1 = kc1

                            for kc2 in range(ic2 - 1, ic2 + 2):
                                if (kc2 == -1) or (kc2 == nc[2]):
                                    jc2 = (kc2 + nc[2]) % nc[2]
                                else:
                                    jc2 = kc2

                                idx_j = index3d((jc0, jc1, jc2), nc)
                                for j in range(cells.num_atoms_per_cell[idx_j]):
                                    nj = cells.atom_index[idx_j, j]

                                    if ni < nj:
                                        dx = apply_pbc(
                                            position[nj, 0] - position[ni, 0],
                                            length[0],
                                        )
                                        dy = apply_pbc(
                                            position[nj, 1] - position[ni, 1],
                                            length[1],
                                        )
                                        dz = apply_pbc(
                                            position[nj, 2] - position[ni, 2],
                                            length[2],
                                        )
                                        r2 = dx * dx + dy * dy + dz * dz

                                        if r2 < neighbor_cutoff2:
                                            neighbors.neighbor_index[
                                                ni,
                                                neighbors.num_neighbors_per_atom[ni],
                                            ] = nj
                                            neighbors.num_neighbors_per_atom[ni] += 1
                                            neighbors.neighbor_index[
                                                nj,
                                                neighbors.num_neighbors_per_atom[nj],
                                            ] = ni
                                            neighbors.num_neighbors_per_atom[nj] += 1

    # Update old position
    for n in range(num_atoms):
        particles.position_old[n, 0] = position[n, 0]
        particles.position_old[n, 1] = position[n, 1]
        particles.position_old[n, 2] = position[n, 2]


@njit
def get_pressure(particles: Particles, virial: Array) -> float:
    vol = np.linalg.det(particles.box)
    temp = get_temperature(particles)
    natoms = particles.position.shape[0]
    pres = (natoms * K_B * temp - virial[0] / 3.0) / vol
    return pres


@njit
def barostat(particles: Particles, params: Parameters, virial: Array) -> None:

    p = get_pressure(particles, virial)
    p0 = params.pressure
    dt = params.time_step
    tau = params.barostat_coupling_time
    beta = params.compressibility
    pos = particles.position
    box = particles.box

    scale = (1.0 - beta * dt / tau * (p0 - p)) ** (1.0 / 3.0)
    pos[:] *= scale
    box[:] *= scale


@njit
def thermostat(particles, params) -> None:
    t = get_temperature(particles)
    t0 = params.temperature
    dt = params.time_step
    tau = params.thermostat_coupling_time
    v = particles.velocity

    scale = FLOAT(math.sqrt(FLOAT(1.0) + (dt / tau) * (t0 / t - FLOAT(1.0))))
    v[:] *= scale


def generate_velocity(
    mass: Array,
    temperature: float,
    seed: int = 1234,
) -> Array:
    np.random.seed(seed)
    velocity = np.zeros(shape=(len(mass), 3), dtype=FLOAT, order=ORDER)
    for i, m in enumerate(mass):
        sigma = np.sqrt(K_B * temperature / m)
        velocity[i] = np.random.normal(0.0, sigma, size=3)
    vcm = (mass.reshape(-1, 1) * velocity).sum(axis=0) / mass.sum()
    velocity -= vcm
    return velocity


def initialize_cells(
    params: Parameters,
    box: Array,
    num_atoms: int,
) -> CellList:

    cell_length = params.cutoff_radius
    length = box[0, 0], box[1, 1], box[2, 2]

    cell_sizes = (
        int(length[0] / cell_length),
        int(length[1] / cell_length),
        int(length[2] / cell_length),
    )
    for size in cell_sizes:
        assert size > 2, print(f"{cell_sizes=}")

    system_volume = math.prod(length)
    cell_volume = cell_length**3
    cell_max_atoms = 3 * int(num_atoms / system_volume * cell_volume)
    # print("Cell max atoms:", cell_max_atoms)

    cell_atom_index = np.empty(
        shape=(math.prod(cell_sizes), cell_max_atoms),
        dtype=np.int32,
    )
    cell_num_atoms = np.empty(
        math.prod(cell_sizes),
        dtype=np.int32,
    )
    cells = CellList(
        sizes=cell_sizes,
        num_atoms_per_cell=cell_num_atoms,
        atom_index=cell_atom_index,
    )
    return cells


def initialize_neighbors(
    params: Parameters,
    box: Array,
    num_atoms: int,
) -> tuple[CellList, NeighborList]:
    # Cell list
    cells = initialize_cells(params, box, num_atoms)
    # Neighbor list
    neighbor_index = np.empty(
        shape=(num_atoms, params.neighbor_max_atoms),
        dtype=np.int32,
    )
    neighbor_num_atoms = np.empty(
        num_atoms,
        dtype=np.int32,
    )
    neighbors = NeighborList(
        num_neighbors_per_atom=neighbor_num_atoms,
        neighbor_index=neighbor_index,
    )
    # print(f"Cell length:", params.cutoff_radius)
    # print(f"Cell sizes: {cells.sizes}")
    return (
        cells,
        neighbors,
    )


@njit
def index3d(ic: tuple[int, int, int], nc: tuple[int, int, int]) -> int:
    return ic[0] + nc[0] * (ic[1] + nc[1] * ic[2])


@njit
def apply_pbc(
    rij: FLOAT,
    length: FLOAT,
) -> FLOAT:
    if rij >= FLOAT(0.5) * length:
        rij -= length
    elif rij <= FLOAT(-0.5) * length:
        rij += length
    return rij


@njit
def check_if_neighbor_need_update(particles: Particles, params: Parameters):
    num_atoms = len(particles.position)
    displacement_cutoff2 = 0.25 * params.skin_radius * params.skin_radius
    r2 = FLOAT(0.0)
    for n in range(num_atoms):
        for dim in range(3):
            dx = particles.position[n, dim] - particles.position_old[n, dim]
            r2 += dx * dx
        if r2 > displacement_cutoff2:
            return True
    return False


@njit
def verlet_integration_position(
    particles: Particles,
    params: Parameters,
) -> None:
    dt = params.time_step
    box = particles.box
    num_atoms = len(particles.position)
    m, r, v, f = particles.mass, particles.position, particles.velocity, particles.force
    length = box[0, 0], box[1, 1], box[2, 2]

    for n in range(num_atoms):
        for dim in range(3):
            r[n, dim] = np.fmod(
                r[n, dim]
                + dt * (v[n, dim] + FLOAT(0.5) * dt / m[n] * f[n, dim])
                + FLOAT(100.0) * length[dim],
                length[dim],
            )
            v[n, dim] += FLOAT(0.5) * dt / m[n] * f[n, dim]


@njit
def verlet_integration_velocity(
    particles: Particles,
    params: Parameters,
) -> None:
    dt = params.time_step
    num_atoms = len(particles.position)
    m, v, f = particles.mass, particles.velocity, particles.force

    for n in range(num_atoms):
        for dim in range(3):
            v[n, dim] += FLOAT(0.5) * dt / m[n] * f[n, dim]


def save_structure(particles: Particles, file: TextIO) -> None:
    position = particles.position
    natoms = position.shape[0]
    file.write(f"{natoms}\n\n")
    for i in range(natoms):
        file.write(f"{'Ar'}\t{position[i, 0]} {position[i, 1]} {position[i, 2]}\n")
    file.flush()


@njit
def get_kinetic_energy(particles: Particles) -> FLOAT:
    natoms = len(particles.velocity)
    m, v = particles.mass, particles.velocity
    ke = FLOAT(0.0)
    for n in range(natoms):
        ke += m[n] * (v[n, 0] * v[n, 0] + v[n, 1] * v[n, 1] + v[n, 2] * v[n, 2])
    return FLOAT(0.5) * ke


@njit
def get_temperature(particles: Particles) -> FLOAT:
    natoms = len(particles.velocity)
    return FLOAT(2.0) / (FLOAT(3) * natoms * K_B) * get_kinetic_energy(particles)

