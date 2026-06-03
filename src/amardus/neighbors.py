import math
import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray
from numba import njit
from amardus.particles import Particles
from amardus.parameters import Parameters, Array, FLOAT


class CellList(NamedTuple):
    num_atoms_per_cell: NDArray
    atom_index: NDArray
    sizes: tuple[int, int, int]


class NeighborList(NamedTuple):
    num_neighbors_per_atom: NDArray
    neighbor_index: NDArray


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
