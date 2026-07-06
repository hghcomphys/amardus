from __future__ import annotations
import math
import numpy as np
from typing import NamedTuple, Protocol, Any
from numpy.typing import NDArray
from numba import njit
from amardus.atoms import Atoms
from amardus.parameters import Array, FLOAT


class CellList(NamedTuple):
    num_atoms_per_cell: NDArray
    atom_index: NDArray
    sizes: tuple[int, int, int]

    @classmethod
    def create(
        cls,
        box: Array,
        num_atoms: int,
        cell_length: float,
        max_atoms_factor: float = 3.0,
    ) -> CellList:

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
        cell_max_atoms = int(max_atoms_factor * num_atoms / system_volume * cell_volume)

        cell_atom_index = np.empty(
            shape=(math.prod(cell_sizes), cell_max_atoms),
            dtype=np.int32,
        )
        cell_num_atoms = np.empty(
            math.prod(cell_sizes),
            dtype=np.int32,
        )
        return cls(
            sizes=cell_sizes,
            num_atoms_per_cell=cell_num_atoms,
            atom_index=cell_atom_index,
        )


class NeighborList(NamedTuple):
    num_neighbors_per_atom: NDArray
    neighbor_index: NDArray

    @classmethod
    def create(
        cls,
        num_atoms: int,
        neighbor_max_atoms: int
    ) -> NeighborList:

        neighbor_index = np.empty(
            shape=(num_atoms, neighbor_max_atoms),
            dtype=np.int32,
        )
        neighbor_num_atoms = np.empty(
            num_atoms,
            dtype=np.int32,
        )
        return cls(
            num_neighbors_per_atom=neighbor_num_atoms,
            neighbor_index=neighbor_index,
        )


@njit(inline="always")
def index3d(ic: tuple[int, int, int], nc: tuple[int, int, int]) -> int:
    return ic[0] + nc[0] * (ic[1] + nc[1] * ic[2])


@njit(inline="always")
def apply_pbc(
    rij: FLOAT,
    length: FLOAT,
) -> FLOAT:
    if rij >= FLOAT(0.5) * length:
        rij -= length
    elif rij <= FLOAT(-0.5) * length:
        rij += length
    return rij

# @njit(inline="always")
# def apply_pbc(
#     rij: FLOAT,
#     length: FLOAT,
# ) -> FLOAT:
#     return rij - length * np.round(rij / length)


@njit(inline="always")
def calculate_distance2(
        position_i: Array, 
        position_j: Array,
        box: Array,
) -> tuple[FLOAT, Any]:
    length = box[0, 0], box[1, 1], box[2, 2]
    dx = apply_pbc(position_j[0] - position_i[0], length[0])
    dy = apply_pbc(position_j[1] - position_i[1], length[1])
    dz = apply_pbc(position_j[2] - position_i[2], length[2])
    r2 = dx * dx + dy * dy + dz * dz
    aux = dx, dy, dz
    return r2, aux


class SystemInterface(Protocol):

    @property
    def atoms(self) -> Atoms: ...

    @property
    def cells(self) -> CellList: ...
    
    @property
    def neighbors(self) -> NeighborList: ...

    @property
    def params(self) -> NamedTuple: ...


@njit
def update_neighbors(sys: SystemInterface) -> None:
    atoms = sys.atoms 
    cells, neighbors = sys.cells, sys.neighbors
    neighbor_cutoff2 = (sys.params.neighbor_cutoff + sys.params.neighbor_skin) ** 2
    box, pos = atoms.box, atoms.positions
    length = box[0, 0], box[1, 1], box[2, 2]
    num_atoms = pos.shape[0] 
    nc = cells.sizes

    # Cell list
    cells.num_atoms_per_cell[:] = 0
    for n in range(num_atoms):
        ic0 = math.floor(pos[n, 0] * nc[0] / length[0])
        ic1 = math.floor(pos[n, 1] * nc[1] / length[1])
        ic2 = math.floor(pos[n, 2] * nc[2] / length[2])
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
                                            pos[nj, 0] - pos[ni, 0],
                                            length[0],
                                        )
                                        dy = apply_pbc(
                                            pos[nj, 1] - pos[ni, 1],
                                            length[1],
                                        )
                                        dz = apply_pbc(
                                            pos[nj, 2] - pos[ni, 2],
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
        atoms.positions_old[n, 0] = pos[n, 0]
        atoms.positions_old[n, 1] = pos[n, 1]
        atoms.positions_old[n, 2] = pos[n, 2]


@njit
def check_if_neighbor_need_update(sys: SystemInterface) -> bool:
    atoms= sys.atoms
    num_atoms = atoms.positions.shape[0] 
    displacement_cutoff2 = 0.25 * sys.params.neighbor_skin **2
    r2 = FLOAT(0.0)
    for n in range(num_atoms):
        for dim in range(3):
            dx = atoms.positions[n, dim] - atoms.positions_old[n, dim]
            r2 += dx * dx
        if r2 > displacement_cutoff2:
            return True
    return False
