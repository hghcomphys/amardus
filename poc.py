import math
from enum import IntEnum
from typing import NamedTuple

import numpy as np
from numba import cuda
from numpy.typing import NDArray

FLOAT = np.double
INDEX = np.int32
Array = NDArray[FLOAT]
IndexArray = NDArray[INDEX]


class Atoms(NamedTuple):
    positions: Array
    types: IndexArray


@cuda.jit(device=True, inline=True)
def distance_squared(xi: Array, xj: Array) -> FLOAT:
    distance = 0.0
    for d in range(xi.shape[0]):
        dx = xi[d] - xj[d]
        distance += dx * dx
    return distance


@cuda.jit(device=True, inline=True)
def distance(xi: Array, xj: Array) -> FLOAT:
    return math.sqrt(distance_squared(xi, xj))


class Neighbors(NamedTuple):
    indices: Array
    sizes: Array


@cuda.jit
def find_neighbors(
    atoms: Atoms,
    r_cut: FLOAT,
    out: Neighbors,
) -> None:
    natoms = atoms.positions.shape[0]
    r_cut2 = r_cut * r_cut
    i = cuda.grid(1)
    if i < natoms:
        neighbors_count = 0
        for j in range(natoms):
            if i == j:
                continue
            dist2 = distance_squared(atoms.positions[i], atoms.positions[j])
            if dist2 < r_cut2:
                out.indices[i][neighbors_count] = j
                neighbors_count += 1
        out.sizes[i] = neighbors_count


@cuda.jit(device=True, inline=True)
def cutoff_tanh(r: FLOAT, r_cut: FLOAT) -> FLOAT:
    return math.tanh(1.0 - r / r_cut) ** 3 if r < r_cut else 0.0


@cuda.jit(device=True, inline=True)
def cutoff_cosine(r: Array, r_cut: FLOAT) -> FLOAT:
    return 0.5 * (math.cos(math.pi * r / r_cut) + 1.0) if r < r_cut else 0.0


class CutoffType(IntEnum):
    TANH = 1
    COSINE = 2


@cuda.jit(device=True)
def cutoff(
    r: Array,
    r_cut: FLOAT,
    cutoff_type: CutoffType,
) -> FLOAT:
    if cutoff_type == CutoffType.TANH:
        return cutoff_tanh(r, r_cut)
    elif cutoff_type == CutoffType.COSINE:
        return cutoff_cosine(r, r_cut)
    else:
        return 1.0


class G2Coefs(NamedTuple):
    eta: FLOAT
    r_shift: FLOAT


@cuda.jit(device=True)
def G2(
    rij: FLOAT,
    coefs: G2Coefs,
    r_cut: FLOAT,
    cutoff_type: CutoffType,
) -> FLOAT:
    return math.exp(-coefs.eta * (rij - coefs.r_shift) ** 2) * cutoff(
        rij, r_cut, cutoff_type
    )


@cuda.jit
def compute_g2(
    atoms: Atoms,
    neighbors: Neighbors,
    between: tuple[INDEX, INDEX],
    coefs: G2Coefs,
    r_cut: FLOAT,
    cutoff_type: CutoffType,
    out: Array,
) -> None:
    i = cuda.grid(1)
    natoms = atoms.positions.shape[0]
    atom_type_center, atom_type_neighbor = between
    total = 0.0
    if i < natoms and atom_type_center == atoms.types[i]:
        for j in range(neighbors.sizes[i]):
            nj = neighbors.indices[i][j]
            if atom_type_neighbor == atoms.types[nj]:
                rij = distance(atoms.positions[i], atoms.positions[nj])
                total += G2(rij, coefs, r_cut, cutoff_type)
        out[i] = total


if __name__ == "__main__":

    # Random 3D atom positions
    N = 10_000
    np.random.seed(0)
    positions = (np.random.rand(N, 3) * 10).astype(FLOAT)
    types = np.ones(N, dtype=INDEX)
    types[::2] = 2
    atoms = Atoms(positions=cuda.to_device(positions), types=cuda.to_device(types))
    neighbors = Neighbors(
        indices=cuda.device_array((N, N), dtype=INDEX),
        sizes=cuda.device_array(N, dtype=INDEX),
    )
    descriptors = cuda.device_array(N, dtype=FLOAT)

    threads_per_block = 32
    blocks = math.ceil(N / threads_per_block)
    print(f"CUDA kernel: [{threads_per_block=}, {blocks=}]")

    r_cut = FLOAT(1.5)
    coefs = G2Coefs(FLOAT(0.1), FLOAT(0.0))
    find_neighbors[blocks, threads_per_block](atoms, r_cut, neighbors)
    for _ in range(1, 10_000):
        compute_g2[blocks, threads_per_block](
            atoms,
            neighbors,
            (1, 1),
            coefs,
            r_cut * 5_000 / _,
            CutoffType.TANH,
            descriptors,
        )
    cuda.synchronize()

    print(neighbors.sizes.copy_to_host()[:5])
    print(descriptors.copy_to_host()[:5])
