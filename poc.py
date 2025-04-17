import math
from typing import Callable, NamedTuple

import numpy as np
from numba import cuda
from numpy.typing import NDArray

Float = np.float64
Index = np.int32
Array = NDArray[Float]
IndexArray = NDArray[Index]


class Atoms(NamedTuple):
    positions: Array
    types: IndexArray


@cuda.jit(device=True, inline=True)
def distance_squared(ri: Array, rj: Array) -> Float:
    distance = 0.0
    for d in range(ri.shape[0]):
        dx = ri[d] - rj[d]
        distance += dx * dx
    return distance


@cuda.jit(device=True, inline=True)
def distance(ri: Array, rj: Array) -> Float:
    return math.sqrt(distance_squared(ri, rj))


class Neighbors(NamedTuple):
    indices: Array
    sizes: Array


@cuda.jit
def find_neighbors(
    atoms: Atoms,
    r_cut: Float,
    result: Neighbors,
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
                result.indices[i][neighbors_count] = j
                neighbors_count += 1
        result.sizes[i] = neighbors_count


@cuda.jit(device=True, inline=True)
def cutoff_tanh(r: Float, r_cut: Float) -> Float:
    return math.tanh(1.0 - r / r_cut) ** 3


@cuda.jit(device=True, inline=True)
def cutoff_cosine(r: Float, r_cut: Float) -> Float:
    return 0.5 * (math.cos(math.pi * r / r_cut) + 1.0)


class G2Coefs(NamedTuple):
    eta: Float
    r_shift: Float


@cuda.jit(device=True, inline=True)
def symmetry_function_g2(rij: Float, coefs: G2Coefs) -> Float:
    return math.exp(-coefs.eta * (rij - coefs.r_shift) ** 2)


def build_radial(
    symmetry_function: Callable,
    cutoff: Callable,
) -> Callable:
    @cuda.jit
    def compute(
        atoms: Atoms,
        neighbors: Neighbors,
        between: tuple[Index, Index],
        coefs: tuple[Float, ...],
        r_cut: Float,
        result: Array,
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
                    if rij < r_cut:
                        total += symmetry_function(rij, coefs) * cutoff(rij, r_cut)
            result[i] = total
    return compute

# def build_angular(
#     symmetry_function: Callable,
#     cutoff: Callable,
# ) -> Callable:
#     @cuda.jit
#     def compute(
#         atoms: Atoms,
#         neighbors: Neighbors,
#         between: tuple[Index, Index, Index],
#         coefs: tuple[Float, ...],
#         r_cut: Float,
#         result: Array,
#     ) -> None:
#         i = cuda.grid(1)
#         natoms = atoms.positions.shape[0]
#         atom_type_center, atom_type_neighbor_1, atom_type_neighbor_2 = between
#         total = 0.0
#         if i < natoms and atom_type_center == atoms.types[i]:
#
#             for j in range(neighbors.sizes[i]):
#                 nj = neighbors.indices[i][j]
#
#                 if atom_type_neighbor_1 == atoms.types[nj]:
#                     rij = distance(atoms.positions[i], atoms.positions[nj])
#
#                     for k in range(neighbors.sizes[i]):
#                         nk = neighbors.indices[i][k]
#                         
#                         if atom_type_neighbor_2 == atoms.types[nk]:
#                         if rij < r_cut:
#                             total += symmetry_function(rij, coefs) * cutoff(rij, r_cut)
#             result[i] = total
#     return compute
#

if __name__ == "__main__":

    N = 1_000
    np.random.seed(0)
    positions = (np.random.rand(N, 3) * 10).astype(Float)
    types = np.ones(N, dtype=Index)
    types[::2] = 2
    atoms = Atoms(positions=cuda.to_device(positions), types=cuda.to_device(types))
    neighbors = Neighbors(
        indices=cuda.device_array((N, N), dtype=Index),
        sizes=cuda.device_array(N, dtype=Index),
    )
    descriptors = cuda.device_array(N, dtype=Float)

    threads_per_block = 32
    blocks = math.ceil(N / threads_per_block)
    print(f"CUDA kernel: [{threads_per_block=}, {blocks=}]")

    r_cut = Float(1.5)
    coefs = G2Coefs(Float(0.1), Float(0.0))
    compute_g2 = [build_radial(symmetry_function_g2, cutoff_tanh) for _ in range(100)]

    find_neighbors[blocks, threads_per_block](atoms, r_cut, neighbors)
    for _ in range(1, 1_000):
        compute_g2[_ % 10][blocks, threads_per_block](
            atoms,
            neighbors,
            (1, 1),
            coefs,
            r_cut * 5_000 / _,
            descriptors,
        )
    cuda.synchronize()

    print(neighbors.sizes.copy_to_host()[:5])
    print(descriptors.copy_to_host()[:5])
