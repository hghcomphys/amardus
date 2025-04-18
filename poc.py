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
    lattice: Array


@cuda.jit(device=True, inline=True)
def apply_pbc(dx: Float, l: Float) -> Float:
    if dx > 0.5 * l:
        dx -= l
    if dx < -0.5 * l:
        dx += l
    return dx


@cuda.jit(device=True, inline=True)
def distance_squared(ri: Array, rj: Array, lattice: Array) -> Float:
    result = 0.0
    for d in range(ri.shape[0]):
        dx = ri[d] - rj[d]
        dx = apply_pbc(dx, lattice[d][d])
        result += dx * dx
    return result


@cuda.jit(device=True, inline=True)
def distance(ri: Array, rj: Array, lattice: Array) -> Float:
    return math.sqrt(distance_squared(ri, rj, lattice))


@cuda.jit(device=True, inline=True)
def subtract(ri: Array, rj: Array, result: Array) -> None:
    for d in range(ri.shape[0]):
        result[d] = ri[d] - rj[d]


@cuda.jit(device=True, inline=True)
def cosine_ijk(rij: Array, rik: Array) -> Float:
    rij2, rik2 = 0.0, 0.0
    dot_product = 0.0
    for d in range(rij.shape[0]):
        rij2 += rij[d] * rij[d]
        rik2 += rik[d] * rik[d]
        dot_product += rij[d] * rik[d]
    return dot_product / math.sqrt(rij2 * rik2)


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
            dist2 = distance_squared(
                atoms.positions[i],
                atoms.positions[j],
                atoms.lattice,
            )
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
def G2(rij: Float, coefs: G2Coefs) -> Float:
    return math.exp(-coefs.eta * (rij - coefs.r_shift) ** 2)


def get_radial(
    symmetry_function: Callable,
    cutoff: Callable,
) -> Callable:
    @cuda.jit
    def compute(
        atoms: Atoms,
        neighbors: Neighbors,
        between: tuple[Index, Index],
        coefs: NamedTuple,
        r_cut: Float,
        result: Array,
    ) -> None:
        i = cuda.grid(1)
        natoms = atoms.positions.shape[0]
        atom_type_center, atom_type_neighbor = between
        total = 0.0
        if i < natoms and atom_type_center == atoms.types[i]:
            # ---
            for j in range(neighbors.sizes[i]):
                nj = neighbors.indices[i][j]
                if atom_type_neighbor == atoms.types[nj]:
                    rij = distance(
                        atoms.positions[i],
                        atoms.positions[nj],
                        atoms.lattice,
                    )
                    if rij > r_cut:
                        continue
                    # ---
                    total += symmetry_function(rij, coefs) * cutoff(rij, r_cut)
            result[i] = total

    return compute


class G3Coefs(NamedTuple):
    zeta: Float
    lambda0: Float
    eta: Float


@cuda.jit(device=True, inline=True)
def G3(rij: Float, rik: Float, rjk: Float, cost: Float, coefs: G3Coefs) -> Array:
    return (
        math.pow(2.0, 1.0 - coefs.zeta)
        * math.pow(1 + coefs.lambda0 * cost, coefs.zeta)
        * math.exp(-coefs.eta * (rij**2 + rik**2 + rjk**2))
    )


def get_angular(
    symmetry_function: Callable,
    cutoff: Callable,
) -> Callable:
    @cuda.jit
    def compute(
        atoms: Atoms,
        neighbors: Neighbors,
        between: tuple[Index, Index, Index],
        coefs: tuple[Float, ...],
        r_cut: Float,
        result: Array,
    ) -> None:
        i = cuda.grid(1)
        natoms = atoms.positions.shape[0]
        atom_type_center, atom_type_neighbor_1, atom_type_neighbor_2 = between

        rij_vec = cuda.local.array(3, dtype=Float)
        rik_vec = cuda.local.array(3, dtype=Float)
        total = 0.0
        if i < natoms and atom_type_center == atoms.types[i]:
            # ---
            for j in range(neighbors.sizes[i]):
                nj = neighbors.indices[i][j]
                if atom_type_neighbor_1 == atoms.types[nj]:
                    rij = distance(
                        atoms.positions[i],
                        atoms.positions[nj],
                        atoms.lattice,
                    )
                    if rij > r_cut:
                        continue
                    # ---
                    for k in range(neighbors.sizes[i]):
                        nk = neighbors.indices[i][k]
                        if atom_type_neighbor_2 == atoms.types[nk]:
                            rik = distance(
                                atoms.positions[i],
                                atoms.positions[nk],
                                atoms.lattice,
                            )
                            if rik > r_cut:
                                continue
                            # ---
                            rjk = distance(
                                atoms.positions[nj],
                                atoms.positions[nk],
                                atoms.lattice,
                            )
                            if rjk > r_cut:
                                continue
                            # ---
                            subtract(atoms.positions[i], atoms.positions[nj], rij_vec)
                            subtract(atoms.positions[i], atoms.positions[nk], rik_vec)
                            cost = cosine_ijk(rij_vec, rik_vec)
                            # ---
                            total += (
                                symmetry_function(rij, rik, rjk, cost, coefs)
                                * cutoff(rij, r_cut)
                                * cutoff(rik, r_cut)
                                * cutoff(rjk, r_cut)
                            )
            result[i] = total

    return compute


if __name__ == "__main__":

    N = 1_000
    np.random.seed(0)
    positions = (np.random.rand(N, 3) * 10).astype(Float)
    lattice = np.eye(3, dtype=Float) * 10
    types = np.ones(N, dtype=Index)
    types[::2] = 2
    atoms = Atoms(
        positions=cuda.to_device(positions),
        types=cuda.to_device(types),
        lattice=cuda.to_device(lattice),
    )
    neighbors = Neighbors(
        indices=cuda.device_array((N, N), dtype=Index),
        sizes=cuda.device_array(N, dtype=Index),
    )
    descriptors = cuda.device_array(N, dtype=Float)

    threads_per_block = 32
    blocks = math.ceil(N / threads_per_block)
    print(f"CUDA kernel: [{threads_per_block=}, {blocks=}]")

    r_cut = Float(1.5)
    coefs_g2 = G2Coefs(Float(0.1), Float(0.0))
    coefs_g3 = G3Coefs(Float(0.1), Float(0.2), Float(0.3))

    compute_g2 = get_radial(G2, cutoff_tanh)
    compute_g3 = get_angular(G3, cutoff_cosine)

    find_neighbors[blocks, threads_per_block](atoms, r_cut, neighbors)
    for _ in range(1, 1_00):
        compute_g2[blocks, threads_per_block](
            atoms,
            neighbors,
            (1, 2),
            coefs_g2,
            r_cut * 5_000 / _,
            descriptors,
        )
        compute_g3[blocks, threads_per_block](
            atoms,
            neighbors,
            (2, 1, 2),
            coefs_g3,
            r_cut * 5_000 / _,
            descriptors,
        )
    cuda.synchronize()

    print(neighbors.sizes.copy_to_host()[:5])
    print(descriptors.copy_to_host()[:5])
