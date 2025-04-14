from numba import cuda
from typing import NamedTuple
from numpy.typing import NDArray as Array
import numpy as np
import math


class Atoms(NamedTuple):
    positions: Array


@cuda.jit(device=True, inline=True) 
def distance_squared(xi, xj):
    distance = 0.0
    for d in range(xi.shape[0]):
        dx = xi[d] - xj[d]
        distance += dx * dx
    return distance


@cuda.jit(device=True, inline=True) 
def distance(xi, xj):
    return math.sqrt(
        distance_squared(xi, xj)
    )


class Neighbors(NamedTuple):
    indices: Array
    sizes: Array


@cuda.jit
def find_neighbors(atoms, r_cut, neighbors):
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
                neighbors.indices[i][neighbors_count] = j
                neighbors_count += 1
        neighbors.sizes[i] = neighbors_count



@cuda.jit(device=True, inline=True) 
def cutoff(r, r_cut):
    return math.tanh(1.0 - r / r_cut) ** 3 if r < r_cut else 0.0


class G2Coefs(NamedTuple):
    eta: np.double
    r_shift: np.double


@cuda.jit(device=True) 
def G2(rij, r_cut, coefs):
    return math.exp(-coefs.eta * (rij - coefs.r_shift) ** 2) * cutoff(rij, r_cut)


@cuda.jit
def compute_g2(atoms, r_cut, neighbors, descriptors):

    natoms = atoms.positions.shape[0] 
    i = cuda.grid(1)

    coefs = G2Coefs(np.double(0.1), np.double(0.0))
    result = 0.0
    if i < natoms:
        for j in range(neighbors.sizes[i]):
            nj = neighbors.indices[i][j]
            rij = distance(atoms.positions[i], atoms.positions[nj])
            result += G2(rij, r_cut, coefs)
        descriptors[i] = result    


if __name__ == "__main__":
 
    # Random 3D atom positions
    N = 10_000
    np.random.seed(0)
    positions = (np.random.rand(N, 3) * 10).astype(np.double)
    atoms = Atoms(
            positions= cuda.to_device(positions),
    )
    neighbors = Neighbors(
        indices=cuda.device_array((N, N), dtype=np.int32),
        sizes=cuda.device_array(N, dtype=np.int32),
    )
    descriptors = cuda.device_array(N, dtype=np.double)

    threads = 32
    blocks = math.ceil(N / threads)
    print("Threads: ", threads)
    print("Blocks: ", blocks)

    r_cut = np.double(1.5)
    find_neighbors[blocks, threads](atoms, r_cut, neighbors)
    compute_g2[blocks, threads](atoms, r_cut, neighbors, descriptors)
    cuda.synchronize()

    print(neighbors.sizes.copy_to_host()[:5])
    print(descriptors.copy_to_host()[:5])


