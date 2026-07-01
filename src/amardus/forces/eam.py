import math
import numpy as np
from numba import njit
from dataclasses import dataclass
from amardus.parameters import FLOAT
from amardus.system import System
from amardus.neighbors import apply_pbc
from amardus.forces.interface import ForceFieldInterface


@dataclass
class EAMForceField(ForceFieldInterface):
    cutoff_radius: float

    def compute_forces(self, sys: System) -> None:
        sys.atoms.forces[:] = FLOAT(0.0)
        sys.energy[0] = FLOAT(0.0)
        sys.virial[0] = FLOAT(0.0)

        _compute_eam_forces(sys, self.cutoff_radius)


def _compute_eam_forces(
    sys: System,
    cutoff_radius: float,
) -> None:
    # virial = sys.virial
    # forces = sys.atoms.forces
    # atom_types = sys.atoms.atom_types
    box = sys.atoms.box
    energy = sys.energy
    neighbors = sys.neighbors
    pos = sys.atoms.positions
    cutoff2 = cutoff_radius**2
    length = box[0, 0], box[1, 1], box[2, 2]
    num_atoms = len(pos)

    rho = np.zeros(num_atoms, dtype=FLOAT)
    # dF = np.zeros(num_atoms, dtype=FLOAT)

    for ni in range(num_atoms):
        # ai = atom_types[ni] 

        for j in range(neighbors.num_neighbors_per_atom[ni]):
            nj = neighbors.neighbor_index[ni, j]
            # aj = atom_types[nj]

            if ni < nj:
                dx = apply_pbc(pos[nj, 0] - pos[ni, 0], length[0])
                dy = apply_pbc(pos[nj, 1] - pos[ni, 1], length[1])
                dz = apply_pbc(pos[nj, 2] - pos[ni, 2], length[2])
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > cutoff2:
                    continue

                r = math.sqrt(r2)
                dens = density_function(r)
                rho[ni] += dens 
                rho[nj] += dens 
                energy[0] += pair_potential(r) 
                

        
    for ni in range(num_atoms):
        # ai = atom_types[ni] 
        # dF[ni] = -FLOAT(0.5) / math.sqrt(rho[ni]) if rho[ni] > 0 else FLOAT(0.0)
        energy[0] += embedding_function(rho[ni])
        pass


@njit
def pair_potential(r: float) -> float:
    return math.exp(-r)


@njit 
def density_function(r: float) -> float:
    return math.exp(-2.0 * r)


@njit
def embedding_function(rho):
    return -math.sqrt(rho) if rho > 0 else 0.0


