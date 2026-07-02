import math
from typing import NamedTuple
import numpy as np
from numba import njit
from dataclasses import dataclass, field
from amardus.parameters import FLOAT
from amardus.system import System
from amardus.neighbors import apply_pbc
from amardus.potentials.interface import ForceFieldInterface


class EAMParams(NamedTuple):
    A: float
    B: float
    c: float
    c0: float
    c1: float
    c2: float
    c3: float
    c4: float
    d: float


@dataclass
class EAMPotential(ForceFieldInterface):
    params: EAMParams = field(repr=False)
    cutoff_radius: float

    def compute_forces(self, sys: System) -> None:
        sys.atoms.forces[:] = FLOAT(0.0)
        sys.energy[0] = FLOAT(0.0)
        sys.virial[0] = FLOAT(0.0)

        _compute_eam_forces(sys, self.params, self.cutoff_radius)


def _compute_eam_forces(
    sys: System,
    params: EAMParams,
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
                dens = params.A**2 * electron_density(r, params)
                rho[ni] += dens 
                rho[nj] += dens 
                energy[0] += pair_potential(r, params) 
                

        
    for ni in range(num_atoms):
        # ai = atom_types[ni] 
        # dF[ni] = -FLOAT(0.5) / math.sqrt(rho[ni]) if rho[ni] > 0 else FLOAT(0.0)
        energy[0] += embedding_function(rho[ni])


@njit(inline="always")
def electron_density(r: float, ps: EAMParams) -> float:
    if r > ps.d: 
        return 0.0
    dr = r - ps.d
    dr2 = dr * dr
    return dr2 * (1.0 + ps.B * dr2 / ps.d)


@njit(inline="always") 
def pair_potential(r: float, ps: EAMParams) -> float:
    if r > ps.c: 
        return 0.0
    dr = r - ps.c
    r2 = r * r
    return dr * dr * (ps.c0 + ps.c1 * r + ps.c2 * r2 + ps.c3 * r2 * r + ps.c4 * r2 * r2) 


@njit(inline="always")
def embedding_function(rho):
    return math.sqrt(rho)


