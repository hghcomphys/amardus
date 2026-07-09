import math
from typing import NamedTuple
import numpy as np
from numba import njit
from dataclasses import dataclass, field
from amardus.parameters import FLOAT
from amardus.system import System
from amardus.parameters import Array
from amardus.neighbors import calculate_distance2
from amardus.potentials.interface import ForceFieldInterface

_EMPTY_ARRAY = lambda: np.empty([])
_ALMOST_ZERO = 1e-14


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
class RhoArrays:
    rho: Array = field(default_factory=_EMPTY_ARRAY)
    drho: Array = field(default_factory=_EMPTY_ARRAY)


@dataclass
class EAMPotential(ForceFieldInterface):
    """Single atom type EAM implementation.

    Extended Finnis–Sinclair potential for bcc and fcc metals and alloys.
    DOI: 10.1088/0953-8984/18/19/008
    """

    params: EAMParams = field(repr=False)
    cutoff_radius: float
    scratch: RhoArrays = field(default_factory=RhoArrays, repr=False)
    allocate_scratch: bool = field(default=True, repr=False)

    def compute_forces(self, sys: System) -> None:
        assert (
            np.sum(sys.atoms.atom_types) == 0
        ), f"{self.__class__.__name__}: only single atom type is supported!"

        if self.allocate_scratch:
            num_atoms = len(sys.atoms)
            self.scratch = RhoArrays(
                rho=np.empty(num_atoms, dtype=FLOAT),
                drho=np.empty((num_atoms, 3), dtype=FLOAT),
            )
            self.allocate_scratch = False

        sys.atoms.forces[:] = FLOAT(0.0)
        sys.energy[0] = FLOAT(0.0)
        sys.virial[0] = FLOAT(0.0)

        _compute_eam_forces(
            sys,
            self.params,
            self.cutoff_radius,
            self.scratch.rho,
            self.scratch.drho,
        )


@njit
def _compute_eam_forces(
    sys: System,
    params: EAMParams,
    cutoff_radius: float,
    rho: Array,
    drho: Array,
) -> None:
    virial = sys.virial
    forces = sys.atoms.forces
    box = sys.atoms.box
    energy = sys.energy
    neighbors = sys.neighbors
    pos = sys.atoms.positions
    cutoff2 = cutoff_radius**2
    num_atoms = len(pos)

    rho[:] = 0.0
    drho[:] = 0.0

    for ni in range(num_atoms):
        for j in range(neighbors.num_neighbors_per_atom[ni]):
            nj = neighbors.neighbor_index[ni, j]

            if ni < nj:
                r2, (dx, dy, dz) = calculate_distance2(pos[ni], pos[nj], box)
                if r2 > cutoff2:
                    continue

                r = math.sqrt(r2)
                r_inv = 1.0 / r
                drdx = dx * r_inv
                drdy = dy * r_inv
                drdz = dz * r_inv

                # Calculate electron density
                if r < params.d:
                    phi, dphidr = _electronic_density_function(r, params)

                    rho[ni] += phi
                    rho[nj] += phi

                    drho_x = dphidr * drdx
                    drho_y = dphidr * drdy
                    drho_z = dphidr * drdz

                    drho[ni, 0] -= drho_x
                    drho[ni, 1] -= drho_y
                    drho[ni, 2] -= drho_z
                    drho[nj, 0] += drho_x
                    drho[nj, 1] += drho_y
                    drho[nj, 2] += drho_z

                # Calculate pair potential
                if r < params.c:
                    pair_pot, dVdr = _pair_potential(r, params)

                    fx = dVdr * drdx
                    fy = dVdr * drdy
                    fz = dVdr * drdz

                    forces[ni, 0] += fx
                    forces[ni, 1] += fy
                    forces[ni, 2] += fz
                    forces[nj, 0] -= fx
                    forces[nj, 1] -= fy
                    forces[nj, 2] -= fz
                    energy[0] += pair_pot

    for ni in range(num_atoms):
        if rho[ni] < _ALMOST_ZERO:
            continue

        eng, dFdrho = _embedding_function(rho[ni])

        forces[ni, 0] += dFdrho * drho[ni, 0]
        forces[ni, 1] += dFdrho * drho[ni, 1]
        forces[ni, 2] += dFdrho * drho[ni, 2]
        energy[0] -= eng

    virial[0] -= np.sum(forces * pos)


@njit(inline="always")
def _pair_potential(r: float, ps: EAMParams) -> tuple[float, float]:
    dr = r - ps.c
    r2 = r * r
    r3 = r2 * r
    term1 = ps.c0 + ps.c1 * r + ps.c2 * r2 + ps.c3 * r3 + ps.c4 * r2 * r2
    term2 = ps.c1 + 2.0 * ps.c2 * r + 3.0 * ps.c3 * r2 + 4.0 * ps.c4 * r3
    V = dr * dr * term1
    dVdr = dr * (2.0 * term1 + dr * term2)
    return V, dVdr


@njit(inline="always")
def _electronic_density_function(r: float, ps: EAMParams) -> tuple[float, float]:
    dr = r - ps.d
    dr2 = dr * dr
    B2 = ps.B * ps.B
    A2 = ps.A * ps.A
    F = A2 * dr2 * (1.0 + B2 * dr2)
    dF = A2 * dr * (2.0 + 4.0 * B2 * dr2)
    return F, dF


@njit(inline="always")
def _embedding_function(rho) -> tuple[float, float]:
    term1 = math.sqrt(rho)
    F = term1
    dF = 0.5 / term1
    return F, dF
