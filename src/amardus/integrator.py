import math
import numpy as np
from numba import njit
from amardus.atoms import Atoms, get_pressure, get_temperature
from amardus.parameters import Parameters, Array, FLOAT


@njit
def verlet_integration_position(
    atoms: Atoms,
    params: Parameters,
) -> None:
    dt = params.time_step
    box = atoms.box
    num_atoms = len(atoms.position)
    m, r, v, f = atoms.mass, atoms.position, atoms.velocity, atoms.force
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
    atoms: Atoms,
    params: Parameters,
) -> None:
    dt = params.time_step
    num_atoms = len(atoms.position)
    m, v, f = atoms.mass, atoms.velocity, atoms.force

    for n in range(num_atoms):
        for dim in range(3):
            v[n, dim] += FLOAT(0.5) * dt / m[n] * f[n, dim]


@njit
def barostat(atoms: Atoms, params: Parameters, virial: Array) -> None:

    p = get_pressure(atoms, virial)
    p0 = params.pressure
    dt = params.time_step
    tau = params.barostat_coupling_time
    beta = params.compressibility
    pos = atoms.position
    box = atoms.box

    scale = (1.0 - beta * dt / tau * (p0 - p)) ** (1.0 / 3.0)
    pos[:] *= scale
    box[:] *= scale


@njit
def thermostat(atoms, params) -> None:
    t = get_temperature(atoms)
    t0 = params.temperature
    dt = params.time_step
    tau = params.thermostat_coupling_time
    v = atoms.velocity

    scale = FLOAT(math.sqrt(FLOAT(1.0) + (dt / tau) * (t0 / t - FLOAT(1.0))))
    v[:] *= scale
