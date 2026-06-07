import math
import numpy as np
from numba import njit
from amardus.system import MDSystem, get_temperature, get_pressure
from amardus.parameters import FLOAT


@njit
def verlet_integration_position(sys: MDSystem) -> None:
    atoms, params = sys.atoms, sys.params
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
def verlet_integration_velocity(sys: MDSystem) -> None:
    atoms, params = sys.atoms, sys.params
    dt = params.time_step
    m, v, f = atoms.mass.reshape(-1, 1), atoms.velocity, atoms.force
    v += FLOAT(0.5) * dt / m * f

@njit
def thermostat(sys: MDSystem) -> None:
    atoms, params = sys.atoms, sys.params
    t0 = params.temperature
    dt = params.time_step
    tau = params.thermostat_coupling_time
    v = atoms.velocity

    t = get_temperature(sys)
    scale = FLOAT(math.sqrt(FLOAT(1.0) + (dt / tau) * (t0 / t - FLOAT(1.0))))
    v *= scale

@njit
def barostat(sys: MDSystem) -> None:
    atoms, params = sys.atoms, sys.params
    p0 = params.pressure
    dt = params.time_step
    tau = params.barostat_coupling_time
    beta = params.compressibility
    pos = atoms.position
    box = atoms.box

    p = get_pressure(sys)
    scale = (1.0 - beta * dt / tau * (p0 - p)) ** (1.0 / 3.0)
    pos *= scale
    box *= scale



