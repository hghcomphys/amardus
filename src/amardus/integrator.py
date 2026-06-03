import math
import numpy as np
from numba import njit
from amardus.particles import Particles, get_pressure, get_temperature
from amardus.parameters import Parameters, Array, FLOAT


@njit
def verlet_integration_position(
    particles: Particles,
    params: Parameters,
) -> None:
    dt = params.time_step
    box = particles.box
    num_atoms = len(particles.position)
    m, r, v, f = particles.mass, particles.position, particles.velocity, particles.force
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
    particles: Particles,
    params: Parameters,
) -> None:
    dt = params.time_step
    num_atoms = len(particles.position)
    m, v, f = particles.mass, particles.velocity, particles.force

    for n in range(num_atoms):
        for dim in range(3):
            v[n, dim] += FLOAT(0.5) * dt / m[n] * f[n, dim]


@njit
def barostat(particles: Particles, params: Parameters, virial: Array) -> None:

    p = get_pressure(particles, virial)
    p0 = params.pressure
    dt = params.time_step
    tau = params.barostat_coupling_time
    beta = params.compressibility
    pos = particles.position
    box = particles.box

    scale = (1.0 - beta * dt / tau * (p0 - p)) ** (1.0 / 3.0)
    pos[:] *= scale
    box[:] *= scale


@njit
def thermostat(particles, params) -> None:
    t = get_temperature(particles)
    t0 = params.temperature
    dt = params.time_step
    tau = params.thermostat_coupling_time
    v = particles.velocity

    scale = FLOAT(math.sqrt(FLOAT(1.0) + (dt / tau) * (t0 / t - FLOAT(1.0))))
    v[:] *= scale
