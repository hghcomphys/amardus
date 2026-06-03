import numpy as np
from numba import njit
from typing import NamedTuple
from amardus.parameters import Array, FLOAT, ORDER


class Particles(NamedTuple):
    mass: Array
    position: Array
    velocity: Array
    force: Array
    charge: Array
    box: Array
    position_old: Array


@njit
def get_pressure(particles: Particles, virial: Array, K_B: float) -> float:
    vol = np.linalg.det(particles.box)
    temp = get_temperature(particles, K_B)
    natoms = particles.position.shape[0]
    pres = (natoms * K_B * temp - virial[0] / FLOAT(3.0)) / vol
    return pres


def generate_velocity(
    mass: Array,
    temperature: float,
    K_B: float,
    seed: int = 1234,
) -> Array:
    np.random.seed(seed)
    velocity = np.zeros(shape=(len(mass), 3), dtype=FLOAT, order=ORDER)
    for i, m in enumerate(mass):
        sigma = np.sqrt(K_B * temperature / m)
        velocity[i] = np.random.normal(0.0, sigma, size=3)
    vcm = (mass.reshape(-1, 1) * velocity).sum(axis=0) / mass.sum()
    velocity -= vcm
    return velocity


@njit
def get_kinetic_energy(particles: Particles) -> FLOAT:
    natoms = len(particles.velocity)
    m, v = particles.mass, particles.velocity
    ke = FLOAT(0.0)
    for n in range(natoms):
        ke += m[n] * (v[n, 0] * v[n, 0] + v[n, 1] * v[n, 1] + v[n, 2] * v[n, 2])
    return FLOAT(0.5) * ke


@njit
def get_temperature(particles: Particles, K_B: float) -> FLOAT:
    natoms = len(particles.velocity)
    return FLOAT(2.0) / (FLOAT(3) * natoms * K_B) * get_kinetic_energy(particles)
