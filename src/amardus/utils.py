import numpy as np
from numpy import random
from amardus.parameters import Array, FLOAT, ORDER, get_default_unit

KB = get_default_unit().boltzman

def generate_velocity(
    mass: Array,
    temperature: float,
    rng: random.Generator = random.default_rng(1234),
) -> Array:
    velocity = np.zeros(shape=(len(mass), 3), dtype=FLOAT, order=ORDER)
    for i, m in enumerate(mass):
        sigma = np.sqrt(KB * temperature / m)
        velocity[i] = rng.normal(0.0, sigma, size=3)
    vcm = (mass.reshape(-1, 1) * velocity).sum(axis=0) / mass.sum()
    velocity -= vcm
    return velocity


