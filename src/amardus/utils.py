import numpy as np
from amardus.parameters import Array, FLOAT, ORDER


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


