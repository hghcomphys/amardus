import numpy as np
from amardus.parameters import Parameters, FLOAT
from amardus.neighbors import (
    initialize_neighbors,
    update_neighbors,
    check_if_neighbor_need_update,
)
from amardus.particles import Particles
from amardus.forces import compute_forces
from amardus.integrator import verlet_integration_velocity, verlet_integration_position
from amardus.io import StateReporter


def simulate(
    params: Parameters,
    particles: Particles,
    reporter: StateReporter, 
    steps: int = 1,
) -> None:
    box = particles.box
    num_atoms = len(particles.position)

    cells, neighbors = initialize_neighbors(params, box, num_atoms)
    update_neighbors(particles, params, cells, neighbors)

    energy = np.empty(1, dtype=FLOAT)
    virial = np.empty(1, dtype=FLOAT)
    compute_forces(particles, params, neighbors, energy, virial)

    num_neighbor_updates = 1
    for step in range(steps):

       reporter.gather(step, energy, virial, particles, params)
       # save_structure(particles, file)

       # Next time step (update r, v, and F)
       verlet_integration_position(particles, params)

       if check_if_neighbor_need_update(particles, params):
           # cells = initialize_cells(params, particles.box)
           update_neighbors(particles, params, cells, neighbors)
           num_neighbor_updates += 1

       compute_forces(particles, params, neighbors, energy, virial)
       verlet_integration_velocity(particles, params)

        # thermostat(particles, params)
        # barostat(particles, params, virial)
