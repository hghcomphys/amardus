import numpy as np
from amardus.parameters import Parameters, FLOAT
from amardus.neighbors import (
    initialize_neighbors,
    update_neighbors,
    check_if_neighbor_need_update,
)
from amardus.atoms import Atoms
from amardus.forces import compute_forces
from amardus.integrator import verlet_integration_velocity, verlet_integration_position
from amardus.io import StateReporter


def simulate(
    params: Parameters,
    atoms: Atoms,
    reporter: StateReporter, 
    steps: int = 1,
) -> None:
    box = atoms.box
    num_atoms = len(atoms.position)

    cells, neighbors = initialize_neighbors(params, box, num_atoms)
    update_neighbors(atoms, params, cells, neighbors)

    energy = np.empty(1, dtype=FLOAT)
    virial = np.empty(1, dtype=FLOAT)
    compute_forces(atoms, params, neighbors, energy, virial)

    num_neighbor_updates = 1
    for step in range(steps):

       reporter.gather(step, energy, virial, atoms, params)
       # save_structure(atoms, file)

       # Next time step (update r, v, and F)
       verlet_integration_position(atoms, params)

       if check_if_neighbor_need_update(atoms, params):
           # cells = initialize_cells(params, atoms.box)
           update_neighbors(atoms, params, cells, neighbors)
           num_neighbor_updates += 1

       compute_forces(atoms, params, neighbors, energy, virial)
       verlet_integration_velocity(atoms, params)

        # thermostat(atoms, params)
        # barostat(atoms, params, virial)
