import math
import numpy as np
from amardus.parameters import Parameters, FLOAT
from amardus.particles import (
    Particles,
    get_kinetic_energy,
    get_temperature,
    get_pressure,
)
from amardus.neighbors import (
    initialize_neighbors,
    update_neighbors,
    check_if_neighbor_need_update,
)
from amardus.forces import compute_forces
from amardus.integrator import verlet_integration_velocity, verlet_integration_position


def simulate(
    params: Parameters,
    particles: Particles,
    steps: int = 1,
    log_frequency: int = 100,
    filename: str = "out.xyz",
) -> None:
    box = particles.box
    num_atoms = len(particles.position)
    unit = params.unit

    cells, neighbors = initialize_neighbors(params, box, num_atoms)
    update_neighbors(particles, params, cells, neighbors)

    energy = np.empty(1, dtype=FLOAT)
    virial = np.empty(1, dtype=FLOAT)
    compute_forces(particles, params, neighbors, energy, virial)

    with open(filename, "w") as file:
        # Simulate
        print(
            f"{'Step':8}"
            f" {'Temp':10}"
            f" {'Pres':10}"
            f" {'PotE':10}"
            f" {'KinE':10}"
            f" {'TotE':10}"
            f" {'UpdateNb':10}"
            f" {'AvgNb':10}"
            f" {'AvgCell':10}"
            f" {'L':10}"
            f" {'Dens':10}"
        )
        num_neighbor_updates = 1
        for step in range(steps):
            if step % log_frequency == 0:

                ke = get_kinetic_energy(particles)
                pe = energy[0]
                temp = get_temperature(particles, unit.K_B)
                pres = get_pressure(particles, virial, unit.K_B)
                vol = np.linalg.det(particles.box)
                dens = np.sum(particles.mass) / vol
                print(
                    f"{step:<8}"
                    f" {temp / unit.kelvin:<10.4f}"
                    f" {pres / unit.bar:<10.4f}"
                    f" {pe / unit.electronvolt:<10.4f}"
                    f" {ke / unit.electronvolt:<10.4f}"
                    f" {(ke + pe) / unit.electronvolt:<10.4f}"
                    f" {num_neighbor_updates:<10}"
                    f" {neighbors.num_neighbors_per_atom.mean():<10.2f}"
                    f" {cells.num_atoms_per_cell.mean():<10.2f}"
                    f" {math.pow(vol, 1/3):<10.4f}"
                    f" {dens / unit.grcm3:<10.6f}"
                )
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
