from amardus.neighbors import (
    update_neighbors,
    check_if_neighbor_need_update,
)
from amardus.system import MDSystem
from amardus.forces import compute_forces
from amardus.io import ReporterInteface
from amardus.integrator import (
        verlet_integration_velocity, 
        verlet_integration_position, 
        thermostat, 
        barostat,
)


def simulate(
    sys: MDSystem,
    steps: int = 1,
    reporters: list[ReporterInteface]=[], 
) -> None:

    update_neighbors(sys)
    compute_forces(sys)
    num_neighbor_updates = 1

    for step in range(steps):

        for reporter in reporters:
           reporter.gather(sys, current_step=step)
        # save_structure(atoms, file)

        # Next time step (update r, v, and F)
        verlet_integration_position(sys)
        if check_if_neighbor_need_update(sys):
            # cells = initialize_cells(params, atoms.box)
            update_neighbors(sys)
            num_neighbor_updates += 1
        compute_forces(sys)
        verlet_integration_velocity(sys)

        # thermostat(sys)
        # barostat(sys)
