from amardus.neighbors import (
    update_neighbors,
    check_if_neighbor_need_update,
)
from amardus.system import MDSystem
from amardus.forces import compute_forces
from amardus.io import ReporterInteface
from amardus.integration import (
        verlet_integration_velocity, 
        verlet_integration_position, 
        thermostat, 
        barostat,
)
from dataclasses import dataclass, field


@dataclass
class MDSimulation:
    system: MDSystem
    reporters: list[ReporterInteface] = field(default_factory=list)
    current_step: int = 0

    def step(
        self,
        number: int = 1,
    ) -> None:

        sys = self.system
        update_neighbors(sys)
        compute_forces(sys)
        num_neighbor_updates = 1

        for _ in range(number):
            self.current_step += 1
            for reporter in self.reporters:
               reporter.gather(sys, self.current_step)

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
