from dataclasses import dataclass, field
from amardus.neighbors import (
    update_neighbors,
    check_if_neighbor_need_update,
)
from amardus.system import System
from amardus.forces import compute_forces
from amardus.io import ReporterInteface
from amardus.integration import (
        IntegrationInterface, 
        brendsen_barostat, 
        brendsen_thermostat,
)


@dataclass
class Simulation:
    system: System
    integration: IntegrationInterface
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
            self.integration.pre_force_udpate(sys)
            
            if check_if_neighbor_need_update(sys):
                # cells = initialize_cells(params, atoms.box)
                update_neighbors(sys)
                num_neighbor_updates += 1

            compute_forces(sys)
            self.integration.post_force_update(sys)

