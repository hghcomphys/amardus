from dataclasses import dataclass, field
from amardus.neighbors import (
    update_neighbors,
    check_if_neighbor_need_update,
)
from amardus.system import System
from amardus.forces import ForceFieldInterface
from amardus.io import ReporterInteface
from amardus.integration import IntegrationInterface


@dataclass
class Simulation:
    system: System
    integration: IntegrationInterface
    forcefield: ForceFieldInterface
    cutoff_radius: float
    skin_radius: float
    current_step: int = 0
    reporters: list[ReporterInteface] = field(default_factory=list)
    
    def step(self, number: int = 1) -> None:

        sys = self.system
        update_neighbors(sys, self.cutoff_radius, self.skin_radius)
        self.forcefield.compute_forces(sys)
        num_neighbor_updates = 1

        for _ in range(number):

            self.current_step += 1
            for reporter in self.reporters:
               reporter.collect(sys, self.current_step)

            # Next time step (update r, v, and F)
            self.integration.pre_forces_udpate(sys)
            
            if check_if_neighbor_need_update(sys, self.skin_radius):
                # cells = initialize_cells(params, atoms.box)
                update_neighbors(sys, self.cutoff_radius, self.skin_radius)
                num_neighbor_updates += 1

            self.forcefield.compute_forces(sys)
            self.integration.post_forces_update(sys)

