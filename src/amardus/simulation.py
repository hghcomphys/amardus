from dataclasses import dataclass, field
from amardus.neighbors import (
    update_neighbors,
    check_if_neighbor_need_update,
)
from amardus.system import System
from amardus.reporters import ReporterInteface
from amardus.integration import IntegrationInterface
from amardus.potentials import ForceFieldInterface 


@dataclass
class Simulation:
    system: System
    integration: IntegrationInterface
    forcefields: list[ForceFieldInterface]
    reporters: list[ReporterInteface] = field(default_factory=list)
    current_step: int = field(default=0, repr=False)
    
    def step(self, number: int = 1) -> None:

        sys = self.system
        update_neighbors(sys)
        for ff in self.forcefields: 
            ff.compute_forces(sys)

        for _ in range(number+1):
            for reporter in self.reporters:
               reporter.collect(sys, self.current_step)

            # Next time step (update r, v, and F)
            self.integration.pre_forces_udpate(sys)
            
            if check_if_neighbor_need_update(sys):
                # cells = initialize_cells(params, atoms.box)
                update_neighbors(sys)

            for ff in self.forcefields:
                ff.compute_forces(sys)

            self.integration.post_forces_update(sys)
            self.current_step += 1

