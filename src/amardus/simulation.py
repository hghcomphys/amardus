from dataclasses import dataclass, field
from amardus.neighbors import (
    update_neighbors,
    check_if_neighbor_need_update,
)
from amardus.system import System
from amardus.reporters import ReporterInteface
from amardus.integration import IntegratorInterface
from amardus.potentials import ForceFieldInterface 
from amardus.minimization import EnergyMinimizerInterface, get_default_minimizer
import numpy as np



@dataclass
class Simulation:
    system: System
    integrator: IntegratorInterface
    forcefields: list[ForceFieldInterface]
    reporters: list[ReporterInteface] = field(default_factory=list)
    current_step: int = field(default=0, repr=False)
    energy_minimizer: EnergyMinimizerInterface = field(default_factory=get_default_minimizer)

    def step(self, number: int = 1) -> None:

        sys = self.system
        update_neighbors(sys)
        for ff in self.forcefields: 
            ff.compute_forces(self.system)  

        for _ in range(number+1):
            for reporter in self.reporters:
               reporter.collect(sys, self.current_step)

            # Next time step (update r, v, and F)
            self.integrator.pre_forces_udpate(sys)
            
            if check_if_neighbor_need_update(sys):
                # cells = initialize_cells(params, atoms.box)
                update_neighbors(sys)
            for ff in self.forcefields: 
                ff.compute_forces(self.system)  

            self.integrator.post_forces_update(sys)
            self.current_step += 1

    def minimize_energy(self) -> None:

        sys = self.system
        forces = sys.atoms.forces

        update_neighbors(sys)
        for ff in self.forcefields: 
            ff.compute_forces(sys)  

        for step in range(self.energy_minimizer.max_steps):

            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force < self.energy_minimizer.force_tolerance:
                print(
                    f"Energy minimization converged after {step} steps."
                )
                return

            if check_if_neighbor_need_update(sys):
                # cells = initialize_cells(params, atoms.box)
                update_neighbors(sys)
            for ff in self.forcefields: 
                ff.compute_forces(self.system)  

            self.energy_minimizer.update(sys)
        else:
            print(f"WARNNING: Energy minmizer didn't converge after {self.minimize_energy.max_steps} steps.")


