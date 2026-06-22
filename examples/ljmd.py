import numpy as np
from ase.io import read as read_ase
from ase import Atoms as AseAtoms
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)
from amardus.io import StateReporter, XyzReporter
from amardus import (
    Atoms,
    System,
    Simulation,
    VerletIntegrator,
)
from amardus.parameters import get_default_unit
from amardus.forces import LennardJonesForceField, LennardJonesParams
from amardus.integration import BrendsenThermostat, BrendsenBarostat 


def main() -> None:

    print(r"""
      _                            _
     / \   _ __ ___   __ _ _ __ __| |_   _ ___
    / _ \ | '_ ` _ \ / _` | '__/ _` | | | / __|
   / ___ \| | | | | | (_| | | | (_| | |_| \__ \
  /_/   \_\_| |_| |_|\__,_|_|  \__,_|\__,_|___/
           Molecular Dynamics Engine
-------------------------------------------------
""")

    unit = get_default_unit()
    time_step = 0.001 * unit.picoseconds
    cutoff = 7.0 * unit.angstrom
    temperature=300.0 * unit.kelvin
    pressure=1.0 * unit.bar
    thermostat_coupling_time=10 * time_step
    barostat_coupling_time=10 * time_step
    # ewald_alpha=3.5 / cutoff
    # ewald_kmax=2

    atoms_ase: AseAtoms = read_ase("argon.xyz")
    MaxwellBoltzmannDistribution(
        atoms_ase,
        temperature_K=temperature / unit.kelvin,
        rng=np.random.default_rng(1234),
    )
    Stationary(atoms_ase)
    atom_types_map = {"Ar": 0, "Ne": 1}
    atoms = Atoms.from_ase(atoms_ase, atom_types_map)
    # print(atoms)

    forcefield = LennardJonesForceField(
        cutoff_radius=cutoff,
        params=LennardJonesParams.from_list(
            epsilon=[e * unit.electronvolt for e in [0.01032, 0.00307]],
            sigma=[s * unit.angstrom for s in [3.405, 2.789]],
        )
    )
    # print(forcefield)

    system = System.create(
        atoms, 
        neighbor_cutoff=cutoff, 
        neighbor_skin = 1.0 * unit.angstrom, 
        neighbor_max_atoms=300,
    )
    print("System: Lennard-Jones particles")
    print(f"Number of atoms: {len(atoms)}")
    # print(system)
        
    integrator = VerletIntegrator(
        time_step=time_step,
        system_coupling=[
            BrendsenThermostat(temperature, thermostat_coupling_time),
            BrendsenBarostat(pressure, barostat_coupling_time, interval=10),
        ]
    )
    # print(integrator)

    state = StateReporter(interval=100)
    xyz  = XyzReporter("out.xyz", atom_types_map, interval=1000)

    simulation = Simulation(
        system,
        integrator,
        forcefield,
        reporters=[state] #, xyz],
    )
    # print(simulation)

    # simulation.step()
    simulation.step(10_000)
    # simulation.reporters.append(xyz)
    # simulation.step(40_000)

    # Heating up system
    # verlet = VerletIntegrator(time_step)
    # md = Simulation(system, verlet, forcefield, reporters=[state, xyz])
    # md.step(10_000)


if __name__ == "__main__":
    main()
