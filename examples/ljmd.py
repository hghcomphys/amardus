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


def run_simulation() -> None:

    unit = get_default_unit()
    time_step = 0.002 * unit.picoseconds
    cutoff = 9.0 * unit.angstrom
    temperature=300.0 * unit.kelvin
    cutoff_radius=cutoff * unit.angstrom
    skin_radius=1.0 * unit.angstrom
    # pressure=1.0 * unit.bar
    # thermostat_coupling_time=10 * time_step
    # barostat_coupling_time=10 * time_step
    # compressibility=1.0
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
        cutoff_radius=cutoff_radius,
        params=LennardJonesParams.from_list(
            epsilon=[e * unit.electronvolt for e in [0.01032, 0.00307]],
            sigma=[s * unit.angstrom for s in [3.405, 2.789]],
        )
    )
    # print(forcefield)

    system = System.create(atoms, cell_length=cutoff_radius, neighbor_max_atoms=300)
    print("System: Lennard-Jones particles")
    print(f"Number of atoms: {len(atoms)}")
    # print(system)
        
    state = StateReporter(interval=1)
    xyz  = XyzReporter("out.xyz", atom_types_map, interval=1000)

    # Heating up system
    # verlet = VerletIntegrator(temperature=300.0 * unit.kelvin)
    # md = Simulation(system, verlet, reporters=[state, xyz])
    # md.step(10_000)

    integrator = VerletIntegrator(
        time_step=time_step,
        # temperature=temperature,
        # pressure=pressure,
    )
    # print(integrator)

    simulation = Simulation(
        system,
        integrator,
        forcefield,
        cutoff_radius=cutoff_radius,
        skin_radius=skin_radius, 
        reporters=[state, xyz],
    )
    # print(simulation)


    simulation.step()
    simulation.step(3)
    # simulation.reporters.append(xyz)
    # simulation.step(30_000)


if __name__ == "__main__":
    run_simulation()
