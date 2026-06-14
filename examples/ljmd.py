import numpy as np
from ase.io import read as read_ase
from ase import Atoms as AseAtoms
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)
from amardus.forces import LennardJones
from amardus.io import StateReporter, XyzReporter
from amardus import (
    Unit,
    Parameters,
    Atoms,
    System,
    Simulation,
    LennardJones,
    ForceField,
    VelocityVerlet,
)


def run_simulation() -> None:

    unit = Unit()
    time_step = 0.001 * unit.picoseconds
    cutoff = 9.0 * unit.angstrom
    params = Parameters(
        time_step=time_step,
        temperature=100.0 * unit.kelvin,
        pressure=1.0 * unit.bar,
        cutoff_radius=cutoff * unit.angstrom,
        skin_radius=1.0 * unit.angstrom,
        thermostat_coupling_time=10 * time_step,
        barostat_coupling_time=10 * time_step,
        compressibility=1.0,
        neighbor_max_atoms=300,
        ewald_alpha=3.5 / cutoff,
        ewald_kmax=2,
        boltzman=unit.boltzman,
        coulomb=unit.coulomb,
    )
    atoms_ase: AseAtoms = read_ase("argon.xyz")
    MaxwellBoltzmannDistribution(
        atoms_ase,
        temperature_K=params.temperature,
        rng=np.random.default_rng(1234),
    )
    Stationary(atoms_ase)
    atom_types_map = {"Ar": 0}
    atoms = Atoms.from_ase(atoms_ase, atom_types_map)
    forcefield = ForceField(
        LennardJones.from_atom_types(
            epsilon=[1.032e-2],
            sigma=[3.405],
        )
    )
    system = System.create(atoms, params, forcefield)
    print("System: Lennard-Jones particles")
    print(f"Number of atoms: {len(atoms)}")

    integration = VelocityVerlet(
        # thermostat=True,
        # barostat=True,
    )
    simulation = Simulation(
        system,
        integration,
        reporters=[StateReporter(interval=100, unit=unit)],
    )
    simulation.step(1_001)
    # simulation.reporters.append(XyzReporter("out.xyz", atom_types_map, interval=1000, unit=unit))
    # simulation.step(30_000)


if __name__ == "__main__":
    run_simulation()
