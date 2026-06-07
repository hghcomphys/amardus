from amardus.io import StateReporter
from amardus import (
    Unit,
    Parameters,
    Atoms,
    MDSystem,
    MDSimulation,
)
from ase.io import read as read_ase
from ase import Atoms as AseAtoms
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)
import numpy as np

from amardus.system import MDSystem


def run_simulation() -> None:

    unit = Unit()
    time_step = 0.001 * unit.picoseconds
    cutoff = 9.0 * unit.angstrom
    params = Parameters(
        time_step=time_step,
        temperature=300.0 * unit.kelvin,
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
    atoms = Atoms.from_ase(atoms_ase)

    system = MDSystem.create(atoms, params)
    reporter = StateReporter(interval=100, unit=unit)
    simulation = MDSimulation(
        system, 
        reporters=[reporter],
    )

    print("MD simulation of Lennard-Jones particles.")
    print(f"Number of atoms: {len(atoms)}")
    simulation.step(1_001)
    # simulation.step(1_001)
    print("Done.")


if __name__ == "__main__":
    run_simulation()
