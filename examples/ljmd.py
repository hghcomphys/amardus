from amardus.io import StateReporter
from amardus import (
    Unit,
    Parameters,
    Atoms,
    simulate,
)
from ase.io import read as ase_read
from ase import Atoms as AseAtoms
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)
import numpy as np


def run_simulation() -> None:

    unit = Unit()
    time_step = 1.0 * unit.femtoseconds
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

    ase_atoms: AseAtoms = ase_read("argon.xyz")
    MaxwellBoltzmannDistribution(
        ase_atoms,
        temperature_K=params.temperature,
        rng=np.random.default_rng(1234),
    )
    Stationary(ase_atoms)

    atoms = Atoms.from_ase(ase_atoms)
    reporter = StateReporter(interval=100, unit=unit)

    print("MD simulation of Lennard-Jones particles.")
    print(f"Number of atoms: {len(atoms)}")
    simulate(params, atoms, steps=1_001, reporter=reporter)
    print("Done.")


if __name__ == "__main__":
    run_simulation()
