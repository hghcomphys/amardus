from amardus import (
    Unit,
    Parameters,
    Particles,
    simulate,
    generate_velocity,
)
from ase.io import read
import numpy as np

ORDER = "C"
FLOAT = np.float64


def run_simulation() -> None:

    ase_atoms = read("argon.xyz")
    num_atoms = len(ase_atoms)

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
        unit=unit,
    )

    charge = ase_atoms.get_initial_charges()
    mass = ase_atoms.get_masses()
    position = np.asarray(ase_atoms.get_positions(), dtype=FLOAT, order=ORDER)
    velocity = generate_velocity(mass, params.temperature, unit.K_B)
    force = np.empty((num_atoms, 3), dtype=FLOAT, order=ORDER)
    box = np.array(ase_atoms.cell.array, dtype=FLOAT)
    particles = Particles(
        mass=mass,
        position=position,
        velocity=velocity,
        force=force,
        charge=charge,
        box=box,
        position_old=position.copy(),
    )
    # print(particles)

    print("MD simulation of Lennard-Jones system.")
    print(f"Number of atoms: {num_atoms}")
    simulate(params, particles, steps=1_001)
    print("Done.")


if __name__ == "__main__":
    run_simulation()
