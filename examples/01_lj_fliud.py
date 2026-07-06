import numpy as np
from ase.io import read as read_ase
from ase import Atoms as AseAtoms
from ase.build import bulk
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)
from amardus import Atoms, System, Simulation
from amardus.parameters import get_default_unit
from amardus.reporters import StateReporter, XyzReporter
from amardus.integration import LangevinIntegrator, VerletIntegrator
from amardus.coupling import BrendsenThermostat, BrendsenBarostat
from amardus.potentials import LennardJonesPotential, LennardJonesParams


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
    temperature = 300.0 * unit.kelvin
    pressure = 1.0 * unit.bar
    thermostat_coupling_time = 10 * time_step
    barostat_coupling_time = 10 * time_step
    # ewald_alpha=3.5 / cutoff
    # ewald_kmax=2

    # atoms_ase: AseAtoms = read_ase("argon-neon.xyz")
    atoms_ase: AseAtoms = bulk("Ar", "fcc", a=5.385, cubic=True).repeat((6, 6, 6))
    MaxwellBoltzmannDistribution(
        atoms_ase,
        temperature_K=temperature / unit.kelvin,
        rng=np.random.default_rng(1234),
    )
    Stationary(atoms_ase)
    atom_types_map = {"Ar": 0, "Ne": 1}
    atoms = Atoms.from_ase(atoms_ase, atom_types_map)
    # print(atoms)

    lennard_jones = LennardJonesPotential(
        cutoff_radius=cutoff,
        params=LennardJonesParams.from_list(
            epsilon=[e * unit.electronvolt for e in [0.01032, 0.00307]],
            sigma=[s * unit.angstrom for s in [3.405, 2.789]],
        ),
    )
    # print(lennard_jones)

    system = System.create(
        atoms,
        neighbor_cutoff=cutoff,
        neighbor_skin=1.0 * unit.angstrom,
        neighbor_max_atoms=300,
    )
    # print(system)

    verlet = VerletIntegrator(
        time_step=time_step,
        system_couplings=[
            BrendsenThermostat(temperature, thermostat_coupling_time),
            BrendsenBarostat(pressure, barostat_coupling_time, interval=10),
        ],
    )
    lagevin = LangevinIntegrator(
        time_step=time_step,
        damping_time=10 * time_step, 
        temperature=temperature,
        system_couplings=[
            BrendsenBarostat(pressure, barostat_coupling_time, interval=10),
        ],
    )
    # print(verlet)

    state = StateReporter(interval=100)
    xyz = XyzReporter("out.xyz", atom_types_map, interval=1000)

    simulation = Simulation(
        system,
        lagevin,
        forcefields=[lennard_jones],
        reporters=[state, xyz],
    )
    print(simulation)

    # simulation.step()
    simulation.step(1_000)
    # simulation.reporters.append(xyz)
    # simulation.step(50_000)

    # Heating up system
    # verlet = VerletIntegrator(time_step)
    # md = Simulation(system, verlet, forcefield, reporters=[state, xyz])
    # md.step(10_000)


if __name__ == "__main__":
    main()
