import numpy as np
from ase import Atoms as AseAtoms
from ase.build import bulk
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)
from amardus import BANNER, Atoms, System, Simulation
from amardus.parameters import get_default_unit
from amardus.reporters import StateReporter, XyzReporter, CSVReporter
from amardus.integration import VerletIntegrator
from amardus.coupling import BrendsenThermostat, BrendsenBarostat
from amardus.potentials import EAMPotential, EAMParams



def main() -> None:

    print(BANNER)
    print("Description: MD simulation of Cu crystal using EAM potential.")

    unit = get_default_unit()
    time_step = 0.001 * unit.picoseconds
    cutoff = 5.3 * unit.angstrom 
    temperature = 300.0 * unit.kelvin
    pressure = 1.0 * unit.bar
    thermostat_coupling_time = 10 * time_step
    barostat_coupling_time = 100 * time_step
    # ewald_alpha=3.5 / cutoff
    # ewald_kmax=2

    atoms_ase: AseAtoms = bulk('Cu', 'fcc', a=3.615, cubic=True).repeat((8, 8, 8))
    MaxwellBoltzmannDistribution(
        atoms_ase,
        temperature_K=temperature / unit.kelvin,
        rng=np.random.default_rng(1234),
    )
    Stationary(atoms_ase)
    atom_types_map = {"Cu": 0}
    atoms = Atoms.from_ase(atoms_ase, atom_types_map)
    # print(atoms)

    cu_prams = EAMParams(
        A=0.391865 * unit.electronvolt * unit.angstrom**-1,
        d=4.32 * unit.angstrom,
        c=4.29 * unit.angstrom,
        c0=10.18724 * unit.electronvolt * unit.angstrom**-2,
        c1=-12.82033 * unit.electronvolt * unit.angstrom**-3,
        c2=6.176587 * unit.electronvolt * unit.angstrom**-4,
        c3=-1.341391 * unit.electronvolt * unit.angstrom**-5,
        c4=0.109842 * unit.electronvolt * unit.angstrom**-6,
        B=-0.881096 * unit.angstrom**-2,
    )
    eam = EAMPotential(
        cutoff_radius=cutoff, 
        params=cu_prams,
    )
    # print(lennard_jones)

    system = System.create(
        atoms,
        neighbor_cutoff=cutoff,
        neighbor_skin=2.0 * unit.angstrom,
        neighbor_max_atoms=300,
    )
    # print(system)

    themorstat = BrendsenThermostat(temperature, thermostat_coupling_time)
    barostat = BrendsenBarostat(
            pressure, 
            barostat_coupling_time, 
            interval=1, 
            compressibility= 1.0 / (1.484e6 * unit.bar),
    )
    verlet = VerletIntegrator(
        time_step=time_step,
        system_couplings=[
            themorstat,
            barostat,
        ],
    )
    # print(verlet)

    state = StateReporter(interval=100)
    xyz = XyzReporter("cu.xyz", atom_types_map, interval=1000)
    csv = CSVReporter("/tmp/thermo.csv", interval=500)

    simulation = Simulation(
        system,
        verlet,
        forcefields=[eam],
        reporters=[state, xyz, csv],
    )
    print(simulation)

    simulation.step()
    simulation.step(10_000)
    simulation.step(100_000)


if __name__ == "__main__":
    main()
