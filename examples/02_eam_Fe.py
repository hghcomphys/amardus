import numpy as np
from ase import Atoms as AseAtoms
from ase.build import bulk
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)
from amardus import Atoms, System, Simulation
from amardus.parameters import get_default_unit
from amardus.reporters import StateReporter, XyzReporter, CSVReporter
from amardus.integration import VerletIntegrator
from amardus.coupling import BrendsenThermostat, BrendsenBarostat
from amardus.potentials import EAMPotential, EAMParams



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
    cutoff = 6.0 * unit.angstrom 
    temperature = 300.0 * unit.kelvin
    pressure = 1.0 * unit.bar
    thermostat_coupling_time = 10 * time_step
    barostat_coupling_time = 100 * time_step
    # ewald_alpha=3.5 / cutoff
    # ewald_kmax=2

    atoms_ase: AseAtoms = bulk('Fe', 'bcc', a=2.866, cubic=True).repeat((8, 8, 8))
    MaxwellBoltzmannDistribution(
        atoms_ase,
        temperature_K=temperature / unit.kelvin,
        rng=np.random.default_rng(1234),
    )
    Stationary(atoms_ase)
    atom_types_map = {"Fe": 0}
    atoms = Atoms.from_ase(atoms_ase, atom_types_map)
    # print(atoms)

    eam = EAMPotential(
        cutoff_radius=cutoff, 
        params=EAMParams(
            A=0.931312 * unit.electronvolt * unit.angstrom**-1,
            d=4.05 * unit.angstrom,
            c=2.96 * unit.angstrom,
            c0=26.27034 * unit.electronvolt * unit.angstrom**-2,
            c1=-24.40109 * unit.electronvolt * unit.angstrom**-3,
            c2=6.957871 * unit.electronvolt * unit.angstrom**-4,
            c3=-0.303077 * unit.electronvolt * unit.angstrom**-5,
            c4=-0.085092 * unit.electronvolt * unit.angstrom**-6,
            B=0.0 * unit.angstrom**-2,
        ),
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
            interval=2, 
            compressibility= 1.0 / (1.694e6 * unit.bar),
    )
    verlet = VerletIntegrator(
        time_step=time_step,
        system_couplings=[
            themorstat,
            barostat,
        ],
    )
    # print(verlet)

    state = StateReporter(interval=100, normalize=False)
    xyz = XyzReporter("fe.xyz", atom_types_map, interval=1000)
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
