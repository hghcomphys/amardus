from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from numba import njit
from amardus.system import System
from amardus.parameters import FLOAT, get_default_unit, Array
from amardus.coupling import ThermoBaroInterface
from numpy import random

KB = get_default_unit().boltzman


class IntegratorInterface(Protocol):
    time_step: float
    current_step: int

    def pre_forces_udpate(self, sys: System) -> None: ...

    def post_forces_update(self, sys: System) -> None: ...


@dataclass
class VerletIntegrator(IntegratorInterface):
    time_step: float
    current_step: int = field(repr=False, default=0)
    system_couplings: list[ThermoBaroInterface] = field(default_factory=list)

    def pre_forces_udpate(self, sys: System) -> None:
        _verlet_integrator_position(sys, self.time_step)

    def post_forces_update(self, sys: System) -> None:
        _verlet_integrator_velocity(sys, self.time_step)
        self.current_step += 1

        for coupling in self.system_couplings:
            coupling.apply(sys, integrator=self)


@dataclass
class LeapfrogIntegrator(IntegratorInterface):
    time_step: float
    current_step: int = field(repr=False, default=0)
    system_couplings: list[ThermoBaroInterface] = field(default_factory=list)

    def pre_forces_udpate(self, sys: System) -> None:
        _leapfrog_integrator_position(sys, self.time_step)

    def post_forces_update(self, sys: System) -> None:
        _leapfrog_integrator_velocity(sys, self.time_step)
        self.current_step += 1

        for coupling in self.system_couplings:
            coupling.apply(sys, integrator=self)



@dataclass
class LangevinIntegrator(IntegratorInterface):
    time_step: float
    damping_time: float
    temperature: float
    current_step: int = field(repr=False, default=0)
    system_couplings: list[ThermoBaroInterface] = field(default_factory=list)
    rng: random.Generator = field(default_factory=lambda: random.default_rng(1234), repr=False)

    def pre_forces_udpate(self, sys: System) -> None:
        _langevin_integrator_position(sys, 
             self.time_step, 
             self.damping_time, 
             self.temperature,
             self.rng.normal(size=sys.atoms.velocities.shape),
        )

    def post_forces_update(self, sys: System) -> None:
        _langevin_integrator_velocity(sys, self.time_step)
        self.current_step += 1

        for coupling in self.system_couplings:
            coupling.apply(sys, integrator=self)


@njit
def _verlet_integrator_position(sys: System, time_step: float) -> None:
    atoms = sys.atoms
    dt = time_step
    box = atoms.box
    num_atoms = atoms.positions.shape[0]
    pos, vel, frc = atoms.positions, atoms.velocities, atoms.forces
    mass = atoms.masses
    bx = box[0, 0], box[1, 1], box[2, 2]

    for n in range(num_atoms):
        for dim in range(3):
            pos[n, dim] = np.fmod(
                pos[n, dim]
                + dt * (vel[n, dim] + FLOAT(0.5) * dt * frc[n, dim] / mass[n])
                + FLOAT(100) * bx[dim],
                bx[dim],
            )
            vel[n, dim] += FLOAT(0.5) * dt * frc[n, dim] / mass[n]


@njit
def _verlet_integrator_velocity(sys: System, time_step: float) -> None:
    atoms = sys.atoms
    dt = time_step
    vel, frc = atoms.velocities, atoms.forces
    mass = atoms.masses.reshape(-1, 1)
    vel += FLOAT(0.5) * dt * frc / mass




@njit
def _leapfrog_integrator_position(sys: System, time_step: float) -> None:
    atoms = sys.atoms
    dt = time_step
    box = atoms.box
    num_atoms = atoms.positions.shape[0]
    pos, vel, frc = atoms.positions, atoms.velocities, atoms.forces
    mass = atoms.masses
    bx = box[0, 0], box[1, 1], box[2, 2]

    for n in range(num_atoms):
        for dim in range(3):
            vel[n, dim] += FLOAT(0.5) * dt * frc[n, dim] / mass[n] 
            pos[n, dim] = np.fmod(
                pos[n, dim] + dt * vel[n, dim] + FLOAT(100) * bx[dim],
                bx[dim],
            )


@njit
def _leapfrog_integrator_velocity(sys: System, time_step: float) -> None:
    atoms = sys.atoms
    dt = time_step
    vel, frc = atoms.velocities, atoms.forces
    mass = atoms.masses.reshape(-1, 1)
    vel += FLOAT(0.5) * dt * frc / mass


@njit
def _langevin_integrator_position(
    sys: System, 
    time_step: float, 
    damping_time: float, 
    temperature: float,
    randomness: Array,
) -> None:
    atoms = sys.atoms
    dt = time_step
    box = atoms.box
    num_atoms = atoms.positions.shape[0]
    pos, vel, frc = atoms.positions, atoms.velocities, atoms.forces
    mass = atoms.masses
    bx = box[0, 0], box[1, 1], box[2, 2]

    c1 = np.exp(- dt / damping_time)
    c2 = np.sqrt((FLOAT(1.0)-c1 * c1) * KB * temperature / mass)

    for n in range(num_atoms):
        for dim in range(3):
            vel[n, dim] += FLOAT(0.5) * dt * frc[n, dim] / mass[n] 
            pos[n, dim] += FLOAT(0.5) * dt * vel[n, dim]
            vel[n, dim] = c1 * vel[n, dim] + c2[n] * randomness[n, dim] 
            pos[n, dim] = np.fmod(
                pos[n, dim] + FLOAT(0.5) * dt * vel[n, dim] + FLOAT(100) * bx[dim],
                bx[dim],
            )

@njit
def _langevin_integrator_velocity(sys: System, time_step: float) -> None:
    atoms = sys.atoms
    dt = time_step
    vel, frc = atoms.velocities, atoms.forces
    mass = atoms.masses.reshape(-1, 1)
    vel += FLOAT(0.5) * dt * frc / mass


