from dataclasses import dataclass, field
import math
from typing import Protocol
import numpy as np
from numba import njit
from amardus.system import System, get_temperature, get_pressure
from amardus.parameters import FLOAT, get_default_unit, Array
from numpy import random

KB = get_default_unit().boltzman


class IntegrationInterface(Protocol):
    time_step: float
    current_step: int

    def pre_forces_udpate(self, sys: System) -> None: ...

    def post_forces_update(self, sys: System) -> None: ...


class ThermoBaroInterface(Protocol):

    def apply(self, sys: System, integrator: IntegrationInterface) -> None: ...


@dataclass
class VerletIntegrator(IntegrationInterface):
    time_step: float
    current_step: int = 0
    system_coupling: list[ThermoBaroInterface] = field(default_factory=list)

    def pre_forces_udpate(self, sys: System) -> None:
        verlet_integrator_position(sys, self.time_step)

    def post_forces_update(self, sys: System) -> None:
        verlet_integrator_velocity(sys, self.time_step)
        self.current_step += 1

        for coupling in self.system_coupling:
            coupling.apply(sys, integrator=self)


@njit
def verlet_integrator_position(sys: System, time_step: float) -> None:
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
def verlet_integrator_velocity(sys: System, time_step: float) -> None:
    atoms = sys.atoms
    dt = time_step
    vel, frc = atoms.velocities, atoms.forces
    mass = atoms.masses.reshape(-1, 1)
    vel += FLOAT(0.5) * dt * frc / mass


@dataclass
class LeapfrogIntegrator(IntegrationInterface):
    time_step: float
    current_step: int = 0
    system_coupling: list[ThermoBaroInterface] = field(default_factory=list)

    def pre_forces_udpate(self, sys: System) -> None:
        leapfrog_integrator_position(sys, self.time_step)

    def post_forces_update(self, sys: System) -> None:
        leapfrog_integrator_velocity(sys, self.time_step)
        self.current_step += 1

        for coupling in self.system_coupling:
            coupling.apply(sys, integrator=self)


@njit
def leapfrog_integrator_position(sys: System, time_step: float) -> None:
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
def leapfrog_integrator_velocity(sys: System, time_step: float) -> None:
    atoms = sys.atoms
    dt = time_step
    vel, frc = atoms.velocities, atoms.forces
    mass = atoms.masses.reshape(-1, 1)
    vel += FLOAT(0.5) * dt * frc / mass



@dataclass
class LangevinIntegrator(IntegrationInterface):
    time_step: float
    damping_time: float
    temperature: float
    current_step: int = 0
    system_coupling: list[ThermoBaroInterface] = field(default_factory=list)
    rng: random.Generator = field(default_factory=lambda: random.default_rng(1234))

    def pre_forces_udpate(self, sys: System) -> None:
        langevin_integrator_position(sys, 
             self.time_step, 
             self.damping_time, 
             self.temperature,
             self.rng.normal(size=sys.atoms.velocities.shape),
        )

    def post_forces_update(self, sys: System) -> None:
        langevin_integrator_velocity(sys, self.time_step)
        self.current_step += 1

        for coupling in self.system_coupling:
            coupling.apply(sys, integrator=self)


@njit
def langevin_integrator_position(
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
def langevin_integrator_velocity(sys: System, time_step: float) -> None:
    atoms = sys.atoms
    dt = time_step
    vel, frc = atoms.velocities, atoms.forces
    mass = atoms.masses.reshape(-1, 1)
    vel += FLOAT(0.5) * dt * frc / mass


@dataclass
class BrendsenThermostat:
    target_temperature: float
    coupling_time: float
    interval: int = 1

    def apply(self, sys: System, integrator: IntegrationInterface) -> None:
        if integrator.current_step % self.interval != 0:
            return

        brendsen_thermostat(
            sys,
            self.target_temperature,
            self.coupling_time,
            integrator.time_step,
        )


@njit
def brendsen_thermostat(
    sys: System,
    temperature: float,
    coupling_time: float,
    time_step: float,
) -> None:
    atoms = sys.atoms
    dt = time_step
    tau = coupling_time
    vel = atoms.velocities
    temp0 = temperature

    temp = get_temperature(sys)
    scale = FLOAT(math.sqrt(FLOAT(1.0) + (dt / tau) * (temp0 / temp - FLOAT(1.0))))
    vel *= scale


@dataclass
class BrendsenBarostat:
    target_pressure: float
    coupling_time: float
    compressibility: float = 1.0
    interval: int = 1

    def apply(self, sys: System, integrator: IntegrationInterface) -> None:

        if integrator.current_step % self.interval != 0:
            return

        brendsen_barostat(
            sys,
            self.target_pressure,
            self.coupling_time,
            self.compressibility,
            integrator.time_step,
        )


@njit
def brendsen_barostat(
    sys: System,
    pressure: float,
    coupling_time: float,
    compressibility: float,
    time_step: float,
) -> None:
    atoms = sys.atoms
    dt = time_step
    tau = coupling_time
    beta = compressibility
    pos = atoms.positions
    box = atoms.box
    pres0 = pressure

    pres = get_pressure(sys)
    scale = (FLOAT(1.0) - beta * dt / tau * (pres0 - pres)) ** (1.0 / 3.0)
    pos *= scale
    box *= scale
