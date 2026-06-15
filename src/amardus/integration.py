import math
from typing import Optional, Protocol
import numpy as np
from numba import njit
from amardus.system import System, get_temperature, get_pressure
from amardus.parameters import FLOAT


class IntegrationInterface(Protocol):
    
    def pre_force_udpate(self, sys: System) -> None:
        ...

    def post_force_update(self, sys: System) -> None:
        ...


class VerletIntegrator(IntegrationInterface):

    def __init__(
        self, 
        temperature: Optional[float] = None, 
        pressure: Optional[float] = None,
    ):
        self.temperature = temperature
        self.pressure = pressure

    def pre_force_udpate(self, sys: System) -> None:
        verlet_integrator_position(sys)
        
    def post_force_update(self, sys: System) -> None:
        verlet_integrator_velocity(sys)

        if self.temperature is not None:
            brendsen_thermostat(sys, self.temperature)
        if self.pressure:
            brendsen_barostat(sys, self.pressure)


@njit
def verlet_integrator_position(sys: System) -> None:
    atoms, params = sys.atoms, sys.params
    dt = params.time_step
    box = atoms.box
    num_atoms = len(atoms.positions)
    m, r, v, f = atoms.masses, atoms.positions, atoms.velocities, atoms.forces
    length = box[0, 0], box[1, 1], box[2, 2]

    for n in range(num_atoms):
        for dim in range(3):
            r[n, dim] = np.fmod(
                r[n, dim]
                + dt * (v[n, dim] + FLOAT(0.5) * dt / m[n] * f[n, dim])
                + FLOAT(100.0) * length[dim],
                length[dim],
            )
            v[n, dim] += FLOAT(0.5) * dt / m[n] * f[n, dim]

@njit
def verlet_integrator_velocity(sys: System) -> None:
    atoms, params = sys.atoms, sys.params
    dt = params.time_step
    m, v, f = atoms.masses.reshape(-1, 1), atoms.velocities, atoms.forces
    v += FLOAT(0.5) * dt / m * f

@njit
def brendsen_thermostat(sys: System, temperature: float) -> None:
    atoms, params = sys.atoms, sys.params
    dt = params.time_step
    tau = params.thermostat_coupling_time
    v = atoms.velocities
    t0 = temperature 

    t = get_temperature(sys)
    scale = FLOAT(math.sqrt(FLOAT(1.0) + (dt / tau) * (t0 / t - FLOAT(1.0))))
    v *= scale

@njit
def brendsen_barostat(sys: System, pressure: float) -> None:
    atoms, params = sys.atoms, sys.params
    dt = params.time_step
    tau = params.barostat_coupling_time
    beta = params.compressibility
    pos = atoms.positions
    box = atoms.box
    p0 = pressure

    p = get_pressure(sys)
    scale = (1.0 - beta * dt / tau * (p0 - p)) ** (1.0 / 3.0)
    pos *= scale
    box *= scale

