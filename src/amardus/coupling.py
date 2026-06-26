import math
from numba import njit
from dataclasses import dataclass
from typing import Protocol
from amardus.system import System, get_temperature, get_pressure
from amardus.parameters import FLOAT


class IntegrationInterface(Protocol):
    time_step: float
    current_step: int


class ThermoBaroInterface(Protocol):

    def apply(self, sys: System, integrator: IntegrationInterface) -> None: ...


@dataclass
class BrendsenThermostat(ThermoBaroInterface):
    target_temperature: float
    coupling_time: float
    interval: int = 1

    def apply(self, sys: System, integrator: IntegrationInterface) -> None:
        if integrator.current_step % self.interval != 0:
            return

        _brendsen_thermostat(
            sys,
            self.target_temperature,
            self.coupling_time,
            integrator.time_step,
        )


@dataclass
class BrendsenBarostat(ThermoBaroInterface):
    target_pressure: float
    coupling_time: float
    compressibility: float = 1.0
    interval: int = 1

    def apply(self, sys: System, integrator: IntegrationInterface) -> None:
        if integrator.current_step % self.interval != 0:
            return

        _brendsen_barostat(
            sys,
            self.target_pressure,
            self.coupling_time,
            self.compressibility,
            integrator.time_step,
        )


@njit
def _brendsen_thermostat(
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


@njit
def _brendsen_barostat(
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
