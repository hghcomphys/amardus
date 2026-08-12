from dataclasses import dataclass
from amardus.system import System
from amardus.parameters import get_default_unit
from typing import Protocol


class EnergyMinimizerInterface(Protocol):

    @property
    def max_steps(self) -> int: ...

    @property
    def force_tolerance(self) -> float: ...

    def update(self, sys: System) -> None: ...


@dataclass
class GradientDescentEnergyMinimizer:
    max_steps: int
    force_tolerance: float
    step_size: float

    def update(self, sys: System) -> None:
        pos = sys.atoms.positions
        pos[:] += self.step_size * sys.atoms.forces


def get_default_minimizer() -> EnergyMinimizerInterface:
    unit = get_default_unit()
    return GradientDescentEnergyMinimizer(
        max_steps=10_000, 
        force_tolerance=0.001 * unit.electronvolt,
        step_size=0.005*unit.angstrom**2*unit.electronvolt**-1, 
    )



