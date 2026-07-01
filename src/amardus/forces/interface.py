from typing import Protocol
from amardus.system import System



class ForceFieldInterface(Protocol):

    def compute_forces(self, sys: System) -> None: ...


