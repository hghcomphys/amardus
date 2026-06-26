import math
import numpy as np
from pathlib import Path
from typing import Protocol
from amardus.parameters import get_default_unit
from amardus.system import (
    System,
    get_kinetic_energy,
    get_pressure,
    get_temperature,
)

PathLike = Path | str


class ReporterInteface(Protocol):
    def collect(self, sys: System, current_step: int) -> None: ...


class StateReporter(ReporterInteface):

    def __init__(
        self,
        interval: int = 1,
        show_header: bool = True,
    ) -> None:
        self.unit = get_default_unit()
        self.interval = interval
        self.show_header: bool = show_header
        self.header: str = (
            f"{'Step':8}"
            f" {'Temp':10}"
            f" {'Pres':10}"
            f" {'PotE':10}"
            f" {'KinE':10}"
            f" {'TotE':10}"
            # f" {'UpdateNb':10}"
            f" {'AvgNb':10}"
            # f" {'AvgCell':10}"
            # f" {'Box':10}"
            f" {'Dens':10}"
        )

    def collect(
        self,
        sys: System,
        current_step: int,
    ) -> None:

        if self.show_header:
            print(self.header)
            self.show_header = False

        if current_step % self.interval != 0:
            return

        unit = self.unit
        ke = get_kinetic_energy(sys)
        pe = sys.energy[0]
        temp = get_temperature(sys)
        pres = get_pressure(sys)
        vol = np.linalg.det(sys.atoms.box)
        dens = np.sum(sys.atoms.masses) / vol

        print(
            f"{current_step:<8}"
            f" {temp / unit.kelvin:<10.4f}"
            f" {pres / unit.bar:<10.4f}"
            f" {pe / unit.electronvolt:<10.4f}"
            f" {ke / unit.electronvolt:<10.4f}"
            f" {(ke + pe) / unit.electronvolt:<10.4f}"
            # f" {num_neighbor_updates:<10}"
            f" {sys.neighbors.num_neighbors_per_atom.mean():<10.2f}"
            # f" {sys.cells.num_atoms_per_cell.mean():<10.2f}"
            # f" {math.pow(vol, 1/3):<10.4f}"
            f" {dens / unit.grcm3:<10.6f}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(interval={self.interval})" 


class XyzReporter(ReporterInteface):

    def __init__(
        self, 
        filename: PathLike,
        atom_types_map: dict[str, int],
        interval: int = 1000,
        flush_immediately: bool = False,
    ) -> None:
        self.unit = get_default_unit()
        self.file = open(str(filename), "w")
        self.atoms_symbol_map = {v: k for k, v in atom_types_map.items()}
        self.interval = interval
        self.flush_immediately = flush_immediately

    def collect(self, sys: System, current_step: int) -> None:

        if current_step % self.interval != 0:
            return 
        
        pos = sys.atoms.positions
        num_atoms = pos.shape[0] 
        unit = self.unit

        self.file.write(f"{num_atoms}\n\n")
        for i in range(num_atoms):
            at = sys.atoms.atom_types[i]
            self.file.write(
                f"{self.atoms_symbol_map[at]}\t"
                f"{pos[i, 0] / unit.angstrom} "
                f"{pos[i, 1] / unit.angstrom} "
                f"{pos[i, 2] / unit.angstrom} "
                "\n"
            )
            if self.flush_immediately:
                self.file.flush()

    def __del__(self):
        if self.file:
            self.file.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(interval={self.interval})" 
