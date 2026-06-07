import math
import numpy as np
from typing import Protocol, TextIO
from amardus.atoms import Atoms
from amardus.parameters import Unit
from amardus.system import (
    MDSystem,
    get_kinetic_energy,
    get_pressure,
    get_temperature,
)


class ReporterInteface(Protocol):
    def gather(self, sys: MDSystem, current_step: int) -> None: ...


class StateReporter(ReporterInteface):

    def __init__(
        self,
        unit: Unit,
        interval: int = 1,
        show_header: bool = True,
    ) -> None:
        self.unit = unit
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
            f" {'AvgCell':10}"
            f" {'Box':10}"
            f" {'Dens':10}"
        )

    def gather(
        self,
        sys: MDSystem,
        current_step: int,
    ) -> None:

        atoms = sys.atoms
        energy = sys.energy
        cells = sys.cells
        neighbors = sys.neighbors

        if self.show_header:
            print(self.header)
            self.show_header = False

        if current_step % self.interval == 0:

            unit = self.unit
            ke = get_kinetic_energy(sys)
            pe = energy[0]
            temp = get_temperature(sys)
            pres = get_pressure(sys)
            vol = np.linalg.det(atoms.box)
            dens = np.sum(atoms.mass) / vol

            print(
                f"{current_step:<8}"
                f" {temp / unit.kelvin:<10.4f}"
                f" {pres / unit.bar:<10.4f}"
                f" {pe / unit.electronvolt:<10.4f}"
                f" {ke / unit.electronvolt:<10.4f}"
                f" {(ke + pe) / unit.electronvolt:<10.4f}"
                # f" {num_neighbor_updates:<10}"
                f" {neighbors.num_neighbors_per_atom.mean():<10.2f}"
                f" {cells.num_atoms_per_cell.mean():<10.2f}"
                f" {math.pow(vol, 1/3):<10.4f}"
                f" {dens / unit.grcm3:<10.6f}"
            )


def save_structure(atoms: Atoms, file: TextIO) -> None:
    position = atoms.position
    natoms = position.shape[0]
    file.write(f"{natoms}\n\n")
    for i in range(natoms):
        file.write(f"{'Ar'}\t{position[i, 0]} {position[i, 1]} {position[i, 2]}\n")
    file.flush()
