from ase.build import bulk
from ase.io import write
import argparse


def build_lj(
    lattice: tuple[int, int, int],
    output_file: str = "argon.xyz",
):
    nx, ny, nz = lattice

    atoms = bulk("Ar", "fcc", a=5.385, cubic=True)
    atoms = atoms.repeat((nx, ny, nz))
    natoms = len(atoms)

    print("Generating Lennard-Jones particles")
    print(f"Number of atoms: {natoms}")
    print(f"Lattice: ({nx}, {ny}, {nz})")
    print(f"Output file: '{output_file}'")

    write(output_file, atoms)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a lattice with given dimensions"
    )
    parser.add_argument(
        "--lattice",
        metavar=("NX", "NY", "NZ"),
        type=int,
        nargs=3,
        default=(10, 10, 10),
        help="Lattice dimensions to repeat (default: 10 10 10)",
    )
    parser.add_argument(
        "--system",
        default="lj",
        choices=["lj"],
        help="System to generate (default: lj)",
    )
    args = parser.parse_args()

    if args.system == "lj":
        build_lj(args.lattice)


if __name__ == "__main__":
    main()
