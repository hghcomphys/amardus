# Amardus

MD engine from scratch.

> *Amardus* is the Greek name for the *Sefid-Rud*, a long river in northern Iran historically known for its abundant fish.


Todos:

- [x] Linear MD
- [x] NVT and NPT ensembles
- [ ] Short-range interaction
- [ ] Electrostatic long-range
- [ ] Bonds
- [ ] Support Amber FF
- [ ] Machine-learning FF 
- [ ] Free energy calculation
- [ ] GPU implementation

---

## Setup the environment

```bash
pixi install
```

## Generate structure

```bash
pixi run python generate_structure.py --lattice 6 6 6
```

## Run benchmark simulation 

```bash
pixi run lmp < in.lmp

```




