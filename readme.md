#Amardus
A GPU-accelerated Python Library for Calculating Atomic Descriptors.

> *Amardus* is the Greek name for the *Sefid-Rud*, a long river in northern Iran historically known for its abundant fish.

## Why?
Pantea's experience has shown that JAX runs slow for non-array computations when system is large or number of descriptor increases. 
JIT compilation for calculating Auto-gradient of functions takes too long as well. 
Therefore, a low-level implementation of atomic descriptors and combining it with JAX potential model seems to be a more efficient approach.  

## Intended Features
- In Python and GPU-accelerated 
- Atomic-centered symmetry functions (ACSF) 
- Relying on numerical differentiation 
- JIT compiled using Numba-CUDA
- To be used together with JAX
