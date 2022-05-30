### Solving PDEs

An attempt to solve PDEs using three methods (Explicit, Implicit and Crank-Nicolson)

A CPU reference is used to confirm I'm in the right direction and hone knowledge.
A toy example will be used (call option on a bank stock) to have a result to play with.

## Structure
build.bat Windows helper to run CMake
CMakeLists.txt
toolchains Compiler optimisations via cmake
cpu/ Reference CPU implementation
gpu/ CUDA implementation

## Papers that help
"GPU implementation of finite difference solvers"
Mark Giles and Jeremy Appleyard
https://people.maths.ox.ac.uk/~gilesm/files/WHPCF14.pdf
