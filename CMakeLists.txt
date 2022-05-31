cmake_minimum_required(VERSION 3.10)

project(CSTask LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit)

# As fast as possible and target my current card
set(CUDA_CUDA_FLAGS -fast-math)
set(CMAKE_CUDA_ARCHITECTURES 75-real)

add_executable(CSCPUTask cpu/main.cpp
                         cpu/PDESolver.cpp
                         cpu/CallExample.cpp)
