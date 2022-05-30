if(DEFINED IMAGUS_FLAGS_AVX2_CMAKE_)
  return()
else()
  set(IMAGUS_FLAGS_AVX2_CMAKE_ 1)
endif()
# We assume AVX2 support when CUDA is enabled
set(CMAKE_CXX_FLAGS "/arch:AVX2" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "/arch:AVX2" CACHE STRING "" FORCE)
message(STATUS "Setting x86_64 optimisations: AVX2")