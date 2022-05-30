if(DEFINED IMAGUS_FLAGS_AVX_CMAKE_)
  return()
else()
  set(IMAGUS_FLAGS_AVX_CMAKE_ 1)
endif()
# We assume AVX support at minimum
set(CMAKE_CXX_FLAGS "/arch:AVX" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "/arch:AVX" CACHE STRING "" FORCE)
message(STATUS "Setting x86_64 optimisations: AVX")