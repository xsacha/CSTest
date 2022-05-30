if(DEFINED IMAGUS_GCC_NATIVE_CMAKE_)
  return()
else()
  set(IMAGUS_GCC_NATIVE_CMAKE_ 1)
endif()

get_filename_component(HUNTER_INSTALL_TAG "${CMAKE_CURRENT_LIST_FILE}" NAME_WE)

find_program(CMAKE_C_COMPILER gcc)
find_program(CMAKE_CXX_COMPILER g++)

if(NOT CMAKE_C_COMPILER)
  message(FATAL_ERROR "gcc not found")
endif()

if(NOT CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "g++ not found")
endif()

set(CMAKE_C_COMPILER "${CMAKE_C_COMPILER}" CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "${CMAKE_CXX_COMPILER}" CACHE STRING "C++ compiler" FORCE)

# Arch-specific optimisations
execute_process(COMMAND uname -m OUTPUT_VARIABLE _output OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_CXX_FLAGS "-fPIC -fuse-ld=gold -fconcepts ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "-fPIC -fuse-ld=gold ${CMAKE_C_FLAGS}")

if(_output MATCHES "amd64.*|x86_64.*|AMD64.*")
  # NIST supports AVX2/AVX512
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma -mpopcnt -mcx16 -msahf -mavx -mavx2" CACHE STRING "" FORCE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma -mpopcnt -mcx16 -msahf -mavx -mavx2" CACHE STRING "" FORCE)
  message(STATUS "Setting x86_64 optimisations: AVX2")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" CACHE STRING "" FORCE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native" CACHE STRING "" FORCE)
  message(STATUS "Setting ${_output} optimisations: Native")
endif()

