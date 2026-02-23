cmake_minimum_required(VERSION 3.16)

project(guc_ls
  VERSION 0.1.0
  DESCRIPTION "Unitary-constraint least squares solvers (unit-norm and Stiefel) with polynomial elimination"
  LANGUAGES CXX
)

option(GUC_LS_FETCH_EIGEN "Fetch Eigen3 automatically if not found" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Eigen dependency ----
find_package(Eigen3 3.3 QUIET NO_MODULE)

if (NOT Eigen3_FOUND)
  if (GUC_LS_FETCH_EIGEN)
    include(FetchContent)
    message(STATUS "Eigen3 not found; fetching Eigen3 (this requires internet access).")
    FetchContent_Declare(
      eigen
      URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    )
    FetchContent_MakeAvailable(eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE "${eigen_SOURCE_DIR}")
  else()
    message(FATAL_ERROR "Eigen3 not found. Install Eigen3 or set GUC_LS_FETCH_EIGEN=ON.")
  endif()
endif()

# ---- Library ----
add_library(guc_ls
  src/roots.cpp
  src/unit_norm_ls.cpp
  src/procrustes.cpp
  src/stiefel_ls.cpp
)

target_include_directories(guc_ls PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(guc_ls PUBLIC Eigen3::Eigen)

target_compile_features(guc_ls PUBLIC cxx_std_17)

## ---- Compile options (requested): -w -O3 for all targets ----
# Notes:
#  * -w disables warnings.
#  * -O3 enables aggressive optimization.
#  * Provide MSVC equivalents for portability.
if (MSVC)
  set(GUC_LS_COMPILE_OPTS /w /O2)
else()
  set(GUC_LS_COMPILE_OPTS -w -O3)
endif()

target_compile_options(guc_ls PRIVATE ${GUC_LS_COMPILE_OPTS})

# ---- Apps ----
add_executable(example_unitnorm apps/example_unitnorm.cpp)
target_link_libraries(example_unitnorm PRIVATE guc_ls)

add_executable(example_procrustes apps/example_procrustes.cpp)
target_link_libraries(example_procrustes PRIVATE guc_ls)

add_executable(example_stiefel apps/example_stiefel.cpp)
target_link_libraries(example_stiefel PRIVATE guc_ls)

add_executable(benchmark_noise_sweep apps/benchmark_noise_sweep.cpp)
target_link_libraries(benchmark_noise_sweep PRIVATE guc_ls)

# Apply the requested compile options to all executables as well.
foreach(_tgt example_unitnorm example_procrustes example_stiefel benchmark_noise_sweep)
  target_compile_options(${_tgt} PRIVATE ${GUC_LS_COMPILE_OPTS})
endforeach()

# ---- Install (optional) ----
include(GNUInstallDirs)
install(TARGETS guc_ls
  EXPORT guc_lsTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
