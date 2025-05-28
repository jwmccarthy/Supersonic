cmake_minimum_required(VERSION 3.18)
project(supersonic LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/torch")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-rdc=true")

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# Find Python using the conda environment path
set(PYTHON_INCLUDE_DIR "/root/miniconda3/envs/rlenv/include/python3.11")
set(PYTHON_LIBRARY "/root/miniconda3/envs/rlenv/lib/libpython3.11.so")

# Common source files (excluding test files)
file(GLOB COMMON_SOURCES 
    "src/AABB.cu"
    "src/ArenaMesh.cu"
    "src/BroadPhaseGrid.cu"
    "src/GameStateDevice.cu"
    "src/InitializerKernels.cu"
    "src/LoadMeshObj.cu"
    "src/PhysicsKernels.cu"
    "src/RLEnvironment.cu"
    "src/StateReset.cu"
)

# BroadPhaseGrid Test executable
add_executable(grid_test ${COMMON_SOURCES} "src/TestBroadPhaseGrid.cu")
target_include_directories(grid_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
)
target_link_libraries(grid_test PRIVATE 
    ${TORCH_LIBRARIES}
    ${PYTHON_LIBRARY}
)
set_target_properties(grid_test PROPERTIES 
    OUTPUT_NAME "grid_test"
    CUDA_SEPARABLE_COMPILATION ON
)
