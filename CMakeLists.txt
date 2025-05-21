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

file(GLOB SOURCES "src/*.cu" "src/*.cpp")

# # Python module
# pybind11_add_module(supersonic
#     # pybind/Supersonic.cpp
#     ${SOURCES}
# )

# target_include_directories(supersonic 
#     PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
# )

# find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
# target_link_libraries(supersonic PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

# Find Python using the conda environment path
set(PYTHON_INCLUDE_DIR "/root/miniconda3/envs/rlenv/include/python3.11")
set(PYTHON_LIBRARY "/root/miniconda3/envs/rlenv/lib/libpython3.11.so")

# Main Test executable
add_executable(mesh_test ${SOURCES})
target_include_directories(mesh_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
)
target_link_libraries(mesh_test PRIVATE 
    ${TORCH_LIBRARIES}
    ${PYTHON_LIBRARY}
)
set_target_properties(mesh_test PROPERTIES 
    OUTPUT_NAME "test"
    CUDA_SEPARABLE_COMPILATION ON
)
