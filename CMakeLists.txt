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

pybind11_add_module(supersonic
    pybind/Supersonic.cpp
    ${SOURCES}
)

target_include_directories(supersonic 
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/bullet
)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(supersonic PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
