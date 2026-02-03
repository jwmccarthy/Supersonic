cmake_minimum_required(VERSION 3.20)
project(supersonic LANGUAGES CXX CUDA)

# CUDA setup
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "89;120")

# Reflection plugin
add_subdirectory(plugin)

# Generate reflection header
set(REFLECTION_HEADER "${CMAKE_BINARY_DIR}/reflection.gen.hpp")
set(REFLECT_HEADERS
  "${CMAKE_SOURCE_DIR}/include/GameState.cuh"
)
add_custom_command(
  OUTPUT "${REFLECTION_HEADER}"
  COMMAND clang++ -std=c++17
          -Wno-pragma-once-outside-header -Wno-attributes
          -fplugin=$<TARGET_FILE:ReflectionPlugin>
          -fsyntax-only -x c++
          -I"${CMAKE_SOURCE_DIR}/include"
          -I"${CUDAToolkit_INCLUDE_DIRS}"
          ${REFLECT_HEADERS}
  DEPENDS ReflectionPlugin ${REFLECT_HEADERS}
  COMMENT "Generating reflection"
)
add_custom_target(reflection DEPENDS "${REFLECTION_HEADER}")

# Sources
file(GLOB CUDA_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cu")

# Main executable
add_executable(supersonic profile.cu ${CUDA_SOURCES})
add_dependencies(supersonic reflection)

target_include_directories(supersonic PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}"
)

set_target_properties(supersonic PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(supersonic PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-include${REFLECTION_HEADER} -Wno-attributes>
  $<$<COMPILE_LANGUAGE:CUDA>:--pre-include=${REFLECTION_HEADER}>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-attributes>
)

target_link_libraries(supersonic PRIVATE CUDA::cudart)

# Enable dynamic parallelism (device runtime)
set_target_properties(supersonic PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
