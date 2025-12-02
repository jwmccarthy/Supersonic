cmake_minimum_required(VERSION 3.20)
project(supersonic LANGUAGES CXX CUDA)

# clangd / tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Plugin that provides ReflectionPlugin
add_subdirectory(plugin)

# CUDA bits
find_package(CUDAToolkit REQUIRED)

# Sources/headers (auto-refresh on re-configure)
file(GLOB_RECURSE PROJECT_HEADERS CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/include/*.hpp")
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/src/*.c*")

# Generated reflection header
set(REFLECTION_HEADER "${CMAKE_BINARY_DIR}/reflection.gen.hpp")
add_custom_command(
  OUTPUT "${REFLECTION_HEADER}"
  COMMAND clang++ -std=c++17
          -Wno-pragma-once-outside-header -Wno-attributes
          -fplugin=$<TARGET_FILE:ReflectionPlugin>
          -fsyntax-only -x c++
          -I"${CMAKE_SOURCE_DIR}/include"
          -I"${CUDAToolkit_INCLUDE_DIRS}"
          ${PROJECT_HEADERS}
  DEPENDS ReflectionPlugin ${PROJECT_HEADERS}
  COMMENT "Generating reflection data"
)
add_custom_target(reflection DEPENDS "${REFLECTION_HEADER}")

# Main executable
add_executable(supersonic
  test.cpp
  ${PROJECT_SOURCES}
)
add_dependencies(supersonic reflection)

target_include_directories(supersonic PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}"
  "${CUDAToolkit_INCLUDE_DIRS}"
)

set_target_properties(supersonic PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# Disable warnings
target_compile_options(supersonic PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-include ${REFLECTION_HEADER}>
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-attributes>
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-attributes>
)

target_link_libraries(supersonic PRIVATE CUDA::cudart)