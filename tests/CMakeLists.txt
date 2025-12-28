# Tests CMakeLists.txt

# Find GTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Collision tests
add_executable(test_collisions
  test_collisions.cu
  ${CMAKE_SOURCE_DIR}/src/CollisionsSAT.cu
  ${CMAKE_SOURCE_DIR}/src/CollisionsEdge.cu
  ${CMAKE_SOURCE_DIR}/src/CollisionsFace.cu
)

target_include_directories(test_collisions PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(test_collisions PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(test_collisions PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

target_link_libraries(test_collisions PRIVATE
  GTest::gtest_main
  CUDA::cudart
)

gtest_discover_tests(test_collisions)

# Math utility tests
add_executable(test_math
  test_math.cu
)

target_include_directories(test_math PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(test_math PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

target_link_libraries(test_math PRIVATE
  GTest::gtest_main
  CUDA::cudart
)

gtest_discover_tests(test_math)

# Collision benchmark (not a gtest, standalone executable)
add_executable(benchmark_collision
  benchmark_collision.cu
  ${CMAKE_SOURCE_DIR}/src/CollisionsSAT.cu
  ${CMAKE_SOURCE_DIR}/src/CollisionsEdge.cu
  ${CMAKE_SOURCE_DIR}/src/CollisionsFace.cu
  ${CMAKE_SOURCE_DIR}/src/RLConstants.cu
)

target_include_directories(benchmark_collision PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(benchmark_collision PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(benchmark_collision PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

target_link_libraries(benchmark_collision PRIVATE
  CUDA::cudart
)

# Scenario benchmark (realistic RL positions)
add_executable(benchmark_scenarios
  benchmark_scenarios.cu
  ${CMAKE_SOURCE_DIR}/src/CollisionsSAT.cu
  ${CMAKE_SOURCE_DIR}/src/CollisionsEdge.cu
  ${CMAKE_SOURCE_DIR}/src/CollisionsFace.cu
  ${CMAKE_SOURCE_DIR}/src/CudaKernels.cu
  ${CMAKE_SOURCE_DIR}/src/StateReset.cu
  ${CMAKE_SOURCE_DIR}/src/RLConstants.cu
  ${CMAKE_SOURCE_DIR}/src/CudaUtils.cu
  ${CMAKE_SOURCE_DIR}/src/RLEnvironment.cu
  ${CMAKE_SOURCE_DIR}/src/GameState.cpp
)

add_dependencies(benchmark_scenarios reflection)

target_include_directories(benchmark_scenarios PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(benchmark_scenarios PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(benchmark_scenarios PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-include ${CMAKE_BINARY_DIR}/reflection.gen.hpp>
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-attributes>
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-attributes>
)

target_link_libraries(benchmark_scenarios PRIVATE
  CUDA::cudart
)
