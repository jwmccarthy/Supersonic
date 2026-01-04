include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Math tests
add_executable(test_math test_math.cu)

target_include_directories(test_math PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(test_math PRIVATE
    GTest::gtest_main
    CUDA::cudart
)

set_target_properties(test_math PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(test_math PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

include(GoogleTest)
gtest_discover_tests(test_math)
