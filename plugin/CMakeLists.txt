cmake_minimum_required(VERSION 3.20)
project(ReflectionPlugin)

# Find LLVM and Clang
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

# C++ standard
set(CMAKE_CXX_STANDARD 17)

# LLVM setup
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})

# Build the plugin
add_library(ReflectionPlugin MODULE plugin.cpp)

# Add dependency tracking to avoid unnecessary rebuilds
set_target_properties(ReflectionPlugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugin"
)

# Link libraries
target_link_libraries(ReflectionPlugin PRIVATE
    clangAST
    clangBasic  
    clangFrontend
)