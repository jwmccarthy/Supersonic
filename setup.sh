#!/bin/bash

# Install dependencies
apt-get update
apt-get install -y \
    build-essential \
    cmake \
    llvm \
    clang \
    libclang-dev

# Create build directory
mkdir -p build

# Run build
./build.sh
