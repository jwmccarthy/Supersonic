#!/bin/bash
set -e

# Clone the repository via HTTPS
git clone https://github.com/jwmccarthy/Supersonic.git
cd Supersonic

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
