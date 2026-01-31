FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    clang \
    llvm-dev \
    libclang-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source files
COPY . .

# Build the project
RUN mkdir -p build && cd build && \
    cmake .. && \
    cmake --build . -j$(nproc)

# Default command runs the profiler
CMD ["./build/supersonic"]
