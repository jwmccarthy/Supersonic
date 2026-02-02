#include <cuda_runtime.h>

#include "CudaCommon.cuh"
#include "CudaKernels.cuh"
#include "StructAllocate.cuh"
#include "ArenaMesh.cuh"
#include "CarArenaCollision.cuh"
#include "RLEnvironment.cuh"

RLEnvironment::RLEnvironment(int sims, int numB, int numO, int seed)
    : sims(sims), numB(numB), numO(numO), seed(seed)
    , cars(sims * (numB + numO))
    , m_arena(MESH_PATH)
    , m_state(sims, numB, numO, seed)
{
    // Copy arena mesh and game state to device
    cudaMallocCpy(d_arena, &m_arena);
    cudaMallocCpy(d_state, &m_state);

    // Allocate intermediate workspaces
    cudaMallocSOA(m_space, cars * MAX_PER_CAR);
    cudaMallocCpy(d_space, &m_space);

    // Allocate output buffer
    CUDA_CHECK(cudaMalloc(&d_output, sims * sizeof(float)));

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&m_stream));

    // Capture step() operations into a graph
    int blockSize = 128;
    int broadGrid = (cars + blockSize - 1) / blockSize;
    int narrowGrid = (cars * MAX_PER_CAR + blockSize - 1) / blockSize;

    CUDA_CHECK(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal));

    cudaMemsetAsync(&d_space->count, 0, sizeof(int), m_stream);
    carArenaBroadPhaseKernel<<<broadGrid, blockSize, 0, m_stream>>>(d_state, d_arena, d_space);
    carArenaNarrowPhaseKernel<<<narrowGrid, blockSize, 0, m_stream>>>(d_state, d_arena, d_space);

    CUDA_CHECK(cudaStreamEndCapture(m_stream, &m_stepGraph));
    CUDA_CHECK(cudaGraphInstantiate(&m_stepGraphExec, m_stepGraph, nullptr, nullptr, 0));
}

RLEnvironment::~RLEnvironment()
{
    cudaGraphExecDestroy(m_stepGraphExec);
    cudaGraphDestroy(m_stepGraph);
    cudaStreamDestroy(m_stream);
}

float* RLEnvironment::step()
{
    CUDA_CHECK(cudaGraphLaunch(m_stepGraphExec, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    return d_output;
}

float* RLEnvironment::reset()
{
    int blockSize = 32;
    int gridSize = (sims + blockSize - 1) / blockSize;

    resetKernel<<<gridSize, blockSize>>>(d_state);
    cudaDeviceSynchronize();

    return d_output;
}