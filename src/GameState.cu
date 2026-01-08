#include "GameState.cuh"
#include "RLConstants.cuh"
#include "StructAllocate.cuh"

GameState::GameState(int sims, int numB, int numO, int seed)
    : sims(sims), nCar(numB + numO), numB(numB), numO(numO), seed(seed)
{
    cudaMallocSOA(ball, sims);
    cudaMallocSOA(cars, sims * nCar);
    cudaMallocSOA(pads, sims * NUM_BOOST_PADS);
}

GameState::~GameState()
{
    cudaFreeSOA(ball);
    cudaFreeSOA(cars);
    cudaFreeSOA(pads);
}
