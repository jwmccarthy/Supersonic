#include "GameState.hpp"
#include "StructAllocate.hpp"
#include "RLConstants.hpp"

GameState::GameState(int sims, int numB, int numO, int seed)
    : sims(sims), nCar(numB + numO),
      numB(numB), numO(numO), seed(seed)
{
    cudaMallocSOA(ball, sims);
    cudaMallocSOA(cars, sims * (numB + numO));
    cudaMallocSOA(pads, sims * NUM_BOOST_PADS);

    // Collision pairs: nCar * (nCar - 1) / 2 pairs per simulation
    int numPairs = nCar * (nCar - 1) / 2;
    cudaMallocSOA(cols, sims * numPairs);
}

GameState::~GameState()
{
    cudaFreeSOA(ball);
    cudaFreeSOA(cars);
    cudaFreeSOA(pads);
    cudaFreeSOA(cols);
}