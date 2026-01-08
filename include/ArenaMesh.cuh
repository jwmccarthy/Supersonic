#pragma once

#include <vector>
#include <cuda_runtime.h>

#include "CudaMath.cuh"
#include "RLConstants.cuh"

#define MESH_PATH "./assets/pitch.obj"

struct Mesh
{
    std::vector<float4> verts;
    std::vector<int4> tris;
};

struct Grid
{
    std::vector<int> triIdx;
    std::vector<int> triPre;
};

struct ArenaMesh
{
    // Mesh dims
    int nTris;
    int nVerts;
    int nCells;

    // Cell size
    float4 cellEx;

    // Device mesh arrays
    float4* verts;
    int4* tris;

    // Broadphase grid
    int* triIdx;
    int* triPre;

    ArenaMesh(const char* path);

    // Mesh & grid construction
    Mesh loadMeshObj(const char* path);
    Grid buildBroadphaseGrid(Mesh m);

    // Indexing helpers
    __host__ __device__ __forceinline__ int3 getCellIdx(float4 p)
    {
        // Convert world position to cell coordinates
        int x = (int)((p.x - ARENA_MIN.x) / CELL_SIZE.x);
        int y = (int)((p.y - ARENA_MIN.y) / CELL_SIZE.y);
        int z = (int)((p.z - ARENA_MIN.z) / CELL_SIZE.z);

        // Clamp to grid bounds
        x = max(0, min(x, GRID_DIMS.x - 1));
        y = max(0, min(y, GRID_DIMS.y - 1));
        z = max(0, min(z, GRID_DIMS.z - 1));

        return { x, y, z };
    }

    __host__ __device__ __forceinline__ int flatCellIdx(int x, int y, int z)
    {
        return x + y * GRID_DIMS.x + z * GRID_DIMS.x * GRID_DIMS.y;
    }
};