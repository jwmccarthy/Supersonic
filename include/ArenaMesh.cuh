#pragma once

#include <vector>
#include <cuda_runtime.h>

#include "CudaMath.cuh"
#include "RLConstants.cuh"

#define MESH_PATH "./assets/pitch.obj"

struct Triangle
{
    float4 v0;
    float4 v1;
    float4 v2;
};

struct Mesh
{
    // Mesh
    std::vector<float4> verts;
    std::vector<float4> norms;
    std::vector<int4>   tris;

    // Bounding boxes
    std::vector<float4> aabbMin;
    std::vector<float4> aabbMax;
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
    int nGroups;

    // Cell size
    float4 cellEx;

    // Device mesh arrays
    float4* verts;
    float4* norms;
    int4*   tris;

    // Tri bounding boxes
    float4* aabbMin;
    float4* aabbMax;

    // Broadphase grid
    int* triIdx;
    int* triPre;

    ArenaMesh(const char* path);

    // Mesh & grid construction
    Mesh loadMeshObj(const char* path);
    Grid buildBroadphaseGrid(Mesh& m);
    void printMeshInfo(const char* path, const Grid& g);

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

    __host__ __device__ __forceinline__ int3 getGroupIdx(int3 cell)
    {
        int x = max(0, min(cell.x, GROUP_DIMS.x - 1));
        int y = max(0, min(cell.y, GROUP_DIMS.y - 1));
        int z = max(0, min(cell.z, GROUP_DIMS.z - 1));
        return { x, y, z };
    }

    __host__ __device__ __forceinline__ int flatGroupIdx(int x, int y, int z)
    {
        return x + y * GROUP_DIMS.x + z * GROUP_DIMS.x * GROUP_DIMS.y;
    }
};

__host__ __device__ float4 getTriNormal(float4 v0, float4 v1, float4 v2);
