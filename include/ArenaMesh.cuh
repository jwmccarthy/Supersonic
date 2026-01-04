#pragma once

#include <vector>

#define MESH_PATH "./assets/pitch.obj"

constexpr int3 GRID_DIMS = { 24, 36, 8 };

struct ArenaMesh
{
    // Mesh dims
    int nTris;
    int nVerts;
    int nCells;

    // Device mesh arrays
    __restrict__ float4* tris;
    __restrict__ int4* verts;

    // Broadphase grid
    __restrict__ int* triIdx;
    __restrict__ int* triPre;

    ArenaMesh(const char* path);
    ~ArenaMesh();

    void loadMeshObj(std::vector<float4>& verts, std::vector<int4>& tris);
}