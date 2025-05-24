#pragma once

#include "CudaCommon.cuh"
#include "LoadMeshObj.cuh"

struct Triangle {
    float4 v0, v1, v2;
};

class BroadPhaseGrid {
public:
    BroadPhaseGrid(const std::string &meshPath,
                   int cellsX, int cellsY, int cellsZ,
                   float arenaX, float arenaY, float arenaZ);

    template <typename Func>
    __device__ void forEachTriangle(float4 aabbMin, float4 aabbMax, Func&& func) const;

private: 
    int m_numCellsX;
    int m_numCellsY;
    int m_numCellsZ;

    // Grid extents
    float4 m_gridMinCorner;
    float4 m_gridMaxCorner;
    float4 m_gridFullExtent;

    // Pre-compute inverse cell size
    float4 m_invCellSize;

    // Triangle mesh data (CSR format)
    const float4* __restrict__ m_vertices;
    const int4*   __restrict__ m_triangles;
    const int*    __restrict__ m_cellOffsets;
    const int*    __restrict__ m_triIndices;

    // Host function for computing per-cell triangle indices
    void buildSpatialGrid(const std::vector<float4>& vertices,
                          const std::vector<int4>& triangles,
                          std::vector<int>& cellOffsets,
                          std::vector<int>& triIndices);

    // Cell indexing helpers
    __host__ __device__ inline int4 worldToCell(float4 point) const;
    __host__ __device__ inline int flattenIndex(int x, int y, int z) const {
        return x * (m_numCellsY * m_numCellsZ) + y * m_numCellsZ + z;
    }
};
