#pragma once

#include "CudaCommon.cuh"
#include "ReadMeshObj.cuh"

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

    float4 m_gridMinCorner;
    float4 m_gridExtents;

    // Triangle mesh data (CSR format)
    const float4* __restrict__ m_vertices        {nullptr};
    const int4*   __restrict__ m_triangles       {nullptr};
    const int*    __restrict__ m_cellOffsets     {nullptr};
    const int*    __restrict__ m_triangleIndices {nullptr};

    __device__ int4 worldToCell(float4 point) const;

    __device__ inline int flattenIndex(int x, int y, int z) const {
        return x * (m_numCellsY * m_numCellsZ) + y * m_numCellsZ + z;
    }
};
