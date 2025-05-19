#pragma once

#include "CudaCommon.cuh"

struct Triangle {
    float3 v0, v1, v2;
};

class BroadPhaseGrid {
public:
    BroadPhaseGrid();

    template <typename Func>
    __device__ void forEachTriangle(float3 aabbMin, float3 aabbMax, Func&& func) const;

private: 
    int m_numCellsX, m_numCellsY, m_numCellsZ;

    float3 m_gridMinCorner;
    float3 m_cellDimensions;

    // Triangle mesh data (CSR format)
    const float3* __restrict__ m_vertices        {nullptr};
    const int3*   __restrict__ m_triangles       {nullptr};
    const int*    __restrict__ m_cellOffsets     {nullptr};
    const int*    __restrict__ m_triangleIndices {nullptr};

    void readMeshData(const std::string &meshPath);

    __device__ int3 worldToCell(float3 point) const;

    __device__ inline int flattenIndex(int x, int y, int z) const {
        return x * (m_numCellsY * m_numCellsZ) + y * m_numCellsZ + z;
    }
};
