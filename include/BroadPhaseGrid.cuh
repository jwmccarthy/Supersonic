#pragma once

#include "ArenaMesh.cuh"
#include "CudaCommon.cuh"

struct TriangleIndexIterator {
    const int* ptr;

    __device__ int operator*() const { return *ptr; }
    __device__ TriangleIndexIterator& operator++() { ++ptr; return *this; }
    __device__ bool operator!=(const TriangleIndexIterator& other) const { return ptr != other.ptr; }
};

struct TriangleIndexRange {
    const int* begin_ptr;
    const int* end_ptr;

    __device__ TriangleIndexIterator begin() const { return {begin_ptr}; }
    __device__ TriangleIndexIterator end()   const { return {end_ptr}; }
};

class BroadPhaseGrid {
public:
    BroadPhaseGrid();

    __device__ void getAABBCellBounds(float3 &minAABB, float3 &maxAABB, int3& minCell, int3& maxCell) const;
    __device__ TriangleIndexRange getTriangles(int x, int y, int z) const;

private:
    int m_numVertices, m_numTriangles;
    int m_numCellX, m_numCellY, m_numCellZ;

    // Mesh information (LDG-enabled)
    const int3*   __restrict__ m_triangles;
    const float3* __restrict__ m_vertices;

    // Uniform grid information
    const int* __restrict__ m_cellOffsets;
    const int* __restrict__ m_triangleIndices;

    __device__ inline int cellIndex(int x, int y, int z) const {
        return (x * m_numCellY * m_numCellZ) + (y * m_numCellZ) + z;
    }
};