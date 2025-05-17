#include "BroadPhaseGrid.cuh"

__device__ TriangleIndexRange BroadPhaseGrid::getTriangles(int x, int y, int z) const {
    int idx = cellIndex(x, y, z);
    int start = m_cellOffsets[idx];
    int end = m_cellOffsets[idx + 1];
    return { m_triangleIndices + start, m_triangleIndices + end };
}