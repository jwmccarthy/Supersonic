#include "Math.cuh"
#include "BroadPhaseGrid.cuh"

BroadPhaseGrid::BroadPhaseGrid(const std::string &meshPath,
                               int cellsX, int cellsY, int cellsZ,
                               float arenaX, float arenaY, float arenaZ)
:   m_numCellsX(cellsX),
    m_numCellsY(cellsY),
    m_numCellsZ(cellsZ)
{
    // Minimum corner of arena extent
    m_gridMinCorner = make_float4(-arenaX, 
                                  -arenaY, 
                                  -arenaZ, 0);

    // Full arena extents (given half-extents)
    m_gridExtents = make_float4(arenaX * 2,
                                arenaY * 2,
                                arenaZ * 2, 0);

    // Read mesh data
    std::vector<float4> vertices;
    std::vector<int4>   triangles;
    loadMeshObj(meshPath, vertices, triangles);

    // 
}

__device__ int4 BroadPhaseGrid::worldToCell(float4 point) const {
    int x = static_cast<int>((point.x - m_gridMinCorner.x) / m_gridExtents.x);
    int y = static_cast<int>((point.y - m_gridMinCorner.y) / m_gridExtents.y);
    int z = static_cast<int>((point.z - m_gridMinCorner.z) / m_gridExtents.z);

    x = clamp(x, 0, m_numCellsX - 1);
    y = clamp(y, 0, m_numCellsY - 1);
    z = clamp(z, 0, m_numCellsZ - 1);

    return {x, y, z};
}

template <typename Func>
__device__ void BroadPhaseGrid::forEachTriangle(float4 aabbMin, float4 aabbMax, Func&& func) const  {
    int4 startCell = worldToCell(aabbMin);
    int4 endCell   = worldToCell(aabbMax);

    for (int cellX = startCell.x; cellX <= endCell.x; ++cellX)
    for (int cellY = startCell.y; cellY <= endCell.y; ++cellY)
    for (int cellZ = startCell.z; cellZ <= endCell.z; ++cellZ)
    {
        int cellIdx = flattenIndex(cellX, cellY, cellZ);

        int triangleStart = m_cellOffsets[cellIdx];
        int triangleEnd   = m_cellOffsets[cellIdx + 1];

        for (int i = triangleStart; i < triangleEnd; ++i) {
            int triIdx = m_triIndices[i];

            int4 vertexIdx = m_triangles[triIdx];

            Triangle triangle {
                m_vertices[vertexIdx.x],
                m_vertices[vertexIdx.y],
                m_vertices[vertexIdx.z]
            };

            func(triangle);
        }
    }
}