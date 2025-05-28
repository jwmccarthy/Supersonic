#include <numeric>
#include <algorithm>

#include "CudaMath.cuh"
#include "BroadPhaseGrid.cuh"

BroadPhaseGrid::BroadPhaseGrid(
    const std::string &meshPath,
    int cellsX, int cellsY, int cellsZ,
    float arenaX, float arenaY, float arenaZ
)
:   m_numCellsX(cellsX),
    m_numCellsY(cellsY),
    m_numCellsZ(cellsZ)
{
    // Min/max corner of arena extent
    m_gridMinCorner = make_float4(-arenaX, -arenaY, 0,      0);
    m_gridMaxCorner = make_float4( arenaX,  arenaY, arenaZ, 0);
    m_gridFullExtent = m_gridMaxCorner - m_gridMinCorner;

    // Pre-compute cell size inverse
    m_invCellSize = make_float4(m_numCellsX / m_gridFullExtent.x,
                                m_numCellsY / m_gridFullExtent.y,
                                m_numCellsZ / m_gridFullExtent.z, 0);

    // Read mesh data
    std::vector<float4> vertices;
    std::vector<int4>   triangles;
    loadMeshObj(meshPath, vertices, triangles);

    // Assign triangles to grid locations and construct offsets
    std::vector<int> cellOffsets;
    std::vector<int> triIndices;
    buildSpatialGrid(vertices, triangles, cellOffsets, triIndices);

    // Device pointers
    float4* d_vertices;
    int4*   d_triangles;
    int*    d_cellOffsets;
    int*    d_triIndices;

    size_t verticesSize = sizeof(float4) * vertices.size();
    size_t trianglesSize = sizeof(int4) * triangles.size();
    size_t cellOffsetsSize = sizeof(int) * cellOffsets.size();
    size_t triIndicesSize = sizeof(int) * triIndices.size();

    // Allocate device memory for triangle data
    CUDA_CHECK(cudaMalloc(&d_vertices,    verticesSize));
    CUDA_CHECK(cudaMalloc(&d_triangles,   trianglesSize));
    CUDA_CHECK(cudaMalloc(&d_cellOffsets, cellOffsetsSize));
    CUDA_CHECK(cudaMalloc(&d_triIndices,  triIndicesSize));

    // Move triangle information to device
    CUDA_CHECK(cudaMemcpy(d_vertices,    vertices.data(),    verticesSize,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_triangles,   triangles.data(),   trianglesSize,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cellOffsets, cellOffsets.data(), cellOffsetsSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_triIndices,  triIndices.data(),  triIndicesSize,  cudaMemcpyHostToDevice));

    // Assign to constant pointers
    m_vertices = d_vertices;
    m_triangles = d_triangles;
    m_cellOffsets = d_cellOffsets;
    m_triIndices = d_triIndices;
}

void BroadPhaseGrid::buildSpatialGrid(
    const std::vector<float4>& vertices,
    const std::vector<int4>& triangles,
    std::vector<int>& cellOffsets,
    std::vector<int>& triIndices
) {
    int totalCells = m_numCellsX * m_numCellsY * m_numCellsZ;

    // Collect cell-triangle pairs
    std::vector<std::pair<int, int>> cellTriPairs;

    // Add triangles to cell lists
    for (int triIdx = 0; triIdx < triangles.size(); ++triIdx) {
        const int4& tri = triangles[triIdx];

        // Triangle vertices
        float4 v0 = vertices[tri.x];
        float4 v1 = vertices[tri.y];
        float4 v2 = vertices[tri.z];
        
        // Get triangle AABB
        float4 triMin = fminf(fminf(v0, v1), v2);
        float4 triMax = fmaxf(fmaxf(v0, v1), v2);

        // Convert points to cell indices
        int4 minCell = worldToCell(triMin);
        int4 maxCell = worldToCell(triMax);

        // Accumulate triangle indices for overlapped cells
        for (int x = minCell.x; x <= maxCell.x; x++)
        for (int y = minCell.y; y <= maxCell.y; y++)
        for (int z = minCell.z; z <= maxCell.z; z++)
        {
            int cellIdx = flattenIndex(x, y, z);
            cellTriPairs.push_back(std::make_pair(cellIdx, triIdx));
        }
    }

    // Sort by cell index
    std::sort(cellTriPairs.begin(), cellTriPairs.end());

    // Build CSR format for triangle access by cell
    cellOffsets.assign(totalCells + 1, 0);
    triIndices.resize(cellTriPairs.size());

    for (size_t i = 0; i < cellTriPairs.size(); i++) {
        auto pair = cellTriPairs[i];
        cellOffsets[pair.first + 1]++;
        triIndices[i] = pair.second;
    }

    // Convert counts to offsets
    std::partial_sum(cellOffsets.begin(), cellOffsets.end(), cellOffsets.begin());
}