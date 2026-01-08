#include <fstream>
#include <sstream>
#include <iostream>

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "CudaCommon.cuh"
#include "CudaMath.cuh"
#include "ArenaMesh.cuh"

Mesh ArenaMesh::loadMeshObj(const char* path)
{
    // File stream
    std::string line;
    std::ifstream file(path);

    // Output vectors
    std::vector<float4> verts;
    std::vector<int4> tris;

    // Read in vertices and triangles
    while (std::getline(file, line))
    {
        std::istringstream s(line);
        char type;
        s >> type;

        if (type == 'v')
        {
            // Vertex world locations
            float x, y, z;
            s >> x >> y >> z;
            verts.push_back({ x, y, z, 0 });
        }
        else if (type == 'f')
        {
            // Triangle vertex indices
            int x, y, z;
            s >> x >> y >> z;
            tris.push_back({ x - 1, y - 1, z - 1, 0 });
        }
    }

    // Set mesh dimensions
    nVerts = verts.size();
    nTris = tris.size();

    return { verts, tris };
}

Grid ArenaMesh::buildBroadphaseGrid(Mesh m)
{
    // Number of cells in grid
    nCells = (int)vec3::prod(GRID_DIMS);

    // Tri accumulators for grid cells
    std::vector<std::vector<int>> cells(nCells);

    for (int i = 0; i < m.tris.size(); ++i)
    {
        int4 tri = m.tris[i];

        // Get cell index for each
        int3 cX = getCellIdx(m.verts[tri.x]);
        int3 cY = getCellIdx(m.verts[tri.y]);
        int3 cZ = getCellIdx(m.verts[tri.z]);

        // Find min and max index
        int3 lo = vec3::min(vec3::min(cX, cY), cZ);
        int3 hi = vec3::max(vec3::max(cX, cY), cZ);

        // Iterate over potential cells
        for (int x = lo.x; x <= hi.x; ++x)
        for (int y = lo.y; y <= hi.y; ++y)
        for (int z = lo.z; z <= hi.z; ++z)
        {
            cells[flatCellIdx(x, y, z)].push_back(i);
        }
    }

    // 1D grid storage via prefix sum
    std::vector<int> triPre(nCells + 1, 0);
    for (int i = 0; i < nCells; ++i)
    {
        std::cout << cells[i].size() << std::endl;
        triPre[i + 1] = triPre[i] + cells[i].size();
    }

    // Construct triangle indices
    std::vector<int> triIdx(triPre.back());
    for (auto& c : cells)
    {
        triIdx.insert(triIdx.end(), c.begin(), c.end());
    }
    
    return { triIdx, triPre };
}

ArenaMesh::ArenaMesh(const char* path)
{
    // Construct broadphase grid
    Mesh m = loadMeshObj(path);
    Grid g = buildBroadphaseGrid(m);

    // Allocate/copy mesh array pointers
    cudaMallocCpy(verts, m.verts.data(), m.verts.size());
    cudaMallocCpy(tris, m.tris.data(), m.tris.size());

    // Allocate/copy grid array pointers
    cudaMallocCpy(triIdx, g.triIdx.data(), g.triIdx.size());
    cudaMallocCpy(triPre, g.triPre.data(), g.triPre.size());
}