#include <fstream>
#include <sstream>
#include <iostream>

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "CudaCommon.cuh"
#include "CudaMath.cuh"
#include "ArenaMesh.cuh"

float4 getTriNormal(float4 v1, float4 v2, float4 v3)
{
    // Get edge vectors
    float4 e1 = vec3::sub(v2, v1);
    float4 e2 = vec3::sub(v3, v1);

    return vec3::norm(vec3::cross(e1, e2));
}

Mesh ArenaMesh::loadMeshObj(const char* path)
{
    // File stream
    std::string line;
    std::ifstream file(path);

    if (!file.is_open())
    {
        std::cerr << "ERROR: Failed to open mesh file: " << path << "\n";
    }

    // Output vectors
    std::vector<float4> verts;
    std::vector<int4> tris;

    // Read in vertices and triangles
    while (std::getline(file, line))
    {
        char type;
        std::istringstream s(line);
        if (!(s >> type)) continue;

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

    // Initialize tri normal array
    std::vector<float4> norms(nTris);

    // Initialize pre-computed AABB
    std::vector<float4> aabbMin(nTris);
    std::vector<float4> aabbMax(nTris);

    return { verts, norms, tris, aabbMin, aabbMax };
}

Grid ArenaMesh::buildBroadphaseGrid(Mesh& m)
{
    // Number of overlapping 2x2x2 groups in grid
    nGroups = (int)vec3::prod(GROUP_DIMS);

    // Tri accumulators for group cells
    std::vector<std::vector<int>> groups(nGroups);

    for (int i = 0; i < m.tris.size(); ++i)
    {
        int4 tri = m.tris[i];

        // Get vertices via index
        float4 v0 = m.verts[tri.x],
               v1 = m.verts[tri.y],
               v2 = m.verts[tri.z];

        // Get cell index for each
        int3 cX = getCellIdx(v0);
        int3 cY = getCellIdx(v1);
        int3 cZ = getCellIdx(v2);

        // Find min and max index
        int3 lo = vec3::min(vec3::min(cX, cY), cZ);
        int3 hi = vec3::max(vec3::max(cX, cY), cZ);

        // Convert cell bounds to overlapping 2x2x2 group bounds
        int3 gLo = max({0, 0, 0}, vec3::sub(lo, 1));
        int3 gHi = min(hi, vec3::sub(GROUP_DIMS, 1));

        // Iterate over potential groups
        for (int x = gLo.x; x <= gHi.x; ++x)
        for (int y = gLo.y; y <= gHi.y; ++y)
        for (int z = gLo.z; z <= gHi.z; ++z)
        {
            groups[flatGroupIdx(x, y, z)].push_back(i);
        }

        // Pre-compute triangle normals
        m.norms[i] = getTriNormal(v0, v1, v2);

        // Pre-compute triangle AABBs
        m.aabbMin[i] = vec3::min(vec3::min(v0, v1), v2);
        m.aabbMax[i] = vec3::max(vec3::max(v0, v1), v2);
    }

    // 1D grid storage via prefix sum
    std::vector<int> triPre(nGroups + 1, 0);
    for (int i = 0; i < nGroups; ++i)
    {
        triPre[i + 1] = triPre[i] + groups[i].size();
    }

    // Construct triangle indices
    std::vector<int> triIdx(triPre.back());
    for (int i = 0; i < nGroups; ++i)
    {
        std::copy(groups[i].begin(), groups[i].end(), triIdx.begin() + triPre[i]);
    }

    return { triIdx, triPre };
}

ArenaMesh::ArenaMesh(const char* path)
{
    // Construct broadphase grid
    Mesh m = loadMeshObj(path);
    Grid g = buildBroadphaseGrid(m);

    printMeshInfo(path, g);

    // Allocate/copy mesh array pointers
    cudaMallocCpy(verts, m.verts.data(), m.verts.size());
    cudaMallocCpy(norms, m.norms.data(), m.norms.size());
    cudaMallocCpy(tris,   m.tris.data(), m.tris.size());

    // Allocate/copy triangle AABB pointers
    cudaMallocCpy(aabbMin, m.aabbMin.data(), m.aabbMin.size());
    cudaMallocCpy(aabbMax, m.aabbMax.data(), m.aabbMax.size());

    // Allocate/copy grid array pointers
    cudaMallocCpy(triIdx, g.triIdx.data(), g.triIdx.size());
    cudaMallocCpy(triPre, g.triPre.data(), g.triPre.size());
}

void ArenaMesh::printMeshInfo(const char* path, const Grid& g)
{
    // Debug: verify mesh loaded
    std::cout << "Loaded mesh: " << path << "\n";
    std::cout << "  Vertices:  " << nVerts << "\n";
    std::cout << "  Triangles: " << nTris << "\n";
    std::cout << "  Grid groups: " << nGroups << " ("
              << GROUP_DIMS.x << "x" 
              << GROUP_DIMS.y << "x" 
              << GROUP_DIMS.z << ")\n";
    std::cout << "  Grid refs:  " << g.triIdx.size() << "\n";
}
