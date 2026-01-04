#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>

#include "ArenaMesh.cuh"
#include "CudaCommon.hpp"

void ArenaMesh::loadMeshObj(std::vector<float4>& verts, std::vector<int4>& tris)
{
    // File stream
    std::string line;
    std::ifstream file(MESH_PATH);

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
            m_tris.push_back(make_float4(x, y, z, 0));
        }
        else if (type == 'f')
        {
            // Triangle vertex indices
            int x, y, z;
            s >> x >> y >> z;
            m_verts.push_back(make_int4(x - 1, y - 1, z - 1, 0));
        }
    }
}

ArenaMesh::ArenaMesh()
{
    // Mesh data accumulators
    std::vector<float4> h_verts;
    std::vector<int4>   h_tris;
    loadMeshObj(h_verts, h_tris);

    // Mesh dims
    nVerts = h_verts.size();
    nTris  = h_tris.size();

    // Build broadphase grid
    buildBroadphaseGrid(h_verts, h_tris);

    // Copy mesh data to device
    cudaMallocCpy(verts, h_verts.data(), nVerts);
    cudaMallocCpy(tris, h_tris.data(), nTris);
}