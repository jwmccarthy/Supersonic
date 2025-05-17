#pragma once

#include <vector>
#include <fstream>

#include "CudaCommon.cuh"

struct ArenaMesh {
    int numVertices; 
    int numTriangles;

    // Mesh info w/ keywords for LDG
    const int3*   __restrict__ triangles;
    const float3* __restrict__ vertices;

    static ArenaMesh* createOnDevice(const std::string& path);
};