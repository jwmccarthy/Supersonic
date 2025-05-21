#pragma once

#include <vector>
#include <sstream>
#include <fstream>

#include "CudaCommon.cuh"

bool loadMeshObj(const std::string &path, 
                 std::vector<float4> &vertices, 
                 std::vector<int4> &triangles);