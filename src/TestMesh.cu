#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include "LoadMeshObj.cuh"
#include "CudaCommon.cuh"

int main()
{
    // 1. Emit a tiny OBJ file
    const std::string objPath = "mini.obj";
    {
        std::ofstream out(objPath);
        out << "v 1 0 0\n"
            << "v 0 1 0\n"
            << "v 0 0 1\n"
            << "f 1 2 3\n";
    }

    // 2. Load
    std::vector<float4> verts;
    std::vector<int4>   tris;
    bool ok = loadMeshObj(objPath, verts, tris);
    assert(ok && "Failed to load OBJ");

    // 3. Checks
    assert(verts.size() == 3);
    assert(tris.size()  == 1);

    // 3a. vertex coords
    assert(verts[0].x == 1.f && verts[0].y == 0.f && verts[0].z == 0.f);
    assert(verts[1].x == 0.f && verts[1].y == 1.f && verts[1].z == 0.f);
    assert(verts[2].x == 0.f && verts[2].y == 0.f && verts[2].z == 1.f);

    // 3b. triangle indices
    assert(tris[0].x == 0 && tris[0].y == 1 && tris[0].z == 2);

    // 4. Success
    std::cout << "OBJ loader test passed ✓\n";
    return 0;
}
