#include <iostream>
#include <vector>
#include <string>

#include "LoadMeshObj.cuh"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.obj>\n";
        return 1;
    }
    std::string path = argv[1];

    // 1. Load with your function
    std::vector<float4> vertices;
    std::vector<int4>   triangles;
    if (!loadMeshObj(path, vertices, triangles)) {
        std::cerr << "Load failed!\n";
        return 1;
    }

    // 2. Print vertices in OBJ format
    std::cout << "# " << vertices.size() << " vertices\n";
    for (const auto& v : vertices)
        std::cout << "v " << v.x << ' ' << v.y << ' ' << v.z << '\n';

    // 3. Print faces (add 1 to convert back to OBJ’s 1-based indices)
    std::cout << "\n# " << triangles.size() << " faces\n";
    for (const auto& t : triangles)
        std::cout << "f "
                  << t.x + 1 << ' '
                  << t.y + 1 << ' '
                  << t.z + 1 << '\n';

    return 0;
}
