#include "LoadMeshObj.cuh"

bool loadMeshObj(const std::string &path, 
    std::vector<float4> &vertices, 
    std::vector<int4> &triangles) 
{
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Cannot open OBJ file: " << path << '\n';
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // Read vertex line
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (std::sscanf(line.c_str() + 2, "%f %f %f", &x, &y, &z) == 3) {
                vertices.push_back(float4{ y, x, z });
            }
        }

        // Read triangle index line (index at 1)
        else if (line[0] == 'f' && line[1] == ' ') {
            int i, j, k;
            if (std::sscanf(line.c_str() + 2, "%d %d %d", &i, &j, &k) == 3) {
                triangles.push_back(int4{ j-1, i-1, k-1 });
            }
        }
    }

    return true;
}