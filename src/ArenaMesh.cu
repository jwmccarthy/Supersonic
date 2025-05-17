#include "ArenaMesh.cuh"
#include "DeviceArray.cuh"

ArenaMesh* ArenaMesh::createOnDevice(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    // Read vertex and triangle counts
    ArenaMesh h_ArenaMesh;
    file.read(reinterpret_cast<char*>(&h_ArenaMesh.numTriangles), sizeof(h_ArenaMesh.numTriangles));
    file.read(reinterpret_cast<char*>(&h_ArenaMesh.numVertices), sizeof(h_ArenaMesh.numVertices));

    // Read mesh data
    std::vector<int3>   h_triangles(h_ArenaMesh.numTriangles);
    std::vector<float3> h_vertices(h_ArenaMesh.numVertices);
    file.read(reinterpret_cast<char*>(h_triangles.data()), sizeof(int3) * h_ArenaMesh.numTriangles);
    file.read(reinterpret_cast<char*>(h_vertices.data()), sizeof(float3) * h_ArenaMesh.numVertices);

    // Move triangle data to device
    DeviceArray<int3>   d_triangles(h_triangles);
    DeviceArray<float3> d_vertices(h_vertices);
    h_ArenaMesh.triangles = d_triangles.data();
    h_ArenaMesh.vertices  = d_vertices.data();

    // Move h_mesh to device
    ArenaMesh* d_ArenaMesh = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ArenaMesh, sizeof(ArenaMesh)));
    CUDA_CHECK(cudaMemcpy(d_ArenaMesh, &h_ArenaMesh, sizeof(ArenaMesh), cudaMemcpyHostToDevice));

    return d_ArenaMesh;
}