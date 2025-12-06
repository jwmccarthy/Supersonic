#include "CudaMath.hpp"
#include "RLConstants.hpp"
#include "CollisionsFace.hpp"
#include "CollisionsSAT.hpp"

struct FaceVertices
{
    float4 p1;
    float4 p2;
    float4 p3;
    float4 p4;
};

// Face-face collision manifold generation
__device__ void generateFaceFaceManifold(SATContext& ctx, SATResult& res)
{
    // Get reference face

    // Get incident face (most negative dot product with reference normal)

    // Get incident face vertices

    // Project incident face vertices onto reference face plane

    // Clip projected quad against reference face edges

    // Transform clipped quad back to 3D space

    // Cull points to best 4

    // Output contact manifold
}