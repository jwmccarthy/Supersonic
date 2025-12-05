#include "Collisions.hpp"
#include "CollisionsSAT.hpp"
#include "CollisionsEdge.hpp"
#include "CollisionsFace.hpp"

// Collision manifold generation
__device__ void carCarCollision(float4 posA, float4 rotA, float4 posB, float4 rotB)
{
    // Build SAT context
    SATContext ctx = buildSATContext(posA, rotA, posB, rotB);

    // Get SAT result (axis, depth)
    SATResult res = carCarSATTest(ctx);

    // Check for face-face or edge-edge
    if (res.axisIdx < 6)
    {
        // Face-face collision
        generateFaceFaceManifold(ctx, res);
    }
    else
    {
        // Edge-edge collision
        generateEdgeEdgeManifold(ctx, res);
    }
}
