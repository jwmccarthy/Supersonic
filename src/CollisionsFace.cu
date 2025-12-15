#include "CudaMath.hpp"
#include "RLConstants.hpp"
#include "CollisionsFace.hpp"
#include "CollisionsSAT.hpp"

// Blend between two axes based on weight (branchless select)
__device__ float4 blendAxes(float4 axA, float4 axB, float b)
{
    return vec3::add(vec3::mult(axA, 1.0f - b), vec3::mult(axB, b));
}

// Find incident face axis (most parallel to reference normal)
__device__ int findIncidentAxis(float4 dir, const float4* axes, float& bestDot)
{
    int bestIdx = 0;
    float maxDot = 0.0f;
    bestDot = 0.0f;

    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        float d = vec3::dot(dir, axes[i]);
        float a = fabsf(d);
        if (a > maxDot)
        {
            maxDot = a;
            bestIdx = i;
            bestDot = d;
        }
    }

    return bestIdx;
}

// Fill 4 face vertices from center and two tangent offsets
__device__ void setFaceVertices(float4 verts[4], float4 center, float4 off1, float4 off2)
{
    verts[0] = vec3::add(center, vec3::add(off1, off2));  // +t1 +t2
    verts[1] = vec3::add(center, vec3::sub(off1, off2));  // +t1 -t2
    verts[2] = vec3::sub(center, vec3::add(off1, off2));  // -t1 -t2
    verts[3] = vec3::sub(center, vec3::sub(off1, off2));  // -t1 +t2
}

// Build reference face from SAT result
__device__ void getReferenceFace(
    const SATContext& ctx, 
    const SATResult& res, 
    ReferenceFace& ref
)
{
    // Build axis indices around reference axis
    int i = res.axisIdx % 3;
    int j = (i + 1) % 3;
    int k = (i + 2) % 3;
    float b = (float)(res.axisIdx >= 3);

    // Get reference face normal (blend between A and B's axis)
    ref.normal = blendAxes(WORLD_AXES[i], ctx.axB[i], b);

    // Point reference normal towards incident car
    float d = vec3::dot(ctx.vecAB, ref.normal);
    float s = sign(d);
    float w = s * (1.0f - b) - s * b;
    ref.normal = vec3::mult(ref.normal, w);

    // Get tangent axes for reference face
    ref.ortho1 = blendAxes(WORLD_AXES[j], ctx.axB[j], b);
    ref.ortho2 = blendAxes(WORLD_AXES[k], ctx.axB[k], b);

    // Get candidate reference face centers
    float4 cA = vec3::mult(WORLD_AXES[i], w * CAR_HALF_EX_ARR[i]);
    float4 cB = vec3::add(ctx.vecAB, vec3::mult(ctx.axB[i], w * CAR_HALF_EX_ARR[i]));

    // Get reference face center & extents
    ref.center = blendAxes(cA, cB, b);
    ref.halfEx = make_float2(CAR_HALF_EX_ARR[j], CAR_HALF_EX_ARR[k]);
}

// Build incident face vertices from reference face
__device__ void getIncidentFace(
    const SATContext& ctx,
    const SATResult& res,
    const ReferenceFace& ref,
    IncidentFace& inc
)
{
    // Determine which box is incident (opposite of reference)
    float b = (float)(res.axisIdx >= 3);

    // Get incident car axes and origin
    const float4* incAxes = (b < 0.5f) ? ctx.axB : WORLD_AXES;
    float4 incOrig = vec3::mult(ctx.vecAB, 1.0f - b);

    // Find most parallel face (axis with largest |dot|)
    float bestDot;
    int bestIdx = findIncidentAxis(ref.normal, incAxes, bestDot);

    // Get face indices tangent to incident axis
    int t1 = (bestIdx + 1) % 3;
    int t2 = (bestIdx + 2) % 3;

    // Compute incident face center
    // If bestDot > 0, axis points same way as ref.normal, so use -axis face
    // If bestDot < 0, axis points opposite to ref.normal, so use +axis face
    float4 incNorm = vec3::mult(incAxes[bestIdx], -sign(bestDot));
    float4 incCent = vec3::add(incOrig, vec3::mult(incNorm, CAR_HALF_EX_ARR[bestIdx]));

    // Compute tangent offsets and fill vertices
    float4 off1 = vec3::mult(incAxes[t1], CAR_HALF_EX_ARR[t1]);
    float4 off2 = vec3::mult(incAxes[t2], CAR_HALF_EX_ARR[t2]);
    setFaceVertices(inc.verts, incCent, off1, off2);
}

// Project incident vertices onto reference plane
__device__ void projectIncidentVertices(
    ReferenceFace& ref, 
    IncidentFace& inc, 
    ClipPolygon& poly
)
{
    poly.count = 4;

    for (int i = 0; i < 4; i++)
    {
        // Relative vector from ref center to inc vert
        float4 r = vec3::sub(inc.verts[i], ref.center);

        // Length along reference tangents
        float u = vec3::dot(r, ref.ortho1);
        float v = vec3::dot(r, ref.ortho2);
        poly.points[i].p = make_float2(u, v);

        // Get depth of incident point
        poly.points[i].d = -vec3::dot(r, ref.normal);
    }
}

// Clip incident vertices against reference polygon edges
__device__ void clipIncidentEdge(ClipPolygon& poly, int a, float s, float h)
{
    ClipPoint points[8];
    int newCount = 0;
    int oldCount = poly.count;

    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        if (i >= oldCount) continue;

        // Current and next vertex w/ wrapping
        ClipPoint curr = poly.points[i];
        ClipPoint next = poly.points[(i+1) % oldCount];

        // Get points defining incident edge
        float currP = (a == 0) ? curr.p.x : curr.p.y;
        float nextP = (a == 0) ? next.p.x : next.p.y;

        // Distance to reference edge line
        // - inside, + outside
        float currD = s * currP - h;
        float nextD = s * nextP - h;
        bool currIn = (currD <= 0.0f);
        bool nextIn = (nextD <= 0.0f);

        // If current index inside, add to poly
        if (currIn)
        {
            points[newCount++] = curr;
        }

        // If edge crosses reference edge, add intersection
        if (currIn != nextIn)
        {
            // Interpolation scalar
            float t = currD / (currD - nextD);

            // Interpolate incident edge
            ClipPoint intersection;
            intersection.p.x = curr.p.x + t * (next.p.x - curr.p.x);
            intersection.p.y = curr.p.y + t * (next.p.y - curr.p.y);

            // Interpolate depth
            intersection.d = curr.d + t * (next.d - curr.d);

            points[newCount++] = intersection;
        }
    }

    // Save results to persistent polygon
    poly.count = newCount;

    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        poly.points[i] = points[i];
    }
}

// Clip all edges of incident rect against reference rect
__device__ void clipIncidentPoly(ClipPolygon& poly, float2 halfEx)
{
    clipIncidentEdge(poly, 0, +1.0f, halfEx.x);
    clipIncidentEdge(poly, 0, -1.0f, halfEx.x);
    clipIncidentEdge(poly, 1, +1.0f, halfEx.y);
    clipIncidentEdge(poly, 1, -1.0f, halfEx.y);
}

// Wrap-around angle difference in [-pi, pi]
__device__ float angleDiff(float a, float b)
{
    float d = a - b;
    if (d > 3.14159265f) d -= 6.28318530f;
    if (d < -3.14159265f) d += 6.28318530f;
    return fabsf(d);
}

// Cull contact points to max 4 using angular distribution
__device__ void cullContactPoints(ContactManifold& contact)
{
    if (contact.count <= 4) return;

    int n = contact.count;

    // Find deepest point index
    int deepIdx = 0;
    for (int i = 1; i < n; ++i)
        if (contact.depths[i] > contact.depths[deepIdx]) deepIdx = i;

    // Compute centroid
    float4 centroid = make_float4(0, 0, 0, 0);
    for (int i = 0; i < n; ++i)
        centroid = vec3::add(centroid, contact.points[i]);
    centroid = vec3::mult(centroid, 1.0f / n);

    // Build tangent frame and compute angles
    float4 t1 = vec3::norm(vec3::sub(contact.points[0], centroid));
    float4 t2 = vec3::cross(contact.normal, t1);

    float angles[8];
    for (int i = 0; i < n; ++i)
    {
        float4 r = vec3::sub(contact.points[i], centroid);
        angles[i] = atan2f(vec3::dot(r, t2), vec3::dot(r, t1));
    }

    // Select: deepest + 3 at 90° intervals
    int keep[4] = {deepIdx, -1, -1, -1};
    bool used[8] = {false};
    used[deepIdx] = true;

    for (int j = 1; j < 4; ++j)
    {
        float target = angles[deepIdx] + j * 1.57079632f;
        int best = -1;
        float bestDiff = 1e9f;

        for (int i = 0; i < n; ++i)
        {
            if (used[i]) continue;
            float d = angleDiff(angles[i], target);
            if (d < bestDiff) { bestDiff = d; best = i; }
        }

        if (best >= 0) { keep[j] = best; used[best] = true; }
    }

    // Compact
    int out = 0;
    for (int j = 0; j < 4; ++j)
    {
        if (keep[j] < 0) continue;
        contact.points[out] = contact.points[keep[j]];
        contact.depths[out] = contact.depths[keep[j]];
        out++;
    }
    contact.count = out;
}

// Construct contact manifold from 2D points and depths
__device__ void reconstructContactManifold(
    ClipPolygon& poly, 
    ReferenceFace& ref, 
    ContactManifold& contact
)
{
    for (int i = 0; i < poly.count; ++i)
    {
        float d = poly.points[i].d;

        // Only keep penetrating points
        if (d <= 0.0f) continue;

        float2 p = poly.points[i].p;

        float4 refP = vec3::add(
            ref.center,
            vec3::add(
                vec3::mult(ref.ortho1, p.x),
                vec3::mult(ref.ortho2, p.y)
            )
        );

        // Use depth to construct point in 3D
        float4 conP = vec3::sub(refP, vec3::mult(ref.normal, d));

        // Add point to manifold
        int outIdx = contact.count;
        if (outIdx >= 8) break;
        contact.points[outIdx] = conP;
        contact.depths[outIdx] = d;
        contact.count = outIdx + 1;
        contact.normal = ref.normal;
    }
}


// Face-face collision manifold generation (main entry point)
__device__ void generateFaceFaceManifold(
    SATContext& ctx, 
    SATResult& res,
    ContactManifold& contact
)
{
    ReferenceFace ref;
    IncidentFace inc;
    ClipPolygon poly;

    // Build reference face
    getReferenceFace(ctx, res, ref);

    // Build incident face in reference space
    getIncidentFace(ctx, res, ref, inc);

    // Project incident vertices on reference plane
    projectIncidentVertices(ref, inc, poly);

    // Clip incident polygon against reference plane
    clipIncidentPoly(poly, ref.halfEx);

    // Extract full contact manifold
    reconstructContactManifold(poly, ref, contact);

    // Cull to max 4 points with good angular distribution
    cullContactPoints(contact);
}
