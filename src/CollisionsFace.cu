#include "CudaMath.hpp"
#include "RLConstants.hpp"
#include "CollisionsFace.hpp"
#include "CollisionsSAT.hpp"

// Branchless blend: b=0 returns axA, b=1 returns axB
__device__ float4 blendAxes(float4 axA, float4 axB, float b)
{
    return vec3::add(vec3::mult(axA, 1.0f - b), vec3::mult(axB, b));
}

// Find axis most antiparallel to dir (largest |dot|)
__device__ int findIncidentAxis(float4 dir, const float4* axes, float& bestDot)
{
    int   idx = 0;
    float max = 0.0f;
    bestDot = 0.0f;

    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        float d = vec3::dot(dir, axes[i]);
        float m = fabsf(d);
        if (m > max)
        {
            max = m;
            idx = i;
            bestDot = d;
        }
    }
    return idx;
}

// Generate 4 face vertices: (+o1+o2), (+o1-o2), (-o1-o2), (-o1+o2)
__device__ void setFaceVertices(float4 verts[4], float4 c, float4 o1, float4 o2)
{
    verts[0] = vec3::add(c, vec3::add(o1, o2));
    verts[1] = vec3::add(c, vec3::sub(o1, o2));
    verts[2] = vec3::sub(c, vec3::add(o1, o2));
    verts[3] = vec3::sub(c, vec3::sub(o1, o2));
}

// Build reference face from the box whose face normal was the separating axis
__device__ void getReferenceFace(const SATContext& ctx, const SATResult& res, ReferenceFace& ref)
{
    // Axis indices: i=separating, j,k=tangent
    int i = res.axisIdx % 3;
    int j = (i + 1) % 3;
    int k = (i + 2) % 3;
    float b = (float)(res.axisIdx >= 3);  // b=0: A's face, b=1: B's face

    // Reference normal (blend between A's and B's axis)
    ref.normal = blendAxes(WORLD_AXES[i], ctx.axB[i], b);

    // Orient normal toward incident box
    float d = vec3::dot(ctx.vecAB, ref.normal);
    float s = sign(d);
    float w = s * (1.0f - b) - s * b;  // Flip sign for B's face
    ref.normal = vec3::mult(ref.normal, w);

    // Tangent axes
    ref.ortho1 = blendAxes(WORLD_AXES[j], ctx.axB[j], b);
    ref.ortho2 = blendAxes(WORLD_AXES[k], ctx.axB[k], b);

    // Face center (blend based on which box)
    float4 cA = vec3::mult(WORLD_AXES[i], w * CAR_HALF_EX_ARR[i]);
    float4 cB = vec3::add(ctx.vecAB, vec3::mult(ctx.axB[i], w * CAR_HALF_EX_ARR[i]));
    ref.center = blendAxes(cA, cB, b);
    ref.halfEx = make_float2(CAR_HALF_EX_ARR[j], CAR_HALF_EX_ARR[k]);
}

// Build incident face (most antiparallel to reference normal)
__device__ void getIncidentFace(const SATContext& ctx, const SATResult& res, const ReferenceFace& ref, IncidentFace& inc)
{
    float dot;

    // b=0: incident is B, b=1: incident is A
    float b = (float)(res.axisIdx >= 3);
    const float4* axes = (b < 0.5f) ? ctx.axB : WORLD_AXES;
    float4 orig = vec3::mult(ctx.vecAB, 1.0f - b);

    // Find face most antiparallel to reference normal
    int idx = findIncidentAxis(ref.normal, axes, dot);
    int t1  = (idx + 1) % 3;
    int t2  = (idx + 2) % 3;

    // Incident face center and vertices
    float4 norm = vec3::mult(axes[idx], -sign(dot));
    float4 cent = vec3::add(orig, vec3::mult(norm, CAR_HALF_EX_ARR[idx]));
    float4 off1 = vec3::mult(axes[t1], CAR_HALF_EX_ARR[t1]);
    float4 off2 = vec3::mult(axes[t2], CAR_HALF_EX_ARR[t2]);
    setFaceVertices(inc.verts, cent, off1, off2);
}

// Add a contact point to manifold, keeping the 4 deepest
__device__ void addContactPoint(ContactManifold& m, float4 pt, float depth)
{
    if (depth <= 0.0f) return;  // Not penetrating

    if (m.count < 4)
    {
        m.points[m.count] = pt;
        m.depths[m.count] = depth;
        m.count++;
    }
    else
    {
        // Replace shallowest if this is deeper
        int minIdx = 0;
        float minD = m.depths[0];

        #pragma unroll
        for (int j = 1; j < 4; ++j)
        {
            if (m.depths[j] < minD)
            {
                minD = m.depths[j];
                minIdx = j;
            }
        }
        
        if (depth > minD)
        {
            m.points[minIdx] = pt;
            m.depths[minIdx] = depth;
        }
    }
}

// Clip a single edge against reference rectangle using Liang-Barsky algorithm
// Returns true if any part of the edge is inside, with t0/t1 as parametric bounds
__device__ bool clipEdgeLiangBarsky(float2 p0, float2 p1, float2 h, float& t0, float& t1)
{
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    t0 = 0.0f;
    t1 = 1.0f;

    // Clip against all 4 edges: -h.x, +h.x, -h.y, +h.y
    float p[4] = {-dx, dx, -dy, dy};
    float q[4] = {p0.x + h.x, h.x - p0.x, p0.y + h.y, h.y - p0.y};

    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if (fabsf(p[i]) < 1e-8f)
        {
            // Edge parallel to clip boundary
            if (q[i] < 0.0f) return false;  // Outside
        }
        else
        {
            float t = q[i] / p[i];
            if (p[i] < 0.0f)
            {
                // Entering
                if (t > t0) t0 = t;
            }
            else
            {
                // Leaving
                if (t < t1) t1 = t;
            }
        }
    }

    return t0 <= t1;
}

// Process one incident edge: clip against reference rect and add contacts
__device__ void processIncidentEdge(
    float2 p0, float d0, float2 p1, float d1,
    const ReferenceFace& ref, ContactManifold& m)
{
    float t0, t1;
    if (!clipEdgeLiangBarsky(p0, p1, ref.halfEx, t0, t1))
        return;  // Edge entirely outside

    // Interpolate clipped endpoints
    float2 cp0 = make_float2(p0.x + t0 * (p1.x - p0.x), p0.y + t0 * (p1.y - p0.y));
    float2 cp1 = make_float2(p0.x + t1 * (p1.x - p0.x), p0.y + t1 * (p1.y - p0.y));
    float cd0 = d0 + t0 * (d1 - d0);
    float cd1 = d0 + t1 * (d1 - d0);

    // Convert to 3D and add to manifold
    if (cd0 > 0.0f)
    {
        float4 pt = vec3::add(ref.center,
            vec3::add(vec3::mult(ref.ortho1, cp0.x), vec3::mult(ref.ortho2, cp0.y)));
        pt = vec3::sub(pt, vec3::mult(ref.normal, cd0));
        addContactPoint(m, pt, cd0);
    }

    // Only add second point if it's different from first (t0 != t1)
    if (t1 > t0 + 1e-6f && cd1 > 0.0f)
    {
        float4 pt = vec3::add(ref.center,
            vec3::add(vec3::mult(ref.ortho1, cp1.x), vec3::mult(ref.ortho2, cp1.y)));
        pt = vec3::sub(pt, vec3::mult(ref.normal, cd1));
        addContactPoint(m, pt, cd1);
    }
}

// Face-face manifold via Liang-Barsky edge clipping
__device__ void generateFaceFaceManifold(SATContext& ctx, SATResult& res, ContactManifold& m)
{
    ReferenceFace ref;
    IncidentFace  inc;

    getReferenceFace(ctx, res, ref);
    getIncidentFace(ctx, res, ref, inc);

    m.normal = ref.normal;
    m.count = 0;

    // Project incident vertices to 2D + depth (small local arrays)
    float2 p2d[4];
    float  depths[4];

    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float4 r = vec3::sub(inc.verts[i], ref.center);
        p2d[i] = make_float2(vec3::dot(r, ref.ortho1), vec3::dot(r, ref.ortho2));
        depths[i] = -vec3::dot(r, ref.normal);  // Positive = penetrating
    }

    // Process each edge of the incident face
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int j = (i + 1) & 3;  // Next vertex (mod 4)
        processIncidentEdge(p2d[i], depths[i], p2d[j], depths[j], ref, m);
    }
}
