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
    float b = (float)(res.axisIdx >= 3);  // b=0: incident is B, b=1: incident is A
    const float4* axes = (b < 0.5f) ? ctx.axB : WORLD_AXES;
    float4 orig = vec3::mult(ctx.vecAB, 1.0f - b);

    // Find face most antiparallel to reference normal
    float  dot;
    int    idx = findIncidentAxis(ref.normal, axes, dot);
    int    t1  = (idx + 1) % 3;
    int    t2  = (idx + 2) % 3;

    // Incident face center and vertices
    float4 norm = vec3::mult(axes[idx], -sign(dot));
    float4 cent = vec3::add(orig, vec3::mult(norm, CAR_HALF_EX_ARR[idx]));
    float4 off1 = vec3::mult(axes[t1], CAR_HALF_EX_ARR[t1]);
    float4 off2 = vec3::mult(axes[t2], CAR_HALF_EX_ARR[t2]);
    setFaceVertices(inc.verts, cent, off1, off2);
}

// Project incident vertices to 2D reference plane coords + depth
__device__ void projectIncidentVertices(ReferenceFace& ref, IncidentFace& inc, ClipPolygon& poly)
{
    poly.count = 4;
    for (int i = 0; i < 4; i++)
    {
        float4 r = vec3::sub(inc.verts[i], ref.center);
        poly.points[i].p = make_float2(vec3::dot(r, ref.ortho1), vec3::dot(r, ref.ortho2));
        poly.points[i].d = -vec3::dot(r, ref.normal);  // Positive = penetrating
    }
}

// Clip polygon against single reference edge (a=0: X, a=1: Y; s=+1/-1; h=half-extent)
__device__ void clipIncidentEdge(ClipPolygon& poly, int a, float s, float h)
{
    ClipPoint pts[8];
    int nOld = poly.count;
    int nNew = 0;

    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        if (i >= nOld)
        {
            continue;
        }

        ClipPoint curr = poly.points[i];
        ClipPoint next = poly.points[(i + 1) % nOld];
        float pC = (a == 0) ? curr.p.x : curr.p.y;
        float pN = (a == 0) ? next.p.x : next.p.y;
        float dC = s * pC - h;  // Signed distance (negative = inside)
        float dN = s * pN - h;

        if (dC <= 0.0f)
        {
            pts[nNew++] = curr;
        }

        // Add intersection if edge crosses clip line
        if ((dC <= 0.0f) != (dN <= 0.0f))
        {
            float t = dC / (dC - dN);
            pts[nNew++] = {
                make_float2(curr.p.x + t * (next.p.x - curr.p.x),
                            curr.p.y + t * (next.p.y - curr.p.y)),
                curr.d + t * (next.d - curr.d)
            };
        }
    }

    poly.count = nNew;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        poly.points[i] = pts[i];
    }
}

// Clip incident polygon against all 4 reference edges
__device__ void clipIncidentPoly(ClipPolygon& poly, float2 h)
{
    clipIncidentEdge(poly, 0, +1.0f, h.x);
    clipIncidentEdge(poly, 0, -1.0f, h.x);
    clipIncidentEdge(poly, 1, +1.0f, h.y);
    clipIncidentEdge(poly, 1, -1.0f, h.y);
}

// Wrap angle difference to [-pi, pi] and return absolute value
__device__ float angleDiff(float a, float b)
{
    float d = a - b;
    if (d > 3.14159265f)
    {
        d -= 6.28318530f;
    }
    if (d < -3.14159265f)
    {
        d += 6.28318530f;
    }
    return fabsf(d);
}

// Cull to max 4 points: keep deepest + 3 at ~90 degree intervals
__device__ void cullContactPoints(ContactManifold& m)
{
    if (m.count <= 4)
    {
        return;
    }

    int n = m.count;

    // Find deepest point
    int deep = 0;
    for (int i = 1; i < n; ++i)
    {
        if (m.depths[i] > m.depths[deep])
        {
            deep = i;
        }
    }

    // Compute centroid
    float4 cent = make_float4(0, 0, 0, 0);
    for (int i = 0; i < n; ++i)
    {
        cent = vec3::add(cent, m.points[i]);
    }
    cent = vec3::mult(cent, 1.0f / n);

    // Build tangent frame and compute angles
    float4 t1 = vec3::norm(vec3::sub(m.points[0], cent));
    float4 t2 = vec3::cross(m.normal, t1);
    float angles[8];
    for (int i = 0; i < n; ++i)
    {
        float4 r = vec3::sub(m.points[i], cent);
        angles[i] = atan2f(vec3::dot(r, t2), vec3::dot(r, t1));
    }

    // Select deepest + 3 points at 90 degree intervals
    int  keep[4] = {deep, -1, -1, -1};
    bool used[8] = {false};
    used[deep] = true;

    for (int j = 1; j < 4; ++j)
    {
        float target = angles[deep] + j * 1.57079632f;
        int   best = -1;
        float diff = 1e9f;

        for (int i = 0; i < n; ++i)
        {
            if (used[i])
            {
                continue;
            }
            float d = angleDiff(angles[i], target);
            if (d < diff)
            {
                diff = d;
                best = i;
            }
        }

        if (best >= 0)
        {
            keep[j] = best;
            used[best] = true;
        }
    }

    // Compact kept points
    int out = 0;
    for (int j = 0; j < 4; ++j)
    {
        if (keep[j] < 0)
        {
            continue;
        }
        m.points[out] = m.points[keep[j]];
        m.depths[out] = m.depths[keep[j]];
        out++;
    }
    m.count = out;
}

// Convert clipped 2D polygon to 3D contact manifold (only penetrating points)
__device__ void reconstructContactManifold(ClipPolygon& poly, ReferenceFace& ref, ContactManifold& m)
{
    for (int i = 0; i < poly.count; ++i)
    {
        float  d = poly.points[i].d;
        float2 p = poly.points[i].p;

        if (d <= 0.0f || m.count >= 8)
        {
            continue;
        }

        // Project 2D back to 3D, offset by depth for contact point
        float4 refP = vec3::add(ref.center,
            vec3::add(vec3::mult(ref.ortho1, p.x), vec3::mult(ref.ortho2, p.y)));
        m.points[m.count] = vec3::sub(refP, vec3::mult(ref.normal, d));
        m.depths[m.count] = d;
        m.normal = ref.normal;
        m.count++;
    }
}

// Face-face manifold via Sutherland-Hodgman clipping
__device__ void generateFaceFaceManifold(SATContext& ctx, SATResult& res, ContactManifold& m)
{
    ReferenceFace ref;
    IncidentFace  inc;
    ClipPolygon   poly;

    getReferenceFace(ctx, res, ref);
    getIncidentFace(ctx, res, ref, inc);
    projectIncidentVertices(ref, inc, poly);
    clipIncidentPoly(poly, ref.halfEx);
    reconstructContactManifold(poly, ref, m);
    cullContactPoints(m);
}
