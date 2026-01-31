# Car-Arena Collision Pipeline Optimization

## Performance Context

**Goal:** Maximize throughput for RL training (1024+ parallel sims)
**CPU reference:** Full physics step = 10 μs/sim → 10.24 ms for 1024 sims sequential
**GPU target:** Complete collision pipeline for 1024 sims in ≪ 10 ms wall-clock

## Current Baseline

| Config | Kernel | Time |
|--------|--------|------|
| THREADS_PER_CAR=16 | `carArenaCollisionKernel` | 27.49 μs |
| THREADS_PER_CAR=2 | `carArenaCollisionKernel` | **16 μs** |

**Key insight:** Fewer threads per car = better ILP, less reduction overhead, more diverse memory access per warp. The 2-thread config runs at 58% of the 16-thread time.

**Reference:** Car-car SAT kernel (6144 pairs) = 18 μs → ~3 ns/pair

---

## Pipeline Architecture

### Option A: Two-Kernel Pipeline (Simplest)
```
┌─────────────────────┐    ┌─────────────────────┐
│  Broad Phase        │───▶│  Narrow Phase       │
│  (AABB + collect)   │    │  (SAT + response)   │
└─────────────────────┘    └─────────────────────┘
      ~20 μs                    ~30-50 μs
```
Broad-phase outputs candidates via atomic append. Narrow-phase consumes them.

### Option B: Fused Single Kernel (Lowest Latency)
```
┌─────────────────────────────────────────────────┐
│  Per-car: Broad → Narrow → Accumulate contacts  │
└─────────────────────────────────────────────────┘
                    ~40-60 μs
```
No intermediate buffer. Each thread does full pipeline for its car's candidates.
Avoids kernel launch overhead and atomic contention.

---

## Thread Strategy: Why 2 Threads/Car Wins

With THREADS_PER_CAR=2:
- Each warp processes 16 different cars → diverse memory access, good coalescing
- Each thread iterates more triangles → better instruction pipelining (ILP)
- Minimal reduction overhead (1 shuffle vs 4 for 16 threads)
- More warps available → better latency hiding

**Recommendation:** Keep THREADS_PER_CAR=2 for broad-phase. Narrow-phase uses 1 thread per candidate pair (no reduction needed).

---

## Implementation Strategy

### Approach: Fused Broad+Narrow Per Car

Instead of outputting candidates to a buffer, each car thread does:
1. Iterate cells/triangles (broad-phase AABB test)
2. On AABB hit → immediately run SAT test (narrow-phase)
3. On SAT hit → accumulate contact into per-car local storage
4. At end → write final contacts

**Advantages:**
- No intermediate buffer allocation
- No atomic contention for candidate list
- Better cache locality (triangle data hot from AABB test)
- Single kernel launch

**Structure:**
```cpp
__device__ void carArenaCollision(GameState* state, ArenaMesh* arena,
                                   int carIdx, int laneIdx,
                                   Contact* carContacts, int* numContacts)
{
    float4 pos = __ldg(&state->cars.position[carIdx]);
    float4 rot = __ldg(&state->cars.rotation[carIdx]);
    auto [aabbMin, aabbMax] = getCarAABB(arena, pos, rot);

    int3 cellMin = arena->getCellIdx(aabbMin);
    int3 cellMax = arena->getCellIdx(aabbMax);

    int localContacts = 0;
    Contact contacts[MAX_CONTACTS_PER_CAR];  // Register/local storage

    for (int x = cellMin.x; x <= cellMax.x; ++x)
    for (int y = cellMin.y; y <= cellMax.y; ++y)
    for (int z = cellMin.z; z <= cellMax.z; ++z)
    {
        int cellIdx = arena->flatCellIdx(x, y, z);
        int triBeg = __ldg(&arena->triPre[cellIdx]);
        int triEnd = __ldg(&arena->triPre[cellIdx + 1]);

        for (int t = triBeg + laneIdx; t < triEnd; t += THREADS_PER_CAR)
        {
            int triIdx = __ldg(&arena->triIdx[t]);

            // Broad-phase: AABB test
            float4 triMin = __ldg(&arena->aabbMin[triIdx]);
            float4 triMax = __ldg(&arena->aabbMax[triIdx]);

            if (!aabbOverlap(aabbMin, aabbMax, triMin, triMax)) continue;

            // Narrow-phase: SAT test (inline)
            Triangle tri = getTriVerts(arena, triIdx);
            float4 normal; float depth;
            if (satTriangleOBB(pos, rot, tri, normal, depth))
            {
                if (localContacts < MAX_CONTACTS_PER_CAR)
                    contacts[localContacts++] = {triIdx, normal, depth};
            }
        }
    }

    // Reduce contacts across lanes, write final result
    // ...
}
```

---

## SAT Implementation

**13 separating axes for Triangle-OBB:**
1. 3 OBB face normals (car X, Y, Z in world space)
2. 1 triangle face normal
3. 9 cross products (3 OBB edges × 3 triangle edges)

```cpp
__device__ __forceinline__ bool satTriangleOBB(
    float4 pos, float4 rot,
    Triangle tri,
    float4& outNormal, float& outDepth)
{
    // OBB axes in world space
    float4 obbX = quat::toWorld({1,0,0,0}, rot);
    float4 obbY = quat::toWorld({0,1,0,0}, rot);
    float4 obbZ = quat::toWorld({0,0,1,0}, rot);

    // Triangle edges and normal
    float4 e0 = vec3::sub(tri.v1, tri.v0);
    float4 e1 = vec3::sub(tri.v2, tri.v1);
    float4 e2 = vec3::sub(tri.v0, tri.v2);
    float4 triN = vec3::cross(e0, e1);

    float minPen = FLT_MAX;
    float4 minAxis;

    // Test each axis, early-out on separation
    // (13 axis tests with projection and overlap check)
    // ...

    outNormal = minAxis;
    outDepth = minPen;
    return true;
}
```

**Note:** Pre-computed triangle normals in `arena->norms[]` can skip the cross product.

---

## Implementation Plan

### Step 1: Add SAT Function
**File:** `include/NarrowPhase.cuh` (new)
- `satTriangleOBB()` device function
- Use existing `quat::toWorld()` from CudaMath.cuh

### Step 2: Fuse Broad+Narrow in Existing Kernel
**File:** `include/CarArenaCollision.cuh`
- Modify `carArenaBroadPhase` to call SAT on AABB hits
- Add contact accumulation (local array per thread)
- Keep THREADS_PER_CAR=2

### Step 3: Contact Output Structure
**File:** `include/GameState.cuh`
- Add `Contact` struct and per-car contact buffer
- Or use shared memory for intra-warp contact merging

### Step 4: Contact Response
**File:** `include/ContactResponse.cuh` (new)
- Separate kernel or fused into collision kernel
- Apply impulse based on penetration depth and normal

---

## Files to Modify

| File | Change |
|------|--------|
| `include/NarrowPhase.cuh` | New: SAT implementation |
| `include/CarArenaCollision.cuh` | Fuse SAT into broad-phase loop |
| `include/GameState.cuh` | Add Contact struct |
| `src/CudaKernels.cu` | Update kernel if signature changes |
| `src/RLEnvironment.cu` | Allocate contact buffers |

---

## Expected Performance

| Stage | Time |
|-------|------|
| Fused broad+narrow | 30-50 μs |
| Contact response | 10-20 μs |
| **Total collision** | **~40-70 μs** |

Rationale: Car-car SAT (6144 pairs) = 18 μs → ~3 ns/pair. Fused approach keeps triangle data cache-hot from AABB test, avoiding memory round-trip of two-kernel approach.

---

## Verification

1. **Correctness:** Place cars at known wall positions, verify contacts detected
2. **Performance:** `./profile.sh` should show fused kernel < 70 μs
3. **Regression:** Ensure broad-phase candidate count unchanged
