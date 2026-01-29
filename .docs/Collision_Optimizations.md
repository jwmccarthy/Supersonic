# Collision Kernel Optimizations

Baseline: **65 μs** for 2048 sims × 8 cars = 16,384 cars

---

## 1. Precompute Triangle AABBs (~18% speedup)

**Problem:** Each triangle test loads 3 vertices (48 bytes) and computes min/max.

**Solution:** Precompute triangle AABBs during mesh loading, store as `{float4 min, float4 max}` (32 bytes).

**Implementation:**
- Add `TriAABB` struct to `ArenaMesh.cuh`
- Add `TriAABB* triAABBs` device pointer to `ArenaMesh`
- Compute AABBs in `buildBroadphaseGrid()` and upload to GPU
- In kernel: load precomputed AABB instead of vertices

**Result:** 65 μs → 53 μs

---

## 2. Fix AABB Overlap Test (correctness)

**Bug:** The overlap test was inverted.

```cpp
// WRONG:
overlaps += vec3::gte(aabbMin, triMax) && vec3::lte(aabbMax, triMin);

// CORRECT:
overlaps += vec3::lte(aabbMin, triMax) && vec3::gte(aabbMax, triMin);
```

---

## 3. Use `__ldg()` for Read-Only Data

**Problem:** Default loads don't use texture cache.

**Solution:** Use `__ldg()` for triangle data loads:
```cpp
int idx = __ldg(&arena->triIdx[triBeg + t]);
float4 triMin = __ldg(&arena->triAABBs[idx].min);
float4 triMax = __ldg(&arena->triAABBs[idx].max);
```

---

## 4. Increase Block Size (~5% speedup)

**Problem:** Block size 32 (1 warp per block) has poor latency hiding.

**Solution:** Use block size 128 (4 warps per block).

```cpp
int blockSize = 128;
```

**Result:** 53 μs → 50 μs

---

## 5. Attempted: AABB Broadcast (no improvement)

**Idea:** Only lane 0 computes car AABB, then broadcast to all lanes via `__shfl_sync`.

**Result:** No measurable improvement. The AABB computation is fast relative to the memory-bound triangle loop.

---

## Summary

| Optimization | Time | Improvement |
|--------------|------|-------------|
| Baseline | 65 μs | - |
| Precomputed triangle AABBs | 53 μs | 18% |
| Block size 128 | 50 μs | 5% |
| **Total** | **50 μs** | **23%** |

---

## Future Opportunities

1. **Better spatial grid tuning** - Reduce triangle duplicates across cells
2. **Coalesced memory access** - Restructure data for better access patterns
3. **Warp-level triangle deduplication** - Avoid testing same triangle multiple times
4. **SOA layout for triangles** - Separate arrays for min.x, min.y, min.z, etc.
