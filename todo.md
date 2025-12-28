### Plan
Optimize collisions
- Split car-car collision into separate kernels
    - SAT kernel (thread per axis)
    - Clipping kernel (thread per edge/face?)
    - Manifold culling kernel
- Persistent kernel to limit data reads?
    - Requires unified kernel? Tricky

To-implement
- Impulse response from collision manifolds
- Suspension (wheels) collision & impulse
- Control input processing
    - Jump/flip/roll/boost impulses
- Ball physics and collision
    - Cars & arena mesh
- Arena mesh create, access, collision
- Register/store game events (boolean checks)