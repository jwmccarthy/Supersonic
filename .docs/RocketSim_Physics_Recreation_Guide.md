# RocketSim Physics Recreation Guide

## Complete Implementation Reference for Rocket League Physics Simulation

This document provides a comprehensive, self-contained reference for recreating the physics simulation used in Rocket League, as implemented in RocketSim. Every algorithm, constant, and formula needed for a complete recreation is documented here.

---

## Table of Contents

1. [Unit System and Coordinate System](#1-unit-system-and-coordinate-system)
2. [Physics Constants](#2-physics-constants)
3. [Data Structures](#3-data-structures)
4. [Car Configuration and Hitboxes](#4-car-configuration-and-hitboxes)
5. [Suspension and Wheel Physics](#5-suspension-and-wheel-physics)
6. [Ground Movement and Driving](#6-ground-movement-and-driving)
7. [Jump and Flip Mechanics](#7-jump-and-flip-mechanics)
8. [Air Control and Aerial Movement](#8-air-control-and-aerial-movement)
9. [Boost System](#9-boost-system)
10. [Ball Physics](#10-ball-physics)
11. [Collision Detection and Resolution](#11-collision-detection-and-resolution)
12. [Car-Ball Collision](#12-car-ball-collision)
13. [Car-Car Collision (Bumps and Demos)](#13-car-car-collision-bumps-and-demos)
14. [Arena and World Collision](#14-arena-and-world-collision)
15. [Special Game Modes](#15-special-game-modes)
16. [Simulation Loop Architecture](#16-simulation-loop-architecture)
17. [Linear Piecewise Curves](#17-linear-piecewise-curves)
18. [Mathematical Foundations](#18-mathematical-foundations)

---

## 1. Unit System and Coordinate System

### Unit Conversion

RocketSim uses **Unreal Units (UU)** as the primary unit system, while the underlying Bullet Physics engine uses metric units.

| Measurement | Unreal Units | Real-World Equivalent |
|-------------|--------------|----------------------|
| 1 UU | 1 | 2 centimeters |
| 50 UU | 50 | 1 meter |

**Conversion Macros:**
```
BT_TO_UU = 50.0    // Bullet to Unreal Units
UU_TO_BT = 0.02    // Unreal Units to Bullet (1/50)
```

When interfacing with Bullet Physics:
- Multiply positions/velocities by `UU_TO_BT` when sending to Bullet
- Multiply results by `BT_TO_UU` when reading from Bullet

### Coordinate System

Rocket League uses a **right-handed** coordinate system:
- **X-axis**: Forward (positive = forward)
- **Y-axis**: Right (positive = right side of car)
- **Z-axis**: Up (positive = up)

The arena is symmetric about the Y=0 plane:
- Blue team defends negative Y
- Orange team defends positive Y

**Arena Dimensions (Standard Soccar):**
```
ARENA_EXTENT_X = 4096 UU    (half-width)
ARENA_EXTENT_Y = 5120 UU    (half-length, excluding goal depth)
ARENA_HEIGHT = 2048 UU
```

**Arena Dimensions (Hoops):**
```
ARENA_EXTENT_X_HOOPS = 2966.67 UU  (8900/3)
ARENA_EXTENT_Y_HOOPS = 3581 UU
ARENA_HEIGHT_HOOPS = 1820 UU
```

**Arena Dimensions (Dropshot):**
```
ARENA_HEIGHT_DROPSHOT = 2024 UU
FLOOR_HEIGHT_DROPSHOT = 1.5 UU  (floor is slightly raised)
```

---

## 2. Physics Constants

### Gravity and Mass

```cpp
GRAVITY_Z = -650.0 UU/s^2    // Downward acceleration

CAR_MASS_BT = 180.0          // Car mass (arbitrary units)
BALL_MASS_BT = 30.0          // Ball mass (1/6 of car mass)
```

### Speed Limits

```cpp
CAR_MAX_SPEED = 2300.0 UU/s      // Maximum car velocity magnitude
CAR_MAX_ANG_SPEED = 5.5 rad/s    // Maximum car angular velocity

BALL_MAX_SPEED = 6000.0 UU/s     // Maximum ball velocity magnitude
BALL_MAX_ANG_SPEED = 6.0 rad/s   // Maximum ball angular velocity

// Goal scoring thresholds
SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y = 5124.25 UU  // Ball Y beyond this = goal
HOOPS_GOAL_SCORE_THRESHOLD_Z = 270.0 UU          // Ball Z below this in hoop = goal

// Hoops goal XY region (elliptical check)
HOOPS_GOAL_SCALE_Y = 0.9
HOOPS_GOAL_OFFSET_Y = 2770 UU
HOOPS_GOAL_RADIUS = 716 UU
```

**Hoops Goal Detection:**

In Hoops mode, a goal is scored when:
1. Ball Z < 270 UU (below the hoop rim), AND
2. Ball is within the hoop XY region

The XY region check uses a transformed circular test:

```python
def ball_within_hoops_goal(x, y):
    # Transform Y coordinate (scales and offsets toward net)
    dy = abs(y) * 0.9 - 2770
    dist_sq = x*x + dy*dy
    return dist_sq < 716*716  # Inside circle radius 716
```

### Complete Goal Scoring Logic

```python
def is_ball_scored(arena):
    ball_pos = arena.ball.rigid_body.world_transform.origin * BT_TO_UU
    mutator = arena.mutator_config

    if arena.game_mode in [SOCCAR, HEATSEEKER, SNOWDAY]:
        # Ball center must pass the goal line by more than its radius
        return abs(ball_pos.y) > (mutator.goal_base_threshold_y + mutator.ball_radius)

    elif arena.game_mode == HOOPS:
        # Ball must be below rim AND within hoop XY region
        if ball_pos.z < HOOPS_GOAL_SCORE_THRESHOLD_Z:  # 270 UU
            return ball_within_hoops_goal(ball_pos.x, ball_pos.y)
        return False

    elif arena.game_mode == DROPSHOT:
        # Ball must fall through the floor (below by 1.75x radius)
        return ball_pos.z < -(mutator.ball_radius * 1.75)

    return False

def get_scoring_team(ball_pos_y):
    # Positive Y = Orange's side = Blue scores
    # Negative Y = Blue's side = Orange scores
    return BLUE if ball_pos_y > 0 else ORANGE
```

### Supersonic State

```cpp
SUPERSONIC_START_SPEED = 2200.0 UU/s      // Speed to become supersonic
SUPERSONIC_MAINTAIN_MIN_SPEED = 2100.0 UU/s  // Minimum to stay supersonic
SUPERSONIC_MAINTAIN_MAX_TIME = 1.0 s      // Grace period below threshold
```

A car becomes supersonic when speed >= 2200 UU/s. It remains supersonic if:
- Speed stays >= 2200 UU/s, OR
- Speed stays >= 2100 UU/s AND time below 2200 is less than 1 second

### Ball Properties

```cpp
BALL_COLLISION_RADIUS_SOCCAR = 91.25 UU
BALL_COLLISION_RADIUS_HOOPS = 96.3831 UU
BALL_COLLISION_RADIUS_DROPSHOT = 100.2565 UU

BALL_REST_Z = 93.15 UU    // Ball resting height on floor
                         // NOTE: Greater than radius (91.25) due to arena mesh collision margin
BALL_DRAG = 0.03          // Velocity drag coefficient
BALL_FRICTION = 0.35      // Friction with world
BALL_RESTITUTION = 0.6    // Bounciness (coefficient of restitution)
```

### Collision Material Properties

```cpp
// Car-Ball collision
CARBALL_COLLISION_FRICTION = 2.0
CARBALL_COLLISION_RESTITUTION = 0.0    // No bounce!

// Car-Car collision
CARCAR_COLLISION_FRICTION = 0.09
CARCAR_COLLISION_RESTITUTION = 0.1

// Car-World collision
CARWORLD_COLLISION_FRICTION = 0.3
CARWORLD_COLLISION_RESTITUTION = 0.3

// Arena base properties
ARENA_COLLISION_BASE_FRICTION = 0.6
ARENA_COLLISION_BASE_RESTITUTION = 0.3
```

---

## 3. Data Structures

### Vector (Vec)

A 3D vector with SIMD alignment (16-byte aligned):

```cpp
struct Vec {
    float x, y, z;
    float _w;  // Padding for SIMD (always 0)

    // Basic operations
    float Length() { return sqrt(x*x + y*y + z*z); }
    float LengthSq() { return x*x + y*y + z*z; }
    float Dot(Vec other) { return x*other.x + y*other.y + z*other.z; }
    Vec Cross(Vec other) {
        return Vec(
            y*other.z - z*other.y,
            z*other.x - x*other.z,
            x*other.y - y*other.x
        );
    }
    Vec Normalized() {
        float len = Length();
        if (len > EPSILON) return Vec(x/len, y/len, z/len);
        return Vec(0, 0, 0);
    }
    Vec To2D() { return Vec(x, y, 0); }
};
```

### Rotation Matrix (RotMat)

A **column-major** 3x3 rotation matrix:

```cpp
struct RotMat {
    Vec forward;  // First column (X-axis in world space)
    Vec right;    // Second column (Y-axis in world space)
    Vec up;       // Third column (Z-axis in world space)

    static RotMat GetIdentity() {
        return RotMat(Vec(1,0,0), Vec(0,1,0), Vec(0,0,1));
    }

    // Transform vector from local to world space
    Vec TransformVector(Vec local) {
        return forward * local.x + right * local.y + up * local.z;
    }
};
```

### Euler Angles (Angle)

Euler angles in **YPR (Yaw-Pitch-Roll)** order, values in **radians**:

```cpp
struct Angle {
    float yaw;    // Rotation around Z-axis
    float pitch;  // Rotation around Y-axis (after yaw)
    float roll;   // Rotation around X-axis (after yaw and pitch)

    RotMat ToRotMat() {
        // Apply in YPR order
        float cy = cos(yaw), sy = sin(yaw);
        float cp = cos(pitch), sp = sin(pitch);
        float cr = cos(roll), sr = sin(roll);

        RotMat result;
        result.forward = Vec(
            cp * cy,
            cp * sy,
            sp
        );
        result.right = Vec(
            cy*sp*sr - sy*cr,
            sy*sp*sr + cy*cr,
            -cp*sr
        );
        result.up = Vec(
            -cy*sp*cr - sy*sr,
            -sy*sp*cr + cy*sr,
            cp*cr
        );
        return result;
    }

    static Angle FromVec(Vec forward) {
        return Angle(
            atan2(forward.y, forward.x),          // yaw
            atan2(forward.z, forward.To2D().Length()),  // pitch
            0  // roll not determinable from forward vector alone
        );
    }
};
```

### Car Controls

```cpp
struct CarControls {
    // Analog inputs (range: -1.0 to 1.0)
    float throttle;  // Forward/backward acceleration
    float steer;     // Left/right steering
    float pitch;     // Nose up/down (air control)
    float yaw;       // Rotate left/right (air control)
    float roll;      // Roll left/right (air control)

    // Digital inputs
    bool jump;       // Jump button
    bool boost;      // Boost button
    bool handbrake;  // Powerslide/handbrake

    void ClampFix() {
        throttle = clamp(throttle, -1, 1);
        steer = clamp(steer, -1, 1);
        pitch = clamp(pitch, -1, 1);
        yaw = clamp(yaw, -1, 1);
        roll = clamp(roll, -1, 1);
    }
};
```

### Physics State

```cpp
struct PhysState {
    Vec pos;        // Position (center of mass)
    RotMat rotMat;  // Rotation matrix
    Vec vel;        // Linear velocity
    Vec angVel;     // Angular velocity (radians/s)
};
```

---

## 4. Car Configuration and Hitboxes

### Hitbox Types

There are 6 standard hitbox presets (indices 0-5) plus a special 3-wheel variant:

| Index | Name | Usage |
|-------|------|-------|
| 0 | Octane | Most common, balanced |
| 1 | Dominus | Flat, wide |
| 2 | Plank (Batmobile) | Very flat |
| 3 | Breakout | Long, narrow |
| 4 | Hybrid | Similar to Octane |
| 5 | Merc | Tall |
| 6 | Psyclops | 3-wheel behavior |

### Hitbox Dimensions

**Hitbox Sizes (full width, not half-extents):**

```cpp
// Vec(length_X, width_Y, height_Z) in UU
HITBOX_SIZES[7] = {
    Vec(120.507, 86.6994, 38.6591),   // OCTANE
    Vec(130.427, 85.7799, 33.8),      // DOMINUS
    Vec(131.32,  87.1704, 31.8944),   // PLANK
    Vec(133.992, 83.021,  32.8),      // BREAKOUT
    Vec(129.519, 84.6879, 36.6591),   // HYBRID
    Vec(123.22,  79.2103, 44.1591),   // MERC
    Vec(120.641, 86.8334, 38.7931),   // PSYCLOPS (slightly larger)
};
```

**Hitbox Position Offsets (from car origin):**

```cpp
HITBOX_OFFSETS[7] = {
    Vec(13.8757, 0, 20.755),   // OCTANE
    Vec(9.0,     0, 15.75),    // DOMINUS
    Vec(9.00857, 0, 12.0942),  // PLANK
    Vec(12.5,    0, 11.75),    // BREAKOUT
    Vec(13.8757, 0, 20.755),   // HYBRID
    Vec(11.3757, 0, 21.505),   // MERC
    Vec(13.8757, 0, 15.0),     // PSYCLOPS
};
```

**Note:** The hitbox is a **compound shape** with the box offset from the car's center of mass (which is at local origin). This offset places the hitbox slightly forward and above the center.

### Car Rigid Body Setup

The car's physics body has specific configuration:

```python
# Create compound shape with offset hitbox
compound_shape = CompoundShape()
hitbox_offset = Transform.identity()
hitbox_offset.origin = hitbox_pos_offset * UU_TO_BT
compound_shape.add_child_shape(hitbox_offset, box_shape)

# Calculate inertia from the child hitbox (not compound)
local_inertia = box_shape.calculate_local_inertia(CAR_MASS)

# Create rigid body
rigid_body = RigidBody(CAR_MASS, compound_shape, local_inertia)

# IMPORTANT: Disable gyroscopic force
# Rocket League doesn't use gyroscopic torque on cars
rigid_body.rigidbody_flags = 0  # Clears BT_ENABLE_GYROSCOPIC_FORCE

# Enable custom material callbacks for collision handling
rigid_body.collision_flags |= CF_CUSTOM_MATERIAL_CALLBACK

# Set default collision properties
rigid_body.friction = 0.3
rigid_body.restitution = 0.1
```

### Vehicle Coordinate System

The raycast vehicle uses Bullet's coordinate indexing:

```python
# setCoordinateSystem(right, up, forward)
vehicle.set_coordinate_system(1, 2, 0)

# This means:
# Index 0 = X = Forward
# Index 1 = Y = Right
# Index 2 = Z = Up

# Wheel direction vectors (in local space)
wheel_direction = Vec(0, 0, -1)  # Points downward
wheel_axle = Vec(0, -1, 0)       # Points left (for right-hand rule rotation)
```

### Wheel Configuration

Each car has front and back wheel pairs with different parameters:

**Wheel Radii:**

```cpp
FRONT_WHEEL_RADII[7] = {12.50, 12.00, 12.50, 13.50, 12.50, 15.00, 12.50};
BACK_WHEEL_RADII[7]  = {15.00, 13.50, 17.00, 15.00, 15.00, 15.00, 15.00};
```

**Suspension Rest Lengths:**

```cpp
FRONT_WHEEL_SUS_REST[7] = {38.755, 33.95, 31.9242, 29.7, 38.755, 39.505, 33.0};
BACK_WHEEL_SUS_REST[7]  = {37.055, 33.85, 27.9242, 29.666, 37.055, 39.105, 31.3};
```

**Wheel Connection Points (from car center):**

```cpp
// Front wheels (Y is mirrored for left/right)
FRONT_WHEELS_OFFSET[7] = {
    Vec(51.25, 25.90, 20.755),   // OCTANE
    Vec(50.30, 31.10, 15.75),    // DOMINUS
    Vec(49.97, 27.80, 12.0942),  // PLANK
    Vec(51.50, 26.67, 11.75),    // BREAKOUT
    Vec(51.25, 25.90, 20.755),   // HYBRID
    Vec(51.25, 25.90, 21.505),   // MERC
    Vec(51.25, 5.000, 15.000),   // PSYCLOPS (centered front wheel)
};

// Back wheels (Y is mirrored for left/right)
BACK_WHEELS_OFFSET[7] = {
    Vec(-33.75, 29.50, 20.755),  // OCTANE
    Vec(-34.75, 33.00, 15.75),   // DOMINUS
    Vec(-35.43, 20.28, 12.0942), // PLANK
    Vec(-35.75, 35.00, 11.75),   // BREAKOUT
    Vec(-34.00, 29.50, 20.755),  // HYBRID
    Vec(-33.75, 29.50, 21.505),  // MERC
    Vec(-33.75, 29.50, 15.000),  // PSYCLOPS
};
```

### Dodge Deadzone

The minimum combined input magnitude to trigger a flip:

```cpp
dodgeDeadzone = 0.5  // Default for all cars
```

A flip is triggered when: `|pitch| + |yaw| + |roll| >= dodgeDeadzone`

### Flip Reset Mechanics

A "flip reset" occurs when the car regains its flip/double-jump ability after being airborne. The condition is simple:

```python
is_on_ground = (num_wheels_in_contact >= 3)
```

When `is_on_ground` becomes True:
- `hasDoubleJumped` is reset to False
- `hasFlipped` is reset to False
- `airTime` and `airTimeSinceJump` are reset to 0

**Important:** Wheel contact is determined by raycasts, which detect *any* collision object - including the ball. This means landing on the ball with 3+ wheels grants a flip reset (commonly called a "ceiling reset" when done on the ceiling, or "ball reset" when landing on the ball mid-air).

---

## 5. Suspension and Wheel Physics

### Suspension Model

RocketSim uses a **raycast vehicle** model where each wheel casts a ray downward to detect ground contact.

**Suspension Constants:**

```cpp
SUSPENSION_STIFFNESS = 500.0
WHEELS_DAMPING_COMPRESSION = 25.0
WHEELS_DAMPING_RELAXATION = 40.0
MAX_SUSPENSION_TRAVEL = 12.0 UU
SUSPENSION_SUBTRACTION = 0.05

SUSPENSION_FORCE_SCALE_FRONT = 35.75  // ~36 - 0.25
SUSPENSION_FORCE_SCALE_BACK = 54.26   // ~54 + 0.26
```

**IMPORTANT: Suspension Rest Length Adjustment**

When setting up wheels, the stored suspension rest length is **reduced by MAX_SUSPENSION_TRAVEL** before being passed to the vehicle:

```python
# During wheel setup
actual_rest_length = config_rest_length - MAX_SUSPENSION_TRAVEL

vehicle.add_wheel(
    connection_point * UU_TO_BT,
    wheel_direction, wheel_axle,
    actual_rest_length * UU_TO_BT,  # Pre-reduced!
    wheel_radius * UU_TO_BT,
    tuning, True
)
```

This means the wheel starts partially compressed and can extend further than it can compress.

### Wheel Raycast Algorithm

```python
def raycast_wheel(wheel, chassis_transform):
    # Calculate ray start point in world space
    hard_point_ws = chassis_transform * wheel.connection_point_local

    # Ray direction (always straight down in world space, but follows chassis rotation)
    wheel_direction_ws = chassis_transform.rotation * Vec(0, 0, -1)

    # Calculate ray length
    suspension_travel = wheel.max_suspension_travel_cm / 100  # Convert to meters
    ray_length = (wheel.suspension_rest_length + suspension_travel +
                  wheel.radius - SUSPENSION_SUBTRACTION)

    # Cast ray
    ray_end = hard_point_ws + wheel_direction_ws * ray_length
    hit = raycast(hard_point_ws, ray_end, ignore=chassis_body)

    if hit:
        wheel.is_in_contact = True
        wheel.contact_point_ws = hit.point
        wheel.contact_normal_ws = hit.normal

        # Calculate suspension length
        wheel_trace_len = dot(hard_point_ws - hit.point, chassis_up_vector)
        wheel.suspension_length = wheel_trace_len - wheel.radius

        # Clamp suspension length
        min_len = wheel.suspension_rest_length - suspension_travel
        max_len = wheel.suspension_rest_length + suspension_travel
        wheel.suspension_length = clamp(wheel.suspension_length, min_len, max_len)

        # Calculate relative velocity for damping
        denominator = dot(hit.normal, chassis_up_vector)
        if denominator > 0.1:
            contact_rel_pos = hit.point - chassis_center
            velocity_at_contact = chassis_velocity + cross(chassis_angular_vel, contact_rel_pos)
            proj_vel = dot(hit.normal, velocity_at_contact)
            wheel.suspension_relative_velocity = proj_vel / denominator
            wheel.inv_contact_dot_suspension = 1.0 / denominator
        else:
            wheel.suspension_relative_velocity = 0
            wheel.inv_contact_dot_suspension = 10.0
    else:
        wheel.is_in_contact = False
        wheel.suspension_length = wheel.suspension_rest_length + suspension_travel
```

### Suspension Force Calculation

```python
def calculate_suspension_force(wheel):
    if not wheel.is_in_contact:
        return 0

    # Spring force (Hooke's law)
    compression = wheel.suspension_rest_length - wheel.suspension_length
    spring_force = compression * SUSPENSION_STIFFNESS * wheel.inv_contact_dot_suspension

    # Damping force
    if wheel.suspension_relative_velocity < 0:
        damping_scale = WHEELS_DAMPING_COMPRESSION
    else:
        damping_scale = WHEELS_DAMPING_RELAXATION

    damping_force = damping_scale * wheel.suspension_relative_velocity

    # Total force (never negative - no pulling down)
    total_force = spring_force - damping_force
    total_force *= wheel.suspension_force_scale  # Front vs back scaling

    if total_force < 0:
        total_force = 0

    return total_force
```

### Suspension Force Application

```python
def apply_suspension_forces(vehicle, dt):
    for wheel in vehicle.wheels:
        if wheel.suspension_force > 0:
            contact_offset = wheel.contact_point_ws - chassis.center_of_mass

            # Force includes suspension force and extra pushback
            force_magnitude = wheel.suspension_force * dt + wheel.extra_pushback
            force = wheel.contact_normal_ws * force_magnitude

            # Apply as impulse at contact point
            chassis.apply_impulse(force, contact_offset)
```

### Extra Pushback (Collision Penetration Resolution)

When a wheel is compressed beyond its rest position against a static object:

```python
def calculate_extra_pushback(wheel, hit, chassis):
    if not hit.object.is_static:
        return 0

    ray_pushback_thresh = (wheel.suspension_rest_length + wheel.radius -
                           SUSPENSION_SUBTRACTION)

    wheel_trace_len = dot(wheel.hard_point_ws - hit.point, chassis.up_vector)

    if wheel_trace_len < ray_pushback_thresh:
        delta = wheel_trace_len - ray_pushback_thresh

        # Use Bullet's collision resolution
        collision_impulse = resolve_single_collision(
            chassis, hit.object, hit.point, hit.normal, delta
        )

        return collision_impulse / num_wheels

    return 0
```

---

## 6. Ground Movement and Driving

### Steering Angle

The maximum steering angle decreases with speed. This is implemented via a **piecewise linear curve**:

**Standard Steering Curve:**

| Speed (UU/s) | Max Angle (radians) | Degrees |
|--------------|---------------------|---------|
| 0 | 0.53356 | 30.6° |
| 500 | 0.31930 | 18.3° |
| 1000 | 0.18203 | 10.4° |
| 1500 | 0.10570 | 6.1° |
| 1750 | 0.08507 | 4.9° |
| 3000 | 0.03454 | 2.0° |

**Powerslide Extended Steering:**

| Speed (UU/s) | Max Angle (radians) |
|--------------|---------------------|
| 0 | 0.39235 |
| 2500 | 0.12610 |

**Three-Wheel Steering Curve (Psyclops):**

| Speed (UU/s) | Max Angle (radians) |
|--------------|---------------------|
| 0 | 0.342473 |
| 2300 | 0.034837 |

```python
def calculate_steer_angle(forward_speed, steer_input, handbrake_val, is_three_wheel):
    abs_speed = abs(forward_speed)

    # Base steering angle from speed curve
    if is_three_wheel:
        base_angle = THREE_WHEEL_STEER_CURVE.get_output(abs_speed)
    else:
        base_angle = STEER_ANGLE_CURVE.get_output(abs_speed)

    # Interpolate with powerslide angle based on handbrake value
    if handbrake_val > 0:
        powerslide_angle = POWERSLIDE_STEER_CURVE.get_output(abs_speed)
        base_angle += (powerslide_angle - base_angle) * handbrake_val

    return base_angle * steer_input
```

### Throttle and Brake Forces

**Engine/Throttle Constants:**

```cpp
THROTTLE_TORQUE_AMOUNT = CAR_MASS * 400.0 = 72000 UU
BRAKE_TORQUE_AMOUNT = CAR_MASS * 14.333... = 2580 UU
STOPPING_FORWARD_VEL = 25.0 UU/s
COASTING_BRAKE_FACTOR = 0.15
THROTTLE_DEADZONE = 0.001
```

**Drive Speed Torque Factor Curve:**

| Speed (UU/s) | Factor |
|--------------|--------|
| 0 | 1.0 |
| 1400 | 0.1 |
| 1410 | 0.0 |

Engine force diminishes as speed increases, reaching zero at 1410 UU/s.

```python
def update_throttle_brake(car, controls, forward_speed):
    abs_speed = abs(forward_speed)
    real_throttle = controls.throttle
    real_brake = 0

    # Boost overrides throttle
    if controls.boost and car.boost > 0:
        real_throttle = 1.0

    if not controls.handbrake:
        abs_throttle = abs(real_throttle)

        if abs_throttle >= THROTTLE_DEADZONE:
            # Check if reversing direction
            if abs_speed > STOPPING_FORWARD_VEL and sign(real_throttle) != sign(forward_speed):
                # Full brake when trying to go opposite direction
                real_brake = 1.0

                if abs_speed > 0.01:
                    real_throttle = 0  # Can't throttle while braking at speed
        else:
            # Coasting (no throttle input)
            real_throttle = 0

            if abs_speed < STOPPING_FORWARD_VEL:
                real_brake = 1.0  # Full stop
            else:
                real_brake = COASTING_BRAKE_FACTOR

    # Calculate forces
    drive_speed_scale = DRIVE_TORQUE_CURVE.get_output(abs_speed)

    # Reduced torque with fewer wheels on ground
    if num_wheels_in_contact < 3:
        drive_speed_scale /= 4

    engine_force = real_throttle * THROTTLE_TORQUE_AMOUNT * drive_speed_scale
    brake_force = real_brake * BRAKE_TORQUE_AMOUNT

    return engine_force, brake_force
```

### Wheel Friction

**Lateral Friction Curve:**

| Slip Ratio | Friction |
|------------|----------|
| 0 | 1.0 |
| 1 | 0.2 |

**Three-Wheel Lateral Friction Curve (Psyclops):**

| Slip Ratio | Friction |
|------------|----------|
| 0 | 0.30 |
| 1 | 0.25 |

**Longitudinal Friction Curve:**

Empty curve - returns default value of 1.0 for all inputs.

**Handbrake Friction Modifiers:**

```python
HANDBRAKE_LAT_FRICTION_FACTOR = 0.1  # Constant (single point curve)
HANDBRAKE_LONG_FRICTION_CURVE = {0: 0.5, 1: 0.9}  # Interpolated
```

**Non-Sticky Friction Curve (when not throttling):**

| Normal Z | Factor |
|----------|--------|
| 0 | 0.1 |
| 0.7075 | 0.5 |
| 1 | 1.0 |

```python
def calculate_wheel_friction(wheel, car, throttle, is_three_wheel):
    if not wheel.has_ground_contact:
        return

    # Get wheel direction vectors
    lat_dir = wheel.world_transform.basis.column(1)  # Y-axis (right)
    long_dir = cross(lat_dir, wheel.contact_normal)   # Forward along surface

    # Calculate velocity at wheel hard point (not contact point!)
    wheel_delta = wheel.hard_point_ws - car.rigid_body.world_transform.origin
    cross_vec = (cross(car.angular_velocity, wheel_delta) + car.velocity) * BT_TO_UU

    # Calculate lateral slip ratio for friction curve input
    base_friction = abs(dot(cross_vec, lat_dir))

    # Only calculate slip ratio if there's significant lateral velocity
    if base_friction > 5:
        # Slip ratio = lateral / (longitudinal + lateral)
        # This gives 0 when going straight, approaching 1 when sliding sideways
        friction_input = base_friction / (abs(dot(cross_vec, long_dir)) + base_friction)
    else:
        friction_input = 0

    # Get base friction values from curves
    if is_three_wheel:
        lat_friction = LAT_FRICTION_CURVE_THREEWHEEL.get_output(friction_input)
    else:
        lat_friction = LAT_FRICTION_CURVE.get_output(friction_input)

    long_friction = LONG_FRICTION_CURVE.get_output(friction_input)  # Returns 1.0 (empty curve)

    # Apply handbrake modifiers using interpolation formula
    if car.handbrake_val > 0:
        # Formula: friction *= (factor - 1) * handbrake_val + 1
        # When handbrake_val=0: friction *= 1 (unchanged)
        # When handbrake_val=1: friction *= factor
        lat_factor = HANDBRAKE_LAT_FRICTION_FACTOR_CURVE.get_output(friction_input)  # 0.1
        long_factor = HANDBRAKE_LONG_FRICTION_FACTOR_CURVE.get_output(friction_input)

        lat_friction *= (lat_factor - 1) * car.handbrake_val + 1
        long_friction *= (long_factor - 1) * car.handbrake_val + 1
    else:
        # When NOT powersliding, longitudinal friction is forced to 1
        long_friction = 1.0

    # Apply non-sticky reduction when not throttling
    # Based on contact normal Z (how vertical the surface is)
    if throttle == 0:
        non_sticky_scale = NON_STICKY_FRICTION_FACTOR_CURVE.get_output(wheel.contact_normal.z)
        lat_friction *= non_sticky_scale
        long_friction *= non_sticky_scale

    wheel.lat_friction = lat_friction
    wheel.long_friction = long_friction
```

**Friction Curve Input Explained:**

The friction curve input is a "slip ratio" measuring how much the car is sliding sideways vs moving forward:
- 0 = Moving straight (all longitudinal velocity)
- 1 = Sliding completely sideways (all lateral velocity)

This slip ratio determines grip reduction - more sideways sliding = less lateral grip (friction decreases from 1.0 toward 0.2).

The 5 UU/s threshold prevents tiny movements from triggering grip reduction.

### Friction Impulse Calculation

```python
def calculate_friction_impulses(vehicle, dt):
    friction_scale = vehicle.chassis.mass / 3  # Mass scaling factor

    for wheel in vehicle.wheels:
        if not wheel.has_contact:
            wheel.impulse = Vec(0, 0, 0)
            continue

        ground_object = wheel.ground_object

        # Get wheel axle direction (includes steering rotation)
        # Axle is the local Y-axis (right) of the wheel transform
        axle_dir = wheel.world_transform.basis.column(1)  # Right axis

        surface_normal = wheel.contact_normal

        # Project axle onto contact surface (remove normal component)
        proj = dot(axle_dir, surface_normal)
        axle_dir = normalize(axle_dir - surface_normal * proj)

        # Forward direction is perpendicular to both axle and surface
        forward_dir = normalize(cross(surface_normal, axle_dir))

        # Calculate side impulse using Bullet's bilateral constraint
        side_impulse = resolve_single_bilateral(
            vehicle.chassis, ground_object,
            wheel.contact_point, wheel.contact_point,
            axle_dir
        )

        # Calculate rolling friction
        ROLLING_FRICTION_SCALE_MAGIC = 113.73963

        if wheel.engine_force == 0:
            if wheel.brake > 0:
                # Braking friction
                car_rel_contact = wheel.contact_point - vehicle.chassis.center_of_mass
                ground_rel_contact = wheel.contact_point - ground_object.center_of_mass

                v1 = vehicle.chassis.get_velocity_in_local_point(car_rel_contact)
                v2 = ground_object.get_velocity_in_local_point(car_rel_contact)
                contact_vel = v1 - v2
                rel_vel = dot(contact_vel, forward_dir)

                # Low tick rate threshold to prevent stuttering
                if dt > 1/80:
                    threshold = -(1 / (dt * 150)) + 0.8
                    if abs(rel_vel) < threshold:
                        rel_vel = 0

                rolling_friction = clamp(
                    -rel_vel * ROLLING_FRICTION_SCALE_MAGIC,
                    -wheel.brake, wheel.brake
                )
            else:
                rolling_friction = 0
        else:
            # Engine force already accounts for mass, cancel out friction_scale
            rolling_friction = -wheel.engine_force / friction_scale

        # Combine friction forces with friction coefficients
        total_friction = (forward_dir * rolling_friction * wheel.long_friction +
                         axle_dir * side_impulse * wheel.lat_friction)

        wheel.impulse = total_friction * friction_scale
```

### Friction Impulse Application

Friction impulses are applied at a modified contact position that excludes the vertical component:

```python
def apply_friction_impulses(vehicle, dt):
    up_dir = vehicle.chassis.world_transform.basis.column(2)  # Up axis

    for wheel in vehicle.wheels:
        if not wheel.impulse.is_zero():
            # Calculate contact offset from chassis center
            wheel_contact_offset = wheel.contact_point - vehicle.chassis.world_transform.origin

            # Remove vertical component of offset
            # This prevents wheel contact from creating pitch torque
            contact_up_dot = dot(up_dir, wheel_contact_offset)
            wheel_rel_pos = wheel_contact_offset - up_dir * contact_up_dot

            # Apply impulse at horizontal-only position
            vehicle.chassis.apply_impulse(wheel.impulse * dt, wheel_rel_pos)
```

**Note:** Removing the vertical component from the impulse position is crucial - it prevents wheel friction from creating unwanted pitch moments.

### Sticky Force

When wheels are in contact with the world, a "sticky" force pulls the car toward the surface:

```python
def apply_sticky_force(vehicle, throttle, forward_speed):
    # Get average contact normal from all wheels
    upward_dir = vehicle.get_upwards_dir_from_wheel_contacts()
    if upward_dir.is_zero():
        return

    full_stick = (throttle != 0) or (abs(forward_speed) > STOPPING_FORWARD_VEL)

    # Three-wheel cars (Psyclops) have no base sticky force
    if vehicle.is_three_wheel:
        sticky_scale = 0.0
    else:
        sticky_scale = 0.5

    # When throttling or moving fast, add extra stick based on surface angle
    # This helps cars stick to walls and ceilings
    if full_stick:
        sticky_scale += 1 - abs(upward_dir.z)

    sticky_force = upward_dir * sticky_scale * GRAVITY_Z * CAR_MASS
    vehicle.chassis.apply_central_force(sticky_force)
```

**Note:** The sticky force direction is the average contact normal from all wheels, effectively pulling the car "into" the surface. The scale increases on walls/ceilings (where `upward_dir.z` is closer to 0) to help cars drive on those surfaces.

### Powerslide (Handbrake) State

Handbrake has analog rise/fall behavior:

```cpp
POWERSLIDE_RISE_RATE = 5.0   // Per second when holding
POWERSLIDE_FALL_RATE = 2.0   // Per second when released
```

```python
def update_handbrake(car, controls, dt):
    if controls.handbrake:
        car.handbrake_val += POWERSLIDE_RISE_RATE * dt
    else:
        car.handbrake_val -= POWERSLIDE_FALL_RATE * dt

    car.handbrake_val = clamp(car.handbrake_val, 0, 1)
```

---

## 7. Jump and Flip Mechanics

### Jump Constants

```cpp
JUMP_IMMEDIATE_FORCE = 291.67 UU/s  (875 / 3)
JUMP_ACCEL = 1458.33 UU/s^2         (4375 / 3)
JUMP_MIN_TIME = 0.025 s
JUMP_MAX_TIME = 0.2 s
JUMP_RESET_TIME_PAD = 0.025 s       (1/40)
```

### Jump Mechanics

```python
def update_jump(car, controls, dt):
    jump_pressed = controls.jump and not car.last_controls.jump

    # Reset jump state when grounded
    if car.is_on_ground and not car.is_jumping:
        if car.has_jumped and car.jump_time < JUMP_MIN_TIME + JUMP_RESET_TIME_PAD:
            pass  # Don't reset yet - prevents ground-touch reset bug
        else:
            car.has_jumped = False
            car.jump_time = 0

    # Continue or end jump
    if car.is_jumping:
        if car.jump_time < JUMP_MIN_TIME or (controls.jump and car.jump_time < JUMP_MAX_TIME):
            car.is_jumping = True  # Continue jumping
        else:
            car.is_jumping = False  # End jump
    elif car.is_on_ground and jump_pressed:
        # Start new jump
        car.is_jumping = True
        car.jump_time = 0

        # Apply immediate impulse in car's up direction
        jump_impulse = car.up_dir * JUMP_IMMEDIATE_FORCE * CAR_MASS
        car.rigid_body.apply_central_impulse(jump_impulse)

    # Apply sustained jump force
    if car.is_jumping:
        car.has_jumped = True

        jump_force = car.up_dir * JUMP_ACCEL

        # Reduced force before minimum time
        if car.jump_time < JUMP_MIN_TIME:
            jump_force *= 0.62

        car.rigid_body.apply_central_force(jump_force * CAR_MASS)

    # Update timer
    if car.is_jumping or car.has_jumped:
        car.jump_time += dt
```

### Flip Constants

```cpp
DOUBLEJUMP_MAX_DELAY = 1.25 s       // Time window after jump
FLIP_INITIAL_VEL_SCALE = 500.0 UU/s
FLIP_TORQUE_TIME = 0.65 s           // Duration of flip rotation
FLIP_TORQUE_MIN_TIME = 0.41 s
FLIP_PITCHLOCK_TIME = 1.0 s
FLIP_PITCHLOCK_EXTRA_TIME = 0.3 s
FLIP_TORQUE_X = 260.0               // Lateral flip torque
FLIP_TORQUE_Y = 224.0               // Forward/back flip torque

// Flip velocity scaling based on current speed
FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE = 1.0
FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE = 1.9
FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE = 2.5
FLIP_BACKWARD_IMPULSE_SCALE_X = 1.0667  (16/15)

// Z-velocity damping during flip
FLIP_Z_DAMP_120 = 0.35              // Damping factor at 120 TPS
FLIP_Z_DAMP_START = 0.15 s          // When to start damping
FLIP_Z_DAMP_END = 0.21 s            // When to stop damping
```

### Flip/Dodge Detection

```python
def check_flip_input(controls, config):
    input_magnitude = abs(controls.pitch) + abs(controls.yaw) + abs(controls.roll)
    return input_magnitude >= config.dodge_deadzone
```

### Flip Direction Calculation

```python
def calculate_flip_direction(controls):
    # Flip direction in car's local XY plane
    # X: forward/backward (negative pitch = forward flip)
    # Y: left/right (positive yaw/roll = right flip)

    dodge_dir = Vec(-controls.pitch, controls.yaw + controls.roll, 0)

    # Apply deadzone to prevent tiny movements
    if abs(controls.yaw + controls.roll) < 0.1 and abs(controls.pitch) < 0.1:
        return Vec(0, 0, 0)  # Stall flip

    return normalize(dodge_dir)
```

### Flip Velocity Impulse

```python
def apply_flip_velocity(car, dodge_dir, forward_speed):
    if dodge_dir.is_zero():
        return  # Stall flip - no velocity impulse

    forward_speed_ratio = abs(forward_speed) / CAR_MAX_SPEED

    # Determine if this is a backwards dodge
    if abs(forward_speed) < 100:
        is_backwards = dodge_dir.x < 0
    else:
        is_backwards = (dodge_dir.x >= 0) != (forward_speed >= 0)

    # Start with base flip velocity
    flip_vel = dodge_dir * FLIP_INITIAL_VEL_SCALE

    # Scale based on speed and direction
    if is_backwards:
        max_scale_x = FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE  # 2.5
    else:
        max_scale_x = FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE   # 1.0

    flip_vel.x *= 1 + (max_scale_x - 1) * forward_speed_ratio
    flip_vel.y *= 1 + (FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE - 1) * forward_speed_ratio

    if is_backwards:
        flip_vel.x *= FLIP_BACKWARD_IMPULSE_SCALE_X  # 16/15

    # Convert to world coordinates (2D only)
    forward_2d = car.forward_dir.To2D().Normalized()
    right_2d = Vec(-forward_2d.y, forward_2d.x, 0)

    world_flip_vel = forward_2d * flip_vel.x + right_2d * flip_vel.y

    car.rigid_body.apply_central_impulse(world_flip_vel * CAR_MASS)
```

### Flip Torque Application

```python
def apply_flip_torque(car, dt):
    if not car.is_flipping or car.flip_rel_torque.is_zero():
        return

    rel_torque = car.flip_rel_torque

    # Flip cancel check (pitch input opposes flip direction)
    pitch_scale = 1.0
    if rel_torque.y != 0 and controls.pitch != 0:
        if sign(rel_torque.y) == sign(controls.pitch):
            pitch_scale = 1 - abs(controls.pitch)

    rel_torque.y *= pitch_scale

    # Apply torque in world space
    world_torque = Vec(
        rel_torque.x * FLIP_TORQUE_X,
        rel_torque.y * FLIP_TORQUE_Y,
        0
    )

    # Transform to world space
    world_torque = car.rotation_matrix * world_torque

    # Apply through inverse inertia tensor
    car.rigid_body.apply_torque(car.inverse_inertia_world * world_torque)
```

### Z-Velocity Damping During Flip

```python
def apply_flip_z_damping(car, dt):
    if not car.is_flipping:
        return

    if car.flip_time < FLIP_TORQUE_TIME:
        if car.flip_time >= FLIP_Z_DAMP_START:
            if car.velocity.z < 0 or car.flip_time < FLIP_Z_DAMP_END:
                # Damp Z velocity
                damp_factor = pow(1 - FLIP_Z_DAMP_120, dt / (1/120))
                car.velocity.z *= damp_factor
```

### Complete Flip State Machine

```python
def update_double_jump_or_flip(car, controls, dt, forward_speed):
    jump_pressed = controls.jump and not car.last_controls.jump

    if car.is_on_ground:
        car.has_double_jumped = False
        car.has_flipped = False
        car.air_time = 0
        car.air_time_since_jump = 0
        car.flip_time = 0
        return

    # Update air time
    car.air_time += dt

    if car.has_jumped and not car.is_jumping:
        car.air_time_since_jump += dt
    else:
        car.air_time_since_jump = 0

    # Check for flip/double-jump
    if jump_pressed and car.air_time_since_jump < DOUBLEJUMP_MAX_DELAY:
        is_flip_input = check_flip_input(controls, car.config)

        can_use = (not car.has_double_jumped and not car.has_flipped)

        if can_use:
            if is_flip_input:
                # Start flip
                car.flip_time = 0
                car.has_flipped = True
                car.is_flipping = True

                dodge_dir = calculate_flip_direction(controls)
                car.flip_rel_torque = Vec(-dodge_dir.y, dodge_dir.x, 0)

                apply_flip_velocity(car, dodge_dir, forward_speed)
            else:
                # Double jump
                jump_impulse = car.up_dir * JUMP_IMMEDIATE_FORCE * CAR_MASS
                car.rigid_body.apply_central_impulse(jump_impulse)
                car.has_double_jumped = True

    # Update flip state
    if car.is_flipping:
        car.flip_time += dt
        apply_flip_z_damping(car, dt)

        if car.flip_time >= FLIP_TORQUE_TIME:
            car.is_flipping = False
    elif car.has_flipped:
        car.flip_time += dt  # Continue counting for pitch lock
```

---

## 8. Air Control and Aerial Movement

### Air Control Constants

```cpp
CAR_AIR_CONTROL_TORQUE = Vec(130, 95, 400)   // Pitch, Yaw, Roll
CAR_AIR_CONTROL_DAMPING = Vec(30, 20, 50)    // Pitch, Yaw, Roll

CAR_TORQUE_SCALE = 2 * PI / 65536 * 1000 = 0.09587...

THROTTLE_AIR_ACCEL = 66.67 UU/s^2            // (200/3)
```

### Air Torque Application

```python
def update_air_torque(car, controls, dt, is_fully_airborne):
    # Direction vectors for each rotation axis
    pitch_axis = -car.right_dir    # Pitch around negative right
    yaw_axis = car.up_dir          # Yaw around up
    roll_axis = -car.forward_dir   # Roll around negative forward

    # Check if we can do air control
    do_air_control = True

    if car.is_flipping:
        # During flip, check if torque time is over
        car.is_flipping = car.has_flipped and car.flip_time < FLIP_TORQUE_TIME

        if car.is_flipping:
            apply_flip_torque(car, dt)

            # Can still yaw/roll during flip, but pitch only for cancel
            if car.flip_rel_torque.is_zero():
                do_air_control = True
            else:
                do_air_control = (sign(car.flip_rel_torque.y) == sign(controls.pitch))

    if not do_air_control or car.is_auto_flipping or not is_fully_airborne:
        return

    # Calculate pitch torque scale (locked during and after flip)
    pitch_torque_scale = 1.0
    if car.is_flipping:
        pitch_torque_scale = 0
    elif car.has_flipped and car.flip_time < FLIP_TORQUE_TIME + FLIP_PITCHLOCK_EXTRA_TIME:
        pitch_torque_scale = 0

    # Calculate control torque
    torque = Vec(0, 0, 0)
    if controls.pitch or controls.yaw or controls.roll:
        torque = (pitch_axis * controls.pitch * pitch_torque_scale * CAR_AIR_CONTROL_TORQUE.x +
                  yaw_axis * controls.yaw * CAR_AIR_CONTROL_TORQUE.y +
                  roll_axis * controls.roll * CAR_AIR_CONTROL_TORQUE.z)

    # Calculate damping (reduces when giving input)
    angular_vel = car.angular_velocity

    damp_pitch = (dot(pitch_axis, angular_vel) * CAR_AIR_CONTROL_DAMPING.x *
                  (1 - abs(controls.pitch * pitch_torque_scale)))
    damp_yaw = (dot(yaw_axis, angular_vel) * CAR_AIR_CONTROL_DAMPING.y *
                (1 - abs(controls.yaw)))
    damp_roll = dot(roll_axis, angular_vel) * CAR_AIR_CONTROL_DAMPING.z

    damping = pitch_axis * damp_pitch + yaw_axis * damp_yaw + roll_axis * damp_roll

    # Apply combined torque
    final_torque = (torque - damping) * CAR_TORQUE_SCALE
    car.rigid_body.apply_torque(car.inverse_inertia_world * final_torque)

    # Air throttle (forward thrust while airborne)
    if controls.throttle != 0:
        thrust = car.forward_dir * controls.throttle * THROTTLE_AIR_ACCEL * CAR_MASS
        car.rigid_body.apply_central_force(thrust)
```

### Auto-Flip (Turtle Recovery)

When the car lands on its roof and jump is pressed:

```cpp
CAR_AUTOFLIP_IMPULSE = 200 UU/s
CAR_AUTOFLIP_TORQUE = 50
CAR_AUTOFLIP_TIME = 0.4 s
CAR_AUTOFLIP_NORMZ_THRESH = 0.7071  // sqrt(0.5)
CAR_AUTOFLIP_ROLL_THRESH = 2.8 rad  // ~160 degrees
```

```python
def update_auto_flip(car, controls, dt):
    jump_pressed = controls.jump and not car.last_controls.jump

    if (jump_pressed and
        car.world_contact.has_contact and
        car.world_contact.normal.z > CAR_AUTOFLIP_NORMZ_THRESH):

        roll_angle = get_roll_angle(car.rotation_matrix)
        abs_roll = abs(roll_angle)

        if abs_roll > CAR_AUTOFLIP_ROLL_THRESH:
            car.auto_flip_timer = CAR_AUTOFLIP_TIME * (abs_roll / PI)
            car.auto_flip_torque_scale = 1 if roll_angle > 0 else -1
            car.is_auto_flipping = True

            # Impulse away from ground
            impulse = -car.up_dir * CAR_AUTOFLIP_IMPULSE * CAR_MASS
            car.rigid_body.apply_central_impulse(impulse)

    if car.is_auto_flipping:
        if car.auto_flip_timer <= 0:
            car.is_auto_flipping = False
        else:
            # Apply roll torque
            car.angular_velocity += (car.forward_dir *
                                     CAR_AUTOFLIP_TORQUE *
                                     car.auto_flip_torque_scale * dt)
            car.auto_flip_timer -= dt
```

### Auto-Roll (Recovery Assistance)

**Trigger Conditions:**

Auto-roll activates when ALL of these are true:
1. `throttle != 0` (player is pressing gas/reverse)
2. Either:
   - `0 < num_wheels_in_contact < 4` (1-3 wheels touching), OR
   - `world_contact.has_contact` (body touching world but not via wheels)

```python
if controls.throttle and ((0 < num_wheels_in_contact < 4) or world_contact.has_contact):
    update_auto_roll(...)
```

**Constants:**

```cpp
CAR_AUTOROLL_FORCE = 100
CAR_AUTOROLL_TORQUE = 80
```

```python
def update_auto_roll(car, controls, dt, num_wheels_in_contact):
    if controls.throttle == 0:
        return

    # Get ground up direction
    if num_wheels_in_contact > 0:
        ground_up = car.vehicle.get_upwards_dir_from_wheel_contacts()
    else:
        ground_up = car.world_contact.normal

    ground_down = -ground_up

    # Calculate how misaligned the car is
    cross_right = cross(ground_up, car.forward_dir)
    cross_forward = cross(ground_down, cross_right)

    right_factor = 1 - clamp(dot(car.right_dir, cross_right), 0, 1)
    forward_factor = 1 - clamp(dot(car.forward_dir, cross_forward), 0, 1)

    # Determine torque direction
    torque_dir_right = car.forward_dir * (-1 if dot(car.right_dir, ground_up) >= 0 else 1)
    torque_dir_forward = car.right_dir * (1 if dot(car.forward_dir, ground_up) >= 0 else -1)

    torque_right = torque_dir_right * right_factor
    torque_forward = torque_dir_forward * forward_factor

    # Apply force and torque
    car.rigid_body.apply_central_force(ground_down * CAR_AUTOROLL_FORCE * CAR_MASS)
    car.rigid_body.apply_torque(car.inverse_inertia_world *
                                (torque_forward + torque_right) * CAR_AUTOROLL_TORQUE)
```

---

## 9. Boost System

### Boost Constants

```cpp
BOOST_MAX = 100.0
BOOST_SPAWN_AMOUNT = 33.33        // (100/3) Starting boost
BOOST_USED_PER_SECOND = 33.33     // (100/3) Consumption rate
BOOST_MIN_TIME = 0.1 s            // Minimum boost duration

BOOST_ACCEL_GROUND = 991.67 UU/s^2   // (2975/3)
BOOST_ACCEL_AIR = 1058.33 UU/s^2     // (3175/3)

// Recharge (optional mutator)
RECHARGE_BOOST_PER_SECOND = 10
RECHARGE_BOOST_DELAY = 0.25 s
```

### Boost State Machine

```python
def update_boost(car, controls, dt, forward_speed):
    has_boost = car.boost > 0

    if has_boost:
        if car.is_boosting:
            # Continue if holding boost OR minimum time not reached
            if controls.boost or car.boosting_time < BOOST_MIN_TIME:
                car.is_boosting = True
            else:
                car.is_boosting = False
        else:
            # Start boost on input
            if controls.boost:
                car.is_boosting = True
    else:
        car.is_boosting = False

    # Update boost timer
    if car.is_boosting:
        car.boosting_time += dt
    else:
        car.boosting_time = 0

    # Apply boost force and consume boost
    if car.is_boosting:
        car.boost = max(0, car.boost - BOOST_USED_PER_SECOND * dt)

        accel = BOOST_ACCEL_GROUND if car.is_on_ground else BOOST_ACCEL_AIR
        boost_force = car.forward_dir * accel * CAR_MASS
        car.rigid_body.apply_central_force(boost_force)

        car.time_since_boosted = 0
    else:
        car.time_since_boosted += dt

        # Optional: Recharge boost
        if mutator.recharge_enabled:
            if car.time_since_boosted >= RECHARGE_BOOST_DELAY:
                car.boost += RECHARGE_BOOST_PER_SECOND * dt

    car.boost = min(car.boost, BOOST_MAX)
```

### Boost Pads

**Boost Pad Properties:**

| Type | Radius | Height | Boost Amount | Cooldown |
|------|--------|--------|--------------|----------|
| Big | 208 UU | 95 UU | 100 | 10 s |
| Small | 144 UU | 95 UU | 12 | 4 s |

**Box Collision (for locked-on pickup):**

| Type | Box Radius | Box Height |
|------|------------|------------|
| Big | 160 UU | 64 UU |
| Small | 120 UU | 64 UU |

**Pickup Detection:**

Initial detection uses cylinder collision (CYL_RAD). Once a car is "locked" to a pad, subsequent frames use box collision (BOX_RAD) with AABB intersection for hysteresis.

**Soccar Boost Pad Locations (28 Small):**

```
Small Pads (x, y, z):
(    0, -4240, 70), (-1792, -4184, 70), ( 1792, -4184, 70),
( -940, -3308, 70), (  940, -3308, 70), (    0, -2816, 70),
(-3584, -2484, 70), ( 3584, -2484, 70), (-1788, -2300, 70),
( 1788, -2300, 70), (-2048, -1036, 70), (    0, -1024, 70),
( 2048, -1036, 70), (-1024,     0, 70), ( 1024,     0, 70),
(-2048,  1036, 70), (    0,  1024, 70), ( 2048,  1036, 70),
(-1788,  2300, 70), ( 1788,  2300, 70), (-3584,  2484, 70),
( 3584,  2484, 70), (    0,  2816, 70), ( -940,  3308, 70),
(  940,  3308, 70), (-1792,  4184, 70), ( 1792,  4184, 70),
(    0,  4240, 70)
```

**Soccar Boost Pad Locations (6 Big):**

```
Big Pads (x, y, z):
(-3584,     0, 73), ( 3584,     0, 73),
(-3072,  4096, 73), ( 3072,  4096, 73),
(-3072, -4096, 73), ( 3072, -4096, 73)
```

**Hoops Boost Pad Locations (14 Small):**

```
Small Pads (x, y, z):
( 1536, -1024, 64), (-1280, -2304, 64), (    0, -2816, 64),
(-1536, -1024, 64), ( 1280, -2304, 64), ( -512,   512, 64),
(-1536,  1024, 64), ( 1536,  1024, 64), ( 1280,  2304, 64),
(    0,  2816, 64), (  512,   512, 64), (  512,  -512, 64),
( -512,  -512, 64), (-1280,  2304, 64)
```

**Hoops Boost Pad Locations (6 Big):**

```
Big Pads (x, y, z):
(-2176,  2944, 72), ( 2176, -2944, 72), (-2176, -2944, 72),
(-2432,     0, 72), ( 2432,     0, 72), ( 2176,  2944, 72)
```

### Car Spawn and Respawn Positions

**Spawn Constants:**

```cpp
CAR_SPAWN_REST_Z = 17.0 UU    // Z position for spawned cars at rest
CAR_RESPAWN_Z = 36.0 UU       // Z position for respawned cars
DEMO_RESPAWN_TIME = 3.0 s     // Time before car respawns after demolition
```

**Orange Team Transformation:**

All spawn/respawn positions are defined for Blue team. For Orange team, apply this transformation:

```python
def transform_for_orange_team(position, yaw):
    # Mirror position across both X and Y axes
    orange_pos = Vec(-position.x, -position.y, position.z)

    # Rotate yaw by 180 degrees
    orange_yaw = yaw + PI

    return orange_pos, orange_yaw
```

**Soccar Spawn Locations (Blue Team):**

| Index | X | Y | Yaw (rad) |
|-------|---|---|-----------|
| 0 | -2048 | -2560 | π/4 (45°) |
| 1 | 2048 | -2560 | 3π/4 (135°) |
| 2 | -256 | -3840 | π/2 (90°) |
| 3 | 256 | -3840 | π/2 (90°) |
| 4 | 0 | -4608 | π/2 (90°) |

**Soccar Respawn Locations (Blue Team):**

| Index | X | Y | Yaw (rad) |
|-------|---|---|-----------|
| 0 | -2304 | -4608 | π/2 |
| 1 | -2688 | -4608 | π/2 |
| 2 | 2304 | -4608 | π/2 |
| 3 | 2688 | -4608 | π/2 |

**Heatseeker Spawn Locations (Blue Team):**

| Index | X | Y | Yaw (rad) |
|-------|---|---|-----------|
| 0 | -1000 | -4620 | π/2 |
| 1 | 1000 | -4620 | π/2 |
| 2 | -2000 | -4620 | π/2 |
| 3 | 2000 | -4620 | π/2 |

**Hoops Spawn Locations (Blue Team):**

| Index | X | Y | Yaw (rad) |
|-------|---|---|-----------|
| 0 | -1536 | -3072 | π/2 |
| 1 | 1536 | -3072 | π/2 |
| 2 | -256 | -2816 | π/2 |
| 3 | 256 | -2816 | π/2 |
| 4 | 0 | -3200 | π/2 |

**Hoops Respawn Locations (Blue Team):**

| Index | X | Y | Yaw (rad) |
|-------|---|---|-----------|
| 0 | -1920 | -3072 | π/2 |
| 1 | -1152 | -3072 | π/2 |
| 2 | 1920 | -3072 | π/2 |
| 3 | 1152 | -3072 | π/2 |

**Dropshot Spawn Locations (Blue Team):**

| Index | X | Y | Yaw (rad) |
|-------|---|---|-----------|
| 0 | -1867 | -2380 | π/4 |
| 1 | 1867 | -2380 | 3π/4 |
| 2 | -256 | -3576 | π/2 |
| 3 | 256 | -3576 | π/2 |
| 4 | 0 | -4088 | π/2 |

**Dropshot Respawn Locations (Blue Team):**

| Index | X | Y | Yaw (rad) |
|-------|---|---|-----------|
| 0 | -2176 | -3410 | π/2 |
| 1 | -1152 | -3100 | π/2 |
| 2 | 2176 | -3410 | π/2 |
| 3 | 1152 | -3100 | π/2 |

---

## 10. Ball Physics

### Ball State

```python
class BallState:
    pos: Vec           # Center position
    rot_mat: RotMat    # Rotation (for visual)
    vel: Vec           # Linear velocity
    ang_vel: Vec       # Angular velocity (rad/s)
```

### Ball Properties

```cpp
// Sphere shape
BALL_RADIUS_SOCCAR = 91.25 UU
BALL_RADIUS_HOOPS = 96.3831 UU
BALL_RADIUS_DROPSHOT = 100.2565 UU

// Physics properties
BALL_MASS = 30.0             // 1/6 of car mass
BALL_DRAG = 0.03             // Linear damping
BALL_FRICTION = 0.35         // World friction
BALL_RESTITUTION = 0.6       // Bounciness

// Limits
BALL_MAX_SPEED = 6000.0 UU/s
BALL_MAX_ANG_SPEED = 6.0 rad/s
```

**Ball No-Rotation Mode:**

RocketSim supports a "noRot" mode where the ball's rotation is locked. This is useful for testing and simplifying physics:

```python
# Only works for sphere shapes (not Snowday puck)
if no_rotation and ball.is_sphere():
    ball.rigid_body.no_rot = True
```

### Ball Physics Update

```python
def ball_finish_physics_tick(ball, mutator):
    # Apply cached velocity impulses
    if not ball.velocity_impulse_cache.is_zero():
        ball.velocity += ball.velocity_impulse_cache
        ball.velocity_impulse_cache = Vec(0, 0, 0)

    # Clamp velocities
    if ball.velocity.length() > mutator.ball_max_speed:
        ball.velocity = ball.velocity.normalized() * mutator.ball_max_speed

    if ball.angular_velocity.length() > BALL_MAX_ANG_SPEED:
        ball.angular_velocity = ball.angular_velocity.normalized() * BALL_MAX_ANG_SPEED
```

### Ball Drag

The ball has a linear drag coefficient applied by Bullet Physics:

```python
# In Bullet setup
rigid_body.linear_damping = BALL_DRAG  # 0.03
```

This means each frame:
```
velocity *= (1 - drag)^dt
```

### Ball Launch Mechanics

In Hoops and Dropshot modes, the ball is launched upward at the start of a round:

**Hoops Ball Launch:**

```cpp
BALL_HOOPS_LAUNCH_Z_VEL = 1000 UU/s  // Z impulse applied to ball
BALL_HOOPS_LAUNCH_DELAY = 0.265 s    // Delay before launch after kickoff
```

**Dropshot Ball Launch:**

```cpp
BALL_LAUNCH_Z_VEL = 985 UU/s    // Z impulse applied to ball
BALL_LAUNCH_DELAY = 0.26 s      // Delay before launch after kickoff
```

### Snowday Puck

The puck is a cylinder collision shape, not a sphere:

```cpp
PUCK_RADIUS = 114.25 UU
PUCK_HEIGHT = 62.5 UU
PUCK_MASS = 50.0
PUCK_GROUND_STICK_FORCE = 70
PUCK_FRICTION = 0.1
PUCK_RESTITUTION = 0.3
PUCK_CIRCLE_POINT_AMOUNT = 20  // Points per circle of convex hull
```

**Puck Collision Shape:**

The puck uses a convex hull approximation of a cylinder, made from two circles of 20 points each (top and bottom):

```python
def create_puck_shape():
    points = []
    for circle in [0, 1]:
        z = (PUCK_HEIGHT / 2) * (1 if circle else -1)
        for i in range(PUCK_CIRCLE_POINT_AMOUNT):
            angle = 2 * PI * i / PUCK_CIRCLE_POINT_AMOUNT
            x = PUCK_RADIUS * cos(angle)
            y = PUCK_RADIUS * sin(angle)
            points.append(Vec(x, y, z) * UU_TO_BT)

    return ConvexHullShape(points)
```

**Ground Stick Force:**

The puck has a downward force applied on world contact to keep it from bouncing excessively:

```python
def puck_on_world_collision(ball, contact_normal):
    # Only apply once per tick
    if not ball.ground_stick_applied:
        ball.rigid_body.apply_central_force(-contact_normal * PUCK_GROUND_STICK_FORCE)
        ball.ground_stick_applied = True
        # Flag is reset to False in _PreTickUpdate
```

**Ball Sleeping Prevention:**

In Snowday mode, the ball is initialized with a tiny Z velocity to prevent Bullet from putting it to sleep:

```python
def reset_game_state():
    if game_mode == SNOWDAY:
        ball_state.vel.z = FLT_EPSILON  # ~1e-38, prevents sleeping
```

---

## 11. Collision Detection and Resolution

### Bullet Physics Integration

RocketSim uses Bullet Physics 3.24 for collision detection and rigid body dynamics.

**World Configuration:**

```python
# Solver settings (matching older Bullet/Rocket League)
solver_info.split_impulse_penetration_threshold = 1e30  # Disabled
solver_info.erp2 = 0.8  # Error reduction parameter
```

### Collision Callback System

All collision events are routed through a global callback:

```python
def bullet_contact_added_callback(contact_point, obj_a, obj_b):
    # Classify collision types by user index
    type_a = obj_a.user_index
    type_b = obj_b.user_index

    # Ensure consistent ordering
    if type_a > type_b:
        swap(obj_a, obj_b)
        swap(type_a, type_b)

    if type_a == TYPE_CAR:
        if type_b == TYPE_BALL:
            on_car_ball_collision(...)
        elif type_b == TYPE_CAR:
            on_car_car_collision(...)
        else:
            on_car_world_collision(...)
    elif type_a == TYPE_BALL:
        if type_b == TYPE_WORLD:
            on_ball_world_collision(...)
```

### Separating Axis Theorem (SAT)

For box-box collision detection (car-car), SAT tests 15 potential separating axes:
- 3 face normals from box A
- 3 face normals from box B
- 9 cross products of edge directions

```python
def detect_box_collision_sat(box_a, box_b):
    axes = []

    # Face normals
    axes.extend(box_a.get_face_normals())  # 3 axes
    axes.extend(box_b.get_face_normals())  # 3 axes

    # Edge cross products
    for edge_a in box_a.get_edge_directions():
        for edge_b in box_b.get_edge_directions():
            axis = cross(edge_a, edge_b)
            if axis.length() > 0.001:
                axes.append(normalize(axis))

    min_overlap = infinity
    best_axis = None

    for axis in axes:
        proj_a = project_box_onto_axis(box_a, axis)
        proj_b = project_box_onto_axis(box_b, axis)

        overlap = min(proj_a.max, proj_b.max) - max(proj_a.min, proj_b.min)

        if overlap <= 0:
            return None  # Found separating axis

        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis

    # Ensure normal points from A to B
    if dot(best_axis, box_b.center - box_a.center) < 0:
        best_axis = -best_axis

    return CollisionInfo(penetration=min_overlap, normal=best_axis)
```

### Impulse Resolution

The general impulse formula for collision response:

```python
def calculate_impulse_magnitude(body_a, body_b, contact_point, normal, restitution):
    # Relative velocity at contact point
    r_a = contact_point - body_a.center_of_mass
    r_b = contact_point - body_b.center_of_mass

    vel_a = body_a.velocity + cross(body_a.angular_velocity, r_a)
    vel_b = body_b.velocity + cross(body_b.angular_velocity, r_b)
    rel_vel = vel_a - vel_b

    # Closing velocity along normal
    vel_along_normal = dot(rel_vel, normal)

    if vel_along_normal > 0:
        return 0  # Separating, no impulse needed

    # Effective mass (inverse)
    inv_mass_sum = body_a.inv_mass + body_b.inv_mass

    # Angular contribution
    r_cross_n_a = cross(r_a, normal)
    r_cross_n_b = cross(r_b, normal)

    angular_a = dot(r_cross_n_a, body_a.inv_inertia_world * r_cross_n_a)
    angular_b = dot(r_cross_n_b, body_b.inv_inertia_world * r_cross_n_b)

    effective_mass_inv = inv_mass_sum + angular_a + angular_b

    # Impulse magnitude
    j = -(1 + restitution) * vel_along_normal / effective_mass_inv

    return j
```

### Applying Impulses

```python
def apply_impulse(body, impulse, contact_point):
    # Linear impulse
    body.velocity += impulse * body.inv_mass

    # Angular impulse
    r = contact_point - body.center_of_mass
    torque = cross(r, impulse)
    body.angular_velocity += body.inv_inertia_world * torque
```

---

## 12. Car-Ball Collision

### Extra Hit Impulse

When a car hits the ball, RocketSim applies an additional impulse beyond standard physics:

**Extra Impulse Constants:**

```cpp
BALL_CAR_EXTRA_IMPULSE_Z_SCALE = 0.35
BALL_CAR_EXTRA_IMPULSE_Z_SCALE_HOOPS_GROUND = 0.5425  // 0.35 * 1.55
BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE = 0.65
BALL_CAR_EXTRA_IMPULSE_MAXDELTAVEL = 4600 UU/s
BALL_CAR_EXTRA_IMPULSE_Z_SCALE_HOOPS_NORMAL_Z_THRESH = 0.1
```

**Extra Impulse Factor Curve:**

| Relative Speed (UU/s) | Factor |
|----------------------|--------|
| 0 | 0.65 |
| 500 | 0.65 |
| 2300 | 0.55 |
| 4600 | 0.30 |

### Car-Ball Collision Handler

The car-ball collision applies Bullet's physics response (with custom friction/restitution) plus an extra "game feel" impulse:

```python
def on_car_ball_collision(car, ball, contact_point, contact_normal, is_ball_body_a):
    # Override material properties - high friction, zero bounce
    contact_point.friction = CARBALL_COLLISION_FRICTION  # 2.0
    contact_point.restitution = CARBALL_COLLISION_RESTITUTION  # 0.0

    # Get the relative position ON THE BALL (for hit info tracking)
    if is_ball_body_a:
        rel_pos_on_ball = contact_point.local_point_a * BT_TO_UU
    else:
        rel_pos_on_ball = contact_point.local_point_b * BT_TO_UU

    # Call ball's hit handler
    ball.on_hit(car, rel_pos_on_ball, game_mode, mutator, tick_count)
```

### Extra Hit Impulse (Ball._OnHit)

This impulse is the "secret sauce" that makes car-ball hits feel right. It's applied in addition to Bullet's physics response:

```python
def ball_on_hit(self, car, rel_pos_on_ball, game_mode, mutator, tick_count):
    car_state = car.get_state()
    ball_state = self.get_state()

    # Record hit info on the car
    car.ball_hit_info.is_valid = True
    car.ball_hit_info.relative_pos_on_ball = rel_pos_on_ball
    car.ball_hit_info.ball_pos = ball_state.pos
    car.ball_hit_info.tick_count_when_hit = tick_count
    car.ball_hit_info.extra_hit_vel = Vec(0, 0, 0)

    # IMPORTANT: Cooldown check - must wait at least 1 tick between extra impulses
    # This prevents multiple impulses from a single sustained contact
    last_impulse_tick = car.ball_hit_info.tick_count_when_extra_impulse_applied
    if tick_count <= last_impulse_tick + 1 and last_impulse_tick <= tick_count:
        return

    car.ball_hit_info.tick_count_when_extra_impulse_applied = tick_count

    # Calculate relative position and velocity
    car_forward = car.get_forward_dir()
    rel_pos = ball_state.pos - car_state.pos   # Ball relative to car
    rel_vel = ball_state.vel - car_state.vel   # Ball velocity relative to car

    # Clamp relative speed to prevent extreme impulses
    rel_speed = min(rel_vel.length(), BALL_CAR_EXTRA_IMPULSE_MAXDELTAVEL)  # 4600 UU/s

    if rel_speed <= 0:
        return

    # Determine Z scale factor
    # Hoops mode uses a higher Z scale when car is grounded and upright
    # This helps pop the ball up into the hoop
    extra_z_scale = (game_mode == HOOPS and
                     car_state.is_on_ground and
                     car_state.rot_mat.up.z > BALL_CAR_EXTRA_IMPULSE_Z_SCALE_HOOPS_NORMAL_Z_THRESH)

    if extra_z_scale:
        z_scale = BALL_CAR_EXTRA_IMPULSE_Z_SCALE_HOOPS_GROUND  # 0.5425
    else:
        z_scale = BALL_CAR_EXTRA_IMPULSE_Z_SCALE  # 0.35

    # Calculate hit direction from car to ball, with Z scaled down
    # This makes hits more horizontal than the actual geometry
    hit_dir = normalize(rel_pos * Vec(1, 1, z_scale))

    # Reduce forward component to prevent "poke" shots being too strong
    # Subtracts (1 - 0.65) = 35% of the forward-aligned component
    forward_component = hit_dir.dot(car_forward)
    forward_adjustment = car_forward * forward_component * (1 - BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE)
    hit_dir = normalize(hit_dir - forward_adjustment)

    # Calculate impulse magnitude from curve (higher speed = lower factor)
    factor = BALL_CAR_EXTRA_IMPULSE_FACTOR_CURVE.get_output(rel_speed)

    # Final velocity to add
    added_vel = hit_dir * rel_speed * factor * mutator.ball_hit_extra_force_scale

    car.ball_hit_info.extra_hit_vel = added_vel

    # Cache impulse - applied at end of physics tick (not immediately)
    self.velocity_impulse_cache += added_vel * UU_TO_BT
```

**Why Cache the Impulse?**

The impulse is cached rather than applied immediately because:
1. Multiple collision callbacks may fire in one tick
2. Applying during collision callback can cause instability
3. The game applies all velocity changes at the end of the tick

---

## 13. Car-Car Collision (Bumps and Demos)

### Bump Constants

```cpp
BUMP_COOLDOWN_TIME = 0.25 s
BUMP_MIN_FORWARD_DIST = 64.5 UU    // Contact must be this far forward
```

### Bump Force Curves

**Ground Bump Velocity:**

| Collision Speed (UU/s) | Bump Velocity (UU/s) |
|------------------------|----------------------|
| 0 | 0.833 |
| 1400 | 1100 |
| 2200 | 1530 |

**Air Bump Velocity:**

| Collision Speed (UU/s) | Bump Velocity (UU/s) |
|------------------------|----------------------|
| 0 | 0.833 |
| 1400 | 1390 |
| 2200 | 1945 |

**Upward Velocity:**

| Collision Speed (UU/s) | Upward Velocity (UU/s) |
|------------------------|------------------------|
| 0 | 0.333 |
| 1400 | 278 |
| 2200 | 417 |

### Car-Car Collision Handler

```python
def on_car_car_collision(car1, car2, contact_point):
    # Override material properties
    contact_point.friction = CARCAR_COLLISION_FRICTION  # 0.09
    contact_point.restitution = CARCAR_COLLISION_RESTITUTION  # 0.1

    # Test collision both ways (car1 bumping car2, car2 bumping car1)
    for attacker, victim in [(car1, car2), (car2, car1)]:
        process_bump_attempt(attacker, victim, contact_point)

def process_bump_attempt(attacker, victim, contact_point):
    attacker_state = attacker.get_state()
    victim_state = victim.get_state()

    # Skip if either car is demoed
    if attacker_state.is_demoed or victim_state.is_demoed:
        return

    # Check cooldown
    if (attacker_state.car_contact.other_car_id == victim.id and
        attacker_state.car_contact.cooldown_timer > 0):
        return

    # Check if attacker is moving toward victim
    delta_pos = victim_state.pos - attacker_state.pos
    if dot(attacker_state.vel, delta_pos) <= 0:
        return

    vel_dir = normalize(attacker_state.vel)
    dir_to_victim = normalize(delta_pos)

    speed_towards_victim = dot(attacker_state.vel, dir_to_victim)
    victim_away_speed = dot(victim_state.vel, vel_dir)

    # Only bump if approaching faster than victim is retreating
    if speed_towards_victim <= victim_away_speed:
        return

    # Check if hit with front bumper
    local_contact = contact_point.local_pos_a if attacker == car1 else contact_point.local_pos_b
    hit_with_bumper = local_contact.x > BUMP_MIN_FORWARD_DIST

    if not hit_with_bumper:
        return

    # Determine demo vs bump
    is_demo = False
    if mutator.demo_mode == DEMO_ON_CONTACT:
        is_demo = True
    elif mutator.demo_mode == DEMO_NORMAL:
        is_demo = attacker_state.is_supersonic

    if is_demo and not mutator.enable_team_demos:
        is_demo = (attacker.team != victim.team)

    if is_demo:
        # Demolish victim
        victim.demolish(mutator.respawn_delay)
    else:
        # Apply bump impulse
        victim_on_ground = victim_state.is_on_ground

        if victim_on_ground:
            base_scale = BUMP_VEL_GROUND_CURVE.get_output(speed_towards_victim)
        else:
            base_scale = BUMP_VEL_AIR_CURVE.get_output(speed_towards_victim)

        upward_scale = BUMP_UPWARD_VEL_CURVE.get_output(speed_towards_victim)

        # Calculate hit direction
        if victim_on_ground:
            hit_up_dir = victim.up_dir
        else:
            hit_up_dir = Vec(0, 0, 1)

        bump_impulse = (vel_dir * base_scale +
                        hit_up_dir * upward_scale) * mutator.bump_force_scale

        victim.velocity_impulse_cache += bump_impulse

    # Set cooldown
    attacker.car_contact.other_car_id = victim.id
    attacker.car_contact.cooldown_timer = mutator.bump_cooldown_time
```

### Demolition

```python
def demolish(car, respawn_delay):
    car.is_demoed = True
    car.demo_respawn_timer = respawn_delay

    # Disable physics simulation
    car.rigid_body.activation_state = DISABLE_SIMULATION
    car.rigid_body.collision_flags |= NO_CONTACT_RESPONSE

def check_respawn(car, dt, game_mode, mutator):
    if car.is_demoed:
        car.demo_respawn_timer = max(0, car.demo_respawn_timer - dt)

        if car.demo_respawn_timer == 0:
            respawn(car, game_mode, mutator)
```

---

## 14. Arena and World Collision

### Car-World Collision

```python
def on_car_world_collision(car, world_obj, contact_point):
    # Record world contact for auto-flip/auto-roll
    car.world_contact.has_contact = True
    car.world_contact.normal = contact_point.normal_world_on_b

    # Override material properties
    contact_point.friction = mutator.car_world_friction  # 0.3
    contact_point.restitution = mutator.car_world_restitution  # 0.3
```

### Ball-World Collision

```python
def on_ball_world_collision(ball, contact_point, contact_normal):
    # Heatseeker: Check for backboard bounce
    if game_mode == HEATSEEKER and ball.hs_info.target_dir != 0:
        check_heatseeker_wall_bounce(ball, contact_normal)

    # Snowday: Apply ground stick force
    if game_mode == SNOWDAY:
        if not ball.ground_stick_applied:
            ball.rigid_body.apply_central_force(-contact_normal * PUCK_GROUND_STICK_FORCE)
            ball.ground_stick_applied = True

    # Mark contact as "special" (except Snowday)
    # This affects Bullet's internal collision handling
    if game_mode != SNOWDAY:
        contact_point.is_special = True
```

### Internal Edge Contact Adjustment

After collision detection, Bullet's `btAdjustInternalEdgeContacts` is called to fix edge artifacts:

```python
def bullet_contact_added_callback(contact_point, obj_a, obj_b, ...):
    # ... collision handling ...

    # Fix triangle edge artifacts in arena collision mesh
    # This prevents objects from catching on edges between triangles
    btAdjustInternalEdgeContacts(contact_point, obj_a, obj_b, part_id, index)
    return True
```

This is essential for smooth collision with the triangle mesh arena walls.

### Collision Groups and Masks

Bullet uses collision groups (what an object is) and masks (what it collides with) to filter collisions:

```python
# Collision mask flags
COLLISION_MASK_HOOPS_NET = 1 << 0      # Hoops net rim
COLLISION_MASK_DROPSHOT_FLOOR = 1 << 1  # Dropshot hexagon tiles
COLLISION_MASK_DROPSHOT_TILE = 1 << 2   # Individual damaged tiles

# Car setup - collides with dropshot floor
car.add_to_world(
    group = DEFAULT_FILTER | DROPSHOT_FLOOR,
    mask = ALL_FILTER
)

# Ball setup - collides with hoops net and dropshot tiles
ball.add_to_world(
    group = DEFAULT_FILTER | HOOPS_NET | DROPSHOT_TILE,
    mask = ALL_FILTER
)
```

This allows:
- Cars to drive on the dropshot floor even before tiles are broken
- Ball to interact with hoops net rims
- Ball to trigger dropshot tile damage

### Arena Collision Mesh

The arena collision is built from:
1. **BVH Triangle Mesh Shapes**: Loaded from collision mesh files (.bin format)
2. **Static Planes**: Floor, ceiling, and side walls

```python
def setup_arena_collision():
    # Load mesh shapes
    for mesh in load_collision_meshes(game_mode):
        shape = BvhTriangleMeshShape(mesh)
        world.add_rigid_body(create_static_body(shape))

    # Add boundary planes
    add_static_plane(Vec(0, 0, 0), Vec(0, 0, 1))  # Floor
    add_static_plane(Vec(0, 0, ARENA_HEIGHT), Vec(0, 0, -1))  # Ceiling
    add_static_plane(Vec(-ARENA_EXTENT_X, 0, 0), Vec(1, 0, 0))  # Left wall
    add_static_plane(Vec(ARENA_EXTENT_X, 0, 0), Vec(-1, 0, 0))  # Right wall
```

---

## 15. Special Game Modes

### Heatseeker

The ball seeks toward the opponent's goal:

**Constants:**

```cpp
HEATSEEKER_INITIAL_TARGET_SPEED = 2900 UU/s
HEATSEEKER_TARGET_SPEED_INCREMENT = 85 UU/s
HEATSEEKER_MIN_SPEEDUP_INTERVAL = 1.0 s
HEATSEEKER_TARGET_Y = 5120 UU
HEATSEEKER_TARGET_Z = 320 UU
HEATSEEKER_HORIZONTAL_BLEND = 1.45
HEATSEEKER_VERTICAL_BLEND = 0.78
HEATSEEKER_SPEED_BLEND = 0.3
HEATSEEKER_MAX_TURN_PITCH = 7000 * PI / 32768 ≈ 0.671 rad
HEATSEEKER_MAX_SPEED = 4600 UU/s

// Wall bounce parameters
WALL_BOUNCE_CHANGE_Y_THRESH = 300 UU    // Backwall distance threshold to change targets
WALL_BOUNCE_CHANGE_Y_NORMAL = 0.5       // Minimum Y normal to trigger bounce-back
WALL_BOUNCE_FORCE_SCALE = 1/3           // Scale of extra bounce impulse
WALL_BOUNCE_UP_FRAC = 0.3               // Fraction of impulse that goes straight up

// Ball starting state (for Blue team - flip Y for Orange)
BALL_START_POS = Vec(-1000, -2220, 92.75)
BALL_START_VEL = Vec(0, -65, 650)
```

**Wall Bounce Mechanics:**

When the heatseeker ball hits a wall near the backboard:

```python
def check_heatseeker_wall_bounce(ball, contact_normal):
    # Check if this is a backboard bounce
    abs_ball_y = abs(ball.pos.y)
    distance_from_backwall = ARENA_EXTENT_Y - abs_ball_y

    if distance_from_backwall < WALL_BOUNCE_CHANGE_Y_THRESH:
        if abs(contact_normal.y) >= WALL_BOUNCE_CHANGE_Y_NORMAL:
            # Switch target to opposite goal
            ball.hs_info.target_dir *= -1

            # Apply extra bounce impulse
            impulse_dir = normalize(Vec(0, ball.hs_info.target_dir, WALL_BOUNCE_UP_FRAC))
            impulse_magnitude = ball.velocity.length() * WALL_BOUNCE_FORCE_SCALE

            ball.velocity += impulse_dir * impulse_magnitude
```

```python
def heatseeker_pre_tick(ball, dt):
    if ball.hs_info.target_dir == 0:
        return

    state = ball.get_state()
    vel_angle = Angle.from_vec(state.vel)

    # Target position in goal
    goal_target = Vec(0, HEATSEEKER_TARGET_Y * ball.hs_info.target_dir, HEATSEEKER_TARGET_Z)
    angle_to_goal = Angle.from_vec(goal_target - state.pos)

    # Find angle difference
    delta_angle = angle_to_goal - vel_angle

    # Speed ratio affects turn rate (faster = more turn per second)
    cur_speed = state.vel.length()
    speed_ratio = cur_speed / HEATSEEKER_MAX_SPEED

    # Interpolate angles
    base_interp_factor = speed_ratio * dt
    new_angle = vel_angle
    new_angle.yaw += delta_angle.yaw * base_interp_factor * HEATSEEKER_HORIZONTAL_BLEND
    new_angle.pitch += delta_angle.pitch * base_interp_factor * HEATSEEKER_VERTICAL_BLEND
    new_angle.normalize_fix()

    # Limit pitch
    new_angle.pitch = clamp(new_angle.pitch, -HEATSEEKER_MAX_TURN_PITCH, HEATSEEKER_MAX_TURN_PITCH)

    # IMPORTANT: Apply UE3 rotator rounding
    # Unreal Engine 3 uses 16-bit angles (0-65535), causing quantization
    # This rounding is "surprisingly important for accuracy"
    new_angle = round_angle_ue3(new_angle)

    # Accelerate toward target speed
    new_speed = cur_speed + (ball.hs_info.target_speed - cur_speed) * HEATSEEKER_SPEED_BLEND

    # Apply new velocity
    new_dir = new_angle.get_forward_vec()
    ball.velocity = new_dir * new_speed

def round_angle_ue3(angle):
    """Round angle to UE3's 16-bit precision"""
    # UE3 uses 65536 units per full rotation
    # This rounds each angle component to nearest representable value
    def round_component(rad):
        # Convert to UE3 units, round, convert back
        ue3_units = rad * 65536 / (2 * PI)
        rounded = round(ue3_units)
        return rounded * (2 * PI) / 65536

    return Angle(
        round_component(angle.yaw),
        round_component(angle.pitch),
        round_component(angle.roll)
    )
```

### Dropshot

Ball charges up with hits and damages floor tiles:

**Ball Mechanics:**

```cpp
DROPSHOT_MIN_DOWNWARD_SPEED_TO_DAMAGE = 250 UU/s   // Min downward speed to damage tiles
DROPSHOT_MIN_CHARGE_HIT_SPEED = 500 UU/s           // Min delta speed to charge ball
DROPSHOT_MIN_ABSORBED_FORCE_FOR_CHARGE = 2500      // Force needed to "break open" ball
DROPSHOT_MIN_ABSORBED_FORCE_FOR_SUPERCHARGE = 11000 // Force needed to super-charge
DROPSHOT_MIN_DAMAGE_INTERVAL = 0.1 s               // Min time between damaging tiles
```

**Charge Levels:**

| Level | Damage Radius | Tiles Affected |
|-------|---------------|----------------|
| 1 | 1 tile | 1 tile |
| 2 | 2 tiles | 7 tiles |
| 3 | 3 tiles | 19 tiles |

**Tile Layout:**

```cpp
NUM_TILES_PER_TEAM = 70
NUM_TILE_ROWS = 7           // Rows 1-7
TILES_IN_FIRST_ROW = 13     // Row closest to center
TILES_IN_LAST_ROW = 7       // Row farthest from center

// Hexagon tile dimensions (in Bullet units, multiply by BT_TO_UU for UU)
TILE_HEXAGON_AABB_MAX = Vec(7.6643, 8.85, 0)  // Half-extents of tile bounding box

// Full tile width from point to point
TILE_WIDTH_X = 7.6643 * 2 * BT_TO_UU = 766.43 UU

// Y-offset between rows (interlocking hexagons)
ROW_OFFSET_Y = (8.85 + 4.425) * BT_TO_UU = 663.75 UU

// Base Y-offset for all tiles
TILE_OFFSET_Y = 2.54736 * BT_TO_UU = 127.37 UU
```

**Hexagon Tile Vertices (Bullet Units):**

The tile shape is a regular hexagon with these vertices (centered at origin):

```
Vertex 0: ( 0.0000, -8.85,  0)   // Bottom point
Vertex 1: ( 7.6643, -4.425, 0)   // Bottom-right
Vertex 2: ( 7.6643,  4.425, 0)   // Top-right
Vertex 3: ( 0.0000,  8.85,  0)   // Top point
Vertex 4: (-7.6643,  4.425, 0)   // Top-left
Vertex 5: (-7.6643, -4.425, 0)   // Bottom-left
```

**Note:** World-space vertices are clamped so they don't cross the Y=0 center line.

**Dropshot Goal Detection:**

A goal in Dropshot mode is scored when the ball falls through a broken tile:

```python
def is_ball_scored_dropshot(ball, mutator):
    # Ball must be below the floor by a significant margin
    ball_z = ball.rigid_body.world_transform.origin.z * BT_TO_UU
    return ball_z < -(mutator.ball_radius * 1.75)
```

---

## 16. Simulation Loop Architecture

### Tick Rate

```cpp
DEFAULT_TICK_RATE = 120 Hz  // Standard physics tick rate
MIN_TICK_RATE = 15 Hz       // Minimum supported
MAX_TICK_RATE = 120 Hz      // Maximum supported

TICK_TIME = 1 / TICK_RATE   // Seconds per tick (default: ~0.00833s)
```

The simulation can run at different tick rates (15-120 Hz), but 120 Hz matches the game's physics rate.

### Tick Order

Each simulation tick follows this order:

```python
def step(arena, num_ticks):
    for i in range(num_ticks):
        # 1. Ball sleep check
        if ball.velocity.is_zero() and ball.angular_velocity.is_zero():
            ball.set_sleeping()
        else:
            ball.set_active()

        # 2. Pre-tick updates
        for car in cars:
            car.pre_tick_update(game_mode, tick_time, mutator)

        for boost_pad in boost_pads:
            boost_pad.pre_tick_update(tick_time)

        ball.pre_tick_update(game_mode, tick_time)

        # 3. Bullet physics step
        bullet_world.step_simulation(tick_time)

        # 4. Post-tick updates
        for car in cars:
            car.post_tick_update(game_mode, tick_time, mutator)
            car.finish_physics_tick(mutator)

            # Boost pad collision
            for boost_pad in boost_pads:
                boost_pad.check_collide(car)

        for boost_pad in boost_pads:
            boost_pad.post_tick_update(tick_time, mutator)

        ball.finish_physics_tick(mutator)

        # 5. Goal scoring check
        if is_ball_scored():
            trigger_goal_callback()

        tick_count += 1
```

### Car Pre-Tick Update

```python
def car_pre_tick_update(car, game_mode, dt, mutator):
    # Input validation
    car.controls.clamp_fix()

    # Handle demo state
    if car.is_demoed:
        car.demo_respawn_timer -= dt
        if car.demo_respawn_timer <= 0:
            car.respawn(game_mode, mutator)
        return

    # Update vehicle first pass (wheel raycasts, friction calculation)
    car.vehicle.update_vehicle_first(dt)

    # Detect jump press
    jump_pressed = car.controls.jump and not car.last_controls.jump

    # Count wheel contacts
    num_wheels_in_contact = sum(1 for w in car.wheels if w.is_in_contact)
    car.is_on_ground = (num_wheels_in_contact >= 3)

    forward_speed = car.vehicle.get_forward_speed()

    # Update subsystems
    car.update_wheels(dt, mutator, num_wheels_in_contact, forward_speed)

    if num_wheels_in_contact < 3:
        car.update_air_torque(dt, mutator, num_wheels_in_contact == 0)
    else:
        car.is_flipping = False

    car.update_jump(dt, mutator, jump_pressed)
    car.update_auto_flip(dt, mutator, jump_pressed)
    car.update_double_jump_or_flip(dt, mutator, jump_pressed, forward_speed)

    if car.controls.throttle and (0 < num_wheels_in_contact < 4 or car.world_contact.has_contact):
        car.update_auto_roll(dt, mutator, num_wheels_in_contact)

    car.world_contact.has_contact = False

    # Update vehicle second pass (suspension, friction application)
    car.vehicle.update_vehicle_second(dt)

    car.update_boost(dt, mutator, forward_speed)
```

### Car Post-Tick Update

```python
def car_post_tick_update(car, game_mode, dt, mutator):
    if car.is_demoed:
        return

    # Update rotation matrix from rigid body
    car.rotation_matrix = car.rigid_body.rotation_matrix

    # Update supersonic state
    speed_sq = car.velocity.length_sq()

    if car.is_supersonic and car.supersonic_time < SUPERSONIC_MAINTAIN_MAX_TIME:
        car.is_supersonic = (speed_sq >= SUPERSONIC_MAINTAIN_MIN_SPEED ** 2)
    else:
        car.is_supersonic = (speed_sq >= SUPERSONIC_START_SPEED ** 2)

    if car.is_supersonic:
        car.supersonic_time += dt
    else:
        car.supersonic_time = 0

    # Update cooldowns
    car.car_contact.cooldown_timer = max(0, car.car_contact.cooldown_timer - dt)

    # Store controls for next frame
    car.last_controls = car.controls
```

### Car Finish Physics Tick

```python
def car_finish_physics_tick(car, mutator):
    if car.is_demoed:
        return

    # Apply cached velocity impulses
    if not car.velocity_impulse_cache.is_zero():
        car.velocity += car.velocity_impulse_cache
        car.velocity_impulse_cache = Vec(0, 0, 0)

    # Clamp velocities
    if car.velocity.length() > CAR_MAX_SPEED:
        car.velocity = car.velocity.normalized() * CAR_MAX_SPEED

    if car.angular_velocity.length() > CAR_MAX_ANG_SPEED:
        car.angular_velocity = car.angular_velocity.normalized() * CAR_MAX_ANG_SPEED
```

---

## 17. Linear Piecewise Curves

Many physics values use piecewise linear interpolation:

```python
class LinearPieceCurve:
    def __init__(self, points):
        # points is a list of (input, output) tuples, sorted by input
        self.points = sorted(points, key=lambda p: p[0])

    def get_output(self, input_val, default=1.0):
        if len(self.points) == 0:
            return default

        # Below minimum
        if input_val <= self.points[0][0]:
            return self.points[0][1]

        # Above maximum
        if input_val >= self.points[-1][0]:
            return self.points[-1][1]

        # Find segment and interpolate
        for i in range(len(self.points) - 1):
            x0, y0 = self.points[i]
            x1, y1 = self.points[i + 1]

            if x0 <= input_val <= x1:
                t = (input_val - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)

        return default
```

### All Curves Summary

| Curve | Input | Output | Points |
|-------|-------|--------|--------|
| STEER_ANGLE_FROM_SPEED | Speed (UU/s) | Angle (rad) | (0, 0.534), (500, 0.319), (1000, 0.182), (1500, 0.106), (1750, 0.085), (3000, 0.035) |
| STEER_ANGLE_THREEWHEEL | Speed (UU/s) | Angle (rad) | (0, 0.342), (2300, 0.035) |
| POWERSLIDE_STEER_ANGLE | Speed (UU/s) | Angle (rad) | (0, 0.392), (2500, 0.126) |
| DRIVE_SPEED_TORQUE | Speed (UU/s) | Factor | (0, 1.0), (1400, 0.1), (1410, 0.0) |
| LAT_FRICTION | Slip ratio | Friction | (0, 1.0), (1, 0.2) |
| LAT_FRICTION_THREEWHEEL | Slip ratio | Friction | (0, 0.30), (1, 0.25) |
| LONG_FRICTION | Slip ratio | Friction | Empty (returns 1.0) |
| NON_STICKY_FRICTION | Normal Z | Factor | (0, 0.1), (0.7075, 0.5), (1, 1.0) |
| HANDBRAKE_LAT_FRICTION | Any | Factor | (0, 0.1) - Constant |
| HANDBRAKE_LONG_FRICTION | Slip ratio | Factor | (0, 0.5), (1, 0.9) |
| BALL_CAR_EXTRA_IMPULSE | Speed (UU/s) | Factor | (0, 0.65), (500, 0.65), (2300, 0.55), (4600, 0.30) |
| BUMP_VEL_GROUND | Speed (UU/s) | Velocity (UU/s) | (0, 0.833), (1400, 1100), (2200, 1530) |
| BUMP_VEL_AIR | Speed (UU/s) | Velocity (UU/s) | (0, 0.833), (1400, 1390), (2200, 1945) |
| BUMP_UPWARD_VEL | Speed (UU/s) | Velocity (UU/s) | (0, 0.333), (1400, 278), (2200, 417) |

---

## 18. Mathematical Foundations

### Inertia Tensor

For a box-shaped car, the diagonal inertia tensor is:

```python
def calculate_box_inertia(mass, half_extents):
    # I = m/12 * diag(h^2+d^2, w^2+d^2, w^2+h^2)
    w, h, d = half_extents.x * 2, half_extents.y * 2, half_extents.z * 2

    Ixx = mass * (h*h + d*d) / 12
    Iyy = mass * (w*w + d*d) / 12
    Izz = mass * (w*w + h*h) / 12

    return Matrix3x3.diagonal(Ixx, Iyy, Izz)
```

The **world-space inverse inertia tensor** is updated when rotation changes:

```python
def update_inverse_inertia_world(body):
    R = body.rotation_matrix
    I_local_inv = body.inverse_inertia_local

    # I_world^-1 = R * I_local^-1 * R^T
    body.inverse_inertia_world = R * I_local_inv * R.transpose()
```

### Quaternion Integration

For angular velocity integration:

```python
def integrate_orientation(q, angular_velocity, dt):
    # qdot = 0.5 * omega * q
    omega_quat = Quaternion(0, angular_velocity.x, angular_velocity.y, angular_velocity.z)
    q_dot = omega_quat * q * 0.5

    # Euler integration
    q_new = q + q_dot * dt

    # Renormalize to prevent drift
    return q_new.normalized()
```

### Bilateral Constraint (Wheel Friction)

The bilateral constraint solver finds the impulse needed to reduce relative velocity along a direction. This is used for wheel lateral friction:

```python
CONTACT_DAMPING = 0.2  # Magic constant from Bullet

def resolve_single_bilateral(body_a, body_b, pos_a, pos_b, direction):
    # Get relative positions
    r_a = pos_a - body_a.center_of_mass
    r_b = pos_b - body_b.center_of_mass

    # Calculate velocities at contact points
    vel_a = body_a.velocity + cross(body_a.angular_velocity, r_a)
    vel_b = body_b.velocity + cross(body_b.angular_velocity, r_b)

    rel_vel = dot(vel_a - vel_b, direction)

    # Calculate Jacobian diagonal (effective mass)
    # This accounts for both linear and angular contributions
    jac = JacobianEntry(
        body_a.rotation.transpose(),
        body_b.rotation.transpose(),
        r_a, r_b, direction,
        body_a.inv_inertia_diag_local, body_a.inv_mass,
        body_b.inv_inertia_diag_local, body_b.inv_mass
    )

    jac_diag_inv = 1.0 / jac.get_diagonal()

    # Return damped impulse (not full velocity cancellation)
    return -CONTACT_DAMPING * rel_vel * jac_diag_inv
```

**Note:** The CONTACT_DAMPING factor of 0.2 means only 20% of the lateral slip is corrected per step, creating gradual grip rather than instant stick.

### Heatseeker Ball Start State

For Heatseeker mode, the ball starts in a specific position with initial velocity (for Blue team - flip Y and velocity.y for Orange):

```cpp
BALL_START_POS = Vec(-1000, -2220, 92.75)  // Position
BALL_START_VEL = Vec(0, -65, 650)          // Initial velocity (upward arc toward goal)
```

---

## Appendix A: Quick Reference Tables

### Speed Thresholds

| Threshold | Value (UU/s) | Purpose |
|-----------|--------------|---------|
| Stopping | 25 | Full brake when coasting below this |
| Throttle cutoff | 1410 | No more engine force above this |
| Supersonic start | 2200 | Become supersonic |
| Supersonic maintain | 2100 | Stay supersonic |
| Max car speed | 2300 | Hard velocity cap |
| Max ball speed | 6000 | Hard velocity cap |

### Timing Values

| Timer | Duration | Purpose |
|-------|----------|---------|
| Jump minimum | 0.025 s | Minimum jump hold time |
| Jump maximum | 0.2 s | Maximum jump hold time |
| Flip window | 1.25 s | Time to flip after jump |
| Flip torque | 0.65 s | Duration of flip rotation |
| Boost minimum | 0.1 s | Minimum boost duration |
| Bump cooldown | 0.25 s | Time between bumps |
| Supersonic grace | 1.0 s | Time to maintain below threshold |

### Mass Values

| Object | Mass |
|--------|------|
| Car | 180 |
| Ball (Soccar) | 30 |
| Puck (Snowday) | 50 |

---

## Appendix B: Implementation Checklist

Use this checklist when implementing a recreation:

**Core Setup:**
- [ ] Unit conversion (UU to physics engine units: 1 UU = 0.02 m)
- [ ] Coordinate system (X forward, Y right, Z up)
- [ ] Gravity (-650 UU/s^2)
- [ ] Tick rate (120 Hz default)

**Car Setup:**
- [ ] Car hitbox setup with offset from center of mass
- [ ] 7 hitbox configurations (Octane, Dominus, Plank, Breakout, Hybrid, Merc, Psyclops)
- [ ] Wheel configuration (positions, radii, suspension rest lengths)
- [ ] Three-wheel car support (Psyclops)

**Suspension and Wheels:**
- [ ] Wheel raycast suspension
- [ ] Suspension force calculation with damping
- [ ] Extra pushback for collision penetration
- [ ] Rolling friction (magic constant: 113.73963)

**Driving:**
- [ ] Steering angle from speed curve (plus three-wheel variant)
- [ ] Powerslide extended steering curve
- [ ] Throttle/brake logic with coasting
- [ ] Drive speed torque factor curve
- [ ] Wheel friction with handbrake modifiers
- [ ] Lateral/longitudinal friction curves (plus three-wheel variant)
- [ ] Non-sticky friction curve
- [ ] Sticky force application (0 for three-wheel, 0.5 base for others)

**Jumping and Flipping:**
- [ ] Jump immediate impulse + sustained force
- [ ] Jump minimum/maximum time
- [ ] Double jump impulse
- [ ] Flip detection (deadzone check)
- [ ] Flip direction calculation
- [ ] Flip velocity impulse with speed scaling
- [ ] Flip torque application
- [ ] Flip Z-velocity damping
- [ ] Flip cancel with pitch input
- [ ] Flip reset when 3+ wheels contact any surface

**Air Control:**
- [ ] Air control torque with damping
- [ ] Air throttle force
- [ ] Auto-flip on turtle (upside down recovery)
- [ ] Auto-roll with partial contact

**Boost:**
- [ ] Boost consumption and force (ground vs air accel)
- [ ] Boost minimum duration
- [ ] Boost pad pickup (cylinder detection, box for locked state)
- [ ] Boost pad cooldown

**Speed and State:**
- [ ] Speed limiting (car: 2300, ball: 6000 UU/s)
- [ ] Angular velocity limiting (car: 5.5, ball: 6.0 rad/s)
- [ ] Supersonic state tracking with grace period
- [ ] Ball drag (linear damping: 0.03)

**Collision:**
- [ ] Car-ball extra impulse with curve
- [ ] Car-car bump detection (forward bumper check)
- [ ] Car-car bump impulse curves (ground/air/upward)
- [ ] Demolition logic with respawn timer
- [ ] Collision material overrides (friction, restitution)
- [ ] Velocity impulse caching

**Game Logic:**
- [ ] Goal scoring detection (Soccar Y threshold, Hoops XY+Z check)
- [ ] Car spawn and respawn positions
- [ ] Tick order (pre-tick, physics, post-tick, finish)

**Special Modes (if implementing):**
- [ ] Heatseeker ball seeking and wall bounces
- [ ] Dropshot tile damage and ball charging
- [ ] Snowday puck physics (cylinder shape, ground stick force)
- [ ] Hoops/Dropshot ball launch mechanics

---

*This document was generated from analysis of the RocketSim codebase (https://github.com/ZealanL/RocketSim), which is an open-source recreation of Rocket League's physics engine for machine learning and analysis purposes.*
