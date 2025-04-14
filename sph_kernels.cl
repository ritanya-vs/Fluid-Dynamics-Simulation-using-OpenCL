// Constants
#define SMOOTHING_LENGTH 3.5f
#define PARTICLE_MASS 0.25f
#define ISOTROPIC_EXPONENT 320.0f  // Slightly increased for wall hits
#define BASE_DENSITY 1.0f
#define DYNAMIC_VISCOSITY 0.05f
#define DAMPING_COEFFICIENT -0.15f  // Softer for continuous splashing
#define MIN_DENSITY 0.1f
#define NORMALIZATION_DENSITY (315.0f * PARTICLE_MASS / (64.0f * 3.141592653589793f * pow(SMOOTHING_LENGTH, 9)))
#define NORMALIZATION_PRESSURE_FORCE (-45.0f * PARTICLE_MASS / (3.141592653589793f * pow(SMOOTHING_LENGTH, 6)))
#define NORMALIZATION_VISCOUS_FORCE (45.0f * DYNAMIC_VISCOSITY * PARTICLE_MASS / (3.141592653589793f * pow(SMOOTHING_LENGTH, 6)))
#define DOMAIN_X_MIN SMOOTHING_LENGTH
#define DOMAIN_X_MAX (40.0f - SMOOTHING_LENGTH)
#define DOMAIN_Y_MIN SMOOTHING_LENGTH
#define DOMAIN_Y_MAX (80.0f - SMOOTHING_LENGTH)
#define GRID_CELL_SIZE SMOOTHING_LENGTH
#define GRID_WIDTH ((int)((40.0f) / GRID_CELL_SIZE) + 1)
#define GRID_HEIGHT ((int)((80.0f) / GRID_CELL_SIZE) + 1)

// Helper function to compute squared distance
float dist_squared(float2 a, float2 b) {
    float2 diff = a - b;
    return dot(diff, diff);
}

// Simple pseudo-random number generator
float rand(float seed) {
    float val = sin(seed * 127.1f) * 43758.5453f;
    return val - floor(val);
}

// Kernel 1: Assign particles to grid cells
kernel void assign_to_grid(
    global float2* positions,
    global int* particle_cell_ids,
    global int* cell_particle_counts,
    const int n_particles
) {
    int gid = get_global_id(0);
    if (gid >= n_particles) return;

    float2 pos = positions[gid];
    int cell_x = (int)(pos.x / GRID_CELL_SIZE);
    int cell_y = (int)(pos.y / GRID_CELL_SIZE);
    cell_x = clamp(cell_x, 0, GRID_WIDTH - 1);
    cell_y = clamp(cell_y, 0, GRID_HEIGHT - 1);
    int cell_id = cell_y * GRID_WIDTH + cell_x;

    particle_cell_ids[gid] = cell_id;
    atomic_inc(&cell_particle_counts[cell_id]);
}

// Kernel 2: Compute density
kernel void compute_density(
    global float2* positions,
    global int* particle_cell_ids,
    global int* cell_particle_counts,
    global int* cell_particle_lists,
    global float* density,
    global float* debug,
    const int n_particles
) {
    int gid = get_global_id(0);
    if (gid >= n_particles) return;

    float2 pos_i = positions[gid];
    int cell_id_i = particle_cell_ids[gid];
    float rho = PARTICLE_MASS;

    int cell_y = cell_id_i / GRID_WIDTH;
    int cell_x = cell_id_i % GRID_WIDTH;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cell_x + dx;
            int ny = cell_y + dy;
            if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;

            int neighbor_cell_id = ny * GRID_WIDTH + nx;
            int count = cell_particle_counts[neighbor_cell_id];
            for (int j_idx = 0; j_idx < count; j_idx++) {
                int j = cell_particle_lists[neighbor_cell_id * n_particles + j_idx];
                float2 pos_j = positions[j];
                float dist_sq = dist_squared(pos_i, pos_j);
                if (dist_sq < SMOOTHING_LENGTH * SMOOTHING_LENGTH) {
                    rho += NORMALIZATION_DENSITY * pow(SMOOTHING_LENGTH * SMOOTHING_LENGTH - dist_sq, 3.0f);
                }
            }
        }
    }

    density[gid] = fmax(rho, MIN_DENSITY);
    debug[gid] = rho;
}

// Kernel 3: Compute forces
kernel void compute_forces(
    global float2* positions,
    global float2* velocities,
    global int* particle_cell_ids,
    global int* cell_particle_counts,
    global int* cell_particle_lists,
    global float* density,
    global float2* forces,
    const int n_particles
) {
    int gid = get_global_id(0);
    if (gid >= n_particles) return;

    float2 pos_i = positions[gid];
    float2 vel_i = velocities[gid];
    float rho_i = density[gid];
    float p_i = ISOTROPIC_EXPONENT * (rho_i - BASE_DENSITY);
    int cell_id_i = particle_cell_ids[gid];
    float2 force = (float2)(0.0f, 0.0f);

    int cell_y = cell_id_i / GRID_WIDTH;
    int cell_x = cell_id_i % GRID_WIDTH;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cell_x + dx;
            int ny = cell_y + dy;
            if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;

            int neighbor_cell_id = ny * GRID_WIDTH + nx;
            int count = cell_particle_counts[neighbor_cell_id];
            for (int j_idx = 0; j_idx < count; j_idx++) {
                int j = cell_particle_lists[neighbor_cell_id * n_particles + j_idx];
                if (j == gid) continue;

                float2 pos_j = positions[j];
                float dist_sq = dist_squared(pos_i, pos_j);
                if (dist_sq < SMOOTHING_LENGTH * SMOOTHING_LENGTH && dist_sq > 0.0001f) {
                    float dist = sqrt(dist_sq);
                    float2 dir = (pos_j - pos_i) / dist;
                    float rho_j = density[j];
                    float p_j = ISOTROPIC_EXPONENT * (rho_j - BASE_DENSITY);

                    // Pressure force
                    force += NORMALIZATION_PRESSURE_FORCE * (-dir) * ((p_i + p_j) / (2.0f * rho_j)) * pow(SMOOTHING_LENGTH - dist, 2.0f);

                    // Viscosity force
                    float2 vel_diff = velocities[j] - vel_i;
                    force += NORMALIZATION_VISCOUS_FORCE * (vel_diff / rho_j) * (SMOOTHING_LENGTH - dist);
                }
            }
        }
    }

    // Add gravity
    force += (float2)(0.0f, -1.0f);

    forces[gid] = force;
}

// Kernel 4: Update particles with enhanced splash effect
kernel void update_particles(
    global float2* positions,
    global float2* velocities,
    global float* density,
    global float2* forces,
    global float2* debug,
    const float dt,
    const int n_particles
) {
    int gid = get_global_id(0);
    if (gid >= n_particles) return;

    float rho_i = density[gid];
    float2 force_i = forces[gid];
    float2 vel_i = velocities[gid];
    float2 pos_i = positions[gid];

    // Update velocity
    vel_i += dt * force_i / rho_i;
    // Update position
    pos_i += dt * vel_i;

    // Flag for splash
    bool is_splashing = false;

    // Enforce boundary conditions with enhanced splash
    if (pos_i.x < DOMAIN_X_MIN) {
        pos_i.x = DOMAIN_X_MIN;
        vel_i.x *= DAMPING_COEFFICIENT;
        vel_i.x += 2.0f;  // Stronger wall repulsion
    }
    if (pos_i.x > DOMAIN_X_MAX) {
        pos_i.x = DOMAIN_X_MAX;
        vel_i.x *= DAMPING_COEFFICIENT;
        vel_i.x -= 2.0f;
    }
    if (pos_i.y < DOMAIN_Y_MIN) {
        pos_i.y = DOMAIN_Y_MIN;
        vel_i.y *= DAMPING_COEFFICIENT;
        vel_i.y += 5.0f;  // High bump for splash
        vel_i.x += 3.0f * (rand(gid * 12.9898f) - 0.5f);  // Wider lateral splash
        is_splashing = true;
    }
    if (pos_i.y > DOMAIN_Y_MAX) {
        pos_i.y = DOMAIN_Y_MAX;
        vel_i.y *= DAMPING_COEFFICIENT;
        vel_i.y -= 1.0f;
    }

    // Clamp velocity to prevent instability
    float vel_mag = length(vel_i);
    if (vel_mag > 12.0f) {
        vel_i *= 12.0f / vel_mag;
    }

    // Propagate splash to nearby particles
    if (is_splashing) {
        int cell_y = (int)(pos_i.y / GRID_CELL_SIZE);
        int cell_x = (int)(pos_i.x / GRID_CELL_SIZE);
        cell_x = clamp(cell_x, 0, GRID_WIDTH - 1);
        cell_y = clamp(cell_y, 0, GRID_HEIGHT - 1);

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cell_x + dx;
                int ny = cell_y + dy;
                if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;

                int neighbor_cell_id = ny * GRID_WIDTH + nx;
                for (int j = 0; j < n_particles; j++) {
                    if (j == gid) continue;
                    float2 pos_j = positions[j];
                    float dist_sq = dist_squared(pos_i, pos_j);
                    if (dist_sq < SMOOTHING_LENGTH * SMOOTHING_LENGTH && dist_sq > 0.0001f) {
                        float dist = sqrt(dist_sq);
                        float splash_factor = (SMOOTHING_LENGTH - dist) / SMOOTHING_LENGTH;
                        float2 vel_j = velocities[j];
                        vel_j.y += 2.5f * splash_factor;  // Stronger ripple
                        vel_j.x += 1.5f * splash_factor * (pos_j.x - pos_i.x) / (dist + 0.01f);  // Wider radial spread
                        float vel_j_mag = length(vel_j);
                        if (vel_j_mag > 12.0f) {
                            vel_j *= 12.0f / vel_j_mag;
                        }
                        velocities[j] = vel_j;
                    }
                }
            }
        }
    }

    // Check for NaN
    if (isnan(pos_i.x) || isnan(pos_i.y) || isnan(vel_i.x) || isnan(vel_i.y)) {
        pos_i = (float2)(DOMAIN_X_MAX * 0.5f, DOMAIN_Y_MAX - SMOOTHING_LENGTH);
        vel_i = (float2)(0.0f, 0.0f);
    }

    positions[gid] = pos_i;
    velocities[gid] = vel_i;
    debug[gid] = vel_i;
}