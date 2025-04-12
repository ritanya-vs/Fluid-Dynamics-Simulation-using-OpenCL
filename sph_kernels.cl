#define SMOOTHING_LENGTH 0.04f
#define MASS 1.0f
#define STIFFNESS 200.0f
#define REST_DENSITY 1000.0f
#define GRAVITY -9.8f
#define DT 0.003f
#define VISCOSITY 10.0f
#define EPSILON 1e-6f

float dist2(float2 a, float2 b) {
    float2 diff = a - b;
    return dot(diff, diff);
}

// Poly6 kernel
float poly6(float r2, float h2) {
    float diff = h2 - r2;
    return (315.0f / (64.0f * M_PI_F * pow(h2, 1.5f))) * pow(diff, 3.0f);
}

// Spiky gradient kernel for pressure force
float2 spiky_gradient(float2 r, float h) {
    float r_len = length(r);
    if (r_len < h && r_len > EPSILON) {
        float coeff = -45.0f / (M_PI_F * pow(h, 6.0f));
        return coeff * pow(h - r_len, 2.0f) * (r / r_len);
    }
    return (float2)(0.0f, 0.0f);
}

// Viscosity kernel Laplacian
float viscosity_laplacian(float r, float h) {
    if (r < h) {
        return 45.0f / (M_PI_F * pow(h, 6.0f)) * (h - r);
    }
    return 0.0f;
}

__kernel void sph_step(
    __global float2* positions,
    __global float2* velocities,
    __global float* densities,
    const int num_particles)
{
    int i = get_global_id(0);
    float2 pos_i = positions[i];
    float2 vel_i = velocities[i];
    float h = SMOOTHING_LENGTH;
    float h2 = h * h;
    float density = 0.0f;

    // Compute density
    for (int j = 0; j < num_particles; j++) {
        float r2 = dist2(pos_i, positions[j]);
        if (r2 < h2) {
            density += MASS * poly6(r2, h2);
        }
    }

    densities[i] = density;

    // Compute forces
    float pressure_i = STIFFNESS * (density - REST_DENSITY);
    float2 pressure_force = (float2)(0.0f, 0.0f);
    float2 viscosity_force = (float2)(0.0f, 0.0f);

    for (int j = 0; j < num_particles; j++) {
        if (i == j) continue;

        float2 r = pos_i - positions[j];
        float r_len = length(r);
        if (r_len < h && r_len > EPSILON) {
            float pressure_j = STIFFNESS * (densities[j] - REST_DENSITY);
            float avg_pressure = 0.5f * (pressure_i + pressure_j);
            pressure_force += -MASS * avg_pressure / densities[j] * spiky_gradient(r, h);

            // Viscosity
            float2 vel_diff = velocities[j] - vel_i;
            viscosity_force += VISCOSITY * MASS * vel_diff / densities[j] * viscosity_laplacian(r_len, h);
        }
    }

    // Total force
    float2 gravity_force = (float2)(0.0f, GRAVITY);
    float2 total_force = pressure_force + viscosity_force + gravity_force;

    // Integrate
    vel_i += DT * total_force;
    pos_i += DT * vel_i;

    // Boundaries
    if (pos_i.x < 0.0f) { pos_i.x = 0.0f; vel_i.x *= -0.5f; }
    if (pos_i.x > 1.0f) { pos_i.x = 1.0f; vel_i.x *= -0.5f; }
    if (pos_i.y < 0.0f) { pos_i.y = 0.0f; vel_i.y *= -0.5f; }
    if (pos_i.y > 1.0f) { pos_i.y = 1.0f; vel_i.y *= -0.5f; }

    velocities[i] = vel_i;
    positions[i] = pos_i;
}