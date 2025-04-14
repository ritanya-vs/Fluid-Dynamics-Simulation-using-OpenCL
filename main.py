import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle
from tqdm import tqdm
import streamlit as st

# Simulation parameters
MAX_PARTICLES = 500
DOMAIN_WIDTH = 40.0
DOMAIN_HEIGHT = 80.0
SMOOTHING_LENGTH = 3.5
TIME_STEP_LENGTH = 0.006
N_TIME_STEPS = 30
ADD_PARTICLES_EVERY = 3
PLOT_EVERY = 10
SCATTER_DOT_SIZE = 180
FIGURE_SIZE = (5, 7)

GRID_CELL_SIZE = SMOOTHING_LENGTH
GRID_WIDTH = int(DOMAIN_WIDTH / GRID_CELL_SIZE) + 1
GRID_HEIGHT = int(DOMAIN_HEIGHT / GRID_CELL_SIZE) + 1

# OpenCL setup
try:
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
except Exception as e:
    st.error(f"Failed to initialize OpenCL: {e}")
    st.stop()

# Load and build OpenCL program
try:
    with open("sph_kernels.cl", "r") as f:
        kernel_src = f.read()
    program = cl.Program(context, kernel_src).build()
except Exception as e:
    st.error(f"Failed to build OpenCL program: {e}")
    st.stop()

# Initialize particle data
n_particles = 0
positions = np.zeros((MAX_PARTICLES, 2), dtype=np.float32)
velocities = np.zeros((MAX_PARTICLES, 2), dtype=np.float32)
density = np.zeros(MAX_PARTICLES, dtype=np.float32)
forces = np.zeros((MAX_PARTICLES, 2), dtype=np.float32)
particle_cell_ids = np.zeros(MAX_PARTICLES, dtype=np.int32)
cell_particle_counts = np.zeros(GRID_WIDTH * GRID_HEIGHT, dtype=np.int32)
cell_particle_lists = np.zeros((GRID_WIDTH * GRID_HEIGHT, MAX_PARTICLES), dtype=np.int32)
debug_density = np.zeros(MAX_PARTICLES, dtype=np.float32)
debug_velocities = np.zeros((MAX_PARTICLES, 2), dtype=np.float32)

# GPU buffers
positions_buf = cl_array.to_device(queue, positions)
velocities_buf = cl_array.to_device(queue, velocities)
density_buf = cl_array.to_device(queue, density)
forces_buf = cl_array.to_device(queue, forces)
particle_cell_ids_buf = cl_array.to_device(queue, particle_cell_ids)
cell_particle_counts_buf = cl_array.to_device(queue, cell_particle_counts)
cell_particle_lists_buf = cl_array.to_device(queue, cell_particle_lists)
debug_density_buf = cl_array.to_device(queue, debug_density)
debug_velocities_buf = cl_array.to_device(queue, debug_velocities)

# Streamlit app setup
st.title("Smoothed Particle Hydrodynamics Simulation")
st.markdown("""
    Visualize a GPU-accelerated SPH simulation with particles, splashes, and bubbles.
    Adjust parameters below to control the simulation.
""")

# Sidebar for controls
st.sidebar.header("Simulation Parameters")
n_time_steps = st.sidebar.slider("Number of Time Steps", 100, 5000, N_TIME_STEPS, step=100)
plot_every = st.sidebar.slider("Plot Every N Steps", 1, 50, PLOT_EVERY)
run_simulation = st.sidebar.button("Run Simulation")

# Placeholder for the plot
plot_placeholder = st.empty()

# Initialize bubbles list
bubbles = []  # Format: [x, y, size, age]

# Main simulation loop (runs only when button is clicked)
if run_simulation:
    progress_bar = st.progress(0)
    for iter in tqdm(range(n_time_steps), desc="Simulating"):
        # Add new particles
        if iter % ADD_PARTICLES_EVERY == 0 and n_particles < MAX_PARTICLES - 8:
            new_positions = np.zeros((8, 2), dtype=np.float32)
            new_velocities = np.zeros((8, 2), dtype=np.float32)
            for i in range(8):
                x_pos = SMOOTHING_LENGTH + (DOMAIN_WIDTH - 2 * SMOOTHING_LENGTH) * (i + np.random.uniform(-0.1, 0.1)) / 8
                x_pos = np.clip(x_pos, SMOOTHING_LENGTH, DOMAIN_WIDTH - SMOOTHING_LENGTH)
                new_positions[i] = [x_pos, DOMAIN_HEIGHT - SMOOTHING_LENGTH]
                new_velocities[i] = [np.random.uniform(-2.5, 2.5), np.random.uniform(-3.0, -1.0)]
            
            positions[n_particles:n_particles+8] = new_positions
            velocities[n_particles:n_particles+8] = new_velocities
            n_particles += 8
            
            positions_buf[:n_particles] = positions[:n_particles]
            velocities_buf[:n_particles] = velocities[:n_particles]
            
            st.write(f"Iter {iter}: Added 8 particles, total={n_particles}")

        # Reset cell counts
        cell_particle_counts.fill(0)
        cell_particle_lists.fill(0)
        cl.enqueue_copy(queue, cell_particle_counts_buf.data, cell_particle_counts)
        cl.enqueue_copy(queue, cell_particle_lists_buf.data, cell_particle_lists)

        # Step 1: Assign particles to grid
        program.assign_to_grid(queue, (MAX_PARTICLES,), None,
                              positions_buf.data, particle_cell_ids_buf.data,
                              cell_particle_counts_buf.data, np.int32(n_particles))

        # Update cell_particle_lists on host
        particle_cell_ids = particle_cell_ids_buf.get()[:n_particles]
        cell_particle_counts.fill(0)
        cell_particle_lists.fill(0)
        for i in range(n_particles):
            cell_id = particle_cell_ids[i]
            if (cell_id >= 0 and cell_id < GRID_WIDTH * GRID_HEIGHT):
                idx = cell_particle_counts[cell_id]
                cell_particle_lists[cell_id, idx] = i
                cell_particle_counts[cell_id] += 1
        cl.enqueue_copy(queue, cell_particle_counts_buf.data, cell_particle_counts)
        cl.enqueue_copy(queue, cell_particle_lists_buf.data, cell_particle_lists)

        # Step 2: Compute density
        program.compute_density(queue, (MAX_PARTICLES,), None,
                               positions_buf.data, particle_cell_ids_buf.data,
                               cell_particle_counts_buf.data, cell_particle_lists_buf.data,
                               density_buf.data, debug_density_buf.data, np.int32(n_particles))

        # Step 3: Compute forces
        program.compute_forces(queue, (MAX_PARTICLES,), None,
                              positions_buf.data, velocities_buf.data,
                              particle_cell_ids_buf.data, cell_particle_counts_buf.data,
                              cell_particle_lists_buf.data, density_buf.data,
                              forces_buf.data, np.int32(n_particles))

        # Step 4: Update particles with splash
        program.update_particles(queue, (MAX_PARTICLES,), None,
                                positions_buf.data, velocities_buf.data,
                                density_buf.data, forces_buf.data,
                                debug_velocities_buf.data, np.float32(TIME_STEP_LENGTH),
                                np.int32(n_particles))

        # Copy data back
        positions[:n_particles] = positions_buf.get()[:n_particles]
        velocities[:n_particles] = velocities_buf.get()[:n_particles]
        debug_density[:n_particles] = debug_density_buf.get()[:n_particles]
        debug_velocities[:n_particles] = debug_velocities_buf.get()[:n_particles]

        # Plot
        if iter % plot_every == 0:
            # Create new figure for each plot
            fig = plt.figure(figsize=FIGURE_SIZE, dpi=160)
            fig.patch.set_facecolor("#001F33")
            ax = fig.add_subplot(111)
            ax.set_facecolor("#001F33")
            
            # Glowing source
            ax.add_patch(Rectangle((SMOOTHING_LENGTH, DOMAIN_HEIGHT - SMOOTHING_LENGTH - 1), 
                                  DOMAIN_WIDTH - 2 * SMOOTHING_LENGTH, 1, 
                                  fill=True, color="#00FFFF", alpha=0.3, zorder=0))
            
            valid_positions = positions[:n_particles]
            valid_velocities = debug_velocities[:n_particles]
            valid_density = debug_density[:n_particles]
            valid_mask = ~np.isnan(valid_positions[:, 0])
            valid_positions = valid_positions[valid_mask]
            valid_velocities = valid_velocities[valid_mask]
            valid_density = valid_density[valid_mask]
            if len(valid_positions) > 0:
                # Identify splashing particles
                splash_mask = (valid_positions[:, 1] < SMOOTHING_LENGTH + 5.0) & (valid_velocities[:, 1] > 0.5)
                # Identify wall-hit particles
                wall_mask = (valid_positions[:, 0] < SMOOTHING_LENGTH + 2.0) | (valid_positions[:, 0] > DOMAIN_WIDTH - SMOOTHING_LENGTH - 2.0)
                sizes = SCATTER_DOT_SIZE * (1 + 0.5 * (valid_density / np.max(valid_density + 1e-6)))
                # Color by height
                colors = np.array([(0.0, 0.75, 1.0)] * len(valid_positions))  # Base #00BFFF
                colors[:, 1] += 0.25 * (valid_positions[:, 1] / DOMAIN_HEIGHT)
                colors = np.clip(colors, 0, 1)
                
                # Plot trails
                trail_positions = valid_positions - 0.1 * valid_velocities
                trail_mask = ~(splash_mask | wall_mask)
                plt.scatter(
                    trail_positions[trail_mask, 0],
                    trail_positions[trail_mask, 1],
                    s=sizes[trail_mask] * 0.5,
                    c=colors[trail_mask],
                    alpha=0.15,
                    zorder=1,
                )
                
                # Plot normal particles
                normal_mask = ~(splash_mask | wall_mask)
                plt.scatter(
                    valid_positions[normal_mask, 0],
                    valid_positions[normal_mask, 1],
                    s=sizes[normal_mask],
                    c=colors[normal_mask],
                    alpha=0.4,
                    edgecolors="white",
                    linewidth=0.5,
                    zorder=2,
                )
                # Plot splashing particles
                if np.any(splash_mask):
                    splash_alpha = 0.25 + 0.1 * (valid_velocities[splash_mask, 1] / 5.0)
                    plt.scatter(
                        valid_positions[splash_mask, 0],
                        valid_positions[splash_mask, 1],
                        s=sizes[splash_mask] * 2.0,
                        c="#ADD8E6",
                        alpha=np.clip(splash_alpha, 0.25, 0.35),
                        edgecolors="white",
                        linewidth=0.7,
                        zorder=3,
                    )
                    # Mist particles
                    mist_count = min(10, np.sum(splash_mask))
                    mist_pos = valid_positions[splash_mask][:mist_count] + np.random.uniform(-0.5, 0.5, size=(mist_count, 2))
                    plt.scatter(
                        mist_pos[:, 0],
                        mist_pos[:, 1],
                        s=sizes[splash_mask][:mist_count] * 0.3,
                        c="#E0FFFF",
                        alpha=0.1,
                        zorder=3,
                    )
                    # Ripples
                    ripple_centers = valid_positions[splash_mask]
                    for center in ripple_centers[:5]:
                        ripple = Ellipse((center[0], center[1]), 
                                        width=2 + 0.5 * np.sin(iter * 0.1), 
                                        height=1 + 0.25 * np.sin(iter * 0.1), 
                                        fill=False, edgecolor="#A0E0E0", alpha=0.2, zorder=1)
                        ax.add_patch(ripple)
                # Plot wall-hit particles
                if np.any(wall_mask):
                    plt.scatter(
                        valid_positions[wall_mask, 0],
                        valid_positions[wall_mask, 1],
                        s=sizes[wall_mask] * 1.2,
                        c="#87CEEB",
                        alpha=0.35,
                        edgecolors="white",
                        linewidth=0.6,
                        zorder=2,
                    )
                # Update bubbles
                new_bubbles = []
                if np.any(splash_mask):
                    for pos in valid_positions[splash_mask][:10]:
                        if np.random.random() < 0.3:
                            bubbles.append([pos[0], pos[1], np.random.uniform(0.5, 1.5), 0])
                for bubble in bubbles:
                    bubble[1] += 0.2 + 0.1 * bubble[2]
                    bubble[3] += 1
                    if bubble[1] < DOMAIN_HEIGHT - SMOOTHING_LENGTH and bubble[3] < 50:
                        new_bubbles.append(bubble)
                        bubble_patch = Circle((bubble[0], bubble[1]), 
                                            radius=bubble[2] * 0.5, 
                                            fill=False, edgecolor="#E0FFFF", alpha=0.3, zorder=1)
                        ax.add_patch(bubble_patch)
                bubbles = new_bubbles[:50]
            else:
                plt.scatter([DOMAIN_WIDTH/2], [DOMAIN_HEIGHT/2], s=SCATTER_DOT_SIZE, c="red", label="No valid particles")
                st.write(f"Iter {iter}: No valid particles, plotting test point")
            
            # Glowing frame
            ax.add_patch(Rectangle((0, 0), DOMAIN_WIDTH, DOMAIN_HEIGHT, 
                                  fill=False, edgecolor="#00FFFF", linewidth=3, alpha=0.7, zorder=5))
            
            plt.xlim([SMOOTHING_LENGTH, DOMAIN_WIDTH - SMOOTHING_LENGTH])
            plt.ylim([SMOOTHING_LENGTH, DOMAIN_HEIGHT - SMOOTHING_LENGTH])
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            
            # Render the plot in Streamlit
            plot_placeholder.pyplot(fig)
            plt.close(fig)  # Close figure to free memory

        progress_bar.progress((iter + 1) / n_time_steps)

    st.success("Simulation complete!")