import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Params
NUM_PARTICLES = 50
DT = 0.003

# Init pos and velocity
positions = np.random.rand(NUM_PARTICLES, 2).astype(np.float32)
velocities = np.zeros((NUM_PARTICLES, 2), dtype=np.float32)
densities = np.ones(NUM_PARTICLES, dtype=np.float32) * 1000.0

# OpenCL setup
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

with open("sph_kernels.cl", "r") as f:
    kernel_code = f.read()

program = cl.Program(ctx, kernel_code).build()

mf = cl.mem_flags
pos_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=positions)
vel_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=velocities)
dens_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=densities)

kernel = program.sph_step
kernel.set_args(pos_buf, vel_buf, dens_buf, np.int32(NUM_PARTICLES))

# Plotting
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1], s=8, c="blue")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Basic SPH Fluid Simulation (GPU)")

def update(frame):
    cl.enqueue_nd_range_kernel(queue, kernel, (NUM_PARTICLES,), None)
    cl.enqueue_copy(queue, positions, pos_buf)
    scat.set_offsets(positions)
    return scat,

ani = animation.FuncAnimation(fig, update, interval=30)
plt.show()