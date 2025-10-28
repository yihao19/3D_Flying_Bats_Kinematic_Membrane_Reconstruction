import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 5  # Grid size (N x N)
mass = 1.0  # Mass of each point mass
k_s = 0.1  # Structural spring constant
k_sh = 0  # Shear spring constant
k_f = 1  # Flexion spring constant
damping = 0.01  # Damping factor
dt = 0.01  # Time step
steps = 1000  # Number of simulation steps

# Initialize positions and velocities
positions = np.zeros((N, N, 2))  # (x, y) positions
velocities = np.zeros_like(positions)

for i in range(N):
    for j in range(N):
        positions[i, j] = np.array([i / (N - 1), j / (N - 1)])

# Define rest lengths for different springs
rest_lengths = {
    'structural': np.ones((N-1, N-1)) / (N-1),
    'shear': np.ones((N-1, N-1)) * np.sqrt(2)/(N-1),
    'flexion': np.ones((N-2, N-2)) * 2.0
}
# Force calculation
def compute_forces():
    forces = np.zeros_like(positions)
    # Structural forces
    for i in range(N-1):
        for j in range(N-1):
            # Horizontal springs
            disp = positions[i+1, j] - positions[i, j]
            dist = np.linalg.norm(disp)
            force = k_s * (dist - rest_lengths['structural'][i, j]) * disp / dist
            forces[i+1, j] -= force
            forces[i, j] += force
            # Vertical springs
            disp = positions[i, j+1] - positions[i, j]
            dist = np.linalg.norm(disp)
            force = k_s * (dist - rest_lengths['structural'][i, j]) * disp / dist
            forces[i, j+1] -= force
            forces[i, j] += force
    # Shear forces
    for i in range(N-1):
        for j in range(N-1):
            # Diagonal springs
            disp = positions[i+1, j+1] - positions[i, j]
            dist = np.linalg.norm(disp)
            force = k_sh * (dist - rest_lengths['shear'][i, j]) * disp / dist
            forces[i+1, j+1] -= force
            forces[i, j] += force
            # Anti-diagonal springs
            disp = positions[i+1, j-1] - positions[i, j]
            dist = np.linalg.norm(disp)
            force = k_sh * (dist - rest_lengths['shear'][i, j]) * disp / dist
            forces[i+1, j-1] -= force
            forces[i, j] += force
    # Flexion forces
    for i in range(N-2):
        for j in range(N-2):
            # Bending springs
            disp = positions[i+2, j+2] - positions[i, j]
            dist = np.linalg.norm(disp)
            force = k_f * (dist - rest_lengths['flexion'][i, j]) * disp / dist
            forces[i+2, j+2] -= force
            forces[i, j] += force
    return forces

# Simulation loop
for step in range(steps):
    forces = compute_forces()
    # Update velocities and positions
    velocities += forces * dt / mass
    positions += velocities * dt
    # Apply damping
    velocities *= (1 - damping)

    # Visualization (every 10 steps)
    if step % 10 == 0:
        plt.clf()
        plt.scatter(positions[:, :, 0], positions[:, :, 1])
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.title(f"Step {step}")
        plt.pause(0.01)

plt.show()
