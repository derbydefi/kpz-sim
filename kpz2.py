import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 200  # System size
dx = .5  # Spatial resolution
dt = 0.001  # Time step
nu = 1.0  # Coefficient of surface tension
lambda_ = 0.5  # Nonlinearity coefficient
T = 10  # Total simulation time
D = 0.1  # Strength of the noise

# Initialize the height array
h = np.zeros(L)

# Time evolution function
def evolve(h, nu, lambda_, D, dx, dt, T):
    N = int(T / dt)
    h_evolution = np.zeros((N, len(h)))
    for n in range(N):
        # Laplacian term
        laplacian = (np.roll(h, -1) - 2 * h + np.roll(h, 1)) / dx**2
        # Gradient squared term
        gradient_sq = ((np.roll(h, -1) - np.roll(h, 1)) / (2 * dx))**2
        # Noise
        eta = np.sqrt(D /dx) * np.random.normal(0, 1, size=h.shape)
        # Update rule
        h += dt * (nu * laplacian + lambda_ * gradient_sq + eta)
        h_evolution[n] = h
    return h_evolution

# Run the simulation
h_evolution = evolve(h, nu, lambda_, D, dx, dt, T)

# Plotting
plt.figure(figsize=(10, 6))
plt.imshow(h_evolution, aspect='auto', origin='lower',
           cmap='viridis', extent=(0, L*dx, 0, T))
plt.colorbar(label='Height')
plt.xlabel('Space')
plt.ylabel('Time')
plt.title('KPZ Equation Simulation')
plt.show()
