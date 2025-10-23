import pysindy as ps
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import derivative as dxdt
import tkinter as tk
from sklearn.linear_model import Lasso, ridge_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation

# Build Lorenz Attractor equations with parameters
def lorenz(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])

# Time parameters
dt = 0.01
num_steps = 10000

# Initialize storage and starting point
xyzs = np.empty((num_steps + 1, 3))
xyzs[0] = (0., 1., 1.05)

# Integrate the system using Euler's method
for i in range(num_steps):
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

# Set up the figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot the Lorenz attractor trajectory
ax.plot(*xyzs.T, lw=0.5, color='blue', alpha=0.6)

# Create a point that will move along the path
(point,) = ax.plot([xyzs[0, 0]], [xyzs[0, 1]], [xyzs[0, 2]],
                   marker='o', color='red', markersize=5)

# Labels and title
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor with Moving Particle")

# Set nice view limits
ax.set_xlim(np.min(xyzs[:, 0]), np.max(xyzs[:, 0]))
ax.set_ylim(np.min(xyzs[:, 1]), np.max(xyzs[:, 1]))
ax.set_zlim(np.min(xyzs[:, 2]), np.max(xyzs[:, 2]))

# Animation update function
def update(i):
    point.set_data(np.array([xyzs[i, 0]]), np.array([xyzs[i, 1]]))
    point.set_3d_properties(xyzs[i, 2])
    return (point,)

# Animate the moving particle
ani = FuncAnimation(fig, update, frames=num_steps, interval=10, blit=True)

plt.show()


