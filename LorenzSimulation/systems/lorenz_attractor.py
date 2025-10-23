import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pysindy as ps
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Lorenz system ---
def lorenz(state, t, s=10, r=28, b=2.667):
    x, y, z = state
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return [x_dot, y_dot, z_dot]

# --- Time parameters ---
dt = 0.01
t = np.arange(0, 40, dt)

# --- Initial condition ---
init_state = [0., 1., 1.05]

# --- Integrate Lorenz system numerically ---
xyzs = odeint(lorenz, init_state, t)

# --- Fit SINDy model ---
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.05),
    feature_library=ps.PolynomialLibrary(degree=2)
)
model.fit(xyzs, t=dt)
model.print()

# --- Simulate SINDy reconstruction ---
x_sim = model.simulate(init_state, t)

# Build GUI
root = tk.Tk()
root.title('Lorenz Attractor with Particle')
root.geometry('800x600')

# --- Plot Lorenz attractor ---
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2],
        lw=0.5, color='blue', alpha=0.6, label="True Trajectory")
ax.plot(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2],
        lw=0.5, color='orange', alpha=0.6, label="SINDy Reconstruction")

(point,) = ax.plot([xyzs[0, 0]], [xyzs[0, 1]], [xyzs[0, 2]],
                   marker='o', color='red', markersize=5)

ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor with Moving Particle")
ax.legend()

# --- Set nice view limits ---
ax.set_xlim(np.min(xyzs[:, 0]), np.max(xyzs[:, 0]))
ax.set_ylim(np.min(xyzs[:, 1]), np.max(xyzs[:, 1]))
ax.set_zlim(np.min(xyzs[:, 2]), np.max(xyzs[:, 2]))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)

# --- Animation update function ---
def update(i):
    point.set_data(np.array([xyzs[i, 0]]), np.array([xyzs[i, 1]]))
    point.set_3d_properties(xyzs[i, 2])
    return (point,)

ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)
plt.show()
canvas.draw()
root.mainloop()