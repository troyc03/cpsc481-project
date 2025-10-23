import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =============================
# Lorenz System Definitions
# =============================

def lorenz(state, *, s=10, r=28, b=2.667):
    """Compute the Lorenz system derivatives."""
    x, y, z = state
    dx = s * (y - x)
    dy = r * x - y - x * z
    dz = x * y - b * z
    return np.array([dx, dy, dz])

# =============================
# Runge-Kutta 4 Integration
# =============================

def rk4(func, initial_state, dt=0.01, steps=10000, **kwargs):
    """Fourth-order Runge-Kutta integration."""
    states = np.empty((steps + 1, len(initial_state)))
    states[0] = initial_state
    for i in range(steps):
        k1 = func(states[i], **kwargs)
        k2 = func(states[i] + dt * k1 / 2, **kwargs)
        k3 = func(states[i] + dt * k2 / 2, **kwargs)
        k4 = func(states[i] + dt * k3, **kwargs)
        states[i + 1] = states[i] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return states

# =============================
# GUI with Animation
# =============================

def launch_gui():
    """Launch a Tkinter GUI with Lorenz attractor animation."""
    root = tk.Tk()
    root.title("Lorenz Attractor Simulation")

    # --- Figure Setup ---
    fig = plt.Figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=5, column=0, columnspan=2)

    # --- Input Fields Helper ---
    def labeled_entry(label_text, row, default):
        tk.Label(root, text=label_text).grid(row=row, column=0, sticky="e")
        entry = tk.Entry(root)
        entry.insert(0, default)
        entry.grid(row=row, column=1)
        return entry

    # Entry fields for initial conditions
    entry_x = labeled_entry("Initial X:", 0, "0.0")
    entry_y = labeled_entry("Initial Y:", 1, "1.0")
    entry_z = labeled_entry("Initial Z:", 2, "1.05")

    # --- Animation Function ---
    def run_sim():
        """Run and animate Lorenz trajectory."""
        # Get initial conditions
        q0 = np.array([
            float(entry_x.get()),
            float(entry_y.get()),
            float(entry_z.get())
        ])

        # Compute trajectory
        dt = 0.01
        steps = 8000
        data = rk4(lorenz, q0, dt=dt, steps=steps)

        # Clear previous plot
        ax.clear()

        # Plot full trajectory (faint blue)
        ax.plot(*data.T, color='blue', lw=0.5, alpha=0.4)

        # Create moving particle and trail
        (point,) = ax.plot([], [], [], 'ro', markersize=5)
        (trail,) = ax.plot([], [], [], 'r-', lw=1.0, alpha=0.7)

        # Configure 3D axes
        ax.set(
            xlabel="X Axis", ylabel="Y Axis", zlabel="Z Axis",
            title="Lorenz Attractor (Animated)",
            xlim=(np.min(data[:, 0]), np.max(data[:, 0])),
            ylim=(np.min(data[:, 1]), np.max(data[:, 1])),
            zlim=(np.min(data[:, 2]), np.max(data[:, 2]))
        )

        trail_length = 150  # number of steps for visible trail

        # Update function for animation
        def update(frame):
            start = max(0, frame - trail_length)
            x, y, z = data[start:frame + 1].T
            point.set_data([data[frame, 0]], [data[frame, 1]])
            point.set_3d_properties([data[frame, 2]])
            trail.set_data(x, y)
            trail.set_3d_properties(z)
            return point, trail

        # Create animation
        ani = FuncAnimation(fig, update, frames=steps, interval=10, blit=True)

        # Draw the animation frame in Tkinter
        canvas.draw()

    # --- Run Button ---
    tk.Button(root, text="Run Simulation", command=run_sim).grid(
        row=3, column=0, columnspan=2, pady=10
    )

    root.mainloop()

# =============================
# Main Execution
# =============================

if __name__ == "__main__":
    launch_gui()
