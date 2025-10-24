import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.integrate import odeint
import pysindy as ps

# --- Lorenz system ---
def lorenz(state, t, s=10, r=28, b=2.667):
    x, y, z = state
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return [x_dot, y_dot, z_dot]


class LorenzApp:
    def __init__(self, master):
        self.master = master
        master.title("Lorenz Attractor Simulator (Interactive)")
        master.geometry("1000x700")

        # Default parameters
        self.s = tk.DoubleVar(value=10)
        self.r = tk.DoubleVar(value=28)
        self.b = tk.DoubleVar(value=2.667)

        # Create layout frames
        self.frame_controls = ttk.Frame(master, padding=10)
        self.frame_controls.pack(side=tk.LEFT, fill=tk.Y)
        self.frame_plot = ttk.Frame(master)
        self.frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Control widgets ---
        ttk.Label(self.frame_controls, text="Lorenz Parameters", font=("Arial", 12, "bold")).pack(pady=5)

        self.add_slider("s", self.s, 0, 30, 0.5)
        self.add_slider("r", self.r, 0, 50, 0.5)
        self.add_slider("b", self.b, 0.5, 5, 0.1)

        ttk.Button(self.frame_controls, text="Run Simulation", command=self.run_simulation).pack(pady=10)
        ttk.Button(self.frame_controls, text="Reset View", command=self.reset_view).pack(pady=5)

        # --- Matplotlib figure setup ---
        self.fig = plt.Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Animation handle
        self.ani = None

        # Run first simulation
        self.run_simulation()

    def add_slider(self, label, var, frm, to, step):
        frame = ttk.Frame(self.frame_controls)
        frame.pack(pady=5)
        ttk.Label(frame, text=label).pack()
        slider = ttk.Scale(frame, variable=var, from_=frm, to=to, orient=tk.HORIZONTAL)
        slider.pack(fill=tk.X, padx=10)
        ttk.Label(frame, textvariable=var).pack()

    def run_simulation(self):
        # Clear previous plots
        self.ax.clear()

        # --- Parameters ---
        s, r, b = self.s.get(), self.r.get(), self.b.get()

        # --- Time and initial condition ---
        dt = 0.01
        t = np.arange(0, 40, dt)
        init_state = [0., 1., 1.05]

        # --- Integrate Lorenz system ---
        xyzs = odeint(lorenz, init_state, t, args=(s, r, b))

        # --- Fit SINDy model ---
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.05),
            feature_library=ps.PolynomialLibrary(degree=2)
        )
        model.fit(xyzs, t=dt)
        print("\n--- SINDy Model ---")
        model.print()

        # --- Simulate with SINDy ---
        x_sim = model.simulate(init_state, t)

        # --- Plot trajectories ---
        self.ax.plot(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2],
                     lw=0.5, color='blue', alpha=0.6, label="True Trajectory")
        self.ax.plot(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2],
                     lw=0.5, color='orange', alpha=0.6, label="SINDy Reconstruction")

        # --- Moving particle ---
        (self.point,) = self.ax.plot([xyzs[0, 0]], [xyzs[0, 1]], [xyzs[0, 2]],
                                     marker='o', color='red', markersize=5)

        # --- Axis labels ---
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Y Axis")
        self.ax.set_zlabel("Z Axis")
        self.ax.set_title(f"Lorenz Attractor (s={s:.2f}, r={r:.2f}, b={b:.2f})")
        self.ax.legend()

        # Set axis limits
        self.ax.set_xlim(np.min(xyzs[:, 0]), np.max(xyzs[:, 0]))
        self.ax.set_ylim(np.min(xyzs[:, 1]), np.max(xyzs[:, 1]))
        self.ax.set_zlim(np.min(xyzs[:, 2]), np.max(xyzs[:, 2]))

        # --- Animation ---
        if self.ani:
            self.ani.event_source.stop()

        def update(i):
            self.point.set_data(np.array([xyzs[i, 0]]), np.array([xyzs[i, 1]]))
            self.point.set_3d_properties(xyzs[i, 2])
            return (self.point,)

        self.ani = FuncAnimation(self.fig, update, frames=len(t), interval=10, blit=True)
        self.canvas.draw()

    def reset_view(self):
        self.ax.view_init(elev=30, azim=45)
        self.canvas.draw()


# --- Run the app ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LorenzApp(root)
    root.mainloop()
