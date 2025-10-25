import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.integrate import odeint
import pysindy as ps
import threading

# --- Lorenz system ---
def lorenz(state, t, s=10, r=28, b=2.667):
    x, y, z = state
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return [x_dot, y_dot, z_dot]

# --- Jacobian ---
def jacobian(q, s=10.0, r=28.0, b=8.0/3.0):
    x, y, z = q
    return np.array([
        [-s,      s,   0],
        [r - z,  -1,  -x],
        [y,       x,  -b]
    ])

class LorenzApp:
    def __init__(self, master):
        self.master = master
        master.title("Lorenz Attractor Simulator")
        master.geometry("1000x700")

        # --- Parameters ---
        self.s = tk.DoubleVar(value=10)
        self.r = tk.DoubleVar(value=28)
        self.b = tk.DoubleVar(value=2.667)

        # --- Layout ---
        self.frame_controls = ttk.Frame(master, padding=10)
        self.frame_controls.pack(side=tk.LEFT, fill=tk.Y)
        self.frame_plot = ttk.Frame(master)
        self.frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(self.frame_controls, text="Lorenz Parameters", font=("Arial", 12, "bold")).pack(pady=5)
        self.add_slider("s", self.s, 0, 30, 0.5)
        self.add_slider("r", self.r, 0, 50, 0.5)
        self.add_slider("b", self.b, 0.5, 5, 0.1)

        ttk.Button(self.frame_controls, text="Run Simulation", command=self.start_thread).pack(pady=10)
        ttk.Button(self.frame_controls, text="Reset View", command=self.reset_view).pack(pady=5)

        self.status = ttk.Label(self.frame_controls, text="Ready", foreground="green")
        self.status.pack(pady=5)

        # --- Figure Setup ---
        self.fig = plt.Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ani = None

        # Initial data placeholders
        self.xyzs = None
        self.x_sim = None
        self.t = None

        # Initial plot
        self.run_initial_plot()

    def add_slider(self, label, var, frm, to, step):
        frame = ttk.Frame(self.frame_controls)
        frame.pack(pady=5)
        ttk.Label(frame, text=label).pack()
        slider = ttk.Scale(frame, variable=var, from_=frm, to=to, orient=tk.HORIZONTAL)
        slider.pack(fill=tk.X, padx=10)
        ttk.Label(frame, textvariable=var).pack()

    def start_thread(self):
        """Starts the simulation computation in a background thread."""
        thread = threading.Thread(target=self.compute_simulation, daemon=True)
        thread.start()

    def compute_simulation(self):
        """Run the numerical computations (in background)."""
        try:
            self.status.config(text="Computing...", foreground="orange")

            # Parameters
            s, r, b = self.s.get(), self.r.get(), self.b.get()
            dt = 0.01
            t = np.arange(0, 5, dt)
            init_state = [0., 1., 1.05]

            # Heavy computations here only (no GUI calls)
            xyzs = odeint(lorenz, init_state, t, args=(s, r, b))

            model = ps.SINDy(
                optimizer=ps.STLSQ(threshold=0.05),
                feature_library=ps.PolynomialLibrary(degree=2)
            )
            model.fit(xyzs, t=dt)
            print("\n--- SINDy Model ---")
            model.print()

            x_sim = model.simulate(init_state, t)

            # Store results for main-thread plotting
            self.xyzs, self.x_sim, self.t = xyzs, x_sim, t

            # Schedule the plot update on main thread
            self.master.after(0, lambda: self.update_plot(s, r, b))

        except:
            self.status.config(text="Error", foreground="red")

    def update_plot(self, s, r, b):
        """Update Matplotlib plot safely from main thread."""
        self.ax.clear()

        xyzs = self.xyzs
        x_sim = self.x_sim
        t = self.t

        self.ax.plot(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2],
                     lw=0.5, color='blue', alpha=0.6, label="True Trajectory")
        self.ax.plot(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2],
                     lw=0.5, color='orange', alpha=0.6, label="SINDy Reconstruction")

        (self.point,) = self.ax.plot([xyzs[0, 0]], [xyzs[0, 1]], [xyzs[0, 2]],
                                     marker='o', color='red', markersize=5)

        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Y Axis")
        self.ax.set_zlabel("Z Axis")
        self.ax.set_title(f"Lorenz Attractor (s={s:.2f}, r={r:.2f}, b={b:.2f})")
        self.ax.legend()

        # Adjust limits
        self.ax.set_xlim(np.min(xyzs[:, 0]), np.max(xyzs[:, 0]))
        self.ax.set_ylim(np.min(xyzs[:, 1]), np.max(xyzs[:, 1]))
        self.ax.set_zlim(np.min(xyzs[:, 2]), np.max(xyzs[:, 2]))

        # Stop old animation safely
        if self.ani:
            self.ani.event_source.stop()

        def update(i):
            self.point.set_data(np.array([xyzs[i, 0]]), np.array([xyzs[i, 1]]))
            self.point.set_3d_properties(xyzs[i, 2])
            return (self.point,)

        self.ani = FuncAnimation(self.fig, update, frames=len(t), interval=10, blit=True)
        self.canvas.draw()

        self.status.config(text="Simulation Complete", foreground="green")

    def run_initial_plot(self):
        """Initial blank plot to make GUI load fast."""
        self.ax.set_title("Lorenz Attractor (waiting for simulation)")
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Y Axis")
        self.ax.set_zlabel("Z Axis")
        self.canvas.draw()

    def reset_view(self):
        self.ax.view_init(elev=30, azim=45)
        self.canvas.draw()


# --- Run the App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LorenzApp(root)
    root.mainloop()
