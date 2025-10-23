import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from integrators.rk4 import rk4
import numpy as np

def launch_gui():
    root = tk.Tk()
    root.title("Lorenz Attractor Simulation")

    fig = plt.Figure(figsize=(6,6))
    ax = fig.add_subplot(projection='3d')

    def run_sim():
        q0 = np.array([float(entry_x.get()), float(entry_y.get()), float(entry_z.get())])
        data = rk4(q0, dt=0.01, steps=10000)
        ax.clear()
        ax.plot(*data.T, color='blue', lw=0.5)
        ax.set_title("Lorenz Trajectory")
        canvas.draw()

    tk.Label(root, text="Initial X:").grid(row=0, column=0)
    tk.Label(root, text="Initial Y:").grid(row=1, column=0)
    tk.Label(root, text="Initial Z:").grid(row=2, column=0)
    entry_x, entry_y, entry_z = tk.Entry(root), tk.Entry(root), tk.Entry(root)
    entry_x.insert(0, "0.0"); entry_y.insert(0, "1.0"); entry_z.insert(0, "1.05")
    entry_x.grid(row=0, column=1); entry_y.grid(row=1, column=1); entry_z.grid(row=2, column=1)

    tk.Button(root, text="Run Simulation", command=run_sim).grid(row=3, column=0, columnspan=2)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=4, column=0, columnspan=2)

    root.mainloop()
