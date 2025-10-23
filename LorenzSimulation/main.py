import argparse
import numpy as np
from integrators.euler import euler_integrate
# from integrators.rk4 import rk4_integrate
from integrators.verlet import verlet_integrate
from visualization.animate_lorenz import animate_trajectory
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Lorenz System Simulation")
    parser.add_argument("--method", choices=["euler", "rk4", "verlet"], default="euler",
                        help="Integration method to use.")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()

    q0 = np.array([0., 1., 1.05])
    print(f"Running Lorenz system using {args.method.upper()} integration...")

    if args.method == "euler":
        xyzs = euler_integrate(q0, dt=args.dt, num_steps=args.steps)
    elif args.method == "rk4":
        pass
        # xyzs = rk4_integrate(q0, dt=args.dt, num_steps=args.steps)
    elif args.method == "verlet":
        xyzs = verlet_integrate(q0, dt=args.dt, num_steps=args.steps)

if __name__ == "__main__":
    main()
