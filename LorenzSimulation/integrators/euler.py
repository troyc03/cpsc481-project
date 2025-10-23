import numpy as np
from systems.lorenz_attractor import lorenz

def euler_integrate(q0, dt=0.01, num_steps=10000):
    xyzs = np.empty((num_steps + 1, 3))
    xyzs[0] = q0
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
    return xyzs

