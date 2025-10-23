import numpy as np
from systems.lorenz_attractor import lorenz, jacobian

def verlet_integrate(q0, v0, a0, dt=0.01, num_steps=10000):
    
    q = np.zeros((num_steps + 1, 3))
    v = np.zeros((num_steps + 1, 3))
    a = np.zeros((num_steps + 1, 3))

    q[0] = q0
    v[0] = lorenz(q[0])
    a[0] = jacobian(q0).dot(v[0])

    for i in range(num_steps):
        v_half = v[i] + 0.5 * dt * a[i]
        q[i + 1] = q[i] + dt * v_half
        a[i + 1] = jacobian(q[i + 1]).dot(v_half)
        v[i + 1] = v_half + dt * a[i + 1]

    return q


