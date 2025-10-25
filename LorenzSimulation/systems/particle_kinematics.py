import numpy as np
from scipy.integrate import solve_ivp
from lorenz_attractor import lorenz
from lorenz_attractor import jacobian

def eom(t, state, s=10.0, r=28.0, b=8.0/3.0):
    """Coupled first-order equations for position and velocity in the Lorenz flow."""
    q = state[:3]
    v = state[3:]
    dqdt = v
    dvdt = jacobian(q, s=s, r=r, b=b) @ v
    return np.hstack([dqdt, dvdt])

def particle_kinematics(q0, dt=0.01, num_steps=10000, s=10.0, r=28.0, b=8.0/3.0):
    """
    Simulate particle motion through Lorenz flow using adaptive integration.
    Returns position q(t), velocity v(t), and acceleration a(t).
    """
    t_span = (0, dt * num_steps)
    t_eval = np.linspace(*t_span, num_steps + 1)
    v0 = lorenz(q0, t=0, s=s, r=r, b=b)
    y0 = np.hstack([q0, v0])

    sol = solve_ivp(eom, t_span, y0, t_eval=t_eval, args=(s, r, b),
                    rtol=1e-6, atol=1e-9, method='RK45')

    q = sol.y[:3].T
    v = sol.y[3:].T
    a = np.zeros_like(v)

    for i in range(len(q)):
        a[i] = jacobian(q[i], s=s, r=r, b=b) @ v[i]

    return q, v, a

# Example usage
q0 = np.array([0.0, 1.0, 1.05])
q, v, a = particle_kinematics(q0, dt=0.01, num_steps=5000)

print("Final position:", q[-1], " m")
print("Velocity magnitude:", np.linalg.norm(v[-1]), " ms^-1")
print("Acceleration magnitude:", np.linalg.norm(a[-1]), " ms^-2")