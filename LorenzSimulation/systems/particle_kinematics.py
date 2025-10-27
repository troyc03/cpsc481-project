import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from lorenz_attractor import lorenz, jacobian

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
    Returns position q(t), velocity v(t), acceleration a(t), and time t.
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

    return q, v, a, sol.t

# Initial conditions & perturbation
q0 = np.array([0.0, 1.0, 1.05])
delta = np.array([1e-6, 0, 0])  # Small perturbation in x

# Run both simulations
q1, v1, a1, t = particle_kinematics(q0)
q2, v2, a2, _ = particle_kinematics(q0 + delta)

# Plot: X component over time
plt.figure(figsize=(10, 5))
plt.plot(t, q1[:, 0], label='Original $x(t)$', color='royalblue')
plt.plot(t, q2[:, 0], label='Perturbed $x(t)$', color='darkorange', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('$x(t)$')
plt.title('Lorenz Attractor – Sensitivity to Initial Conditions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Divergence of trajectories
diff = np.linalg.norm(q1 - q2, axis=1)

plt.figure(figsize=(8, 5))
plt.semilogy(t, diff, color='crimson')
plt.xlabel('Time (s)')
plt.ylabel(r'$||\Delta \mathbf{q}(t)||$')
plt.title('Exponential Divergence of Nearby Trajectories')
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

print("Final position:", q1[-1], " m") 
print("Velocity magnitude:", np.linalg.norm(v1[-1]), " ms^-1") 
print("Acceleration magnitude:", np.linalg.norm(a1[-1]), " ms^-2")