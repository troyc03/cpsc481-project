# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 11:28:45 2025

@author: WINDOWS
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lorenz_extended(t, X, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    Extended Lorenz system that includes the variational equations.
    """
    # Unpack main variables
    x, y, z = X[0], X[1], X[2]

    # Lorenz system
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    # Jacobian
    Jac = np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]
    ])

    # Variational matrix (reshape the remaining 9 elements into 3x3)
    Y = np.array(X[3:]).reshape((3, 3))

    # Variational equation
    dYdt = Jac @ Y

    # Flatten the matrix back
    return np.concatenate(([dxdt, dydt, dzdt], dYdt.flatten()))


def lorenz_lyapunov(tstart=0, stept=0.1, tend=1000, ystart=np.array([1.0, 1.0, 1.0])):
    """
    Computes Lyapunov exponents for the Lorenz system.
    """
    n = 3
    nits = int((tend - tstart) / stept)

    # Initial extended state (3 + 9 = 12)
    y = np.zeros(n + n * n)
    y[:n] = ystart
    y[n:] = np.eye(n).flatten()  # Identity for variational matrix

    cum = np.zeros(n)
    Texp = []
    Lexp = []

    t = tstart

    for iterlya in range(nits):
        # Integrate one step
        sol = solve_ivp(lorenz_extended, [t, t + stept], y, t_eval=[t + stept])
        y = sol.y[:, -1]
        t += stept

        # Reorganize variational matrix
        Y = np.array(y[n:]).reshape((n, n)).T

        # --- Gram-Schmidt reorthonormalization ---
        Q = np.zeros_like(Y)
        znorm = np.zeros(n)

        for j in range(n):
            v = Y[:, j]
            for k in range(j):
                v -= np.dot(Q[:, k], Y[:, j]) * Q[:, k]
            znorm[j] = np.linalg.norm(v)
            Q[:, j] = v / znorm[j]

        # Update cumulative log stretch
        cum += np.log(znorm)

        # Normalize exponents
        lp = cum / (t - tstart)

        # Store data
        Texp.append(t)
        Lexp.append(lp.copy())

        # Replace Y with Q for next integration step
        y[n:] = Q.T.flatten()

    Texp = np.array(Texp)
    Lexp = np.array(Lexp)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(Texp, Lexp[:, 0], 'b', label=r'$\lambda_1$')
    plt.plot(Texp, Lexp[:, 1], 'g', label=r'$\lambda_2$')
    plt.plot(Texp, Lexp[:, 2], 'r', label=r'$\lambda_3$')
    plt.legend()
    plt.title('Dynamics of Lyapunov Exponents (Lorenz System)')
    plt.xlabel('Time')
    plt.ylabel('Lyapunov Exponents')
    plt.grid(True)
    plt.show()

    print("Final Lyapunov Exponents:")
    print(f"λ[1] = {Lexp[-1,0]:.4f}")
    print(f"λ[2] = {Lexp[-1,1]:.4f}")
    print(f"λ[3] = {Lexp[-1,2]:.4f}")

    return Texp, Lexp


if __name__ == "__main__":
    lorenz_lyapunov(tend=1000)
