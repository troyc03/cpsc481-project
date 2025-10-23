import matplotlib.pyplot as plt
import numpy as np

# Define the Lorenz system
def lorenz(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])

# Define the Jacobian matrix 
def jacobian(q, *, s=10.0, r=28.0, b=8.0/3.0):
    x, y, z = q
    return np.array([
        [-s,s,0],
        [r-z,-1,-x],
        [y,x,b]
        ])
