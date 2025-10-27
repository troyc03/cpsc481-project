import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
import pandas as pd

def lorenz_true(state, t, s=10, r=28, b=2.667):
    x, y, z = state
    dx = s * (y - x)
    dy = r * x - r * z - y
    dz = x * y - b * z
    return [dx, dy, dz]

def lorenz_trained(state, t, s=10, r=28, b=2.667):
    pass





