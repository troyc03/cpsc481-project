import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
import pandas as pd
from scipy.integrate import odeint
import pysindy as ps

def lorenz_true(state, t, s=10, r=28, b=2.667):
    x, y, z = state
    dx = s * (y - x)
    dy = r * x - y - x * z
    dz = x * y - b * z
    return [dx, dy, dz]

def lorenz_trained(state, t, s=10, r=28, b=2.667):
    # This will be implemented after fitting the model
    pass

def generate_training_data(init_state, t):
    return odeint(lorenz_true, init_state, t)

def fit_sindy_model(X_train, t, threshold=0.01):
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=2)
    )
    model.fit(X_train, t=t)
    return model

def simulate_trajectory(func, init_state, t):
    return odeint(func, init_state, t)

def compute_rmse(true_traj, trained_traj):
    return rmse(true_traj, trained_traj)

def plot_trajectories(true_traj, trained_traj, t):
    fig = plt.figure(figsize=(12, 5))

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(*true_traj.T, label='True Lorenz', color='blue', alpha=0.7)
    ax1.plot(*trained_traj.T, label='Trained Lorenz', color='red', alpha=0.7)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Lorenz Trajectories')
    ax1.legend()

    # RMSE over time
    rmse_vals = [rmse(true_traj[:i+1], trained_traj[:i+1]) for i in range(len(t))]
    ax2 = fig.add_subplot(122)
    ax2.plot(t, rmse_vals, color='green')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE over Time')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    dt = 0.01
    num_steps = 1000
    t = np.arange(0, num_steps * dt, dt)
    init_state = [0., 1., 1.05]

    # Generate training data
    X_train = generate_training_data(init_state, t)

    # Fit SINDy model
    model = fit_sindy_model(X_train, t)

    # Implement lorenz_trained using the model
    def lorenz_trained(state, t, s=10, r=28, b=2.667):
        return model.predict(state[np.newaxis, :])[0]

    # Simulate trajectories
    true_traj = simulate_trajectory(lorenz_true, init_state, t)
    trained_traj = simulate_trajectory(lorenz_trained, init_state, t)

    # Compute overall RMSE
    overall_rmse = compute_rmse(true_traj, trained_traj)
    print(f"Overall RMSE: {overall_rmse}")

    # Plot
    plot_trajectories(true_traj, trained_traj, t)

    # Save results to CSV
    df = pd.DataFrame({
        'time': t,
        'true_x': true_traj[:, 0],
        'true_y': true_traj[:, 1],
        'true_z': true_traj[:, 2],
        'trained_x': trained_traj[:, 0],
        'trained_y': trained_traj[:, 1],
        'trained_z': trained_traj[:, 2]
    })
    df.to_csv('lorenz_evaluation_results.csv', index=False)
    print("Results saved to lorenz_evaluation_results.csv")





