import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import pandas as pd
from scipy.integrate import odeint
import pysindy as ps

# -----------------------
# True Lorenz System
# -----------------------
def lorenz_true(state, t, s=10, r=28, b=2.667):
    x, y, z = state
    dx = s * (y - x)
    dy = r * x - y - x * z
    dz = x * y - b * z
    return [dx, dy, dz]

# Generate training data
def generate_training_data(init_state, t):
    return odeint(lorenz_true, init_state, t)

# Train SINDy model
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

# -----------------------
# Preprocessing
# -----------------------
def preprocess_csv(file_path, seq_len=10):
    df = pd.read_csv(file_path)

    # Corrected: select_dtypes not select_types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    state_cols = ['true_x', 'true_y', 'true_z']
    states = df[state_cols].values.astype(np.float32)

    X, y = [], []
    for i in range(len(states) - seq_len):
        X.append(states[i:i+seq_len])
        y.append(states[i+seq_len])
    X, y = np.stack(X), np.stack(y)  # Vectorize X and y

    # Corrected: test_size should be a fraction (e.g., 0.2), not 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test, scaler

# -----------------------
# Plotting
# -----------------------
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
    ax2.plot(t, rmse_vals, color='green', marker='o', mfc='black')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE over Time')

    plt.tight_layout()
    plt.show()

# -----------------------
# Main pipeline
# -----------------------
if __name__ == "__main__":
    np.random.seed(42)

    # 1. Generate Lorenz data
    dt = 0.01
    num_steps = 1000
    t = np.arange(0, num_steps * dt, dt)
    init_state = [0., 1., 1.05]

    X_train_data = generate_training_data(init_state, t)

    # 2. Fit SINDy model
    sindy_model = fit_sindy_model(X_train_data, t)

    # 3. Define trained dynamics using SINDy (SINDy models derivatives -> OK for odeint)
    def lorenz_trained(state, t):
        return sindy_model.predict(state[np.newaxis, :])[0]

    # 4. Simulate true and SINDy trajectories
    true_traj = simulate_trajectory(lorenz_true, init_state, t)
    trained_traj = simulate_trajectory(lorenz_trained, init_state, t)

    # 5. Compute RMSE (SINDy) and save results CSV
    overall_rmse = compute_rmse(true_traj, trained_traj)
    print(f"Overall RMSE (SINDy): {overall_rmse:.6f}")

    df = pd.DataFrame({
        'time': t,
        'true_x': true_traj[:, 0],
        'true_y': true_traj[:, 1],
        'true_z': true_traj[:, 2],
        'trained_x': trained_traj[:, 0],
        'trained_y': trained_traj[:, 1],
        'trained_z': trained_traj[:, 2]
    })
    df.to_csv("lorenz_evaluation_results.csv", index=False)
    print('\nFirst ten datapoints:\n', df.head(10), '\n')
    print('\nNull values:\n', df.isnull().sum(), '\n')
    print("Saved results to lorenz_evaluation_results.csv")

    # 6. Plot SINDy results (optional)
    plot_trajectories(true_traj, trained_traj, t)

    # -------------------------
    # Learned-attractor (MLP) pipeline
    # -------------------------
    # 7. Preprocess once for learned attractor
    X_train, X_test, y_train, y_test, scaler = preprocess_csv("lorenz_evaluation_results.csv")

    # 8. Flatten time-series windows for scikit-learn regressors
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # (n_samples, seq_len * features)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)

    # 9. Train an MLPRegressor (multi-output)
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='tanh',
        solver='adam',
        max_iter=500,
        random_state=42,
        verbose=False
    )
    mlp.fit(X_train_flat, y_train)   # y_train shape: (n_samples, 3)

    # 10. Predict on test set
    y_pred = mlp.predict(X_test_flat)

    # 11. Compute and report RMSE (values are in scaled space)
    learned_rmse = rmse(y_test, y_pred)
    print(f"Overall RMSE (Learned Attractor, scaled units): {learned_rmse:.6f}")

    # 12. Visual comparisons (phase-space and x time-series)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(y_test[:,0], y_test[:,2], label="True (scaled)", alpha=0.7)
    plt.plot(y_pred[:,0], y_pred[:,2], label="Learned (scaled)", alpha=0.7)
    plt.xlabel("x (scaled)"); plt.ylabel("z (scaled)")
    plt.title("Phase Space Comparison (scaled)"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(y_test[:,0], label="True x (scaled)")
    plt.plot(y_pred[:,0], label="Pred x (scaled)", alpha=0.7)
    plt.title("Time Series Comparison (x, scaled)"); plt.legend()

    plt.tight_layout()
    plt.show()
