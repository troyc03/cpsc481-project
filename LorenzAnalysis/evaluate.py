import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy.integrate import odeint
import pysindy as ps
import os

# -----------------------
# Utility functions
# -----------------------
def rmse_from_array(true, pred):
    """Root mean squared error between two arrays of shape (T, d)."""
    return np.sqrt(mean_squared_error(true, pred))

# -----------------------
# True Lorenz System
# -----------------------
def lorenz_true(state, t, s=10.0, r=28.0, b=8.0/3.0):
    x, y, z = state
    dx = s * (y - x)
    dy = r * x - y - x * z
    dz = x * y - b * z
    return [dx, dy, dz]

# -----------------------
# Data generation / SINDy
# -----------------------
def generate_training_data(init_state, t):
    """Integrate the true Lorenz ODE to produce states (T,3)."""
    return odeint(lorenz_true, init_state, t)

def fit_sindy_model(X_train, t, threshold=0.01, poly_degree=2):
    """Fit a PySINDy model to state data X_train with timestep vector t or scalar dt."""
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_degree)
    )
    model.fit(X_train, t=t)
    return model

def sindy_rhs_wrapper(sindy_model):
    """
    Return a function f(state, t) that computes derivatives using the fitted SINDy model.
    Useful for embedding into odeint if you want to simulate using the fitted right-hand side.
    """
    def f(state, t):
        deriv = sindy_model.predict(state.reshape(1, -1))[0]
        return deriv
    return f

def simulate_with_callable(rhs_func, init_state, t):
    """Simulate an ODE given rhs_func(state,t) via odeint."""
    return odeint(rhs_func, init_state, t)

# -----------------------
# CSV preprocessing for learned-attractor
# -----------------------
def preprocess_csv(file_path, seq_len=10, test_fraction=0.2):
    """
    Read CSV with columns ['true_x','true_y','true_z','sindy_x','sindy_y','sindy_z',...]
    Build sliding windows X of shape (N, seq_len, 3) using true_* columns and targets y of shape (N,3)
    Returns: X_train, X_test, y_train, y_test, raw_time (for reference)
    NOTE: this function does NOT scale the data. Scaling is handled in main() to avoid leakage.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessing CSV not found: {file_path}")

    df = pd.read_csv(file_path)

    required_cols = ['true_x', 'true_y', 'true_z']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' in CSV for preprocessing.")

    states = df[required_cols].values.astype(np.float32)
    times = df['time'].values if 'time' in df.columns else np.arange(len(states)).astype(np.float32)

    X, y = [], []
    for i in range(len(states) - seq_len):
        X.append(states[i:i+seq_len])
        y.append(states[i+seq_len])
    X = np.stack(X)  # (N, seq_len, 3)
    y = np.stack(y)  # (N, 3)

    # Train/test split (preserve time ordering: shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False)
    # Also compute corresponding time windows for test optionally (t for the y_test indices)
    # We return raw times for convenience (aligned to original df); user can slice as needed.
    return X_train, X_test, y_train, y_test, times

# -----------------------
# Plot helpers
# -----------------------
def plot_trajectories(true_traj, trained_traj, t, title_prefix=""):
    """
    Plot 3D true vs trained trajectories and RMSE over time.
    true_traj, trained_traj: arrays (T,3)
    t: times (T,)
    """
    fig = plt.figure(figsize=(12, 5))

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(true_traj[:,0], true_traj[:,1], true_traj[:,2], label='True Lorenz', alpha=0.7)
    ax1.plot(trained_traj[:,0], trained_traj[:,1], trained_traj[:,2], label='Trained (SINDy)', alpha=0.7)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title(f'{title_prefix} Trajectories')
    ax1.legend()

    # RMSE over time (cumulative RMSE up to time i)
    rmse_vals = [rmse_from_array(true_traj[:i+1], trained_traj[:i+1]) for i in range(len(t))]
    ax2 = fig.add_subplot(122)
    ax2.plot(t, rmse_vals)
    ax2.set_xlabel('Time'); ax2.set_ylabel('RMSE'); ax2.set_title(f'{title_prefix} Cumulative RMSE')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# -----------------------
# Main pipeline
# -----------------------
def main():
    np.random.seed(42)

    # 1. Generate Lorenz data
    dt = 0.01
    num_steps = 1000
    t = np.arange(0, num_steps * dt, dt)
    init_state = [0.0, 1.0, 1.05]

    print("Generating true Lorenz trajectory...")
    X_train_data = generate_training_data(init_state, t)  # shape (T,3)

    # 2. Fit SINDy model to the true states
    print("Fitting SINDy model...")
    sindy_model = fit_sindy_model(X_train_data, t, threshold=0.05, poly_degree=2)

    print("\n--- Discovered SINDy model ---")
    try:
        sindy_model.print()
    except Exception:
        print(str(sindy_model))

    # 3. Simulate SINDy dynamics
    print("Simulating SINDy prediction using model.simulate() ...")
    try:
        x_sim = sindy_model.simulate(init_state, t)
    except Exception as e:
        print("model.simulate failed, falling back to odeint with SINDy-predicted derivatives:", e)
        rhs = sindy_rhs_wrapper(sindy_model)
        x_sim = simulate_with_callable(rhs, init_state, t)

    # 4. Compute RMSE between true and SINDy-predicted trajectories
    overall_rmse = rmse_from_array(X_train_data, x_sim)
    print(f"\nOverall RMSE (SINDy vs True): {overall_rmse:.6f}")

    # 5. Save results to CSV for learned-attractor pipeline
    results_df = pd.DataFrame({
        'time': t,
        'true_x': X_train_data[:,0],
        'true_y': X_train_data[:,1],
        'true_z': X_train_data[:,2],
        'sindy_x': x_sim[:,0],
        'sindy_y': x_sim[:,1],
        'sindy_z': x_sim[:,2]
    })
    csv_path = "lorenz_evaluation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved SINDy evaluation results to {csv_path}")

    # 6. Plot SINDy vs True
    plot_trajectories(X_train_data, x_sim, t, title_prefix="SINDy")

    # -------------------------
    # Learned-attractor (SGD) pipeline
    # -------------------------
    print("\nStarting learned-attractor (SGD) pipeline...")

    seq_len = 10  # window length in timesteps
    X_train, X_test, y_train, y_test, raw_times = preprocess_csv(csv_path, seq_len=seq_len, test_fraction=0.2)

    # Flatten windows for sklearn regressors (each sample: seq_len * 3 features)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)

    # SCALE: Fit scalers only on training data (no leakage)
    feat_scaler = StandardScaler()
    X_train_flat_scaled = feat_scaler.fit_transform(X_train_flat)
    X_test_flat_scaled = feat_scaler.transform(X_test_flat)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    # we will transform y_test for evaluation in the scaled domain if needed, but final RMSE computed in original units
    y_test_scaled = target_scaler.transform(y_test)

    # Multi-output SGD regressor (wrap SGDRegressor)
    base_sgd = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=1e-4,
        learning_rate="invscaling",  # more stable than "optimal" for some versions
        eta0=1e-2,
        power_t=0.25,
        max_iter=5000,
        tol=1e-4,
        random_state=42,
        verbose=0
    )
    sgd_model = MultiOutputRegressor(base_sgd)
    print("Training SGD multi-output regressor on sliding-window features...")
    sgd_model.fit(X_train_flat_scaled, y_train_scaled)

    # Predict on test set (returns scaled targets)
    y_pred_scaled = sgd_model.predict(X_test_flat_scaled)

    # Check for numerical issues
    if np.any(np.isnan(y_pred_scaled)) or np.any(np.isinf(y_pred_scaled)):
        raise RuntimeError("Predictions contain NaN or Inf values. Try reducing learning rate or regularization.")

    # Inverse transform predictions back to original (unscaled) units for meaningful RMSE
    y_pred = target_scaler.inverse_transform(y_pred_scaled)

    # Evaluate learned model (RMSE in original units)
    learned_rmse = rmse_from_array(y_test, y_pred)
    print(f"Overall RMSE (SGD Learned Attractor, original units): {learned_rmse:.6f}")

    # Visual comparisons: true vs predicted on test set (phase space and x-time-series)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(y_test[:,0], y_test[:,2], label="True")
    plt.plot(y_pred[:,0], y_pred[:,2], label="SGD Learned", alpha=0.7)
    plt.xlabel("x"); plt.ylabel("z")
    plt.title("Phase Space Comparison"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(y_test[:,0], label="True x")
    plt.plot(y_pred[:,0], label="Pred x", alpha=0.7)
    plt.title("Time Series Comparison (x)"); plt.legend()

    plt.tight_layout()
    plt.show()

    # Save learned predictions (original units)
    learned_df = pd.DataFrame({
        'true_x': y_test[:,0],
        'true_y': y_test[:,1],
        'true_z': y_test[:,2],
        'pred_x': y_pred[:,0],
        'pred_y': y_pred[:,1],
        'pred_z': y_pred[:,2]
    })
    learned_csv = "learned_attractor_sgd_results.csv"
    learned_df.to_csv(learned_csv, index=False)
    print(f"Saved learned-attractor results to {learned_csv}")

if __name__ == "__main__":
    main()
