import numpy as np

def simulate_gbm(S0, r, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    for t in range(n_steps):
        print(t)
        Z = np.random.randn(n_paths)
        S[:, t+1] = S[:, t] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    return S
