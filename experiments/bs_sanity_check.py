import numpy as np
import matplotlib.pyplot as plt

from models.black_scholes import bs_call_price
from pricing.monte_carlo import simulate_gbm
from hedging.delta_hedge import delta_hedge_pnl

# ----------------------------
# Parameters
# ----------------------------
S0 = 100
K = 100
T = 1.0
r = 0.05
sigma = 0.2

n_steps = 252
n_paths = 5000

# ----------------------------
# Simulate BS World
# ----------------------------
paths = simulate_gbm(
    S0=S0,
    r=r,
    sigma=sigma,
    T=T,
    n_steps=n_steps,
    n_paths=n_paths
)

# ----------------------------
# Pricing sanity check
# ----------------------------
mc_price = np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0).mean()
bs_price = bs_call_price(S0, K, T, r, sigma)

print(f"BS Price: {bs_price:.4f}")
print(f"MC Price: {mc_price:.4f}")

# ----------------------------
# Delta hedging PnL
# ----------------------------
pnl = delta_hedge_pnl(paths, K, T, r, sigma)

print("Mean PnL:", pnl.mean())
print("Std PnL:", pnl.std())

# ----------------------------
# Plot
# ----------------------------
plt.hist(pnl, bins=50)
plt.title("Delta Hedging PnL (BS World)")
plt.xlabel("PnL")
plt.ylabel("Frequency")
plt.show()
