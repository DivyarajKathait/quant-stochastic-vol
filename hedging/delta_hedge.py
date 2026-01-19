import numpy as np
from models.black_scholes import bs_delta, bs_call_price

def delta_hedge_pnl(S_paths, K, T, r, sigma):
    n_paths, n_steps = S_paths.shape
    dt = T / (n_steps - 1)

    pnl = np.zeros(n_paths)

    for i in range(n_paths):
        print(i)
        cash = bs_call_price(S_paths[i, 0], K, T, r, sigma)
        delta_prev = 0.0

        for t in range(n_steps - 1):
            tau = T - t * dt
            delta = bs_delta(S_paths[i, t], K, tau, r, sigma)

            cash -= (delta - delta_prev) * S_paths[i, t]
            cash *= np.exp(r * dt)

            delta_prev = delta

        pnl[i] = cash + delta_prev * S_paths[i, -1] - max(S_paths[i, -1] - K, 0)

    return pnl
