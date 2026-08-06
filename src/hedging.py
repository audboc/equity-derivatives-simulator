from math import exp

import numpy as np

from black_scholes import black_scholes_call
from greeks import call_delta
from simulation import simulate_gbm_path


def delta_hedge_call(S0, K, T, r, sigma_realized, sigma_implied=None, q=0, n_steps=252,
                      rebalance_every=1, rebalance_threshold=None, seed=None, price_path=None):
    """
    Simulates selling a European call, hedged via delta hedging, over a path
    of the underlying (simulated by default, or a real path if price_path is given).

    sigma_realized : vol used to simulate the underlying's path (the "real" world).
                      Ignored for path generation if price_path is given
                      (but still useful as a reference, e.g. realized vol measured on that path).
    sigma_implied : vol used to price the option and compute the delta (the price at which
                    the option was sold). Defaults to sigma_realized if not specified.
    rebalance_every : number of steps between two rebalances (1 = every step, 5 = roughly
                       weekly if n_steps=252). Ignored if rebalance_threshold is specified.
    rebalance_threshold : if specified (e.g. 0.05), rebalancing happens as soon as the
                           theoretical delta drifts from the held delta by more than this
                           threshold, instead of on a fixed interval. Overrides rebalance_every
                           when used.
    price_path : an already-known path (e.g. real prices fetched via market_data.py) to
                 use instead of a GBM simulation. Must contain exactly n_steps+1
                 points, with price_path[0] == S0. If None (default), the path is
                 simulated with simulate_gbm_path as before.

    Logic (from the option seller's point of view):
    - collect the premium by selling the call
    - buy delta shares to hedge, financed by the remaining cash
    - at each step: cash accrues at the risk-free rate, the theoretical delta is recomputed;
      rebalancing (buying/selling shares) only happens if the chosen trigger is reached
    - at expiry: rebalance one last time to liquidate, then pay the payoff

    Returns a dictionary with the path, the delta history (target and held),
    the number of rebalances, and the final P&L of the hedging strategy.
    """

    if sigma_implied is None:
        sigma_implied = sigma_realized

    dt = T / n_steps

    if price_path is not None:
        path = np.asarray(price_path, dtype=float)
        if len(path) != n_steps + 1:
            raise ValueError(f"price_path must have n_steps+1={n_steps + 1} points, got {len(path)}.")
        if not np.isclose(path[0], S0):
            raise ValueError(f"price_path[0]={path[0]} must equal S0={S0}.")
    else:
        path = simulate_gbm_path(S0, mu=r, sigma=sigma_realized, T=T, n_steps=n_steps, q=q, seed=seed)

    option_price_0 = black_scholes_call(S0, K, T, r, sigma_implied, q)
    delta_0 = call_delta(S0, K, T, r, sigma_implied, q)

    # Premium collected minus the cost of buying the hedging shares
    cash = option_price_0 - delta_0 * S0
    shares_held = delta_0

    deltas_target = [delta_0]
    deltas_held = [delta_0]
    cash_history = [cash]
    portfolio_values = [cash + shares_held * S0]
    option_values = [option_price_0]
    n_rebalances = 1
    total_shares_traded = abs(delta_0)

    for t in range(1, n_steps + 1):
        time_remaining = T - t * dt
        S_t = path[t]

        cash = cash * exp(r * dt)

        if time_remaining > 1e-8:
            target_delta = call_delta(S_t, K, time_remaining, r, sigma_implied, q)
            option_value = black_scholes_call(S_t, K, time_remaining, r, sigma_implied, q)
        else:
            target_delta = 1.0 if S_t > K else 0.0
            option_value = max(S_t - K, 0)

        # Rebalance either by threshold or on a fixed interval, and always on the last step (liquidation)
        if rebalance_threshold is not None:
            should_rebalance = abs(target_delta - shares_held) >= rebalance_threshold
        else:
            should_rebalance = t % rebalance_every == 0

        if should_rebalance or t == n_steps:
            shares_to_trade = target_delta - shares_held
            cash -= shares_to_trade * S_t
            shares_held = target_delta
            n_rebalances += 1
            total_shares_traded += abs(shares_to_trade)

        deltas_target.append(target_delta)
        deltas_held.append(shares_held)
        cash_history.append(cash)
        portfolio_values.append(cash + shares_held * S_t)
        option_values.append(option_value)

    S_T = path[-1]
    option_payoff = max(S_T - K, 0)
    portfolio_value = cash + shares_held * S_T
    final_pnl = portfolio_value - option_payoff

    # "Mark-to-market" P&L at each point in time: hedge portfolio value
    # minus the theoretical value of the option that will eventually need to be paid
    mark_to_market_pnl = np.array(portfolio_values) - np.array(option_values)

    return {
        "path": path,
        "deltas_target": np.array(deltas_target),
        "deltas_held": np.array(deltas_held),
        "cash_history": np.array(cash_history),
        "portfolio_values": np.array(portfolio_values),
        "option_values": np.array(option_values),
        "mark_to_market_pnl": mark_to_market_pnl,
        "option_price_0": option_price_0,
        "option_payoff": option_payoff,
        "final_pnl": final_pnl,
        "n_rebalances": n_rebalances,
        "total_shares_traded": total_shares_traded,
    }


if __name__ == "__main__":

    # A simple simulation: realized vol = implied vol
    result = delta_hedge_call(S0=100, K=100, T=1, r=0.05, sigma_realized=0.20, seed=42)

    print(f"Premium collected at sale: {result['option_price_0']:.4f}")
    print(f"Payoff paid at expiry    : {result['option_payoff']:.4f}")
    print(f"Final hedge P&L          : {result['final_pnl']:.4f}")

    # Statistical check: if realized vol = implied vol, the average P&L
    # over many paths should be close to 0 (the hedge replicates the option)
    n_paths = 2000
    pnls_matched = [delta_hedge_call(S0=100, K=100, T=1, r=0.05, sigma_realized=0.20)["final_pnl"]
                    for _ in range(n_paths)]

    print(f"\nOver {n_paths} paths, realized vol = implied vol (20%):")
    print(f"Mean P&L: {np.mean(pnls_matched):.4f}")
    print(f"P&L std dev: {np.std(pnls_matched):.4f}")

    # If realized vol is higher than implied vol (option underpriced
    # at sale), the hedged seller should on average LOSE money
    pnls_high_vol = [delta_hedge_call(S0=100, K=100, T=1, r=0.05,
                                       sigma_realized=0.35, sigma_implied=0.20)["final_pnl"]
                     for _ in range(n_paths)]

    print(f"\nOver {n_paths} paths, realized vol (35%) > implied vol (20%):")
    print(f"Mean P&L: {np.mean(pnls_high_vol):.4f}")

    # Comparison: daily rebalancing (every day) vs weekly (every 5 days)
    print(f"\nDaily vs weekly comparison, over {n_paths} paths (realized vol = implied = 20%):")

    for label, rebalance_every in [("Daily (every day)", 1), ("Weekly (every 5 days)", 5)]:
        results = [delta_hedge_call(S0=100, K=100, T=1, r=0.05, sigma_realized=0.20, rebalance_every=rebalance_every)
                   for _ in range(n_paths)]

        pnls = [res["final_pnl"] for res in results]
        n_rebalances = [res["n_rebalances"] for res in results]

        print(f"\n{label}:")
        print(f"  Mean P&L: {np.mean(pnls):.4f}")
        print(f"  P&L std dev: {np.std(pnls):.4f}")
        print(f"  Average number of rebalances: {np.mean(n_rebalances):.1f}")

    # Third mode: rebalancing triggered by a delta drift threshold,
    # instead of a fixed interval
    print(f"\nComparison with threshold-based rebalancing, over {n_paths} paths:")

    for label, threshold in [("5% threshold", 0.05), ("10% threshold", 0.10)]:
        results = [delta_hedge_call(S0=100, K=100, T=1, r=0.05, sigma_realized=0.20, rebalance_threshold=threshold)
                   for _ in range(n_paths)]

        pnls = [res["final_pnl"] for res in results]
        n_rebalances = [res["n_rebalances"] for res in results]

        print(f"\n{label}:")
        print(f"  Mean P&L: {np.mean(pnls):.4f}")
        print(f"  P&L std dev: {np.std(pnls):.4f}")
        print(f"  Average number of rebalances: {np.mean(n_rebalances):.1f}")
