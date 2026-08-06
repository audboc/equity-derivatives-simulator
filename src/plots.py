# %%
import numpy as np
import matplotlib.pyplot as plt

from black_scholes import black_scholes_call, black_scholes_put
from greeks import call_delta, put_delta, gamma, vega, call_theta, put_theta, call_rho, put_rho
from implied_volatility import implied_volatility_call
from hedging import delta_hedge_call
from market_data import fetch_price_history, compute_log_returns, compute_realized_volatility, \
    compute_rolling_realized_volatility
from dispersion import analyze_dispersion


def plot_price_vs_spot(K=100, T=1, r=0.05, q=0, save_path="../plots/price_vs_spot.png", show=True):
    """
    Plots the call and put price against spot, for several
    volatility levels, with intrinsic value as a reference.
    """

    spot_range = np.linspace(50, 150, 200)
    vols = [0.10, 0.20, 0.40]
    vol_colors = ["#9ecae1", "#4292c6", "#08519c"]  # sequential ramp, light -> dark

    call_intrinsic = np.maximum(spot_range - K, 0)
    put_intrinsic = np.maximum(K - spot_range, 0)

    fig, (ax_call, ax_put) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for sigma, color in zip(vols, vol_colors):
        call_prices = [black_scholes_call(S, K, T, r, sigma, q) for S in spot_range]
        put_prices = [black_scholes_put(S, K, T, r, sigma, q) for S in spot_range]

        ax_call.plot(spot_range, call_prices, color=color, linewidth=2, label=f"σ = {sigma:.0%}")
        ax_put.plot(spot_range, put_prices, color=color, linewidth=2, label=f"σ = {sigma:.0%}")

    ax_call.plot(spot_range, call_intrinsic, color="#555555", linewidth=1.5,
                 linestyle="--", alpha=0.7, label="Intrinsic value")
    ax_put.plot(spot_range, put_intrinsic, color="#555555", linewidth=1.5,
                linestyle="--", alpha=0.7, label="Intrinsic value")

    for ax, title in ((ax_call, "European call"), (ax_put, "European put")):
        ax.set_title(title)
        ax.set_xlabel("Spot")
        ax.axvline(K, color="#bbbbbb", linewidth=1, linestyle=":")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)

    ax_call.set_ylabel("Option price")

    fig.suptitle(f"Black-Scholes price vs spot  (K={K}, T={T}, r={r:.0%}, q={q:.0%})")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Chart saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def plot_delta_vs_spot(K=100, T=1, r=0.05, sigma=0.20, q=0, save_path="../plots/delta_vs_spot.png", show=True):
    """
    Plots the call and put delta against spot.
    """

    spot_range = np.linspace(50, 150, 200)

    call_deltas = [call_delta(S, K, T, r, sigma, q) for S in spot_range]
    put_deltas = [put_delta(S, K, T, r, sigma, q) for S in spot_range]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(spot_range, call_deltas, color="#0072B2", linewidth=2, label="Call delta")
    ax.plot(spot_range, put_deltas, color="#D55E00", linewidth=2, label="Put delta")

    ax.axhline(0, color="#bbbbbb", linewidth=1)
    ax.axhline(1, color="#bbbbbb", linewidth=1, linestyle=":")
    ax.axhline(-1, color="#bbbbbb", linewidth=1, linestyle=":")
    ax.axvline(K, color="#bbbbbb", linewidth=1, linestyle=":")

    ax.set_title(f"Delta vs spot  (K={K}, T={T}, r={r:.0%}, σ={sigma:.0%}, q={q:.0%})")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Delta")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Chart saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_gamma_vs_spot(K=100, r=0.05, sigma=0.20, q=0, save_path="../plots/gamma_vs_spot.png", show=True):
    """
    Plots gamma against spot, for several maturities.
    Identical for a call and a put (one curve per maturity).
    """

    spot_range = np.linspace(50, 150, 200)
    maturities = [0.25, 1, 2]
    maturity_colors = ["#9ecae1", "#4292c6", "#08519c"]  # sequential ramp, light -> dark

    fig, ax = plt.subplots(figsize=(7, 5))

    for T, color in zip(maturities, maturity_colors):
        gamma_values = [gamma(S, K, T, r, sigma, q) for S in spot_range]
        ax.plot(spot_range, gamma_values, color=color, linewidth=2, label=f"T = {T} year(s)")

    ax.axvline(K, color="#bbbbbb", linewidth=1, linestyle=":")

    ax.set_title(f"Gamma vs spot  (K={K}, r={r:.0%}, σ={sigma:.0%}, q={q:.0%})")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Gamma")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Chart saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_vega_vs_spot(K=100, r=0.05, sigma=0.20, q=0, save_path="../plots/vega_vs_spot.png", show=True):
    """
    Plots vega against spot, for several maturities.
    Identical for a call and a put (one curve per maturity).
    """

    spot_range = np.linspace(50, 150, 200)
    maturities = [0.25, 1, 2]
    maturity_colors = ["#9ecae1", "#4292c6", "#08519c"]  # sequential ramp, light -> dark

    fig, ax = plt.subplots(figsize=(7, 5))

    for T, color in zip(maturities, maturity_colors):
        vega_values = [vega(S, K, T, r, sigma, q) for S in spot_range]
        ax.plot(spot_range, vega_values, color=color, linewidth=2, label=f"T = {T} year(s)")

    ax.axvline(K, color="#bbbbbb", linewidth=1, linestyle=":")

    ax.set_title(f"Vega vs spot  (K={K}, r={r:.0%}, σ={sigma:.0%}, q={q:.0%})")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Vega")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Chart saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_theta_vs_spot(K=100, T=1, r=0.05, sigma=0.20, q=0, save_path="../plots/theta_vs_spot.png", show=True):
    """
    Plots the call and put theta against spot.
    """

    spot_range = np.linspace(40, 160, 200)

    call_thetas = [call_theta(S, K, T, r, sigma, q) for S in spot_range]
    put_thetas = [put_theta(S, K, T, r, sigma, q) for S in spot_range]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(spot_range, call_thetas, color="#0072B2", linewidth=2, label="Call theta")
    ax.plot(spot_range, put_thetas, color="#D55E00", linewidth=2, label="Put theta")

    ax.axhline(0, color="#555555", linewidth=1)
    ax.axvline(K, color="#bbbbbb", linewidth=1, linestyle=":")

    ax.set_title(f"Theta vs spot  (K={K}, T={T}, r={r:.0%}, σ={sigma:.0%}, q={q:.0%})")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Theta (annual)")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Chart saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_theta_dividend_effect(K=100, T=1, r=0.05, sigma=0.20,
                                save_path="../plots/theta_dividend_effect.png", show=True):
    """
    Special case: theta of a deep-ITM call against spot, for several
    dividend levels. Without a dividend, call theta always stays negative;
    a high dividend can push it positive when the call is deep ITM.
    """

    spot_range = np.linspace(60, 200, 200)
    dividends = [0.00, 0.10, 0.20]
    dividend_colors = ["#9ecae1", "#4292c6", "#08519c"]  # sequential ramp, light -> dark

    fig, ax = plt.subplots(figsize=(7, 5))

    for q, color in zip(dividends, dividend_colors):
        call_thetas = [call_theta(S, K, T, r, sigma, q) for S in spot_range]
        ax.plot(spot_range, call_thetas, color=color, linewidth=2, label=f"q = {q:.0%}")

    ax.axhline(0, color="#555555", linewidth=1)
    ax.axvline(K, color="#bbbbbb", linewidth=1, linestyle=":")

    ax.set_title(f"Deep-ITM call theta vs dividend  (K={K}, T={T}, r={r:.0%}, σ={sigma:.0%})")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Call theta (annual)")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Chart saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_rho_vs_spot(K=100, T=1, r=0.05, sigma=0.20, q=0, save_path="../plots/rho_vs_spot.png", show=True):
    """
    Plots the call and put rho against spot.
    """

    spot_range = np.linspace(50, 150, 200)

    call_rhos = [call_rho(S, K, T, r, sigma, q) for S in spot_range]
    put_rhos = [put_rho(S, K, T, r, sigma, q) for S in spot_range]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(spot_range, call_rhos, color="#0072B2", linewidth=2, label="Call rho")
    ax.plot(spot_range, put_rhos, color="#D55E00", linewidth=2, label="Put rho")

    ax.axhline(0, color="#555555", linewidth=1)
    ax.axvline(K, color="#bbbbbb", linewidth=1, linestyle=":")

    ax.set_title(f"Rho vs spot  (K={K}, T={T}, r={r:.0%}, σ={sigma:.0%}, q={q:.0%})")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Rho")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Chart saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_implied_vol_smile(S=100, T=1, r=0.05, q=0, base_vol=0.20, skew_slope=-0.15, convexity=0.25,
                            save_path="../plots/implied_vol_smile.png", show=True):
    """
    Builds a SYNTHETIC volatility smile (not real market data):
    call prices are built from a made-up vol curve (higher for low
    strikes, like the skew observed on equities, with a bit of
    convexity to also rise for deep OTM calls), then that curve is
    recovered strike by strike with implied_volatility_call. Used to
    verify that the inversion works correctly across a whole range of strikes.
    """

    strikes = np.linspace(70, 130, 40)
    moneyness = strikes / S

    true_vols = base_vol + skew_slope * (moneyness - 1) + convexity * (moneyness - 1) ** 2

    market_prices = [black_scholes_call(S, K, T, r, sigma, q) for K, sigma in zip(strikes, true_vols)]
    recovered_vols = [implied_volatility_call(price, S, K, T, r, q)
                       for price, K in zip(market_prices, strikes)]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(strikes, true_vols, color="#555555", linewidth=3, alpha=0.5, label="\"Market\" vol (synthetic)")
    ax.plot(strikes, recovered_vols, color="#0072B2", linewidth=1.5, linestyle="--",
            label="Recovered implied vol")

    ax.axvline(S, color="#bbbbbb", linewidth=1, linestyle=":")

    ax.set_title(f"Implied volatility smile (synthetic)  (S={S}, T={T}, r={r:.0%}, q={q:.0%})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Chart saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def _plot_hedge_panels(result, K, T, r, n_steps, sigma_label, rebalance_every, rebalance_threshold,
                        title_prefix, save_path, show):
    """
    Draws the 3 panels common to every delta hedging simulation (spot, delta,
    P&L), from a result already computed by delta_hedge_call. Factored out to be
    reused both for a simulated path and for a real market path
    (the upstream computation changes, the drawing is identical).
    """

    time_axis = np.linspace(0, T, n_steps + 1)

    fig, (ax_spot, ax_delta, ax_pnl) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    ax_spot.plot(time_axis, result["path"], color="#0072B2", linewidth=1.5)
    ax_spot.axhline(K, color="#bbbbbb", linewidth=1, linestyle=":", label="Strike")
    ax_spot.set_ylabel("Spot")
    ax_spot.set_title("Underlying path")
    ax_spot.legend(frameon=False)

    ax_delta.plot(time_axis, result["deltas_target"], color="#bbbbbb", linewidth=1,
                  label="Target delta (theoretical)")
    ax_delta.plot(time_axis, result["deltas_held"], color="#D55E00", linewidth=1.5,
                  label="Held delta (actual hedge)")
    ax_delta.set_ylabel("Delta")
    if rebalance_threshold is not None:
        rule_label = f"{rebalance_threshold:.0%} threshold"
    else:
        rule_label = f"every {rebalance_every} step(s)"
    ax_delta.set_title(f"Delta — rebalancing {rule_label} ({result['n_rebalances']} rebalances)")
    ax_delta.legend(frameon=False)

    ax_pnl.plot(time_axis, result["mark_to_market_pnl"], color="#009E73", linewidth=1.5)
    ax_pnl.axhline(0, color="#555555", linewidth=1)
    ax_pnl.set_ylabel("Mark-to-market P&L")
    ax_pnl.set_xlabel("Time (years)")
    ax_pnl.set_title("Hedge P&L (call seller)")

    for ax in (ax_spot, ax_delta, ax_pnl):
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{title_prefix}  (K={K}, T={T:.2f}, r={r:.0%}, "
                 f"{sigma_label}, final P&L={result['final_pnl']:.2f})", fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Chart saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def plot_hedge_simulation(S0=100, K=100, T=1, r=0.05, sigma_realized=0.20, sigma_implied=None,
                           q=0, n_steps=252, rebalance_every=1, rebalance_threshold=None, seed=42,
                           save_path="../plots/hedge_simulation.png", show=True):
    """
    Visualizes a delta hedging simulation on a single path: spot,
    delta (target vs actually held), and mark-to-market P&L, over time.
    """

    result = delta_hedge_call(S0, K, T, r, sigma_realized, sigma_implied=sigma_implied, q=q,
                               n_steps=n_steps, rebalance_every=rebalance_every,
                               rebalance_threshold=rebalance_threshold, seed=seed)

    sigma_label = f"σ_realized={sigma_realized:.0%}"

    _plot_hedge_panels(result, K, T, r, n_steps, sigma_label, rebalance_every, rebalance_threshold,
                        "Delta hedging simulation", save_path, show)


def plot_real_price_and_volatility(ticker="^GSPC", period="2y", window=21, trading_days=252,
                                    save_path="../plots/real_price_and_vol.png", show=True):
    """
    Plots an asset's real price (via yfinance) and its rolling realized
    volatility, to see how much volatility is NOT constant in reality
    (contrary to the Black-Scholes assumption).
    """

    prices = fetch_price_history(ticker, period=period)
    log_returns = compute_log_returns(prices)
    rolling_vol = compute_rolling_realized_volatility(log_returns, window=window, trading_days=trading_days)
    full_period_vol = compute_realized_volatility(log_returns, trading_days=trading_days)

    fig, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_price.plot(prices.index, prices.values, color="#0072B2", linewidth=1.2)
    ax_price.set_ylabel("Price")
    ax_price.set_title(f"{ticker} price")

    ax_vol.plot(rolling_vol.index, rolling_vol.values, color="#D55E00", linewidth=1.2,
                label=f"Rolling realized vol ({window}d)")
    ax_vol.axhline(full_period_vol, color="#555555", linewidth=1.5, linestyle="--",
                    label=f"Average realized vol ({full_period_vol:.0%})")
    ax_vol.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax_vol.set_ylabel("Realized volatility")
    ax_vol.set_xlabel("Date")
    ax_vol.set_title("Rolling realized volatility")
    ax_vol.legend(frameon=False)

    for ax in (ax_price, ax_vol):
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Real data (yfinance)  —  {ticker}, {period}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Chart saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_real_hedge_simulation(ticker="AAPL", K=None, r=0.05, sigma_implied=None, q=0,
                                period="1y", trading_days=252, rebalance_every=1,
                                rebalance_threshold=None,
                                save_path="../plots/hedge_simulation_real_aapl.png", show=True):
    """
    Simulates delta hedging on a REAL price path (via yfinance), rather than
    on a path simulated by geometric Brownian motion. The implied vol used
    for pricing is, by default, the historical realized vol itself measured
    on that same path (the project has no real source of market implied vol
    — see plots/README.md for the full caveat).
    """

    prices = fetch_price_history(ticker, period=period)
    price_path = prices.to_numpy()
    n_steps = len(price_path) - 1
    S0 = float(price_path[0])

    if K is None:
        K = round(S0)

    T = n_steps / trading_days
    sigma_realized_hist = compute_realized_volatility(compute_log_returns(prices), trading_days=trading_days)

    result = delta_hedge_call(S0, K, T, r, sigma_realized_hist, sigma_implied=sigma_implied, q=q,
                               n_steps=n_steps, rebalance_every=rebalance_every,
                               rebalance_threshold=rebalance_threshold, price_path=price_path)

    sigma_label = f"σ_realized (historical)={sigma_realized_hist:.0%}"

    _plot_hedge_panels(result, K, T, r, n_steps, sigma_label, rebalance_every, rebalance_threshold,
                        f"Delta hedging on a real path ({ticker})", save_path, show)


def plot_dispersion_analysis(tickers=None, weights=None, period="2y",
                              save_path="../plots/dispersion_analysis.png", show=True):
    """
    Realized dispersion analysis on a basket of real stocks (via yfinance):
    correlation matrix of returns (heatmap), and comparison between the
    realized vol of the basket ("mini-index"), its reconstruction "via the
    formula" (individual vols + correlation — the verification round-trip),
    and the weighted average of the individual vols. The gap between these
    last two illustrates the diversification effect due to correlation:
    that's what a dispersion trade tries to exploit (selling index vol,
    buying the constituents' vol, betting on future correlation).
    """

    if tickers is None:
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]

    result = analyze_dispersion(tickers, weights=weights, period=period)
    corr_matrix = result["corr_matrix"]
    n = len(tickers)

    fig, (ax_corr, ax_vol) = plt.subplots(1, 2, figsize=(12, 5))

    im = ax_corr.imshow(corr_matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax_corr.set_xticks(range(n))
    ax_corr.set_yticks(range(n))
    ax_corr.set_xticklabels(tickers)
    ax_corr.set_yticklabels(tickers)
    for i in range(n):
        for j in range(n):
            value = corr_matrix.iloc[i, j]
            ax_corr.text(j, i, f"{value:.2f}", ha="center", va="center",
                         color="white" if abs(value) > 0.6 else "black", fontsize=9)
    ax_corr.set_title("Correlation of returns")
    fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    bar_labels = ["Basket vol\n(direct)", "Basket vol\n(formula)", "Weighted average\nof components"]
    bar_values = [result["basket_vol_direct"], result["basket_vol_formula"],
                  result["weighted_avg_component_vol"]]
    bar_colors = ["#0072B2", "#56B4E9", "#D55E00"]

    ax_vol.bar(bar_labels, bar_values, color=bar_colors)
    for i, value in enumerate(bar_values):
        ax_vol.text(i, value, f"{value:.1%}", ha="center", va="bottom")
    ax_vol.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax_vol.set_ylabel("Annualized volatility")
    ax_vol.set_title("Correlation effect (diversification)")
    ax_vol.grid(True, axis="y", alpha=0.3)
    ax_vol.spines["top"].set_visible(False)
    ax_vol.spines["right"].set_visible(False)

    fig.suptitle(f"Realized dispersion — basket {', '.join(tickers)}  ({period})")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Chart saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig, result


# %%
if __name__ == "__main__":
    plot_price_vs_spot()
    plot_delta_vs_spot()
    plot_gamma_vs_spot()
    plot_vega_vs_spot()
    plot_theta_vs_spot()
    plot_theta_dividend_effect()
    plot_rho_vs_spot()
    plot_implied_vol_smile()
    plot_hedge_simulation()
    plot_real_price_and_volatility()
    plot_real_price_and_volatility(ticker="AAPL", save_path="../plots/real_price_and_vol_aapl.png")
    plot_real_hedge_simulation()
    plot_dispersion_analysis()
