import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

from black_scholes import black_scholes_call, black_scholes_put
from greeks import call_delta, put_delta, gamma, vega, call_theta, put_theta, call_rho, put_rho
from hedging import delta_hedge_call
from dispersion import analyze_dispersion
from plots import plot_price_vs_spot, plot_implied_vol_smile, _plot_hedge_panels, plot_dispersion_analysis


st.set_page_config(page_title="Equity Derivatives Simulator", layout="wide")

st.title("Equity Derivatives Pricing & Hedging — Interactive Simulator")
st.caption("Interactive interface on top of the project — see the README for the theory and detailed limitations.")

tab_pricing, tab_smile, tab_hedging, tab_dispersion = st.tabs(
    ["Pricing & Greeks", "Implied vol smile", "Delta hedging", "Dispersion"]
)


with tab_pricing:
    st.subheader("Black-Scholes price and Greeks")

    col_inputs, col_output = st.columns([1, 2])

    with col_inputs:
        S = st.slider("Spot (S)", 50, 150, 100)
        K = st.slider("Strike (K)", 50, 150, 100)
        T = st.slider("Maturity T (years)", 0.05, 3.0, 1.0, step=0.05)
        r = st.slider("Risk-free rate r", 0.0, 0.10, 0.05, step=0.005, format="%.3f")
        sigma = st.slider("Volatility σ", 0.05, 0.80, 0.20, step=0.01)
        q = st.slider("Dividend q", 0.0, 0.10, 0.0, step=0.005, format="%.3f")

    call_price = black_scholes_call(S, K, T, r, sigma, q)
    put_price = black_scholes_put(S, K, T, r, sigma, q)

    with col_output:
        price_cols = st.columns(3)
        price_cols[0].metric("Call price", f"{call_price:.2f}")
        price_cols[1].metric("Put price", f"{put_price:.2f}")
        price_cols[2].metric("Parity C − P", f"{call_price - put_price:.2f}")

        greek_cols = st.columns(5)
        greek_cols[0].metric("Delta call / put", f"{call_delta(S, K, T, r, sigma, q):.3f} / "
                                                   f"{put_delta(S, K, T, r, sigma, q):.3f}")
        greek_cols[1].metric("Gamma", f"{gamma(S, K, T, r, sigma, q):.4f}")
        greek_cols[2].metric("Vega", f"{vega(S, K, T, r, sigma, q):.2f}")
        greek_cols[3].metric("Theta call / put", f"{call_theta(S, K, T, r, sigma, q):.2f} / "
                                                   f"{put_theta(S, K, T, r, sigma, q):.2f}")
        greek_cols[4].metric("Rho call / put", f"{call_rho(S, K, T, r, sigma, q):.2f} / "
                                                f"{put_rho(S, K, T, r, sigma, q):.2f}")

        st.caption("The chart below compares 3 fixed vol levels (10/20/40%) to show where the "
                   "volatility chosen above sits.")
        fig_price = plot_price_vs_spot(K=K, T=T, r=r, q=q, save_path=None, show=False)
        st.pyplot(fig_price)


with tab_smile:
    st.subheader("Implied volatility smile (synthetic)")
    st.caption("No real options data in this project — see the Limitations section of the README. "
               "This smile is built from a made-up vol curve, then recovered via numerical "
               "inversion of Black-Scholes: the two curves should overlap exactly.")

    col_inputs, col_output = st.columns([1, 2])

    with col_inputs:
        S_smile = st.slider("Spot (S)", 50, 150, 100, key="S_smile")
        T_smile = st.slider("Maturity T (years)", 0.05, 3.0, 1.0, step=0.05, key="T_smile")
        r_smile = st.slider("Risk-free rate r", 0.0, 0.10, 0.05, step=0.005, format="%.3f", key="r_smile")
        q_smile = st.slider("Dividend q", 0.0, 0.10, 0.0, step=0.005, format="%.3f", key="q_smile")
        base_vol = st.slider("Base vol (ATM)", 0.05, 0.60, 0.20, step=0.01)
        skew_slope = st.slider("Skew slope", -0.50, 0.50, -0.15, step=0.01)
        convexity = st.slider("Convexity", 0.0, 1.0, 0.25, step=0.05)

    with col_output:
        fig_smile = plot_implied_vol_smile(S=S_smile, T=T_smile, r=r_smile, q=q_smile,
                                            base_vol=base_vol, skew_slope=skew_slope, convexity=convexity,
                                            save_path=None, show=False)
        st.pyplot(fig_smile)


with tab_hedging:
    st.subheader("Delta hedging simulation")
    st.caption("Selling a delta-hedged European call on a simulated path of the underlying, "
               "from the seller's point of view.")

    col_inputs, col_output = st.columns([1, 2])

    with col_inputs:
        S0_hedge = st.slider("Initial spot (S0)", 50, 150, 100)
        K_hedge = st.slider("Strike (K)", 50, 150, 100, key="K_hedge")
        T_hedge = st.slider("Maturity T (years)", 0.1, 3.0, 1.0, step=0.1, key="T_hedge")
        r_hedge = st.slider("Risk-free rate r", 0.0, 0.10, 0.05, step=0.005, format="%.3f", key="r_hedge")
        sigma_realized = st.slider("Realized vol (simulated path)", 0.05, 0.80, 0.20, step=0.01)
        sigma_implied_differs = st.checkbox("Implied vol (pricing) differs from realized vol")
        sigma_implied = None
        if sigma_implied_differs:
            sigma_implied = st.slider("Implied vol (pricing)", 0.05, 0.80, 0.20, step=0.01)

        rebalance_mode = st.radio("Rebalancing mode", ["Fixed interval", "Adaptive threshold"])
        rebalance_every = 1
        rebalance_threshold = None
        if rebalance_mode == "Fixed interval":
            rebalance_every = st.slider("Rebalance every N steps", 1, 21, 1)
        else:
            rebalance_threshold = st.slider("Rebalancing threshold (delta drift)", 0.01, 0.20, 0.05,
                                             step=0.01)

        seed = st.number_input("Seed (reproducibility)", value=42, step=1)

    result = delta_hedge_call(S0_hedge, K_hedge, T_hedge, r_hedge, sigma_realized,
                               sigma_implied=sigma_implied, q=0, n_steps=252,
                               rebalance_every=rebalance_every, rebalance_threshold=rebalance_threshold,
                               seed=int(seed))

    with col_output:
        result_cols = st.columns(2)
        result_cols[0].metric("Final P&L", f"{result['final_pnl']:.2f}")
        result_cols[1].metric("Number of rebalances", result["n_rebalances"])

        sigma_label = f"σ_realized={sigma_realized:.0%}"
        fig_hedge = _plot_hedge_panels(result, K_hedge, T_hedge, r_hedge, 252, sigma_label,
                                        rebalance_every, rebalance_threshold,
                                        "Delta hedging simulation", save_path=None, show=False)
        st.pyplot(fig_hedge)


with tab_dispersion:
    st.subheader("Realized dispersion on a stock basket")
    st.caption("**Realized** dispersion, not an actual desk trade (no real implied vols "
               "in this project — see the Limitations section of the README).")

    default_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
    tickers = st.multiselect("Basket tickers (yfinance)", options=default_tickers + ["GOOGL", "TSLA", "JPM"],
                              default=default_tickers)
    period = st.selectbox("Period", ["1y", "2y", "5y"], index=1)

    if len(tickers) < 2:
        st.warning("Choose at least 2 tickers.")
    elif st.button("Analyze"):
        try:
            with st.spinner("Downloading prices and computing..."):
                fig_dispersion, dispersion_result = plot_dispersion_analysis(tickers=tickers, period=period,
                                                                               save_path=None, show=False)
        except (ValueError, OSError) as error:
            st.error(f"Could not fetch data via yfinance: {error}")
        else:
            metric_cols = st.columns(3)
            metric_cols[0].metric("Basket vol (direct)", f"{dispersion_result['basket_vol_direct']:.1%}")
            metric_cols[1].metric("Basket vol (formula)", f"{dispersion_result['basket_vol_formula']:.1%}")
            metric_cols[2].metric("Weighted average of components",
                                   f"{dispersion_result['weighted_avg_component_vol']:.1%}")

            st.pyplot(fig_dispersion)
