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

st.title("Equity Derivatives Pricing & Hedging — Simulateur interactif")
st.caption("Interface interactive au-dessus du projet — voir le README pour la théorie et les limites détaillées.")

tab_pricing, tab_smile, tab_hedging, tab_dispersion = st.tabs(
    ["Pricing & Greeks", "Smile de vol implicite", "Delta hedging", "Dispersion"]
)


with tab_pricing:
    st.subheader("Prix Black-Scholes et Greeks")

    col_inputs, col_output = st.columns([1, 2])

    with col_inputs:
        S = st.slider("Spot (S)", 50, 150, 100)
        K = st.slider("Strike (K)", 50, 150, 100)
        T = st.slider("Maturité T (années)", 0.05, 3.0, 1.0, step=0.05)
        r = st.slider("Taux sans risque r", 0.0, 0.10, 0.05, step=0.005, format="%.3f")
        sigma = st.slider("Volatilité σ", 0.05, 0.80, 0.20, step=0.01)
        q = st.slider("Dividende q", 0.0, 0.10, 0.0, step=0.005, format="%.3f")

    call_price = black_scholes_call(S, K, T, r, sigma, q)
    put_price = black_scholes_put(S, K, T, r, sigma, q)

    with col_output:
        price_cols = st.columns(3)
        price_cols[0].metric("Prix du call", f"{call_price:.2f}")
        price_cols[1].metric("Prix du put", f"{put_price:.2f}")
        price_cols[2].metric("Parité C − P", f"{call_price - put_price:.2f}")

        greek_cols = st.columns(5)
        greek_cols[0].metric("Delta call / put", f"{call_delta(S, K, T, r, sigma, q):.3f} / "
                                                   f"{put_delta(S, K, T, r, sigma, q):.3f}")
        greek_cols[1].metric("Gamma", f"{gamma(S, K, T, r, sigma, q):.4f}")
        greek_cols[2].metric("Vega", f"{vega(S, K, T, r, sigma, q):.2f}")
        greek_cols[3].metric("Theta call / put", f"{call_theta(S, K, T, r, sigma, q):.2f} / "
                                                   f"{put_theta(S, K, T, r, sigma, q):.2f}")
        greek_cols[4].metric("Rho call / put", f"{call_rho(S, K, T, r, sigma, q):.2f} / "
                                                f"{put_rho(S, K, T, r, sigma, q):.2f}")

        st.caption("Le graphique ci-dessous compare 3 niveaux de vol fixes (10/20/40%) pour situer "
                   "où se place la volatilité choisie ci-dessus.")
        fig_price = plot_price_vs_spot(K=K, T=T, r=r, q=q, save_path=None, show=False)
        st.pyplot(fig_price)


with tab_smile:
    st.subheader("Smile de volatilité implicite (synthétique)")
    st.caption("Pas de vraies données d'options dans ce projet — voir la section Limites du README. "
               "Ce smile est fabriqué à partir d'une courbe de vol imaginaire, puis retrouvé par "
               "inversion numérique de Black-Scholes : les deux courbes doivent se superposer exactement.")

    col_inputs, col_output = st.columns([1, 2])

    with col_inputs:
        S_smile = st.slider("Spot (S)", 50, 150, 100, key="S_smile")
        T_smile = st.slider("Maturité T (années)", 0.05, 3.0, 1.0, step=0.05, key="T_smile")
        r_smile = st.slider("Taux sans risque r", 0.0, 0.10, 0.05, step=0.005, format="%.3f", key="r_smile")
        q_smile = st.slider("Dividende q", 0.0, 0.10, 0.0, step=0.005, format="%.3f", key="q_smile")
        base_vol = st.slider("Vol de base (ATM)", 0.05, 0.60, 0.20, step=0.01)
        skew_slope = st.slider("Pente du skew", -0.50, 0.50, -0.15, step=0.01)
        convexity = st.slider("Convexité", 0.0, 1.0, 0.25, step=0.05)

    with col_output:
        fig_smile = plot_implied_vol_smile(S=S_smile, T=T_smile, r=r_smile, q=q_smile,
                                            base_vol=base_vol, skew_slope=skew_slope, convexity=convexity,
                                            save_path=None, show=False)
        st.pyplot(fig_smile)


with tab_hedging:
    st.subheader("Simulation de delta hedging")
    st.caption("Vente d'un call européen couvert en delta sur une trajectoire simulée du sous-jacent, "
               "point de vue du vendeur.")

    col_inputs, col_output = st.columns([1, 2])

    with col_inputs:
        S0_hedge = st.slider("Spot initial (S0)", 50, 150, 100)
        K_hedge = st.slider("Strike (K)", 50, 150, 100, key="K_hedge")
        T_hedge = st.slider("Maturité T (années)", 0.1, 3.0, 1.0, step=0.1, key="T_hedge")
        r_hedge = st.slider("Taux sans risque r", 0.0, 0.10, 0.05, step=0.005, format="%.3f", key="r_hedge")
        sigma_realized = st.slider("Vol réalisée (trajectoire simulée)", 0.05, 0.80, 0.20, step=0.01)
        sigma_implied_differs = st.checkbox("Vol implicite (pricing) différente de la vol réalisée")
        sigma_implied = None
        if sigma_implied_differs:
            sigma_implied = st.slider("Vol implicite (pricing)", 0.05, 0.80, 0.20, step=0.01)

        rebalance_mode = st.radio("Mode de rebalancement", ["Intervalle fixe", "Seuil adaptatif"])
        rebalance_every = 1
        rebalance_threshold = None
        if rebalance_mode == "Intervalle fixe":
            rebalance_every = st.slider("Rebalancer tous les N pas", 1, 21, 1)
        else:
            rebalance_threshold = st.slider("Seuil de rebalancement (écart de delta)", 0.01, 0.20, 0.05,
                                             step=0.01)

        seed = st.number_input("Seed (reproductibilité)", value=42, step=1)

    result = delta_hedge_call(S0_hedge, K_hedge, T_hedge, r_hedge, sigma_realized,
                               sigma_implied=sigma_implied, q=0, n_steps=252,
                               rebalance_every=rebalance_every, rebalance_threshold=rebalance_threshold,
                               seed=int(seed))

    with col_output:
        result_cols = st.columns(2)
        result_cols[0].metric("P&L final", f"{result['final_pnl']:.2f}")
        result_cols[1].metric("Nombre de rebalancements", result["n_rebalances"])

        sigma_label = f"σ_réalisée={sigma_realized:.0%}"
        fig_hedge = _plot_hedge_panels(result, K_hedge, T_hedge, r_hedge, 252, sigma_label,
                                        rebalance_every, rebalance_threshold,
                                        "Simulation de delta hedging", save_path=None, show=False)
        st.pyplot(fig_hedge)


with tab_dispersion:
    st.subheader("Dispersion réalisée sur un panier d'actions")
    st.caption("Dispersion **réalisée**, pas un vrai trade de desk (pas de vraies vols implicites "
               "dans ce projet — voir la section Limites du README).")

    default_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
    tickers = st.multiselect("Tickers du panier (yfinance)", options=default_tickers + ["GOOGL", "TSLA", "JPM"],
                              default=default_tickers)
    period = st.selectbox("Période", ["1y", "2y", "5y"], index=1)

    if len(tickers) < 2:
        st.warning("Choisir au moins 2 tickers.")
    elif st.button("Analyser"):
        try:
            with st.spinner("Téléchargement des prix et calcul en cours..."):
                fig_dispersion, dispersion_result = plot_dispersion_analysis(tickers=tickers, period=period,
                                                                               save_path=None, show=False)
        except (ValueError, OSError) as error:
            st.error(f"Impossible de récupérer les données via yfinance : {error}")
        else:
            metric_cols = st.columns(3)
            metric_cols[0].metric("Vol du panier (directe)", f"{dispersion_result['basket_vol_direct']:.1%}")
            metric_cols[1].metric("Vol du panier (formule)", f"{dispersion_result['basket_vol_formula']:.1%}")
            metric_cols[2].metric("Moyenne pondérée des composants",
                                   f"{dispersion_result['weighted_avg_component_vol']:.1%}")

            st.pyplot(fig_dispersion)
