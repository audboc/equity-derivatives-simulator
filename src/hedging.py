from math import exp

import numpy as np

from black_scholes import black_scholes_call
from greeks import call_delta
from simulation import simulate_gbm_path


def delta_hedge_call(S0, K, T, r, sigma_realized, sigma_implied=None, q=0, n_steps=252,
                      rebalance_every=1, rebalance_threshold=None, seed=None, price_path=None):
    """
    Simule la vente d'un call européen, couvert par delta hedging, sur une trajectoire
    du sous-jacent (simulée par défaut, ou une vraie trajectoire si price_path est fourni).

    sigma_realized : vol utilisée pour simuler la trajectoire du sous-jacent (le "vrai" monde).
                      Ignorée pour la génération de la trajectoire si price_path est fourni
                      (mais reste utile comme référence, ex : vol réalisée mesurée sur ce chemin).
    sigma_implied : vol utilisée pour pricer et calculer le delta (le prix auquel l'option a
                    été vendue). Égale à sigma_realized par défaut si non précisée.
    rebalance_every : nombre de pas entre deux rebalancements (1 = à chaque pas, 5 = environ
                       chaque semaine si n_steps=252). Ignoré si rebalance_threshold est précisé.
    rebalance_threshold : si précisé (ex : 0.05), on rebalance dès que le delta théorique
                           s'écarte du delta détenu de plus de ce seuil, plutôt qu'à intervalle
                           fixe. Remplace rebalance_every quand il est utilisé.
    price_path : trajectoire déjà connue (ex : prix réels récupérés via market_data.py) à
                 utiliser à la place d'une simulation GBM. Doit contenir exactement n_steps+1
                 points, avec price_path[0] == S0. Si None (par défaut), la trajectoire est
                 simulée avec simulate_gbm_path comme avant.

    Logique (point de vue du vendeur de l'option) :
    - on encaisse la prime en vendant le call
    - on achète delta actions pour se couvrir, financé par le cash restant
    - à chaque pas : le cash accrue au taux sans risque, le delta théorique est recalculé ;
      on ne rebalance (achète/vend des actions) que si le déclencheur choisi est atteint
    - à l'échéance : on rebalance une dernière fois pour liquider, puis on paie le payoff

    Retourne un dictionnaire avec la trajectoire, l'historique du delta (cible et détenu),
    le nombre de rebalancements, et le P&L final de la stratégie de couverture.
    """

    if sigma_implied is None:
        sigma_implied = sigma_realized

    dt = T / n_steps

    if price_path is not None:
        path = np.asarray(price_path, dtype=float)
        if len(path) != n_steps + 1:
            raise ValueError(f"price_path doit avoir n_steps+1={n_steps + 1} points, reçu {len(path)}.")
        if not np.isclose(path[0], S0):
            raise ValueError(f"price_path[0]={path[0]} doit être égal à S0={S0}.")
    else:
        path = simulate_gbm_path(S0, mu=r, sigma=sigma_realized, T=T, n_steps=n_steps, q=q, seed=seed)

    option_price_0 = black_scholes_call(S0, K, T, r, sigma_implied, q)
    delta_0 = call_delta(S0, K, T, r, sigma_implied, q)

    # Prime encaissée moins le coût d'achat des actions de couverture
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

        # On rebalance soit par seuil, soit à intervalle fixe, et toujours au dernier pas (liquidation)
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

    # P&L "mark-to-market" à chaque instant : valeur du portefeuille de couverture
    # moins la valeur théorique de l'option qu'on doit finir par payer
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

    # Une simulation simple : vol réalisée = vol implicite
    result = delta_hedge_call(S0=100, K=100, T=1, r=0.05, sigma_realized=0.20, seed=42)

    print(f"Prime encaissée à la vente : {result['option_price_0']:.4f}")
    print(f"Payoff payé à l'échéance   : {result['option_payoff']:.4f}")
    print(f"P&L final de la couverture : {result['final_pnl']:.4f}")

    # Vérification statistique : si vol réalisée = vol implicite, le P&L moyen
    # sur beaucoup de trajectoires doit être proche de 0 (la couverture réplique l'option)
    n_paths = 2000
    pnls_matched = [delta_hedge_call(S0=100, K=100, T=1, r=0.05, sigma_realized=0.20)["final_pnl"]
                    for _ in range(n_paths)]

    print(f"\nSur {n_paths} trajectoires, vol réalisée = vol implicite (20%) :")
    print(f"P&L moyen : {np.mean(pnls_matched):.4f}")
    print(f"Écart-type du P&L : {np.std(pnls_matched):.4f}")

    # Si la vol réalisée est plus élevée que la vol implicite (option sous-évaluée
    # à la vente), le vendeur couvert doit en moyenne PERDRE de l'argent
    pnls_high_vol = [delta_hedge_call(S0=100, K=100, T=1, r=0.05,
                                       sigma_realized=0.35, sigma_implied=0.20)["final_pnl"]
                     for _ in range(n_paths)]

    print(f"\nSur {n_paths} trajectoires, vol réalisée (35%) > vol implicite (20%) :")
    print(f"P&L moyen : {np.mean(pnls_high_vol):.4f}")

    # Comparaison : rebalancement quotidien (tous les jours) vs hebdomadaire (tous les 5 jours)
    print(f"\nComparaison quotidien vs hebdomadaire, sur {n_paths} trajectoires (vol réalisée = implicite = 20%) :")

    for label, rebalance_every in [("Quotidien (tous les jours)", 1), ("Hebdomadaire (tous les 5 jours)", 5)]:
        results = [delta_hedge_call(S0=100, K=100, T=1, r=0.05, sigma_realized=0.20, rebalance_every=rebalance_every)
                   for _ in range(n_paths)]

        pnls = [res["final_pnl"] for res in results]
        n_rebalances = [res["n_rebalances"] for res in results]

        print(f"\n{label} :")
        print(f"  P&L moyen : {np.mean(pnls):.4f}")
        print(f"  Écart-type du P&L : {np.std(pnls):.4f}")
        print(f"  Nombre moyen de rebalancements : {np.mean(n_rebalances):.1f}")

    # Troisième mode : rebalancement déclenché par un seuil de mouvement du delta,
    # plutôt qu'à intervalle fixe
    print(f"\nComparaison avec un rebalancement par seuil, sur {n_paths} trajectoires :")

    for label, threshold in [("Seuil 5%", 0.05), ("Seuil 10%", 0.10)]:
        results = [delta_hedge_call(S0=100, K=100, T=1, r=0.05, sigma_realized=0.20, rebalance_threshold=threshold)
                   for _ in range(n_paths)]

        pnls = [res["final_pnl"] for res in results]
        n_rebalances = [res["n_rebalances"] for res in results]

        print(f"\n{label} :")
        print(f"  P&L moyen : {np.mean(pnls):.4f}")
        print(f"  Écart-type du P&L : {np.std(pnls):.4f}")
        print(f"  Nombre moyen de rebalancements : {np.mean(n_rebalances):.1f}")
