# Equity Derivatives Pricing and Hedging Simulator

## Introduction

Ce projet part du modèle de Black-Scholes pour construire, étape par étape, un simulateur de pricing et de couverture d'options sur actions : calcul du prix d'un call/put européen, des Greeks (delta, gamma, vega, theta, rho), de la volatilité implicite, simulation du sous-jacent, et delta hedging avec analyse du P&L résultant.

L'objectif n'est pas de reproduire l'outil d'un desk professionnel, mais de comprendre en profondeur — et de pouvoir montrer concrètement — comment ces briques théoriques s'articulent entre elles : pourquoi un vendeur d'option couvert en delta peut quand même perdre de l'argent, comment la fréquence de rebalancement affecte le risque, et ce que change le fait de tester tout ça sur de vraies données de marché plutôt que sur des trajectoires simulées.

## Théorie

### Black-Scholes

Le modèle suppose que le sous-jacent suit un mouvement brownien géométrique (rendements log-normaux, volatilité σ et taux sans risque r constants, pas de coûts de transaction, exercice européen uniquement). Sous ces hypothèses, le prix d'un call et d'un put avec dividende continu q sont donnés par :

```
C = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
P = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1)

d1 = [ln(S/K) + (r − q + σ²/2)·T] / (σ√T)
d2 = d1 − σ√T
```

Ces deux prix ne sont pas indépendants : la **parité call-put** (`C − P = S·e^(−qT) − K·e^(−rT)`) doit toujours être vérifiée, et sert de test de cohérence de base dans `tests/test_pricing.py`.

### Greeks

Chaque Greek mesure la sensibilité du prix à une variable : delta (spot), gamma (variation du delta, donc convexité en spot), vega (volatilité), theta (temps), rho (taux). `src/greeks.py` les implémente en dérivant directement les formules ci-dessus. La section [Vega vs gamma](#vega-en-fonction-du-spot) plus bas détaille la différence entre les deux sensibilités de second ordre.

### Volatilité implicite et hedging

La volatilité implicite est l'opération inverse : au lieu de partir de σ pour calculer un prix, on part d'un prix observé et on retrouve le σ qui le justifie (inversion numérique de Black-Scholes, voir `src/implied_volatility.py`). Le delta hedging, lui, exploite le delta comme ratio de couverture : détenir `Δ` actions par option vendue neutralise (localement) l'exposition au spot. Comme Δ change avec le spot (à cause du gamma), la couverture doit être rebalancée — d'où le P&L résiduel étudié dans la section Résultats.

## Résultats

Chaque image ci-dessous est générée par une fonction de `src/plots.py`. Pour les régénérer, ouvrir `plots.py` dans VS Code et utiliser *Run Current File in Interactive Window*.

### Prix Black-Scholes en fonction du spot

`plot_price_vs_spot()`

Call et put européens, K=100, T=1, r=5%, q=0%, pour trois niveaux de volatilité (10%, 20%, 40%), avec la valeur intrinsèque en référence.

![Prix du call et du put en fonction du spot](plots/price_vs_spot.png)

### Delta en fonction du spot

`plot_delta_vs_spot()`

Delta du call et du put, K=100, T=1, r=5%, σ=20%, q=0%. Le call tend vers 1 quand le spot monte, le put tend vers −1 quand le spot baisse — la pente est maximale près du strike (K=100).

![Delta du call et du put en fonction du spot](plots/delta_vs_spot.png)

### Gamma en fonction du spot

`plot_gamma_vs_spot()`

Gamma (identique call/put), K=100, r=5%, σ=20%, q=0%, pour trois maturités (0.25, 1, 2 ans). Plus l'échéance est proche, plus le pic de gamma est haut et étroit autour du strike — c'est ce qui rend une option courte difficile à couvrir près de l'échéance.

![Gamma en fonction du spot, pour plusieurs maturités](plots/gamma_vs_spot.png)

### Vega en fonction du spot

`plot_vega_vs_spot()`

Vega (identique call/put), K=100, r=5%, σ=20%, q=0%, pour trois maturités (0.25, 1, 2 ans). Contrairement au gamma, le vega augmente avec la maturité — plus il reste de temps, plus une variation de volatilité a d'impact sur le prix.

![Vega en fonction du spot, pour plusieurs maturités](plots/vega_vs_spot.png)

#### Vega vs gamma : quelle différence ?

Les deux mesurent une sensibilité de second ordre, mais pas par rapport à la même variable :

- **Gamma** = sensibilité du delta au **spot**. Répond à : *"si le sous-jacent bouge de 1€ maintenant, de combien mon delta (ratio de couverture) change-t-il ?"* Il se matérialise via de vrais mouvements du sous-jacent — c'est pour ça qu'il est lié au delta hedging (il dicte la vitesse de rebalancement).
- **Vega** = sensibilité du prix à la **volatilité**. Répond à : *"si le marché change d'avis sur la volatilité future, de combien le prix change-t-il, même si le spot ne bouge pas ?"* Il se matérialise même sans mouvement du sous-jacent — juste parce que le marché réévalue le risque futur (ex : une annonce de résultats approche, la vol implicite grimpe, l'option prend de la valeur alors que le spot n'a pas bougé).

Les deux graphiques ci-dessus l'illustrent en miroir : le gamma est fort quand la maturité est **courte** (le prix doit "se décider" vite), le vega est fort quand la maturité est **longue** (plus de temps = plus d'incertitude potentielle sur le chemin de la vol). Une option courte est plutôt "gamma", une option longue plutôt "vega".

### Theta en fonction du spot

`plot_theta_vs_spot()`

Theta du call et du put, K=100, T=1, r=5%, σ=20%, q=0%. Le call reste toujours négatif. Le put, lui, passe spontanément en **positif** quand il est assez ITM (spot ≲ 88) — sans forcer aucun paramètre extrême : encaisser le strike tôt (via le terme `+r·K·e^(−rT)·N(−d2)`) vaut alors plus que l'optionalité perdue.

![Theta du call et du put en fonction du spot](plots/theta_vs_spot.png)

### Cas particulier : theta du call très ITM selon le dividende

`plot_theta_dividend_effect()`

Le call, contrairement au put, ne passe jamais en theta positif sans dividende (courbe q=0%, toujours négative). Avec un dividende élevé (q=10%, 20%), le theta du call très ITM devient positif à son tour — le terme `+q·S·e^(−qT)·N(d1)` finit par dominer.

![Theta du call très ITM en fonction du spot, pour plusieurs niveaux de dividende](plots/theta_dividend_effect.png)

### Rho en fonction du spot

`plot_rho_vs_spot()`

Rho du call et du put, K=100, T=1, r=5%, σ=20%, q=0%. Le rho du call est toujours positif (un taux plus élevé augmente la valeur du call), celui du put toujours négatif — les deux tendent vers 0 aux extrêmes (options très OTM/ITM).

![Rho du call et du put en fonction du spot](plots/rho_vs_spot.png)

### Smile de volatilité implicite (synthétique)

`plot_implied_vol_smile()`

**Attention : vol "marché" synthétique, pas de vraies données.** On fabrique des prix de call à partir d'une courbe de vol imaginaire (plus élevée pour les strikes bas + un peu de convexité, comme le skew observé sur les indices actions), puis on retrouve cette courbe strike par strike avec `implied_volatility_call` (`brentq`). Les deux courbes se superposent exactement — ça valide que l'inversion fonctionne sur toute une gamme de strikes, pas juste au point testé dans `implied_volatility.py`.

![Smile de volatilité implicite synthétique en fonction du strike](plots/implied_vol_smile.png)

### Simulation de delta hedging

`plot_hedge_simulation()`

Une trajectoire simulée (K=100, T=1, r=5%, σ=20%, rebalancement à chaque pas), vue du vendeur d'un call couvert en delta. Trois panneaux synchronisés dans le temps : le spot, le delta rebalancé (qui suit fidèlement le spot et tombe à 0 quand l'option finit très OTM), et le P&L mark-to-market (valeur du portefeuille de couverture moins valeur théorique de l'option restant à payer). Le P&L progresse par petits à-coups à chaque rebalancement — c'est le frottement du hedging discret (rebalancer en continu serait l'idéal théorique, impossible en pratique).

![Simulation de delta hedging : spot, delta et P&L dans le temps](plots/hedge_simulation.png)

### Rebalancement quotidien vs hebdomadaire

`plot_hedge_simulation(rebalance_every=5)`

Même trajectoire, mais rebalancée seulement tous les 5 pas (~hebdomadaire) au lieu de chaque jour. Le delta détenu (orange) suit le delta cible (gris) en escalier, avec un vrai écart entre deux rebalancements — visible en particulier autour de t=0.55 et t=0.9, où le P&L fait des sauts nets plus marqués.

![Simulation de delta hedging avec rebalancement hebdomadaire](plots/hedge_simulation_weekly.png)

**Comparaison statistique** (2000 trajectoires, K=100, T=1, r=5%, vol réalisée = vol implicite = 20%) :

| Fréquence | P&L moyen | Écart-type du P&L | Rebalancements moyens |
|---|---|---|---|
| Quotidien (tous les pas) | −0.006 | 0.436 | 253 |
| Hebdomadaire (tous les 5 pas) | −0.019 | 0.953 | 52 |

Le P&L moyen reste proche de zéro dans les deux cas (la couverture n'est pas biaisée), mais l'**écart-type double presque** en hebdomadaire pour 5x moins de rebalancements. C'est le vrai compromis du delta hedging discret : rebalancer moins souvent réduit les coûts de transaction mais augmente fortement le risque (dispersion du P&L) — rebalancer en continu serait l'idéal théorique (risque nul), impossible en pratique.

### Rebalancement par seuil (adaptatif)

`plot_hedge_simulation(rebalance_threshold=0.05)`

Troisième mode : au lieu d'un intervalle fixe, on rebalance dès que le delta théorique s'écarte du delta détenu de plus d'un seuil donné (ici 5%). Le delta détenu colle beaucoup plus fidèlement au delta cible qu'en hebdomadaire — le seuil rebalance davantage quand le marché bouge vraiment, et moins quand c'est calme.

![Simulation de delta hedging avec rebalancement par seuil de 5%](plots/hedge_simulation_threshold.png)

**Comparaison complète** (2000 trajectoires, mêmes paramètres) :

| Stratégie | P&L moyen | Écart-type du P&L | Rebalancements moyens |
|---|---|---|---|
| Quotidien | −0.013 | 0.431 | 253 |
| Hebdomadaire | −0.025 | 0.959 | 52 |
| Seuil 5% | 0.006 | 0.678 | 38.7 |
| Seuil 10% | 0.029 | 1.058 | 15.6 |

Résultat notable : à nombre de rebalancements comparable, le **seuil bat le calendrier fixe**. Le seuil 5% (38.7 rebalancements, un peu moins que l'hebdomadaire) obtient un écart-type de 0.678, nettement inférieur aux 0.959 de l'hebdomadaire (52 rebalancements). Rebalancer *quand le marché bouge* plutôt que *selon un calendrier* réduit le risque à budget de transactions équivalent — c'est ce que font en pratique la plupart des desks.

### Prix réel et volatilité réalisée : S&P 500

`plot_real_price_and_volatility()`

**Attention : données réelles utilisées à titre illustratif et pédagogique, pas au niveau de rigueur d'un desk professionnel** (pas de traitement avancé des jours fériés/splits/dividendes au-delà de ce que fournit `yfinance`, pas de nettoyage de données). Prix de clôture ajustés du S&P 500 (`^GSPC`) sur 2 ans via `yfinance`, et volatilité réalisée glissante sur une fenêtre de 21 jours (~1 mois), annualisée. La ligne pointillée grise est la vol réalisée moyenne sur toute la période (16%). On voit très clairement que **la volatilité n'est pas constante dans la réalité** — contrairement à l'hypothèse de Black-Scholes — avec un pic à ~49% coïncidant exactement avec la chute brutale du prix visible sur le panneau du haut.

![Prix du S&P 500 et volatilité réalisée glissante](plots/real_price_and_vol.png)

### Prix réel et volatilité réalisée : Apple

`plot_real_price_and_volatility(ticker="AAPL", save_path="../plots/real_price_and_vol_aapl.png")`

Même lecture, sur Apple (`AAPL`). Même pic de volatilité (marché large, même période), mais nettement plus prononcé (jusqu'à 78%) et une moyenne plus élevée (29% vs 16% pour l'indice) — une action individuelle est toujours plus volatile qu'un indice diversifié.

![Prix d'Apple et volatilité réalisée glissante](plots/real_price_and_vol_aapl.png)

### Delta hedging sur une trajectoire réelle : Apple

`plot_real_hedge_simulation()`

**Attention : il n'y a pas de vraie source de volatilité implicite de marché dans ce projet** — seule la vol implicite synthétique de `plot_implied_vol_smile()` existe (pas de vraies données d'options). La "vol implicite" utilisée ici pour pricer l'option et calculer le delta théorique est donc simplement la vol réalisée historique mesurée sur la même trajectoire, pas une vraie vol de marché observée sur des prix d'options réels. Cette simulation réutilise exactement le même moteur (`delta_hedge_call`) que les trajectoires simulées plus haut, mais avec `price_path` pointant vers une vraie série de prix récupérée via `market_data.py` au lieu d'un chemin brownien géométrique — elle sert à voir à quoi ressemble un delta hedging sur un vrai chemin de marché (irrégulier, avec ses propres régimes de volatilité), pas à mesurer un vrai P&L de trading historique.

![Simulation de delta hedging sur le prix réel d'Apple](plots/hedge_simulation_real_aapl.png)

### Dispersion réalisée : panier AAPL / MSFT / NVDA / AMZN / META

`plot_dispersion_analysis()`

**Attention : dispersion "réalisée", pas un vrai trade de dispersion de desk.** Un vrai trade de dispersion vend de la vol *implicite* d'indice et achète de la vol *implicite* des composants — ce projet n'a pas de vraies données d'options (voir la mise en garde sur `plot_implied_vol_smile()` plus haut). Ce qui suit utilise uniquement de la vol et de la corrélation *réalisées* sur des prix réels, pour illustrer le mécanisme sous-jacent : pourquoi la vol d'un indice est presque toujours inférieure à la moyenne de ses composants, et le rôle exact que joue la corrélation dans cet écart.

Cinq actions réelles (`AAPL`, `MSFT`, `NVDA`, `AMZN`, `META`), équipondérées, prix sur 2 ans via `yfinance`. Le "mini-indice" est construit en indexant chaque action à 100 au départ puis en sommant les niveaux pondérés — pour que le prix en dollars de chaque action ne fausse pas son poids.

**Round-trip de vérification** : la vol du panier peut se calculer de deux façons indépendantes.
- *Directe* : on calcule les rendements du panier lui-même, comme pour n'importe quelle action → **26.2%**.
- *Par la formule* : on part uniquement des vols individuelles + de la matrice de corrélation, et on reconstruit `σ_panier² = wᵀ Σ w` (avec `Σᵢⱼ = σᵢ σⱼ ρᵢⱼ`) → **25.8%**.

Les deux méthodes concordent à ~0.4 point près (petit écart attendu : la formule est exacte pour des rendements simples, on l'applique ici à des rendements logarithmiques, dont la somme pondérée n'est pas rigoureusement égale au rendement log du panier). Ce round-trip valide le calcul de corrélation, exactement comme le round-trip prix → vol implicite → prix avait validé l'inversion de Black-Scholes.

Une fois la formule validée, elle isole l'effet de la corrélation : la moyenne pondérée des vols individuelles est de **34.9%**, bien au-dessus des ~26% du panier. L'écart (~9 points) est entièrement dû à la diversification — les corrélations entre paires (0.31 à 0.58 sur la heatmap) sont bien inférieures à 1, donc les mouvements des actions s'annulent partiellement au lieu de s'additionner. C'est précisément cet écart qu'un trade de dispersion cherche à monétiser : si le marché price une corrélation implicite plus élevée que la corrélation qui se réalisera vraiment, vendre la vol de l'indice et acheter celle des composants est gagnant (et inversement).

![Corrélation et effet de diversification sur un panier de 5 actions](plots/dispersion_analysis.png)

## Fonctionnalités

| Module | Contenu |
|---|---|
| `src/black_scholes.py` | Prix call/put européens, parité call-put |
| `src/greeks.py` | Delta, gamma, vega, theta, rho (call et put) |
| `src/implied_volatility.py` | Inversion numérique de Black-Scholes (`brentq`) pour retrouver σ à partir d'un prix |
| `src/simulation.py` | Simulation de trajectoires par mouvement brownien géométrique |
| `src/hedging.py` | Delta hedging (rebalancement à intervalle fixe ou par seuil), P&L mark-to-market |
| `src/market_data.py` | Récupération de prix réels (`yfinance`), rendements log, volatilité réalisée (globale et glissante) |
| `src/dispersion.py` | Vol réalisée + corrélation d'un panier d'actions, construction d'un panier pondéré, vérification bottom-up de sa vol |
| `src/plots.py` | Toutes les visualisations du projet, une fonction par graphique, chaque fonction sauvegarde son PNG dans `plots/` |
| `app.py` | Interface Streamlit interactive au-dessus des modules ci-dessus (voir section dédiée plus bas) |

## Limites

Ce projet est un outil pédagogique, pas un outil de desk. Principales limites, à garder à l'esprit en lisant les résultats ci-dessus :

- **Pas de vraies données de volatilité implicite.** Aucune source de prix d'options réels n'est utilisée : le smile de la section Résultats est entièrement synthétique, et la "vol implicite" utilisée dans le delta hedging sur données réelles est en fait de la vol réalisée historique. Un vrai trade de dispersion se joue sur des vols et une corrélation *implicites*, pas réalisées — ce projet ne peut illustrer que la mécanique sous-jacente (voir la section Dispersion).
- **Black-Scholes lui-même est une simplification** : volatilité et taux constants, pas de sauts de prix, exercice européen uniquement (pas d'options américaines), pas de coûts de transaction ni de contrainte de liquidité dans le delta hedging.
- **Données réelles non nettoyées à un niveau professionnel** : ce que fournit `yfinance` tel quel (jours fériés, splits, dividendes gérés par `auto_adjust=True`), sans traitement additionnel.
- **Le "mini-indice" de la section Dispersion est une construction simplifiée** (panier équipondéré de 5 actions indexées à 100), pas un vrai indice avec pondérations par capitalisation flottante et méthodologie de rebalancement.

## Tests

`tests/test_pricing.py` et `tests/test_greeks.py` valident, sur plusieurs jeux de paramètres (`tests/params.py`) : la parité call-put, la positivité des prix, la monotonicité en spot et en volatilité, les bornes du delta ([0,1] pour le call, [−1,0] pour le put), et le signe de gamma/vega/rho. Pour les lancer :

```
.venv/bin/python -m pytest tests/
```

## Interface interactive (Streamlit)

**App en ligne : [equity-derivatives-simulator-wgrj8rugktksm3x3qh4jpz.streamlit.app](https://equity-derivatives-simulator-wgrj8rugktksm3x3qh4jpz.streamlit.app)** (hébergement gratuit Streamlit Community Cloud — l'app peut mettre quelques secondes à se réveiller après une période d'inactivité).

`app.py`, à la racine du projet, expose 4 onglets pour explorer les paramètres en direct sans toucher au code : **Pricing & Greeks** (sliders S/K/T/r/σ/q, prix et Greeks recalculés en temps réel + graphique de comparaison sur 3 niveaux de vol), **Smile de vol implicite** (paramètres de la courbe synthétique ajustables), **Delta hedging** (choix du mode de rebalancement — intervalle fixe ou seuil —, P&L et graphique à 3 panneaux), **Dispersion** (sélection des tickers du panier, corrélation et vols recalculés via `yfinance`).

Pour la lancer en local :

```
.venv/bin/streamlit run app.py
```

L'app réutilise directement les fonctions de `src/` (aucune logique dupliquée) — les fonctions de tracé de `src/plots.py` acceptent `save_path=None` pour ne rien écrire sur disque en mode interactif, et retournent la figure matplotlib au lieu de simplement l'afficher/sauvegarder.
