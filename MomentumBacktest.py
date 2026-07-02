import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Paramètres
# 10 grandes valeurs du CAC 40
TICKERS = [
    "AI.PA",   # Air Liquide
    "BNP.PA",  # BNP Paribas
    "MC.PA",   # LVMH
    "OR.PA",   # L'Oréal
    "SAN.PA",  # Sanofi
    "TTE.PA",  # TotalEnergies
    "SU.PA",   # Schneider Electric
    "KER.PA",  # Kering
    "ACA.PA",  # Crédit Agricole
    "GLE.PA"   # Société Générale
]

BENCHMARK = "^FCHI"       # CAC 40 comme référence
DEBUT = "2020-01-01"
FIN = "2026-01-01"
MOMENTUM = 252          # fenêtre momentum = 12 mois (252 jours de bourse)
TOP_N = 3            # on achète les 3 meilleurs

# Téléchargement des données
print("Téléchargement des données...")
data = yf.download(TICKERS, start=DEBUT, end=FIN, auto_adjust=True)["Close"]
benchmark = yf.download(BENCHMARK, start=DEBUT, end=FIN, auto_adjust=True)["Close"]

data = data.dropna(how="all")
benchmark = benchmark.dropna()

print(f"\nNombre d'actions : {data.shape[1]}")
print(f"Période : {data.index[0].date()} -> {data.index[-1].date()}")
print(f"Nombre de jours : {len(data)}")

# Calcul du signal momentum
# Performance de chaque action sur les 12 derniers mois
momentum_signal = data.pct_change(MOMENTUM)

print("\nAperçu du signal momentum (5 dernières dates) :")
print(momentum_signal.tail())

print("\nClassement momentum au 31/12/2025 :")
print(momentum_signal.loc["2025-12-31"].sort_values(ascending=False))


# Construction des positions
# On décale le signal d'un jour pour éviter le biais look-ahead
signal_decale = momentum_signal.shift(1)

def construire_positions(row):
    """
    Pour chaque jour, sélectionne les TOP_N actions avec le meilleur momentum
    et leur attribue un poids égal (1/TOP_N chacune).
    Les autres actions ont un poids de 0 (non détenues ce jour-là).
    Exemple : si TOP_N=3, alors 33% sur chacune des 3 meilleures actions.
    """
    valides = row.dropna()
    if len(valides) < TOP_N:
        return pd.Series(0, index=row.index)
    top = valides.nlargest(TOP_N).index
    poids = pd.Series(0.0, index=row.index)
    poids[top] = 1.0 / TOP_N
    return poids

# On applique la fonction sur chaque ligne (chaque jour)
# résultat : un tableau avec les poids de chaque action pour chaque jour
positions = signal_decale.apply(construire_positions, axis=1)

# Calcul des rendements
# Rendements journaliers réels de chaque action
rendements_actions = data.pct_change()

# Rendement journalier de notre stratégie = somme des (poids de chaque action * son rendement ce jour-là)
# Exemple : 33% * rendement BNP + 33% * rendement SG + 33% * rendement ACA
rendements_strategie = (positions * rendements_actions).sum(axis=1)

# Rendement journalier du benchmark (CAC 40) pour comparer
rendements_benchmark = benchmark.pct_change()

# Alignement des dates entre stratégie et benchmark
rendements_strategie = rendements_strategie.dropna()
rendements_benchmark = rendements_benchmark.reindex(rendements_strategie.index).dropna()
rendements_strategie = rendements_strategie.reindex(rendements_benchmark.index)

print("Aperçu des positions (5 dernières dates) :")
print(positions.tail())
print(f"\nNombre de jours avec positions actives : {(positions.sum(axis=1) > 0).sum()}")


# Indices de performance
def calculer_indices(rendements, nom):
    """
    Calcule les indices clés d'une stratégie à partir de ses rendements journaliers.
    - Rendement annualisé : performance moyenne sur 1 an
    - Volatilité annualisée : instabilité des rendements sur 1 an
    - Sharpe : rendement obtenu par unité de risque (plus c'est élevé, mieux c'est)
    - Drawdown max : pire perte depuis un pic (mesure le risque de ruine)
    """
    # 252 jours de bourse par an = facteur d'annualisation
    rendement_annuel = rendements.mean() * 252
    volatilite_annuelle = rendements.std() * np.sqrt(252)

    # Sharpe = rendement excédentaire / volatilité
    # On soustrait le taux sans risque (3% ici)
    taux_sans_risque = 0.03
    sharpe = (rendement_annuel - taux_sans_risque) / volatilite_annuelle

    # Drawdown max : on calcule la valeur du portefeuille jour par jour, puis on mesure la pire chute depuis un sommet
    valeur_portefeuille = (1 + rendements).cumprod()
    pic = valeur_portefeuille.cummax()
    drawdown = (valeur_portefeuille - pic) / pic
    drawdown_max = drawdown.min()

    print(f"\n{nom}")
    print(f"Rendement annualisé  : {rendement_annuel:.2%}")
    print(f"Volatilité annualisée : {volatilite_annuelle:.2%}")
    print(f"Ratio de Sharpe      : {sharpe:.2f}")
    print(f"Drawdown maximum     : {drawdown_max:.2%}")

    return {
        "rendement": rendement_annuel,
        "volatilite": volatilite_annuelle,
        "sharpe": sharpe,
        "drawdown_max": drawdown_max,
        "valeur": valeur_portefeuille
    }

# Calcul pour la stratégie et le benchmark
indices_strategie = calculer_indices(rendements_strategie, "Stratégie Momentum")
indices_benchmark = calculer_indices(rendements_benchmark.squeeze(), "Benchmark CAC 40")


# Visualisation

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Backtest Stratégie Momentum — CAC 40 (2020-2025)",
             fontsize=14, fontweight="bold")

# Graphique 1 : Performance cumulée stratégie vs benchmark
ax1 = axes[0, 0]
indices_strategie["valeur"].plot(ax=ax1, label="Stratégie Momentum",
                                    color="#4C72B0", linewidth=2)
indices_benchmark["valeur"].plot(ax=ax1, label="CAC 40",
                                    color="#E74C3C", linewidth=2)
ax1.set_title("Performance cumulée (1€ investi)")
ax1.set_ylabel("Valeur du portefeuille (€)")
ax1.legend()
ax1.axhline(1, color="gray", linestyle="--", linewidth=0.8)

# Graphique 2 : Drawdown stratégie vs benchmark
ax2 = axes[0, 1]
valeur_s = indices_strategie["valeur"]
valeur_b = indices_benchmark["valeur"]

drawdown_s = (valeur_s - valeur_s.cummax()) / valeur_s.cummax()
drawdown_b = (valeur_b - valeur_b.cummax()) / valeur_b.cummax()

drawdown_s.plot(ax=ax2, label="Stratégie Momentum", color="#4C72B0", linewidth=1.5)
drawdown_b.plot(ax=ax2, label="CAC 40", color="#E74C3C", linewidth=1.5)
ax2.fill_between(drawdown_s.index, drawdown_s, 0, alpha=0.2, color="#4C72B0")
ax2.set_title("Drawdown (chute depuis un pic)")
ax2.set_ylabel("Drawdown (%)")
ax2.legend()

# Graphique 3 : Distribution des rendements journaliers
ax3 = axes[1, 0]
rendements_strategie.hist(ax=ax3, bins=80, color="#4C72B0",
                           alpha=0.7, edgecolor="none", label="Stratégie")
rendements_benchmark.squeeze().hist(ax=ax3, bins=80, color="#E74C3C",
                                     alpha=0.5, edgecolor="none", label="CAC 40")
ax3.axvline(0, color="black", linewidth=1, linestyle="--")
ax3.set_title("Distribution des rendements journaliers")
ax3.set_xlabel("Rendement journalier")
ax3.set_ylabel("Fréquence")
ax3.legend()

# Graphique 4 : Tableau récapitulatif des indices
ax4 = axes[1, 1]
ax4.axis("off")

indices_table = [
    ["Indice", "Momentum", "CAC 40"],
    ["Rendement annualisé",
     f"{indices_strategie['rendement']:.2%}",
     f"{indices_benchmark['rendement']:.2%}"],
    ["Volatilité annualisée",
     f"{indices_strategie['volatilite']:.2%}",
     f"{indices_benchmark['volatilite']:.2%}"],
    ["Ratio de Sharpe",
     f"{indices_strategie['sharpe']:.2f}",
     f"{indices_benchmark['sharpe']:.2f}"],
    ["Drawdown maximum",
     f"{indices_strategie['drawdown_max']:.2%}",
     f"{indices_benchmark['drawdown_max']:.2%}"],
]

table = ax4.table(cellText=indices_table[1:],
                  colLabels=indices_table[0],
                  loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.3, 2.2)
ax4.set_title("Récapitulatif des performances")

plt.tight_layout()
plt.savefig("momentum_backtest.png", dpi=150, bbox_inches="tight")
print("\nGraphique sauvegardé : momentum_backtest.png")