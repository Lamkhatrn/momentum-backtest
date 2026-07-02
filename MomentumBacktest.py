import numpy as np
import pandas as pd
import yfinance as yf

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

# 1. Téléchargement des données
print("Téléchargement des données...")
data = yf.download(TICKERS, start=DEBUT, end=FIN, auto_adjust=True)["Close"]
benchmark = yf.download(BENCHMARK, start=DEBUT, end=FIN, auto_adjust=True)["Close"]

data = data.dropna(how="all")
benchmark = benchmark.dropna()

print(f"\nNombre d'actions : {data.shape[1]}")
print(f"Période : {data.index[0].date()} -> {data.index[-1].date()}")
print(f"Nombre de jours : {len(data)}")

# 2. Calcul du signal momentum
# Performance de chaque action sur les 12 derniers mois
momentum_signal = data.pct_change(MOMENTUM)

print("\nAperçu du signal momentum (5 dernières dates) :")
print(momentum_signal.tail())

print("\nClassement momentum au 31/12/2025 :")
print(momentum_signal.loc["2025-12-31"].sort_values(ascending=False))


# 3. Construction des positions
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

# 4. Calcul des rendements
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


# 5. Métriques de performance
def calculer_metriques(rendements, nom):
    """
    Calcule les métriques clés d'une stratégie à partir de ses rendements journaliers.
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
metriques_strategie = calculer_metriques(rendements_strategie, "Stratégie Momentum")
metriques_benchmark = calculer_metriques(rendements_benchmark.squeeze(), "Benchmark CAC 40")