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