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