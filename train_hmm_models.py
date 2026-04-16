import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import os
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_hmm_regimes(symbol, period="5y"):
    logging.info(f"--- Training HMM for {symbol} ---")
    
    # 1. Fetch Data
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. Feature Engineering
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['Range'] = (df['High'] - df['Low']) / df['Close']
    df['Return_ZScore'] = (df['Returns'] - df['Returns'].rolling(50).mean()) / df['Returns'].rolling(50).std()
    
    df = df.dropna()
    
    # 3. Prepare Features
    feature_cols = ['Returns', 'Volatility', 'Range', 'Return_ZScore']
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train HMM
    n_states = 3
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X_scaled)
    
    # 5. Predict and Label States
    states = model.predict(X_scaled)
    df['Regime'] = states
    
    # Sort states by volatility to assign consistent labels
    # Rank 0: Low Vol, Rank 1: Med Vol, Rank 2: High Vol
    vol_means = df.groupby('Regime')['Volatility'].mean().sort_values()
    state_map = {old: new for new, old in enumerate(vol_means.index)}
    df['Regime'] = df['Regime'].map(state_map)
    
    regime_labels = {0: "LOW_VOL", 1: "MEDIUM_VOL", 2: "HIGH_VOL"}
    
    # 6. Save Model
    model_filename = f"hmm_model_{symbol.replace('=X', '')}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'state_map': state_map,
            'regime_labels': regime_labels
        }, f)
    
    logging.info(f"Model saved to {model_filename}")
    
    # 7. Analysis Output
    for state, label in regime_labels.items():
        mask = df['Regime'] == state
        count = mask.sum()
        avg_ret = df.loc[mask, 'Returns'].mean() * 100
        avg_vol = df.loc[mask, 'Volatility'].mean() * 100
        logging.info(f"State {state} ({label}): {count} days, Avg Ret: {avg_ret:.3f}%, Avg Vol: {avg_vol:.2f}%")
        
    return df

if __name__ == "__main__":
    symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
    for sym in symbols:
        try:
            train_hmm_regimes(sym)
        except Exception as e:
            logging.error(f"Error training {sym}: {e}")
