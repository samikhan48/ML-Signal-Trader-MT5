# =============================================================================
# COLAB NOTEBOOK V5: ADVANCED TRIPLE-BARRIER DYNAMIC ONNX
# =============================================================================
# This script represents the V5 institutional upgrade. 
# It trains LightGBM models for ANY SYMBOL using the Triple-Barrier Method.
# Instead of a rigid time limit, it mathematically simulates an MT5 ATR-based 
# Stop Loss and Take Profit, training the AI to find trades that hit 
# TP before SL within 48 hours.
#
# USAGE:
# 1. Upload your mt5_data_*_H1.csv files to Colab
# 2. Run this entire script
# 3. Download the generated ONNX files
# 4. Copy to MT5 Common Files folder: MQL5\Files\Common\ML_ONNX_Models\
# =============================================================================

# Install dependencies (Uncomment when running in Colab)
# !pip install lightgbm onnxmltools skl2onnx onnx scikit-learn pandas numpy -q

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx
from onnx import numpy_helper
import warnings
warnings.filterwarnings('ignore')

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# =============================================================================
# CONFIGURATION - V5 TRIPLE BARRIER
# =============================================================================
FEATURE_COUNT = 21  # 20 Technical + 1 Macro Proxy (VIX Z-Score)

# Triple Barrier Parameters (Mirrors your MT5 EA Logic)
TP_ATR_MULTIPLIER = 2.0      # Upper Barrier (Take Profit distance)
SL_ATR_MULTIPLIER = 1.0      # Lower Barrier (Stop Loss distance)
MAX_HOLDING_BARS = 48        # Time Barrier (If neither hit in 48 hours = Loss)

# =============================================================================
# FEATURE ENGINEERING & TRIPLE-BARRIER LABELING
# =============================================================================
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_adx(df, window=14):
    df = df.copy()
    df['tr'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                     abs(df['low'] - df['close'].shift(1))))
    
    df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                             np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                              np.maximum(df['low'].shift(1) - df['low'], 0), 0)
    
    alpha = 1 / window
    df['atr'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
    df['dm_plus_smooth'] = df['dm_plus'].ewm(alpha=alpha, adjust=False).mean()
    df['dm_minus_smooth'] = df['dm_minus'].ewm(alpha=alpha, adjust=False).mean()
    
    df['di_plus'] = 100 * (df['dm_plus_smooth'] / df['atr'])
    df['di_minus'] = 100 * (df['dm_minus_smooth'] / df['atr'])
    
    df['dx'] = 100 * np.abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
    df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
    
    return df['adx']

def calculate_atr(df, window=14):
    df = df.copy()
    df['tr'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                     abs(df['low'] - df['close'].shift(1))))
    return df['tr'].rolling(window=window).mean()

def apply_triple_barrier(df):
    """
    Applies the Triple-Barrier Method to organically discover winning trades.
    1. Upper Barrier = Dynamic Take Profit (ATR based)
    2. Lower Barrier = Dynamic Stop Loss (ATR based)
    3. Time Barrier = Maximum bars to hold trade
    """
    print("  Applying Triple-Barrier Method for Institutional Labeling...")
    df = df.copy()
    targets = []
    
    # Pre-extract arrays for speed
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atrs = df['ATR'].values
    
    for i in range(len(df)):
        if i + MAX_HOLDING_BARS >= len(df) or np.isnan(atrs[i]):
            targets.append(np.nan)
            continue
            
        entry_price = closes[i]
        current_atr = atrs[i]
        
        # Calculate dynamic barriers (simulating your MT5 EA exact logic)
        take_profit = entry_price + (current_atr * TP_ATR_MULTIPLIER)
        stop_loss = entry_price - (current_atr * SL_ATR_MULTIPLIER)
        
        target = 0 # Default to 0 (Hit SL or Time expired = Loss)
        
        # Look forward through the time barrier
        for j in range(1, MAX_HOLDING_BARS + 1):
            future_high = highs[i+j]
            future_low = lows[i+j]
            
            # Did it hit the Lower Barrier (Stop Loss) first?
            if future_low <= stop_loss:
                target = 0
                break
                
            # Did it hit the Upper Barrier (Take Profit) first?
            if future_high >= take_profit:
                target = 1  # Winner!
                break
                
        targets.append(target)
        
    df['Target'] = targets
    return df.dropna(subset=['Target'])

def add_features(df, symbol, proxy_df=None):
    df = df.copy()
    
    # Time features
    if 'time' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
            
        # Strip timezone if present to standardise
        if df['time'].dt.tz is not None:
             df['time'] = df['time'].dt.tz_localize(None)
             
        df['Hour'] = df['time'].dt.hour
        df['DayOfWeek'] = df['time'].dt.dayofweek
    else:
        df['Hour'] = 12
        df['DayOfWeek'] = 2
    
    # Technical indicators
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['close'], window=14)
    df['ADX'] = calculate_adx(df, window=14)
    
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['EMA_200_Position'] = (df['close'] > df['EMA_200']).astype(float)
    
    df['ATR'] = calculate_atr(df, window=14)
    df['ATR_50'] = calculate_atr(df, window=50)
    df['ATR_Ratio'] = df['ATR'] / df['ATR_50']
    
    # Volume features
    vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
    df['Volume_MA_20'] = df[vol_col].rolling(window=20).mean()
    df['Volume_Ratio'] = df[vol_col] / df['Volume_MA_20']
    
    df['Spread_Normalized'] = (df['high'] - df['low']) / df['ATR']
    df['Returns'] = df['close'].pct_change()
    
    for lag in range(1, 4):
        df[f'Return_Lag_{lag}'] = df['Returns'].shift(lag)
        df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
        
    df['High_Low_Diff'] = df['high'] - df['low']
    df['Close_Open_Diff'] = df['close'] - df['open']
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # ==========================================
    # PHASE 7: MACRO CONTEXT INJECTION (21st Feature)
    # ==========================================
    if proxy_df is not None and not proxy_df.empty:
        print("    -> Injecting Macro Proxy Context (VIX/GOLD)...")
        # Ensure proxy_df has 'Date' and 'Close'
        if 'Date' in proxy_df.columns:
            proxy_df['Date'] = pd.to_datetime(proxy_df['Date'])
            if proxy_df['Date'].dt.tz is not None:
                proxy_df['Date'] = proxy_df['Date'].dt.tz_localize(None)
        
        # Merge exactly on the hour
        df = pd.merge(df, proxy_df[['Date', 'Close']], left_on='time', right_on='Date', how='left')
        df.rename(columns={'Close': 'Proxy_Close'}, inplace=True)
        
        # Calculate Z-Score of Proxy to measure contextual panic/euphoria
        df['Proxy_Close'].fillna(method='ffill', inplace=True) 
        df['Proxy_ZScore'] = (df['Proxy_Close'] - df['Proxy_Close'].rolling(50).mean()) / df['Proxy_Close'].rolling(50).std()
        df['Proxy_ZScore'].fillna(0, inplace=True)
        # Drop temporary merge column
        df.drop(columns=['Date', 'Proxy_Close'], inplace=True, errors='ignore')
    else:
        # Fallback to 0 if no proxy provided
        df['Proxy_ZScore'] = 0.0
    
    df = df.dropna()
    
    # V5 Upgrade: Replace static pip lookahead with Triple-Barrier Method
    df = apply_triple_barrier(df)
    df['Target'] = df['Target'].astype(int)
    
    return df

# =============================================================================
# ONNX MQL5 COMPATIBILITY FIX
# =============================================================================
def fix_onnx_for_mql5(model_path, feature_count):
    print("  Fixing ONNX for MQL5 compatibility...")
    model = onnx.load(model_path)
    
    zipmap_node = None
    for node in model.graph.node:
        if node.op_type == "ZipMap":
            zipmap_node = node
            break
            
    if zipmap_node:
        prob_tensor_name = zipmap_node.input[0]
        zipmap_output_name = zipmap_node.output[0]
        for out in model.graph.output:
            if out.name == zipmap_output_name:
                out.name = prob_tensor_name
        model.graph.node.remove(zipmap_node)
    
    new_initializers = []
    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.DOUBLE:
            arr = numpy_helper.to_array(init)
            arr32 = arr.astype(np.float32)
            new_init = numpy_helper.from_array(arr32, init.name)
            new_initializers.append(new_init)
        else:
            new_initializers.append(init)
            
    while len(model.graph.initializer) > 0:
        model.graph.initializer.pop()
    model.graph.initializer.extend(new_initializers)
    
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == onnx.TensorProto.DOUBLE:
                    attr.i = onnx.TensorProto.FLOAT
    
    for output in model.graph.output:
        output_name = output.name.lower()
        while len(output.type.tensor_type.shape.dim) > 0:
            output.type.tensor_type.shape.dim.pop()
        if 'label' in output_name:
            output.type.tensor_type.elem_type = onnx.TensorProto.INT64
            dim = output.type.tensor_type.shape.dim.add()
            dim.dim_value = 1
        else:
            output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            dim1 = output.type.tensor_type.shape.dim.add()
            dim1.dim_value = 1
            dim2 = output.type.tensor_type.shape.dim.add()
            dim2.dim_value = 3  # MT5 expects 3 class probs for LightGBM
            
    for node in model.graph.node:
        if node.op_type == 'TreeEnsembleClassifier':
            for attr in node.attribute:
                if attr.name == 'classlabels_int64s':
                    if len(list(attr.ints)) == 2:
                        attr.ints[:] = [0, 1, 2]
    
    onnx.save(model, model_path)
    print("    - Model fixed and saved!")

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_symbol(csv_file, symbol, proxy_df=None):
    print("\n" + "=" * 60)
    print(f"🏆 TRAINING V5 TRIPLE-BARRIER: {symbol}")
    print("=" * 60)
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.lower()
    df.rename(columns={'date': 'time', 'vol': 'volume', 'vol.': 'volume'}, inplace=True)
    if 'tick_volume' not in df.columns and 'volume' in df.columns:
        df['tick_volume'] = df['volume']
        
    print(f"📊 Loaded {len(df)} bars. Processing features...")
    df_processed = add_features(df, symbol, proxy_df=proxy_df)
    
    feature_cols = [
        'RSI', 'SMA_10', 'SMA_50', 'ADX', 'Returns', 'Volatility', 
        'High_Low_Diff', 'Close_Open_Diff', 'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3',
        'RSI_Lag_1', 'RSI_Lag_2', 'RSI_Lag_3', 'Hour', 'DayOfWeek',
        'EMA_200_Position', 'ATR_Ratio', 'Volume_Ratio', 'Spread_Normalized',
        'Proxy_ZScore'  # Phase 7: Macro Context Feature
    ]
    
    X = df_processed[feature_cols].values.astype(np.float32)
    y = df_processed['Target'].values
    
    print(f"📈 Samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"🎯 Win/Loss Distribution from Triple-Barrier: Losers={sum(y==0)}, Winners={sum(y==1)}")
    
    if sum(y==1) == 0 or sum(y==0) == 0:
        raise ValueError(f"CRITICAL ERROR: One of the classes has 0 samples for {symbol}. Adjust Barrier Parameters.")

    tscv = TimeSeriesSplit(n_splits=5)
    scale_pos = sum(y==0) / max(sum(y==1), 1)
    
    lgb_params = {
        'n_estimators': 200,
        'learning_rate': 0.02,
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 40,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos, # Balances out the naturally lower number of winners
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1,
        'random_state': 42
    }
    
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        model_cv = lgb.LGBMClassifier(**lgb_params)
        model_cv.fit(X[train_idx], y[train_idx])
        preds = model_cv.predict(X[test_idx])
        cv_scores.append(accuracy_score(y[test_idx], preds))
    
    print(f"📊 CV Accuracy: {np.mean(cv_scores):.2%} (+/- {np.std(cv_scores):.2%})")
    
    final_model = lgb.LGBMClassifier(**lgb_params)
    final_model.fit(X, y)
    
    onnx_path = f"ml_strategy_v4_{symbol}.onnx"  # Kept v4 nomenclature so MT5 reads it natively
    initial_type = [('float_input', FloatTensorType([None, FEATURE_COUNT]))]
    onx = onnxmltools.convert_lightgbm(final_model, initial_types=initial_type, target_opset=12)
    
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    
    fix_onnx_for_mql5(onnx_path, FEATURE_COUNT)
    return onnx_path

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 INITIALIZING V5 TRIPLE-BARRIER TRAINING")
    print("=" * 60)
    print(f"Barrier Settings: TP = {TP_ATR_MULTIPLIER}x ATR | SL = {SL_ATR_MULTIPLIER}x ATR | Timeout = {MAX_HOLDING_BARS}h")
    
    trained_models = []
    proxy_df = None
    
    import os
    if os.path.exists("mt5_data_VIX_H1.csv"):
        print("📥 Found VIX Macro Proxy Data! Loading...")
        proxy_df = pd.read_csv("mt5_data_VIX_H1.csv")
        
    if IN_COLAB:
        uploaded = files.upload()
        if "mt5_data_VIX_H1.csv" in uploaded.keys():
             proxy_df = pd.read_csv("mt5_data_VIX_H1.csv")
             
        for filename in uploaded.keys():
            if "VIX" in filename or "GOLD" in filename: continue
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 3:
                symbol = parts[2].upper()
                try:
                    onnx_path = train_symbol(filename, symbol, proxy_df=proxy_df)
                    trained_models.append(onnx_path)
                except Exception as e:
                    print(f"[ERROR] Failed to train {symbol}: {e}")
                    
        for model_path in trained_models:
            files.download(model_path)
    else:
        print("Note: Run this script inside Google Colab (or ensure CSV files are in the same directory locally).")
        import glob
        csv_files = glob.glob('mt5_data_*_H1.csv')
        for filename in csv_files:
            if "VIX" in filename or "GOLD" in filename: continue
            parts = filename.replace('.csv', '').split('_')
            symbol = parts[2].upper()
            try:
                train_symbol(filename, symbol, proxy_df=proxy_df)
            except Exception as e:
                print(f"[ERROR] Failed: {e}")
