# -*- coding: utf-8 -*-
"""
TWII T+20 模型最佳化系統 (Model Optimizer System - 20 Day)
自動搜索最佳超參數並訓練 20 日預測模型

功能：
- optimize 模式：Grid Search 自動搜索最佳超參數組合
- train 模式：使用最佳參數進行全量訓練
- predict 模式：載入最佳模型進行預測

預測策略：
- 使用 Direct Strategy（直接預測法）
- 模型輸出：第 20 個交易日後的 Adj Close

搜索參數範圍：
- Lookback: [30, 60, 90]
- LSTM Units: [64, 128, 256]
- Dropout Rate: [0.2, 0.3, 0.4]
- Batch Size: [32, 64]

使用方式：
  最佳化：python twii_model_optimizer_20d.py optimize --start 2019-07-01 --end 2025-12-12
  訓練：python twii_model_optimizer_20d.py train
  預測：python twii_model_optimizer_20d.py predict
"""

import argparse
import json
import pickle
import itertools
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import sys
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# =============================================================================
# 設定
# =============================================================================
models_dir_name = "saved_models_optimized_20d"
MODELS_DIR = Path(__file__).parent / models_dir_name
BEST_PARAMS_FILE = MODELS_DIR / "best_params.json"
CSV_FILE_PATH = Path(__file__).parent / "twii_data_from_2000_01_01.csv"
UPDATE_SCRIPT_PATH = Path(__file__).parent / "update_twii_data.py"

# 預測範圍
FORECAST_HORIZON = 20  # 預測未來第 20 個交易日

# 預設超參數（當沒有最佳參數時使用）
# 針對 T+20 的預設建議值 (基於 2025-12-13 Optimizer 最佳化結果)
DEFAULT_LOOKBACK = 60      # 鎖定 60 天
DEFAULT_LSTM_UNITS = 128   # 小模型 (128) 泛化能力優於 256
DEFAULT_DROPOUT_RATE = 0.5 # 最大正則化 (Maximum Regularization)
DEFAULT_BATCH_SIZE = 16    # 小批量有助於跳出局部最優
DEFAULT_EPOCHS = 50

# 最佳化搜索範圍 (可根據需求調整)
# 最佳化搜索範圍 (通用設定)
SEARCH_SPACE = {
    "lookback": [60, 90],
    "lstm_units": [128, 256],
    "dropout_rate": [0.3, 0.4, 0.5],
    "batch_size": [16, 32]
}

# 最佳化設定
OPTIMIZE_EPOCHS = 30  # 搜索時使用較少 epochs 加速
VALIDATION_RATIO = 0.2  # 驗證集比例

# 訓練設定
TRAIN_RATIO = 0.9
MODEL_STALE_DAYS = 180
MIN_TRAIN_DAYS = 1460

# 技術指標參數
KD_PARAMS = (9, 3, 3)
MACD_PARAMS = (12, 26, 9)
MIN_INDICATOR_DAYS = 50

# 預設訓練區間 (針對 T+20 測試後的最佳區間)
DEFAULT_START_DATE = "2019-07-01"
DEFAULT_END_DATE = "2025-12-12"

# 中文字型設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 自訂 Self-Attention Layer
# =============================================================================
class SelfAttention(layers.Layer):
    """
    Sequential Self-Attention Layer (論文 SSAM 架構)
    """
    
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.units = input_shape[-1]
        
        self.W_q = self.add_weight(
            name='W_query',
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.W_k = self.add_weight(
            name='W_key',
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.W_v = self.add_weight(
            name='W_value',
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(SelfAttention, self).build(input_shape)
    
    def call(self, inputs):
        Q = tf.matmul(inputs, self.W_q)
        K = tf.matmul(inputs, self.W_k)
        V = tf.matmul(inputs, self.W_v)
        
        attention_scores = tf.matmul(Q, K, transpose_b=True)
        d_k = tf.cast(self.units, tf.float32)
        attention_scores = attention_scores / tf.math.sqrt(d_k)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        output = tf.matmul(attention_weights, V)
        
        return output
    
    def get_config(self):
        config = super(SelfAttention, self).get_config()
        return config


# =============================================================================
# 特徵工程 (Feature Engineering)
# =============================================================================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    新增技術指標到 DataFrame
    
    新增欄位：
    - Volume_Log: 成交量（Log 轉換）
    - K: KD 指標的 K 值
    - D: KD 指標的 D 值
    - MACD_Hist: MACD 柱狀圖
    """
    df = df.copy()
    
    # Volume Log 轉換
    df['Volume_Log'] = np.log1p(df['Volume'])
    
    # KD 指標
    k_period, k_smooth, d_smooth = KD_PARAMS
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    raw_k = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = raw_k.rolling(window=k_smooth).mean()
    df['D'] = df['K'].rolling(window=d_smooth).mean()
    
    # MACD 指標
    fast_period, slow_period, signal_period = MACD_PARAMS
    ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    df['MACD_Hist'] = macd_line - signal_line
    
    # 移除 NaN
    original_len = len(df)
    df = df.dropna()
    removed_len = original_len - len(df)
    
    if removed_len > 0:
        print(f"[特徵工程] 已移除 {removed_len} 筆含 NaN 的資料")
    
    return df


# =============================================================================
# 模型架構（含 Dropout）
# =============================================================================
def build_lstm_ssam_model(
    time_steps: int,
    n_features: int,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT_RATE
) -> Model:
    """
    建立 LSTM + Dropout + Self-Attention 混合模型
    
    架構：
    Input -> LSTM -> Dropout -> Self-Attention -> Flatten -> Dense(1)
    
    Args:
        time_steps: 回看天數（動態調整）
        n_features: 輸入特徵數量
        lstm_units: LSTM 隱藏層單元數
        dropout_rate: Dropout 比率（防止過擬合）
    
    Returns:
        編譯好的 Keras 模型
    """
    inputs = layers.Input(shape=(time_steps, n_features), name='input_layer')
    
    # LSTM 層
    lstm_out = layers.LSTM(units=lstm_units, return_sequences=True, name='lstm_layer')(inputs)
    
    # Dropout 層（新增：防止過擬合）
    dropout_out = layers.Dropout(rate=dropout_rate, name='dropout_layer')(lstm_out)
    
    # Self-Attention 層
    attention_out = SelfAttention(name='self_attention')(dropout_out)
    
    # 輸出層
    flatten_out = layers.Flatten(name='flatten_layer')(attention_out)
    outputs = layers.Dense(units=1, activation='linear', name='output_layer')(flatten_out)
    
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_SSAM_Optimized_Model')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


# =============================================================================
# 資料處理
# =============================================================================
def run_update_script():
    """執行 update_twii_data.py 更新資料"""
    print("[系統] 嘗試呼叫外部腳本更新資料...")
    try:
        if not UPDATE_SCRIPT_PATH.exists():
            print(f"[警告] 找不到更新腳本: {UPDATE_SCRIPT_PATH}")
            return
        
        result = subprocess.run(
            [sys.executable, str(UPDATE_SCRIPT_PATH)],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            print("[系統] 資料更新程序執行完畢")
        else:
            print(f"[錯誤] 更新腳本執行失敗 (Return Code: {result.returncode})")
            print(result.stderr)
            
    except Exception as e:
        print(f"[錯誤] 呼叫更新腳本時發生例外: {e}")


def load_local_csv() -> pd.DataFrame:
    """
    讀取並格式化本地 CSV 資料
    """
    if not CSV_FILE_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        
        df['Date'] = pd.to_datetime(df['date'])
        df = df.set_index('Date').sort_index()
        
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
            
        return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
    except Exception as e:
        print(f"[錯誤] 讀取 CSV 失敗: {e}")
        return pd.DataFrame()


def download_data_by_date_range(start_date: str, end_date: str) -> pd.DataFrame:
    """
    取得指定日期範圍的 TWII 資料
    策略：優先讀取本地 CSV -> 若資料不足則自動更新 -> 重試 -> 若仍不足則報錯
    """
    print(f"[資料獲取] 正在讀取 ^TWII 資料 ({start_date} ~ {end_date})...")
    
    target_start = pd.Timestamp(start_date)
    target_end = pd.Timestamp(end_date)
    
    df = load_local_csv()
    
    today = pd.Timestamp.now().normalize()
    needs_update = False
    
    if df.empty:
        needs_update = True
    else:
        last_date = df.index[-1]
        if target_end > last_date and target_end <= today:
            needs_update = True
    
    if needs_update:
        print(f"[資料獲取] 資料庫資料不足 (最新: {df.index[-1].date() if not df.empty else '無'}), 嘗試更新...")
        run_update_script()
        df = load_local_csv()
    
    if not df.empty:
        mask = (df.index >= target_start) & (df.index <= target_end)
        df_filtered = df.loc[mask]
        
        if not df_filtered.empty:
            print(f"[資料獲取] 成功取得 {len(df_filtered)} 筆資料")
            print(f"[資料獲取] 實際期間：{df_filtered.index[0].strftime('%Y-%m-%d')} ~ {df_filtered.index[-1].strftime('%Y-%m-%d')}")
            return df_filtered
            
    raise ValueError(
        f"無法取得完整資料區間 ({start_date} ~ {end_date})。\n"
        f"本地資料範圍: {df.index[0].date() if not df.empty else '無'} ~ {df.index[-1].date() if not df.empty else '無'}\n"
        "請確認日期範圍是否正確，或檢查網路連線與更新腳本。"
    )


def download_recent_data(lookback_days: int = 100) -> pd.DataFrame:
    """
    取得最近的資料用於預測
    策略：自動嘗試更新至最新 -> 讀取 CSV -> 取最後 N 筆
    """
    required_days = lookback_days + MIN_INDICATOR_DAYS
    
    print(f"[資料獲取] 準備獲取最近 {required_days} 天的資料...")
    
    run_update_script()
    
    df = load_local_csv()
    
    if df.empty:
        raise ValueError("無法讀取本地資料庫 (CSV 為空)，請檢查 data 目錄")
    
    if len(df) < required_days:
        raise ValueError(f"歷史資料不足，僅有 {len(df)} 筆，需要 {required_days} 筆")
    
    df_recent = df.tail(required_days).copy()
    
    print(f"[資料獲取] 成功取得 {len(df_recent)} 筆資料 (最新日期: {df_recent.index[-1].date()})")
    
    return df_recent


def get_feature_columns() -> list:
    """取得特徵欄位名稱"""
    return ['Adj Close', 'Volume_Log', 'K', 'D', 'MACD_Hist']


def prepare_data(
    df: pd.DataFrame,
    lookback: int,
    forecast_horizon: int = FORECAST_HORIZON
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, int]:
    """
    準備訓練/驗證資料（Direct Strategy）
    
    資料對齊邏輯：
    - 輸入 X：時間點 [t-lookback, t) 的特徵
    - 目標 y：時間點 t+forecast_horizon-1 的 Adj Close
    
    Returns:
        X, y, feature_scaler, target_scaler, n_features
    """
    # 新增技術指標
    df = add_technical_indicators(df)
    
    # 確保有 Adj Close 欄位
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    
    # 準備特徵和目標
    feature_columns = get_feature_columns()
    features = df[feature_columns].values
    n_features = len(feature_columns)
    target = df['Adj Close'].values.reshape(-1, 1)
    
    # 建立縮放器
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)
    
    # 建立時序資料集（Direct Strategy）
    X, y = [], []
    max_idx = len(scaled_features) - forecast_horizon
    
    for i in range(lookback, max_idx):
        X.append(scaled_features[i - lookback:i])
        y.append(scaled_target[i + forecast_horizon - 1, 0])
    
    X, y = np.array(X), np.array(y)
    
    return X, y, feature_scaler, target_scaler, n_features


# =============================================================================
# 超參數最佳化（Grid Search）
# =============================================================================
def run_optimization(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    執行 Grid Search 超參數最佳化
    
    搜索範圍：
    - Lookback: [30, 60, 90]
    - LSTM Units: [64, 128]
    - Dropout Rate: [0.2, 0.3, 0.4]
    - Batch Size: [32, 64]
    
    使用 Time Series Split：最後 20% 資料作為驗證集
    
    Returns:
        最佳參數字典
    """
    print("\n" + "=" * 70)
    print("  TWII 模型最佳化系統 - Grid Search 超參數搜索")
    print("=" * 70)
    
    # 計算參數組合總數
    param_combinations = list(itertools.product(
        SEARCH_SPACE['lookback'],
        SEARCH_SPACE['lstm_units'],
        SEARCH_SPACE['dropout_rate'],
        SEARCH_SPACE['batch_size']
    ))
    total_combinations = len(param_combinations)
    
    print(f"\n[設定] 搜索參數範圍：")
    for key, values in SEARCH_SPACE.items():
        print(f"  - {key}: {values}")
    print(f"\n[設定] 總共 {total_combinations} 種參數組合")
    print(f"[設定] 每組合訓練 {OPTIMIZE_EPOCHS} epochs")
    print(f"[設定] 驗證集比例：{VALIDATION_RATIO * 100:.0f}%")
    
    # 下載資料
    df_raw = download_data_by_date_range(start_date, end_date)
    
    # 記錄所有結果
    results = []
    best_rmse = float('inf')
    best_params = None
    
    # 設定隨機種子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # ==========================================================================
    # Grid Search 迴圈
    # ==========================================================================
    for idx, (lookback, lstm_units, dropout_rate, batch_size) in enumerate(param_combinations):
        print(f"\n{'='*60}")
        print(f"[搜索] 組合 {idx + 1}/{total_combinations}")
        print(f"  Lookback: {lookback} | LSTM Units: {lstm_units}")
        print(f"  Dropout: {dropout_rate} | Batch Size: {batch_size}")
        print("=" * 60)
        
        try:
            # 1. 準備資料（根據當前 lookback 動態調整）
            X, y, feature_scaler, target_scaler, n_features = prepare_data(
                df_raw.copy(), lookback=lookback
            )
            
            # 2. 分割訓練集和驗證集（時間序列分割）
            val_size = int(len(X) * VALIDATION_RATIO)
            train_size = len(X) - val_size
            
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            print(f"[資料] 訓練集: {len(X_train)} | 驗證集: {len(X_val)}")
            
            # 3. 建立模型（根據當前參數動態調整）
            model = build_lstm_ssam_model(
                time_steps=lookback,
                n_features=n_features,
                lstm_units=lstm_units,
                dropout_rate=dropout_rate
            )
            
            # 4. 訓練（使用較少 epochs 加速搜索）
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=0
            )
            
            history = model.fit(
                X_train, y_train,
                epochs=OPTIMIZE_EPOCHS,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stop],
                verbose=0
            )
            
            # 5. 評估（計算驗證集 RMSE）
            y_pred_scaled = model.predict(X_val, verbose=0)
            y_val_actual = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            y_pred_actual = target_scaler.inverse_transform(y_pred_scaled).flatten()
            
            rmse = np.sqrt(mean_squared_error(y_val_actual, y_pred_actual))
            r2 = r2_score(y_val_actual, y_pred_actual)
            
            # 記錄結果
            result = {
                'lookback': lookback,
                'lstm_units': lstm_units,
                'dropout_rate': dropout_rate,
                'batch_size': batch_size,
                'val_rmse': float(rmse),
                'val_r2': float(r2),
                'epochs_trained': len(history.history['loss'])
            }
            results.append(result)
            
            print(f"[結果] Val RMSE: {rmse:.2f} | Val R²: {r2:.4f}")
            
            # 更新最佳參數
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = result.copy()
                print(f"  🏆 新的最佳組合！")
            
            # 清理 GPU 記憶體
            keras.backend.clear_session()
            
        except Exception as e:
            print(f"[錯誤] 組合失敗: {e}")
            continue
    
    # ==========================================================================
    # 輸出最佳化結果
    # ==========================================================================
    print("\n" + "=" * 70)
    print("📊 Grid Search 搜索完成")
    print("=" * 70)
    
    if best_params is None:
        print("❌ 沒有找到有效的參數組合")
        return None
    
    print(f"\n🏆 最佳參數組合：")
    print(f"  Lookback (回看天數)  : {best_params['lookback']}")
    print(f"  LSTM Units          : {best_params['lstm_units']}")
    print(f"  Dropout Rate        : {best_params['dropout_rate']}")
    print(f"  Batch Size          : {best_params['batch_size']}")
    print(f"  驗證集 RMSE         : {best_params['val_rmse']:.2f}")
    print(f"  驗證集 R²           : {best_params['val_r2']:.4f}")
    
    # 儲存最佳參數
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    best_params_output = {
        'best_params': {
            'lookback': best_params['lookback'],
            'lstm_units': best_params['lstm_units'],
            'dropout_rate': best_params['dropout_rate'],
            'batch_size': best_params['batch_size']
        },
        'validation_metrics': {
            'rmse': best_params['val_rmse'],
            'r2': best_params['val_r2']
        },
        'search_space': SEARCH_SPACE,
        'total_combinations_tested': len(results),
        'optimization_timestamp': datetime.now().isoformat(),
        'data_range': {
            'start': start_date,
            'end': end_date
        }
    }
    
    with open(BEST_PARAMS_FILE, 'w', encoding='utf-8') as f:
        json.dump(best_params_output, f, ensure_ascii=False, indent=2)
    
    print(f"\n[儲存] 最佳參數已儲存至：{BEST_PARAMS_FILE}")
    
    # 儲存完整搜索結果
    results_file = MODELS_DIR / "optimization_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'best_params': best_params_output
        }, f, ensure_ascii=False, indent=2)
    
    print(f"[儲存] 完整搜索結果已儲存至：{results_file}")
    
    return best_params_output


# =============================================================================
# 載入最佳參數
# =============================================================================
def load_best_params() -> Dict[str, Any]:
    """
    載入最佳參數，如果不存在則返回預設值
    """
    if BEST_PARAMS_FILE.exists():
        with open(BEST_PARAMS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[參數] 已載入最佳參數：{BEST_PARAMS_FILE}")
        return data['best_params']
    else:
        print("[參數] 未找到最佳參數檔案，使用預設值")
        return {
            'lookback': DEFAULT_LOOKBACK,
            'lstm_units': DEFAULT_LSTM_UNITS,
            'dropout_rate': DEFAULT_DROPOUT_RATE,
            'batch_size': DEFAULT_BATCH_SIZE
        }


# =============================================================================
# 模型成品管理
# =============================================================================
def get_artifact_paths(start_date: str, end_date: str) -> Tuple[Path, Path, Path, Path]:
    """取得模型成品路徑"""
    model_path = MODELS_DIR / f"model_{start_date}_{end_date}.keras"
    feature_scaler_path = MODELS_DIR / f"feature_scaler_{start_date}_{end_date}.pkl"
    target_scaler_path = MODELS_DIR / f"target_scaler_{start_date}_{end_date}.pkl"
    meta_path = MODELS_DIR / f"meta_{start_date}_{end_date}.json"
    return model_path, feature_scaler_path, target_scaler_path, meta_path


def save_artifacts(
    model,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    start_date: str,
    end_date: str,
    hyperparams: Dict[str, Any],
    price_min: float,
    price_max: float,
    n_features: int,
    rmse: float = None,
    r2: float = None
):
    """儲存模型成品（含超參數設定）"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path, feature_scaler_path, target_scaler_path, meta_path = get_artifact_paths(start_date, end_date)
    
    model.save(model_path)
    print(f"[儲存] 模型已儲存至：{model_path}")
    
    with open(feature_scaler_path, 'wb') as f:
        pickle.dump(feature_scaler, f)
    print(f"[儲存] 特徵縮放器已儲存至：{feature_scaler_path}")
    
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    print(f"[儲存] 目標縮放器已儲存至：{target_scaler_path}")
    
    metadata = {
        "model_type": "optimized_20day_direct",
        "train_start": start_date,
        "train_end": end_date,
        "forecast_horizon": FORECAST_HORIZON,
        "n_features": n_features,
        "feature_columns": get_feature_columns(),
        "price_min": price_min,
        "price_max": price_max,
        "training_timestamp": datetime.now().isoformat(),
        "hyperparameters": hyperparams,
        "technical_indicators": {
            "kd_params": list(KD_PARAMS),
            "macd_params": list(MACD_PARAMS)
        },
        "metrics": {
            "rmse": round(rmse, 2) if rmse is not None else None,
            "r2": round(r2, 4) if r2 is not None else None
        }
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[儲存] 元資料已儲存至：{meta_path}")


def load_artifacts(start_date: str, end_date: str) -> Tuple[Model, MinMaxScaler, MinMaxScaler, Dict[str, Any]]:
    """載入模型成品"""
    model_path, feature_scaler_path, target_scaler_path, meta_path = get_artifact_paths(start_date, end_date)
    
    model = keras.models.load_model(
        model_path,
        custom_objects={'SelfAttention': SelfAttention}
    )
    
    with open(feature_scaler_path, 'rb') as f:
        feature_scaler = pickle.load(f)
    
    with open(target_scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return model, feature_scaler, target_scaler, metadata


# =============================================================================
# 智慧模型選擇
# =============================================================================
def select_best_model(target_date: date) -> Optional[Dict[str, Any]]:
    """智慧選擇最適合的模型"""
    if not MODELS_DIR.exists():
        print("[搜尋] 模型目錄不存在")
        return None
    
    meta_files = list(MODELS_DIR.glob("meta_*.json"))
    if not meta_files:
        print("[搜尋] 找不到任何模型檔案")
        return None
    
    candidates = []
    
    for meta_file in meta_files:
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            train_start = datetime.strptime(metadata['train_start'], '%Y-%m-%d').date()
            train_end = datetime.strptime(metadata['train_end'], '%Y-%m-%d').date()
            
            duration_days = (train_end - train_start).days
            model_name = f"model_{metadata['train_start']}_{metadata['train_end']}"
            
            if duration_days < MIN_TRAIN_DAYS:
                continue
            
            if train_end < target_date:
                candidates.append({
                    'metadata': metadata,
                    'train_start': train_start,
                    'train_end': train_end,
                    'model_name': model_name,
                    'r2': metadata.get('metrics', {}).get('r2', 0.0) or 0.0
                })
        except Exception as e:
            continue
    
    if not candidates:
        print(f"[搜尋] 沒有符合條件的模型")
        return None
    
    candidates.sort(key=lambda x: (x['train_end'], x['r2'], x['train_start']), reverse=True)
    
    print(f"\n[搜尋] 找到 {len(candidates)} 個可用模型：")
    for i, c in enumerate(candidates):
        r2_display = f"R²: {c['r2']:.4f}" if c['r2'] else "R²: N/A"
        status = "✅ Selected" if i == 0 else "Backup"
        print(f"  {i+1}. {c['model_name']} ({r2_display}) -> {status}")
    
    return candidates[0]['metadata']


# =============================================================================
# 計算未來交易日
# =============================================================================
def get_future_trading_date(start_date: date, trading_days: int) -> date:
    """計算未來第 N 個交易日的日期"""
    current_date = start_date
    days_counted = 0
    
    while days_counted < trading_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            days_counted += 1
    
    return current_date


# =============================================================================
# 訓練結果視覺化
# =============================================================================
def plot_training_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    start_date: str,
    end_date: str,
    rmse: float,
    r2: float,
    hyperparams: Dict[str, Any]
) -> Path:
    """繪製訓練結果視覺化圖表"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_axis = range(len(y_true))
    ax.plot(x_axis, y_true, label='Actual (T+20)', color='blue', linewidth=1.5, alpha=0.8)
    ax.plot(x_axis, y_pred, label='Predicted (T+20)', color='red', linewidth=1.5, alpha=0.8)
    
    title = f"TWII Optimized 20-Day Forecast ({start_date} ~ {end_date}) | R²: {r2:.4f} | RMSE: {rmse:.2f}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('測試集樣本索引', fontsize=12)
    ax.set_ylabel('價格 (Price)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 超參數資訊
    textstr = (f"R² = {r2:.4f}\nRMSE = {rmse:.2f}\n"
               f"Lookback = {hyperparams['lookback']}\n"
               f"LSTM = {hyperparams['lstm_units']}\n"
               f"Dropout = {hyperparams['dropout_rate']}")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.97, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    plot_path = MODELS_DIR / f"plot_{start_date}_{end_date}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[視覺化] 訓練結果圖表已儲存至：{plot_path}")
    
    return plot_path


# =============================================================================
# 訓練模式
# =============================================================================
def train_mode(args):
    """使用最佳參數進行全量訓練"""
    print("\n" + "=" * 60)
    print("  TWII 模型最佳化系統 - 訓練模式")
    print("=" * 60)
    
    start_date = args.start
    end_date = args.end
    
    # 載入最佳參數
    params = load_best_params()
    lookback = params['lookback']
    lstm_units = params['lstm_units']
    dropout_rate = params['dropout_rate']
    batch_size = params['batch_size']
    
    print(f"\n[設定] 訓練期間：{start_date} ~ {end_date}")
    print(f"[設定] 使用超參數：")
    print(f"  Lookback: {lookback} | LSTM Units: {lstm_units}")
    print(f"  Dropout: {dropout_rate} | Batch Size: {batch_size}")
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 下載資料
    df = download_data_by_date_range(start_date, end_date)
    
    # 準備資料
    X, y, feature_scaler, target_scaler, n_features = prepare_data(df, lookback=lookback)
    
    # 分割訓練集和測試集
    train_size = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\n[預處理] 訓練集：{len(X_train)} 筆 | 測試集：{len(X_test)} 筆")
    
    # 建立模型
    print("\n[模型] 建立 LSTM-SSAM 最佳化模型...")
    model = build_lstm_ssam_model(
        time_steps=lookback,
        n_features=n_features,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate
    )
    model.summary()
    
    # 訓練
    print(f"\n[訓練] 開始訓練（{DEFAULT_EPOCHS} epochs）...")
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        epochs=DEFAULT_EPOCHS,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # 評估
    print("\n[評估] 計算測試集指標...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_predicted = target_scaler.inverse_transform(y_pred_scaled).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    r2 = r2_score(y_actual, y_predicted)
    
    # 取得價格範圍
    df_processed = add_technical_indicators(df)
    if 'Adj Close' not in df_processed.columns:
        df_processed['Adj Close'] = df_processed['Close']
    price_min = float(df_processed['Adj Close'].min())
    price_max = float(df_processed['Adj Close'].max())
    
    print("\n" + "=" * 50)
    print("📊 模型評估結果 (最佳化 20 日預測)")
    print("=" * 50)
    print(f"  RMSE (均方根誤差)  : {rmse:.2f} 點")
    print(f"  R² Score (決定係數): {r2:.4f}")
    print("=" * 50)
    
    # 儲存成品
    save_artifacts(
        model, feature_scaler, target_scaler,
        start_date, end_date,
        hyperparams=params,
        price_min=price_min, price_max=price_max,
        n_features=n_features,
        rmse=rmse, r2=r2
    )
    
    # 繪製圖表
    plot_training_results(y_actual, y_predicted, start_date, end_date, rmse, r2, params)
    
    print("\n✅ 訓練完成！模型成品已儲存至 saved_models_optimized/ 目錄")


# =============================================================================
# 預測模式
# =============================================================================
def predict_mode(args):
    """預測模式"""
    print("\n" + "=" * 60)
    print("  TWII 模型最佳化系統 - 預測模式")
    print("=" * 60)
    
    today = date.today()
    target_date = get_future_trading_date(today, FORECAST_HORIZON)
    
    print(f"\n[設定] 今日日期：{today}")
    print(f"[設定] 預測目標：未來第 {FORECAST_HORIZON} 個交易日 ({target_date})")
    
    # 選擇最佳模型（基於預測目標日期而非今天）
    print("\n[搜尋] 正在搜尋合適的模型...")
    metadata = select_best_model(target_date)
    
    if metadata is None:
        print(f"\n❌ 找不到適合的歷史模型。")
        print("   請先執行 optimize 和 train 指令。")
        return
    
    train_start = metadata['train_start']
    train_end = metadata['train_end']
    hyperparams = metadata.get('hyperparameters', load_best_params())
    lookback = hyperparams.get('lookback', DEFAULT_LOOKBACK)
    
    print(f"\n✅ 使用模型版本：{train_start} ~ {train_end}")
    print(f"   超參數：Lookback={lookback}, LSTM={hyperparams.get('lstm_units')}")
    
    # 載入模型
    print("\n[載入] 正在載入模型和縮放器...")
    model, feature_scaler, target_scaler, metadata = load_artifacts(train_start, train_end)
    
    # 下載資料
    df = download_recent_data(lookback_days=lookback + 20)
    
    # 預處理
    df_processed = add_technical_indicators(df)
    if 'Adj Close' not in df_processed.columns:
        df_processed['Adj Close'] = df_processed['Close']
    
    feature_columns = get_feature_columns()
    features = df_processed[feature_columns].values
    scaled_features = feature_scaler.transform(features)
    
    if len(scaled_features) < lookback:
        raise ValueError(f"資料不足")
    
    X = scaled_features[-lookback:].reshape(1, lookback, len(feature_columns))
    
    current_price = df_processed['Adj Close'].iloc[-1]
    last_data_date = df_processed.index[-1].date()
    
    # 預測
    print(f"\n[預測] 最近資料日期：{last_data_date}")
    print(f"[預測] 使用過去 {lookback} 天資料進行預測")
    
    y_pred_scaled = model.predict(X, verbose=0)
    predicted_price = target_scaler.inverse_transform(y_pred_scaled)[0, 0]
    
    predicted_date = get_future_trading_date(last_data_date, FORECAST_HORIZON)
    
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    trend = "📈 看漲" if price_change > 0 else "📉 看跌"
    
    # 輸出結果
    print("\n" + "=" * 55)
    print(f"🔮 TWII 20 日預測結果 (最佳化模型)")
    print("=" * 55)
    print(f"  最近收盤價 ({last_data_date})     : {current_price:.2f}")
    print(f"  預測價格   ({predicted_date}) : {predicted_price:.2f}")
    print(f"  預期變化                        : {price_change:+.2f} ({price_change_pct:+.2f}%)")
    print(f"  趨勢判斷                        : {trend}")
    print("=" * 55)
    print(f"  預測策略   : Direct Strategy（最佳化）")
    print(f"  回看天數   : {lookback} 天")
    print(f"  使用模型   : {train_start} ~ {train_end}")
    print("=" * 55)


# =============================================================================
# CLI 入口
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='TWII T+20 模型最佳化系統 - 自動搜索最佳超參數',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  超參數搜索：
    python twii_model_optimizer_20d.py optimize --start 2019-07-01 --end 2025-12-12
  
  使用最佳參數訓練：
    python twii_model_optimizer_20d.py train
  
  預測 20 個交易日後：
    python twii_model_optimizer_20d.py predict

搜索參數範圍：
  - Lookback: [60, 90]
  - LSTM Units: [128, 256]
  - Dropout Rate: [0.3, 0.4, 0.5]
  - Batch Size: [16, 32]
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='運作模式')
    
    # optimize 子命令
    optimize_parser = subparsers.add_parser('optimize', help='執行超參數搜索')
    optimize_parser.add_argument(
        '--start',
        type=str,
        default=DEFAULT_START_DATE,
        help=f'訓練資料起始日期 (預設: {DEFAULT_START_DATE})'
    )
    optimize_parser.add_argument(
        '--end',
        type=str,
        default=DEFAULT_END_DATE,
        help=f'訓練資料結束日期 (預設: {DEFAULT_END_DATE})'
    )
    
    # train 子命令
    train_parser = subparsers.add_parser('train', help='使用最佳參數訓練')
    train_parser.add_argument(
        '--start',
        type=str,
        default=DEFAULT_START_DATE,
        help=f'訓練資料起始日期 (預設: {DEFAULT_START_DATE})'
    )
    train_parser.add_argument(
        '--end',
        type=str,
        default=DEFAULT_END_DATE,
        help=f'訓練資料結束日期 (預設: {DEFAULT_END_DATE})'
    )
    
    # predict 子命令
    predict_parser = subparsers.add_parser('predict', help='預測 20 個交易日後的價格')
    
    args = parser.parse_args()
    
    if args.mode == 'optimize':
        run_optimization(args.start, args.end)
    elif args.mode == 'train':
        train_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
