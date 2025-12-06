# -*- coding: utf-8 -*-
"""
TWII 5 日預測模型註冊系統 (5-Day Forecast Model Registry System)
版本管理與自動模型選擇

功能：
- train 模式：使用多變量輸入訓練 LSTM-SSAM 模型，直接預測 5 個交易日後的收盤價
- predict 模式：智慧選擇合適模型進行 5 日後預測

預測策略：
- 使用 Direct Strategy（直接預測法），不使用遞迴預測
- 模型輸入：過去 30 天的特徵資料
- 模型輸出：第 5 個交易日後的 Adj Close

輸入特徵 (Features)：
- Adj Close: 調整後收盤價
- Volume (Log): 成交量（Log 轉換）
- K, D: KD 指標（9, 3, 3）
- MACD_Hist: MACD 柱狀圖（12, 26, 9）

使用方式：
  訓練：python twii_model_registry_5d.py train --start 2020-01-01 --end 2025-12-05
  預測：python twii_model_registry_5d.py predict
"""

import argparse
import json
import pickle
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

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
MODELS_DIR = Path(__file__).parent / "saved_models_5d"  # 5 日預測專用目錄
LOOKBACK = 30  # 回看天數（增加以捕捉更長趨勢）
FORECAST_HORIZON = 5  # 預測未來第 5 個交易日
LSTM_UNITS = 256
DROPOUT_RATE = 0.05  # Dropout 比率（防止過擬合）
EPOCHS = 50
BATCH_SIZE = 12
TRAIN_RATIO = 0.9
MODEL_STALE_DAYS = 180  # 模型過期警告閾值（天）
MIN_TRAIN_DAYS = 1460   # 最低訓練天數（4 年 = 4 × 365 = 1460 天）

# 技術指標參數
KD_PARAMS = (9, 3, 3)  # (K period, K smooth, D smooth)
MACD_PARAMS = (12, 26, 9)  # (快線, 慢線, 訊號線)

# 技術指標計算所需的最小資料筆數
MIN_INDICATOR_DAYS = 50  # 保守估計，確保指標穩定

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
    
    Args:
        df: 原始 OHLCV 資料
    
    Returns:
        包含技術指標的 DataFrame（已移除 NaN）
    """
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # 1. Volume Log 轉換
    # -------------------------------------------------------------------------
    df['Volume_Log'] = np.log1p(df['Volume'])
    
    # -------------------------------------------------------------------------
    # 2. KD 指標 (Stochastic Oscillator)
    # 參數：(K period, K smooth, D smooth) = (9, 3, 3)
    # -------------------------------------------------------------------------
    k_period, k_smooth, d_smooth = KD_PARAMS
    
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    
    raw_k = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = raw_k.rolling(window=k_smooth).mean()
    df['D'] = df['K'].rolling(window=d_smooth).mean()
    
    # -------------------------------------------------------------------------
    # 3. MACD 指標
    # 參數：(快線期數, 慢線期數, 訊號線期數) = (12, 26, 9)
    # -------------------------------------------------------------------------
    fast_period, slow_period, signal_period = MACD_PARAMS
    
    ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    df['MACD_Hist'] = macd_line - signal_line
    
    # -------------------------------------------------------------------------
    # 4. 移除 NaN
    # -------------------------------------------------------------------------
    original_len = len(df)
    df = df.dropna()
    removed_len = original_len - len(df)
    
    if removed_len > 0:
        print(f"[特徵工程] 已移除 {removed_len} 筆含 NaN 的資料（技術指標暖機期）")
    
    print(f"[特徵工程] 新增特徵：Volume_Log, K, D, MACD_Hist")
    print(f"[特徵工程] 最終資料筆數：{len(df)}")
    
    return df


# =============================================================================
# 模型架構
# =============================================================================
def build_lstm_ssam_model(
    time_steps: int = LOOKBACK, 
    n_features: int = 5, 
    lstm_units: int = LSTM_UNITS,
    dropout_rate: float = DROPOUT_RATE
):
    """
    建立 LSTM + Dropout + Self-Attention 混合模型（5 日預測版本）
    
    架構：Input -> LSTM -> Dropout -> Self-Attention -> Flatten -> Dense(1)
    
    Args:
        time_steps: 回看天數（預設 30）
        n_features: 輸入特徵數量（預設 5）
        lstm_units: LSTM 隱藏層單元數
        dropout_rate: Dropout 比率（防止過擬合）
    
    Returns:
        編譯好的 Keras 模型
    """
    inputs = layers.Input(shape=(time_steps, n_features), name='input_layer')
    
    # LSTM 層
    lstm_out = layers.LSTM(units=lstm_units, return_sequences=True, name='lstm_layer')(inputs)
    
    # Dropout 層（防止過擬合）
    dropout_out = layers.Dropout(rate=dropout_rate, name='dropout_layer')(lstm_out)
    
    # Self-Attention 層
    attention_out = SelfAttention(name='self_attention')(dropout_out)
    
    # 輸出層
    flatten_out = layers.Flatten(name='flatten_layer')(attention_out)
    outputs = layers.Dense(units=1, activation='linear', name='output_layer')(flatten_out)
    
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_SSAM_5Day_Model')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


# =============================================================================
# 資料處理
# =============================================================================
def download_data_by_date_range(start_date: str, end_date: str) -> pd.DataFrame:
    """下載指定日期範圍的 TWII 資料"""
    print(f"[資料獲取] 正在下載 ^TWII 資料 ({start_date} ~ {end_date})...")
    
    ticker = yf.Ticker("^TWII")
    df = ticker.history(start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError("無法取得 ^TWII 資料，請檢查網路連線或日期範圍")
    
    print(f"[資料獲取] 成功下載 {len(df)} 筆資料")
    print(f"[資料獲取] 實際期間：{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    
    return df


def download_recent_data(lookback_days: int = 60) -> pd.DataFrame:
    """下載最近的資料用於預測"""
    required_days = lookback_days + MIN_INDICATOR_DAYS
    
    print(f"[資料獲取] 正在下載最近 {required_days} 天的資料（含技術指標暖機期）...")
    
    ticker = yf.Ticker("^TWII")
    df = ticker.history(period=f"{required_days}d")
    
    if df.empty:
        raise ValueError("無法取得最近的 ^TWII 資料")
    
    print(f"[資料獲取] 成功下載 {len(df)} 筆資料")
    
    return df


def get_feature_columns() -> list:
    """取得特徵欄位名稱"""
    return ['Adj Close', 'Volume_Log', 'K', 'D', 'MACD_Hist']


def preprocess_for_training(df: pd.DataFrame, lookback: int = LOOKBACK, forecast_horizon: int = FORECAST_HORIZON, train_ratio: float = TRAIN_RATIO):
    """
    訓練用資料預處理（5 日預測版本 - Direct Strategy）
    
    資料對齊邏輯（Direct Strategy）：
    - 輸入 X：時間點 t-lookback 到 t 的特徵
    - 目標 y：時間點 t+forecast_horizon 的 Adj Close
    
    Returns:
        X_train, y_train, X_test, y_test, feature_scaler, target_scaler, price_min, price_max, n_features
    """
    # 1. 新增技術指標
    df = add_technical_indicators(df)
    
    # 2. 確保有 Adj Close 欄位
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
        print("[預處理] 使用 Close 欄位作為 Adj Close")
    
    # 3. 準備特徵矩陣和目標變數
    feature_columns = get_feature_columns()
    
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必要欄位：{col}")
    
    features = df[feature_columns].values
    n_features = len(feature_columns)
    target = df['Adj Close'].values.reshape(-1, 1)
    
    # 4. 建立雙縮放器
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)
    
    # 5. 建立時序資料集（Direct Strategy）
    # 注意：迴圈結束點需扣除 forecast_horizon，因為最後幾筆沒有未來的答案
    X, y = [], []
    max_idx = len(scaled_features) - forecast_horizon  # 可用資料的最後索引
    
    for i in range(lookback, max_idx):
        # X: 時間點 i-lookback 到 i-1 的特徵 (共 lookback 天)
        X.append(scaled_features[i - lookback:i])
        # y: 時間點 i + forecast_horizon - 1 的 Adj Close（即未來第 5 天）
        # 因為 i 是當前時間點的「下一天」，所以要加 forecast_horizon - 1
        # 實際上：i 代表的是 lookback 窗口結束後的第一天
        # 我們要預測的是從這天算起的第 forecast_horizon 天
        y.append(scaled_target[i + forecast_horizon - 1, 0])
    
    X, y = np.array(X), np.array(y)
    
    print(f"[預處理] Direct Strategy 資料對齊完成")
    print(f"[預處理] X 使用時間點 [t-{lookback}, t)，y 使用時間點 t+{forecast_horizon-1}")
    
    # 6. 分割訓練集與測試集
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 7. 記錄價格範圍
    price_min = float(df['Adj Close'].min())
    price_max = float(df['Adj Close'].max())
    
    print(f"[預處理] 輸入形狀：{X_train.shape} (samples, time_steps, n_features)")
    print(f"[預處理] 特徵數量：{n_features} ({', '.join(feature_columns)})")
    print(f"[預處理] 預測目標：未來第 {forecast_horizon} 個交易日")
    print(f"[預處理] 訓練集：{len(X_train)} 筆 | 測試集：{len(X_test)} 筆")
    print(f"[預處理] 價格範圍：{price_min:.2f} ~ {price_max:.2f}")
    
    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler, price_min, price_max, n_features


def preprocess_for_prediction(
    df: pd.DataFrame, 
    feature_scaler: MinMaxScaler, 
    lookback: int = LOOKBACK
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    預測用資料預處理
    
    Returns:
        X: 模型輸入 shape (1, lookback, n_features)
        df_processed: 處理後的 DataFrame
    """
    df_processed = add_technical_indicators(df)
    
    if 'Adj Close' not in df_processed.columns:
        df_processed['Adj Close'] = df_processed['Close']
    
    feature_columns = get_feature_columns()
    features = df_processed[feature_columns].values
    
    scaled_features = feature_scaler.transform(features)
    
    if len(scaled_features) < lookback:
        raise ValueError(f"資料不足，需要至少 {lookback} 筆資料，目前只有 {len(scaled_features)} 筆")
    
    X = scaled_features[-lookback:].reshape(1, lookback, len(feature_columns))
    
    return X, df_processed


# =============================================================================
# 訓練結果視覺化
# =============================================================================
def plot_training_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    start_date: str,
    end_date: str,
    rmse: float,
    r2: float
) -> Path:
    """繪製訓練結果視覺化圖表"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_axis = range(len(y_true))
    ax.plot(x_axis, y_true, label='Actual (T+5)', color='blue', linewidth=1.5, alpha=0.8)
    ax.plot(x_axis, y_pred, label='Predicted (T+5)', color='red', linewidth=1.5, alpha=0.8)
    
    title = f"TWII 5-Day Forecast ({start_date} ~ {end_date}) | R²: {r2:.4f} | RMSE: {rmse:.2f}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('測試集樣本索引', fontsize=12)
    ax.set_ylabel('價格 (Price)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\n5日直接預測'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.97, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    plot_path = MODELS_DIR / f"plot_{start_date}_{end_date}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[視覺化] 訓練結果圖表已儲存至：{plot_path}")
    
    return plot_path


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
    price_min: float,
    price_max: float,
    n_features: int,
    rmse: float = None,
    r2: float = None,
    lookback: int = LOOKBACK,
    forecast_horizon: int = FORECAST_HORIZON
):
    """儲存模型成品"""
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
        "model_type": "5day_direct",
        "train_start": start_date,
        "train_end": end_date,
        "lookback": lookback,
        "forecast_horizon": forecast_horizon,
        "n_features": n_features,
        "feature_columns": get_feature_columns(),
        "price_min": price_min,
        "price_max": price_max,
        "dropout_rate": DROPOUT_RATE,
        "lstm_units": LSTM_UNITS,
        "batch_size": BATCH_SIZE,
        "training_timestamp": datetime.now().isoformat(),
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
                print(f"[略過] 模型 {model_name} 訓練天數 {duration_days} 天不足 4 年 ({MIN_TRAIN_DAYS} 天)")
                continue
            
            if train_end < target_date:
                candidates.append({
                    'metadata': metadata,
                    'train_start': train_start,
                    'train_end': train_end,
                    'duration_days': duration_days,
                    'gap_days': (target_date - train_end).days,
                    'model_name': model_name,
                    'r2': metadata.get('metrics', {}).get('r2', 0.0) or 0.0
                })
        except Exception as e:
            print(f"[警告] 無法解析 {meta_file}: {e}")
            continue
    
    if not candidates:
        print(f"[搜尋] 沒有符合條件的模型（train_end < {target_date}）")
        return None
    
    candidates.sort(key=lambda x: (x['train_end'], x['r2'], x['train_start']), reverse=True)
    
    print(f"\n[搜尋] 找到 {len(candidates)} 個可用模型：")
    for i, c in enumerate(candidates):
        r2_display = f"R²: {c['r2']:.4f}" if c['r2'] else "R²: N/A"
        status = "Selected (Best Match)" if i == 0 else "Backup"
        print(f"  {i+1}. {c['model_name']} ({r2_display}) -> {status}")
    
    return candidates[0]['metadata']


def validate_model(metadata: Dict[str, Any], target_date: date, current_price: Optional[float] = None):
    """驗證模型並發出警告"""
    train_end = datetime.strptime(metadata['train_end'], '%Y-%m-%d').date()
    gap_days = (target_date - train_end).days
    
    if gap_days > MODEL_STALE_DAYS:
        print(f"\n⚠️ 警告：選擇的模型已訓練超過 {MODEL_STALE_DAYS} 天（距今 {gap_days} 天），建議重新訓練。")
    
    if current_price is not None:
        price_min = metadata.get('price_min', 0)
        price_max = metadata.get('price_max', float('inf'))
        
        if current_price < price_min or current_price > price_max:
            print(f"\n⚠️ 警告：當前價格 {current_price:.2f} 超出訓練時的價格範圍 [{price_min:.2f}, {price_max:.2f}]")


# =============================================================================
# 計算未來交易日
# =============================================================================
def get_future_trading_date(start_date: date, trading_days: int) -> date:
    """
    計算未來第 N 個交易日的日期（跳過週末）
    
    Args:
        start_date: 起始日期
        trading_days: 要前進的交易日數
    
    Returns:
        未來第 N 個交易日的日期
    """
    current_date = start_date
    days_counted = 0
    
    while days_counted < trading_days:
        current_date += timedelta(days=1)
        # 週一到週五才算交易日
        if current_date.weekday() < 5:
            days_counted += 1
    
    return current_date


# =============================================================================
# 訓練模式
# =============================================================================
def train_mode(args):
    """訓練模式"""
    print("\n" + "=" * 60)
    print("  TWII 5 日預測模型註冊系統 - 訓練模式")
    print("  (Direct Strategy - 直接預測法)")
    print("=" * 60)
    
    start_date = args.start
    end_date = args.end
    
    print(f"\n[設定] 訓練期間：{start_date} ~ {end_date}")
    print(f"[設定] Lookback: {LOOKBACK} | Forecast Horizon: {FORECAST_HORIZON}")
    print(f"[設定] LSTM Units: {LSTM_UNITS} | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}")
    print(f"[設定] KD 參數: {KD_PARAMS} | MACD 參數: {MACD_PARAMS}")
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 1. 下載資料
    df = download_data_by_date_range(start_date, end_date)
    
    # 2. 預處理（Direct Strategy）
    X_train, y_train, X_test, y_test, feature_scaler, target_scaler, price_min, price_max, n_features = preprocess_for_training(df)
    
    # 3. 建立模型
    print("\n[模型] 建立 LSTM-SSAM 5 日預測模型...")
    model = build_lstm_ssam_model(time_steps=LOOKBACK, n_features=n_features)
    model.summary()
    
    # 4. 訓練
    print(f"\n[訓練] 開始訓練...")
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # 5. 評估
    print("\n[評估] 計算測試集指標...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_predicted = target_scaler.inverse_transform(y_pred_scaled).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    r2 = r2_score(y_actual, y_predicted)
    
    print("\n" + "=" * 50)
    print("📊 模型評估結果 (5 日直接預測)")
    print("=" * 50)
    print(f"  RMSE (均方根誤差)  : {rmse:.2f} 點")
    print(f"  R² Score (決定係數): {r2:.4f}")
    print(f"  預測目標           : 未來第 {FORECAST_HORIZON} 個交易日")
    print("=" * 50)
    
    # 6. 儲存成品
    save_artifacts(
        model, feature_scaler, target_scaler, 
        start_date, end_date, 
        price_min, price_max, n_features,
        rmse=rmse, r2=r2
    )
    
    # 7. 繪製訓練結果圖表
    plot_training_results(y_actual, y_predicted, start_date, end_date, rmse, r2)
    
    print("\n✅ 訓練完成！模型成品已儲存至 saved_models_5d/ 目錄")


# =============================================================================
# 預測模式
# =============================================================================
def predict_mode(args):
    """預測模式 - 直接預測 5 個交易日後的收盤價"""
    print("\n" + "=" * 60)
    print("  TWII 5 日預測模型註冊系統 - 預測模式")
    print("  (Direct Strategy - 直接預測法)")
    print("=" * 60)
    
    # 計算今天和 5 個交易日後的日期
    today = date.today()
    target_date = get_future_trading_date(today, FORECAST_HORIZON)
    
    print(f"\n[設定] 今日日期：{today}")
    print(f"[設定] 預測目標：未來第 {FORECAST_HORIZON} 個交易日 ({target_date})")
    
    # 選擇最佳模型（基於今天的日期）
    print("\n[搜尋] 正在搜尋合適的模型...")
    metadata = select_best_model(today)
    
    if metadata is None:
        print(f"\n❌ 找不到適合的歷史模型。")
        print("   請先訓練模型。")
        print(f"   範例：python {Path(__file__).name} train --start 2020-01-01 --end {today - timedelta(days=1)}")
        return
    
    train_start = metadata['train_start']
    train_end = metadata['train_end']
    lookback = metadata.get('lookback', LOOKBACK)
    forecast_horizon = metadata.get('forecast_horizon', FORECAST_HORIZON)
    
    print(f"\n✅ 使用模型版本：訓練期間 {train_start} 至 {train_end}")
    print(f"   模型類型：{forecast_horizon} 日直接預測")
    print(f"   特徵欄位：{metadata.get('feature_columns', get_feature_columns())}")
    
    # 載入模型成品
    print("\n[載入] 正在載入模型和縮放器...")
    model, feature_scaler, target_scaler, metadata = load_artifacts(train_start, train_end)
    
    # 下載最近資料
    df = download_recent_data(lookback_days=lookback + 20)
    
    # 預處理
    X, df_processed = preprocess_for_prediction(df, feature_scaler, lookback)
    
    current_price = df_processed['Adj Close'].iloc[-1]
    last_data_date = df_processed.index[-1].date()
    
    # 驗證模型
    validate_model(metadata, today, current_price)
    
    # 直接預測（無需遞迴迴圈）
    print(f"\n[預測] 最近資料日期：{last_data_date}")
    print(f"[預測] 使用過去 {lookback} 天資料進行單次預測")
    
    y_pred_scaled = model.predict(X, verbose=0)
    predicted_price = target_scaler.inverse_transform(y_pred_scaled)[0, 0]
    
    # 計算預測目標日期（從最後資料日開始算 5 個交易日）
    predicted_date = get_future_trading_date(last_data_date, forecast_horizon)
    
    # 計算漲跌幅
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    trend = "📈 看漲" if price_change > 0 else "📉 看跌"
    
    # 輸出結果
    print("\n" + "=" * 55)
    print(f"🔮 TWII 5 日預測結果")
    print("=" * 55)
    print(f"  最近收盤價 ({last_data_date})     : {current_price:.2f}")
    print(f"  預測價格   ({predicted_date}) : {predicted_price:.2f}")
    print(f"  預期變化                        : {price_change:+.2f} ({price_change_pct:+.2f}%)")
    print(f"  趨勢判斷                        : {trend}")
    print("=" * 55)
    print(f"  預測策略   : Direct Strategy（直接預測）")
    print(f"  預測範圍   : 未來第 {forecast_horizon} 個交易日")
    print(f"  使用模型   : {train_start} ~ {train_end}")
    print(f"  回看天數   : {lookback} 天")
    print("=" * 55)


# =============================================================================
# CLI 入口
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='TWII 5 日預測模型註冊系統 - 直接預測法',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  訓練模型：
    python twii_model_registry_5d.py train --start 2020-01-01 --end 2024-01-01
  
  預測 5 個交易日後：
    python twii_model_registry_5d.py predict

預測策略：
  - 使用 Direct Strategy（直接預測法）
  - 模型輸入：過去 30 天的多變量特徵
  - 模型輸出：第 5 個交易日後的 Adj Close

輸入特徵（多變量）：
  - Adj Close: 調整後收盤價
  - Volume_Log: 成交量（Log 轉換）
  - K, D: KD 指標（9, 3, 3）
  - MACD_Hist: MACD 柱狀圖（12, 26, 9）
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='運作模式')
    
    # train 子命令
    train_parser = subparsers.add_parser('train', help='訓練新模型')
    train_parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='訓練資料起始日期 (YYYY-MM-DD)'
    )
    train_parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='訓練資料結束日期 (YYYY-MM-DD)'
    )
    
    # predict 子命令
    predict_parser = subparsers.add_parser('predict', help='預測 5 個交易日後的價格')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
