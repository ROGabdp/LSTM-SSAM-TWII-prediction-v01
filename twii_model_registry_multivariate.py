# -*- coding: utf-8 -*-
"""
TWII 多變量模型註冊系統 (Multivariate Model Registry System)
版本管理與自動模型選擇

功能：
- train 模式：使用多變量輸入（技術指標）訓練 LSTM-SSAM 模型並儲存成品
- predict 模式：智慧選擇合適模型進行預測

輸入特徵 (Features)：
- Adj Close: 調整後收盤價
- Volume (Log): 成交量（Log 轉換）
- K, D: KD 指標（9, 3, 3）
- MACD_Hist: MACD 柱狀圖（12, 26, 9）

使用方式：
  訓練：python twii_model_registry_multivariate.py train --start 2020-07-01 --end 2025-12-05
  預測：python twii_model_registry_multivariate.py predict --target_date 2024-12-10
  預測（明天）：python twii_model_registry_multivariate.py predict
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
MODELS_DIR = Path(__file__).parent / "saved_models_multivariate"
LOOKBACK = 10  # 回看天數（論文規格）
LSTM_UNITS = 50
EPOCHS = 50
BATCH_SIZE = 10
TRAIN_RATIO = 0.9
MODEL_STALE_DAYS = 180  # 模型過期警告閾值（天）
MIN_TRAIN_DAYS = 1460   # 最低訓練天數（4 年 = 4 × 365 = 1460 天）

# 技術指標參數
KD_PARAMS = (9, 3, 3)  # (K period, K smooth, D smooth)
MACD_PARAMS = (12, 26, 9)  # (快線, 慢線, 訊號線)

# 技術指標計算所需的最小資料筆數
# MACD 需要 26 日慢線 + 9 日訊號線 = 至少 35 天
# KD 需要 9 + 3 + 3 = 15 天
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
    # 使用 log1p 避免 log(0) 的問題
    df['Volume_Log'] = np.log1p(df['Volume'])
    
    # -------------------------------------------------------------------------
    # 2. KD 指標 (Stochastic Oscillator)
    # 參數：(K period, K smooth, D smooth) = (9, 3, 3)
    # -------------------------------------------------------------------------
    k_period, k_smooth, d_smooth = KD_PARAMS
    
    # 計算最低價和最高價的滾動窗口
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    
    # 計算 Raw Stochastic (%K 原始值)
    # %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    raw_k = (df['Close'] - low_min) / (high_max - low_min) * 100
    
    # 平滑 K 值（使用 SMA）
    df['K'] = raw_k.rolling(window=k_smooth).mean()
    
    # D 值是 K 值的 SMA
    df['D'] = df['K'].rolling(window=d_smooth).mean()
    
    # -------------------------------------------------------------------------
    # 3. MACD 指標
    # 參數：(快線期數, 慢線期數, 訊號線期數) = (12, 26, 9)
    # -------------------------------------------------------------------------
    fast_period, slow_period, signal_period = MACD_PARAMS
    
    # 計算 EMA（指數移動平均）
    ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
    
    # MACD 線 = 快線 EMA - 慢線 EMA
    macd_line = ema_fast - ema_slow
    
    # 訊號線 = MACD 線的 EMA
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # MACD 柱狀圖 = MACD 線 - 訊號線
    df['MACD_Hist'] = macd_line - signal_line
    
    # -------------------------------------------------------------------------
    # 4. 移除 NaN（技術指標計算初期會產生空值）
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
def build_lstm_ssam_model(time_steps: int = LOOKBACK, n_features: int = 5, lstm_units: int = LSTM_UNITS):
    """
    建立 LSTM + Self-Attention 混合模型（多變量版本）
    
    Args:
        time_steps: 回看天數
        n_features: 輸入特徵數量（預設 5：Adj Close, Volume_Log, K, D, MACD_Hist）
        lstm_units: LSTM 隱藏層單元數
    
    Returns:
        編譯好的 Keras 模型
    """
    inputs = layers.Input(shape=(time_steps, n_features), name='input_layer')
    lstm_out = layers.LSTM(units=lstm_units, return_sequences=True, name='lstm_layer')(inputs)
    attention_out = SelfAttention(name='self_attention')(lstm_out)
    flatten_out = layers.Flatten(name='flatten_layer')(attention_out)
    outputs = layers.Dense(units=1, activation='linear', name='output_layer')(flatten_out)
    
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_SSAM_Multivariate_Model')
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


def download_recent_data(lookback_days: int = 30) -> pd.DataFrame:
    """下載最近的資料用於預測（需額外下載計算技術指標所需的歷史資料）"""
    # 計算需要的總資料量：lookback + 技術指標暖機期
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


def preprocess_for_training(df: pd.DataFrame, lookback: int = LOOKBACK, train_ratio: float = TRAIN_RATIO):
    """
    訓練用資料預處理（多變量版本，使用雙縮放器策略）
    
    策略說明：
    - feature_scaler: 正規化所有輸入特徵（X），用於模型輸入
    - target_scaler: 專門正規化目標欄位（Adj Close），用於還原預測結果
    
    Returns:
        X_train, y_train, X_test, y_test, feature_scaler, target_scaler, price_min, price_max, n_features
    """
    # 1. 新增技術指標
    df = add_technical_indicators(df)
    
    # 2. 確保有 Adj Close 欄位（yfinance 預設會包含）
    if 'Adj Close' not in df.columns:
        # 如果沒有 Adj Close，使用 Close 代替
        df['Adj Close'] = df['Close']
        print("[預處理] 使用 Close 欄位作為 Adj Close")
    
    # 3. 準備特徵矩陣和目標變數
    feature_columns = get_feature_columns()
    
    # 確認所有欄位都存在
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必要欄位：{col}")
    
    # 特徵矩陣 shape: (samples, n_features)
    features = df[feature_columns].values
    n_features = len(feature_columns)
    
    # 目標變數 shape: (samples, 1)
    target = df['Adj Close'].values.reshape(-1, 1)
    
    # 4. 建立雙縮放器
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 擬合並轉換
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)
    
    # 5. 建立時序資料集
    X, y = [], []
    for i in range(lookback, len(scaled_features)):
        # X: 過去 lookback 天的所有特徵 shape: (lookback, n_features)
        X.append(scaled_features[i - lookback:i])
        # y: 目標日的 Adj Close（已縮放）
        y.append(scaled_target[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # 6. 分割訓練集與測試集
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # X 形狀已經是 (samples, time_steps, n_features)，不需要 reshape
    
    # 7. 記錄價格範圍（使用原始 Adj Close）
    price_min = float(df['Adj Close'].min())
    price_max = float(df['Adj Close'].max())
    
    print(f"[預處理] 輸入形狀：{X_train.shape} (samples, time_steps, n_features)")
    print(f"[預處理] 特徵數量：{n_features} ({', '.join(feature_columns)})")
    print(f"[預處理] 訓練集：{len(X_train)} 筆 | 測試集：{len(X_test)} 筆")
    print(f"[預處理] 價格範圍：{price_min:.2f} ~ {price_max:.2f}")
    
    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler, price_min, price_max, n_features


def preprocess_for_prediction(
    df: pd.DataFrame, 
    feature_scaler: MinMaxScaler, 
    lookback: int = LOOKBACK
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    預測用資料預處理（多變量版本）
    
    Args:
        df: 原始資料（需包含足夠的歷史資料計算技術指標）
        feature_scaler: 訓練時擬合的特徵縮放器
        lookback: 回看天數
    
    Returns:
        X: 模型輸入 shape (1, lookback, n_features)
        df_processed: 處理後的 DataFrame（用於取得最後日期等資訊）
    """
    # 1. 計算技術指標
    df_processed = add_technical_indicators(df)
    
    # 2. 確保有 Adj Close 欄位
    if 'Adj Close' not in df_processed.columns:
        df_processed['Adj Close'] = df_processed['Close']
    
    # 3. 準備特徵矩陣
    feature_columns = get_feature_columns()
    features = df_processed[feature_columns].values
    
    # 4. 使用訓練時的縮放器轉換
    scaled_features = feature_scaler.transform(features)
    
    # 5. 確認資料量足夠
    if len(scaled_features) < lookback:
        raise ValueError(f"資料不足，需要至少 {lookback} 筆資料，目前只有 {len(scaled_features)} 筆")
    
    # 6. 取最後 lookback 筆資料
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
    """
    繪製訓練結果視覺化圖表
    
    Args:
        y_true: 實際價格
        y_pred: 預測價格
        start_date: 訓練起始日期
        end_date: 訓練結束日期
        rmse: 均方根誤差
        r2: R² 分數
    
    Returns:
        圖表儲存路徑
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 繪製實際與預測曲線
    x_axis = range(len(y_true))
    ax.plot(x_axis, y_true, label='Actual', color='blue', linewidth=1.5, alpha=0.8)
    ax.plot(x_axis, y_pred, label='Predicted', color='red', linewidth=1.5, alpha=0.8)
    
    # 標題（含指標）
    title = f"TWII Multivariate Prediction ({start_date} ~ {end_date}) | R²: {r2:.4f} | RMSE: {rmse:.2f}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('測試集樣本索引', fontsize=12)
    ax.set_ylabel('價格 (Price)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 文字注釋方塊（右下角）
    textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\n多變量模型'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.97, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    # 儲存圖表
    plot_path = MODELS_DIR / f"plot_{start_date}_{end_date}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[視覺化] 訓練結果圖表已儲存至：{plot_path}")
    
    return plot_path


# =============================================================================
# 模型成品管理
# =============================================================================
def get_artifact_paths(start_date: str, end_date: str) -> Tuple[Path, Path, Path, Path]:
    """
    取得模型成品路徑（多變量版本需要兩個縮放器檔案）
    
    Returns:
        model_path, feature_scaler_path, target_scaler_path, meta_path
    """
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
    lookback: int = LOOKBACK
):
    """儲存模型成品（含雙縮放器與效能指標）"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path, feature_scaler_path, target_scaler_path, meta_path = get_artifact_paths(start_date, end_date)
    
    # 儲存模型
    model.save(model_path)
    print(f"[儲存] 模型已儲存至：{model_path}")
    
    # 儲存特徵縮放器
    with open(feature_scaler_path, 'wb') as f:
        pickle.dump(feature_scaler, f)
    print(f"[儲存] 特徵縮放器已儲存至：{feature_scaler_path}")
    
    # 儲存目標縮放器
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    print(f"[儲存] 目標縮放器已儲存至：{target_scaler_path}")
    
    # 儲存元資料（含效能指標）
    metadata = {
        "model_type": "multivariate",
        "train_start": start_date,
        "train_end": end_date,
        "lookback": lookback,
        "n_features": n_features,
        "feature_columns": get_feature_columns(),
        "price_min": price_min,
        "price_max": price_max,
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
    """
    載入模型成品（多變量版本）
    
    Returns:
        model, feature_scaler, target_scaler, metadata
    """
    model_path, feature_scaler_path, target_scaler_path, meta_path = get_artifact_paths(start_date, end_date)
    
    # 載入模型（需註冊自訂層）
    model = keras.models.load_model(
        model_path,
        custom_objects={'SelfAttention': SelfAttention}
    )
    
    # 載入特徵縮放器
    with open(feature_scaler_path, 'rb') as f:
        feature_scaler = pickle.load(f)
    
    # 載入目標縮放器
    with open(target_scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    # 載入元資料
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return model, feature_scaler, target_scaler, metadata


# =============================================================================
# 智慧模型選擇
# =============================================================================
def parse_date_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """從檔名解析日期"""
    try:
        # 格式：meta_YYYY-MM-DD_YYYY-MM-DD.json
        parts = filename.replace('meta_', '').replace('.json', '').split('_')
        if len(parts) == 2:
            return parts[0], parts[1]
    except Exception:
        pass
    return None, None


def select_best_model(target_date: date) -> Optional[Dict[str, Any]]:
    """
    智慧選擇最適合的模型（含 Tie-Breaker 邏輯）
    
    選擇邏輯：
    1. 掃描所有 meta_*.json 檔案
    2. 篩選 train_end_date < target_date（避免資料洩漏）
    3. 排序優先順序：
       - 主鍵 (Recency): train_end_date 降冪（越新越好）
       - 次鍵 (Tie-breaker): train_start_date 降冪（較晚開始的模型更專精於近期市場）
    """
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
            
            # 計算訓練天數
            duration_days = (train_end - train_start).days
            model_name = f"model_{metadata['train_start']}_{metadata['train_end']}"
            
            # 篩選 1：訓練天數必須至少 4 年
            if duration_days < MIN_TRAIN_DAYS:
                print(f"[略過] 模型 {model_name} 訓練天數 {duration_days} 天不足 4 年 ({MIN_TRAIN_DAYS} 天)")
                continue
            
            # 篩選 2：train_end 必須早於 target_date（避免 look-ahead bias）
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
    
    # 排序：主鍵 train_end 降冪，次鍵 r2 降冪，第三鍵 train_start 降冪
    # train_end 最新 -> R² 最高 -> train_start 最新（更專精）
    candidates.sort(key=lambda x: (x['train_end'], x['r2'], x['train_start']), reverse=True)
    
    # 輸出候選模型列表
    print(f"\n[搜尋] 找到 {len(candidates)} 個可用模型：")
    for i, c in enumerate(candidates):
        r2_display = f"R²: {c['r2']:.4f}" if c['r2'] else "R²: N/A"
        status = "Selected (Best Match)" if i == 0 else ""
        if i > 0:
            # 判斷為何未被選中
            if c['train_end'] < candidates[0]['train_end']:
                status = "Backup (Older end date)"
            elif c['r2'] < candidates[0]['r2']:
                status = "Backup (Lower R²)"
            elif c['train_start'] < candidates[0]['train_start']:
                status = "Backup (Older start date)"
            else:
                status = "Backup"
        
        print(f"  {i+1}. {c['model_name']} ({r2_display}) -> {status}")
    
    # 返回排名第一的模型
    return candidates[0]['metadata']



def validate_model(metadata: Dict[str, Any], target_date: date, current_price: Optional[float] = None):
    """驗證模型並發出警告"""
    train_end = datetime.strptime(metadata['train_end'], '%Y-%m-%d').date()
    gap_days = (target_date - train_end).days
    
    # 檢查模型是否過期
    if gap_days > MODEL_STALE_DAYS:
        print(f"\n⚠️ 警告：選擇的模型已訓練超過 {MODEL_STALE_DAYS} 天（距今 {gap_days} 天），建議重新訓練。")
    
    # 檢查價格範圍
    if current_price is not None:
        price_min = metadata.get('price_min', 0)
        price_max = metadata.get('price_max', float('inf'))
        
        if current_price < price_min or current_price > price_max:
            print(f"\n⚠️ 警告：當前價格 {current_price:.2f} 超出訓練時的價格範圍 [{price_min:.2f}, {price_max:.2f}]")


# =============================================================================
# 訓練模式
# =============================================================================
def train_mode(args):
    """訓練模式"""
    print("\n" + "=" * 60)
    print("  TWII 多變量模型註冊系統 - 訓練模式")
    print("=" * 60)
    
    start_date = args.start
    end_date = args.end
    
    print(f"\n[設定] 訓練期間：{start_date} ~ {end_date}")
    print(f"[設定] Lookback: {LOOKBACK} | LSTM Units: {LSTM_UNITS}")
    print(f"[設定] Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}")
    print(f"[設定] KD 參數: {KD_PARAMS} | MACD 參數: {MACD_PARAMS}")
    
    # 設定隨機種子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 1. 下載資料
    df = download_data_by_date_range(start_date, end_date)
    
    # 2. 預處理（含技術指標計算）
    X_train, y_train, X_test, y_test, feature_scaler, target_scaler, price_min, price_max, n_features = preprocess_for_training(df)
    
    # 3. 建立模型
    print("\n[模型] 建立 LSTM-SSAM 多變量模型...")
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
    
    # 使用 target_scaler 還原價格
    y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_predicted = target_scaler.inverse_transform(y_pred_scaled).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    r2 = r2_score(y_actual, y_predicted)
    
    print("\n" + "=" * 50)
    print("📊 模型評估結果 (多變量)")
    print("=" * 50)
    print(f"  RMSE (均方根誤差)  : {rmse:.2f} 點")
    print(f"  R² Score (決定係數): {r2:.4f}")
    print("=" * 50)
    
    # 6. 儲存成品（含雙縮放器和效能指標）
    save_artifacts(
        model, feature_scaler, target_scaler, 
        start_date, end_date, 
        price_min, price_max, n_features,
        rmse=rmse, r2=r2
    )
    
    # 7. 繪製訓練結果圖表
    plot_training_results(y_actual, y_predicted, start_date, end_date, rmse, r2)
    
    print("\n✅ 訓練完成！模型成品已儲存至 saved_models_multivariate/ 目錄")


# =============================================================================
# 預測模式
# =============================================================================
def predict_mode(args):
    """預測模式 - 支援多步遞迴預測（多變量版本）"""
    print("\n" + "=" * 60)
    print("  TWII 多變量模型註冊系統 - 預測模式")
    print("=" * 60)
    
    # 解析目標日期
    if args.target_date == 'tomorrow':
        target_date = date.today() + timedelta(days=1)
        # 跳過週末
        while target_date.weekday() >= 5:
            target_date += timedelta(days=1)
        print(f"\n[設定] 預測目標日期：{target_date}（明日）")
    else:
        target_date = datetime.strptime(args.target_date, '%Y-%m-%d').date()
        print(f"\n[設定] 預測目標日期：{target_date}")
    
    # 選擇最佳模型
    print("\n[搜尋] 正在搜尋合適的模型...")
    metadata = select_best_model(target_date)
    
    if metadata is None:
        print(f"\n❌ 找不到適合目標日期 {target_date} 的歷史模型。")
        print("   請先訓練一個結束日期早於此日期的模型。")
        print(f"   範例：python {Path(__file__).name} train --start 2020-01-01 --end {target_date - timedelta(days=1)}")
        return
    
    train_start = metadata['train_start']
    train_end = metadata['train_end']
    lookback = metadata.get('lookback', LOOKBACK)
    
    print(f"\n✅ 使用模型版本：訓練期間 {train_start} 至 {train_end}")
    print(f"   特徵欄位：{metadata.get('feature_columns', get_feature_columns())}")
    
    # 載入模型成品（包含雙縮放器）
    print("\n[載入] 正在載入模型和縮放器...")
    model, feature_scaler, target_scaler, metadata = load_artifacts(train_start, train_end)
    
    # 下載最近資料（需下載足夠的歷史資料來計算技術指標）
    df = download_recent_data(lookback_days=lookback + 10)
    
    # 預處理並計算技術指標
    X, df_processed = preprocess_for_prediction(df, feature_scaler, lookback)
    
    current_price = df_processed['Adj Close'].iloc[-1]
    last_data_date = df_processed.index[-1].date()
    
    # 驗證模型
    validate_model(metadata, target_date, current_price)
    
    # 計算需要預測多少步
    days_diff = (target_date - last_data_date).days
    if days_diff <= 0:
        print(f"\n⚠️ 目標日期 {target_date} 已有歷史資料，請選擇未來日期。")
        return
    
    # 估算交易日數量（排除週末）
    trading_days = 0
    check_date = last_data_date
    while check_date < target_date:
        check_date += timedelta(days=1)
        if check_date.weekday() < 5:  # 週一到週五
            trading_days += 1
    
    print(f"\n[預測] 最近資料日期：{last_data_date}")
    print(f"[預測] 目標日期：{target_date}")
    print(f"[預測] 需要進行 {trading_days} 步遞迴預測")
    
    # 準備輸入資料（多變量序列）
    feature_columns = get_feature_columns()
    n_features = len(feature_columns)
    
    # 取最後 lookback 筆的縮放後特徵
    features = df_processed[feature_columns].values
    scaled_features = feature_scaler.transform(features)
    current_sequence = scaled_features[-lookback:].tolist()
    
    # 多步遞迴預測
    predictions = []
    predict_dates = []
    check_date = last_data_date
    
    for step in range(trading_days):
        # 準備輸入 shape: (1, lookback, n_features)
        X = np.array(current_sequence[-lookback:]).reshape(1, lookback, n_features)
        
        # 預測下一天
        y_pred_scaled = model.predict(X, verbose=0)
        predicted_scaled = y_pred_scaled[0, 0]
        
        # 更新序列：對於多變量預測，需要用預測值更新 Adj Close 特徵
        # 其他特徵（Volume_Log, K, D, MACD_Hist）無法預測，使用最後已知值
        # 這是多步遞迴預測的限制，但對於短期預測影響較小
        new_row = current_sequence[-1].copy()  # 複製最後一行
        new_row[0] = predicted_scaled  # 更新第一個特徵（Adj Close 的縮放值）
        current_sequence.append(new_row)
        
        # 記錄預測結果
        # 使用 target_scaler 還原真實價格
        predicted_price = target_scaler.inverse_transform([[predicted_scaled]])[0, 0]
        predictions.append(predicted_price)
        
        # 計算對應的交易日
        check_date += timedelta(days=1)
        while check_date.weekday() >= 5:  # 跳過週末
            check_date += timedelta(days=1)
        predict_dates.append(check_date)
    
    # 最終預測價格（目標日期）
    final_predicted_price = predictions[-1] if predictions else current_price
    
    # 計算漲跌幅
    price_change = final_predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    trend = "📈 看漲" if price_change > 0 else "📉 看跌"
    
    # 輸出結果
    print("\n" + "=" * 50)
    print(f"🔮 TWII 預測結果 (多變量模型) - 目標日期：{target_date}")
    print("=" * 50)
    print(f"  最近收盤價 ({last_data_date}) : {current_price:.2f}")
    print(f"  預測價格   ({target_date})   : {final_predicted_price:.2f}")
    print(f"  預期變化   : {price_change:+.2f} ({price_change_pct:+.2f}%)")
    print(f"  趨勢判斷   : {trend}")
    print("=" * 50)
    print(f"  使用模型   : {train_start} ~ {train_end}")
    print(f"  預測步數   : {trading_days} 個交易日")
    print(f"  輸入特徵   : {n_features} 個")
    print("=" * 50)
    
    # 顯示逐日預測（如果步數不多）
    if trading_days <= 60:
        print("\n📊 逐日預測明細：")
        print("-" * 40)
        prev_price = current_price
        for i, (pred_date, pred_price) in enumerate(zip(predict_dates, predictions)):
            daily_change = pred_price - prev_price
            daily_pct = (daily_change / prev_price) * 100
            print(f"  {pred_date} : {pred_price:.2f} ({daily_change:+.2f}, {daily_pct:+.2f}%)")
            prev_price = pred_price
        print("-" * 40)


# =============================================================================
# CLI 入口
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='TWII 多變量模型註冊系統 - 版本管理與自動模型選擇',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  訓練模型：
    python twii_model_registry_multivariate.py train --start 2020-01-01 --end 2024-01-01
  
  預測明天：
    python twii_model_registry_multivariate.py predict
  
  預測指定日期：
    python twii_model_registry_multivariate.py predict --target_date 2024-12-10

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
    predict_parser = subparsers.add_parser('predict', help='預測價格')
    predict_parser.add_argument(
        '--target_date',
        type=str,
        default='tomorrow',
        help='預測目標日期 (YYYY-MM-DD)，預設為明天'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
