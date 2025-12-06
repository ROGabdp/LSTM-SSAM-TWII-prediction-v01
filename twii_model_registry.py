# -*- coding: utf-8 -*-
"""
TWII 模型註冊系統 (Model Registry System)
版本管理與自動模型選擇

功能：
- train 模式：訓練 LSTM-SSAM 模型並儲存成品
- predict 模式：智慧選擇合適模型進行預測

使用方式：
  訓練：python twii_model_registry.py train --start 2020-01-01 --end 2024-01-01
  預測：python twii_model_registry.py predict --target_date 2024-12-10
  預測（明天）：python twii_model_registry.py predict
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
MODELS_DIR = Path(__file__).parent / "saved_models"
LOOKBACK = 10  # 回看天數（論文規格）
LSTM_UNITS = 50
EPOCHS = 50
BATCH_SIZE = 10
TRAIN_RATIO = 0.9
MODEL_STALE_DAYS = 180  # 模型過期警告閾值（天）
MIN_TRAIN_DAYS = 1460   # 最低訓練天數（4 年 = 4 × 365 = 1460 天）

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
# 模型架構
# =============================================================================
def build_lstm_ssam_model(time_steps: int = LOOKBACK, features: int = 1, lstm_units: int = LSTM_UNITS):
    """建立 LSTM + Self-Attention 混合模型"""
    inputs = layers.Input(shape=(time_steps, features), name='input_layer')
    lstm_out = layers.LSTM(units=lstm_units, return_sequences=True, name='lstm_layer')(inputs)
    attention_out = SelfAttention(name='self_attention')(lstm_out)
    flatten_out = layers.Flatten(name='flatten_layer')(attention_out)
    outputs = layers.Dense(units=1, activation='linear', name='output_layer')(flatten_out)
    
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_SSAM_Model')
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
    """下載最近的資料用於預測"""
    print(f"[資料獲取] 正在下載最近 {lookback_days} 天的資料...")
    
    ticker = yf.Ticker("^TWII")
    df = ticker.history(period=f"{lookback_days}d")
    
    if df.empty:
        raise ValueError("無法取得最近的 ^TWII 資料")
    
    print(f"[資料獲取] 成功下載 {len(df)} 筆資料")
    
    return df


def preprocess_for_training(df: pd.DataFrame, lookback: int = LOOKBACK, train_ratio: float = TRAIN_RATIO):
    """訓練用資料預處理"""
    data = df['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # 記錄價格範圍
    price_min = float(df['Close'].min())
    price_max = float(df['Close'].max())
    
    print(f"[預處理] 訓練集：{len(X_train)} 筆 | 測試集：{len(X_test)} 筆")
    print(f"[預處理] 價格範圍：{price_min:.2f} ~ {price_max:.2f}")
    
    return X_train, y_train, X_test, y_test, scaler, price_min, price_max


def preprocess_for_prediction(df: pd.DataFrame, scaler: MinMaxScaler, lookback: int = LOOKBACK) -> np.ndarray:
    """預測用資料預處理"""
    data = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)
    
    # 使用最後 lookback 筆資料
    if len(scaled_data) < lookback:
        raise ValueError(f"資料不足，需要至少 {lookback} 筆資料")
    
    X = scaled_data[-lookback:].reshape(1, lookback, 1)
    
    return X


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
    title = f"TWII Prediction ({start_date} ~ {end_date}) | R²: {r2:.4f} | RMSE: {rmse:.2f}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('測試集樣本索引', fontsize=12)
    ax.set_ylabel('價格 (Price)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 文字注釋方塊（右下角）
    textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}'
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
def get_artifact_paths(start_date: str, end_date: str) -> Tuple[Path, Path, Path]:
    """取得模型成品路徑"""
    model_path = MODELS_DIR / f"model_{start_date}_{end_date}.keras"
    scaler_path = MODELS_DIR / f"scaler_{start_date}_{end_date}.pkl"
    meta_path = MODELS_DIR / f"meta_{start_date}_{end_date}.json"
    return model_path, scaler_path, meta_path


def save_artifacts(
    model,
    scaler: MinMaxScaler,
    start_date: str,
    end_date: str,
    price_min: float,
    price_max: float,
    rmse: float = None,
    r2: float = None,
    lookback: int = LOOKBACK
):
    """儲存模型成品（含效能指標）"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path, scaler_path, meta_path = get_artifact_paths(start_date, end_date)
    
    # 儲存模型
    model.save(model_path)
    print(f"[儲存] 模型已儲存至：{model_path}")
    
    # 儲存縮放器
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[儲存] 縮放器已儲存至：{scaler_path}")
    
    # 儲存元資料（含效能指標）
    metadata = {
        "train_start": start_date,
        "train_end": end_date,
        "lookback": lookback,
        "price_min": price_min,
        "price_max": price_max,
        "training_timestamp": datetime.now().isoformat(),
        "metrics": {
            "rmse": round(rmse, 2) if rmse is not None else None,
            "r2": round(r2, 4) if r2 is not None else None
        }
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[儲存] 元資料已儲存至：{meta_path}")


def load_artifacts(start_date: str, end_date: str) -> Tuple[Model, MinMaxScaler, Dict[str, Any]]:
    """載入模型成品"""
    model_path, scaler_path, meta_path = get_artifact_paths(start_date, end_date)
    
    # 載入模型（需註冊自訂層）
    model = keras.models.load_model(
        model_path,
        custom_objects={'SelfAttention': SelfAttention}
    )
    
    # 載入縮放器
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 載入元資料
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return model, scaler, metadata


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
            
            # 篩選 1：訓練天數必須至少 8 年
            if duration_days < MIN_TRAIN_DAYS:
                print(f"[略過] 模型 {model_name} 訓練天數 {duration_days} 天不足 8 年 ({MIN_TRAIN_DAYS} 天)")
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
    print("  TWII 模型註冊系統 - 訓練模式")
    print("=" * 60)
    
    start_date = args.start
    end_date = args.end
    
    print(f"\n[設定] 訓練期間：{start_date} ~ {end_date}")
    print(f"[設定] Lookback: {LOOKBACK} | LSTM Units: {LSTM_UNITS}")
    print(f"[設定] Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}")
    
    # 設定隨機種子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 1. 下載資料
    df = download_data_by_date_range(start_date, end_date)
    
    # 2. 預處理
    X_train, y_train, X_test, y_test, scaler, price_min, price_max = preprocess_for_training(df)
    
    # 3. 建立模型
    print("\n[模型] 建立 LSTM-SSAM 模型...")
    model = build_lstm_ssam_model()
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
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_predicted = scaler.inverse_transform(y_pred_scaled).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    r2 = r2_score(y_actual, y_predicted)
    
    print("\n" + "=" * 50)
    print("📊 模型評估結果")
    print("=" * 50)
    print(f"  RMSE (均方根誤差)  : {rmse:.2f} 點")
    print(f"  R² Score (決定係數): {r2:.4f}")
    print("=" * 50)
    
    # 6. 儲存成品（含效能指標）
    save_artifacts(model, scaler, start_date, end_date, price_min, price_max, rmse=rmse, r2=r2)
    
    # 7. 繪製訓練結果圖表
    plot_training_results(y_actual, y_predicted, start_date, end_date, rmse, r2)
    
    print("\n✅ 訓練完成！模型成品已儲存至 saved_models/ 目錄")


# =============================================================================
# 預測模式
# =============================================================================
def predict_mode(args):
    """預測模式 - 支援多步遞迴預測"""
    print("\n" + "=" * 60)
    print("  TWII 模型註冊系統 - 預測模式")
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
    
    # 載入模型成品
    print("\n[載入] 正在載入模型和縮放器...")
    model, scaler, metadata = load_artifacts(train_start, train_end)
    
    # 下載最近資料
    df = download_recent_data(lookback_days=lookback + 10)
    current_price = df['Close'].iloc[-1]
    last_data_date = df.index[-1].date()
    
    # 驗證模型
    validate_model(metadata, target_date, current_price)
    
    # 計算需要預測多少步
    # 計算交易日差距（簡化處理：假設每週 5 個交易日）
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
    
    # 準備輸入資料
    data = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)
    
    # 取最後 lookback 筆作為初始序列
    current_sequence = scaled_data[-lookback:].flatten().tolist()
    
    # 多步遞迴預測
    predictions = []
    predict_dates = []
    check_date = last_data_date
    
    for step in range(trading_days):
        # 準備輸入
        X = np.array(current_sequence[-lookback:]).reshape(1, lookback, 1)
        
        # 預測下一天
        y_pred_scaled = model.predict(X, verbose=0)
        predicted_scaled = y_pred_scaled[0, 0]
        
        # 更新序列（將預測值加入）
        current_sequence.append(predicted_scaled)
        
        # 記錄預測結果
        predicted_price = scaler.inverse_transform([[predicted_scaled]])[0, 0]
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
    print(f"🔮 TWII 預測結果 - 目標日期：{target_date}")
    print("=" * 50)
    print(f"  最近收盤價 ({last_data_date}) : {current_price:.2f}")
    print(f"  預測價格   ({target_date})   : {final_predicted_price:.2f}")
    print(f"  預期變化   : {price_change:+.2f} ({price_change_pct:+.2f}%)")
    print(f"  趨勢判斷   : {trend}")
    print("=" * 50)
    print(f"  使用模型   : {train_start} ~ {train_end}")
    print(f"  預測步數   : {trading_days} 個交易日")
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
        description='TWII 模型註冊系統 - 版本管理與自動模型選擇',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  訓練模型：
    python twii_model_registry.py train --start 2020-01-01 --end 2024-01-01
  
  預測明天：
    python twii_model_registry.py predict
  
  預測指定日期：
    python twii_model_registry.py predict --target_date 2024-12-10
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
