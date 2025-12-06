# -*- coding: utf-8 -*-
"""
TWII 投資顧問機器人 (Trade Advisor)
智慧整合多模型，產生動態定期定額操作建議

功能：
- 智慧模型選擇：自動掃描並選擇最佳的 1日/5日 預測模型
- 雙模型推論：同時取得短期(T+1)與波段(T+5)預測
- 投資建議產生：根據預測漲幅提供資金控管與進場時機建議

使用方式：
  python trade_advisor.py

輸入來源：
  - 短期訊號 (T+1): saved_models_multivariate/
  - 波段趨勢 (T+5): saved_models_optimized/
"""

import json
import pickle
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =============================================================================
# 設定
# =============================================================================
BASE_DIR = Path(__file__).parent

# 模型目錄
MODELS_DIR_1D = BASE_DIR / "saved_models_multivariate"  # T+1 短期模型
MODELS_DIR_5D = BASE_DIR / "saved_models_optimized"     # T+5 波段模型

# 模型篩選條件
MIN_TRAIN_DAYS = 1460  # 最低訓練天數（4 年）

# 技術指標參數
KD_PARAMS = (9, 3, 3)
MACD_PARAMS = (12, 26, 9)

# 投資建議閾值
TREND_BULLISH_THRESHOLD = 0.02    # 5日漲幅 > 2% 為大晴天
TREND_BEARISH_THRESHOLD = -0.02   # 5日漲幅 < -2% 為暴風雨


# =============================================================================
# 自訂 Self-Attention Layer（載入模型需要）
# =============================================================================
class SelfAttention(layers.Layer):
    """Sequential Self-Attention Layer"""
    
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
# 特徵工程（與訓練時完全相同）
# =============================================================================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    新增技術指標到 DataFrame（與訓練時完全相同）
    
    新增欄位：
    - Volume_Log: 成交量（Log 轉換）
    - K: KD 指標的 K 值 (9, 3, 3)
    - D: KD 指標的 D 值
    - MACD_Hist: MACD 柱狀圖 (12, 26, 9)
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
    df = df.dropna()
    
    return df


def get_feature_columns() -> list:
    """取得特徵欄位名稱"""
    return ['Adj Close', 'Volume_Log', 'K', 'D', 'MACD_Hist']


# =============================================================================
# 智慧模型選擇機制
# =============================================================================
def select_best_model(model_dir: Path) -> Optional[Dict[str, Any]]:
    """
    智慧選擇最佳模型
    
    選擇邏輯：
    1. 訓練期間過濾：train_end - train_start >= 1460 天 (4年)
    2. 避免未來數據：train_end <= today
    3. 排序（降冪）：train_end -> r2 -> train_start
    
    Args:
        model_dir: 模型目錄路徑
    
    Returns:
        最佳模型的 metadata，如果找不到則返回 None
    """
    if not model_dir.exists():
        print(f"[錯誤] 模型目錄不存在：{model_dir}")
        return None
    
    meta_files = list(model_dir.glob("meta_*.json"))
    if not meta_files:
        print(f"[錯誤] 在 {model_dir} 找不到任何模型檔案")
        return None
    
    today = date.today()
    candidates = []
    
    for meta_file in meta_files:
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            train_start = datetime.strptime(metadata['train_start'], '%Y-%m-%d').date()
            train_end = datetime.strptime(metadata['train_end'], '%Y-%m-%d').date()
            
            # 計算訓練天數
            duration_days = (train_end - train_start).days
            
            # 篩選條件 1：訓練期間必須至少 4 年
            if duration_days < MIN_TRAIN_DAYS:
                continue
            
            # 篩選條件 2：避免未來數據（train_end <= today）
            if train_end > today:
                continue
            
            # 取得 R² 分數
            r2 = metadata.get('metrics', {}).get('r2', 0.0) or 0.0
            
            candidates.append({
                'metadata': metadata,
                'meta_file': meta_file,
                'train_start': train_start,
                'train_end': train_end,
                'duration_days': duration_days,
                'r2': r2
            })
            
        except Exception as e:
            print(f"[警告] 無法解析 {meta_file}: {e}")
            continue
    
    if not candidates:
        print(f"[錯誤] 在 {model_dir} 找不到符合條件的模型（訓練 >= 4 年）")
        return None
    
    # 排序邏輯（降冪）：train_end -> r2 -> train_start
    candidates.sort(
        key=lambda x: (x['train_end'], x['r2'], x['train_start']),
        reverse=True
    )
    
    # 返回最佳模型
    return candidates[0]['metadata']


# =============================================================================
# 模型載入
# =============================================================================
def load_model_artifacts(model_dir: Path, metadata: Dict[str, Any]) -> Tuple:
    """
    載入模型及相關資源
    
    Returns:
        (model, feature_scaler, target_scaler, metadata)
    """
    train_start = metadata['train_start']
    train_end = metadata['train_end']
    
    # 構建檔案路徑
    model_path = model_dir / f"model_{train_start}_{train_end}.keras"
    
    # 嘗試兩種可能的縮放器命名格式
    # 格式 1：feature_scaler_*.pkl（optimized 目錄）
    # 格式 2：scaler_*.pkl（multivariate 目錄，舊格式）
    feature_scaler_path = model_dir / f"feature_scaler_{train_start}_{train_end}.pkl"
    target_scaler_path = model_dir / f"target_scaler_{train_start}_{train_end}.pkl"
    
    # 如果新格式不存在，嘗試舊格式（multivariate 目錄可能使用）
    if not feature_scaler_path.exists():
        # multivariate 目錄的格式
        feature_scaler_path = model_dir / f"feature_scaler_{train_start}_{train_end}.pkl"
    if not target_scaler_path.exists():
        target_scaler_path = model_dir / f"target_scaler_{train_start}_{train_end}.pkl"
    
    # 載入模型
    model = keras.models.load_model(
        model_path,
        custom_objects={'SelfAttention': SelfAttention}
    )
    
    # 載入縮放器
    with open(feature_scaler_path, 'rb') as f:
        feature_scaler = pickle.load(f)
    
    with open(target_scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    return model, feature_scaler, target_scaler, metadata


# =============================================================================
# 資料獲取
# =============================================================================
def download_market_data(lookback_1d: int, lookback_5d: int) -> pd.DataFrame:
    """
    下載市場資料
    
    下載長度：max(lookback_1d, lookback_5d) + 50 天（確保技術指標暖機）
    """
    required_days = max(lookback_1d, lookback_5d) + 50
    
    print(f"[資料獲取] 正在下載 ^TWII 最近 {required_days} 天資料...")
    
    ticker = yf.Ticker("^TWII")
    df = ticker.history(period=f"{required_days}d")
    
    if df.empty:
        raise ValueError("無法取得 ^TWII 資料")
    
    print(f"[資料獲取] 成功下載 {len(df)} 筆資料")
    
    return df


# =============================================================================
# 模型推論
# =============================================================================
def run_inference(
    model,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    df_processed: pd.DataFrame,
    lookback: int
) -> float:
    """
    執行模型推論
    
    Args:
        model: Keras 模型
        feature_scaler: 特徵縮放器
        target_scaler: 目標縮放器
        df_processed: 已處理的資料（含技術指標）
        lookback: 回看天數
    
    Returns:
        預測價格（真實價格，非縮放後）
    """
    feature_columns = get_feature_columns()
    
    # 確保有 Adj Close 欄位
    if 'Adj Close' not in df_processed.columns:
        df_processed = df_processed.copy()
        df_processed['Adj Close'] = df_processed['Close']
    
    # 取得特徵
    features = df_processed[feature_columns].values
    
    # 縮放特徵
    scaled_features = feature_scaler.transform(features)
    
    # 確保資料量足夠
    if len(scaled_features) < lookback:
        raise ValueError(f"資料不足，需要至少 {lookback} 筆")
    
    # 取最後 lookback 筆
    X = scaled_features[-lookback:].reshape(1, lookback, len(feature_columns))
    
    # 預測
    y_pred_scaled = model.predict(X, verbose=0)
    
    # 還原為真實價格
    predicted_price = target_scaler.inverse_transform(y_pred_scaled)[0, 0]
    
    return predicted_price


# =============================================================================
# 投資建議產生
# =============================================================================
def generate_advice(change_1d: float, change_5d: float) -> Dict[str, str]:
    """
    根據預測漲幅產生投資建議
    
    Args:
        change_1d: T+1 預期漲幅（例如 0.01 = 1%）
        change_5d: T+5 預期漲幅
    
    Returns:
        包含 trend_advice 和 timing_advice 的字典
    """
    # 資金控管建議（5日模型）
    if change_5d > TREND_BULLISH_THRESHOLD:
        trend_emoji = "🌞"
        trend_status = "大晴天"
        trend_advice = "市場樂觀，建議加碼扣款 1.5~2 倍"
    elif change_5d < TREND_BEARISH_THRESHOLD:
        trend_emoji = "⛈️"
        trend_status = "暴風雨"
        trend_advice = "市場悲觀，建議暫停扣款或減碼 50%"
    else:
        trend_emoji = "☁️"
        trend_status = "多雲盤整"
        trend_advice = "市場中性，維持標準扣款金額"
    
    # 進場時機建議（1日模型）
    if change_1d > 0:
        timing_emoji = "✅"
        timing_status = "綠燈通行"
        timing_advice = "短期看漲，建議今日進場扣款"
    else:
        timing_emoji = "🛑"
        timing_status = "紅燈停看聽"
        timing_advice = "短期看跌，建議觀望等待更好時機"
    
    return {
        'trend_emoji': trend_emoji,
        'trend_status': trend_status,
        'trend_advice': trend_advice,
        'timing_emoji': timing_emoji,
        'timing_status': timing_status,
        'timing_advice': timing_advice
    }


# =============================================================================
# 計算未來交易日
# =============================================================================
def get_future_trading_date(start_date: date, trading_days: int) -> date:
    """計算未來第 N 個交易日的日期（跳過週末）"""
    current_date = start_date
    days_counted = 0
    
    while days_counted < trading_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            days_counted += 1
    
    return current_date


# =============================================================================
# 主程式
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("  🤖 TWII 投資顧問機器人 (Trade Advisor)")
    print("  智慧整合多模型，產生動態定期定額操作建議")
    print("=" * 70)
    
    today = date.today()
    print(f"\n📅 今日日期：{today}")
    
    # =========================================================================
    # 1. 智慧選擇最佳模型
    # =========================================================================
    print("\n" + "-" * 50)
    print("📊 模型選擇")
    print("-" * 50)
    
    # 選擇 T+1 模型（短期訊號）
    print(f"\n[T+1 模型] 掃描 {MODELS_DIR_1D}...")
    metadata_1d = select_best_model(MODELS_DIR_1D)
    
    if metadata_1d is None:
        print("\n❌ 無法載入 T+1 模型，程式終止。")
        print("   請先執行：python twii_model_registry_multivariate.py train ...")
        return
    
    # 選擇 T+5 模型（波段趨勢）
    print(f"\n[T+5 模型] 掃描 {MODELS_DIR_5D}...")
    metadata_5d = select_best_model(MODELS_DIR_5D)
    
    if metadata_5d is None:
        print("\n❌ 無法載入 T+5 模型，程式終止。")
        print("   請先執行：python twii_model_optimizer.py train")
        return
    
    # 取得 lookback 參數
    lookback_1d = metadata_1d.get('lookback', 10)
    lookback_5d = metadata_5d.get('hyperparameters', {}).get('lookback', 
                  metadata_5d.get('lookback', 30))
    
    # 顯示模型資訊
    r2_1d = metadata_1d.get('metrics', {}).get('r2', 'N/A')
    r2_5d = metadata_5d.get('metrics', {}).get('r2', 'N/A')
    
    print(f"\n✅ 已選擇模型：")
    print(f"  [T+1] {metadata_1d['train_start']} ~ {metadata_1d['train_end']} (R²: {r2_1d}, Lookback: {lookback_1d})")
    print(f"  [T+5] {metadata_5d['train_start']} ~ {metadata_5d['train_end']} (R²: {r2_5d}, Lookback: {lookback_5d})")
    
    # =========================================================================
    # 2. 載入模型
    # =========================================================================
    print("\n" + "-" * 50)
    print("🔧 載入模型")
    print("-" * 50)
    
    try:
        model_1d, scaler_feat_1d, scaler_tgt_1d, _ = load_model_artifacts(MODELS_DIR_1D, metadata_1d)
        print(f"  [T+1] 模型載入成功")
        
        model_5d, scaler_feat_5d, scaler_tgt_5d, _ = load_model_artifacts(MODELS_DIR_5D, metadata_5d)
        print(f"  [T+5] 模型載入成功")
    except Exception as e:
        print(f"\n❌ 模型載入失敗：{e}")
        return
    
    # =========================================================================
    # 3. 下載並處理市場資料
    # =========================================================================
    print("\n" + "-" * 50)
    print("📈 市場資料")
    print("-" * 50)
    
    try:
        df_raw = download_market_data(lookback_1d, lookback_5d)
        df_processed = add_technical_indicators(df_raw)
        
        # 確保有 Adj Close
        if 'Adj Close' not in df_processed.columns:
            df_processed['Adj Close'] = df_processed['Close']
        
        current_price = df_processed['Adj Close'].iloc[-1]
        last_date = df_processed.index[-1].date()
        
        print(f"  最近交易日：{last_date}")
        print(f"  目前收盤價：{current_price:.2f}")
    except Exception as e:
        print(f"\n❌ 資料獲取失敗：{e}")
        return
    
    # =========================================================================
    # 4. 執行預測
    # =========================================================================
    print("\n" + "-" * 50)
    print("🔮 模型預測")
    print("-" * 50)
    
    try:
        # T+1 預測
        pred_1d = run_inference(model_1d, scaler_feat_1d, scaler_tgt_1d, df_processed, lookback_1d)
        change_1d = (pred_1d - current_price) / current_price
        date_1d = get_future_trading_date(last_date, 1)
        
        print(f"  [T+1] 預測 {date_1d}：{pred_1d:.2f} ({change_1d:+.2%})")
        
        # T+5 預測
        pred_5d = run_inference(model_5d, scaler_feat_5d, scaler_tgt_5d, df_processed, lookback_5d)
        change_5d = (pred_5d - current_price) / current_price
        date_5d = get_future_trading_date(last_date, 5)
        
        print(f"  [T+5] 預測 {date_5d}：{pred_5d:.2f} ({change_5d:+.2%})")
    except Exception as e:
        print(f"\n❌ 預測失敗：{e}")
        return
    
    # =========================================================================
    # 5. 產生投資建議
    # =========================================================================
    advice = generate_advice(change_1d, change_5d)
    
    # =========================================================================
    # 6. 輸出報表
    # =========================================================================
    print("\n" + "=" * 70)
    print("  📋 投資顧問報告")
    print("=" * 70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                        📊 模型履歷                                  │
├─────────────────────────────────────────────────────────────────────┤
│  短期模型 (T+1)：{metadata_1d['train_start']} ~ {metadata_1d['train_end']}                     │
│                  R² = {r2_1d}  |  Lookback = {lookback_1d} 天                     │
├─────────────────────────────────────────────────────────────────────┤
│  波段模型 (T+5)：{metadata_5d['train_start']} ~ {metadata_5d['train_end']}                     │
│                  R² = {r2_5d}  |  Lookback = {lookback_5d} 天                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        🔮 預測數據                                  │
├─────────────────────────────────────────────────────────────────────┤
│  目前價格 ({last_date})              ：{current_price:>10.2f}                     │
│  T+1 預測 ({date_1d})              ：{pred_1d:>10.2f}  ({change_1d:>+6.2%})           │
│  T+5 預測 ({date_5d})              ：{pred_5d:>10.2f}  ({change_5d:>+6.2%})           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        💡 操作建議                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  {advice['trend_emoji']} 資金控管 (5日趨勢)：{advice['trend_status']}                              │
│     → {advice['trend_advice']}                       │
│                                                                     │
│  {advice['timing_emoji']} 進場時機 (1日訊號)：{advice['timing_status']}                            │
│     → {advice['timing_advice']}                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # 綜合建議
    print("=" * 70)
    if change_5d > TREND_BULLISH_THRESHOLD and change_1d > 0:
        print("  🎯 綜合建議：市場短期看漲、中期樂觀，建議「加碼進場」")
    elif change_5d < TREND_BEARISH_THRESHOLD and change_1d < 0:
        print("  🎯 綜合建議：市場短期看跌、中期悲觀，建議「暫停觀望」")
    elif change_5d > TREND_BULLISH_THRESHOLD and change_1d < 0:
        print("  🎯 綜合建議：中期樂觀但短期回檔，建議「等待低接」")
    elif change_5d < TREND_BEARISH_THRESHOLD and change_1d > 0:
        print("  🎯 綜合建議：中期悲觀但短期反彈，建議「逢高減碼」")
    else:
        print("  🎯 綜合建議：市場盤整中，建議「維持標準定期定額」")
    print("=" * 70)
    
    print("\n⚠️  免責聲明：本報告僅供參考，不構成投資建議。投資有風險，請謹慎決策。\n")


if __name__ == "__main__":
    main()
