# -*- coding: utf-8 -*-
"""
TWII 投資顧問機器人 (Trade Advisor)
智慧整合多模型，產生動態定期定額操作建議

功能：
- 智慧模型選擇：自動掃描並選擇最佳的 1日/5日 預測模型
- 雙模型推論：同時取得短期(T+1)與波段(T+5)預測
- 信心度評估：MC Dropout (5日) + RMSE 區間判斷 (1日)
- 投資建議產生：根據預測漲幅與信心度提供資金控管與進場時機建議

使用方式：
  python trade_advisor.py

輸入來源：
  - 短期訊號 (T+1): saved_models_multivariate/
  - 波段趨勢 (T+5): saved_models_5d/
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
MODELS_DIR_5D = BASE_DIR / "saved_models_5d"            # T+5 波段模型

# 模型篩選條件
MIN_TRAIN_DAYS = 1460  # 最低訓練天數（4 年）

# 技術指標參數
KD_PARAMS = (9, 3, 3)
MACD_PARAMS = (12, 26, 9)

# 投資建議閾值
TREND_BULLISH_THRESHOLD = 0.02    # 5日漲幅 > 2% 為大晴天
TREND_BEARISH_THRESHOLD = -0.02   # 5日漲幅 < -2% 為暴風雨

# MC Dropout 設定
MC_DROPOUT_ITERATIONS = 30        # MC Dropout 預測迭代次數

# 信心度閾值
CV_HIGH_CONFIDENCE = 0.005        # CV < 0.5% 為高信心度
CV_LOW_CONFIDENCE = 0.01          # CV > 1% 為低信心度


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
            
            duration_days = (train_end - train_start).days
            
            if duration_days < MIN_TRAIN_DAYS:
                continue
            
            if train_end > today:
                continue
            
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
    
    candidates.sort(
        key=lambda x: (x['train_end'], x['r2'], x['train_start']),
        reverse=True
    )
    
    return candidates[0]['metadata']


# =============================================================================
# 模型載入
# =============================================================================
def load_model_artifacts(model_dir: Path, metadata: Dict[str, Any]) -> Tuple:
    """
    載入模型及相關資源
    
    支援兩種 Scaler 命名格式：
    - 格式 A：feature_scaler_YYYY-MM-DD_YYYY-MM-DD.pkl（新格式）
    - 格式 B：scaler_YYYY-MM-DD_YYYY-MM-DD.pkl（舊版相容）
    
    Returns:
        (model, feature_scaler, target_scaler, metadata)
    """
    train_start = metadata['train_start']
    train_end = metadata['train_end']
    
    model_path = model_dir / f"model_{train_start}_{train_end}.keras"
    
    feature_scaler_path = model_dir / f"feature_scaler_{train_start}_{train_end}.pkl"
    target_scaler_path = model_dir / f"target_scaler_{train_start}_{train_end}.pkl"
    
    if not feature_scaler_path.exists():
        legacy_scaler_path = model_dir / f"scaler_{train_start}_{train_end}.pkl"
        if legacy_scaler_path.exists():
            feature_scaler_path = legacy_scaler_path
            target_scaler_path = legacy_scaler_path
            print(f"  [注意] 使用舊版 Scaler 格式：{legacy_scaler_path.name}")
    
    model = keras.models.load_model(
        model_path,
        custom_objects={'SelfAttention': SelfAttention}
    )
    
    with open(feature_scaler_path, 'rb') as f:
        feature_scaler = pickle.load(f)
    
    with open(target_scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    return model, feature_scaler, target_scaler, metadata


# =============================================================================
# 資料獲取
# =============================================================================
def download_market_data(lookback_1d: int, lookback_5d: int) -> pd.DataFrame:
    """下載市場資料"""
    required_days = max(lookback_1d, lookback_5d) + 50
    
    print(f"[資料獲取] 正在下載 ^TWII 最近 {required_days} 天資料...")
    
    ticker = yf.Ticker("^TWII")
    df = ticker.history(period=f"{required_days}d")
    
    if df.empty:
        raise ValueError("無法取得 ^TWII 資料")
    
    print(f"[資料獲取] 成功下載 {len(df)} 筆資料")
    
    return df


# =============================================================================
# MC Dropout 不確定性評估（針對 5 日模型）
# =============================================================================
def predict_with_uncertainty(
    model,
    X: np.ndarray,
    target_scaler: MinMaxScaler,
    n_iter: int = MC_DROPOUT_ITERATIONS
) -> Tuple[float, float, str]:
    """
    使用 MC Dropout 進行不確定性評估
    
    原理：
    - 強制開啟 Dropout 模式（training=True），重複預測 n_iter 次
    - 計算預測結果的平均值作為最終預測，標準差作為不確定性
    
    Args:
        model: Keras 模型（必須包含 Dropout 層）
        X: 輸入特徵 (shape: 1, lookback, n_features)
        target_scaler: 目標變數縮放器
        n_iter: MC Dropout 迭代次數
    
    Returns:
        (mean_price, std_price, confidence_level)
        - mean_price: 預測價格平均值
        - std_price: 預測價格標準差（風險波動）
        - confidence_level: 信心度等級 ('高', '中', '低')
    """
    predictions = []
    
    for _ in range(n_iter):
        # 強制開啟 Dropout 模式
        y_pred_scaled = model(X, training=True)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.numpy())[0, 0]
        predictions.append(y_pred)
    
    predictions = np.array(predictions)
    
    # 計算統計量
    mean_price = np.mean(predictions)
    std_price = np.std(predictions)
    
    # 計算變異係數 (CV = Std / Mean)
    cv = std_price / mean_price if mean_price != 0 else 0
    
    # 判斷信心度
    if cv < CV_HIGH_CONFIDENCE:
        confidence_level = "高"
    elif cv > CV_LOW_CONFIDENCE:
        confidence_level = "低"
    else:
        confidence_level = "中"
    
    return mean_price, std_price, confidence_level


# =============================================================================
# RMSE 區間信心度評估（針對 1 日模型）- 寬鬆門檻版本
# =============================================================================
def evaluate_1d_confidence(
    pred_price: float,
    current_price: float,
    rmse: float
) -> str:
    """
    根據 RMSE 評估 1 日模型的信心度（寬鬆門檻）
    
    邏輯（寬鬆版）：
    - 預期獲利點數 = abs(預測價 - 現價)
    - 若 預期獲利 > 0.8 * RMSE -> 信心度：高
    - 若 預期獲利 > 0.4 * RMSE -> 信心度：中
    - 若 預期獲利 < 0.4 * RMSE -> 信心度：低（可能只是雜訊）
    
    Args:
        pred_price: 預測價格
        current_price: 當前價格
        rmse: 模型 RMSE
    
    Returns:
        confidence_level: 信心度等級 ('高', '中', '低')
    """
    expected_profit = abs(pred_price - current_price)
    
    # 寬鬆門檻：0.8x 和 0.4x RMSE
    if expected_profit > 0.8 * rmse:
        return "高"
    elif expected_profit > 0.4 * rmse:
        return "中"
    else:
        return "低"


# =============================================================================
# 趨勢共振 (Trend Alignment) 加分機制
# =============================================================================
def apply_trend_alignment(
    confidence_1d: str,
    change_5d: float,
    confidence_5d: str
) -> tuple:
    """
    根據 T+5 趨勢對 T+1 信心度進行加分調整
    
    邏輯：
    - 若 T+5 看漲 (change > 0) 且信心度為高/中：
      → T+1 信心度升一級（低→中，中→高）
    - 若 T+5 看跌：
      → T+1 維持原判（逆勢需高標準）
    
    Args:
        confidence_1d: 原始 T+1 信心度
        change_5d: T+5 預期漲跌幅
        confidence_5d: T+5 信心度
    
    Returns:
        (adjusted_confidence, upgraded): 調整後信心度, 是否有升級
    """
    # 檢查是否符合順勢加分條件
    is_t5_bullish = change_5d > 0
    is_t5_confident = confidence_5d in ["高", "中"]
    
    if is_t5_bullish and is_t5_confident:
        # 順勢交易，信心度升一級
        if confidence_1d == "低":
            return "中", True
        elif confidence_1d == "中":
            return "高", True
        else:
            return "高", False  # 已經是高，不變
    else:
        # 逆勢或 T+5 信心不足，維持原判
        return confidence_1d, False


# =============================================================================
# 模型推論（標準版）
# =============================================================================
def prepare_input_data(
    df_processed: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    lookback: int
) -> np.ndarray:
    """準備模型輸入資料"""
    feature_columns = get_feature_columns()
    
    if 'Adj Close' not in df_processed.columns:
        df_processed = df_processed.copy()
        df_processed['Adj Close'] = df_processed['Close']
    
    features = df_processed[feature_columns].values
    scaled_features = feature_scaler.transform(features)
    
    if len(scaled_features) < lookback:
        raise ValueError(f"資料不足，需要至少 {lookback} 筆")
    
    X = scaled_features[-lookback:].reshape(1, lookback, len(feature_columns))
    
    return X


def run_inference(
    model,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    df_processed: pd.DataFrame,
    lookback: int
) -> float:
    """執行標準模型推論"""
    X = prepare_input_data(df_processed, feature_scaler, lookback)
    y_pred_scaled = model.predict(X, verbose=0)
    predicted_price = target_scaler.inverse_transform(y_pred_scaled)[0, 0]
    
    return predicted_price


# =============================================================================
# 投資建議產生（含信心度考量）
# =============================================================================
def generate_advice(
    change_1d: float,
    change_5d: float,
    confidence_1d: str,
    confidence_5d: str
) -> Dict[str, str]:
    """
    根據預測漲幅與信心度產生投資建議
    
    信心度調整邏輯：
    - 5日策略：大晴天 + 低信心度 -> 降級為維持標準扣款
    - 1日策略：綠燈 + 低信心度 -> 降級為觀望
    """
    # ==========================================================================
    # 資金控管建議（5日模型 + 信心度調整）
    # ==========================================================================
    if change_5d > TREND_BULLISH_THRESHOLD:
        if confidence_5d == "低":
            # 大晴天但信心度低，降級處理
            trend_emoji = "🌤️"
            trend_status = "晴時多雲"
            trend_advice = "趨勢樂觀但信心不足，建議維持標準扣款"
        else:
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
    
    # ==========================================================================
    # 進場時機建議（1日模型 + 信心度調整）
    # ==========================================================================
    if change_1d > 0:
        if confidence_1d == "低":
            # 綠燈但信心度低，降級處理
            timing_emoji = "🟡"
            timing_status = "黃燈謹慎"
            timing_advice = "短期微漲但信心不足，建議觀望"
        else:
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
# 信心度 Emoji
# =============================================================================
def get_confidence_emoji(level: str) -> str:
    """根據信心度等級返回 Emoji"""
    if level == "高":
        return "🟢"
    elif level == "中":
        return "🟡"
    else:
        return "🔴"


# =============================================================================
# 主程式
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("  🤖 TWII 投資顧問機器人 (Trade Advisor)")
    print("  智慧整合多模型，產生動態定期定額操作建議")
    print("  v2.1 - MC Dropout + 趨勢共振加分機制")
    print("=" * 70)
    
    today = date.today()
    print(f"\n📅 今日日期：{today}")
    
    # =========================================================================
    # 1. 智慧選擇最佳模型
    # =========================================================================
    print("\n" + "-" * 50)
    print("📊 模型選擇")
    print("-" * 50)
    
    # 選擇 T+1 模型
    print(f"\n[T+1 模型] 掃描 {MODELS_DIR_1D}...")
    metadata_1d = select_best_model(MODELS_DIR_1D)
    
    if metadata_1d is None:
        print("\n❌ 無法載入 T+1 模型，程式終止。")
        print("   請先執行：python twii_model_registry_multivariate.py train ...")
        return
    
    # 選擇 T+5 模型
    print(f"\n[T+5 模型] 掃描 {MODELS_DIR_5D}...")
    metadata_5d = select_best_model(MODELS_DIR_5D)
    
    if metadata_5d is None:
        print("\n❌ 無法載入 T+5 模型，程式終止。")
        print("   請先執行：python twii_model_registry_5d.py train ...")
        return
    
    # 取得參數
    lookback_1d = metadata_1d.get('lookback', 10)
    lookback_5d = metadata_5d.get('hyperparameters', {}).get('lookback', 
                  metadata_5d.get('lookback', 30))
    
    # 取得 RMSE 用於 1D 信心度評估
    rmse_1d = metadata_1d.get('metrics', {}).get('rmse', 100.0) or 100.0
    
    r2_1d = metadata_1d.get('metrics', {}).get('r2', 'N/A')
    r2_5d = metadata_5d.get('metrics', {}).get('r2', 'N/A')
    
    print(f"\n✅ 已選擇模型：")
    print(f"  [T+1] {metadata_1d['train_start']} ~ {metadata_1d['train_end']} (R²: {r2_1d}, RMSE: {rmse_1d:.2f})")
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
        print(f"  [T+5] 模型載入成功（含 Dropout 層）")
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
    # 4. 執行預測（含信心度評估）
    # =========================================================================
    print("\n" + "-" * 50)
    print("🔮 模型預測 + 信心度評估")
    print("-" * 50)
    
    try:
        # ---------------------------------------------------------------------
        # T+1 預測（標準推論 + RMSE 信心度）
        # ---------------------------------------------------------------------
        pred_1d = run_inference(model_1d, scaler_feat_1d, scaler_tgt_1d, df_processed, lookback_1d)
        change_1d = (pred_1d - current_price) / current_price
        date_1d = get_future_trading_date(last_date, 1)
        
        # 評估 1D 原始信心度
        raw_confidence_1d = evaluate_1d_confidence(pred_1d, current_price, rmse_1d)
        
        # 暫存，稍後在 T+5 預測完成後進行趨勢共振調整
        print(f"  [T+1] 預測 {date_1d}：{pred_1d:.2f} ({change_1d:+.2%}) | (信心度稍後評估)")
        
        # ---------------------------------------------------------------------
        # T+5 預測（MC Dropout 不確定性評估）
        # ---------------------------------------------------------------------
        print(f"  [T+5] 執行 MC Dropout ({MC_DROPOUT_ITERATIONS} 次迭代)...")
        
        X_5d = prepare_input_data(df_processed, scaler_feat_5d, lookback_5d)
        pred_5d, std_5d, confidence_5d = predict_with_uncertainty(
            model_5d, X_5d, scaler_tgt_5d, MC_DROPOUT_ITERATIONS
        )
        
        change_5d = (pred_5d - current_price) / current_price
        date_5d = get_future_trading_date(last_date, 5)
        conf_emoji_5d = get_confidence_emoji(confidence_5d)
        
        print(f"  [T+5] 預測 {date_5d}：{pred_5d:.2f} ({change_5d:+.2%}) | 信心度: {conf_emoji_5d} {confidence_5d}")
        
        # ---------------------------------------------------------------------
        # 趨勢共振 (Trend Alignment) 調整 T+1 信心度
        # ---------------------------------------------------------------------
        confidence_1d, trend_aligned = apply_trend_alignment(
            raw_confidence_1d, change_5d, confidence_5d
        )
        conf_emoji_1d = get_confidence_emoji(confidence_1d)
        
        # 產生加分備註
        trend_bonus_text = " (順勢加分)" if trend_aligned else ""
        
        print(f"  [T+1] 信心度評估：{conf_emoji_1d} {confidence_1d}{trend_bonus_text}")
        print(f"         風險波動 (Std): ±{std_5d:.2f} 點")
        
    except Exception as e:
        print(f"\n❌ 預測失敗：{e}")
        return
    
    # =========================================================================
    # 5. 產生投資建議（含信心度調整）
    # =========================================================================
    advice = generate_advice(change_1d, change_5d, confidence_1d, confidence_5d)
    
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
│                  R² = {r2_1d}  |  RMSE = {rmse_1d:<6.2f}                      │
├─────────────────────────────────────────────────────────────────────┤
│  波段模型 (T+5)：{metadata_5d['train_start']} ~ {metadata_5d['train_end']}                     │
│                  R² = {r2_5d}  |  Lookback = {lookback_5d} 天                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     🔮 預測數據 (含信心度)                          │
├─────────────────────────────────────────────────────────────────────┤
│  目前價格 ({last_date})              ：{current_price:>10.2f}                     │
│  T+1 預測 ({date_1d})              ：{pred_1d:>10.2f}  ({change_1d:>+6.2%}) {conf_emoji_1d} {confidence_1d}{trend_bonus_text}  │
│  T+5 預測 ({date_5d})              ：{pred_5d:>10.2f}  ({change_5d:>+6.2%}) {conf_emoji_5d} {confidence_5d}    │
│      → 風險波動 (Std)              ：   ±{std_5d:<6.2f} 點                      │
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
    
    # 綜合建議（考量信心度）
    print("=" * 70)
    
    # 高信心度條件下的綜合建議
    if change_5d > TREND_BULLISH_THRESHOLD and change_1d > 0:
        if confidence_1d == "低" or confidence_5d == "低":
            print("  🎯 綜合建議：趨勢樂觀但信心不足，建議「謹慎觀察」")
        else:
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
