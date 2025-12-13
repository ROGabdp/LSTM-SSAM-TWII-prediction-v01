# -*- coding: utf-8 -*-
"""
TWII 投資顧問機器人 (Trade Advisor) v3.0
三刀流架構：智慧整合多模型，產生立體化投資建議

架構層次：
1. 【戰略層】月線 (T+20)：決定大方向與基本水位 (0% ~ 200%)
2. 【戰術層】週線 (T+5) ：進行短波段攻擊或調節 (+/- 水位)
3. 【執行層】日線 (T+1) ：判斷當下精確買賣點 (進場/觀望)

功能：
- 智慧模型選擇：自動掃描並載入最佳的 T+1, T+5, T+20 模型
- 不確定性評估：全面採用 MC Dropout (T+5, T+20) 與 RMSE (T+1) 信心度指標
- 趨勢共振機制：當長中短週期趨勢一致時，大幅提升訊號權重
- 動態資金控管：根據信心度與趨勢強度，動態計算建議扣款比例

使用方式：
  python trade_advisor.py

輸入來源：
  - 短期狙擊 (T+1): saved_models_multivariate/
  - 波段戰術 (T+5): saved_models_5d/
  - 長期戰略 (T+20): saved_models_20d/
"""

import json
import pickle
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import subprocess
import sys

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
MODELS_DIR_20D = BASE_DIR / "saved_models_20d"          # T+20 長期模型
CSV_FILE_PATH = BASE_DIR / "twii_data_from_2000_01_01.csv"
UPDATE_SCRIPT_PATH = BASE_DIR / "update_twii_data.py"

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
# =============================================================================
# 資料獲取與處理
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
    """讀取並格式化本地 CSV 資料"""
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


def download_market_data(lookback_1d: int, lookback_5d: int) -> pd.DataFrame:
    """
    取得市場資料
    策略：自動嘗試更新至最新 -> 讀取 CSV -> 取最後 N 筆
    """
    required_days = max(lookback_1d, lookback_5d) + 50
    
    print(f"[資料獲取] 準備獲取最近 {required_days} 天的資料...")
    
    # 嘗試更新資料
    run_update_script()
    
    # 讀取本地資料
    df = load_local_csv()
    
    if df.empty:
        raise ValueError("無法讀取本地資料庫 (CSV 為空)，請檢查目錄")
    
    if len(df) < required_days:
        raise ValueError(f"歷史資料不足，僅有 {len(df)} 筆，需要 {required_days} 筆")
    
    # 取最近 N 筆
    df_recent = df.tail(required_days).copy()
    
    print(f"[資料獲取] 成功取得 {len(df_recent)} 筆資料 (最新日期: {df_recent.index[-1].date()})")
    
    return df_recent


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
# =============================================================================
# 趨勢共振 (Trend Alignment) 加分機制 (v3.0 三刀流版)
# =============================================================================
def apply_trend_alignment(
    confidence_1d: str,
    change_5d: float,
    confidence_5d: str,
    change_20d: float,
    confidence_20d: str
) -> tuple:
    """
    根據 T+5 與 T+20 趨勢對 T+1 信心度進行加分
    
    邏輯 (三線共振)：
    - 若 T+20 與 T+5 皆看漲，且信心度足夠：
      → T+1 信心度大幅升級 (低->高, 中->高)
    - 若僅 T+5 看漲：
      → T+1 信心度微幅升級
    """
    is_t20_bullish = change_20d > 0
    is_t5_bullish = change_5d > 0
    
    score = 0
    if is_t20_bullish and confidence_20d in ["高", "中"]:
        score += 1
    if is_t5_bullish and confidence_5d in ["高", "中"]:
        score += 1
        
    rank_map = {"低": 0, "中": 1, "高": 2}
    current_rank = rank_map[confidence_1d]
    
    if score == 2: # 三線共振（月、週皆多）
        current_rank = 2 # 直接升高
        is_aligned = True
    elif score == 1: # 部分共振
        current_rank = min(2, current_rank + 1)
        is_aligned = True
    else:
        is_aligned = False
        
    inv_map = {0: "低", 1: "中", 2: "高"}
    return inv_map[current_rank], is_aligned


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
# =============================================================================
# 投資建議產生 (v3.0 三刀流版)
# =============================================================================
def generate_advice(
    change_1d: float,
    change_5d: float,
    change_20d: float,
    confidence_1d: str,
    confidence_5d: str,
    confidence_20d: str
) -> Dict[str, str]:
    """
    產生三層次投資建議
    
    層次結構：
    1. 戰略 (Strategy) - T+20: 決定水位 (0% ~ 200%)
    2. 戰術 (Tactics)  - T+5 : 微調水位 (+/-)
    3. 執行 (Action)   - T+1 : 決定進出
    """
    
    # --------------------------------------------------------------------------
    # 1. 戰略層 (T+20 月線) - 決定大方向
    # --------------------------------------------------------------------------
    strategy_score = 2 # 預設標準水位 (2分)
    
    if change_20d > 0.05:          # 大漲 > 5%
        strategy_status = "極度樂觀"
        strategy_emoji = "🚀"
        strategy_score = 4
    elif change_20d > 0.01:        # 緩漲 > 1%
        strategy_status = "樂觀看多"
        strategy_emoji = "📈" 
        strategy_score = 3
    elif change_20d > -0.01:       # 盤整 -1% ~ 1%
        strategy_status = "區間震盪"
        strategy_emoji = "⚖️"
        strategy_score = 2
    elif change_20d > -0.05:       # 緩跌 > -5%
        strategy_status = "保守看空"
        strategy_emoji = "📉"
        strategy_score = 1
    else:                          # 大跌 < -5%
        strategy_status = "極度悲觀"
        strategy_emoji = "�️"
        strategy_score = 0
        
    # 信心度懲罰
    if confidence_20d == "低":
        strategy_score = max(2, strategy_score - 1) if strategy_score > 2 else strategy_score # 樂觀時收斂
        strategy_note = "(信心不足，收斂水位)"
    else:
        strategy_note = ""

    # --------------------------------------------------------------------------
    # 2. 戰術層 (T+5 週線) - 決定攻擊力度
    # --------------------------------------------------------------------------
    t5_score = 0
    if change_5d > 0.015:
        t5_emoji = "🌤️"
        t5_score = 1
    elif change_5d < -0.015:
        t5_emoji = "🌧️"
        t5_score = -1
    else:
        t5_emoji = "☁️"
        
    # 信心度過濾：若 T+5 信心低，忽略其訊號
    if confidence_5d == "低":
        t5_score = 0
        t5_note = "(信心不足，忽略短波)"
    else:
        t5_note = ""
        
    # 計算最終建議水位
    final_score = strategy_score + t5_score
    final_score = max(0, min(5, final_score)) # 限制在 0~5 分
    
    # 水位對照表
    allocation_map = {
        0: "暫停扣款 (0%)",
        1: "減碼扣款 (50%)",
        2: "標準扣款 (100%)",
        3: "積極扣款 (120%)",
        4: "強力加碼 (150%)",
        5: "全力進攻 (200%)"
    }
    allocation_advice = allocation_map[final_score]

    # --------------------------------------------------------------------------
    # 3. 執行層 (T+1 日線) - 決定今日動作
    # --------------------------------------------------------------------------
    if change_1d > 0:
        if confidence_1d == "低":
            action_emoji = "🟡"
            action_status = "黃燈 (微漲但信心低)"
            action_advice = "可小額進場，或觀望"
        else:
            action_emoji = "🟢"
            action_status = "綠燈 (看漲)"
            action_advice = f"建議今日執行：{allocation_advice.split(' ')[0]}"
    else:
        action_emoji = "�"
        action_status = "紅燈 (看跌)"
        action_advice = "建議暫緩，等待更低點"

    return {
        'strategy_status': f"{strategy_emoji} {strategy_status} {strategy_note}",
        'tactics_status': f"{t5_emoji} 週線修正 {t5_score:+d} {t5_note}",
        'allocation_advice': f"💰 建議水位：{allocation_advice}",
        'action_status': f"{action_emoji} {action_status}",
        'action_advice': action_advice,
        'final_score': final_score
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
# =============================================================================
# 主程式
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("  🤖 TWII 投資顧問機器人 (Trade Advisor v3.0)")
    print("  三刀流架構：月線戰略(T+20) -> 週線戰術(T+5) -> 日線狙擊(T+1)")
    print("=" * 70)
    
    today = date.today()
    print(f"\n📅 今日日期：{today}")
    
    # -------------------------------------------------------------------------
    # 1. 智慧選擇最佳模型
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("📊 模型選擇")
    print("-" * 50)
    
    print(f"\n[T+1 短期] 掃描 {MODELS_DIR_1D}...")
    meta_1d = select_best_model(MODELS_DIR_1D)
    
    print(f"\n[T+5 波段] 掃描 {MODELS_DIR_5D}...")
    meta_5d = select_best_model(MODELS_DIR_5D)
    
    print(f"\n[T+20 長期] 掃描 {MODELS_DIR_20D}...")
    meta_20d = select_best_model(MODELS_DIR_20D)
    
    if any(m is None for m in [meta_1d, meta_5d, meta_20d]):
        print("\n❌ 部分模型缺失，請先確認所有模型皆已訓練完畢。")
        return

    # 顯示模型資訊
    print(f"\n✅ 已就緒模型：")
    print(f"  [T+1]  {meta_1d['train_start']}~{meta_1d['train_end']} (RMSE: {meta_1d['metrics'].get('rmse', 0):.2f})")
    print(f"  [T+5]  {meta_5d['train_start']}~{meta_5d['train_end']} (R²: {meta_5d['metrics'].get('r2', 0)})")
    print(f"  [T+20] {meta_20d['train_start']}~{meta_20d['train_end']} (R²: {meta_20d['metrics'].get('r2', 0)})")

    # -------------------------------------------------------------------------
    # 2. 載入與推論
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("🔧 載入與推論")
    print("-" * 50)
    
    try:
        # 下載資料
        df_raw = download_market_data(
            lookback_1d=meta_1d.get('lookback', 10),
            lookback_5d=meta_5d.get('hyperparameters', {}).get('lookback', 30)
        )
        # 用最長的 lookback 確保資料足夠，這裡簡化處理取最大值加緩衝
        # 注意：正確做法應該要看 20d 的 lookback，但通常 20d 與 5d 差不多或更長
        # 我們直接用下載下來的 df 進行處理
        
        df_processed = add_technical_indicators(df_raw)
        if 'Adj Close' not in df_processed.columns:
            df_processed['Adj Close'] = df_processed['Close']
            
        current_price = df_processed['Adj Close'].iloc[-1]
        last_date = df_processed.index[-1].date()
        
        # 載入模型並預測
        # T+1
        m1, sc_f1, sc_t1, _ = load_model_artifacts(MODELS_DIR_1D, meta_1d)
        p1 = run_inference(m1, sc_f1, sc_t1, df_processed, meta_1d.get('lookback', 10))
        c1_pct = (p1 - current_price) / current_price
        d1 = get_future_trading_date(last_date, 1)
        rmse_1d = meta_1d.get('metrics', {}).get('rmse', 100.0)
        conf_1d_raw = evaluate_1d_confidence(p1, current_price, rmse_1d)
        
        # T+5 (MC Dropout)
        m5, sc_f5, sc_t5, _ = load_model_artifacts(MODELS_DIR_5D, meta_5d)
        lb5 = meta_5d.get('hyperparameters', {}).get('lookback', 30)
        X5 = prepare_input_data(df_processed, sc_f5, lb5)
        p5, std5, conf5 = predict_with_uncertainty(m5, X5, sc_t5, MC_DROPOUT_ITERATIONS)
        c5_pct = (p5 - current_price) / current_price
        d5 = get_future_trading_date(last_date, 5)

        # T+20 (MC Dropout) - 新增
        print(f"  [T+20] 執行 MC Dropout ({MC_DROPOUT_ITERATIONS} 次)...")
        m20, sc_f20, sc_t20, _ = load_model_artifacts(MODELS_DIR_20D, meta_20d)
        lb20 = meta_20d.get('hyperparameters', {}).get('lookback', 60)
        X20 = prepare_input_data(df_processed, sc_f20, lb20)
        p20, std20, conf20 = predict_with_uncertainty(m20, X20, sc_t20, MC_DROPOUT_ITERATIONS)
        c20_pct = (p20 - current_price) / current_price
        d20 = get_future_trading_date(last_date, 20)
        
        # 趨勢共振調整 T+1
        conf_1d_final, is_aligned = apply_trend_alignment(
            conf_1d_raw, c5_pct, conf5, c20_pct, conf20
        )
        
    except Exception as e:
        print(f"\n❌ 執行過程發生錯誤：{e}")
        import traceback
        traceback.print_exc()
        return

    # -------------------------------------------------------------------------
    # 3. 產生報告
    # -------------------------------------------------------------------------
    advice = generate_advice(c1_pct, c5_pct, c20_pct, conf_1d_final, conf5, conf20)
    
    e1 = get_confidence_emoji(conf_1d_final)
    e5 = get_confidence_emoji(conf5)
    e20 = get_confidence_emoji(conf20)
    bonus = " (🔥三線共振)" if is_aligned else ""
    
    print("\n" + "=" * 70)
    print("  📋 三刀流投資分析報告")
    print("=" * 70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  📊 市場預測                                                        │
├─────────────────────────────────────────────────────────────────────┤
│  基準日：{last_date:<12} 收盤：{current_price:<10.2f}                         │
├─────────────────────────────────────────────────────────────────────┤
│  [戰略] 月線 T+20 ({d20}): {p20:>8.2f} ({c20_pct:>+6.2%}) {e20} {conf20} (±{std20:.0f})      │
│  [戰術] 週線 T+5  ({d5}): {p5:>8.2f} ({c5_pct:>+6.2%}) {e5} {conf5} (±{std5:.0f})      │
│  [執行] 日線 T+1  ({d1}): {p1:>8.2f} ({c1_pct:>+6.2%}) {e1} {conf_1d_final}{bonus}       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  💡 三刀流決策                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  1. 戰略方向：{advice['strategy_status']:<46}│
│  2. 戰術調節：{advice['tactics_status']:<46}│
│  3. 資金水位：{advice['allocation_advice']:<46}│
│  ───────────────────────────────────────────────────────────────────│
│  ⚡ 今日執行：{advice['action_advice']:<46}│
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print("=" * 70)
    if advice['final_score'] >= 4:
        print("  🚀 綜合評價：強力進攻訊號，把握機會！")
    elif advice['final_score'] <= 1:
        print("  🛡️ 綜合評價：風險偏高，建議防守為上。")
    else:
        print("  ⚖️ 綜合評價：市場穩健，按紀律執行。")
    print("=" * 70)
    print("\n⚠️  純屬 AI 預測研究，不代表投資建議。")

if __name__ == "__main__":
    main()
