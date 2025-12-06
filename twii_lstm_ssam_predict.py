# -*- coding: utf-8 -*-
"""
TWII 台股大盤走勢預測模型
使用 LSTM + Sequential Self-Attention (SSAM) 架構

基於論文規格實作：
- 資料來源：yfinance (^TWII)
- 模型架構：LSTM(50) + Self-Attention + Dense(1)
- 訓練配置：Adam, MSE, batch_size=10, epochs=50
"""

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
# 中文字型設定（解決 matplotlib 中文顯示問題）
# =============================================================================
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# =============================================================================
# 1. 資料獲取與預處理
# =============================================================================

def download_twii_data(years: int = 10) -> pd.DataFrame:
    """
    使用 yfinance 下載台股大盤 (^TWII) 歷史資料
    
    Args:
        years: 下載過去幾年的資料
    
    Returns:
        DataFrame 包含歷史價格資料
    """
    print(f"[資料獲取] 正在下載 ^TWII 過去 {years} 年歷史資料...")
    ticker = yf.Ticker("^TWII")
    df = ticker.history(period=f"{years}y")
    
    if df.empty:
        raise ValueError("無法取得 ^TWII 資料，請檢查網路連線")
    
    print(f"[資料獲取] 成功下載 {len(df)} 筆資料")
    print(f"[資料獲取] 資料期間：{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    
    return df


def preprocess_data(df: pd.DataFrame, lookback: int = 10, train_ratio: float = 0.9):
    """
    資料預處理：正規化、分割、建立滑動視窗序列
    
    Args:
        df: 原始價格資料
        lookback: 回看天數（時間步長）
        train_ratio: 訓練集比例
    
    Returns:
        X_train, y_train, X_test, y_test, scaler
    """
    # 論文規格：僅使用 Adj Close（調整後收盤價）作為輸入特徵
    # 注意：yfinance 的 history() 回傳的 Close 已經是調整後價格
    data = df['Close'].values.reshape(-1, 1)
    
    # 論文規格：Min-Max Scaling 正規化至 0~1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    print(f"[預處理] 資料正規化完成 (Min-Max Scaling)")
    
    # 論文規格：滑動視窗建立序列 (Lookback = 10)
    # 使用 t-10 到 t-1 的資料來預測第 t 天
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # 論文規格：訓練集 90%、測試集 10%
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 重塑為 LSTM 輸入格式：(samples, time_steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    print(f"[預處理] 訓練集：{len(X_train)} 筆 | 測試集：{len(X_test)} 筆")
    print(f"[預處理] 輸入形狀：{X_train.shape} (樣本數, 時間步長, 特徵數)")
    
    return X_train, y_train, X_test, y_test, scaler


# =============================================================================
# 2. 自訂 Self-Attention Layer
# =============================================================================

class SelfAttention(layers.Layer):
    """
    Sequential Self-Attention Layer (論文 SSAM 架構)
    
    實作注意力機制：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    此層接收 LSTM 的 3D 輸出 (batch, steps, units)，
    透過注意力機制學習序列中各時間步的相對重要性。
    """
    
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """
        建立 Q, K, V 的權重矩陣
        
        input_shape: (batch_size, time_steps, units)
        """
        self.units = input_shape[-1]  # LSTM 輸出的隱藏單元數
        
        # Query 權重矩陣
        self.W_q = self.add_weight(
            name='W_query',
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Key 權重矩陣
        self.W_k = self.add_weight(
            name='W_key',
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Value 權重矩陣
        self.W_v = self.add_weight(
            name='W_value',
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(SelfAttention, self).build(input_shape)
    
    def call(self, inputs):
        """
        前向傳播：計算 Self-Attention
        
        Args:
            inputs: LSTM 輸出 (batch, steps, units)
        
        Returns:
            注意力加權後的輸出 (batch, steps, units)
        """
        # 計算 Q, K, V
        Q = tf.matmul(inputs, self.W_q)  # (batch, steps, units)
        K = tf.matmul(inputs, self.W_k)  # (batch, steps, units)
        V = tf.matmul(inputs, self.W_v)  # (batch, steps, units)
        
        # 計算注意力分數：QK^T
        # (batch, steps, units) @ (batch, units, steps) = (batch, steps, steps)
        attention_scores = tf.matmul(Q, K, transpose_b=True)
        
        # 論文規格：除以 sqrt(d_k) 進行縮放，避免梯度消失
        d_k = tf.cast(self.units, tf.float32)
        attention_scores = attention_scores / tf.math.sqrt(d_k)
        
        # Softmax 正規化得到注意力權重
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # 加權求和：注意力權重 @ V
        output = tf.matmul(attention_weights, V)
        
        return output
    
    def get_config(self):
        """序列化配置（用於模型儲存）"""
        config = super(SelfAttention, self).get_config()
        return config


# =============================================================================
# 3. LSTM-SSAM 模型架構
# =============================================================================

def build_lstm_ssam_model(time_steps: int = 10, features: int = 1, lstm_units: int = 50):
    """
    建立 LSTM + Self-Attention 混合模型（論文 LSTM-SSAM 架構）
    
    架構流程：
    Input(10,1) → LSTM(50, return_sequences=True) → SelfAttention → Flatten → Dense(1)
    
    Args:
        time_steps: 時間步長（回看天數）
        features: 輸入特徵數
        lstm_units: LSTM 隱藏單元數
    
    Returns:
        編譯完成的 Keras Model
    """
    # 輸入層：形狀 (time_steps, features) = (10, 1)
    inputs = layers.Input(shape=(time_steps, features), name='input_layer')
    
    # LSTM 層：50 個隱藏單元，return_sequences=True（關鍵：輸出完整序列給 Attention）
    lstm_out = layers.LSTM(
        units=lstm_units,
        return_sequences=True,  # 論文規格：必須回傳完整序列
        name='lstm_layer'
    )(inputs)
    
    # Self-Attention 層：學習序列中各時間步的相對重要性
    attention_out = SelfAttention(name='self_attention')(lstm_out)
    
    # Flatten 層：將 3D 輸出攤平為 1D
    flatten_out = layers.Flatten(name='flatten_layer')(attention_out)
    
    # 輸出層：Dense(1)，線性激活函數（預測連續值）
    outputs = layers.Dense(units=1, activation='linear', name='output_layer')(flatten_out)
    
    # 建立模型
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_SSAM_Model')
    
    # 論文規格：Adam 優化器、MSE 損失函數
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    print("\n[模型架構] LSTM-SSAM 模型建立完成")
    model.summary()
    
    return model


# =============================================================================
# 4. 訓練與評估
# =============================================================================

def train_model(model, X_train, y_train, X_test, y_test, epochs: int = 50, batch_size: int = 10):
    """
    訓練模型
    
    Args:
        model: LSTM-SSAM 模型
        X_train, y_train: 訓練資料
        X_test, y_test: 驗證資料
        epochs: 訓練輪數
        batch_size: 批次大小
    
    Returns:
        訓練歷史紀錄
    """
    print(f"\n[訓練] 開始訓練... (epochs={epochs}, batch_size={batch_size})")
    
    # 設定早停機制（可選，防止過擬合）
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    print("[訓練] 訓練完成！")
    
    return history


def evaluate_model(model, X_test, y_test, scaler):
    """
    評估模型並輸出指標
    
    Args:
        model: 訓練完成的模型
        X_test, y_test: 測試資料（正規化後）
        scaler: 用於反正規化的 MinMaxScaler
    
    Returns:
        y_actual, y_predicted: 反正規化後的實際值與預測值
    """
    print("\n[評估] 正在進行預測...")
    
    # 預測
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # 反正規化：將 0~1 的值轉換回原始價格
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_predicted = scaler.inverse_transform(y_pred_scaled).flatten()
    
    # 計算評估指標
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    r2 = r2_score(y_actual, y_predicted)
    
    print("\n" + "=" * 50)
    print("📊 模型評估結果")
    print("=" * 50)
    print(f"  RMSE (均方根誤差)  : {rmse:.2f} 點")
    print(f"  R² Score (決定係數): {r2:.4f}")
    print("=" * 50)
    
    return y_actual, y_predicted


def plot_results(y_actual, y_predicted, df, train_ratio: float = 0.9, lookback: int = 10):
    """
    繪製預測結果圖表
    
    Args:
        y_actual: 實際價格
        y_predicted: 預測價格
        df: 原始資料 DataFrame（用於取得日期索引）
        train_ratio: 訓練集比例
        lookback: 回看天數
    """
    # 計算測試集對應的日期索引
    total_samples = len(df) - lookback
    train_size = int(total_samples * train_ratio)
    test_dates = df.index[lookback + train_size:]
    
    # 確保日期數量與預測數量一致
    test_dates = test_dates[:len(y_actual)]
    
    # 繪圖
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, y_actual, label='實際價格 (Actual)', color='blue', linewidth=1.5)
    plt.plot(test_dates, y_predicted, label='預測價格 (Predicted)', color='red', linewidth=1.5, alpha=0.8)
    
    plt.title('TWII 台股大盤預測 - LSTM + Self-Attention Model', fontsize=14)
    plt.xlabel('日期 (Date)', fontsize=12)
    plt.ylabel('價格 (Price)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 儲存圖片
    output_path = 'twii_prediction_result.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n[視覺化] 圖表已儲存至：{output_path}")
    
    plt.show()


# =============================================================================
# 5. 主程式
# =============================================================================

def main():
    """主程式入口"""
    
    print("\n" + "=" * 60)
    print("  TWII 台股大盤走勢預測 - LSTM + Self-Attention Model")
    print("=" * 60)
    
    # 設定隨機種子以確保結果可重現
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 論文規格參數
    LOOKBACK = 10        # 時間步長（回看天數）
    TRAIN_RATIO = 0.9    # 訓練集比例
    LSTM_UNITS = 50      # LSTM 隱藏單元數
    EPOCHS = 50          # 訓練輪數
    BATCH_SIZE = 10      # 批次大小
    
    # 1. 下載資料
    df = download_twii_data(years=10)
    
    # 2. 預處理
    X_train, y_train, X_test, y_test, scaler = preprocess_data(
        df, lookback=LOOKBACK, train_ratio=TRAIN_RATIO
    )
    
    # 3. 建立模型
    model = build_lstm_ssam_model(
        time_steps=LOOKBACK,
        features=1,
        lstm_units=LSTM_UNITS
    )
    
    # 4. 訓練模型
    history = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=EPOCHS, batch_size=BATCH_SIZE
    )
    
    # 5. 評估模型
    y_actual, y_predicted = evaluate_model(model, X_test, y_test, scaler)
    
    # 6. 繪製結果
    plot_results(y_actual, y_predicted, df, train_ratio=TRAIN_RATIO, lookback=LOOKBACK)
    
    print("\n✅ 預測完成！")


if __name__ == "__main__":
    main()
