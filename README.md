# TWII 台股大盤預測系統

基於 **LSTM + Self-Attention (SSAM)** 架構的台股加權指數預測模型，具備完整的模型版本管理與自動選擇功能。

## ✨ 功能特色

- 🧠 **LSTM-SSAM 架構**：結合長短期記憶網路與自注意力機制
- 📦 **模型註冊系統**：自動管理多版本模型，智慧選擇最佳版本
- 🔮 **多步遞迴預測**：支援預測未來多個交易日
- 📊 **效能指標追蹤**：自動記錄 R² Score 和 RMSE
- 📈 **視覺化輸出**：訓練完成後自動生成預測結果圖表

## 📁 專案結構

```
LSTM-SSAM-TWII-prediction-v01/
├── twii_model_registry.py      # 主程式（模型註冊系統）
├── twii_lstm_ssam_predict.py   # 基礎版本（單次訓練/預測）
├── saved_models/               # 模型倉庫
│   ├── model_*.keras           # Keras 模型檔
│   ├── scaler_*.pkl            # MinMaxScaler 縮放器
│   ├── meta_*.json             # 元資料（含效能指標）
│   └── plot_*.png              # 訓練結果視覺化圖表
└── README.md
```

## 🚀 快速開始

### 安裝依賴

```bash
pip install numpy pandas matplotlib yfinance scikit-learn tensorflow
```

### 訓練模型

```bash
python twii_model_registry.py train --start 2020-01-01 --end 2025-12-05
```

訓練完成後會在 `saved_models/` 目錄產生：
- `model_{start}_{end}.keras` - 訓練好的模型
- `scaler_{start}_{end}.pkl` - 資料縮放器
- `meta_{start}_{end}.json` - 元資料（含 R²、RMSE）
- `plot_{start}_{end}.png` - 預測結果視覺化

### 預測價格

```bash
# 預測明天
python twii_model_registry.py predict

# 預測指定日期
python twii_model_registry.py predict --target_date 2025-12-15
```

輸出範例：
```
[搜尋] 找到 6 個可用模型：
  1. model_2021-01-01_2025-12-05 (R²: 0.9682) -> Selected (Best Match)
  2. model_2017-01-01_2025-12-05 (R²: 0.9532) -> Backup (Lower R²)

==================================================
🔮 TWII 預測結果 - 目標日期：2025-12-08
==================================================
  最近收盤價 (2025-12-05) : 27980.89
  預測價格   (2025-12-08) : 27634.40
  預期變化   : -346.49 (-1.24%)
  趨勢判斷   : 📉 看跌
==================================================
```

## 🧮 模型選擇邏輯

系統會自動選擇最適合的模型，排序優先順序：

1. **Recency（時效性）**：`train_end` 越新越好
2. **Performance（效能）**：R² Score 越高越好
3. **Freshness（專精度）**：`train_start` 越新，模型越專精於近期市場

### 過濾條件
- ✅ 訓練天數 ≥ 1460 天（4 年）
- ✅ `train_end` < 目標預測日期（避免資料洩漏）

## ⚙️ 設定參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `LOOKBACK` | 10 | 回看天數（時間步長） |
| `LSTM_UNITS` | 50 | LSTM 隱藏單元數 |
| `EPOCHS` | 50 | 訓練輪數 |
| `BATCH_SIZE` | 10 | 批次大小 |
| `MIN_TRAIN_DAYS` | 1460 | 最低訓練天數（4 年） |
| `MODEL_STALE_DAYS` | 180 | 模型過期警告閾值 |

## 📊 元資料格式

```json
{
  "train_start": "2021-01-01",
  "train_end": "2025-12-05",
  "lookback": 10,
  "price_min": 15159.86,
  "price_max": 24416.67,
  "training_timestamp": "2025-12-06T11:45:00",
  "metrics": {
    "rmse": 430.40,
    "r2": 0.9682
  }
}
```

## 📝 參考論文

本專案架構參考自論文：
> Sequential Self-Attention Model for Stock Price Prediction

## 📜 授權

MIT License
