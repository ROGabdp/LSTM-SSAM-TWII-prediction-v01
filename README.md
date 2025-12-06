# TWII 台股大盤預測與投資顧問系統

本專案是一個整合多種 AI 模型的台股加權指數 (TWII) 預測系統，結合了 **LSTM**, **Self-Attention (SSAM)**, **Dropout** 等深度學習技術，並包含一個智慧投資顧問機器人，提供定期定額的資金控管建議。

## ✨ 主要功能

- **� 多樣化預測模型**：
    - **T+1 短期模型** (`twii_model_registry_multivariate.py`)：預測隔日收盤價，精準捕捉短期波動。
    - **T+5 波段模型** (`twii_model_registry_5d.py`)：預測未來第 5 個交易日價格，用於判斷中期趨勢。
    - **模型優化器** (`twii_model_optimizer.py`)：自動化 Grid Search，尋找最佳超參數。
- **🤖 智慧投資顧問** (`trade_advisor.py`)：
    - 整合長短期模型訊號。
    - **信心度評估**：結合 MC Dropout (風險波動) 與 RMSE 區間判斷。
    - **動態建議**：提供「進場時機」與「資金控管」雙重建議。
- **📊 完整技術指標**：整合 KD, MACD, 成交量 (Log) 等多變量特徵。
- **🛡️ 嚴謹的模型管理**：防止資料洩漏 (Data Leakage)，自動過濾過期模型。

## 📁 檔案說明

### 1. 🤖 投資顧問機器人 (`trade_advisor.py`)
這是系統的核心使用者介面，整合所有模型的預測結果。

- **功能**：
    - 自動掃描並載入最佳的 T+1 與 T+5 模型。
    - 執行 **MC Dropout** (30次迭代) 計算 T+5 預測的不確定性 (Confidence)。
    - 結合 **RMSE** 評估 T+1 預測的信心度。
    - 根據「趨勢共振」邏輯，動態調整信心評級。
- **使用方式**：
    ```bash
    python trade_advisor.py
    ```
- **輸出範例**：
    > 🎯 綜合建議：市場短期看漲、中期樂觀，建議「加碼進場」

### 2. 📉 T+5 波段預測模型 (`twii_model_registry_5d.py`)
專為波段交易設計的模型，直接預測 5 天後的價格 (Direct Strategy)。

- **特色**：
    - **Dropout 機制**：防止過擬合，並支援 MC Dropout 不確定性估計。
    - **Direct Strategy**：直接映射 $X_t \to y_{t+5}$，避免遞迴累積誤差。
- **使用方式**：
    ```bash
    # 訓練 (自動存入 saved_models_5d/)
    python twii_model_registry_5d.py train --start 2020-01-01 --end 2025-12-05
    
    # 預測
    python twii_model_registry_5d.py predict
    ```

### 3. ⏱️ T+1 短期預測模型 (`twii_model_registry_multivariate.py`)
用於捕捉隔日行情的短期模型。

- **特色**：
    - **多變量輸入**：整合 OHLCV + KD + MACD。
    - **精細縮放**：特徵與目標使用獨立的 Scaler。
- **使用方式**：
    ```bash
    # 訓練 (自動存入 saved_models_multivariate/)
    python twii_model_registry_multivariate.py train --start 2020-07-01 --end 2025-12-05
    
    # 預測
    python twii_model_registry_multivariate.py predict
    ```

### 4. ⚡ 模型優化器 (`twii_model_optimizer.py`)
用於 T+5 模型的超參數自動搜尋。

- **功能**：
    - 支援 Grid Search (LSTM Units, Dropout Rate, Batch Size等)。
    - Time Series Cross-Validation 驗證。
    - 自動保存最佳參數至 `saved_models_optimized/best_params.json`。
- **使用方式**：
    ```bash
    # 開始搜尋最佳參數
    python twii_model_optimizer.py optimize
    
    # 使用最佳參數進行全量訓練
    python twii_model_optimizer.py train
    ```

## 📂 目錄結構

```
LSTM-SSAM-TWII-prediction-v01/
├── trade_advisor.py                    # [核心] 投資顧問機器人
├── twii_model_registry_5d.py           # [核心] T+5 模型訓練/預測
├── twii_model_registry_multivariate.py # [核心] T+1 模型訓練/預測
├── twii_model_optimizer.py             # [工具] 超參數優化
├── saved_models_5d/                    # T+5 模型存檔
├── saved_models_multivariate/          # T+1 模型存檔
└── saved_models_optimized/             # 優化後的模型存檔
```

## ⚙️ 系統需求

- Python 3.8+
- TensorFlow 2.x
- Pandas, NumPy, Scikit-learn, Yfinance, Matplotlib

```bash
pip install tensorflow pandas numpy scikit-learn yfinance matplotlib
```

## 📝 備註

本系統參考論文架構：*Sequential Self-Attention Model for Stock Price Prediction*，並針對台股特性進行了在地化調整與功能擴充。

## 📜 License

MIT License
