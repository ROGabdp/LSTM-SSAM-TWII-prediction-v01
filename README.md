# TWII 台股大盤預測與投資顧問系統

本專案是一個整合多種 AI 模型的台股加權指數 (TWII) 預測系統，結合了 **LSTM**, **Self-Attention (SSAM)**, **Dropout** 等深度學習技術，並包含一個智慧投資顧問機器人，提供定期定額的資金控管建議。

## ✨ 主要功能

- **🧩 多樣化預測模型**：
    - **T+1 短期模型** (`twii_model_registry_multivariate.py`)：預測隔日收盤價，精準捕捉短期波動。
    - **T+5 波段模型** (`twii_model_registry_5d.py`)：預測未來第 5 個交易日價格，用於判斷中期趨勢。
    - **T+20 長期模型** (`twii_model_registry_20d.py`)：預測未來第 20 個交易日價格，用於長線佈局參考。
    - **模型優化器** (`twii_model_optimizer.py`)：自動化 Grid Search，尋找最佳超參數。
- **🤖 智慧投資顧問 (三刀流架構)** (`trade_advisor.py`)：
    - **v3.0 升級**：整合 T+20 (戰略)、T+5 (戰術)、T+1 (執行) 三層次決策系統。
    - **不確定性評估**：全面導入 **MC Dropout** (T+5/T+20) 與 **RMSE** (T+1) 信心度指標。
    - **趨勢共振**：當月線、週線、日線趨勢一致時，提供強力訊號。
    - **立體化建議**：提供從「總資金水位」、「加減碼調節」到「今日進出」的完整策略。
- **📊 完整技術指標**：整合 KD, MACD, 成交量 (Log) 等多變量特徵。
- **🛡️ 穩定資料處理**：優先讀取本地 `twii_data_from_2000_01_01.csv`，並具備自動增量更新機制 (`update_twii_data.py`)，減少網路依賴。
- **🛡️ 嚴謹的模型管理**：防止資料洩漏 (Data Leakage)，自動過濾過期模型，並基於預測目標日期 (Target Date) 智慧選擇最佳模型。

## 📁 檔案說明

### 1. 🤖 投資顧問機器人 (`trade_advisor.py`) [v3.0]
這是系統的核心決策中樞，採用「三刀流」架構，提供立體化的投資視角。

- **三刀流架構**：
    1. **戰略層 (T+20 月線)**：決定戰略方向與基本水位 (0% ~ 200%)。
       - *參考模型：`saved_models_20d/` (128 units, dropout 0.5)*
    2. **戰術層 (T+5 週線)**：在戰略基礎上進行短波段攻擊或調節。
       - *參考模型：`saved_models_5d/` (256 units, dropout 0.2)*
    3. **執行層 (T+1 日線)**：判斷當下的精確買賣點。
       - *參考模型：`saved_models_multivariate/`*

- **特色**：
    - **MC Dropout 全面導入**：T+5 與 T+20 模型皆執行 30 次迭代，計算預測分佈與風險波動 (Std)。
    - **趨勢共振**：當「月線多頭」遇上「週線攻擊」且「日線轉強」，觸發 🔥 **三線共振** 強力訊號。
    - **資料一致性**：自動呼叫 `update_twii_data.py`，確保與訓練環境使用相同的本地數據。

- **使用方式**：
    ```bash
    python trade_advisor.py
    ```
- **輸出範例**：
    > 📊 [戰略] 月線 T+20: ↗ 極度樂觀 (信心低，收斂水位)
    > 💡 [決策] 建議水位：積極扣款 (120%) | 今日執行：進場

### 2. 📉 T+5 波段預測模型 (`twii_model_registry_5d.py`)
專為波段交易設計的模型，直接預測 5 天後的價格 (Direct Strategy)。

- **特色**：
    - **Dropout 機制**：防止過擬合，並支援 MC Dropout 不確定性估計。
    - **Direct Strategy**：直接映射 $X_t \to y_{t+5}$，避免遞迴累積誤差。
    - **智慧模型選擇**：預測時自動根據目標日期挑選包含最新資料的模型。
- **使用方式**：
    ```bash
    # 訓練 (參數可選，未指定則自動計算預設長度)
    python twii_model_registry_5d.py train --start 2020-01-01 --end 2025-12-05
    
    # 預測
    python twii_model_registry_5d.py predict
    ```

### 3. 📅 T+20 長期預測模型 (`twii_model_registry_20d.py`)
專為長期趨勢判斷設計，直接預測 20 天後的價格。

- **最佳化設定** (基於 2025-12-13 Grid Search)：
    - Lookback: **60 天** (鎖定長期趨勢)
    - LSTM Units: **128** (小模型泛化能力更佳)
    - Dropout: **0.5** (最大化正則化，對抗長期雜訊)
    - Batch Size: **16** (小批量增加隨機性，跳出局部最優)
- **使用方式**：
    ```bash
    # 訓練 (預設區間為 2019-07-01 ~ 2025-12-13，經測試最佳)
    python twii_model_registry_20d.py train
    
    # 預測
    python twii_model_registry_20d.py predict
    ```

### 4. ⏱️ T+1 短期預測模型 (`twii_model_registry_multivariate.py`)
用於捕捉隔日行情的短期模型。

- **特色**：
    - **多變量輸入**：整合 OHLCV + KD + MACD。
    - **精細縮放**：特徵與目標使用獨立的 Scaler。
- **使用方式**：
    ```bash
    # 訓練 (自動存入 saved_models_multivariate/)
    python twii_model_registry_multivariate.py train
    
    # 預測
    python twii_model_registry_multivariate.py predict
    ```

### 5. ⚡ 模型優化器 (`twii_model_optimizer.py` / `_20d.py`)
用於 T+5 與 T+20 模型的超參數自動搜尋。

- **功能**：
    - **T+5 優化器** (`twii_model_optimizer_5d.py`): 針對波段預測優化，模型存於 `saved_models_optimized_5d/`。
    - **T+20 優化器** (`twii_model_optimizer_20d.py`): 針對長期預測優化，重點在於防止過擬合 (Dropout/Small Batch)，模型存於 `saved_models_optimized_20d/`。
- **特色**：
    - 支援 Grid Search (LSTM Units, Dropout Rate, Batch Size等)。
    - Time Series Cross-Validation 驗證。
    - **邊界效應分析**：自動識別參數邊界，引導進一步的 Search Space 調整。
- **使用方式**：
    ```bash
    # T+20 優化
    python twii_model_optimizer_20d.py optimize
    ```

## 📂 目錄結構

```
LSTM-SSAM-TWII-prediction-v01/
├── trade_advisor.py                    # [核心] 投資顧問機器人
├── twii_model_registry_5d.py           # [核心] T+5 模型訓練/預測
├── twii_model_registry_20d.py          # [核心] T+20 模型訓練/預測
├── twii_model_registry_multivariate.py # [核心] T+1 模型訓練/預測
├── twii_model_optimizer.py             # [工具] 超參數優化
├── update_twii_data.py                 # [工具] 自動更新股價資料腳本
├── twii_data_from_2000_01_01.csv       # [資料] 本地歷史資料庫
├── saved_models_5d/                    # T+5 模型存檔
├── saved_models_20d/                   # T+20 模型存檔
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
