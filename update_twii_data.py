"""
TWII 資料更新腳本

功能說明：
1. 從 twii_data_from_2000_01_01.csv 讀取已有的歷史資料
2. 從台灣證交所 (TWSE) API 下載最新的成交值資料
3. 從 yfinance 下載 OHLC 資料
4. 合併並更新 CSV 檔案

重要：
- volume 欄位的單位是「億元」(成交金額/1e8)
- 證交所 API 回傳的成交金額單位是「元」
- yfinance 回傳的 Volume 是「成交股數」，我們只使用其 OHLC 資料

使用方式：
python update_twii_data.py
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time
import os


# ============================================================================
# 常數定義
# ============================================================================
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "twii_data_from_2000_01_01.csv")
TWSE_API_URL = "https://www.twse.com.tw/exchangeReport/FMTQIK"
YFINANCE_TICKER = "^TWII"

# 注意：volume 欄位的單位是「億元」
# 證交所 API 回傳的「成交金額」單位是「元」，需除以 1e8 轉換為「億元」
VOLUME_CONVERSION_FACTOR = 1e8  # 元 -> 億元


# ============================================================================
# 輔助函數
# ============================================================================
def parse_roc_date(roc_date_str: str) -> pd.Timestamp:
    """
    將民國年日期字串轉換為 pandas Timestamp
    例如: "114/12/09" -> Timestamp('2025-12-09')
    """
    parts = roc_date_str.split('/')
    year = int(parts[0]) + 1911  # 民國年轉西元年
    month = int(parts[1])
    day = int(parts[2])
    return pd.Timestamp(year, month, day)


def format_date_for_csv(ts: pd.Timestamp) -> str:
    """
    將 Timestamp 格式化為 CSV 中的日期格式
    例如: Timestamp('2025-12-09') -> "2025/12/9"
    """
    return f"{ts.year}/{ts.month}/{ts.day}"


def fetch_twse_monthly_data(year: int, month: int) -> pd.DataFrame:
    """
    從證交所 API 下載指定月份的每日市場成交資訊
    
    參數:
        year: 西元年 (例如 2025)
        month: 月份 (1-12)
    
    回傳:
        DataFrame 包含 ['date', 'close', 'volume'] 欄位
        volume 單位已轉換為「億元」
    """
    date_str = f"{year}{month:02d}01"
    
    params = {
        'response': 'json',
        'date': date_str
    }
    
    try:
        response = requests.get(TWSE_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('stat') != 'OK' or 'data' not in data:
            print(f"  警告: {year}/{month:02d} 無資料或 API 回傳錯誤")
            return pd.DataFrame()
        
        records = []
        for row in data['data']:
            # row format: [日期, 成交股數, 成交金額, 成交筆數, 發行量加權股價指數, 漲跌點數]
            roc_date = row[0]
            trading_value_str = row[2].replace(',', '')  # 成交金額 (元)
            index_str = row[4].replace(',', '')  # 發行量加權股價指數
            
            date = parse_roc_date(roc_date)
            
            # 成交金額單位: 元 -> 億元
            trading_value = float(trading_value_str)
            volume_in_yi = round(trading_value / VOLUME_CONVERSION_FACTOR, 2)
            
            close = float(index_str)
            
            records.append({
                'date': date,
                'close': close,
                'volume': volume_in_yi  # 單位: 億元
            })
        
        df = pd.DataFrame(records)
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"  錯誤: 下載 {year}/{month:02d} 資料失敗 - {e}")
        return pd.DataFrame()


def fetch_twse_data_range(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    從證交所 API 下載指定日期範圍的資料
    
    參數:
        start_date: 開始日期 (包含)
        end_date: 結束日期 (包含)
    
    回傳:
        DataFrame 包含 ['date', 'close', 'volume'] 欄位
    """
    all_data = []
    
    # 計算需要下載的月份範圍
    current = start_date.replace(day=1)
    end_month = end_date.replace(day=1)
    
    while current <= end_month:
        print(f"  下載證交所資料: {current.year}/{current.month:02d}...")
        
        monthly_data = fetch_twse_monthly_data(current.year, current.month)
        if not monthly_data.empty:
            all_data.append(monthly_data)
        
        # 下一個月
        if current.month == 12:
            current = pd.Timestamp(current.year + 1, 1, 1)
        else:
            current = pd.Timestamp(current.year, current.month + 1, 1)
        
        # 避免請求過於頻繁
        time.sleep(1.0)
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # 過濾日期範圍
    combined = combined[(combined['date'] >= start_date) & (combined['date'] <= end_date)]
    combined = combined.sort_values('date').reset_index(drop=True)
    
    return combined


def fetch_yfinance_ohlc(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    從 yfinance 下載 TWII 的 OHLC 資料
    
    參數:
        start_date: 開始日期 (包含)
        end_date: 結束日期 (包含)
    
    回傳:
        DataFrame 包含 ['date', 'open', 'high', 'low', 'close'] 欄位
    """
    print(f"  下載 yfinance OHLC 資料...")
    
    # yfinance 的 end 參數是 exclusive，需要加一天
    end_plus_one = end_date + timedelta(days=1)
    
    try:
        ticker = yf.Ticker(YFINANCE_TICKER)
        df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                           end=end_plus_one.strftime('%Y-%m-%d'))
        
        if df.empty:
            print("  警告: yfinance 無法取得資料")
            return pd.DataFrame()
        
        df = df.reset_index()
        df['date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        result = df[['date', 'Open', 'High', 'Low', 'Close']].copy()
        result.columns = ['date', 'open', 'high', 'low', 'close']
        
        return result
        
    except Exception as e:
        print(f"  錯誤: yfinance 下載失敗 - {e}")
        return pd.DataFrame()


def load_existing_data() -> pd.DataFrame:
    """
    載入現有的 CSV 資料
    
    回傳:
        DataFrame 包含所有歷史資料
    """
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"找不到檔案: {CSV_FILE}")
    
    df = pd.read_csv(CSV_FILE)
    
    # 轉換日期格式 (例如: "2025/11/11" -> Timestamp)
    df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
    
    return df


def save_data(df: pd.DataFrame):
    """
    儲存資料到 CSV
    
    參數:
        df: 包含所有資料的 DataFrame
    """
    # 格式化日期為原始格式
    df_save = df.copy()
    df_save['date'] = df_save['date'].apply(format_date_for_csv)
    
    # 確保欄位順序正確
    df_save = df_save[['date', 'open', 'high', 'low', 'close', 'volume']]
    
    # 儲存 (不含 index)
    df_save.to_csv(CSV_FILE, index=False)
    print(f"  已儲存至: {CSV_FILE}")


# ============================================================================
# 主程式
# ============================================================================
def main():
    print("=" * 60)
    print("TWII 資料更新腳本")
    print("=" * 60)
    
    # 1. 讀取現有資料
    print("\n[步驟 1] 讀取現有歷史資料...")
    existing_df = load_existing_data()
    print(f"  現有資料筆數: {len(existing_df)}")
    print(f"  資料範圍: {existing_df['date'].min().strftime('%Y-%m-%d')} ~ "
          f"{existing_df['date'].max().strftime('%Y-%m-%d')}")
    
    # 取得最新一筆資料的日期 (作為下載起始日)
    last_date = existing_df['date'].max()
    print(f"  最新資料日期: {last_date.strftime('%Y-%m-%d')}")
    
    # 2. 確認今天的日期
    print("\n[步驟 2] 確認今天的日期...")
    today = pd.Timestamp.now().normalize()
    print(f"  今天日期: {today.strftime('%Y-%m-%d')}")
    
    # 如果今天是週末或假日，可能沒有新資料
    if last_date >= today:
        print("\n資料已是最新，無需更新。")
        return
    
    # 3. 從證交所 API 下載成交值資料
    # 從最新資料「當天」開始下載，以確保可以覆蓋更新
    print(f"\n[步驟 3] 從證交所 API 下載成交金額資料 ({last_date.strftime('%Y-%m-%d')} ~ {today.strftime('%Y-%m-%d')})...")
    
    twse_df = fetch_twse_data_range(last_date, today)
    
    if twse_df.empty:
        print("  警告: 證交所無新資料")
        twse_df = pd.DataFrame(columns=['date', 'close', 'volume'])
    else:
        print(f"  證交所資料筆數: {len(twse_df)}")
    
    # 4. 從 yfinance 下載 OHLC 資料
    print(f"\n[步驟 4] 從 yfinance 下載 OHLC 資料...")
    yf_df = fetch_yfinance_ohlc(last_date, today)
    
    if yf_df.empty:
        print("  警告: yfinance 無新資料")
    else:
        print(f"  yfinance 資料筆數: {len(yf_df)}")
    
    # 5. 合併資料
    print("\n[步驟 5] 合併並更新資料...")
    
    if twse_df.empty and yf_df.empty:
        print("  無新資料可更新")
        return
    
    # 建立更新用的 DataFrame
    # 優先使用證交所的 close 和 volume (成交金額，單位：億元)
    # 使用 yfinance 的 open, high, low
    
    # 先以 yfinance 為基礎
    if not yf_df.empty:
        update_df = yf_df.copy()
    else:
        update_df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close'])
    
    # 合併證交所的 volume 和 close
    if not twse_df.empty:
        # 若 update_df 是空的，則直接使用 twse_df 為基礎
        if update_df.empty:
            update_df = twse_df[['date', 'close']].copy()
            update_df['open'] = update_df['close']
            update_df['high'] = update_df['close']
            update_df['low'] = update_df['close']
        
        # 合併 volume
        twse_volume = twse_df.set_index('date')['volume']
        update_df = update_df.set_index('date')
        update_df['volume'] = twse_volume
        update_df = update_df.reset_index()
        
        # 使用證交所的 close 覆蓋 yfinance 的 close
        twse_close = twse_df.set_index('date')['close']
        update_df = update_df.set_index('date')
        
        # 只覆蓋有證交所資料的日期
        for date in twse_close.index:
            if date in update_df.index:
                update_df.loc[date, 'close'] = twse_close[date]
        
        update_df = update_df.reset_index()
    
    # 過濾掉缺少 volume 的資料
    update_df = update_df.dropna(subset=['volume'])
    
    if update_df.empty:
        print("  合併後無完整資料可更新")
        return
    
    # 更新到現有資料
    # 從 existing_df 中移除會被覆蓋的日期
    dates_to_update = set(update_df['date'])
    existing_df = existing_df[~existing_df['date'].isin(dates_to_update)]
    
    # 合併
    final_df = pd.concat([existing_df, update_df], ignore_index=True)
    final_df = final_df.sort_values('date').reset_index(drop=True)
    
    # 確保欄位順序和類型
    final_df['open'] = final_df['open'].round(2)
    final_df['high'] = final_df['high'].round(2)
    final_df['low'] = final_df['low'].round(2)
    final_df['close'] = final_df['close'].round(2)
    final_df['volume'] = final_df['volume'].round(2)
    
    print(f"  更新後資料筆數: {len(final_df)}")
    print(f"  新增/更新筆數: {len(update_df)}")
    
    # 6. 儲存
    print("\n[步驟 6] 儲存更新後的資料...")
    save_data(final_df)
    
    # 顯示最後幾筆資料
    print("\n最後 5 筆資料:")
    print(final_df.tail().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("更新完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
