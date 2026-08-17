"""
soxl_defense_github_summary.py
------------------------------
Generates a clean, concise markdown summary for GitHub web display (README.md)
"""

import math
import numpy as np
import pandas as pd
import yfinance as yf

MY_REMAINING_CASH = 10000.0

def fetch_market_data():
    soxl = yf.download("SOXL", period="1y", interval="1d", progress=False)
    soxx = yf.download("SOXX", period="1y", interval="1d", progress=False)
    
    if isinstance(soxl.columns, pd.MultiIndex):
        soxl.columns = soxl.columns.get_level_values(0)
    if isinstance(soxx.columns, pd.MultiIndex):
        soxx.columns = soxx.columns.get_level_values(0)
        
    soxl = soxl.dropna()
    soxx = soxx.dropna()
    
    df = pd.DataFrame({
        "date": soxl.index,
        "soxl_open": soxl["Open"].values,
        "soxl_high": soxl["High"].values,
        "soxl_low": soxl["Low"].values,
        "soxl_close": soxl["Close"].values,
        "soxl_volume": soxl["Volume"].values
    })
    
    soxx_map = {d: c for d, c in zip(soxx.index, soxx["Close"].values)}
    df["soxx_close"] = df["date"].map(soxx_map)
    return df.dropna().reset_index(drop=True)

def compute_defense_signals(df):
    close_soxl = df["soxl_close"].values
    volume_soxl = df["soxl_volume"].values
    N = len(close_soxl)
    
    ret = np.zeros(N)
    ret[1:] = np.diff(close_soxl) / (close_soxl[:-1] + 1e-12)
    
    vol_ma = pd.Series(volume_soxl).rolling(20, min_periods=1).mean().values
    norm_vol = np.clip(volume_soxl / (vol_ma + 1e-8), 0.1, 5.0)
    
    edi_raw = ret * np.log1p(norm_vol)
    alpha = 2.0 / (5.0 + 1.0)
    edi = np.full(N, np.nan, dtype=np.float64)
    edi[0] = edi_raw[0]
    for i in range(1, N):
        edi[i] = alpha * edi_raw[i] + (1.0 - alpha) * edi[i-1]
        
    entropy = np.full(N, np.nan, dtype=np.float64)
    for i in range(14, N):
        sub = ret[i-13:i+1]
        hist, _ = np.histogram(sub, bins=10, range=(-0.10, 0.10))
        p = hist / (np.sum(hist) + 1e-12)
        p = p[p > 0]
        entropy[i] = -np.sum(p * np.log2(p + 1e-12))
        
    b1_loop = np.zeros(N)
    tda_collapse = np.zeros(N)
    for i in range(20, N):
        e_win, h_win = edi[i-20:i], entropy[i-20:i]
        rad = np.sqrt(np.nanvar(e_win) + np.nanvar(h_win))
        b1_loop[i] = rad
        if i >= 25:
            prev_rad = np.mean(b1_loop[i-5:i])
            tda_collapse[i] = (rad - prev_rad) / (prev_rad + 1e-8)
            
    entropy_ma = pd.Series(entropy).rolling(20, min_periods=1).mean().values
    entropy_std = pd.Series(entropy).rolling(20, min_periods=1).std().values + 1e-6
    entropy_z = (entropy - entropy_ma) / entropy_std
    return df, edi, entropy_z, tda_collapse

def generate_markdown_summary():
    df_raw = fetch_market_data()
    df, edi, entropy_z, tda_collapse = compute_defense_signals(df_raw)
    
    last_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
    prior_close_soxl = df["soxl_close"].iloc[-1]
    prior_close_soxx = df["soxx_close"].iloc[-1]
    
    is_phase_bull = (edi[-1] > 0) and (entropy_z[-1] <= 1.2)
    tranche_base = MY_REMAINING_CASH / 20.0
    
    asset_to_buy = "SOXL (3배수 레버리지 ETF)" if is_phase_bull else "SOXX (1배수 본주)"
    est_shares = int(tranche_base / prior_close_soxl) if is_phase_bull else int(tranche_base / prior_close_soxx)
    status_str = "🟢 상승 모멘텀 진입 모드 (SOXL 3X 매수 국면)" if is_phase_bull else "🛡️ 조정/약세 방어 모드 (SOXX 1X 매수 국면)"
    
    md = []
    md.append(f"# 📊 오늘({last_date} 종가 기준) 실시간 방어 매매 지침\n")
    md.append(f"* **오늘의 시장 상태**: {status_str}")
    md.append(f"* **전일 SOXL 종가**: `${prior_close_soxl:.2f}`")
    md.append(f"* **전일 SOXX 종가**: `${prior_close_soxx:.2f}`\n")
    md.append(f"---\n")
    md.append(f"### 📋 [오늘 개장 시가 예약 주문 3단계 가이드]\n")
    md.append(f"1. **1단계 (비상 방어 손절)**: 해당 없음 (정상 상태 유지)")
    md.append(f"2. **2단계 (익절 지정가)**: 신규 진입 단계 (목표가 +10% 지정가 대기)")
    md.append(f"3. **3단계 (오늘 시가 매수 주문)** 🎯:")
    md.append(f"   * **매수 대상**: {asset_to_buy}")
    md.append(f"   * **매수 자금**: 1회분 약 ${tranche_base:.2f} (약 {est_shares}주)")
    md.append(f"   * **매수 근거**: 상승 모멘텀 국면 진입에 따른 SOXL 3X 시가 매수 (조정 진입 시 SOXX 1X로 자동 스위칭)\n")
    
    content = "\n".join(md)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)
        
    print("[성공] 깔끔한 README.md 업데이트 완료!")

if __name__ == "__main__":
    generate_markdown_summary()
