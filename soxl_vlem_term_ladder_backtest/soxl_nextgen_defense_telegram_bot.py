"""
================================================================================
 SOXL NextGen Multi-Leverage Defense Engine : 텔레그램 무인 자동 알림 봇
 (PC 필요 없음 / 무료 클라우드 서버 & 스마트폰 알림 전용)
================================================================================
설명:
이 스크립트는 최신 SOXL/SOXX 일봉 데이터를 계산하여
스마트폰 텔레그램 메신저로 오늘 개장 시 걸어야 할 매수/익절/손절 가이드를 자동 전송합니다.
================================================================================
"""

import os
import sys
import math
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# 텔레그램 봇 토큰 및 채널/사용자 ID (환경변수 또는 아래 직접 입력)
# ==============================================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

MY_REMAINING_CASH = float(os.environ.get("MY_REMAINING_CASH", "10000.0"))
MY_CURRENT_SOXL_SHARES = int(os.environ.get("MY_CURRENT_SOXL_SHARES", "0"))
MY_CURRENT_SOXX_SHARES = int(os.environ.get("MY_CURRENT_SOXX_SHARES", "0"))
MY_AVG_COST_SOXL = float(os.environ.get("MY_AVG_COST_SOXL", "0.0"))
MY_AVG_COST_SOXX = float(os.environ.get("MY_AVG_COST_SOXX", "0.0"))

def send_telegram_message(message: str, token: str, chat_id: str):
    if not token or not chat_id:
        print("[알림] 텔레그램 토큰(TELEGRAM_BOT_TOKEN) 또는 채팅 ID(TELEGRAM_CHAT_ID)가 입력되지 않았습니다.")
        print("-> 텔레그램 BotFather에서 만든 토큰과 Chat ID를 입력하시면 스마트폰으로 전송됩니다!")
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=payload, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as response:
            print("[성공] 스마트폰 텔레그램으로 오늘 매매 신호 자동 전송 완료!")
            return True
    except Exception as e:
        print(f"[오류] 텔레그램 전송 실패: {e}")
        return False

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

def generate_message():
    df_raw = fetch_market_data()
    df, edi, entropy_z, tda_collapse = compute_defense_signals(df_raw)
    
    last_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
    prior_close_soxl = df["soxl_close"].iloc[-1]
    prior_close_soxx = df["soxx_close"].iloc[-1]
    
    is_phase_bull = (edi[-1] > 0) and (entropy_z[-1] <= 1.2)
    is_crash_confirmed = (tda_collapse[-1] <= -0.015 or entropy_z[-1] > 1.8) and (tda_collapse[-2] <= -0.015 or entropy_z[-2] > 1.8)
    
    tranche_base = MY_REMAINING_CASH / 20.0
    
    msg = []
    msg.append(f"🛡️ *[SOXL 방어 엔진]* 오늘({last_date}) 매매 신호\n")
    msg.append(f"• SOXL 전일 종가: `${prior_close_soxl:.2f}`")
    msg.append(f"• SOXX 전일 종가: `${prior_close_soxx:.2f}`")
    
    if is_phase_bull:
        msg.append("• 오늘의 상태: 🟢 *상승 모멘텀 (SOXL 3X 매수)*\n")
    else:
        msg.append("• 오늘의 상태: 🛡️ *조정/약세 방어 (SOXX 1X 매수)*\n")
        
    msg.append("📋 *[오늘 장 시작 주문 가이드]*")
    
    if (MY_CURRENT_SOXL_SHARES > 0 or MY_CURRENT_SOXX_SHARES > 0) and is_crash_confirmed:
        cut_soxl = int(math.ceil(MY_CURRENT_SOXL_SHARES * 0.50))
        cut_soxx = int(math.ceil(MY_CURRENT_SOXX_SHARES * 0.50))
        msg.append(f"1️⃣ ⚠️ *비상 방어 손절 경보!* SOXL {cut_soxl}주 + SOXX {cut_soxx}주 (50%) 시가 손절!")
    else:
        msg.append("1️⃣ *비상 손절*: 해당 없음 (정상 유지)")
        
    if MY_CURRENT_SOXL_SHARES > 0 and MY_AVG_COST_SOXL > 0:
        msg.append(f"2️⃣ *SOXL 익절 지정가*: `${MY_AVG_COST_SOXL * 1.10:.2f}` (+10%)")
    elif MY_CURRENT_SOXX_SHARES > 0 and MY_AVG_COST_SOXX > 0:
        msg.append(f"2️⃣ *SOXX 익절 지정가*: `${MY_AVG_COST_SOXX * 1.10:.2f}` (+10%)")
    else:
        msg.append("2️⃣ *익절 지정가*: 신규 진입 대기")
        
    asset_to_buy = "SOXL (3배수)" if is_phase_bull else "SOXX (1배수)"
    est_shares = int(tranche_base / prior_close_soxl) if is_phase_bull else int(tranche_base / prior_close_soxx)
    msg.append(f"3️⃣ 🎯 *오늘 시가 매수*: *[{asset_to_buy}] 약 ${tranche_base:.2f} (약 {est_shares}주)*\n")
    
    full_text = "\n".join(msg)
    return full_text

if __name__ == "__main__":
    text = generate_message()
    send_telegram_message(text, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
