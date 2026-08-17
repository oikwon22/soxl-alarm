"""
================================================================================
 SOXL NextGen Defense Engine : 비밀번호 보안 웹 애플리케이션 (Streamlit)
================================================================================
설명:
이 웹 앱은 올바른 비밀번호(기본값: 7777)를 입력해야만
오늘의 SOXL/SOXX 실시간 매매 지침 카드를 띄워주는 보안 웹 앱입니다.
================================================================================
"""

import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

SECRET_PASSWORD = "7777"

st.set_page_config(
    page_title="SOXL Defense Engine Dashboard",
    page_icon="🛡️",
    layout="centered"
)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# 1. 비밀번호 인증 화면 (접속 시 바로 표출)
if not st.session_state["authenticated"]:
    st.title("🔒 보안 접근 시스템")
    st.markdown("본 시스템은 허가된 사용자만 접근할 수 있습니다.")
    
    password_input = st.text_input("비밀번호 (PIN)", type="password")
    if st.button("확인 (Unlock)"):
        if password_input == SECRET_PASSWORD:
            st.session_state["authenticated"] = True
            st.success("🔓 인증되었습니다!")
            st.rerun()
        else:
            st.error("⛔ 접근 권한이 없습니다. 올바른 비밀번호를 입력하세요.")
    st.stop()

# 2. 인증 완료 후 메인 대시보드 표출
st.title("🛡️ SOXL NextGen Defense Engine")
st.caption("100% Strict Causal Real-Time Trading Signal Dashboard")

@st.cache_data(ttl=3600)
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

with st.spinner("최신 SOXL/SOXX 주가 수집 및 위상 계산 중..."):
    df_raw = fetch_market_data()
    df, edi, entropy_z, tda_collapse = compute_defense_signals(df_raw)

last_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
prior_close_soxl = df["soxl_close"].iloc[-1]
prior_close_soxx = df["soxx_close"].iloc[-1]

is_phase_bull = (edi[-1] > 0) and (entropy_z[-1] <= 1.2)
is_crash_confirmed = (tda_collapse[-1] <= -0.015 or entropy_z[-1] > 1.8) and (tda_collapse[-2] <= -0.015 or entropy_z[-2] > 1.8)

# 사이드바 설정
st.sidebar.header("⚙️ 사용자 계좌 세팅")
my_cash = st.sidebar.number_input("현재 잔여 현금 ($)", value=10000.0, step=500.0)
my_soxl_shares = st.sidebar.number_input("보유 SOXL 주식 수 (주)", value=0, step=1)
my_soxx_shares = st.sidebar.number_input("보유 SOXX 주식 수 (주)", value=0, step=1)

if st.sidebar.button("🔒 잠금 (Lock)"):
    st.session_state["authenticated"] = False
    st.rerun()

tranche_base = my_cash / 20.0

st.subheader(f"📊 오늘({last_date} 종가 기준) 실시간 방어 매매 지침")

col1, col2 = st.columns(2)
with col1:
    st.metric("전일 SOXL 종가", f"${prior_close_soxl:.2f}")
with col2:
    st.metric("전일 SOXX 종가", f"${prior_close_soxx:.2f}")

if is_phase_bull:
    st.success("🟢 오늘의 시장 상태: 상승 모멘텀 진입 모드 (SOXL 3X 매수 국면)")
else:
    st.warning("🛡️ 오늘의 시장 상태: 조정/약세 방어 모드 (SOXX 1X 매수 국면)")

st.divider()
st.subheader("📋 오늘 개장 시가 예약 주문 3단계 가이드")

if (my_soxl_shares > 0 or my_soxx_shares > 0) and is_crash_confirmed:
    cut_soxl = int(math.ceil(my_soxl_shares * 0.50))
    cut_soxx = int(math.ceil(my_soxx_shares * 0.50))
    st.error(f"1️⃣ ⚠️ 비상 방어 손절 경보! SOXL {cut_soxl}주 + SOXX {cut_soxx}주 (50%) 시가 손절!")
else:
    st.info("1️⃣ 비상 방어 손절: 해당 없음 (정상 상태 유지)")

st.info("2️⃣ 익절 지정가: 신규 진입 단계 (목표가 +10% 지정가 대기)")

asset_to_buy = "SOXL (3배수 레버리지 ETF)" if is_phase_bull else "SOXX (1배수 본주)"
est_shares = int(tranche_base / prior_close_soxl) if is_phase_bull else int(tranche_base / prior_close_soxx)

st.success(f"""
3️⃣ 🎯 오늘 시가 매수 주문:
* **매수 대상**: **{asset_to_buy}**
* **매수 자금**: **약 ${tranche_base:.2f} (약 {est_shares}주)**
* **매수 근거**: {("상승 모멘텀 진입에 따른 SOXL 3X 시가 매수" if is_phase_bull else "조정/약세 진입에 따른 SOXX 1X 본주 스위칭 매수")}
""")
