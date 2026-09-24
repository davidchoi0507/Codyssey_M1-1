# -*- coding: utf-8 -*-
"""
Streamlit 오디오 테크 웹 프로덕트 (Sound Matrix 100 Pro)
현대 전기 녹음 100년사 (1926~2025) 사운드 진화 연구소
- [개선 1] 사이드바 테마 스위치 전면 제거 & 우측 상단 Streamlit 메뉴(⋮ -> Settings -> Theme)로 테마 제어 100% 일원화
- [개선 2] 그래프 깜빡임/번쩍임 원인(JS setInterval DOM 뮤테이션 루프 & Plotly 애니메이션 충돌) 원천 제거
- [개선 3] 순수 정적 CSS 변수(CSS Variables & Media Queries)로 안정적 듀얼 테마 정착
- [개선 4] 우측 상단 점 3개 메뉴(⋮) 직사각형 외곽선 완전 제거 (순수 3점 아이콘)
- [개선 5] 탭 오버플로우 방지 (5대 콤팩트 세그먼트 탭)
- [개선 6] 슬라이더 조작 즉시 0ms 반응 페이드아웃 & 0.4s 안정적 페이드인
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -------------------------------------------------------------
# 1. 페이지 기본 설정
# -------------------------------------------------------------
st.set_page_config(
    page_title="SOUND MATRIX 100 | 현대 전기 녹음 100년사 (1926-2025)",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 2. 사이드바 오디오 콘솔 제어기 (테마 스위치 완전 제거)
# -------------------------------------------------------------
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px;">
    <div style="width: 34px; height: 34px; background: #1DB954; border-radius: 9px; display: flex; align-items: center; justify-content: center; font-size: 1.15rem; color: #000;">🎛️</div>
    <div>
        <div style="font-size: 1.15rem; font-weight: 800; letter-spacing: -0.5px; line-height: 1.2;">CONSOLE RACK</div>
        <div style="font-size: 0.72rem; opacity: 0.65; font-weight: 600;">AUDIO PARAMS ENGINE</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.subheader("⚡ 시대별 퀵 프리셋")
preset = st.sidebar.radio(
    "역사적 시대 바로가기",
    options=[
        "전체 100년사 (1926-2025)",
        "마이크 탄생 & SP 시대 (1926-1950)",
        "비틀즈 & 핑크 플로이드 (1960-1980)",
        "CD 전성기 (1980-2000)",
        "스트리밍 & 숏폼 완결 (2000-2025)"
    ],
    index=0
)

if preset == "마이크 탄생 & SP 시대 (1926-1950)":
    default_range = (1926, 1950)
elif preset == "비틀즈 & 핑크 플로이드 (1960-1980)":
    default_range = (1960, 1980)
elif preset == "CD 전성기 (1980-2000)":
    default_range = (1980, 2000)
elif preset == "스트리밍 & 숏폼 완결 (2000-2025)":
    default_range = (2000, 2025)
else:
    default_range = (1926, 2025)

year_range = st.sidebar.slider(
    "연도 타임라인 슬라이더",
    min_value=1926,
    max_value=2025,
    value=default_range,
    step=1
)

# -------------------------------------------------------------
# 3. 안정적인 정적 CSS 테마 시스템 (무한루프/깜빡임 100% 제거)
# -------------------------------------------------------------
DYNAMIC_CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
html, body, [class*="css"] {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* 1. 기본 라이트 테마 변수 */
:root {
    --card-bg: rgba(255, 255, 255, 0.92);
    --card-border: rgba(226, 232, 240, 0.95);
    --card-shadow: 0 4px 18px rgba(0, 0, 0, 0.05);
    --text-primary: #0F172A;
    --text-secondary: #334155;
    --text-muted: #64748B;
    --track-bg: rgba(255, 255, 255, 0.95);
    --pill-bg: #F1F5F9;
    --gnb-bg: rgba(255, 255, 255, 0.92);
    --insight-bg: rgba(255, 255, 255, 0.95);
    --tab-list-bg: #E2E8F0;
    --tab-text: #1E293B;
    --tab-active-bg: #1DB954;
    --tab-active-text: #FFFFFF;
    --hero-title-grad: linear-gradient(90deg, #0F172A 0%, #1DB954 50%, #0077B6 100%);
    --tooltip-bg: rgba(255, 255, 255, 0.96);
    --tooltip-stroke: rgba(29, 185, 84, 0.75);
    --tooltip-text: #0F172A;
    --tooltip-desc: #475569;
    --tooltip-sep: #E2E8F0;
    --hover-bg: rgba(0, 0, 0, 0.06);
    --tag-cyan-bg: rgba(0, 150, 199, 0.12);
    --tag-cyan-col: #0284C7;
    --tag-purple-bg: rgba(142, 68, 173, 0.12);
    --tag-purple-col: #8E44AD;
    --tag-green-bg: rgba(29, 185, 84, 0.12);
    --tag-green-col: #15803D;
    --tag-crimson-bg: rgba(231, 76, 60, 0.12);
    --tag-crimson-col: #DC2626;
}

/* 2. 다크 모드 미디어 쿼리 (시스템/브라우저 연동) */
@media (prefers-color-scheme: dark) {
    :root {
        --card-bg: rgba(18, 22, 31, 0.85);
        --card-border: rgba(255, 255, 255, 0.08);
        --card-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.55);
        --text-primary: #FFFFFF;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --track-bg: rgba(18, 22, 31, 0.75);
        --pill-bg: rgba(255, 255, 255, 0.05);
        --gnb-bg: rgba(18, 22, 31, 0.88);
        --insight-bg: rgba(18, 22, 31, 0.82);
        --tab-list-bg: rgba(18, 22, 31, 0.75);
        --tab-text: #E2E8F0;
        --tab-active-bg: #1DB954;
        --tab-active-text: #0B0E14;
        --hero-title-grad: linear-gradient(90deg, #FFFFFF 0%, #1DB954 50%, #00F2FE 100%);
        --tooltip-bg: rgba(15, 23, 42, 0.94);
        --tooltip-stroke: rgba(29, 185, 84, 0.6);
        --tooltip-text: #FFFFFF;
        --tooltip-desc: #94A3B8;
        --tooltip-sep: #334155;
        --hover-bg: rgba(255, 255, 255, 0.08);
        --tag-cyan-bg: rgba(0, 150, 199, 0.16);
        --tag-cyan-col: #38BDF8;
        --tag-purple-bg: rgba(142, 68, 173, 0.16);
        --tag-purple-col: #C084FC;
        --tag-green-bg: rgba(29, 185, 84, 0.16);
        --tag-green-col: #4ADE80;
        --tag-crimson-bg: rgba(231, 76, 60, 0.16);
        --tag-crimson-col: #F87171;
    }
}

/* 3. 앰비언트 글로우 오버레이 (네이티브 배경색을 절대 덮지 않고 은은하게만 투과) */
.stApp {
    background-image: radial-gradient(circle at 85% 10%, rgba(0, 150, 199, 0.06), transparent 45%),
                      radial-gradient(circle at 15% 20%, rgba(29, 185, 84, 0.06), transparent 35%);
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* 4. 상단 스트림릿 기본 툴바/메뉴 (직사각형 외곽선 완전 제거) */
header[data-testid="stHeader"] {
    background: transparent !important;
    visibility: visible !important;
}
#MainMenu {
    visibility: visible !important;
}
[data-testid="stToolbar"] {
    visibility: visible !important;
}

header[data-testid="stHeader"] button,
#MainMenu button,
[data-testid="stToolbar"] button {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
    border-radius: 8px !important;
    transition: background-color 0.2s ease !important;
}
header[data-testid="stHeader"] button:focus,
header[data-testid="stHeader"] button:active,
#MainMenu button:focus,
#MainMenu button:active {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
header[data-testid="stHeader"] button:hover,
#MainMenu button:hover,
[data-testid="stToolbar"] button:hover {
    background: var(--hover-bg) !important;
}
header[data-testid="stHeader"] svg,
#MainMenu svg,
[data-testid="stToolbar"] svg {
    border: none !important;
    outline: none !important;
    stroke: none !important;
}
header[data-testid="stHeader"] svg *,
#MainMenu svg *,
[data-testid="stToolbar"] svg * {
    stroke: none !important;
    border: none !important;
    outline: none !important;
}

/* 5. 상단 GNB 바 */
.app-gnb {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    background: var(--gnb-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: var(--card-shadow);
}

.gnb-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-logo-badge {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #1DB954 0%, #0096C7 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    box-shadow: 0 0 12px rgba(29, 185, 84, 0.4);
}
.brand-title {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--text-primary);
}
.brand-version {
    font-size: 0.7rem;
    font-weight: 700;
    background: rgba(29, 185, 84, 0.16);
    border: 1px solid rgba(29, 185, 84, 0.4);
    color: #1DB954;
    padding: 2px 7px;
    border-radius: 10px;
    margin-left: 6px;
}

.gnb-status {
    display: flex;
    align-items: center;
    gap: 8px;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--pill-bg);
    border: 1px solid var(--card-border);
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-primary);
}
.dot-green {
    width: 7px;
    height: 7px;
    background: #1DB954;
    border-radius: 50%;
    box-shadow: 0 0 8px #1DB954;
}

/* 6. 히어로 섹션 */
.hero-container {
    margin-bottom: 24px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(29, 185, 84, 0.12);
    border: 1px solid rgba(29, 185, 84, 0.35);
    padding: 5px 14px;
    border-radius: 30px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #1DB954;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
}
.hero-title {
    font-size: 2.35rem;
    font-weight: 900;
    letter-spacing: -0.8px;
    line-height: 1.2;
    margin-bottom: 8px;
    background: var(--hero-title-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-desc {
    font-size: 1.02rem;
    line-height: 1.65;
    color: var(--text-secondary);
    max-width: 1000px;
}

/* 7. 100년 타임라인 트랙 */
.timeline-track-card {
    background: var(--track-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 24px;
    box-shadow: var(--card-shadow);
}
.timeline-track-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.5px;
}
.timeline-pills-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
}
.timeline-pill-item {
    background: var(--pill-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 9px 10px;
    text-align: center;
    transition: all 0.2s ease;
}
.timeline-pill-item:hover {
    border-color: rgba(29, 185, 84, 0.5);
    transform: translateY(-2px);
}
.pill-year {
    font-size: 0.85rem;
    font-weight: 800;
    color: var(--text-primary);
}
.pill-desc {
    font-size: 0.73rem;
    color: var(--text-muted);
    margin-top: 2px;
}
.pill-active-peak {
    border-color: rgba(245, 158, 11, 0.6) !important;
    background: rgba(245, 158, 11, 0.12) !important;
}

/* 8. KPI 계측 카드 */
.metric-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 26px;
}
.glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 18px 20px;
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    overflow: hidden;
    box-shadow: var(--card-shadow);
}
.glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(29, 185, 84, 0.55);
    box-shadow: 0 14px 32px -8px rgba(29, 185, 84, 0.25);
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
}
.card-green::before { background: #1DB954; box-shadow: 0 0 12px #1DB954; }
.card-cyan::before { background: #0096C7; box-shadow: 0 0 12px #0096C7; }
.card-purple::before { background: #8E44AD; box-shadow: 0 0 12px #8E44AD; }
.card-crimson::before { background: #E74C3C; box-shadow: 0 0 12px #E74C3C; }

.card-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 700;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.card-tag {
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 800;
}
.tag-cyan { background: var(--tag-cyan-bg); color: var(--tag-cyan-col); }
.tag-purple { background: var(--tag-purple-bg); color: var(--tag-purple-col); }
.tag-green { background: var(--tag-green-bg); color: var(--tag-green-col); }
.tag-crimson { background: var(--tag-crimson-bg); color: var(--tag-crimson-col); }

.card-value {
    font-size: 1.95rem;
    font-weight: 900;
    color: var(--text-primary);
    letter-spacing: -0.8px;
    line-height: 1.15;
}
.card-unit {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-left: 5px;
}
.card-sub {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 6px;
    font-weight: 500;
}

/* 9. 세그먼트 필 탭 바 (오버플로우 방지) */
.stTabs [data-baseweb="tab-list"] {
    background: var(--tab-list-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 30px !important;
    padding: 4px !important;
    gap: 4px !important;
    margin-bottom: 18px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 24px !important;
    padding: 6px 14px !important;
    height: 38px !important;
    font-weight: 700 !important;
    font-size: 0.84rem !important;
    color: var(--tab-text) !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(128, 128, 128, 0.1) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--tab-active-bg) !important;
    color: var(--tab-active-text) !important;
    font-weight: 800 !important;
    box-shadow: 0 4px 12px rgba(29, 185, 84, 0.35) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* 10. 스튜디오 오디오 인텔리전스 카드 */
.studio-insight-box {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    background: var(--insight-bg);
    border: 1px solid var(--card-border);
    border-left: 4px solid #1DB954;
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 18px;
    margin-bottom: 12px;
    box-shadow: var(--card-shadow);
}
.insight-icon {
    font-size: 1.3rem;
    line-height: 1;
    margin-top: 2px;
}
.insight-body {
    flex: 1;
}
.insight-title {
    font-size: 0.86rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.3px;
    margin-bottom: 4px;
}
.insight-desc {
    font-size: 0.85rem;
    line-height: 1.6;
    color: var(--text-secondary);
}
.insight-amber {
    border-left-color: #F59E0B !important;
}
.insight-cyan {
    border-left-color: #0096C7 !important;
}

/* 11. 인터랙션 모션 (0ms 터치 페이드아웃 & 0.4s 안정적 페이드인) */
.stApp:has([data-testid="stSidebar"] [role="slider"]:active) [data-testid="stMain"],
.stApp:has([data-testid="stSidebar"] [data-baseweb="slider"]:active) [data-testid="stMain"],
.stApp:has([data-testid="stSidebar"] input[type="range"]:active) [data-testid="stMain"] {
    opacity: 0.35 !important;
    filter: blur(1.5px) !important;
    transition: opacity 0.15s ease-out, filter 0.15s ease-out !important;
}
[data-stale="true"], .element-container-stale {
    opacity: 0.4 !important;
    filter: blur(1.2px) !important;
    transition: opacity 0.2s ease-out, filter 0.2s ease-out !important;
}
[data-testid="stMain"] {
    transition: opacity 0.4s cubic-bezier(0.16, 1, 0.3, 1), filter 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

/* 12. Plotly 툴팁 글래스모피즘 & 부드러운 라운딩 (차트 애니메이션 완전 배제) */
.js-plotly-plot .plotly .hoverlayer {
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
}
.js-plotly-plot .plotly .hoverlayer path {
    stroke-linejoin: round !important;
    stroke-linecap: round !important;
    stroke-width: 5px !important;
    paint-order: stroke fill !important;
    fill: var(--tooltip-bg) !important;
    stroke: var(--tooltip-stroke) !important;
    filter: drop-shadow(0 10px 25px rgba(0, 0, 0, 0.12)) !important;
}

.sidebar-tech-box {
    background: var(--pill-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 24px;
    font-size: 0.8rem;
    color: var(--text-muted);
}
</style>
"""
st.markdown(DYNAMIC_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. 데이터 로드 및 전처리
# -------------------------------------------------------------
DATA_PATH = os.path.join("data", "yearly_features.csv")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("데이터 파일을 찾을 수 없습니다. 먼저 python analysis.py를 실행해 주세요.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])].copy()

# 이동평균 파라미터
ma_win = st.sidebar.selectbox(
    "롤링 스무딩 필터 (Rolling Filter)",
    options=[3, 5, 10],
    index=1,
    format_func=lambda x: f"{x}년 스무딩 필터 (MA)"
)
filtered_df['dynamic_ma'] = filtered_df['duration_sec'].rolling(ma_win, min_periods=1).mean()

# 비교 오디오 피처
compare_feat = st.sidebar.selectbox(
    "비교 오디오 파형 피처",
    options=["acousticness", "danceability", "energy", "tempo", "loudness", "valence"],
    index=0,
    format_func=lambda x: {
        "acousticness": "어쿠스틱함 (Acousticness)",
        "danceability": "댄서빌리티 (Danceability)",
        "energy": "에너지 (Energy)",
        "tempo": "템포 (Tempo BPM)",
        "loudness": "음압 (Loudness dB)",
        "valence": "밝기/긍정도 (Valence)"
    }[x]
)
filtered_df['dynamic_feat_ma'] = filtered_df[compare_feat].rolling(ma_win, min_periods=1).mean()

st.sidebar.markdown("""
<div class="sidebar-tech-box">
    <b>SYSTEM SPECIFICATIONS</b><br>
    • Track Archive: 169,457 Tracks<br>
    • Timespan: 100 Continuous Years<br>
    • Anchor Date: September 2026<br>
    • Theme: Auto Sync via Top-Right Menu<br>
    • Data Source: Spotify Web API
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 5. 글로벌 GNB & 히어로 섹션
# -------------------------------------------------------------
st.markdown(f"""
<div class="app-gnb">
    <div class="gnb-brand">
        <div class="brand-logo-badge">🎧</div>
        <div>
            <span class="brand-title">SOUND MATRIX 100</span>
            <span class="brand-version">PRO v2.6</span>
        </div>
    </div>
    <div class="gnb-status">
        <div class="status-pill"><div class="dot-green"></div> 169,457 TRACKS</div>
        <div class="status-pill">📅 {year_range[0]} — {year_range[1]}</div>
        <div class="status-pill">🎛️ AUDIO LAB</div>
    </div>
</div>

<div class="hero-container">
    <div class="hero-badge">
        <span>● LIVE TIME-SERIES AUDIO INTELLIGENCE</span>
    </div>
    <div class="hero-title">현대 전기 녹음 100년사 사운드 진화 연구소</div>
    <div class="hero-desc">
        1926년 <b>전기 마이크 탄생(3분 물리적 한계)</b>에서 출발하여, 
        <b>비틀즈와 핑크 플로이드(1970s)의 5분대 대곡 황금기</b>를 거쳐, 
        <b>2025년 숏폼/스트리밍 완결</b>로 다시 100년 전의 3분대로 완벽 회귀한 대서사시를 실시간 오디오 콘솔로 분석합니다.
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 6. 100년 시대 타임라인 인터랙티브 트랙
# -------------------------------------------------------------
st.markdown(f"""
<div class="timeline-track-card">
    <div class="timeline-track-header">
        <span>⏳ 100-YEAR HISTORICAL TIMELINE TRACK</span>
        <span>CURRENT WINDOW: <b>{year_range[0]}년 — {year_range[1]}년 ({year_range[1]-year_range[0]+1}개년)</b></span>
    </div>
    <div class="timeline-pills-row">
        <div class="timeline-pill-item">
            <div class="pill-year">🎙️ 1926년</div>
            <div class="pill-desc">마이크 전기 녹음 (3분 SP 한계)</div>
        </div>
        <div class="timeline-pill-item">
            <div class="pill-year">💿 1948년</div>
            <div class="pill-desc">33.3 RPM LP 음반 (앨범 개막)</div>
        </div>
        <div class="timeline-pill-item">
            <div class="pill-year">🎸 1963년</div>
            <div class="pill-desc">비틀즈 데뷔 (Hey Jude 싱글 혁명)</div>
        </div>
        <div class="timeline-pill-item pill-active-peak">
            <div class="pill-year">👑 1976년 ★</div>
            <div class="pill-desc">핑크 플로이드 정점 (266.4초)</div>
        </div>
        <div class="timeline-pill-item">
            <div class="pill-year">📱 2025년</div>
            <div class="pill-desc">숏폼 스트리밍 완결 (188.4초 회귀)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 7. 스튜디오 KPI 디지털 계측 카드
# -------------------------------------------------------------
start_d = filtered_df['duration_sec'].iloc[0]
end_d = filtered_df['duration_sec'].iloc[-1]
diff_d = end_d - start_d
max_row = filtered_df.loc[filtered_df['duration_sec'].idxmax()]

st.markdown(f"""
<div class="metric-container">
    <div class="glass-card card-cyan">
        <div class="card-label">
            <span>시작 시점 평균 ({year_range[0]}년)</span>
            <span class="card-tag tag-cyan">START</span>
        </div>
        <div class="card-value">{start_d:.1f}<span class="card-unit">초 ({start_d/60:.2f}분)</span></div>
        <div class="card-sub">기준 윈도우 첫 발매 연도</div>
    </div>
    <div class="glass-card card-purple">
        <div class="card-label">
            <span>역사상 최고 정점 (Peak)</span>
            <span class="card-tag tag-purple">👑 {int(max_row['year'])} PEAK</span>
        </div>
        <div class="card-value">{max_row['duration_sec']:.1f}<span class="card-unit">초 ({max_row['duration_min']:.2f}분)</span></div>
        <div class="card-sub">핑크 플로이드 & 앨범 록 황금기</div>
    </div>
    <div class="glass-card card-green">
        <div class="card-label">
            <span>종료 시점 평균 ({year_range[1]}년)</span>
            <span class="card-tag tag-green">{"▲" if diff_d >= 0 else "▼"} {abs(diff_d):.1f}s</span>
        </div>
        <div class="card-value">{end_d:.1f}<span class="card-unit">초 ({end_d/60:.2f}분)</span></div>
        <div class="card-sub">기준 대비 변동률: {(diff_d/start_d)*100:+.1f}%</div>
    </div>
    <div class="glass-card card-crimson">
        <div class="card-label">
            <span>2030년 미래 하한선 (Floor)</span>
            <span class="card-tag tag-crimson">LIMIT</span>
        </div>
        <div class="card-value">150.0<span class="card-unit">초 (2.50분)</span></div>
        <div class="card-sub">음악 서사 & 30초 정산 마지노선</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 8. 역사적 마일스톤 데이터 & 툴팁 빌더
# -------------------------------------------------------------
milestone_dict = {
    1926: "🎙️ 마이크 전기 녹음 상용화 (78 RPM SP 3분 한계)",
    1948: "💿 33.3 RPM LP 음반 발명 (앨범 시대 개막)",
    1963: "🎸 비틀즈(The Beatles) 데뷔 (라디오 팝 3분 룰 준수)",
    1968: "🔥 Hey Jude 싱글 발표 (7분 11초 싱글 혁명)",
    1973: "🌈 핑크 플로이드 The Dark Side of the Moon 발표",
    1976: "👑 핑크 플로이드 & 앨범 록 전성기 (100년 역사상 최고점 266.4초)",
    1982: "💿 디지털 CD 발매 및 MTV 방송 시작",
    2008: "🎧 Spotify 스트리밍 서비스 론칭 (30초 정산 정책)",
    2016: "📱 TikTok 숏폼 글로벌 확산 시작",
    2025: "🏁 숏폼 스트리밍 완결 (188.4초, 100년 만의 원점 회귀)"
}

def build_hover_text(sub_df):
    hover_texts = []
    for _, r in sub_df.iterrows():
        y = int(r['year'])
        sec = r['duration_sec']
        mins = int(sec // 60)
        rem_sec = int(round(sec % 60))
        min_str = f"{mins}분 {rem_sec:02d}초"
        ma = r.get('dynamic_ma', sec)
        ma_mins = int(ma // 60)
        ma_rem = int(round(ma % 60))
        event = milestone_dict.get(y, "대중음악 시계열 트렌드 관측 구간")
        
        text = (
            f"<br>&nbsp;<span style='background: rgba(29, 185, 84, 0.18); border: 1px solid rgba(29, 185, 84, 0.55); border-radius: 12px; padding: 3px 10px; font-size: 12.5px; font-weight: 700; color: #1DB954;'>🎧 {y}년 오디오 세션 로그</span><br>"
            f"<span style='color: var(--tooltip-sep, #CBD5E1);'>──────────────────────────────</span><br>"
            f"<span style='font-size: 10.5px; color: var(--tooltip-desc, #64748B);'>⏱️ 평균 곡 길이 (DURATION)</span><br>"
            f"&nbsp;&nbsp;<b><span style='font-size: 16.5px; color: var(--tooltip-text, inherit);'>{sec:.1f}초</span></b> "
            f"<span style='font-size: 12px; color: #0284C7;'>({min_str} / {sec/60.0:.2f}분)</span><br><br>"
            f"<span style='font-size: 10.5px; color: var(--tooltip-desc, #64748B);'>📈 {ma_win}년 스무딩 필터 (TREND MA)</span><br>"
            f"&nbsp;&nbsp;<b><span style='font-size: 13.5px; color: var(--tooltip-text, inherit);'>{ma:.1f}초</span></b> "
            f"<span style='font-size: 11px; color: var(--tooltip-desc, #64748B);'>({ma_mins}분 {ma_rem:02d}초)</span><br><br>"
            f"<span style='font-size: 10.5px; color: var(--tooltip-desc, #64748B);'>🎛️ 사운드 텍스처 (SOUND MATRIX)</span><br>"
            f"&nbsp;&nbsp;댄서빌리티 <b style='color: #1DB954;'>{r['danceability']:.3f}</b> &nbsp;|&nbsp; "
            f"어쿠스틱 <b style='color: #D97706;'>{r['acousticness']:.3f}</b><br>"
            f"<span style='color: var(--tooltip-sep, #CBD5E1);'>──────────────────────────────</span><br>"
            f"&nbsp;<span style='background: rgba(245, 158, 11, 0.18); border: 1px solid rgba(245, 158, 11, 0.55); border-radius: 6px; padding: 2px 7px; font-size: 10px; font-weight: 700; color: #D97706;'>💡 HISTORICAL HIGHLIGHT</span><br>"
            f"&nbsp;<b><span style='font-size: 12px; color: var(--tooltip-text, inherit);'>{event}</span></b><br>"
        )
        hover_texts.append(text)
    return hover_texts

filtered_df['hover_info'] = build_hover_text(filtered_df)

# Plotly 공통 레이아웃 프리셋 (Streamlit 네이티브 테마와 100% 호환, 애니메이션 충돌 제거)
ADAPTIVE_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Pretendard, -apple-system, sans-serif"),
    xaxis=dict(
        showline=True,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1.2,
        spikecolor="rgba(29, 185, 84, 0.6)",
        spikedash="dot"
    ),
    yaxis=dict(
        showline=True
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(0,0,0,0)"
    ),
    hoverdistance=20,
    margin=dict(l=40, r=40, t=60, b=40)
)

# -------------------------------------------------------------
# 9. 간결하고 콤팩트한 5대 세그먼트 필 탭 메뉴
# -------------------------------------------------------------
t1, t2, t3, t4, t5 = st.tabs([
    "📈 곡 길이 순환 파형",
    "🎚️ 사운드 EQ 비교",
    "🔍 10년 주기 분해",
    "🔮 2030 미래 예측",
    "📋 데이터 아카이브"
])

# -------------------------------------------------------------
# Tab 1: 곡 길이 순환 궤적
# -------------------------------------------------------------
with t1:
    st.markdown("<div style='font-size: 1.1rem; font-weight: 800; margin-bottom: 10px;'>대중음악 곡 길이 100년 거대 순환 파형 (1926 ~ 2025)</div>", unsafe_allow_html=True)

    fig1 = go.Figure()

    # 1. 연도별 실측 곡 길이 (로열 블루 #0096C7)
    fig1.add_trace(go.Scatter(
        x=filtered_df['year'],
        y=filtered_df['duration_sec'],
        mode='lines+markers',
        name='연도별 평균 곡 길이 (Raw Signal)',
        line=dict(color='#0096C7', width=1.6),
        marker=dict(size=5.5, color='#0096C7', line=dict(width=1, color='#FFFFFF')),
        hoverinfo='text',
        hovertext=filtered_df['hover_info'],
        opacity=0.75
    ))

    # 2. 이동평균선 (Spotify Neon Green #1DB954)
    fig1.add_trace(go.Scatter(
        x=filtered_df['year'],
        y=filtered_df['dynamic_ma'],
        mode='lines',
        name=f'{ma_win}년 롤링 스무딩 필터 (Trend MA)',
        line=dict(color='#1DB954', width=3.8),
        hoverinfo='skip'
    ))

    # 3. 주요 역사적 마일스톤
    milestones = [
        (1926, 170.4, "🎙️ 마이크 전기 녹음 (1926)", "#0096C7"),
        (1948, 220.0, "💿 LP 발명 (1948)", "#64748B"),
        (1963, 195.7, "🎸 비틀즈 데뷔 (1963)", "#8E44AD"),
        (1968, 217.3, "🔥 Hey Jude (1968)", "#8E44AD"),
        (1976, 266.4, "👑 핑크 플로이드 정점 (1976)", "#D97706"),
        (1982, 250.2, "💿 CD & MTV (1982)", "#64748B"),
        (2008, 240.5, "🎧 Spotify 론칭 (2008)", "#1DB954"),
        (2025, 188.4, "🏁 숏폼 완결 (188초)", "#E74C3C")
    ]

    for yr, yval, lbl, col in milestones:
        if year_range[0] <= yr <= year_range[1]:
            fig1.add_vline(x=yr, line_width=1.2, line_dash="dash", line_color=col, opacity=0.5)
            fig1.add_annotation(
                x=yr, y=yval,
                text=lbl,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.2,
                arrowcolor=col,
                ax=0, ay=-38,
                bgcolor="rgba(128, 128, 128, 0.15)",
                bordercolor=col,
                borderwidth=1.5,
                font=dict(size=9.5, family="Pretendard")
            )

    fig1.update_layout(
        **ADAPTIVE_PLOT_LAYOUT,
        title=dict(text="<b>곡 길이 100년의 거대 순환: 마이크 탄생(3분) ➔ 핑크 플로이드(5분대) ➔ 숏폼 회귀(3분)</b>", font=dict(size=14)),
        xaxis_title="발매 연도 (Year)",
        yaxis_title="평균 곡 길이 (초 / Seconds)",
        hovermode="closest",
        height=520
    )

    st.plotly_chart(fig1, use_container_width=True, theme="streamlit")

    st.markdown("""
    <div class="studio-insight-box">
        <div class="insight-icon">💡</div>
        <div class="insight-body">
            <div class="insight-title">STUDIO WAVEFORM INTERACTION TIP</div>
            <div class="insight-desc">
                파형 곡선이나 포인트 위에 마우스를 올리면 <b>부드러운 라운드 글래스모피즘 툴팁</b>이 나타납니다. 
                마우스 드래그로 특정 시대(예: 비틀즈 & 핑크 플로이드 60~70s)를 확대(Zoom-in)할 수 있으며 더블 클릭 시 즉시 전체 뷰로 복귀합니다.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Tab 2: 어쿠스틱 쇠퇴 vs 댄서빌리티 부흥
# -------------------------------------------------------------
with t2:
    st.markdown(f"<div style='font-size: 1.1rem; font-weight: 800; margin-bottom: 10px;'>사운드 패러다임 대역전: 곡 길이 vs {compare_feat}</div>", unsafe_allow_html=True)

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    fig2.add_trace(
        go.Scatter(
            x=filtered_df['year'],
            y=filtered_df['dynamic_ma'],
            name=f'곡 길이 ({ma_win}년 필터, 초)',
            line=dict(color='#D97706', width=3.2),
            hovertemplate="&nbsp;<b>⏱️ 곡 길이 필터</b>: <b>%{y:.1f}초</b><extra></extra>"
        ),
        secondary_y=False
    )

    fig2.add_trace(
        go.Scatter(
            x=filtered_df['year'],
            y=filtered_df['dynamic_feat_ma'],
            name=f'{compare_feat} ({ma_win}년 필터)',
            line=dict(color='#1DB954', width=3.2, dash='dash'),
            hovertemplate=f"&nbsp;<b>🎛️ {compare_feat}</b>: <b>%{{y:.3f}}</b><extra></extra>"
        ),
        secondary_y=True
    )

    fig2.update_layout(
        **ADAPTIVE_PLOT_LAYOUT,
        title=dict(text=f"<b>곡 길이와 {compare_feat}의 100년 동적 상관관계 (1926-2025)</b>", font=dict(size=14)),
        xaxis_title="발매 연도 (Year)",
        hovermode="x unified",
        height=500
    )
    fig2.update_yaxes(title_text="곡 길이 (초)", secondary_y=False)
    fig2.update_yaxes(title_text=f"{compare_feat} 지수 (0.0 ~ 1.0)", secondary_y=True)

    st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

    if compare_feat == 'acousticness':
        st.markdown("""
        <div class="studio-insight-box insight-cyan">
            <div class="insight-icon">🎻</div>
            <div class="insight-body">
                <div class="insight-title">ACOUSTIC RETREAT ANALYSIS (어쿠스틱의 수직 낙하)</div>
                <div class="insight-desc">
                    1920년대 0.90에 육박하던 어쿠스틱 지수는 전자 기타(1950s), 신시사이저(1980s), 컴퓨터 DAW 디지털 프로덕션(2000s~)을 거치며 
                    <b>2025년 0.18 수준으로 수직 낙하</b>했습니다.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif compare_feat == 'danceability':
        st.markdown("""
        <div class="studio-insight-box insight-cyan">
            <div class="insight-icon">🕺</div>
            <div class="insight-body">
                <div class="insight-title">RHYTHM & BEAT ASCENDANCE (비트/리듬의 부흥)</div>
                <div class="insight-desc">
                    곡 길이가 줄어드는 최근 30년간 댄서빌리티는 지속 상승했습니다. 이는 현대 대중음악이 '서사 감상'에서 
                    <b>'신체적 비트 반응' 중심으로 패러다임이 전면 재편</b>되었음을 실증합니다.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Tab 3: 시계열 분해
# -------------------------------------------------------------
with t3:
    st.markdown("<div style='font-size: 1.1rem; font-weight: 800; margin-bottom: 10px;'>100년 오디오 신호 분해 (10년 데케이드 주기 가법 모델)</div>", unsafe_allow_html=True)
    if len(filtered_df) >= 20:
        ts = filtered_df.set_index(pd.date_range(start=str(year_range[0]), periods=len(filtered_df), freq='YS'))['duration_sec']
        decomp = seasonal_decompose(ts, model='additive', period=min(10, len(filtered_df)//2))

        fig3 = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=["관측 파형 (Observed Signal)", "장기 추세선 (Trend)", "10년 주기 파동 (Decade Cycle)", "불규칙 노이즈 잔차 (Residual)"]
        )

        fig3.add_trace(go.Scatter(x=ts.index.year, y=decomp.observed, name='Observed', line=dict(color='#0096C7', width=1.6)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=ts.index.year, y=decomp.trend, name='Trend', line=dict(color='#1DB954', width=2.8)), row=2, col=1)
        fig3.add_trace(go.Scatter(x=ts.index.year, y=decomp.seasonal, name='Cycle', line=dict(color='#8E44AD', width=2.2)), row=3, col=1)
        fig3.add_trace(go.Scatter(x=ts.index.year, y=decomp.resid, name='Residual', mode='markers', marker=dict(color='#E74C3C', size=5, opacity=0.8)), row=4, col=1)

        fig3.update_layout(
            **ADAPTIVE_PLOT_LAYOUT,
            height=700,
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True, theme="streamlit")
    else:
        st.markdown("""
        <div class="studio-insight-box insight-amber">
            <div class="insight-icon">⚠️</div>
            <div class="insight-body">
                <div class="insight-title">INSUFFICIENT SAMPLE WINDOW</div>
                <div class="insight-desc">시계열 분해를 위해 사이드바 슬라이더에서 최소 20년 이상의 기간을 선택해 주세요.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Tab 4: 미래 5년 예측
# -------------------------------------------------------------
with t4:
    st.markdown("<div style='font-size: 1.1rem; font-weight: 800; margin-bottom: 10px;'>미래 5년(2026~2030) 베이스라인 시계열 예측 (현재 2026년 9월 기준)</div>", unsafe_allow_html=True)

    ts_all = df.set_index(pd.date_range(start='1926', periods=len(df), freq='YS'))['duration_sec']
    hw = ExponentialSmoothing(ts_all, trend='add', seasonal=None).fit()
    fc_5 = hw.forecast(5)
    fc_years = pd.date_range(start='2026', periods=5, freq='YS').year
    std_all = np.nanstd(hw.resid)

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=ts_all.index.year[-30:],
        y=ts_all.iloc[-30:],
        mode='lines+markers',
        name='실측 및 보정치 (1996-2025)',
        line=dict(color='#0096C7', width=2),
        marker=dict(size=4.5, color='#0096C7')
    ))

    fig4.add_trace(go.Scatter(
        x=fc_years,
        y=fc_5,
        mode='lines+markers',
        name='미래 5년 예측선 (2026-2030)',
        line=dict(color='#E74C3C', width=3.2, dash='dash'),
        marker=dict(size=7, color='#E74C3C', line=dict(width=1.5, color='#FFFFFF'))
    ))

    upper = fc_5 + 1.96 * std_all
    lower = fc_5 - 1.96 * std_all
    fig4.add_trace(go.Scatter(
        x=list(fc_years) + list(fc_years[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.16)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='95% 신뢰구간'
    ))

    fig4.add_hline(y=150.0, line_dash="dot", line_color="#D97706", line_width=1.8, opacity=0.8)
    fig4.add_annotation(
        x=2028, y=150.0,
        text="⚠️ 도메인 물리적 하한선 (Floor Limit: 150초 / 2분 30초)",
        showarrow=False,
        bgcolor="rgba(128, 128, 128, 0.15)",
        bordercolor="#D97706",
        borderwidth=1.2,
        font=dict(color="#D97706", size=10, family="Pretendard")
    )

    val_2026 = fc_5.iloc[0]
    fig4.add_vline(x=2026, line_width=1.5, line_dash="dot", line_color="#1DB954")
    fig4.add_annotation(
        x=2026, y=val_2026,
        text=f"<b>현재 시점(2026년): {val_2026:.1f}초</b>",
        showarrow=True, arrowhead=2, arrowcolor="#1DB954",
        ax=40, ay=-30,
        bgcolor="rgba(128, 128, 128, 0.15)",
        bordercolor="#1DB954",
        borderwidth=1.2,
        font=dict(color="#1DB954", size=10, family="Pretendard")
    )

    val_2030 = fc_5.iloc[-1]
    fig4.add_annotation(
        x=2030, y=val_2030,
        text=f"<b>2030년 예측: {val_2030:.1f}초 ({val_2030/60:.2f}분)</b>",
        showarrow=True, arrowhead=2, arrowcolor="#E74C3C",
        ax=-40, ay=-35,
        bgcolor="rgba(128, 128, 128, 0.15)",
        bordercolor="#E74C3C",
        borderwidth=1.2,
        font=dict(color="#E74C3C", size=10, family="Pretendard")
    )

    fig4.update_layout(
        **ADAPTIVE_PLOT_LAYOUT,
        title=dict(text="<b>100년(1926-2025) 데이터 기반 미래 5년(2026-2030) 대중음악 곡 길이 예측 레이더</b>", font=dict(size=14)),
        xaxis_title="연도 (Year)",
        yaxis_title="평균 곡 길이 (초 / Seconds)",
        hovermode="x unified",
        height=520
    )

    st.plotly_chart(fig4, use_container_width=True, theme="streamlit")

    st.markdown("""
    <div class="studio-insight-box insight-amber">
        <div class="insight-icon">⚠️</div>
        <div class="insight-body">
            <div class="insight-title">DOMAIN FLOOR LIMIT FORECAST (도메인 물리적 하한선 전망)</div>
            <div class="insight-desc">
                선형 지수평활 모델은 2030년 약 170.8초(2분 51초) 수렴을 가리킵니다. 
                그러나 스트리밍 30초 정산 기준과 보컬/인트로/후렴이라는 최소한의 음악적 서사 구조를 충족하기 위해 
                <b>2분 30초(150초)가 물리적 마지노선</b>이 될 것으로 분석됩니다.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Tab 5: 인터랙티브 데이터 테이블 (st.dataframe)
# -------------------------------------------------------------
with t5:
    st.markdown("<div style='font-size: 1.1rem; font-weight: 800; margin-bottom: 10px;'>100년 음원 시계열 데이터 콘솔 아카이브</div>", unsafe_allow_html=True)
    st.markdown("컬럼 헤더를 클릭하여 **오름차순/내림차순 정렬**하거나, 우측 상단 메뉴를 통해 **CSV로 즉시 다운로드**할 수 있습니다.")

    display_df = filtered_df[['year', 'duration_sec', 'duration_min', 'danceability', 'energy', 'acousticness', 'tempo', 'track_count']].copy()
    display_df.rename(columns={
        'year': '연도',
        'duration_sec': '곡 길이 (초)',
        'duration_min': '곡 길이 (분)',
        'danceability': '댄서빌리티',
        'energy': '에너지',
        'acousticness': '어쿠스틱함',
        'tempo': '템포(BPM)',
        'track_count': '트랙 수'
    }, inplace=True)

    st.dataframe(
        display_df.style.format({
            '곡 길이 (초)': '{:.1f}',
            '곡 길이 (분)': '{:.2f}',
            '댄서빌리티': '{:.3f}',
            '에너지': '{:.3f}',
            '어쿠스틱함': '{:.3f}',
            '템포(BPM)': '{:.1f}',
            '트랙 수': '{:,}'
        }),
        use_container_width=True,
        height=450
    )
