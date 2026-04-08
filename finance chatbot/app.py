"""
AI Finance Control Center - PRODUCTION VERSION
Three Views: Dashboard | AI Revenue Prediction | AI Copilot
Design: thebricks.com inspired - Clean, Professional, Effective
"""

import sys
from pathlib import Path

# Ensure project root is importable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

from database import DatabaseHandler
from data_processor import DataProcessor
from models.revenue_forecaster import RevenueForecaster
from models.customer_scorer import CustomerScorer
from models.churn_predictor import ChurnPredictor
from models.ai_copilot import AICopilot
from utils import Formatter, MetricsCalculator
from config import Config

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Finance Control Center",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ============================================================================
# PROFESSIONAL CSS - thebricks.com Inspired
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #e8eff8;
        --bg-layer: #f0f5fb;
        --surface: #ffffff;
        --surface-soft: #f4f8fd;
        --text-strong: #0d1829;
        --text-muted: #5a6a82;
        --line-soft: rgba(15, 24, 42, 0.09);
        --line-mid: rgba(15, 24, 42, 0.15);
        --brand-900: #081526;
        --brand-800: #0f2240;
        --brand-700: #1a3a6b;
        --brand-600: #1e4d8c;
        --brand-500: #2563eb;
        --accent: #6366f1;
        --accent-soft: rgba(99, 102, 241, 0.12);
        --mint: #5eead4;
        --mint-soft: rgba(94, 234, 212, 0.15);
        --shadow-xs: 0 2px 8px rgba(12, 24, 48, 0.06);
        --shadow-soft: 0 8px 28px rgba(12, 24, 48, 0.09);
        --shadow-strong: 0 20px 48px rgba(12, 24, 48, 0.14);
        --shadow-glow: 0 0 0 3px rgba(99, 102, 241, 0.12);
        --radius-sm: 10px;
        --radius-md: 14px;
        --radius-lg: 20px;
        --radius-xl: 28px;
    }

    * { font-family: 'Outfit', 'Inter', sans-serif; }

    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* === BACKGROUND === */
    .stApp {
        background:
            radial-gradient(ellipse 80% 60% at 10% -10%, rgba(94, 234, 212, 0.22) 0%, transparent 55%),
            radial-gradient(ellipse 60% 50% at 92% 8%, rgba(99, 102, 241, 0.16) 0%, transparent 50%),
            radial-gradient(ellipse 100% 80% at 50% 100%, rgba(37, 99, 235, 0.08) 0%, transparent 60%),
            linear-gradient(150deg, #dce8f5 0%, #eaf1f9 40%, #f2f6fb 100%);
        min-height: 100vh;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }

    /* === SIDEBAR === */
    section[data-testid="stSidebar"] {
        background:
            radial-gradient(ellipse 120% 60% at 80% 0%, rgba(99, 102, 241, 0.18) 0%, transparent 50%),
            radial-gradient(ellipse 80% 40% at 20% 100%, rgba(94, 234, 212, 0.12) 0%, transparent 50%),
            linear-gradient(175deg, #060f1f 0%, #0d2040 50%, #091628 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
        box-shadow: 4px 0 32px rgba(0,0,0,0.22);
    }

    section[data-testid="stSidebar"] * { color: #dde8f7 !important; }

    section[data-testid="stSidebar"] > div {
        padding-top: 22px;
        padding-left: 16px;
        padding-right: 16px;
    }

    .sidebar-brand { padding: 6px 2px 18px 2px; }

    .sidebar-kicker {
        color: rgba(99, 102, 241, 0.9);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 9.5px;
        font-weight: 800;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .sidebar-kicker::before {
        content: '';
        display: inline-block;
        width: 18px;
        height: 2px;
        background: rgba(99, 102, 241, 0.8);
        border-radius: 99px;
    }

    .sidebar-title {
        color: #ffffff;
        font-size: 38px;
        font-weight: 900;
        line-height: 0.95;
        letter-spacing: -1.5px;
        margin-bottom: 14px;
        background: linear-gradient(135deg, #ffffff 30%, rgba(147,197,253,0.9));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .sidebar-copy {
        color: rgba(210, 225, 245, 0.72);
        font-size: 13.5px;
        line-height: 1.7;
        margin-bottom: 22px;
        max-width: 220px;
        font-weight: 400;
    }

    section[data-testid="stSidebar"] .stRadio { margin-top: 6px; margin-bottom: 22px; }
    section[data-testid="stSidebar"] .stRadio > div { gap: 8px; }

    section[data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 13px !important;
        min-height: 50px;
        padding: 13px 16px !important;
        transition: all 0.25s cubic-bezier(0.22, 1, 0.36, 1);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.07) !important;
        border-color: rgba(147, 197, 253, 0.35);
        transform: translateX(3px);
    }

    section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: linear-gradient(135deg, rgba(37,99,235,0.3) 0%, rgba(99,102,241,0.25) 100%) !important;
        border-color: rgba(147, 197, 253, 0.55) !important;
        box-shadow: 0 8px 24px rgba(8, 18, 36, 0.3), inset 0 1px 0 rgba(255,255,255,0.1);
    }

    section[data-testid="stSidebar"] .stRadio label > div:first-child { display: none; }

    section[data-testid="stSidebar"] .stRadio label p {
        font-size: 14.5px !important;
        font-weight: 600 !important;
        color: #eaf2ff !important;
        letter-spacing: 0.1px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        margin-top: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.06);
        color: #d4e5ff !important;
        min-height: 44px;
        font-weight: 600;
        font-size: 13.5px;
        transition: all 0.22s ease;
        letter-spacing: 0.2px;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(147, 197, 253, 0.4);
        background: rgba(255,255,255,0.12);
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
    }

    .sidebar-panel {
        margin-top: 18px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px;
        padding: 14px 16px;
        backdrop-filter: blur(12px);
    }

    .sidebar-panel-title {
        color: rgba(99, 102, 241, 0.85);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 9.5px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .sidebar-panel-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        padding: 7px 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .sidebar-panel-row:last-child { border-bottom: 0; }
    .sidebar-panel-row strong { color: #a8c4ea !important; font-weight: 700; }

    /* === TYPOGRAPHY === */
    h1 {
        color: var(--text-strong) !important;
        font-size: 40px !important;
        font-weight: 900 !important;
        letter-spacing: -1.2px !important;
        margin-bottom: 4px !important;
        line-height: 1.1 !important;
    }

    h2 {
        color: var(--text-strong) !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        margin-top: 30px !important;
    }

    h3 {
        color: #152b4e !important;
        font-size: 19px !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
    }

    p {
        color: var(--text-muted) !important;
        line-height: 1.65 !important;
    }

    .section-copy { margin-bottom: 18px; font-size: 15px; }

    /* === METRIC CARDS === */
    div[data-testid="metric-container"] {
        background: var(--surface);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius-md);
        padding: 22px 20px;
        box-shadow: var(--shadow-soft);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-strong);
    }

    div[data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        left: 0; top: 0;
        width: 100%; height: 3.5px;
        background: linear-gradient(90deg, #2563eb 0%, #6366f1 50%, #8b5cf6 100%);
    }

    div[data-testid="metric-container"]::after {
        content: '';
        position: absolute;
        right: -20px; top: -20px;
        width: 80px; height: 80px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 70%);
    }

    div[data-testid="metric-container"] label {
        color: var(--text-muted) !important;
        font-size: 10.5px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--text-strong) !important;
        font-size: 30px !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
    }

    /* === BUTTONS === */
    .stButton > button {
        border-radius: 11px;
        border: 1px solid var(--line-mid);
        background: #ffffff;
        color: #0f2240;
        box-shadow: var(--shadow-xs);
        font-weight: 600;
        font-size: 13.5px;
        letter-spacing: 0.1px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #93c5fd;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.14);
        transform: translateY(-1px);
    }

    div[data-testid="column"] .stButton > button {
        border-radius: 999px;
        min-height: 40px;
    }

    /* === CHARTS === */
    .js-plotly-plot {
        background: var(--surface);
        border-radius: var(--radius-md);
        border: 1px solid var(--line-soft);
        box-shadow: var(--shadow-soft);
        padding: 10px;
    }

    /* === DATAFRAMES === */
    .dataframe {
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        border: 1px solid var(--line-soft) !important;
        box-shadow: var(--shadow-soft) !important;
    }

    .dataframe thead tr th {
        background: linear-gradient(135deg, #0f2240 0%, #1e3d68 100%) !important;
        color: #ffffff !important;
        font-size: 11.5px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
        padding: 12px 14px !important;
    }

    /* === EXPANDABLE === */
    .streamlit-expanderHeader {
        border: 1px solid var(--line-soft) !important;
        border-radius: var(--radius-sm) !important;
        background: #ffffff !important;
        color: var(--text-strong) !important;
        font-weight: 600 !important;
    }

    /* === CHAT === */
    .stChatMessage {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--line-soft) !important;
        background: #ffffff !important;
        box-shadow: var(--shadow-xs) !important;
    }

    /* === INPUTS === */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stTextArea > div > div > textarea {
        border-radius: 11px !important;
        border: 1.5px solid rgba(15, 24, 42, 0.16) !important;
        background: #ffffff !important;
        font-size: 14px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
        outline: none !important;
    }

    /* === STATUS CHIP === */
    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 999px;
        border: 1px solid rgba(16, 185, 129, 0.25);
        background: rgba(16, 185, 129, 0.08);
        color: #065f46;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        padding: 7px 14px;
    }

    .status-chip::before {
        content: '';
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.2);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.2); }
        50% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0.08); }
    }

    /* === CARDS === */
    .insight-card, .tier-card, .alert-card {
        background: var(--surface);
        border: 1px solid var(--line-soft);
        box-shadow: var(--shadow-soft);
    }

    .insight-card {
        border-radius: var(--radius-md);
        padding: 22px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .insight-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-strong);
    }

    .insight-title {
        color: var(--text-strong);
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 10px;
        letter-spacing: -0.2px;
    }

    .insight-text { color: var(--text-muted); font-size: 14px; line-height: 1.65; }

    .tier-card {
        border-radius: var(--radius-md);
        padding: 18px;
        transition: transform 0.2s ease;
    }

    .tier-card:hover { transform: translateY(-2px); }

    .tier-card-title {
        color: var(--text-strong);
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .tier-card-copy { color: var(--text-muted); font-size: 13px; line-height: 1.6; }

    .alert-card {
        border-radius: var(--radius-md);
        padding: 16px;
        margin-bottom: 10px;
        transition: transform 0.2s ease;
    }

    .alert-card:hover { transform: translateX(2px); }
    .alert-card-title { color: var(--text-strong); font-size: 14px; font-weight: 700; margin-bottom: 4px; }
    .alert-card-copy { color: var(--text-muted); font-size: 12.5px; line-height: 1.6; }

    /* === DIVIDERS === */
    hr {
        border: none;
        border-top: 1px solid var(--line-soft);
        margin: 8px 0 20px 0;
    }

    /* === LOGIN PAGE SPECIAL STYLES === */
    .login-page-bg {
        min-height: 100vh;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding-top: 40px;
    }

    .login-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 999px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #4338ca;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 18px;
    }

    @media (max-width: 1024px) {
        .main .block-container { padding-top: 1.2rem; }
        h1 { font-size: 32px !important; }
        .sidebar-title { font-size: 32px; }
    }
</style>
""", unsafe_allow_html=True)
# ============================================================================
# SESSION STATE
# ============================================================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.kpis = None
    st.session_state.customer_scores = None
    st.session_state.churn_predictions = None
    st.session_state.revenue_forecast = None
    st.session_state.copilot = None
    st.session_state.forecast_days = 30

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False

if 'auth_user' not in st.session_state:
    st.session_state.auth_user = None

if 'auth_bootstrapped' not in st.session_state:
    st.session_state.auth_bootstrapped = False


def _safe_cache_key(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)

def _cache_paths(invoice_collection: str):
    os.makedirs("cache", exist_ok=True)
    key = _safe_cache_key(invoice_collection)
    return (
        f"cache/{key}_processed.parquet",
        f"cache/{key}_meta.txt",
    )

def _read_processed_cache(invoice_collection: str):
    parquet_path, _ = _cache_paths(invoice_collection)
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return None

def _write_processed_cache(invoice_collection: str, df: pd.DataFrame):
    parquet_path, _ = _cache_paths(invoice_collection)
    df.to_parquet(parquet_path, index=False)

# ============================================================================
# AUTH / ACCESS CONTROL HELPERS
# ============================================================================

def reset_user_data_cache():
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.kpis = None
    st.session_state.customer_scores = None
    st.session_state.churn_predictions = None
    st.session_state.revenue_forecast = None
    st.session_state.copilot = None
    st.session_state.chat_history = []
    st.session_state.forecast_days = 30

def init_auth_bootstrap():
    if st.session_state.auth_bootstrapped:
        return
    db = DatabaseHandler()
    if db.connect():
        try:
            db.ensure_auth_indexes()
            db.bootstrap_default_users()
        finally:
            db.close()
    st.session_state.auth_bootstrapped = True

def login_view():
    # Hero / brand header
    st.markdown("""
    <div style='text-align:center; padding: 48px 0 32px 0;'>
        <div style='
            display:inline-flex; align-items:center; gap:10px;
            padding:7px 18px; border-radius:999px;
            background:rgba(99,102,241,0.10); border:1px solid rgba(99,102,241,0.22);
            color:#4338ca; font-size:11px; font-weight:800;
            text-transform:uppercase; letter-spacing:1.4px; margin-bottom:22px;
        '>✦ AI Finance Control Center</div>
        <div style='
            font-size:48px; font-weight:900; color:#0d1829;
            letter-spacing:-2px; line-height:1.05; margin-bottom:12px;
            background: linear-gradient(135deg, #0d1829 0%, #1e3a6e 60%, #6366f1 100%);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
        '>Your Finance<br>Command Center</div>
        <div style='color:#5a6a82; font-size:16px; max-width:480px; margin:0 auto; line-height:1.6;'>
            Secure access to real-time revenue intelligence, predictive analytics,
            and AI-powered financial insights.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Three feature pills
    st.markdown("""
    <div style='display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-bottom:36px;'>
        <div style='display:flex;align-items:center;gap:8px;padding:10px 18px;
                    background:#fff;border:1px solid rgba(15,24,42,0.09);
                    border-radius:999px;box-shadow:0 4px 12px rgba(12,24,48,0.07);
                    font-size:13px;font-weight:600;color:#1e3a6e;'>
            <span style='font-size:16px;'>📊</span> Live Revenue Dashboard
        </div>
        <div style='display:flex;align-items:center;gap:8px;padding:10px 18px;
                    background:#fff;border:1px solid rgba(15,24,42,0.09);
                    border-radius:999px;box-shadow:0 4px 12px rgba(12,24,48,0.07);
                    font-size:13px;font-weight:600;color:#1e3a6e;'>
            <span style='font-size:16px;'>🤖</span> AI Revenue Forecasting
        </div>
        <div style='display:flex;align-items:center;gap:8px;padding:10px 18px;
                    background:#fff;border:1px solid rgba(15,24,42,0.09);
                    border-radius:999px;box-shadow:0 4px 12px rgba(12,24,48,0.07);
                    font-size:13px;font-weight:600;color:#1e3a6e;'>
            <span style='font-size:16px;'>💬</span> Finance AI Copilot
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Login card
    _, card_col, _ = st.columns([1, 2.2, 1])
    with card_col:
        st.markdown("""
        <div style='
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(15,24,42,0.10);
            border-radius: 24px;
            padding: 36px 40px 32px 40px;
            box-shadow: 0 24px 64px rgba(12,24,48,0.13), 0 1px 0 rgba(255,255,255,0.8) inset;
            backdrop-filter: blur(20px);
        '>
            <div style='font-size:22px;font-weight:800;color:#0d1829;letter-spacing:-0.4px;margin-bottom:4px;'>
                🔐 Sign in to your workspace
            </div>
            <div style='color:#5a6a82;font-size:13.5px;margin-bottom:24px;line-height:1.5;'>
                Enter your credentials to access your assigned finance dashboard.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                username = st.text_input("Username", placeholder="Enter username")
            with c2:
                password = st.text_input("Password", type="password", placeholder="Enter password")

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("✦ Sign In", use_container_width=True)

        st.markdown("""
        <div style='text-align:center; margin-top:14px;'>
            <span style='font-size:12px; color:#94a3b8;'>
                🔒 Secured workspace access &nbsp;·&nbsp; <em>admin/admin123 &nbsp;·&nbsp; user_a/usera123</em>
            </span>
        </div>
        """, unsafe_allow_html=True)

    if submitted:
        if not username or not password:
            st.error("Please enter username and password.")
            return

        db = DatabaseHandler()
        if not db.connect():
            st.error("Database connection failed.")
            return

        try:
            user = db.get_user(username.strip())
            if not user:
                st.error("Invalid credentials.")
                return

            if not db.verify_password(password, user.get("password_hash", "")):
                st.error("Invalid credentials.")
                return

            st.session_state.is_authenticated = True
            st.session_state.auth_user = {
                "username": user["username"],
                "role": user["role"],
                "assigned_invoice_collection": user.get("assigned_invoice_collection"),
                "assigned_line_collection": user.get("assigned_line_collection"),
            }
            reset_user_data_cache()
            st.success(f"✅ Welcome back, {user['username']}!")
            st.rerun()
        finally:
            db.close()

def admin_panel():
    st.title("Admin Control Panel")
    st.markdown("**Create users and assign/change user-specific collections.**")
    st.markdown("---")

    db = DatabaseHandler()
    if not db.connect():
        st.error("Unable to connect to MongoDB for admin operations.")
        return

    try:
        allowed_invoice_collections = [
            "ZATCA_API_INVOICES_ALL",
            "zatca-pro",
        ]
        allowed_line_collections = [
            "ZATCA_API_INVOICES_LINES_ALL",
            "zatca-pro-line",
        ]

        tab1, tab2 = st.tabs(["Create User", "Manage Users"])

        with tab1:
            with st.form("create_user_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                new_role = st.selectbox("Role", ["user", "admin"])
                new_invoice_collection = st.selectbox(
                    "Assigned Invoice Collection (user role)",
                    allowed_invoice_collections,
                    disabled=(new_role == "admin")
                )
                new_line_collection = st.selectbox(
                    "Assigned Line Collection (user role)",
                    allowed_line_collections,
                    disabled=(new_role == "admin")
                )
                create_btn = st.form_submit_button("Create User", use_container_width=True)

            if create_btn:
                if not new_username.strip() or not new_password.strip():
                    st.error("Username and password are required.")
                else:
                    invoice_collection = new_invoice_collection if new_role == "user" else None
                    line_collection = new_line_collection if new_role == "user" else None
                    ok, msg = db.create_user(
                        username=new_username.strip(),
                        password=new_password.strip(),
                        role=new_role,
                        assigned_invoice_collection=invoice_collection,
                        assigned_line_collection=line_collection,
                    )
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        with tab2:
            users = db.list_users()
            if users:
                users_df = pd.DataFrame(users)
                st.dataframe(users_df, use_container_width=True, hide_index=True)

                st.markdown("### Update User Assignment")
                with st.form("update_user_form"):
                    usernames = [u["username"] for u in users]
                    selected_user = st.selectbox("Select User", usernames)

                    # prefill friendly defaults
                    selected_role = st.selectbox("Role", ["user", "admin"])
                    selected_invoice_collection = st.selectbox(
                        "Assigned Invoice Collection",
                        allowed_invoice_collections,
                        disabled=(selected_role == "admin")
                    )
                    selected_line_collection = st.selectbox(
                        "Assigned Line Collection",
                        allowed_line_collections,
                        disabled=(selected_role == "admin")
                    )
                    update_btn = st.form_submit_button("Update User", use_container_width=True)

                if update_btn:
                    invoice_collection = selected_invoice_collection if selected_role == "user" else None
                    line_collection = selected_line_collection if selected_role == "user" else None
                    ok, msg = db.update_user_assignment(
                        username=selected_user,
                        role=selected_role,
                        assigned_invoice_collection=invoice_collection,
                        assigned_line_collection=line_collection,
                    )
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                st.info("No users found.")
    finally:
        db.close()

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data(ttl=300)
def load_and_process_data(invoice_collection: str, line_items_collection: str):
    """Option A: full-data load once, then fast parquet reuse."""
    cached_df = _read_processed_cache(invoice_collection)
    if cached_df is not None and not cached_df.empty:
        print(f"⚡ Loaded processed cache for {invoice_collection}")
        return cached_df

    db = DatabaseHandler(
        database_name=Config.DATABASE_NAME,
        invoice_collection=invoice_collection,
        line_items_collection=line_items_collection,
    )

    if not db.connect():
        return None

    projection = {
        "INVOICE_ID": 1,
        "ISSUE_DATE": 1,
        "DOCUMENTCURRENCYCODE": 1,
        "CUSTOMER_PARTY_LEGAL_ENTITY_REGISTRATION_NAME": 1,
        "CUSTOMER_PARTY_TAX_SCHEME_COMPANYID": 1,
        "CUSTOMER_POSTA_ADDRESS_STREET_NAME": 1,
        "CUSTOMER_POSTALADDRESS_CITY_NAME": 1,
        "TAXTOTAL_TAX_AMOUNT_VALUE": 1,
        "TAX_EXCLUSIVE_AMOUNT_VALUE": 1,
        "TAX_INCLUSIVE_AMOUNT_VALUE": 1,
        "DISCOUNT_AMOUNT_VALUE": 1,
        "PAYABLE_AMOUNT_VALUE": 1,
        "INVOICE_TYPE_CODE_VALUE": 1,
        "INVOICE_TYPE_CODE_NAME": 1,
        "Status": 1,
    }

    df_raw = db.get_invoices(projection=projection)
    db.close()

    if df_raw.empty:
        return None

    processor = DataProcessor()
    df_clean = processor.clean_invoices(df_raw)
    df_processed = processor.engineer_features(df_clean)

    _write_processed_cache(invoice_collection, df_processed)
    return df_processed
# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_revenue_trend(df: pd.DataFrame):
    """Monthly revenue and invoice volume trend"""

    monthly = df.groupby('ISSUE_MONTH').agg({
        'PAYABLE_AMOUNT_VALUE': 'sum',
        'INVOICE_ID': 'count',
        'IS_PAID': 'mean'
    }).reset_index()

    monthly.columns = ['MONTH', 'REVENUE', 'INVOICES', 'PAYMENT_RATE']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=monthly['MONTH'],
            y=monthly['INVOICES'],
            name='Invoice Volume',
            marker=dict(color='rgba(158, 221, 217, 0.95)', line=dict(width=0)),
            hovertemplate='<b>%{x}</b><br>Invoice volume: %{y}<extra></extra>'
        ),
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(
            x=monthly['MONTH'],
            y=monthly['REVENUE'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#4f5ff0', width=3),
            marker=dict(size=7, color='#4f5ff0', line=dict(color='white', width=1.5)),
            hovertemplate='<b>%{x}</b><br>Revenue: SAR %{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )

    fig.update_layout(
        title=dict(
            text='Revenue & Volume Trends',
            font=dict(size=32, family='Outfit, sans-serif', color='#111a2b'),
            x=0,
            y=0.98,
            xanchor='left'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.22,
            xanchor='center',
            x=0.5,
            font=dict(size=12, color='#5f6f87')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=430,
        hovermode='x unified',
        margin=dict(t=72, b=80, l=60, r=50)
    )

    fig.update_xaxes(
        showgrid=False,
        tickfont=dict(size=11, color='#72839b')
    )
    fig.update_yaxes(
        title_text='Revenue (SAR Millions)',
        title_font=dict(size=11, color='#6e8099'),
        tickfont=dict(size=10, color='#7a8ca4'),
        gridcolor='rgba(17,26,43,0.08)',
        zeroline=False,
        secondary_y=False
    )
    fig.update_yaxes(
        title_text='Invoice Count',
        title_font=dict(size=11, color='#6e8099'),
        tickfont=dict(size=10, color='#7a8ca4'),
        showgrid=False,
        zeroline=False,
        secondary_y=True
    )

    return fig

def plot_invoice_status(df: pd.DataFrame):
    """Invoice status distribution donut chart"""
    
    status_counts = df['Status'].value_counts()
    
    colors = {
        'Cleared': '#10b981',
        'Cleared-With-Warning': '#f59e0b',
        'Non-Taxable': '#6366f1',
        'pending': '#ef4444',
        'Exception': '#dc2626'
    }
    
    chart_colors = [colors.get(s, '#64748b') for s in status_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.6,
        marker=dict(colors=chart_colors, line=dict(color='white', width=3)),
        textinfo='label+percent',
        textfont=dict(size=13, weight='bold'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
    )])
    
    total = status_counts.sum()
    fig.add_annotation(
        text=f"<b style='font-size:26px;'>{total}</b><br><span style='font-size:13px; color:#64748b;'>Total<br>Invoices</span>",
        x=0.5, y=0.5,
        font=dict(size=14, color='#0f172a'),
        showarrow=False,
        align='center'
    )
    
    fig.update_layout(
        title=dict(text='Invoice Status', font=dict(size=18, weight='bold', color='#0f172a')),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=350,
        margin=dict(t=60, b=60, l=40, r=40)
    )
    
    return fig

def plot_business_type_split(df: pd.DataFrame):
    """B2B vs B2C revenue split"""
    
    business_revenue = df.groupby('BUSINESS_TYPE')['PAYABLE_AMOUNT_VALUE'].sum()
    
    colors = {'B2B': '#6366f1', 'B2C': '#8b5cf6', 'Unknown': '#94a3b8'}
    chart_colors = [colors.get(bt, '#64748b') for bt in business_revenue.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=business_revenue.index,
        values=business_revenue.values,
        marker=dict(colors=chart_colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        textfont=dict(size=14, weight='bold'),
        hovertemplate='<b>%{label}</b><br>Revenue: SAR %{value:,.0f}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text='B2B vs B2C Revenue', font=dict(size=18, weight='bold', color='#0f172a')),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=350,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    return fig

def plot_top_customers(customer_scores: pd.DataFrame):
    """Top customers horizontal bar chart"""

    top_10 = customer_scores.head(10).sort_values('TOTAL_REVENUE', ascending=True).copy()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top_10['CUSTOMER_NAME'].str[:52],
        x=top_10['TOTAL_REVENUE'],
        orientation='h',
        marker=dict(
            color=top_10['TOTAL_REVENUE'],
            colorscale=[[0, '#3b82f6'], [1, '#5b62f1']],
            line=dict(width=0)
        ),
        text=top_10['TOTAL_REVENUE'].apply(lambda x: f"{x/1_000_000:.0f}M"),
        textposition='outside',
        cliponaxis=False,
        textfont=dict(size=12, color='#111a2b', family='Outfit, sans-serif'),
        hovertemplate='<b>%{y}</b><br>Revenue: SAR %{x:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text='Top 10 Customers by Revenue',
            font=dict(size=30, family='Outfit, sans-serif', color='#111a2b'),
            x=0
        ),
        annotations=[
            dict(
                text='VALUES IN SAR (MILLIONS)',
                x=1,
                y=1.12,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=11, color='#8aa0ba', family='Outfit, sans-serif')
            )
        ],
        xaxis=dict(
            title='',
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            title='',
            showgrid=False,
            tickfont=dict(size=12, color='#2a3f5d'),
            automargin=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=430,
        margin=dict(t=80, b=30, l=330, r=70)
    )

    return fig

def plot_churn_risk(churn_predictions: pd.DataFrame):
    """Churn risk distribution"""
    
    risk_counts = churn_predictions['CHURN_RISK_LEVEL'].value_counts()
    
    colors = {'Critical': '#dc2626', 'High': '#f59e0b', 'Medium': '#3b82f6', 'Low': '#10b981'}
    chart_colors = [colors.get(level, '#64748b') for level in risk_counts.index]
    
    fig = go.Figure(data=[go.Bar(
        x=risk_counts.index,
        y=risk_counts.values,
        marker=dict(color=chart_colors, line=dict(width=0)),
        text=risk_counts.values,
        textposition='outside',
        textfont=dict(size=13, weight='bold'),
        hovertemplate='<b>%{x} Risk</b><br>Customers: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text='Customer Churn Risk Distribution', font=dict(size=18, weight='bold', color='#0f172a')),
        xaxis=dict(title='Risk Level', showgrid=False, categoryorder='array', categoryarray=['Critical', 'High', 'Medium', 'Low']),
        yaxis=dict(title='Number of Customers', showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=350,
        showlegend=False,
        margin=dict(t=60, b=60, l=60, r=40)
    )
    
    return fig

def plot_revenue_forecast(forecast_df: pd.DataFrame, title: str = "Revenue Forecast"):
    """Revenue forecast with confidence intervals"""
    
    if forecast_df is None or forecast_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Upper confidence bound
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['estimate_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Lower confidence bound with fill
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['estimate_lower'],
        mode='lines',
        name='Confidence Interval (80%)',
        line=dict(width=0),
        fillcolor='rgba(99,102,241,0.15)',
        fill='tonexty',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Estimation'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#6366f1', width=3),
        marker=dict(size=7, color='#6366f1', line=dict(color='white', width=2)),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Forecast: SAR %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, weight='bold', color='#0f172a')),
        xaxis=dict(title='Date', showgrid=False),
        yaxis=dict(title='Estimated Revenue (SAR)', showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=480,
        hovermode='x',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60, l=80, r=40)
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""

    # Initialize auth seed users/index once
    init_auth_bootstrap()

    # Login gate
    if not st.session_state.is_authenticated:
        login_view()
        return

    current_user = st.session_state.auth_user

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class='sidebar-brand'>
            <div class='sidebar-kicker'>Finance Intelligence</div>
            <div class='sidebar-title'>Finance<br>Control</div>
            <div class='sidebar-copy'>Live visibility across revenue, collections, and customer risk.</div>
        </div>
        """, unsafe_allow_html=True)

        if current_user and current_user.get("role") == "admin":
            view_options = ["Admin Panel"]
        else:
            view_options = ["Dashboard", "Revenue Forecast", "AI Copilot"]

        view_mode = st.radio(
            "Select View",
            view_options,
            label_visibility="collapsed"
        )

        st.markdown(f"**User:** {current_user.get('username', '-')}")
        st.markdown(f"**Role:** {current_user.get('role', '-')}")

        if st.button("Logout", use_container_width=True):
            st.session_state.is_authenticated = False
            st.session_state.auth_user = None
            reset_user_data_cache()
            st.rerun()

        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.data_loaded = False
            if current_user.get("role") == "user":
                parquet_path, _ = _cache_paths(current_user.get("assigned_invoice_collection", "default"))
                if os.path.exists(parquet_path):
                    os.remove(parquet_path)
            st.rerun()

        invoice_count = f"{len(st.session_state.df):,}" if st.session_state.data_loaded and st.session_state.df is not None else "-"
        customer_count = f"{st.session_state.df['CUSTOMER_NAME'].nunique():,}" if st.session_state.data_loaded and st.session_state.df is not None else "-"
        active_workspace = current_user.get("assigned_invoice_collection") if current_user.get("role") == "user" else "Admin Panel"

        st.markdown(f"""
        <div class='sidebar-panel'>
            <div class='sidebar-panel-title'>Workspace</div>
            <div class='sidebar-panel-row'><span>Database</span><strong>{Config.DATABASE_NAME}</strong></div>
            <div class='sidebar-panel-row'><span>Collection</span><strong>{active_workspace if active_workspace else '-'}</strong></div>
            <div class='sidebar-panel-row'><span>Invoices</span><strong>{invoice_count}</strong></div>
            <div class='sidebar-panel-row'><span>Customers</span><strong>{customer_count}</strong></div>
            <div class='sidebar-panel-row'><span>Updated</span><strong>{datetime.now().strftime('%H:%M:%S')}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # Admin route
    if current_user.get("role") == "admin":
        admin_panel()
        return

    # Enforce strict per-user collection isolation
    assigned_invoice_collection = current_user.get("assigned_invoice_collection")
    assigned_line_collection = current_user.get("assigned_line_collection")

    if not assigned_invoice_collection or not assigned_line_collection:
        st.error("No collection assignment found for this user. Please contact admin.")
        return

    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading finance data..."):
            df = load_and_process_data(assigned_invoice_collection, assigned_line_collection)

            if df is None or df.empty:
                st.error("No data available. Check the database/collection configuration.")
                return

            # Calculate KPIs
            kpis = MetricsCalculator.calculate_kpis(df)

            # Customer scoring
            scorer = CustomerScorer()
            customer_scores = scorer.calculate_scores(df)

            # Churn prediction
            churn_predictor = ChurnPredictor()
            churn_predictions = churn_predictor.predict_churn(df)

            # Revenue forecasting
            forecaster = RevenueForecaster()
            forecaster.train(df)
            revenue_forecast = forecaster.predict(periods=30)

            # AI Copilot
            copilot = AICopilot(df)

            # Store in session
            st.session_state.df = df
            st.session_state.kpis = kpis
            st.session_state.customer_scores = customer_scores
            st.session_state.churn_predictions = churn_predictions
            st.session_state.revenue_forecast = revenue_forecast
            st.session_state.copilot = copilot
            st.session_state.data_loaded = True

    # Get data from session
    df = st.session_state.df
    kpis = st.session_state.kpis
    customer_scores = st.session_state.customer_scores
    churn_predictions = st.session_state.churn_predictions
    revenue_forecast = st.session_state.revenue_forecast
    copilot = st.session_state.copilot

    # Page routing
    if view_mode == "Dashboard":
        show_dashboard(df, kpis, customer_scores, churn_predictions)

    elif view_mode == "Revenue Forecast":
        show_revenue_prediction(df, revenue_forecast)

    elif view_mode == "AI Copilot":
        show_ai_copilot(copilot, df)

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================

def show_dashboard(df, kpis, customer_scores, churn_predictions):
    """Main Dashboard - Executive Overview"""

    # Header
    col_h1, col_h2 = st.columns([5, 1])

    with col_h1:
        st.title("Finance Control Dashboard")
        st.markdown("**Trend performance and account concentration across your portfolio.**")

    with col_h2:
        st.markdown("""
        <div style='text-align: right; padding-top: 18px;'>
            <span class='status-chip'>
                Live data
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # KPIs
    st.markdown("### Key Metrics")
    st.markdown("<p class='section-copy'>Revenue, cash, and collections at a glance.</p>", unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="medium")

    # Calculate Revenue Run Rate
    days_in_month = datetime.now().day
    days_total = pd.Timestamp.now().days_in_month
    revenue_run_rate = (kpis['mtd_revenue'] / days_in_month) * days_total if days_in_month > 0 else 0

    with kpi1:
        st.metric(
            "Revenue (MTD)",
            Formatter.format_currency(kpis['mtd_revenue']),
            f"{kpis['total_invoices']} invoices"
        )

    with kpi2:
        st.metric(
            "Revenue Run Rate",
            Formatter.format_currency(revenue_run_rate),
            "Monthly projection"
        )

    with kpi3:
        st.metric(
            "Outstanding",
            Formatter.format_currency(kpis['outstanding_invoices']),
            f"{kpis['unpaid_invoices']} unpaid"
        )

    with kpi4:
        st.metric(
            "Expected Cash (30d)",
            Formatter.format_currency(kpis['expected_cash_inflow_30d']),
            "Next 30 days"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Customer Trends & Analytics
    st.markdown("### Revenue and Customers")
    st.markdown("<p class='section-copy'>Trend performance and account concentration.</p>", unsafe_allow_html=True)

    st.plotly_chart(plot_top_customers(customer_scores), use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(plot_revenue_trend(df), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Invoice Status
    st.markdown("### Status and Mix")
    st.markdown("<p class='section-copy'>Payment status and business composition.</p>", unsafe_allow_html=True)

    status1, status2 = st.columns([1, 1], gap="large")

    with status1:
        st.plotly_chart(plot_invoice_status(df), use_container_width=True)

    with status2:
        st.plotly_chart(plot_business_type_split(df), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Customer Value Scoring
    st.markdown("### Customer Segments")
    st.markdown("<p class='section-copy'>Tiers based on revenue, frequency, recency, and reliability.</p>", unsafe_allow_html=True)

    tier_summary = customer_scores.groupby('TIER').agg({
        'CUSTOMER_NAME': 'count',
        'TOTAL_REVENUE': 'sum'
    }).reset_index()

    tier_col1, tier_col2, tier_col3, tier_col4 = st.columns(4)

    for idx, (col, tier) in enumerate(zip([tier_col1, tier_col2, tier_col3, tier_col4], ['A', 'B', 'C', 'D'])):
        tier_data = tier_summary[tier_summary['TIER'] == tier]
        if not tier_data.empty:
            count = int(tier_data['CUSTOMER_NAME'].values[0])
            revenue = tier_data['TOTAL_REVENUE'].values[0]
        else:
            count = 0
            revenue = 0

        tier_colors = {'A': '#10b981', 'B': '#3b82f6', 'C': '#f59e0b', 'D': '#ef4444'}

        with col:
            st.markdown(f"""
            <div class='tier-card' style='border-top: 4px solid {tier_colors[tier]};'>
                <div class='tier-card-title'>Tier {tier}</div>
                <div class='tier-card-copy'>
                    {count} customers<br>
                    {Formatter.format_currency(revenue)}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top customers table
    display_customers = customer_scores.head(10).copy()
    display_customers['TOTAL_REVENUE'] = display_customers['TOTAL_REVENUE'].apply(Formatter.format_currency)
    display_customers['CUSTOMER_SCORE'] = display_customers['CUSTOMER_SCORE'].apply(lambda x: f"{x:.1f}")
    display_customers['PAYMENT_RATE'] = display_customers['PAYMENT_RATE'].apply(lambda x: f"{x*100:.0f}%")

    st.dataframe(
        display_customers[['CUSTOMER_NAME', 'TIER', 'CUSTOMER_SCORE', 'TOTAL_REVENUE', 'INVOICE_COUNT', 'PAYMENT_RATE', 'RELIABILITY_GRADE']],
        use_container_width=True,
        hide_index=True,
        height=350
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Customer Churn Prediction
    st.markdown("### Churn Risk")
    st.markdown("<p class='section-copy'>Accounts that need attention now.</p>", unsafe_allow_html=True)

    churn_col1, churn_col2 = st.columns([1.5, 2], gap="large")

    with churn_col1:
        st.plotly_chart(plot_churn_risk(churn_predictions), use_container_width=True)

    with churn_col2:
        st.markdown("#### Priority Accounts")

        high_risk = churn_predictions[churn_predictions['CHURN_RISK_LEVEL'].isin(['High', 'Critical'])].head(5)

        if not high_risk.empty:
            for _, row in high_risk.iterrows():
                risk_color = '#dc2626' if row['CHURN_RISK_LEVEL'] == 'Critical' else '#f59e0b'

                st.markdown(f"""
                <div class='alert-card' style='border-left: 4px solid {risk_color};'>
                    <div class='alert-card-title'>{row['CUSTOMER_NAME'][:50]}</div>
                    <div class='alert-card-copy'>
                        <strong>Risk</strong> {row['CHURN_PROBABILITY']:.0f}% | 
                        <strong>Open invoices</strong> {row['UNPAID_INVOICE_COUNT']}<br>
                        <strong>Next step</strong> {row['RECOMMENDED_ACTIONS']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No priority churn accounts identified.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Business Insights
    st.markdown("### Executive Notes")

    ins1, ins2, ins3 = st.columns(3, gap="medium")

    with ins1:
        payment_rate = (kpis['paid_invoices'] / kpis['total_invoices'] * 100)
        st.markdown(f"""
        <div class='insight-card' style='border-left-color: #10b981;'>
            <div class='insight-title'>Collections</div>
            <div class='insight-text'>
                <strong>{payment_rate:.1f}%</strong> of invoices have been collected.
                <br><br>
                <strong>Revenue</strong> {Formatter.format_currency(kpis['total_revenue'])}<br>
                <strong>Paid invoices</strong> {kpis['paid_invoices']:,}<br>
                <strong>Average invoice</strong> {Formatter.format_currency(kpis['avg_invoice_value'])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ins2:
        b2b_revenue = df[df['BUSINESS_TYPE'] == 'B2B']['PAYABLE_AMOUNT_VALUE'].sum()
        b2b_pct = (b2b_revenue / kpis['total_revenue'] * 100) if kpis['total_revenue'] > 0 else 0

        st.markdown(f"""
        <div class='insight-card' style='border-left-color: #6366f1;'>
            <div class='insight-title'>Business Mix</div>
            <div class='insight-text'>
                <strong>B2B revenue</strong> {Formatter.format_currency(b2b_revenue)} ({b2b_pct:.1f}%)
                <br><br>
                Enterprise accounts remain the primary revenue source.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ins3:
        high_churn_count = len(churn_predictions[churn_predictions['CHURN_PROBABILITY'] >= 60])

        st.markdown(f"""
        <div class='insight-card' style='border-left-color: #f59e0b;'>
            <div class='insight-title'>Retention</div>
            <div class='insight-text'>
                <strong>{high_churn_count}</strong> customers are above the 60% churn threshold.
                <br><br>
                Focus outreach on high-value at-risk accounts.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: AI REVENUE PREDICTION
# ============================================================================

def show_revenue_prediction(df, revenue_forecast):
    """AI Revenue Prediction Page"""

    st.title("Revenue Forecast")
    st.markdown("**Short-term revenue forecast with confidence bands.**")
    st.markdown("---")

    # Forecast period selector (buttons)
    st.markdown("### Forecast Horizon")
    st.markdown("<p class='section-copy'>Select a planning window.</p>", unsafe_allow_html=True)

    period_col1, period_col2, period_col3, period_col4, period_col5, spacer = st.columns([1, 1, 1, 1, 1, 2])

    with period_col1:
        if st.button("7 Days", use_container_width=True):
            st.session_state.forecast_days = 7
            st.rerun()

    with period_col2:
        if st.button("14 Days", use_container_width=True):
            st.session_state.forecast_days = 14
            st.rerun()

    with period_col3:
        if st.button("30 Days", use_container_width=True):
            st.session_state.forecast_days = 30
            st.rerun()

    with period_col4:
        if st.button("60 Days", use_container_width=True):
            st.session_state.forecast_days = 60
            st.rerun()

    with period_col5:
        if st.button("90 Days", use_container_width=True):
            st.session_state.forecast_days = 90
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Generate forecast
    forecaster = RevenueForecaster()
    forecaster.train(df)
    forecast_df = forecaster.predict(periods=st.session_state.forecast_days)

    # Summary metrics
    if not forecast_df.empty:
        metric1, metric2, metric3, metric4 = st.columns(4)

        total_forecast = forecast_df['Estimation'].sum()
        avg_daily = forecast_df['Estimation'].mean()
        max_day = forecast_df.loc[forecast_df['Estimation'].idxmax(), 'Date']
        max_value = forecast_df['Estimation'].max()

        with metric1:
            st.metric("Total Forecast", Formatter.format_currency(total_forecast), f"{st.session_state.forecast_days} days")

        with metric2:
            st.metric("Daily Average", Formatter.format_currency(avg_daily))

        with metric3:
            st.metric("Peak Day", max_day.strftime('%b %d'))

        with metric4:
            st.metric("Peak Revenue", Formatter.format_currency(max_value))

    st.markdown("<br>", unsafe_allow_html=True)

    # Main forecast chart
    title = f"Revenue Forecast: Next {st.session_state.forecast_days} Days"
    st.plotly_chart(plot_revenue_forecast(forecast_df, title), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast data table
    with st.expander("Detailed forecast data", expanded=False):
        if not forecast_df.empty:
            display_forecast = forecast_df.copy()
            display_forecast['Date'] = display_forecast['Date'].dt.strftime('%Y-%m-%d')
            display_forecast['Estimation'] = display_forecast['Estimation'].apply(lambda x: f"{x:,.2f}")
            display_forecast['estimate_lower'] = display_forecast['estimate_lower'].apply(lambda x: f"{x:,.2f}")
            display_forecast['estimate_upper'] = display_forecast['estimate_upper'].apply(lambda x: f"{x:,.2f}")

            st.dataframe(display_forecast, use_container_width=True, height=400, hide_index=True)

            st.markdown(f"<p style='color: #64748b; margin-top: 12px; font-size: 13px;'>Showing {len(display_forecast)} forecast days with 80% confidence intervals.</p>", unsafe_allow_html=True)

# ============================================================================
# PAGE 3: AI COPILOT
# ============================================================================

def show_ai_copilot(copilot, df):
    """AI Copilot Chat Interface"""

    st.title("Financial Copilot")
    st.markdown("**Ask direct questions about revenue, customers, and collections.**")
    st.markdown("---")

    # Sample questions
    st.markdown("### Suggested Prompts")
    st.markdown("<p class='section-copy'>Use a prompt below or enter your own.</p>", unsafe_allow_html=True)

    sample_col1, sample_col2, sample_col3 = st.columns(3)

    with sample_col1:
        if st.button("This month revenue", use_container_width=True):
            query = "What is total revenue this month?"
            result = copilot.process_query(query)
            st.session_state.chat_history.append({'query': query, 'result': result})
            st.rerun()

        if st.button("Top customers", use_container_width=True):
            query = "Show me top 10 customers"
            result = copilot.process_query(query)
            st.session_state.chat_history.append({'query': query, 'result': result})
            st.rerun()

        if st.button("Revenue by city", use_container_width=True):
            query = "Show revenue in Riyadh"
            result = copilot.process_query(query)
            st.session_state.chat_history.append({'query': query, 'result': result})
            st.rerun()

    with sample_col2:
        if st.button("Large unpaid invoices", use_container_width=True):
            query = "Which customers have unpaid invoices above 10,000 SAR?"
            result = copilot.process_query(query)
            st.session_state.chat_history.append({'query': query, 'result': result})
            st.rerun()

        if st.button("B2B vs B2C", use_container_width=True):
            query = "Compare revenue between B2B and B2C"
            result = copilot.process_query(query)
            st.session_state.chat_history.append({'query': query, 'result': result})
            st.rerun()

        if st.button("Tax collected", use_container_width=True):
            query = "What's the total tax collected this month?"
            result = copilot.process_query(query)
            st.session_state.chat_history.append({'query': query, 'result': result})
            st.rerun()

    with sample_col3:
        if st.button("Credit memos", use_container_width=True):
            query = "Show me all credit memos"
            result = copilot.process_query(query)
            st.session_state.chat_history.append({'query': query, 'result': result})
            st.rerun()

        if st.button("Monthly trend", use_container_width=True):
            query = "What's the monthly revenue trend?"
            result = copilot.process_query(query)
            st.session_state.chat_history.append({'query': query, 'result': result})
            st.rerun()

        if st.button("Debit notes", use_container_width=True):
            query = "Total debit notes this month"
            result = copilot.process_query(query)
            st.session_state.chat_history.append({'query': query, 'result': result})
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat input
    user_query = st.text_input(
        "Question",
        placeholder="Which customers in Riyadh owe the most?",
        key="copilot_input"
    )

    send_col1, send_col2 = st.columns([1, 5])

    with send_col1:
        if st.button("Send", use_container_width=True) and user_query:
            result = copilot.process_query(user_query)
            st.session_state.chat_history.append({'query': user_query, 'result': result})
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat history
    if st.session_state.chat_history:
        st.markdown("### Recent Conversation")

        for idx, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Last 10 messages
            with st.chat_message("user"):
                st.markdown(f"**{chat['query']}**")

            with st.chat_message("assistant"):
                result = chat['result']

                if result.get('success'):
                    st.markdown(result['explanation'])

                    # Display data table if available
                    if 'data' in result and isinstance(result['data'], pd.DataFrame) and not result['data'].empty:
                        st.dataframe(result['data'].head(15), use_container_width=True, hide_index=True)
                else:
                    st.warning(result.get('message', 'Query failed'))

                    if 'suggestions' in result:
                        st.markdown("**Try one of these:**")
                        for suggestion in result['suggestions'][:5]:
                            st.markdown(f"- {suggestion}")
    else:
        st.info("Ask a question about revenue, customers, or collections to get started.")

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()