"""
app.py  —  Agri-Edge  |  Final Demo Build
==========================================
End-to-end agricultural advisory dashboard for Indian farmers.

Pipeline:
  [1] Sidebar inputs     →  crop · location · language
  [2] Disease detection  →  models/disease_detector.predict_disease()
  [3] Weather fetch      →  utils/weather_fetcher.get_weather()
  [4] Market prices      →  utils/market_prices.get_market_prices()
  [5] AI advisory        →  utils/advisor.generate_advisory()
  [6] Dashboard render   →  this file

Run:
    streamlit run app.py
"""

# ── Page config — must be FIRST Streamlit call ───────────────────────────────
import streamlit as st

st.set_page_config(
    page_title = "Agri-Edge · AI Farm Advisor",
    page_icon  = "🌿",
    layout     = "wide",
    initial_sidebar_state = "expanded",
    menu_items = {"About": "Agri-Edge — AI-Powered Advisory for Indian Farmers 🇮🇳"},
)

# ── Standard library ──────────────────────────────────────────────────────────
import os, sys, time
from datetime import datetime

# ── Project imports ───────────────────────────────────────────────────────────
try:
    from models.disease_detector import predict_disease, get_disease_info
    from utils.weather_fetcher   import get_weather
    from utils.market_prices     import get_market_prices
    from utils.advisor           import (
        generate_advisory, compute_risk_score,
        translate_advisory, SUPPORTED_LANGUAGES,
        DEFAULT_LANGUAGE, get_language_config, LANGUAGE_CONFIG,
    )
    from config.settings import OPENWEATHER_API_KEY, GROQ_API_KEY
    _OK = True
except ImportError as _e:
    _OK = False; _ERR = str(_e)

LC = LANGUAGE_CONFIG if _OK else {}  # short alias


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init():
    defs = {
        "disease_result"  : None,
        "weather_data"    : None,
        "market_data"     : None,
        "advisory_result" : None,
        "last_img"        : None,
        "last_loc"        : None,
        "last_crop"       : None,
        "s_disease"       : "idle",   # idle | running | ok | error
        "s_weather"       : "idle",
        "s_advisory"      : "idle",
        "err_disease"     : None,
        "err_weather"     : None,
        "pipeline_run"    : False,
        "language"        : "English",
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  CSS  —  Premium dark-earth theme
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

/* ── Tokens ──────────────────────────────────────────────────────────── */
:root {
    --bg      : #0B150B;   --bg2     : #111B11;   --bg3     : #172017;
    --card    : #1A2A1A;   --card2   : #203020;
    --b       : #273D27;   --b2      : #334F33;   --b3      : #3E6040;
    --green   : #4ADE80;   --green2  : #22C55E;   --green3  : #16A34A;
    --gold    : #F0C84E;   --gold2   : #D4A827;
    --red     : #EF4444;   --red2    : #DC2626;   --redl    : #FCA5A5;
    --orange  : #F97316;   --orangel : #FB923C;
    --yellow  : #EAB308;   --yellowl : #FCD34D;
    --blue    : #3B82F6;   --bluel   : #93C5FD;
    --txt     : #E9F5E9;   --txt2    : #8FB88F;   --txt3    : #506850;
    --serif   : 'DM Serif Display', Georgia, serif;
    --sans    : 'DM Sans', system-ui, sans-serif;
    --r       : 12px;      --rl      : 18px;      --rxl     : 24px;
    --shadow  : 0 4px 24px rgba(0,0,0,.45);
    --glow    : 0 0 28px rgba(74,222,128,.12);
    --t       : all .22s cubic-bezier(.4,0,.2,1);
}

/* ── Reset ───────────────────────────────────────────────────────────── */
.stApp { background:var(--bg); font-family:var(--sans); color:var(--txt); }
#MainMenu,footer,header,.stDeployButton { display:none !important; visibility:hidden !important; }
hr { border-color:var(--b) !important; margin:1.6rem 0 !important; }
* { box-sizing:border-box; }

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg2); }
::-webkit-scrollbar-thumb { background:var(--b2); border-radius:3px; }

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background:var(--bg2) !important;
    border-right:1px solid var(--b) !important;
}
[data-testid="stSidebar"] * { font-family:var(--sans) !important; }
[data-testid="stSidebar"] label {
    color:var(--txt2) !important; font-size:.71rem !important;
    font-weight:700 !important; text-transform:uppercase !important;
    letter-spacing:.09em !important;
}
[data-testid="stSidebar"] p { color:var(--txt2) !important; }

/* ── Inputs ──────────────────────────────────────────────────────────── */
.stTextInput input {
    background:var(--card) !important; border:1px solid var(--b) !important;
    border-radius:9px !important; color:var(--txt) !important;
    font-size:.88rem !important; padding:10px 14px !important;
    transition:var(--t) !important;
}
.stTextInput input:focus {
    border-color:var(--green) !important;
    box-shadow:0 0 0 3px rgba(74,222,128,.14) !important;
}
[data-baseweb="select"] > div {
    background:var(--card) !important; border-color:var(--b) !important;
    border-radius:9px !important;
}

/* ── File uploader ───────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background:var(--card) !important;
    border:2px dashed var(--b2) !important;
    border-radius:var(--rl) !important; transition:var(--t) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color:var(--green) !important;
    background:var(--card2) !important;
    box-shadow:var(--glow) !important;
}
[data-testid="stFileUploader"] p { color:var(--txt2) !important; }

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton>button {
    font-family:var(--sans) !important; font-weight:700 !important;
    border-radius:11px !important; transition:var(--t) !important;
    letter-spacing:.025em !important; font-size:.88rem !important;
}
.stButton>button[kind="primary"] {
    background:linear-gradient(135deg,#22C55E,#16A34A) !important;
    border:none !important; color:#fff !important;
    box-shadow:0 4px 18px rgba(34,197,94,.38) !important;
    padding:12px 28px !important;
}
.stButton>button[kind="primary"]:hover {
    background:linear-gradient(135deg,#4ADE80,#22C55E) !important;
    transform:translateY(-2px) !important;
    box-shadow:0 8px 28px rgba(74,222,128,.5) !important;
}
.stButton>button[kind="primary"]:active { transform:translateY(0) !important; }
.stButton>button[kind="secondary"] {
    background:var(--card) !important; border:1px solid var(--b2) !important;
    color:var(--txt) !important;
}
.stButton>button[kind="secondary"]:hover {
    border-color:var(--green) !important; color:var(--green) !important;
}

/* ── Spinner ─────────────────────────────────────────────────────────── */
.stSpinner>div { border-top-color:var(--green) !important; }
[data-testid="stSpinner"] p { color:var(--txt2) !important; font-size:.84rem !important; }

/* ── Alerts ──────────────────────────────────────────────────────────── */
.stAlert { border-radius:var(--r) !important; font-size:.84rem !important; font-family:var(--sans) !important; }

/* ── Progress bar ────────────────────────────────────────────────────── */
.stProgress>div>div { background:linear-gradient(90deg,var(--green2),var(--green)) !important; border-radius:8px !important; }
.stProgress>div { background:var(--b) !important; border-radius:8px !important; }

/* ── Images ──────────────────────────────────────────────────────────── */
[data-testid="stImage"] img { border-radius:var(--r); border:1px solid var(--b2); }

/* ════════════════════════════════════════════════════════════════════════════
   DESIGN TOKENS  —  custom HTML components
   ════════════════════════════════════════════════════════════════════════════ */

/* ─── Hero ────────────────────────────────────────────────────────────── */
.hero {
    background:linear-gradient(135deg, #111B11 0%, #172017 50%, #0B150B 100%);
    border:1px solid var(--b); border-radius:var(--rxl);
    padding:28px 36px 24px; margin-bottom:20px;
    position:relative; overflow:hidden;
}
.hero::before {
    content:''; position:absolute; top:-80px; right:-80px;
    width:280px; height:280px; border-radius:50%; pointer-events:none;
    background:radial-gradient(circle, rgba(74,222,128,.07) 0%, transparent 65%);
}
.hero::after {
    content:''; position:absolute; bottom:-40px; left:20%;
    width:160px; height:160px; border-radius:50%; pointer-events:none;
    background:radial-gradient(circle, rgba(240,200,78,.04) 0%, transparent 65%);
}
.hero-eyebrow {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(74,222,128,.09); border:1px solid rgba(74,222,128,.25);
    color:var(--green); font-size:.67rem; font-weight:800;
    text-transform:uppercase; letter-spacing:.13em;
    padding:4px 12px; border-radius:20px; margin-bottom:12px;
}
.hero-h1 {
    font-family:var(--serif); font-size:2.3rem; font-weight:400;
    color:var(--txt); margin:0 0 6px; line-height:1.12; letter-spacing:-.02em;
}
.hero-sub { color:var(--txt2); font-size:.88rem; margin:0; line-height:1.5; }
.hero-pill {
    display:inline-flex; align-items:center; gap:4px;
    background:rgba(255,255,255,.06); border:1px solid var(--b2);
    color:var(--green); font-weight:700; font-size:.82rem;
    padding:3px 10px; border-radius:20px;
}
.hero-divider {
    height:1px; background:linear-gradient(90deg,transparent,var(--b2),transparent);
    margin:16px 0 12px;
}
.hero-stats {
    display:flex; gap:24px; flex-wrap:wrap;
}
.hero-stat {
    display:flex; flex-direction:column;
}
.hero-stat-val {
    font-size:1.4rem; font-weight:700; color:var(--txt);
    font-variant-numeric:tabular-nums; line-height:1;
}
.hero-stat-lbl { font-size:.66rem; color:var(--txt3); text-transform:uppercase; letter-spacing:.07em; margin-top:2px; }

/* ─── Pipeline bar ────────────────────────────────────────────────────── */
.pipe-wrap {
    display:flex; align-items:stretch; background:var(--bg2);
    border:1px solid var(--b); border-radius:var(--r);
    padding:14px 20px; margin-bottom:20px; gap:0;
    overflow:hidden;
}
.pipe-step {
    flex:1; display:flex; flex-direction:column; align-items:center;
    gap:6px; position:relative; padding:0 8px;
}
.pipe-step+.pipe-step::before {
    content:''; position:absolute; left:0; top:12px;
    width:1px; height:20px; background:var(--b2);
}
.pipe-dot {
    width:28px; height:28px; border-radius:50%; border:2px solid var(--b2);
    background:var(--bg3); display:flex; align-items:center; justify-content:center;
    font-size:.82rem; transition:var(--t);
}
.pipe-lbl { font-size:.64rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; color:var(--txt3); text-align:center; }
.ps-idle  .pipe-dot { border-color:var(--b2); }
.ps-ok    .pipe-dot { border-color:var(--green2); background:rgba(34,197,94,.15); }
.ps-run   .pipe-dot { border-color:var(--yellow);  background:rgba(234,179,8,.12); animation:spin-border 1.2s linear infinite; }
.ps-err   .pipe-dot { border-color:var(--red);     background:rgba(239,68,68,.12); }
.ps-ok    .pipe-lbl { color:var(--green); }
.ps-err   .pipe-lbl { color:var(--redl); }
@keyframes spin-border { to { transform:rotate(360deg); } }

/* ─── Section cards ───────────────────────────────────────────────────── */
.panel {
    background:var(--card); border:1px solid var(--b);
    border-radius:var(--rl); padding:20px 22px;
    transition:var(--t); height:100%;
}
.panel:hover { border-color:var(--b2); box-shadow:var(--glow); }
.panel-hdr {
    font-size:.67rem; font-weight:800; text-transform:uppercase;
    letter-spacing:.11em; color:var(--txt2);
    margin:0 0 16px; padding-bottom:11px;
    border-bottom:1px solid var(--b);
    display:flex; align-items:center; gap:8px;
}
.panel-hdr-badge {
    margin-left:auto; font-size:.6rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.07em;
    padding:2px 8px; border-radius:20px;
}
.badge-ok   { background:rgba(34,197,94,.12); color:var(--green2); border:1px solid rgba(34,197,94,.25); }
.badge-warn { background:rgba(249,115,22,.12); color:var(--orangel); border:1px solid rgba(249,115,22,.25); }
.badge-err  { background:rgba(239,68,68,.1);  color:var(--redl);   border:1px solid rgba(239,68,68,.25); }
.badge-demo { background:rgba(234,179,8,.1);  color:var(--yellowl); border:1px solid rgba(234,179,8,.22); }

/* ─── Empty states ────────────────────────────────────────────────────── */
.empty {
    border:1.5px dashed var(--b2); border-radius:var(--r);
    padding:28px 20px; text-align:center;
    color:var(--txt3); font-size:.8rem; line-height:1.7;
}
.empty-ico { font-size:2rem; display:block; margin-bottom:10px; opacity:.4; }
.empty strong { color:var(--txt2); }

/* ─── Disease result ──────────────────────────────────────────────────── */
.dx-card {
    border-radius:var(--r); padding:14px 16px; margin-top:12px;
    display:flex; align-items:flex-start; gap:12px;
}
.dx-healthy { background:rgba(34,197,94,.08); border:1px solid rgba(34,197,94,.28); }
.dx-sick    { background:rgba(239,68,68,.07); border:1px solid rgba(239,68,68,.26); }
.dx-critical{ background:rgba(239,68,68,.12); border:1.5px solid rgba(239,68,68,.42); }
.dx-icon    { font-size:1.6rem; line-height:1; flex-shrink:0; margin-top:2px; }
.dx-label   { font-size:1rem; font-weight:700; margin:0 0 2px; }
.dx-h-lbl   { color:var(--green); }
.dx-s-lbl   { color:var(--redl); }
.dx-sub     { font-size:.72rem; color:var(--txt2); margin:0; }
.dx-urgency {
    display:inline-block; margin-top:6px; font-size:.67rem; font-weight:800;
    text-transform:uppercase; letter-spacing:.08em;
    padding:2px 8px; border-radius:20px;
    background:rgba(239,68,68,.15); color:var(--redl); border:1px solid rgba(239,68,68,.3);
}

/* ─── Confidence bars ─────────────────────────────────────────────────── */
.cbar { margin-top:14px; }
.cbar-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; }
.cbar-lbl { font-size:.67rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:var(--txt2); }
.cbar-pct { font-size:.82rem; font-weight:700; font-variant-numeric:tabular-nums; }
.cbar-pct-hi   { color:var(--green); }
.cbar-pct-mid  { color:var(--yellowl); }
.cbar-pct-lo   { color:var(--redl); }
.cbar-track { height:8px; background:var(--b); border-radius:10px; overflow:hidden; }
.cbar-fill  { height:100%; border-radius:10px; transition:width .9s cubic-bezier(.4,0,.2,1); }
.cf-hi  { background:linear-gradient(90deg,#16A34A,#4ADE80); }
.cf-mid { background:linear-gradient(90deg,#CA8A04,#FCD34D); }
.cf-lo  { background:linear-gradient(90deg,#DC2626,#FCA5A5); }

/* ─── Top-3 ───────────────────────────────────────────────────────────── */
.t3-section { margin-top:14px; border-top:1px solid var(--b); padding-top:12px; }
.t3-hdr { font-size:.64rem; text-transform:uppercase; letter-spacing:.08em; color:var(--txt3); font-weight:700; margin-bottom:8px; }
.t3-row {
    display:flex; align-items:center; gap:9px;
    padding:5px 0; border-bottom:1px solid rgba(255,255,255,.04);
    font-size:.75rem;
}
.t3-row:last-child { border:none; }
.t3-rank { font-weight:800; color:var(--txt3); width:16px; flex-shrink:0; }
.t3-lbl  { flex:1; color:var(--txt); font-weight:500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.t3-bar  { width:52px; height:4px; background:var(--b); border-radius:4px; overflow:hidden; flex-shrink:0; }
.t3-fill { height:100%; background:var(--green2); border-radius:4px; }
.t3-pct  { width:38px; text-align:right; color:var(--txt2); font-variant-numeric:tabular-nums; font-weight:700; flex-shrink:0; }

/* ─── Low-confidence alert ────────────────────────────────────────────── */
.warn-box {
    display:flex; align-items:center; gap:8px;
    margin-top:10px; padding:8px 12px;
    background:rgba(234,179,8,.08); border:1px solid rgba(234,179,8,.22);
    border-radius:8px; font-size:.75rem; color:var(--yellowl);
}

/* ─── Weather tiles ───────────────────────────────────────────────────── */
.wx-grid { display:grid; grid-template-columns:1fr 1fr; gap:9px; margin:6px 0 12px; }
.wx-tile {
    background:var(--bg2); border:1px solid var(--b);
    border-radius:11px; padding:11px 13px; transition:var(--t);
}
.wx-tile:hover { border-color:var(--b2); }
.wx-ico { font-size:.95rem; display:block; margin-bottom:3px; }
.wx-val { font-size:1.15rem; font-weight:700; color:var(--txt); display:block; font-variant-numeric:tabular-nums; }
.wx-lbl { font-size:.6rem; text-transform:uppercase; letter-spacing:.08em; color:var(--txt3); font-weight:700; display:block; margin-top:1px; }

/* ─── Farming flags ───────────────────────────────────────────────────── */
.flags-grid { display:grid; grid-template-columns:1fr 1fr; gap:7px; margin-top:6px; }
.flag {
    display:flex; align-items:center; gap:6px;
    font-size:.73rem; font-weight:600;
    padding:6px 10px; border-radius:8px;
    border:1px solid var(--b); background:var(--bg2);
    transition:var(--t);
}
.flag-on  { color:var(--green); border-color:rgba(74,222,128,.25); background:rgba(74,222,128,.06); }
.flag-off { color:var(--txt3); }
.flag-warn{ color:var(--orangel); border-color:rgba(249,115,22,.25); background:rgba(249,115,22,.06); }

/* ─── Forecast rows ───────────────────────────────────────────────────── */
.fc-hdr-row, .fc-row {
    display:grid; grid-template-columns:40px 46px 1fr 60px 80px;
    align-items:center; gap:6px; font-size:.75rem; padding:6px 0;
    border-bottom:1px solid rgba(255,255,255,.04);
}
.fc-row:last-child { border:none; }
.fc-hdr-row { font-size:.6rem; text-transform:uppercase; letter-spacing:.07em; color:var(--txt3); font-weight:700; padding-bottom:7px; }
.fc-day  { font-weight:700; color:var(--txt2); }
.fc-date { color:var(--txt3); font-size:.7rem; }
.fc-cond { color:var(--txt); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.fc-rain { color:var(--bluel); text-align:right; font-variant-numeric:tabular-nums; }
.fc-temp { color:var(--txt2); text-align:right; font-variant-numeric:tabular-nums; font-size:.72rem; }
.fc-today { background:rgba(74,222,128,.04); border-radius:6px; padding:6px 4px; }

/* ─── Market card ─────────────────────────────────────────────────────── */
.mkt-main  { font-family:var(--serif); font-size:2.1rem; color:var(--gold); font-weight:400; line-height:1; margin:4px 0 2px; }
.mkt-unit  { font-size:.7rem; color:var(--txt3); margin-bottom:14px; }
.mkt-range { display:flex; gap:9px; margin-bottom:12px; }
.mkt-box   { flex:1; background:var(--bg2); border:1px solid var(--b); border-radius:9px; padding:9px 10px; text-align:center; }
.mkt-bv    { display:block; font-weight:700; font-size:.95rem; }
.mkt-bl    { display:block; font-size:.6rem; text-transform:uppercase; letter-spacing:.08em; color:var(--txt3); margin-top:2px; }
.mkt-trend {
    display:inline-flex; align-items:center; gap:5px;
    font-size:.69rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em;
    padding:4px 11px; border-radius:20px;
}
.t-rising  { background:rgba(34,197,94,.12); color:var(--green2); border:1px solid rgba(34,197,94,.28); }
.t-falling { background:rgba(239,68,68,.1);  color:var(--redl);   border:1px solid rgba(239,68,68,.28); }
.t-stable  { background:rgba(240,200,78,.1); color:var(--gold);   border:1px solid rgba(240,200,78,.25); }
.msp-row   {
    display:flex; justify-content:space-between; align-items:center;
    margin-top:11px; padding-top:10px; border-top:1px solid var(--b);
    font-size:.76rem;
}
.msp-lbl { color:var(--txt3); }
.msp-val { font-weight:700; }
.msp-ok   { color:var(--gold); }
.msp-warn { color:var(--redl); }
.msp-alert {
    margin-top:8px; padding:8px 12px;
    background:rgba(239,68,68,.08); border:1px solid rgba(239,68,68,.25);
    border-radius:8px; font-size:.75rem; color:var(--redl);
    display:flex; align-items:center; gap:7px;
}
.mkt-footer { font-size:.68rem; color:var(--txt3); margin-top:10px; line-height:1.6; }

/* ─── Risk block ──────────────────────────────────────────────────────── */
.risk-wrap {
    border-radius:var(--rl); padding:20px 22px;
    border:1px solid; position:relative; overflow:hidden; margin-bottom:16px;
}
.risk-wrap::after {
    content:''; position:absolute; top:-40px; right:-40px;
    width:140px; height:140px; border-radius:50%; pointer-events:none;
}
.rk-none    { background:rgba(34,197,94,.05);  border-color:rgba(34,197,94,.2);  }
.rk-low     { background:rgba(59,130,246,.05); border-color:rgba(59,130,246,.2); }
.rk-moderate{ background:rgba(234,179,8,.05);  border-color:rgba(234,179,8,.22); }
.rk-high    { background:rgba(249,115,22,.07); border-color:rgba(249,115,22,.28);}
.rk-critical{ background:rgba(239,68,68,.09);  border-color:rgba(239,68,68,.4);  }
.rk-none::after    { background:radial-gradient(circle,rgba(74,222,128,.08) 0%,transparent 70%); }
.rk-moderate::after{ background:radial-gradient(circle,rgba(234,179,8,.07) 0%,transparent 70%); }
.rk-high::after    { background:radial-gradient(circle,rgba(249,115,22,.08) 0%,transparent 70%); }
.rk-critical::after{ background:radial-gradient(circle,rgba(239,68,68,.1)  0%,transparent 70%); }
.risk-lbl { font-family:var(--serif); font-size:1.65rem; font-weight:400; margin:0 0 2px; }
.rk-none     .risk-lbl { color:var(--green); }
.rk-low      .risk-lbl { color:#60A5FA; }
.rk-moderate .risk-lbl { color:var(--gold); }
.rk-high     .risk-lbl { color:var(--orangel); }
.rk-critical .risk-lbl { color:var(--redl); }
.risk-score-txt { font-size:.77rem; color:var(--txt2); margin-bottom:8px; }
.risk-track { height:8px; background:rgba(255,255,255,.07); border-radius:10px; overflow:hidden; margin-bottom:12px; }
.risk-fill  { height:100%; border-radius:10px; transition:width 1s cubic-bezier(.4,0,.2,1); }
.rk-none     .risk-fill { background:linear-gradient(90deg,#15803D,#4ADE80); }
.rk-low      .risk-fill { background:linear-gradient(90deg,#1D4ED8,#60A5FA); }
.rk-moderate .risk-fill { background:linear-gradient(90deg,#A16207,#FCD34D); }
.rk-high     .risk-fill { background:linear-gradient(90deg,#C2410C,#FB923C); }
.rk-critical .risk-fill { background:linear-gradient(90deg,#B91C1C,#FCA5A5); }
.risk-reason {
    display:flex; align-items:flex-start; gap:7px;
    font-size:.74rem; color:var(--txt2); padding:5px 0;
    border-bottom:1px solid rgba(255,255,255,.05); line-height:1.5;
}
.risk-reason:last-child { border:none; }
.risk-reason-dot { flex-shrink:0; margin-top:3px; font-size:.55rem; }

/* ─── Component score rows ────────────────────────────────────────────── */
.comp-section { margin-top:14px; }
.comp-hdr { font-size:.62rem; text-transform:uppercase; letter-spacing:.09em; color:var(--txt3); font-weight:700; margin-bottom:8px; }
.comp-row { display:flex; align-items:center; gap:9px; padding:4px 0; font-size:.72rem; }
.comp-name { width:130px; color:var(--txt2); flex-shrink:0; text-transform:capitalize; line-height:1.3; }
.comp-track { flex:1; height:5px; background:var(--b); border-radius:5px; overflow:hidden; }
.comp-fill  { height:100%; border-radius:5px; background:linear-gradient(90deg,var(--green2),var(--green)); }
.comp-pts   { width:30px; text-align:right; color:var(--txt3); font-weight:800; font-variant-numeric:tabular-nums; font-size:.68rem; }

/* ─── Advisory cards ──────────────────────────────────────────────────── */
.adv-card {
    background:var(--card); border:1px solid var(--b);
    border-radius:var(--r); padding:16px 18px; margin-bottom:10px;
    transition:var(--t); position:relative; overflow:hidden;
}
.adv-card:hover { border-color:var(--b2); box-shadow:0 2px 18px rgba(0,0,0,.3); }
.adv-card-disease { border-left:3px solid var(--redl) !important; }
.adv-card-irrigation { border-left:3px solid var(--bluel) !important; }
.adv-card-inputs { border-left:3px solid var(--green) !important; }
.adv-card-market { border-left:3px solid var(--gold) !important; }
.adv-hdr {
    font-weight:700; font-size:.88rem; color:var(--txt);
    margin:0 0 8px; display:flex; align-items:center; gap:8px;
}
.adv-body { font-size:.83rem; color:var(--txt2); line-height:1.65; margin:0; }
.urgency-tag {
    display:inline-flex; align-items:center; gap:5px;
    margin-top:9px; padding:3px 10px; border-radius:20px;
    font-size:.67rem; font-weight:800; text-transform:uppercase; letter-spacing:.07em;
    background:rgba(249,115,22,.13); color:var(--orangel); border:1px solid rgba(249,115,22,.28);
}
.urgency-critical {
    background:rgba(239,68,68,.13); color:var(--redl); border-color:rgba(239,68,68,.3);
    animation:urgent-pulse 2s ease-in-out infinite;
}
@keyframes urgent-pulse { 0%,100%{opacity:1} 50%{opacity:.7} }

/* ─── Action plan ─────────────────────────────────────────────────────── */
.plan-card {
    background:linear-gradient(135deg,rgba(74,222,128,.04),rgba(34,197,94,.02));
    border:1px solid rgba(74,222,128,.18);
    border-radius:var(--r); padding:16px 18px; margin-bottom:10px;
}
.plan-hdr { font-weight:700; font-size:.88rem; color:var(--green); margin:0 0 12px; display:flex; align-items:center; gap:8px; }
.plan-step {
    display:flex; align-items:flex-start; gap:12px;
    padding:8px 0; border-bottom:1px solid rgba(255,255,255,.05);
    font-size:.82rem; color:var(--txt2); line-height:1.55;
}
.plan-step:last-child { border:none; padding-bottom:0; }
.plan-num {
    width:24px; height:24px; border-radius:50%; flex-shrink:0; margin-top:1px;
    background:rgba(74,222,128,.13); border:1px solid rgba(74,222,128,.28);
    color:var(--green); font-size:.68rem; font-weight:800;
    display:flex; align-items:center; justify-content:center;
}

/* ─── Summary callout ─────────────────────────────────────────────────── */
.summary-box {
    background:linear-gradient(135deg,rgba(74,222,128,.07),rgba(22,163,74,.03));
    border:1px solid rgba(74,222,128,.22); border-left:3px solid var(--green2);
    border-radius:var(--r); padding:14px 18px; margin-bottom:16px;
    font-size:.87rem; color:var(--txt); line-height:1.6;
    display:flex; align-items:flex-start; gap:10px;
}
.summary-icon { font-size:1.1rem; flex-shrink:0; margin-top:1px; }

/* ─── Language / source tags ──────────────────────────────────────────── */
.lang-badge {
    display:inline-flex; align-items:center; gap:5px;
    font-size:.67rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em;
    color:var(--green); background:rgba(74,222,128,.09);
    border:1px solid rgba(74,222,128,.22);
    padding:3px 9px; border-radius:20px; margin-bottom:10px;
}
.src-tag { font-size:.64rem; color:var(--txt3); text-align:right; margin-top:14px; font-style:italic; line-height:1.5; }

/* ─── Section divider labels ──────────────────────────────────────────── */
.sec-divider {
    display:flex; align-items:center; gap:12px;
    margin:24px 0 18px; font-size:.67rem; font-weight:800;
    text-transform:uppercase; letter-spacing:.11em; color:var(--txt3);
}
.sec-divider::before,.sec-divider::after {
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,transparent,var(--b),transparent);
}

/* ─── Status strip (success / error) ─────────────────────────────────── */
.status-ok {
    display:flex; align-items:center; gap:9px;
    background:rgba(34,197,94,.07); border:1px solid rgba(34,197,94,.22);
    border-radius:9px; padding:10px 14px; margin-bottom:14px;
    font-size:.82rem; color:var(--green);
}
.status-err {
    display:flex; align-items:center; gap:9px;
    background:rgba(239,68,68,.07); border:1px solid rgba(239,68,68,.22);
    border-radius:9px; padding:10px 14px; margin-bottom:14px;
    font-size:.82rem; color:var(--redl);
}
.status-warn {
    display:flex; align-items:center; gap:9px;
    background:rgba(234,179,8,.07); border:1px solid rgba(234,179,8,.22);
    border-radius:9px; padding:10px 14px; margin-bottom:14px;
    font-size:.82rem; color:var(--yellowl);
}

/* ─── CTA section ─────────────────────────────────────────────────────── */
.cta-wrap {
    background:linear-gradient(135deg,rgba(34,197,94,.05),rgba(22,163,74,.02));
    border:1px solid rgba(74,222,128,.16);
    border-radius:var(--rl); padding:24px 28px; margin-bottom:20px;
    display:flex; align-items:center; gap:24px; flex-wrap:wrap;
}
.cta-text h3 { font-family:var(--serif); font-size:1.3rem; font-weight:400; color:var(--txt); margin:0 0 3px; }
.cta-text p  { font-size:.82rem; color:var(--txt2); margin:0; line-height:1.5; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _h(html: str):
    st.markdown(html, unsafe_allow_html=True)


def _divider(label: str):
    _h(f'<div class="sec-divider">{label}</div>')


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def _run_disease(f) -> bool:
    st.session_state.s_disease = "running"
    try:
        r = predict_disease(f)
        if r.get("error"): raise ValueError(r["error"])
        st.session_state.disease_result = r
        st.session_state.err_disease    = None
        st.session_state.s_disease      = "ok"
        return True
    except Exception as e:
        st.session_state.disease_result = None
        st.session_state.err_disease    = str(e)
        st.session_state.s_disease      = "error"
        return False


def _run_weather(loc: str) -> bool:
    if not loc:
        st.session_state.weather_data = None
        st.session_state.s_weather    = "idle"
        return False
    st.session_state.s_weather = "running"
    try:
        d = get_weather(loc, api_key=OPENWEATHER_API_KEY)
        if d.get("error"): raise ValueError(d["error"])
        st.session_state.weather_data = d
        st.session_state.err_weather  = None
        st.session_state.s_weather    = "ok"
        return True
    except Exception as e:
        st.session_state.weather_data = None
        st.session_state.err_weather  = str(e)
        st.session_state.s_weather    = "error"
        return False


def _run_market(crop: str) -> dict:
    try:
        d = get_market_prices(crop)
        st.session_state.market_data = d
        return d
    except Exception:
        fb = {"modal_price":"—","min_price":"—","max_price":"—",
              "trend":"stable","market":"Local Mandi","date":"—","msp":None}
        st.session_state.market_data = fb
        return fb


def _run_advisory(crop: str, language: str) -> bool:
    st.session_state.s_advisory = "running"
    try:
        r = generate_advisory({
            "crop"         : crop,
            "disease_label": (st.session_state.disease_result or {}).get("label", ""),
            "confidence"   : (st.session_state.disease_result or {}).get("confidence", 0.0),
            "weather"      : st.session_state.weather_data,
            "market"       : st.session_state.market_data,
        }, api_key=GROQ_API_KEY, language=language)
        st.session_state.advisory_result = r
        st.session_state.s_advisory      = "ok"
        return True
    except Exception as e:
        st.session_state.advisory_result = None
        st.session_state.s_advisory      = "error"
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  HTML COMPONENT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _pipe_bar():
    steps = [
        ("s_disease",  "🔬", "Disease"),
        ("s_weather",  "🌦", "Weather"),
        ("market_data","📊", "Market"),
        ("s_advisory", "🤖", "Advisory"),
    ]
    icons = {"idle":"○","running":"◉","ok":"✓","error":"✕"}

    def _status(key):
        if key == "market_data":
            return "ok" if st.session_state.market_data else "idle"
        return st.session_state.get(key, "idle")

    items = ""
    for key, emoji, label in steps:
        s   = _status(key)
        cls = f"pipe-step ps-{s}"
        items += f'<div class="{cls}"><div class="pipe-dot">{icons[s]}</div><div class="pipe-lbl">{emoji} {label}</div></div>'

    _h(f'<div class="pipe-wrap">{items}</div>')


def _disease_panel_html(r: dict) -> str:
    label      = r.get("label", "Unknown")
    conf       = float(r.get("confidence", 0))
    top3       = r.get("top3", [])
    source     = r.get("source", "model")
    is_healthy = "healthy" in label.lower()
    conf_pct   = conf * 100
    sev        = (r.get("disease_info") or {}).get("severity", "unknown")

    # Card style
    if is_healthy:
        card_cls, lbl_cls, icon = "dx-healthy", "dx-h-lbl", "✅"
    elif sev in ("severe", "critical"):
        card_cls, lbl_cls, icon = "dx-critical", "dx-s-lbl", "🚨"
    else:
        card_cls, lbl_cls, icon = "dx-sick", "dx-s-lbl", "⚠️"

    # Confidence bar
    if conf_pct >= 75: bar_cls, pct_cls = "cf-hi", "cbar-pct-hi"
    elif conf_pct >= 50: bar_cls, pct_cls = "cf-mid", "cbar-pct-mid"
    else: bar_cls, pct_cls = "cf-lo", "cbar-pct-lo"

    urgency_html = ""
    if not is_healthy:
        urg = (r.get("disease_info") or {}).get("treatment", [""])
        if sev in ("severe", "critical"):
            urgency_html = '<span class="dx-urgency">⏱ Act within 24 hours</span>'
        elif sev == "moderate":
            urgency_html = '<span class="dx-urgency">⏱ Treat within 2–3 days</span>'

    demo = ""
    if source == "mock":
        demo = '<div class="panel-hdr-badge badge-demo" style="display:inline-block;margin-bottom:8px">🎭 Demo predictions</div>'

    low_conf = ""
    if conf < 0.60:
        low_conf = '<div class="warn-box">⚠️ Low confidence — use a clearer, well-lit close-up photo of the leaf</div>'

    t3_rows = "".join(
        f'<div class="t3-row">'
        f'<span class="t3-rank">#{i}</span>'
        f'<span class="t3-lbl">{p["label"]}</span>'
        f'<div class="t3-bar"><div class="t3-fill" style="width:{int(p["confidence"]*100)}%"></div></div>'
        f'<span class="t3-pct">{p["confidence"]*100:.1f}%</span>'
        f'</div>'
        for i, p in enumerate(top3[:3], 1)
    )

    return f"""
{demo}
<div class="dx-card {card_cls}">
  <div class="dx-icon">{icon}</div>
  <div>
    <p class="dx-label {lbl_cls}">{label}</p>
    <p class="dx-sub">MobileNetV2 · Transfer Learning</p>
    {urgency_html}
  </div>
</div>
<div class="cbar">
  <div class="cbar-row">
    <span class="cbar-lbl">Model Confidence</span>
    <span class="cbar-pct {pct_cls}">{conf_pct:.1f}%</span>
  </div>
  <div class="cbar-track"><div class="cbar-fill {bar_cls}" style="width:{conf_pct}%"></div></div>
</div>
{low_conf}
<div class="t3-section">
  <div class="t3-hdr">All predictions</div>
  {t3_rows}
</div>"""


def _weather_tiles(cur: dict) -> str:
    temp = cur.get("temp_c", "—")
    # Colour-code temperature
    try:
        t = float(temp)
        t_color = "var(--redl)" if t >= 35 else ("var(--yellowl)" if t >= 30 else "var(--bluel)" if t < 15 else "var(--txt)")
    except: t_color = "var(--txt)"

    hum = cur.get("humidity_pct", "—")
    try:
        h = float(hum)
        h_color = "var(--orangel)" if h >= 80 else "var(--txt)"
    except: h_color = "var(--txt)"

    return f"""
<div class="wx-grid">
  <div class="wx-tile"><span class="wx-ico">🌡</span><span class="wx-val" style="color:{t_color}">{temp}°C</span><span class="wx-lbl">Temperature</span></div>
  <div class="wx-tile"><span class="wx-ico">💧</span><span class="wx-val" style="color:{h_color}">{hum}%</span><span class="wx-lbl">Humidity</span></div>
  <div class="wx-tile"><span class="wx-ico">💨</span><span class="wx-val">{cur.get('wind_speed_kmh','—')} km/h</span><span class="wx-lbl">Wind Speed</span></div>
  <div class="wx-tile"><span class="wx-ico">🌧</span><span class="wx-val">{cur.get('rainfall_1h_mm',0)} mm</span><span class="wx-lbl">Rainfall 1h</span></div>
</div>"""


def _flags_html(adv: dict) -> str:
    flags = [
        ("irrigation_needed",          "🚿", "Irrigation Needed",   "warn"),
        ("heat_stress_risk",           "🌡", "Heat Stress",         "warn"),
        ("frost_risk",                 "🥶", "Frost Risk",          "warn"),
        ("rain_expected_24h",          "🌧", "Rain in 24h",         "on"),
        ("spray_conditions_ok",        "✅", "Good for Spraying",   "on"),
        ("high_humidity_disease_risk", "🍄", "Fungal Disease Risk", "warn"),
    ]
    items = ""
    for key, ico, lbl, active_cls in flags:
        on  = adv.get(key, False)
        cls = f"flag-{active_cls}" if on else "flag-off"
        dot = "●" if on else "○"
        items += f'<div class="flag {cls}">{dot} {ico} {lbl}</div>'
    return f'<div class="flags-grid">{items}</div>'


def _forecast_html(fc: list) -> str:
    header = '<div class="fc-hdr-row"><span class="fc-day">Day</span><span class="fc-date">Date</span><span class="fc-cond">Condition</span><span class="fc-rain">Rain</span><span class="fc-temp">Temp Range</span></div>'
    rows = ""
    for i, d in enumerate(fc[:5]):
        extra = ' fc-today' if i == 0 else ''
        prob  = int(d.get("rain_probability", 0) * 100)
        rain_color = "var(--bluel)" if d.get("rainfall_mm", 0) > 2 else "var(--txt2)"
        rows += (
            f'<div class="fc-row{extra}">'
            f'<span class="fc-day">{d["day_name"][:3]}</span>'
            f'<span class="fc-date">{d["date"][5:]}</span>'
            f'<span class="fc-cond">{d["condition"]}</span>'
            f'<span class="fc-rain" style="color:{rain_color}">💧{d["rainfall_mm"]}mm <span style="font-size:.65rem;color:var(--txt3)">({prob}%)</span></span>'
            f'<span class="fc-temp">{d["temp_min_c"]}° – {d["temp_max_c"]}°C</span>'
            f'</div>'
        )
    return header + rows


def _risk_html(level: str, score: int, reasons: list, components: dict, ui: dict) -> str:
    cls_map = {"NONE":"rk-none","LOW":"rk-low","MODERATE":"rk-moderate","HIGH":"rk-high","CRITICAL":"rk-critical"}
    emo_map = {"NONE":"🟢","LOW":"🔵","MODERATE":"🟡","HIGH":"🟠","CRITICAL":"🔴"}
    cls = cls_map.get(level, "rk-none")
    emo = emo_map.get(level, "⚪")

    reasons_html = "".join(
        f'<div class="risk-reason"><span class="risk-reason-dot">◆</span>{r}</div>'
        for r in reasons[:4]
    )

    max_w = {"disease_confidence":35,"disease_severity":25,"weather_stress":20,"market_loss_risk":10,"rainfall_deficit":10}
    comp_rows = ""
    for fac, pts in components.items():
        mw  = max_w.get(fac, 10)
        pct = int((pts / mw) * 100) if mw else 0
        comp_rows += (
            f'<div class="comp-row">'
            f'<span class="comp-name">{fac.replace("_"," ")}</span>'
            f'<div class="comp-track"><div class="comp-fill" style="width:{pct}%"></div></div>'
            f'<span class="comp-pts">{pts}</span>'
            f'</div>'
        )

    score_label = ui.get("score_label", "Score")
    risk_label  = ui.get("risk_label", "RISK")
    breakdown   = ui.get("breakdown_label", "Risk Breakdown")

    return f"""
<div class="risk-wrap {cls}">
  <p class="risk-lbl">{emo} {level} {risk_label}</p>
  <p class="risk-score-txt">{score_label}: <strong style="color:var(--txt);font-size:.9rem">{score}</strong>/100</p>
  <div class="risk-track"><div class="risk-fill" style="width:{score}%"></div></div>
  {reasons_html}
</div>
<div class="comp-section">
  <div class="comp-hdr">{breakdown}</div>
  {comp_rows}
</div>"""


def _adv_card(icon: str, heading: str, body: str, urgency: str = "", card_cls: str = "") -> str:
    is_critical = any(w in urgency.lower() for w in ("24 hour", "immediate", "urgent"))
    urg_cls  = "urgency-critical" if is_critical else ""
    urg_html = f'<div class="urgency-tag {urg_cls}">⏱ {urgency}</div>' if urgency else ""
    return (
        f'<div class="adv-card {card_cls}">'
        f'<p class="adv-hdr">{icon} {heading}</p>'
        f'<p class="adv-body">{body}</p>'
        f'{urg_html}'
        f'</div>'
    )


def _action_plan(steps: list, heading: str = "7-Day Action Plan") -> str:
    rows = "".join(
        f'<div class="plan-step"><span class="plan-num">{i}</span><span>{s}</span></div>'
        for i, s in enumerate(steps, 1)
    )
    return f'<div class="plan-card"><p class="plan-hdr">📋 {heading}</p>{rows}</div>'


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def _sidebar():
    with st.sidebar:
        _h("""
        <div style="padding:6px 0 20px">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <span style="font-size:1.6rem">🌿</span>
            <div>
              <p style="font-family:var(--serif);font-size:1.25rem;color:var(--green);margin:0;line-height:1.1">Agri-Edge</p>
              <p style="font-size:.64rem;color:var(--txt3);text-transform:uppercase;letter-spacing:.09em;margin:0">AI Farm Advisor · India</p>
            </div>
          </div>
        </div>""")
        st.divider()

        crop = st.selectbox(
            "🌾 Crop Type",
            ["Tomato","Potato","Rice","Wheat","Cotton","Maize"],
            help="The crop you want to analyse",
        )
        if crop!= st.session_state.get("last_crop"):
            st.session_state.market_data=None
            st.session_state.last_crop=None
        location = st.text_input(
            "📍 Location",
            placeholder="PIN code or city  (e.g. 411001)",
            help="6-digit PIN code for hyperlocal weather, or any Indian city name",
        )

        lang_opts = [f"{LC.get(l,{}).get('flag','🌐')} {l}" for l in SUPPORTED_LANGUAGES]
        lang_disp = st.selectbox(
            "🗣 Advisory Language",
            options=lang_opts,
            index=SUPPORTED_LANGUAGES.index(st.session_state.get("language", DEFAULT_LANGUAGE)),
            help="The language for the AI advisory text",
        )
        sel_lang = lang_disp.split(" ", 1)[1] if " " in lang_disp else lang_disp
        if sel_lang != st.session_state.get("language", DEFAULT_LANGUAGE):
            st.session_state.language        = sel_lang
            st.session_state.advisory_result = None
            st.session_state.s_advisory      = "idle"
        else:
            st.session_state.language = sel_lang

        st.divider()
        _h("""
        <div style="font-size:.73rem;color:var(--txt2);line-height:1.85">
          <strong style="color:var(--txt);display:block;margin-bottom:4px">How to use</strong>
          <span style="color:var(--green)">1</span> &nbsp;Select crop type<br>
          <span style="color:var(--green)">2</span> &nbsp;Enter PIN code or city<br>
          <span style="color:var(--green)">3</span> &nbsp;Upload a leaf photo<br>
          <span style="color:var(--green)">4</span> &nbsp;Click <em>Analyse Crop</em><br>
          <span style="color:var(--green)">5</span> &nbsp;Review your advisory
        </div>""")
        st.divider()
        _h("""
        <div style="font-size:.65rem;color:var(--txt3);text-align:center;line-height:1.6">
          Built for Indian farmers 🇮🇳<br>
          <span style="color:var(--b2)">Powered by MobileNetV2 + LLM · v1.0</span>
        </div>""")

    return crop, location, st.session_state.language


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════

def _hero(crop: str, location: str, language: str):
    lcfg     = get_language_config(language)
    loc_pill = f'<span class="hero-pill">📍 {location}</span>' if location else '<span style="color:var(--txt3)">—</span>'
    lang_pill= f'<span class="hero-pill">{lcfg["flag"]} {lcfg["native_name"]}</span>'
    crop_pill= f'<span class="hero-pill">🌾 {crop}</span>'

    # Live stats from session
    d_ok = st.session_state.s_disease  == "ok"
    w_ok = st.session_state.s_weather  == "ok"
    a_ok = st.session_state.s_advisory == "ok"

    risk_lbl = ""
    if a_ok and st.session_state.advisory_result:
        rl = st.session_state.advisory_result.get("risk_level","—")
        emo= {"NONE":"🟢","LOW":"🔵","MODERATE":"🟡","HIGH":"🟠","CRITICAL":"🔴"}.get(rl,"⚪")
        risk_lbl = f'{emo} {rl}'
    else:
        risk_lbl = "—"

    disease_lbl = "—"
    if d_ok and st.session_state.disease_result:
        dl = st.session_state.disease_result.get("label","—")
        disease_lbl = dl[:28] + "…" if len(dl) > 28 else dl

    weather_lbl = "—"
    if w_ok and st.session_state.weather_data:
        temp = st.session_state.weather_data.get("current",{}).get("temp_c","—")
        cond = st.session_state.weather_data.get("current",{}).get("condition","—")
        weather_lbl = f'{temp}°C · {cond}'

    _h(f"""
    <div class="hero">
      <div class="hero-eyebrow">🌿 Agricultural Intelligence Platform</div>
      <h1 class="hero-h1">Agri-Edge Dashboard</h1>
      <p class="hero-sub">
        AI disease detection &nbsp;·&nbsp; hyperlocal weather &nbsp;·&nbsp; mandi price advisory
        &nbsp;&nbsp;
        Crop: {crop_pill} &nbsp; Location: {loc_pill} &nbsp; Language: {lang_pill}
      </p>
      <div class="hero-divider"></div>
      <div class="hero-stats">
        <div class="hero-stat">
          <span class="hero-stat-val" style="font-size:1rem;color:var(--txt2)">{disease_lbl}</span>
          <span class="hero-stat-lbl">Diagnosis</span>
        </div>
        <div class="hero-stat">
          <span class="hero-stat-val" style="font-size:1rem;color:var(--txt2)">{weather_lbl}</span>
          <span class="hero-stat-lbl">Current Weather</span>
        </div>
        <div class="hero-stat">
          <span class="hero-stat-val" style="font-size:1rem;color:var(--txt2)">{risk_lbl}</span>
          <span class="hero-stat-lbl">Risk Level</span>
        </div>
      </div>
    </div>""")


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL — DISEASE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _panel_disease():
    status = st.session_state.s_disease
    badge  = (
        '<span class="panel-hdr-badge badge-ok">✓ Analysed</span>'    if status == "ok"    else
        '<span class="panel-hdr-badge badge-err">✕ Error</span>'      if status == "error" else
        '<span class="panel-hdr-badge badge-demo">⬆ Upload</span>'
    )
    _h(f'<div class="panel"><div class="panel-hdr">🔬 Disease Detection {badge}</div>')

    f = st.file_uploader(
        "Upload leaf image",
        type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed",
        help="Take a sharp close-up of the affected leaf in natural light",
    )

    result = None
    if f:
        st.image(f, use_column_width=True, caption="Uploaded leaf image", output_format="JPEG")
        if f.name != st.session_state.last_img:
            with st.spinner("🔬 Running MobileNetV2 inference…"):
                _run_disease(f)
            st.session_state.last_img = f.name

        if st.session_state.s_disease == "error":
            _h(f'<div class="status-err">✕ Detection failed: {st.session_state.err_disease}</div>')
        elif st.session_state.disease_result:
            _h(_disease_panel_html(st.session_state.disease_result))
            result = st.session_state.disease_result
    else:
        _h("""<div class="empty">
        <span class="empty-ico">📷</span>
        Upload a <strong>clear, well-lit</strong> photo of a crop leaf<br>
        <span style="font-size:.72rem;color:var(--txt3)">JPEG · PNG · WebP supported</span>
        </div>""")

    _h("</div>")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL — WEATHER
# ══════════════════════════════════════════════════════════════════════════════

def _panel_weather(location: str):
    status = st.session_state.s_weather
    badge  = (
        '<span class="panel-hdr-badge badge-ok">✓ Live</span>'          if status == "ok"    else
        '<span class="panel-hdr-badge badge-err">✕ Error</span>'        if status == "error" else
        '<span class="panel-hdr-badge badge-demo">⌛ Waiting</span>'    if not location else ""
    )
    if status == "ok" and st.session_state.weather_data and st.session_state.weather_data.get("_mock"):
        badge = '<span class="panel-hdr-badge badge-demo">🎭 Demo</span>'

    _h(f'<div class="panel"><div class="panel-hdr">🌦 Hyperlocal Weather {badge}</div>')
    data = None

    if not location:
        _h("""<div class="empty">
        <span class="empty-ico">📍</span>
        Enter a <strong>PIN code</strong> or <strong>city name</strong><br>in the sidebar to load weather
        </div>""")
    else:
        if location != st.session_state.last_loc:
            with st.spinner("🌐 Fetching weather data…"):
                _run_weather(location)
            st.session_state.last_loc = location

        if st.session_state.s_weather == "error":
            _h(f'<div class="status-err">✕ {st.session_state.err_weather}</div>')
            _h('<div style="font-size:.75rem;color:var(--txt3);margin-top:4px">Verify your PIN code, city name, or OPENWEATHER_API_KEY.</div>')
        elif st.session_state.weather_data:
            wd  = st.session_state.weather_data
            cur = wd.get("current", {})
            loc = wd.get("location", {})
            adv = wd.get("farming_advisory", {})
            fc  = wd.get("forecast", [])
            data = wd

            city_str = f"{loc.get('city', location)}, {loc.get('country','IN')}"
            _h(f'<div style="font-size:.8rem;color:var(--txt2);margin-bottom:10px;display:flex;align-items:center;gap:6px">📍 <strong style="color:var(--txt)">{city_str}</strong> &nbsp;·&nbsp; ☁️ {cur.get("condition","—")} &nbsp;·&nbsp; {cur.get("pressure_hpa","—")} hPa</div>')

            _h(_weather_tiles(cur))

            if adv:
                _h('<div style="font-size:.63rem;text-transform:uppercase;letter-spacing:.09em;color:var(--txt3);font-weight:700;margin:12px 0 6px">Farming Conditions</div>')
                _h(_flags_html(adv))

            if adv.get("advisory_notes"):
                urgent_notes = [n for n in adv["advisory_notes"] if any(w in n.upper() for w in ("URGENT","FROST","HEAT","🚨","🥶"))]
                if urgent_notes:
                    _h(f'<div class="warn-box" style="margin-top:10px">⚠️ {urgent_notes[0]}</div>')

            if fc:
                _h('<div style="font-size:.63rem;text-transform:uppercase;letter-spacing:.09em;color:var(--txt3);font-weight:700;margin:14px 0 6px">5-Day Forecast</div>')
                _h(_forecast_html(fc))

    _h("</div>")
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL — MARKET PRICES
# ══════════════════════════════════════════════════════════════════════════════

def _panel_market(crop: str):
    if crop != st.session_state.get("last_crop") or not st.session_state.market_data:
        _run_market(crop)
        st.session_state.last_crop = crop
        st.session_state.advisory_result = None   # clear old advisory when crop changes
        st.session_state.s_advisory = "idle"
    p     = st.session_state.market_data or {}
    trend = p.get("trend", "stable").lower()
    tmap  = {"rising":("t-rising","↑ Rising"),"falling":("t-falling","↓ Falling")}
    tclass, tlabel = tmap.get(trend, ("t-stable","→ Stable"))
    msp   = p.get("msp")

    # MSP comparison
    msp_html    = ""
    msp_alert   = ""
    if msp:
        modal_raw = str(p.get("modal_price","0")).replace("—","0")
        is_below  = float(modal_raw) < float(msp) if modal_raw.replace(".","").isdigit() else False
        msp_cls   = "msp-warn" if is_below else "msp-ok"
        msp_note  = " ⚠ Below MSP" if is_below else " ✓ Above MSP"
        msp_html  = f'<div class="msp-row"><span class="msp-lbl">MSP (Govt. floor)</span><span class="msp-val {msp_cls}">₹{msp}/q{msp_note}</span></div>'
        if is_below:
            msp_alert = f'<div class="msp-alert">⚠️ Market price is below MSP (₹{msp}/q). Approach your nearest APMC or FCI.</div>'

    badge = '<span class="panel-hdr-badge badge-ok">✓ Live</span>'
    _h(f'<div class="panel"><div class="panel-hdr">📊 Mandi Market Prices {badge}</div>')
    _h(f"""
    <div style="font-size:.68rem;color:var(--txt3);margin-bottom:2px">Modal price &nbsp;·&nbsp; {p.get('crop', crop)}</div>
    <div class="mkt-main">₹{p.get('modal_price','—')}</div>
    <div class="mkt-unit">per quintal (100 kg)</div>
    <div class="mkt-range">
      <div class="mkt-box">
        <span class="mkt-bv" style="color:var(--green)">₹{p.get('min_price','—')}</span>
        <span class="mkt-bl">Min</span>
      </div>
      <div class="mkt-box">
        <span class="mkt-bv" style="color:var(--redl)">₹{p.get('max_price','—')}</span>
        <span class="mkt-bl">Max</span>
      </div>
    </div>
    <span class="mkt-trend {tclass}">{tlabel}</span>
    {msp_html}
    {msp_alert}
    <div class="mkt-footer">📍 {p.get('market','Local Mandi')}<br>🗓 {p.get('date','—')}</div>
    </div>""")

    return p


# ══════════════════════════════════════════════════════════════════════════════
#  ADVISORY SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _section_advisory(crop: str, language: str):
    _divider("AI Advisory Engine")

    # CTA block
    d_ok = st.session_state.s_disease  == "ok"
    w_ok = st.session_state.s_weather  == "ok"

    hints = []
    if not d_ok: hints.append("📷 upload a leaf image")
    if not w_ok: hints.append("📍 enter a location")
    hint_str = " and ".join(hints) + " for a richer advisory" if hints else ""

    _h(f"""
    <div class="cta-wrap">
      <div class="cta-text">
        <h3>Generate Your Advisory</h3>
        <p>Receive disease treatment, irrigation schedule, fertilizer plan and market strategy in <strong style="color:var(--green)">{language}</strong>.{(' Also ' + hint_str + '.') if hint_str else ''}</p>
      </div>
    </div>""")

    btn_col, _ = st.columns([1, 2.5])
    with btn_col:
        run = st.button(
            "🌱 Analyse Crop & Generate Advisory",
            type="primary",
            use_container_width=True,
        )

    if not run and not st.session_state.pipeline_run:
        _h("""<div class="empty" style="max-width:560px;margin:12px auto">
        <span class="empty-ico">🤖</span>
        <strong>Ready when you are.</strong><br>
        Upload a leaf image, enter your location, then click the button above<br>to run the complete analysis pipeline.
        </div>""")
        return

    if run:
        st.session_state.pipeline_run = True

        # Weather (if not already fetched)
        loc = st.session_state.last_loc
        if loc and st.session_state.s_weather == "idle":
            with st.spinner("🌐 Fetching weather…"):
                _run_weather(loc)

        # Market (instant)
        if not st.session_state.market_data:
            _run_market(crop)

        # Advisory
        with st.spinner("🤖 Generating AI advisory… this takes a few seconds"):
            _run_advisory(crop, language)

        st.rerun()

    # ── Display results ────────────────────────────────────────────────────────
    if st.session_state.s_advisory == "error":
        _h('<div class="status-err">✕ Advisory generation failed. Check your GROQ_API_KEY or try again.</div>')
        return

    if not st.session_state.advisory_result:
        return

    res      = st.session_state.advisory_result
    sections = res.get("sections", {})
    lcfg     = get_language_config(language)
    ui       = lcfg["ui"]

    # Success strip
    source = res.get("source", "mock")
    model  = res.get("model_used", "rule-based")
    gen_at = res.get("generated_at", "")
    if source == "mock":
        _h(f'<div class="status-warn">🎭 {ui.get("demo_badge","Demo Mode")} — Add GROQ_API_KEY for real AI advisory</div>')
    else:
        _h(f'<div class="status-ok">✓ Advisory generated by {model} in {lcfg["native_name"]} · {gen_at}</div>')

    left, right = st.columns([1, 1.7])

    # ── Left: Risk assessment ──────────────────────────────────────────────────
    with left:
        _divider(ui.get("risk_title", "Risk Assessment"))

        _h(f'<div class="lang-badge">{lcfg["flag"]} {lcfg["native_name"]}</div>')

        _h(_risk_html(
            level      = res.get("risk_level", "NONE"),
            score      = res.get("risk_score", 0),
            reasons    = res.get("risk_reasons", []),
            components = res.get("component_scores", {}),
            ui         = ui,
        ))

        _h(f'<div class="src-tag">Model: {model}<br>Provider: {source}<br>{gen_at}</div>')

    # ── Right: Advisory sections ───────────────────────────────────────────────
    with right:
        _divider(ui.get("report_title", "Advisory Report"))

        # Summary
        if res.get("summary"):
            _h(f'<div class="summary-box"><span class="summary-icon">📌</span><span>{res["summary"]}</span></div>')

        # Section cards with left-border colour coding
        sec_cfg = [
            ("disease_warning",   "🦠", "adv-card-disease"),
            ("irrigation_advice", "💧", "adv-card-irrigation"),
            ("crop_inputs",       "🌱", "adv-card-inputs"),
            ("market_advice",     "📊", "adv-card-market"),
        ]
        for key, icon, card_cls in sec_cfg:
            sec = sections.get(key, {})
            if not sec: continue
            _h(_adv_card(
                icon     = icon,
                heading  = sec.get("heading", key.replace("_"," ").title()),
                body     = sec.get("body", ""),
                urgency  = sec.get("urgency", ""),
                card_cls = card_cls,
            ))

        # Action plan
        action = sections.get("action_plan", {})
        if action and action.get("steps"):
            _h(_action_plan(action["steps"], heading=ui.get("plan_heading","7-Day Action Plan")))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not _OK:
        st.error(f"❌ Import error: {_ERR}")
        st.code("cd agri_edge && streamlit run app.py", language="bash")
        st.stop()

    st.markdown(_CSS, unsafe_allow_html=True)
    _init()

    crop, location, language = _sidebar()
    _hero(crop, location, language)
    _pipe_bar()

    # ── Three data panels ──────────────────────────────────────────────────────
    _divider("Farm Data")
    col1, col2, col3 = st.columns([1.15, 1.1, 0.9])
    with col1: _panel_disease()
    with col2: _panel_weather(location)
    with col3: _panel_market(crop)

    # ── Advisory ───────────────────────────────────────────────────────────────
    _section_advisory(crop, language)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _divider("Agri-Edge · AI-Powered Agricultural Advisory · India 🇮🇳")


if __name__ == "__main__":
    main()
