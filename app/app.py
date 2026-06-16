"""
Migros · Location Intelligence — Aurora dark dashboard.

A modern, vibrant dark interface: deep midnight surfaces, an aurora
violet→cyan signature gradient, glassmorphism cards, ambient glow and
fluid entrance animations. Same analytical pipeline (load → spatial join →
composite scoring → OLS opportunity gap), entirely re-skinned and re-laid.
"""

import os
import warnings

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Migros · Location Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# DESIGN TOKENS — aurora dark (midnight + violet→cyan)
# ──────────────────────────────────────────────────────────────────────
BG       = "#070811"   # page background
BG_2     = "#0A0C18"   # secondary background
SURF     = "#0F1222"   # card surface
SURF_2   = "#151A2E"   # raised surface
SURF_3   = "#1C2240"   # hover / active
HAIR     = "#1E2440"   # hairline divider
BORDER   = "#272E50"   # standard border
BORDER_H = "#3D477A"   # hover border
TEXT     = "#F3F5FE"   # primary text
TEXT_2   = "#BCC2E0"   # secondary text
MUTED    = "#7A82AC"   # tertiary / labels
FAINT    = "#474E78"   # quaternary

# Vibrant aurora accents
VIOLET   = "#7C6BF8"   # primary accent
VIOLET_L = "#9A8BFF"
INDIGO   = "#6366F1"
CYAN     = "#22D3EE"   # secondary accent
CYAN_L   = "#5AE3F5"
MAGENTA  = "#F472B6"
EMERALD  = "#34D399"
AMBER    = "#FBBF24"
ROSE     = "#FB7185"

ACCENT   = VIOLET      # signature
ACCENT_2 = CYAN
POS      = EMERALD
NEG      = ROSE

GRAD     = f"linear-gradient(135deg, {VIOLET} 0%, {CYAN} 100%)"
GRAD_3   = f"linear-gradient(120deg, {VIOLET} 0%, {MAGENTA} 45%, {CYAN} 100%)"

# Chart palette — vivid but balanced
PAL = [VIOLET, CYAN, MAGENTA, EMERALD, AMBER, INDIGO]
# Continuous violet→cyan scale for value-encoded bars
SCALE = [[0.0, INDIGO], [0.5, VIOLET], [1.0, CYAN]]


# ──────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@200;300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif;
    color: {TEXT};
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    font-feature-settings: 'ss01', 'cv11', 'tnum';
}}
.stApp {{
    background:
      radial-gradient(1100px 700px at 12% -5%, rgba(124,107,248,0.16) 0%, transparent 55%),
      radial-gradient(1000px 720px at 100% 0%, rgba(34,211,238,0.12) 0%, transparent 50%),
      radial-gradient(900px 800px at 85% 110%, rgba(244,114,182,0.10) 0%, transparent 55%),
      {BG};
    background-attachment: fixed;
}}
.block-container {{ padding: 1.4rem 2.6rem 5rem; max-width: 1520px; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

/* ── ANIMATIONS ────────────────────────────────────────── */
@keyframes fadeUp {{
    0%   {{ opacity: 0; transform: translateY(16px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    0% {{ opacity: 0; }} 100% {{ opacity: 1; }}
}}
@keyframes gradShift {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
@keyframes floatY {{
    0%,100% {{ transform: translateY(0); }}
    50%     {{ transform: translateY(-18px); }}
}}
@keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(52,211,153,0.55); }}
    70%  {{ box-shadow: 0 0 0 7px rgba(52,211,153,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(52,211,153,0); }}
}}
@keyframes sheen {{
    0%   {{ transform: translateX(-120%); }}
    100% {{ transform: translateX(220%); }}
}}
@keyframes spinGlow {{
    0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }}
}}

.block-container > div {{ animation: fadeIn .5s ease both; }}

/* ── SIDEBAR ───────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #080A15 0%, #06070F 100%);
    border-right: 1px solid {HAIR};
}}
[data-testid="stSidebar"] * {{ color: {TEXT}; }}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 0; }}

.brand {{
    padding: 30px 22px 26px;
    border-bottom: 1px solid {HAIR};
    position: relative; overflow: hidden;
}}
.brand::after {{
    content: ''; position: absolute; left: 22px; bottom: -1px;
    width: 46px; height: 2px; background: {GRAD}; border-radius: 2px;
}}
.brand-row {{ display: flex; align-items: center; gap: 13px; }}
.brand-logo {{
    width: 40px; height: 40px; border-radius: 12px;
    background: {GRAD};
    display: flex; align-items: center; justify-content: center;
    font-family: 'Sora', sans-serif;
    font-size: 19px; font-weight: 700; color: #fff;
    letter-spacing: -0.5px;
    box-shadow: 0 6px 22px rgba(124,107,248,0.45),
                0 0 0 1px rgba(255,255,255,0.08) inset;
    position: relative; overflow: hidden;
}}
.brand-logo::after {{
    content: ''; position: absolute; top: 0; left: 0;
    width: 40%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
    animation: sheen 4.5s ease-in-out infinite;
}}
.brand-text .name {{
    font-family: 'Sora', sans-serif;
    font-size: 15px; font-weight: 600; letter-spacing: -0.3px;
    color: {TEXT}; line-height: 1;
}}
.brand-text .sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px; color: {MUTED}; letter-spacing: 2.4px;
    text-transform: uppercase; margin-top: 6px; font-weight: 500;
}}
.brand-meta {{
    margin-top: 24px; display: grid; grid-template-columns: 1fr 1fr; gap: 13px 16px;
}}
.bm-cell {{
    background: rgba(255,255,255,0.02); border: 1px solid {HAIR};
    border-radius: 9px; padding: 10px 12px;
}}
.bm-cell .l {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: {FAINT}; letter-spacing: 1.6px;
    text-transform: uppercase; margin-bottom: 5px;
}}
.bm-cell .v {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: {TEXT_2}; font-weight: 500; letter-spacing: 0.4px;
}}

.nav-label {{
    padding: 24px 24px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px; letter-spacing: 2.8px; color: {FAINT};
    text-transform: uppercase; font-weight: 500;
}}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 4px; padding: 0 12px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    width: 100%; padding: 11px 14px !important; margin: 0 !important;
    border-radius: 11px; border: 1px solid transparent;
    cursor: pointer; font-size: 13.5px; font-weight: 500;
    color: {MUTED}; transition: all 0.18s cubic-bezier(.4,0,.2,1);
    letter-spacing: 0.1px; position: relative; overflow: hidden;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background: {SURF_2}; color: {TEXT};
    transform: translateX(3px);
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background: linear-gradient(135deg, rgba(124,107,248,0.18) 0%, rgba(34,211,238,0.10) 100%);
    border-color: rgba(124,107,248,0.45);
    color: {TEXT}; font-weight: 600;
    box-shadow: 0 4px 18px rgba(124,107,248,0.18);
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked)::before {{
    content: ''; position: absolute; left: 0; top: 18%; bottom: 18%;
    width: 3px; border-radius: 3px; background: {GRAD};
}}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}

.side-foot {{
    margin: 26px 22px 22px;
    padding: 16px 14px;
    border: 1px solid {HAIR}; border-radius: 12px;
    background: rgba(255,255,255,0.015);
}}
.side-foot .row {{
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: {MUTED}; letter-spacing: 1.2px;
    padding: 5px 0;
}}
.side-foot .row span:last-child {{ color: {TEXT_2}; }}
.side-foot .row.gold span:last-child {{ color: {CYAN_L}; }}
.status-pulse {{
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: {POS}; margin-right: 7px;
    animation: pulse 2.4s infinite; vertical-align: middle;
}}

/* ── PAGE HEADER ───────────────────────────────────────── */
.page-bar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 11px 18px; margin-bottom: 26px;
    background: rgba(255,255,255,0.02);
    border: 1px solid {HAIR}; border-radius: 12px;
    backdrop-filter: blur(8px);
    animation: fadeUp .5s ease both;
}}
.crumb {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: {MUTED}; letter-spacing: 1.8px;
    text-transform: uppercase; font-weight: 500;
}}
.crumb .sep {{ color: {FAINT}; margin: 0 9px; }}
.crumb .now {{
    background: {GRAD}; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; font-weight: 700;
}}
.page-tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; color: {MUTED}; letter-spacing: 1.5px;
}}
.page-tag .ts {{ color: {TEXT_2}; }}

.hero {{
    display: grid; grid-template-columns: 1.7fr 1fr;
    gap: 48px; align-items: end; margin-bottom: 8px;
    animation: fadeUp .6s ease both;
}}
.hero-title {{
    font-family: 'Sora', sans-serif;
    font-size: 50px; font-weight: 300; line-height: 1.05;
    letter-spacing: -1.8px; color: {TEXT};
}}
.hero-title b {{ font-weight: 700; color: {TEXT}; }}
.hero-title .grad {{
    background: linear-gradient(120deg, {VIOLET}, {MAGENTA}, {CYAN}, {VIOLET});
    background-size: 280% 280%;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; font-weight: 600;
    animation: gradShift 7s ease infinite;
}}
.hero-lede {{
    font-size: 13.5px; color: {TEXT_2};
    line-height: 1.75; font-weight: 300;
    border-left: 2px solid transparent;
    border-image: {GRAD} 1;
    padding-left: 18px;
}}
.hero-lede b {{ color: {TEXT}; font-weight: 600; }}

.section-bar {{
    display: flex; align-items: baseline; justify-content: space-between;
    margin: 46px 0 4px; padding-bottom: 12px;
    border-bottom: 1px solid {HAIR};
}}
.section-left {{ display: flex; align-items: baseline; gap: 16px; }}
.section-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: 2px; font-weight: 700;
    background: {GRAD}; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.section-title {{
    font-family: 'Sora', sans-serif;
    font-size: 22px; font-weight: 600; letter-spacing: -0.5px; color: {TEXT};
}}
.section-right {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; color: {MUTED}; letter-spacing: 1.5px;
    text-transform: uppercase;
}}
.section-sub {{
    font-size: 13px; color: {MUTED};
    line-height: 1.6; max-width: 720px;
    margin: 14px 0 22px; font-weight: 300;
}}

/* ── KPI STRIP ─────────────────────────────────────────── */
.kpi-row {{
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 14px; margin: 28px 0 6px;
}}
.kpi-cell {{
    padding: 22px 22px 24px; border-radius: 16px;
    background: linear-gradient(165deg, {SURF_2} 0%, {SURF} 100%);
    border: 1px solid {HAIR};
    position: relative; overflow: hidden;
    transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
    animation: fadeUp .6s ease both;
}}
.kpi-cell::before {{
    content: ''; position: absolute; inset: 0; border-radius: 16px;
    padding: 1px; background: {GRAD};
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    opacity: 0; transition: opacity .22s ease;
}}
.kpi-cell:hover {{
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.4), 0 0 28px rgba(124,107,248,0.12);
}}
.kpi-cell:hover::before {{ opacity: 1; }}
.kpi-cell .icn {{
    width: 30px; height: 30px; border-radius: 9px; margin-bottom: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; background: rgba(124,107,248,0.12);
    border: 1px solid rgba(124,107,248,0.25); color: {VIOLET_L};
}}
.kpi-cell .label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px; letter-spacing: 1.8px;
    text-transform: uppercase; color: {MUTED};
    margin-bottom: 12px; font-weight: 500;
}}
.kpi-cell .val {{
    font-family: 'Sora', sans-serif;
    font-size: 36px; line-height: 1; letter-spacing: -1.4px;
    color: {TEXT}; font-weight: 300;
    font-variant-numeric: tabular-nums;
}}
.kpi-cell .val.grad {{
    background: {GRAD}; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; font-weight: 600;
}}
.kpi-cell .unit {{ font-size: 12px; color: {MUTED}; margin-left: 4px; }}
.kpi-cell .delta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: {FAINT}; margin-top: 12px;
    letter-spacing: 1.2px; text-transform: uppercase;
}}
.kpi-cell .delta .pos {{ color: {POS}; font-weight: 600; }}
.kpi-cell .delta .neg {{ color: {NEG}; font-weight: 600; }}

/* ── RECOMMENDATION CARD ───────────────────────────────── */
.rec {{
    position: relative; border-radius: 20px; margin: 28px 0 24px;
    background:
      radial-gradient(600px 300px at 0% 0%, rgba(124,107,248,0.14) 0%, transparent 60%),
      radial-gradient(500px 300px at 100% 100%, rgba(34,211,238,0.10) 0%, transparent 55%),
      linear-gradient(165deg, {SURF_2} 0%, {SURF} 100%);
    border: 1px solid {BORDER};
    display: grid; grid-template-columns: 1.6fr 1fr;
    overflow: hidden; box-shadow: 0 20px 60px rgba(0,0,0,0.45);
    animation: fadeUp .7s ease both;
}}
.rec::after {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: {GRAD_3}; background-size: 200% 100%;
    animation: gradShift 6s ease infinite;
}}
.rec-left {{ padding: 38px 42px 40px; border-right: 1px solid {HAIR}; }}
.rec-eyebrow {{ display: flex; align-items: center; gap: 11px; margin-bottom: 24px; }}
.rec-eyebrow .tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 2.2px;
    text-transform: uppercase; color: #fff;
    background: {GRAD}; padding: 5px 11px; border-radius: 7px; font-weight: 600;
    box-shadow: 0 4px 14px rgba(124,107,248,0.4);
}}
.rec-eyebrow .ref {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: {MUTED}; letter-spacing: 1.5px;
}}
.rec-name {{
    font-family: 'Sora', sans-serif;
    font-size: 58px; font-weight: 700; letter-spacing: -2.4px;
    color: {TEXT}; line-height: 1; margin-bottom: 12px;
    background: linear-gradient(120deg, {TEXT} 0%, {VIOLET_L} 55%, {CYAN_L} 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}}
.rec-place {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: {MUTED}; letter-spacing: 1.8px;
    text-transform: uppercase;
}}
.rec-summary {{
    margin-top: 26px; padding-top: 22px;
    border-top: 1px solid {HAIR};
    font-size: 13px; color: {TEXT_2}; line-height: 1.75; font-weight: 300;
}}
.rec-summary b {{ color: {TEXT}; font-weight: 600; }}
.rec-right {{ padding: 0; }}
.rec-metric {{
    padding: 17px 28px; border-bottom: 1px solid {HAIR};
    display: flex; justify-content: space-between; align-items: baseline;
    transition: background .18s ease;
}}
.rec-metric:hover {{ background: rgba(124,107,248,0.06); }}
.rec-metric:last-child {{ border-bottom: none; }}
.rec-metric .l {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 1.5px;
    text-transform: uppercase; color: {MUTED};
}}
.rec-metric .v {{
    font-family: 'Sora', sans-serif;
    font-size: 18px; font-weight: 500; color: {TEXT};
    font-variant-numeric: tabular-nums; letter-spacing: -0.3px;
}}
.rec-metric .v.grad {{
    background: {GRAD}; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; font-weight: 700;
}}

/* ── PROCESS FLOW ──────────────────────────────────────── */
.flow {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 14px; margin: 6px 0 14px;
}}
.flow-cell {{
    padding: 24px 26px 26px; border-radius: 15px;
    background: linear-gradient(165deg, {SURF_2} 0%, {SURF} 100%);
    border: 1px solid {HAIR}; position: relative; overflow: hidden;
    transition: transform .2s ease, border-color .2s ease;
    animation: fadeUp .6s ease both;
}}
.flow-cell:hover {{ transform: translateY(-3px); border-color: {BORDER_H}; }}
.flow-cell.win {{
    background:
      radial-gradient(400px 200px at 50% 0%, rgba(124,107,248,0.18) 0%, transparent 70%),
      linear-gradient(165deg, {SURF_2} 0%, {SURF} 100%);
    border-color: rgba(124,107,248,0.45);
    box-shadow: 0 0 28px rgba(124,107,248,0.15);
}}
.flow-cell .step {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px; letter-spacing: 2px; color: {MUTED};
    text-transform: uppercase; margin-bottom: 14px; font-weight: 500;
}}
.flow-cell.win .step {{ color: {CYAN_L}; }}
.flow-cell .num {{
    font-family: 'Sora', sans-serif;
    font-size: 44px; line-height: 1; font-weight: 300;
    letter-spacing: -1.8px; color: {TEXT};
    font-variant-numeric: tabular-nums;
}}
.flow-cell.win .num {{
    background: {GRAD}; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; font-weight: 600;
}}
.flow-cell .ttl {{
    font-size: 13px; font-weight: 600; color: {TEXT};
    margin: 12px 0 4px; letter-spacing: -0.2px;
}}
.flow-cell .dsc {{ font-size: 11.5px; color: {MUTED}; line-height: 1.55; font-weight: 300; }}
.flow-cell .arrow {{
    position: absolute; right: -8px; top: 38px; z-index: 4;
    font-family: 'JetBrains Mono', monospace;
    color: {VIOLET}; font-size: 15px; font-weight: 700;
}}

/* ── PANEL (glass) ─────────────────────────────────────── */
.panel {{
    background: linear-gradient(165deg, rgba(21,26,46,0.7) 0%, rgba(15,18,34,0.7) 100%);
    border: 1px solid {HAIR}; border-radius: 16px;
    padding: 18px 22px 20px; margin-bottom: 16px;
    backdrop-filter: blur(10px);
    animation: fadeUp .6s ease both;
}}
.panel-head {{
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 14px; margin-bottom: 6px;
    border-bottom: 1px solid {HAIR};
}}
.panel-title {{
    font-family: 'Sora', sans-serif;
    font-size: 13.5px; font-weight: 600; color: {TEXT}; letter-spacing: -0.1px;
}}
.panel-sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: {CYAN_L}; letter-spacing: 1.4px; text-transform: uppercase;
}}

/* ── METHOD CARDS ──────────────────────────────────────── */
.method {{
    background: linear-gradient(165deg, {SURF_2} 0%, {SURF} 100%);
    border: 1px solid {HAIR}; border-radius: 16px;
    padding: 26px 28px; height: 224px;
    position: relative; overflow: hidden;
    transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
    display: flex; flex-direction: column;
    animation: fadeUp .6s ease both;
}}
.method:hover {{
    transform: translateY(-4px); border-color: {BORDER_H};
    box-shadow: 0 16px 40px rgba(0,0,0,0.4);
}}
.method::before {{
    content: ''; position: absolute; top: 0; left: 26px; width: 36px; height: 2px;
    background: {GRAD}; border-radius: 2px;
}}
.method-idx {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; letter-spacing: 2px; font-weight: 700;
    margin-bottom: 18px;
    background: {GRAD}; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.method-title {{
    font-family: 'Sora', sans-serif;
    font-size: 16px; font-weight: 600; color: {TEXT};
    letter-spacing: -0.3px; margin-bottom: 10px;
}}
.method-body {{ font-size: 12.5px; color: {MUTED}; line-height: 1.6; font-weight: 300; flex: 1; }}
.method-foot {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: {CYAN_L}; letter-spacing: 1.4px;
    padding-top: 14px; border-top: 1px solid {HAIR}; text-transform: uppercase;
}}

/* ── TOOLBAR ───────────────────────────────────────────── */
.toolbar {{
    background: linear-gradient(165deg, rgba(21,26,46,0.6) 0%, rgba(15,18,34,0.6) 100%);
    border: 1px solid {HAIR}; border-radius: 14px;
    padding: 16px 22px 4px; margin-bottom: 22px;
    backdrop-filter: blur(8px);
}}
.toolbar-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 2.2px; text-transform: uppercase;
    color: {CYAN_L}; margin-bottom: 10px; font-weight: 600;
}}

/* ── TABLE ─────────────────────────────────────────────── */
.tbl-wrap {{
    background: linear-gradient(165deg, {SURF_2} 0%, {SURF} 100%);
    border: 1px solid {HAIR}; border-radius: 16px; overflow: hidden;
    animation: fadeUp .6s ease both;
}}
.tbl-head {{
    padding: 15px 22px; border-bottom: 1px solid {HAIR};
    display: flex; align-items: center; gap: 12px;
}}
table.et {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
table.et th {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 1.6px; text-transform: uppercase;
    color: {MUTED}; padding: 12px 18px; border-bottom: 1px solid {HAIR};
    text-align: left; font-weight: 500; background: rgba(0,0,0,0.25);
}}
table.et td {{
    padding: 13px 18px; border-bottom: 1px solid {HAIR};
    color: {TEXT_2}; font-variant-numeric: tabular-nums;
}}
table.et tr:last-child td {{ border-bottom: none; }}
table.et tbody tr {{ transition: background .14s ease; }}
table.et tbody tr:hover td {{ background: rgba(124,107,248,0.07); }}
table.et td.mono {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: {FAINT}; font-weight: 500;
}}
table.et td.name {{ color: {TEXT}; font-weight: 600; letter-spacing: -0.1px; }}
table.et td.grad {{
    font-weight: 700;
    background: {GRAD}; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}}
table.et tr.win td {{
    background: linear-gradient(90deg, rgba(124,107,248,0.12), rgba(34,211,238,0.04)) !important;
}}
table.et tr.win td:first-child {{
    box-shadow: inset 3px 0 0 0 {VIOLET}; padding-left: 16px;
}}

/* ── BADGES & CHIPS ────────────────────────────────────── */
.badge {{
    display: inline-flex; align-items: center;
    background: transparent; color: {TEXT_2};
    border: 1px solid {BORDER}; border-radius: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 600; letter-spacing: 1.5px;
    padding: 5px 11px; text-transform: uppercase;
}}
.badge.grad {{ background: {GRAD}; color: #fff; border-color: transparent;
    box-shadow: 0 4px 14px rgba(124,107,248,0.35); }}
.chip {{
    display: inline-flex; align-items: center;
    background: rgba(255,255,255,0.03); border: 1px solid {HAIR};
    color: {MUTED}; border-radius: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; padding: 5px 11px; letter-spacing: 1px; margin-right: 6px;
}}
.chip .dot {{
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: {GRAD}; margin-right: 7px;
}}

/* ── INPUTS ────────────────────────────────────────────── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div {{
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important; color: {TEXT} !important;
}}
div[data-testid="stMultiSelect"] [data-baseweb="tag"] {{
    background: {VIOLET} !important; border-radius: 6px !important;
}}
label p, .stSlider label p, .stCheckbox label p {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important; letter-spacing: 1.5px !important;
    text-transform: uppercase !important; color: {MUTED} !important;
    font-weight: 600 !important;
}}
.stSlider [data-baseweb="slider"] > div > div > div {{ background: {GRAD} !important; }}
.stSlider [data-baseweb="slider"] > div > div {{ background: {BORDER} !important; }}
.stSlider [role="slider"] {{ box-shadow: 0 0 0 4px rgba(124,107,248,0.25) !important; }}
.stCheckbox [role="checkbox"][aria-checked="true"] {{
    background: {VIOLET} !important; border-color: {VIOLET} !important;
}}
.stButton > button {{
    background: {GRAD} !important; color: #fff !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'JetBrains Mono', sans-serif !important;
    font-weight: 600 !important; font-size: 11px !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    box-shadow: 0 6px 18px rgba(124,107,248,0.35) !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(124,107,248,0.5) !important;
}}
div[data-testid="stExpander"] {{
    background: rgba(255,255,255,0.02); border: 1px solid {HAIR};
    border-radius: 12px;
}}
div[data-testid="stExpander"] summary {{ color: {TEXT_2} !important; }}

::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 8px; }}
::-webkit-scrollbar-thumb:hover {{ background: {BORDER_H}; }}

/* folium iframe rounding */
iframe {{ border-radius: 14px; }}

/* ── COLOPHON ──────────────────────────────────────────── */
.colophon {{
    margin-top: 52px; padding-top: 24px;
    border-top: 1px solid {HAIR};
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 30px;
}}
.colo-cell {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 1.4px; color: {FAINT};
    text-transform: uppercase; line-height: 1.85;
}}
.colo-cell b {{
    display: block; margin-bottom: 7px; letter-spacing: 1.8px; font-weight: 700;
    background: {GRAD}; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.colo-cell span {{ color: {TEXT_2}; }}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────
# DATA  (pipeline unchanged)
# ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA_URL = "https://raw.githubusercontent.com/so-rn/Migros-Location-Optimizer/main/data/"


def _path(rel: str) -> str:
    local = os.path.join(DATA_DIR, rel)
    return local if os.path.exists(local) else DATA_URL + rel


@st.cache_data(show_spinner=False)
def load_data():
    df_stores = pd.read_csv(_path("geneva_supermarkets_data_with_address.csv"))
    stores_gdf = gpd.GeoDataFrame(
        df_stores,
        geometry=gpd.points_from_xy(df_stores.longitude, df_stores.latitude),
        crs="EPSG:4326",
    )
    df_pop = pd.read_csv(_path("OCS_POPBATLOG_COMMUNE.csv"), sep=";")
    df_pop["COMMUNE"] = df_pop["COMMUNE"].str.strip()
    df_power = pd.read_csv(_path("finance/geneva_purchasing_power_proxy_all_years.csv"))
    df_p22 = df_power[df_power["year"] == 2022].copy()
    df_p22["commune"] = df_p22["commune"].str.strip()
    return stores_gdf, df_pop, df_p22


@st.cache_data(show_spinner=False)
def load_boundaries():
    b = gpd.read_file(_path("geneva_communes_boundaries.geojson"))
    return b[["COMMUNE_NAME", "geometry"]].set_crs(epsg=4326, allow_override=True)


@st.cache_data(show_spinner=False)
def build_master(_stores, _pop, _power, _bounds):
    joined = gpd.sjoin(_stores, _bounds, how="inner", predicate="intersects")
    store_ct = joined.groupby("COMMUNE_NAME").size().reset_index(name="STORE_COUNT")
    df = (
        _bounds.merge(_pop, left_on="COMMUNE_NAME", right_on="COMMUNE", how="left")
        .merge(
            _power[["commune", "proxy_purchasing_power_median_chf"]],
            left_on="COMMUNE_NAME", right_on="commune", how="left",
        )
        .merge(store_ct, on="COMMUNE_NAME", how="left")
    )
    df["STORE_COUNT"] = df["STORE_COUNT"].fillna(0)
    df["PCT_WORKING_AGE"] = (df["AGE_20_64"] / df["POPULATION"]) * 100
    df["PCT_SINGLE_FAMILY"] = (df["MAISON_INDIV"] / df["BATLOG_TOT"]) * 100
    df["PCT_FOREIGNERS"] = (df["POP_ETR"] / df["POPULATION"]) * 100
    df_clean = df.dropna(
        subset=["POPULATION", "proxy_purchasing_power_median_chf"]
    ).copy()
    df_clean = df_clean[
        ~df_clean["COMMUNE_NAME"].isin(["Genève", "Geneve", "Geneva"])
    ].reset_index(drop=True)
    return df_clean, joined


@st.cache_data(show_spinner=False)
def run_pipeline(_df):
    top20 = _df.sort_values("POPULATION", ascending=False).head(20).copy().reset_index(drop=True)
    top20["RANK"] = range(1, 21)

    def mm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0

    s2 = top20.copy()
    s2["SC_INC"] = mm(s2["proxy_purchasing_power_median_chf"])
    s2["SC_FOR"] = mm(s2["PCT_FOREIGNERS"])
    s2["SC_AGE"] = mm(s2["PCT_WORKING_AGE"])
    s2["SC_URB"] = 1 - mm(s2["PCT_SINGLE_FAMILY"])
    W = {"SC_INC": 0.35, "SC_FOR": 0.25, "SC_AGE": 0.20, "SC_URB": 0.20}
    s2["COMPOSITE_SCORE"] = sum(s2[k] * v for k, v in W.items())
    s2 = s2.sort_values("COMPOSITE_SCORE", ascending=False).reset_index(drop=True)
    top5 = s2.head(5).copy()
    top5["RANK"] = range(1, 6)

    X0 = sm.add_constant(_df[["POPULATION", "proxy_purchasing_power_median_chf"]])
    m0 = sm.OLS(_df["STORE_COUNT"], X0).fit()
    cooks = m0.get_influence().cooks_distance[0]
    df_f = _df.copy()
    df_f["COOKS_D"] = cooks
    df_f = df_f[df_f["COOKS_D"] <= 4 / len(df_f)].copy().reset_index(drop=True)
    Xf = sm.add_constant(df_f[["POPULATION", "proxy_purchasing_power_median_chf"]])
    mf = sm.OLS(df_f["STORE_COUNT"], Xf).fit()
    df_f["PREDICTED"] = mf.predict(Xf)
    df_f["OPPORTUNITY"] = df_f["PREDICTED"] - df_f["STORE_COUNT"]

    top5_ols = (
        top5.merge(df_f[["COMMUNE_NAME", "PREDICTED", "OPPORTUNITY"]],
                   on="COMMUNE_NAME", how="left")
        .sort_values("OPPORTUNITY", ascending=False).reset_index(drop=True)
    )
    top5_ols["RANK"] = range(1, len(top5_ols) + 1)
    champion = top5_ols.iloc[0].copy()
    return top20, top5, top5_ols, champion, df_f, mf


def base_layout(**kw):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=TEXT, size=11),
        xaxis=dict(
            gridcolor=HAIR, zerolinecolor=HAIR, linecolor=HAIR,
            tickfont=dict(color=MUTED, size=10),
        ),
        yaxis=dict(
            gridcolor=HAIR, zerolinecolor=HAIR, linecolor=HAIR,
            tickfont=dict(color=TEXT_2, size=11),
        ),
        margin=dict(l=10, r=10, t=14, b=10),
        hoverlabel=dict(
            bgcolor=SURF_2, font_color=TEXT, bordercolor=VIOLET,
            font=dict(family="JetBrains Mono", size=11),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=HAIR,
            font=dict(size=10, color=MUTED),
        ),
    )
    base.update(kw)
    return base


PLOT_CFG = {"displayModeBar": False}


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
    <div class="brand">
      <div class="brand-row">
        <div class="brand-logo">M</div>
        <div class="brand-text">
          <div class="name">Migros Group</div>
          <div class="sub">Location Intelligence</div>
        </div>
      </div>
      <div class="brand-meta">
        <div class="bm-cell"><div class="l">Brief</div><div class="v">N° 01</div></div>
        <div class="bm-cell"><div class="l">FY</div><div class="v">2022</div></div>
        <div class="bm-cell"><div class="l">Canton</div><div class="v">Geneva · CH</div></div>
        <div class="bm-cell"><div class="l">Build</div><div class="v">v 2.0</div></div>
      </div>
    </div>
    <div class="nav-label">Briefing</div>
    """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "nav",
        [
            "◆   Executive Summary",
            "◇   Population Pool",
            "▣   Composite Scoring",
            "◈   Regression Model",
            "⬡   Demographic Atlas",
            "◉   Geographic Map",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        f"""
    <div class="side-foot">
      <div class="row gold"><span>Status</span><span><span class="status-pulse"></span>Live</span></div>
      <div class="row"><span>Data</span><span>OCS · OFS</span></div>
      <div class="row"><span>Stores</span><span>OSM</span></div>
      <div class="row"><span>Model</span><span>OLS · CookD</span></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────────────────
try:
    stores_gdf, df_pop, df_p22 = load_data()
    bounds = load_boundaries()
    df_clean, joined = build_master(stores_gdf, df_pop, df_p22, bounds)
    top20, top5, top5_ols, champion, df_f, model = run_pipeline(df_clean)
except Exception as exc:
    st.error(f"Data load failed: {exc}")
    st.stop()


def render_page_bar(crumb_now: str, ref_code: str):
    st.markdown(
        f"""
    <div class="page-bar">
      <div class="crumb">Migros<span class="sep">/</span>Geneva<span class="sep">/</span>
        <span class="now">{crumb_now}</span></div>
      <div class="page-tag">REF<span class="ts"> · {ref_code}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        VINTAGE<span class="ts"> · 2022</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        CONFIDENTIAL<span class="ts"> · INTERNAL</span></div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_section(num: str, title: str, right: str, sub: str = ""):
    st.markdown(
        f"""
    <div class="section-bar">
      <div class="section-left">
        <span class="section-num">§ {num}</span>
        <span class="section-title">{title}</span>
      </div>
      <div class="section-right">{right}</div>
    </div>
    {'<div class="section-sub">' + sub + '</div>' if sub else ''}
    """,
        unsafe_allow_html=True,
    )


def render_colophon():
    st.markdown(
        f"""
    <div class="colophon">
      <div class="colo-cell"><b>Document</b>
        <span>Migros · Location Intelligence</span><br>
        <span>Briefing N° 01 · v2.0</span></div>
      <div class="colo-cell"><b>Scope</b>
        <span>Canton of Geneva · CH</span><br>
        <span>City excluded · {len(df_clean)} communes</span></div>
      <div class="colo-cell"><b>Method</b>
        <span>Composite scoring</span><br>
        <span>OLS · Cook's D trim</span></div>
      <div class="colo-cell"><b>Sources</b>
        <span>OCS · OFS · OSM</span><br>
        <span>Vintage 2022</span></div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# 00 · EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════
if page == "◆   Executive Summary":
    cn = champion["COMMUNE_NAME"]
    render_page_bar("Executive Summary", "MIG-GE-00")

    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-title">A single optimal site for the<br>
        next <b>Migros</b> branch in <span class="grad">Geneva</span>.</div>
      <div class="hero-lede">A three-stage quantitative funnel —
        <b>population pool</b>, <b>composite scoring</b>, and an
        <b>OLS opportunity-gap model</b> — synthesises demographic,
        retail-saturation and purchasing-power signals into one
        defensible recommendation.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    kpis = [
        ("◴", "Communes", f"{len(df_clean)}", "", "CANTON SCOPE", False),
        ("◍", "Stores", f"{int(df_clean['STORE_COUNT'].sum())}", "", "GEOLOCATED · OSM", False),
        ("◔", "Stage 1 Pool", "20", "", "BY POPULATION", False),
        ("◑", "Finalists", "5", "", "COMPOSITE SCORE", False),
        ("★", "Gap", f"+{champion['OPPORTUNITY']:.2f}", "", "▲ CHAMPION", True),
    ]
    cells = ""
    for i, (icn, label, val, unit, delta, grad) in enumerate(kpis):
        vc = "val grad" if grad else "val"
        dl = f'<span class="pos">▲</span> CHAMPION' if grad else delta
        cells += (
            f'<div class="kpi-cell" style="animation-delay:{i*0.06:.2f}s">'
            f'<div class="icn">{icn}</div>'
            f'<div class="label">{label}</div>'
            f'<div class="{vc}">{val}<span class="unit">{unit}</span></div>'
            f'<div class="delta">{dl}</div></div>'
        )
    st.markdown(f'<div class="kpi-row">{cells}</div>', unsafe_allow_html=True)

    metrics_html = "".join(
        [
            f'<div class="rec-metric"><div class="l">Population</div><div class="v">{int(champion["POPULATION"]):,}</div></div>',
            f'<div class="rec-metric"><div class="l">Active Stores</div><div class="v">{int(champion["STORE_COUNT"])}</div></div>',
            f'<div class="rec-metric"><div class="l">Predicted</div><div class="v">{champion["PREDICTED"]:.2f}</div></div>',
            f'<div class="rec-metric"><div class="l">Opportunity Gap</div><div class="v grad">+{champion["OPPORTUNITY"]:.2f}</div></div>',
            f'<div class="rec-metric"><div class="l">Median Income</div><div class="v">CHF {int(champion["proxy_purchasing_power_median_chf"]):,}</div></div>',
            f'<div class="rec-metric"><div class="l">Foreign Residents</div><div class="v">{champion["PCT_FOREIGNERS"]:.1f}%</div></div>',
        ]
    )
    st.markdown(
        f"""
    <div class="rec">
      <div class="rec-left">
        <div class="rec-eyebrow">
          <span class="tag">Recommendation</span>
          <span class="ref">REF · MIG-GE-CHAMP-01</span>
        </div>
        <div class="rec-name">{cn}</div>
        <div class="rec-place">Canton of Geneva · Switzerland</div>
        <div class="rec-summary">
          The model identifies <b>{cn}</b> as the single highest-priority site
          for a new Migros branch. The commune carries
          <b>{int(champion['POPULATION']):,}</b> residents,
          <b>{int(champion['STORE_COUNT'])}</b> active supermarkets, and a modelled
          shortfall of <b>+{champion['OPPORTUNITY']:.2f}</b> stores against expectation —
          the largest opportunity gap among the Stage-2 finalists.
        </div>
      </div>
      <div class="rec-right">{metrics_html}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    render_section("01", "Selection Funnel", "FOUR · CHECKPOINTS",
                   "Four sequential filters reduce the canton to one defensible site.")
    items = [
        (str(len(df_clean)), "Canton", "All communes — city excluded", "STAGE 0", False),
        ("20", "Population Pool", "Top by resident count", "STAGE 1", False),
        ("5", "Composite Cut", "Multi-factor weighted index", "STAGE 2", False),
        ("1", f"{cn}", "Highest OLS opportunity gap", "STAGE 3", True),
    ]
    cells = []
    for i, (num, ttl, dsc, step, win) in enumerate(items):
        arr = '<div class="arrow">→</div>' if i < len(items) - 1 else ""
        cls = "flow-cell win" if win else "flow-cell"
        cells.append(
            f'<div class="{cls}" style="animation-delay:{i*0.08:.2f}s">'
            f'<div class="step">{step}</div>'
            f'<div class="num">{num}</div><div class="ttl">{ttl}</div>'
            f'<div class="dsc">{dsc}</div>{arr}</div>'
        )
    st.markdown(f'<div class="flow">{"".join(cells)}</div>', unsafe_allow_html=True)

    render_section("02", "Methodology", "THREE · STAGES",
                   "Each stage filters on an independent signal — demand, affinity, and saturation.")
    m1, m2, m3 = st.columns(3, gap="medium")
    methods = [
        (m1, "01", "Population Shortlist",
         "Rank communes by resident count. Caps the search to the 20 communes where store volume is viable.",
         "VAR · POPULATION"),
        (m2, "02", "Composite Scoring",
         "Min–max weighted index over four dimensions: income, foreign share, working-age share, urban density.",
         "WEIGHTS · 35 / 25 / 20 / 20"),
        (m3, "03", "OLS Opportunity Gap",
         "Predicts store count from population and income. Cook's-D trimmed. Positive residual signals under-service.",
         "OUTLIERS · COOK'S D ≤ 4/n"),
    ]
    for i, (col, idx, ttl, body, foot) in enumerate(methods):
        with col:
            st.markdown(
                f"""
            <div class="method" style="animation-delay:{i*0.08:.2f}s">
              <div class="method-idx">§ {idx}</div>
              <div class="method-title">{ttl}</div>
              <div class="method-body">{body}</div>
              <div class="method-foot">{foot}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    render_colophon()


# ══════════════════════════════════════════════════════════════════════
# 01 · POPULATION POOL
# ══════════════════════════════════════════════════════════════════════
elif page == "◇   Population Pool":
    render_page_bar("Population Pool", "MIG-GE-01")

    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-title">Top <b>20 communes</b><br>by resident <span class="grad">population</span>.</div>
      <div class="hero-lede">The starting pool. City centre excluded —
        saturation and store ubiquity make it noise. Filters refine which
        subset of the pool to inspect.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="toolbar"><div class="toolbar-title">▸ Filter Options</div>',
        unsafe_allow_html=True,
    )
    f1, f2, f3 = st.columns(3)
    with f1:
        pop_range = st.slider(
            "Population Range",
            int(top20["POPULATION"].min()), int(top20["POPULATION"].max()),
            (int(top20["POPULATION"].min()), int(top20["POPULATION"].max())),
            step=500,
        )
    with f2:
        inc_f = st.selectbox("Income Bracket", ["All", "< CHF 70K", "CHF 70K – 80K", "> CHF 80K"])
    with f3:
        st_f = st.selectbox("Store Count", ["All", "No stores", "1–3 stores", "4+ stores"])
    st.markdown("</div>", unsafe_allow_html=True)

    df_s1 = top20[top20["POPULATION"].between(*pop_range)].copy()
    if inc_f == "< CHF 70K":
        df_s1 = df_s1[df_s1["proxy_purchasing_power_median_chf"] < 70000]
    elif inc_f == "CHF 70K – 80K":
        df_s1 = df_s1[df_s1["proxy_purchasing_power_median_chf"].between(70000, 80000)]
    elif inc_f == "> CHF 80K":
        df_s1 = df_s1[df_s1["proxy_purchasing_power_median_chf"] > 80000]
    if st_f == "No stores":
        df_s1 = df_s1[df_s1["STORE_COUNT"] == 0]
    elif st_f == "1–3 stores":
        df_s1 = df_s1[df_s1["STORE_COUNT"].between(1, 3)]
    elif st_f == "4+ stores":
        df_s1 = df_s1[df_s1["STORE_COUNT"] >= 4]

    st.markdown(
        f'<span class="chip"><span class="dot"></span>{len(df_s1)} / {len(top20)} COMMUNES</span>'
        f'<span class="chip">SORT · POPULATION ▼</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    if df_s1.empty:
        st.warning("No communes match the current filters.")
    else:
        # value-encoded gradient bars; winner gets magenta highlight
        vals = df_s1["POPULATION"].tolist()
        colors = list(vals)
        fig = go.Figure(
            go.Bar(
                x=df_s1["POPULATION"], y=df_s1["COMMUNE_NAME"], orientation="h",
                marker=dict(
                    color=colors, colorscale=SCALE, showscale=False,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                text=[f"{v:,}" for v in df_s1["POPULATION"]],
                textposition="outside",
                textfont=dict(color=TEXT_2, size=10),
                hovertemplate="<b>%{y}</b><br>Population · %{x:,}<extra></extra>",
            )
        )
        # highlight rank-1 with a glowing magenta outline marker
        if len(df_s1):
            w = df_s1.iloc[0]
            fig.add_trace(
                go.Bar(
                    x=[w["POPULATION"]], y=[w["COMMUNE_NAME"]], orientation="h",
                    marker=dict(color="rgba(0,0,0,0)",
                                line=dict(color=MAGENTA, width=2)),
                    hoverinfo="skip", showlegend=False,
                )
            )
        fig.update_layout(
            **base_layout(
                height=max(340, len(df_s1) * 32 + 60), barmode="overlay",
                xaxis=dict(gridcolor=HAIR, tickfont=dict(color=MUTED, size=9),
                           tickformat=",.0f", zerolinecolor=HAIR),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT, size=11),
                           autorange="reversed"),
                margin=dict(l=10, r=80, t=14, b=10), showlegend=False,
            )
        )
        st.markdown(
            '<div class="panel"><div class="panel-head">'
            '<div class="panel-title">Resident Population · by commune</div>'
            '<div class="panel-sub">PERSONS · 2022</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
        st.markdown("</div>", unsafe_allow_html=True)

        rows = ""
        for i, (_, r) in enumerate(df_s1.iterrows()):
            row_cls = "win" if i == 0 else ""
            rows += f"""<tr class="{row_cls}">
              <td class="mono">{int(r['RANK']):02d}</td>
              <td class="name">{r['COMMUNE_NAME']}</td>
              <td>{int(r['POPULATION']):,}</td>
              <td>{int(r['STORE_COUNT'])}</td>
              <td>{r['PCT_FOREIGNERS']:.1f}%</td>
              <td>CHF {int(r['proxy_purchasing_power_median_chf']):,}</td>
            </tr>"""
        st.markdown(
            f"""
        <div class="tbl-wrap" style="margin-top:18px;">
          <div class="tbl-head">
            <span class="badge grad">Stage 1</span>
            <span class="chip">{len(df_s1)} COMMUNES</span>
          </div>
          <table class="et">
            <thead><tr>
              <th>Rk</th><th>Commune</th><th>Population</th>
              <th>Stores</th><th>Foreign %</th><th>Median Income</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """,
            unsafe_allow_html=True,
        )

    render_colophon()


# ══════════════════════════════════════════════════════════════════════
# 02 · COMPOSITE SCORING
# ══════════════════════════════════════════════════════════════════════
elif page == "▣   Composite Scoring":
    render_page_bar("Composite Scoring", "MIG-GE-02")

    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-title">Four dimensions, one <span class="grad">composite</span> <b>score</b>.</div>
      <div class="hero-lede">A min–max weighted index over income,
        foreign-resident share, working-age share, and urban density.
        Adjust the weights — the ranking recomputes live.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="toolbar"><div class="toolbar-title">▸ Score Weights · target sum 100%</div>',
        unsafe_allow_html=True,
    )
    w1, w2, w3, w4 = st.columns(4)
    with w1: w_inc = st.slider("Income Weight %", 0, 60, 35, 5)
    with w2: w_for = st.slider("Foreign % Weight", 0, 60, 25, 5)
    with w3: w_age = st.slider("Working Age %", 0, 60, 20, 5)
    with w4: w_urb = st.slider("Urban Density %", 0, 60, 20, 5)
    total_w = w_inc + w_for + w_age + w_urb
    tc = POS if total_w == 100 else NEG
    st.markdown(
        f"""<div style="font-family:'JetBrains Mono',monospace;font-size:10px;
        margin-top:8px;color:{tc};letter-spacing:1.5px;font-weight:600;">
        WEIGHT SUM · {total_w}% {"· BALANCED ✓" if total_w == 100 else "· ADJUST →  100%"}</div>""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    def mm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0

    s2d = top5.copy()
    s2d["SC_INC"] = mm(s2d["proxy_purchasing_power_median_chf"])
    s2d["SC_FOR"] = mm(s2d["PCT_FOREIGNERS"])
    s2d["SC_AGE"] = mm(s2d["PCT_WORKING_AGE"])
    s2d["SC_URB"] = 1 - mm(s2d["PCT_SINGLE_FAMILY"])
    if total_w > 0:
        s2d["COMPOSITE_SCORE"] = (
            s2d["SC_INC"] * w_inc + s2d["SC_FOR"] * w_for
            + s2d["SC_AGE"] * w_age + s2d["SC_URB"] * w_urb
        ) / 100
        s2d = s2d.sort_values("COMPOSITE_SCORE", ascending=False).reset_index(drop=True)

    dims = [
        ("01", "Income",            w_inc, VIOLET),
        ("02", "Foreign Residents", w_for, CYAN),
        ("03", "Working Age",       w_age, MAGENTA),
        ("04", "Urban Density",     w_urb, EMERALD),
    ]
    cols = st.columns(4)
    for col, (idx, lbl, pct, hex_c) in zip(cols, dims):
        with col:
            st.markdown(
                f"""
            <div class="method" style="height:auto;padding:22px 24px 24px;">
              <div class="method-idx" style="color:{hex_c};-webkit-text-fill-color:{hex_c};
                   background:none;">§ {idx} · {lbl.upper()}</div>
              <div style="font-family:'Sora',sans-serif;font-size:46px;font-weight:300;
                   color:{hex_c};line-height:1;letter-spacing:-1.8px;
                   font-variant-numeric:tabular-nums;margin-top:8px;">
                   {pct}<span style="font-size:18px;color:{MUTED};">%</span></div>
              <div style="margin-top:14px;height:5px;border-radius:3px;
                   background:{HAIR};overflow:hidden;">
                <div style="width:{min(pct/60*100,100):.0f}%;height:100%;
                     background:{hex_c};border-radius:3px;"></div></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    c_a, c_b = st.columns(2, gap="medium")

    with c_a:
        fig1 = go.Figure(
            go.Bar(
                y=s2d["COMMUNE_NAME"], x=s2d["COMPOSITE_SCORE"], orientation="h",
                marker=dict(color=s2d["COMPOSITE_SCORE"], colorscale=SCALE, showscale=False),
                text=[f"{v:.3f}" for v in s2d["COMPOSITE_SCORE"]],
                textposition="outside", textfont=dict(color=TEXT_2, size=10),
                hovertemplate="<b>%{y}</b><br>Score · %{x:.4f}<extra></extra>",
            )
        )
        fig1.update_layout(
            **base_layout(
                height=300,
                xaxis=dict(gridcolor=HAIR, tickfont=dict(color=MUTED, size=9), range=[0, 1.15]),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT, size=11),
                           autorange="reversed"),
                margin=dict(l=10, r=70, t=14, b=10),
            )
        )
        st.markdown(
            '<div class="panel"><div class="panel-head">'
            '<div class="panel-title">Composite Score · Stage 2 finalists</div>'
            '<div class="panel-sub">INDEX · 0–1</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig1, use_container_width=True, config=PLOT_CFG)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_b:
        keys = ["SC_INC", "SC_FOR", "SC_AGE", "SC_URB"]
        labs = [f"Income · {w_inc}%", f"Foreign · {w_for}%",
                f"Age · {w_age}%", f"Urban · {w_urb}%"]
        pal4 = [VIOLET, CYAN, MAGENTA, EMERALD]
        ws = [w_inc / 100, w_for / 100, w_age / 100, w_urb / 100]
        fig2 = go.Figure()
        for d, lbl, c, w in zip(keys, labs, pal4, ws):
            fig2.add_trace(
                go.Bar(
                    name=lbl, y=s2d["COMMUNE_NAME"], x=s2d[d] * w,
                    orientation="h", marker_color=c,
                    marker_line_color=SURF, marker_line_width=1,
                    hovertemplate=f"<b>%{{y}}</b><br>{lbl} · %{{x:.3f}}<extra></extra>",
                )
            )
        fig2.update_layout(
            **base_layout(
                barmode="stack", height=300,
                xaxis=dict(gridcolor=HAIR, tickfont=dict(color=MUTED, size=9)),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT, size=11),
                           autorange="reversed"),
                margin=dict(l=10, r=10, t=14, b=32),
                legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                            font=dict(size=9.5, color=MUTED)),
            )
        )
        st.markdown(
            '<div class="panel"><div class="panel-head">'
            '<div class="panel-title">Score Breakdown · stacked contributions</div>'
            '<div class="panel-sub">WEIGHT × Z-NORM</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig2, use_container_width=True, config=PLOT_CFG)
        st.markdown("</div>", unsafe_allow_html=True)

    rows = ""
    for i, (_, r) in enumerate(s2d.iterrows()):
        row_cls = "win" if i == 0 else ""
        rows += f"""<tr class="{row_cls}">
          <td class="mono">{i+1:02d}</td>
          <td class="name">{r['COMMUNE_NAME']}</td>
          <td>CHF {int(r['proxy_purchasing_power_median_chf']):,}</td>
          <td>{r['PCT_FOREIGNERS']:.1f}%</td>
          <td>{r['PCT_WORKING_AGE']:.1f}%</td>
          <td>{r['PCT_SINGLE_FAMILY']:.1f}%</td>
          <td class="grad">{r['COMPOSITE_SCORE']:.4f}</td>
        </tr>"""
    st.markdown(
        f"""
    <div class="tbl-wrap" style="margin-top:14px;">
      <div class="tbl-head"><span class="badge grad">Stage 2</span>
      <span class="chip">LIVE · WEIGHTS</span></div>
      <table class="et">
        <thead><tr>
          <th>Rk</th><th>Commune</th><th>Income</th>
          <th>Foreign %</th><th>Working Age %</th><th>Single-Family %</th><th>Score</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """,
        unsafe_allow_html=True,
    )

    render_colophon()


# ══════════════════════════════════════════════════════════════════════
# 03 · REGRESSION MODEL
# ══════════════════════════════════════════════════════════════════════
elif page == "◈   Regression Model":
    render_page_bar("Regression Model", "MIG-GE-03")

    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-title">What the model <span class="grad">expected</span><br>to <b>find</b>.</div>
      <div class="hero-lede">An OLS regression predicts store count from
        population and purchasing power. The residual — predicted minus
        actual — surfaces markets that are demonstrably under-served.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="toolbar"><div class="toolbar-title">▸ View Options</div>',
        unsafe_allow_html=True,
    )
    o1, o2 = st.columns(2)
    with o1: show_neg = st.checkbox("Show negative opportunity gaps", value=True)
    with o2: sort_by = st.selectbox("Sort by", ["Opportunity Gap ↓", "Population ↓", "Predicted ↓"])
    st.markdown("</div>", unsafe_allow_html=True)

    tv = top5_ols.copy()
    if not show_neg:
        tv = tv[tv["OPPORTUNITY"] >= 0]
    if sort_by == "Population ↓":
        tv = tv.sort_values("POPULATION", ascending=False)
    elif sort_by == "Predicted ↓":
        tv = tv.sort_values("PREDICTED", ascending=False)

    c_a, c_b = st.columns(2, gap="medium")

    with c_a:
        bar_cols = [VIOLET if v >= 0 else ROSE for v in tv["OPPORTUNITY"]]
        if len(bar_cols):
            # champion (max gap, first row already sorted desc by default) → magenta
            top_idx = tv["OPPORTUNITY"].values.argmax()
            bar_cols[top_idx] = MAGENTA
        fig1 = go.Figure(
            go.Bar(
                y=tv["COMMUNE_NAME"], x=tv["OPPORTUNITY"], orientation="h",
                marker=dict(color=bar_cols, line=dict(color=SURF, width=1)),
                text=[f"+{v:.2f}" if v >= 0 else f"{v:.2f}" for v in tv["OPPORTUNITY"]],
                textposition="outside", textfont=dict(color=TEXT_2, size=10),
                hovertemplate="<b>%{y}</b><br>Gap · %{x:.2f}<extra></extra>",
            )
        )
        fig1.add_vline(x=0, line_color=FAINT, line_dash="dot", line_width=1)
        fig1.update_layout(
            **base_layout(
                height=300,
                xaxis=dict(gridcolor=HAIR, tickfont=dict(color=MUTED, size=9), zerolinecolor=FAINT),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT, size=11),
                           autorange="reversed"),
                margin=dict(l=10, r=70, t=14, b=10),
            )
        )
        st.markdown(
            '<div class="panel"><div class="panel-head">'
            '<div class="panel-title">Opportunity Gap · predicted − actual</div>'
            '<div class="panel-sub">RESIDUAL</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig1, use_container_width=True, config=PLOT_CFG)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_b:
        if not tv.empty:
            all_vals = list(tv["STORE_COUNT"]) + list(tv["PREDICTED"].dropna())
            lim = max(all_vals) + 1
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=[0, lim], y=[0, lim], mode="lines",
                    line=dict(color=FAINT, dash="dot", width=1),
                    name="Equilibrium", hoverinfo="skip",
                )
            )
            cmax = tv["OPPORTUNITY"].max()
            for i, (_, r) in enumerate(tv.iterrows()):
                c = MAGENTA if r["OPPORTUNITY"] == cmax else CYAN
                sz = 18 if r["OPPORTUNITY"] == cmax else 13
                fig2.add_trace(
                    go.Scatter(
                        x=[r["STORE_COUNT"]], y=[r["PREDICTED"]],
                        mode="markers+text",
                        marker=dict(color=c, size=sz, line=dict(color=BG, width=1.5),
                                    opacity=0.92),
                        text=[r["COMMUNE_NAME"]], textposition="top right",
                        textfont=dict(color=TEXT_2, size=10),
                        name=r["COMMUNE_NAME"],
                        hovertemplate=f"<b>{r['COMMUNE_NAME']}</b><br>"
                        f"Actual · {r['STORE_COUNT']}<br>"
                        f"Predicted · {r['PREDICTED']:.2f}<extra></extra>",
                    )
                )
            fig2.update_layout(
                **base_layout(
                    height=300,
                    xaxis=dict(title="ACTUAL", gridcolor=HAIR,
                               tickfont=dict(color=MUTED, size=9), range=[-0.2, lim],
                               title_font=dict(size=9, color=MUTED)),
                    yaxis=dict(title="PREDICTED", gridcolor=HAIR,
                               tickfont=dict(color=MUTED, size=9), range=[-0.2, lim],
                               title_font=dict(size=9, color=MUTED)),
                    margin=dict(l=10, r=10, t=14, b=10), showlegend=False,
                )
            )
            st.markdown(
                '<div class="panel"><div class="panel-head">'
                '<div class="panel-title">Actual vs Predicted</div>'
                '<div class="panel-sub">STORES · OLS</div></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig2, use_container_width=True, config=PLOT_CFG)
            st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("▸  OLS · Model Summary"):
        Xf2 = sm.add_constant(df_f[["POPULATION", "proxy_purchasing_power_median_chf"]])
        st.code(str(sm.OLS(df_f["STORE_COUNT"], Xf2).fit().summary()), language="text")

    rows = ""
    for i, (_, r) in enumerate(tv.iterrows()):
        gap = f"+{r['OPPORTUNITY']:.2f}" if r["OPPORTUNITY"] >= 0 else f"{r['OPPORTUNITY']:.2f}"
        row_cls = "win" if i == 0 else ""
        rows += f"""<tr class="{row_cls}">
          <td class="mono">{int(r['RANK']):02d}</td>
          <td class="name">{r['COMMUNE_NAME']}</td>
          <td>{int(r['POPULATION']):,}</td>
          <td>{int(r['STORE_COUNT'])}</td>
          <td>{r['PREDICTED']:.2f}</td>
          <td class="grad">{gap}</td>
          <td>{r['COMPOSITE_SCORE']:.4f}</td>
        </tr>"""
    st.markdown(
        f"""
    <div class="tbl-wrap" style="margin-top:14px;">
      <div class="tbl-head"><span class="badge grad">Stage 3</span>
      <span class="chip">OLS · RESIDUALS</span></div>
      <table class="et">
        <thead><tr>
          <th>Rk</th><th>Commune</th><th>Population</th><th>Stores</th>
          <th>Predicted</th><th>Gap</th><th>Score</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """,
        unsafe_allow_html=True,
    )

    cn = champion["COMMUNE_NAME"]
    metrics_html = "".join(
        [
            f'<div class="rec-metric"><div class="l">Population</div><div class="v">{int(champion["POPULATION"]):,}</div></div>',
            f'<div class="rec-metric"><div class="l">Active Stores</div><div class="v">{int(champion["STORE_COUNT"])}</div></div>',
            f'<div class="rec-metric"><div class="l">Predicted</div><div class="v">{champion["PREDICTED"]:.2f}</div></div>',
            f'<div class="rec-metric"><div class="l">Gap</div><div class="v grad">+{champion["OPPORTUNITY"]:.2f}</div></div>',
            f'<div class="rec-metric"><div class="l">Median Income</div><div class="v">CHF {int(champion["proxy_purchasing_power_median_chf"]):,}</div></div>',
            f'<div class="rec-metric"><div class="l">Foreign %</div><div class="v">{champion["PCT_FOREIGNERS"]:.1f}%</div></div>',
        ]
    )
    st.markdown(
        f"""
    <div class="rec" style="margin-top:28px;">
      <div class="rec-left">
        <div class="rec-eyebrow">
          <span class="tag">Stage 3 · Champion</span>
          <span class="ref">REF · MIG-GE-CHAMP-01</span>
        </div>
        <div class="rec-name">{cn}</div>
        <div class="rec-place">Canton of Geneva · Switzerland</div>
        <div class="rec-summary">
          Highest opportunity gap among finalists.
          Predicted store count of <b>{champion['PREDICTED']:.2f}</b> against
          <b>{int(champion['STORE_COUNT'])}</b> observed — a shortfall of
          <b>+{champion['OPPORTUNITY']:.2f}</b> stores.
        </div>
      </div>
      <div class="rec-right">{metrics_html}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    render_colophon()


# ══════════════════════════════════════════════════════════════════════
# 04 · DEMOGRAPHIC ATLAS
# ══════════════════════════════════════════════════════════════════════
elif page == "⬡   Demographic Atlas":
    render_page_bar("Demographic Atlas", "MIG-GE-04")

    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-title">Five lenses on store <span class="grad">demand</span>.</div>
      <div class="hero-lede">Each panel pairs a demographic axis with active
        store count across the canton. The accent marker pins the highlighted
        commune.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="toolbar"><div class="toolbar-title">▸ Options</div>',
        unsafe_allow_html=True,
    )
    d1, d2, d3 = st.columns(3)
    with d1: min_pop = st.slider("Min Population", 0, 20000, 0, 1000)
    with d2: hl_opt = st.selectbox("Highlight Commune", ["Champion"] + list(df_f["COMMUNE_NAME"].sort_values()))
    with d3: show_tl = st.checkbox("Show trend line", value=True)
    st.markdown("</div>", unsafe_allow_html=True)

    cn = champion["COMMUNE_NAME"]
    df_demo = df_f.copy()
    if min_pop > 0:
        df_demo = df_demo[df_demo["POPULATION"] >= min_pop]
    hl_name = cn if hl_opt == "Champion" else hl_opt

    panels = [
        ("POPULATION", "Population", CYAN),
        ("PCT_WORKING_AGE", "Working-Age %", VIOLET),
        ("PCT_SINGLE_FAMILY", "Single-Family Housing %", EMERALD),
        ("PCT_FOREIGNERS", "Foreign Residents %", AMBER),
        ("proxy_purchasing_power_median_chf", "Median Income (CHF)", INDIGO),
    ]

    def scatter_panel(xcol, xlabel, color, height=240):
        hl_data = df_demo[df_demo["COMMUNE_NAME"] == hl_name]
        fig = go.Figure()
        if show_tl:
            x_all = df_demo[xcol].values
            y_all = df_demo["STORE_COUNT"].values
            mask = ~np.isnan(x_all) & ~np.isnan(y_all)
            if mask.sum() > 2:
                c = np.polyfit(x_all[mask], y_all[mask], 1)
                x_line = np.linspace(x_all[mask].min(), x_all[mask].max(), 80)
                fig.add_trace(
                    go.Scatter(
                        x=x_line, y=np.polyval(c, x_line),
                        mode="lines", line=dict(color=VIOLET, width=1.6),
                        name="Trend", hoverinfo="skip",
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=df_demo[xcol], y=df_demo["STORE_COUNT"],
                mode="markers", name="Commune",
                marker=dict(color=color, size=8, opacity=0.72,
                            line=dict(color=BG, width=0.7)),
                customdata=df_demo[["COMMUNE_NAME"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>"
                + xlabel + " · %{x}<br>Stores · %{y}<extra></extra>",
            )
        )
        if not hl_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=hl_data[xcol], y=hl_data["STORE_COUNT"],
                    mode="markers", name=f"★ {hl_name}",
                    marker=dict(color=MAGENTA, size=17, symbol="diamond",
                                line=dict(color=BG, width=1.6)),
                    hovertemplate=f"<b>★ {hl_name}</b><br>{xlabel} · %{{x}}"
                    "<br>Stores · %{y}<extra></extra>",
                )
            )
        fig.update_layout(
            **base_layout(
                height=height,
                xaxis=dict(title=dict(text=xlabel.upper(), font=dict(size=9, color=MUTED)),
                           gridcolor=HAIR, tickfont=dict(color=MUTED, size=9)),
                yaxis=dict(title=dict(text="STORES", font=dict(size=9, color=MUTED)),
                           gridcolor=HAIR, tickfont=dict(color=MUTED, size=9)),
                margin=dict(l=10, r=10, t=14, b=10), showlegend=False,
            )
        )
        return fig

    row1 = st.columns(3, gap="medium")
    for (xc, xl, col_), col in zip(panels[:3], row1):
        with col:
            st.markdown(
                f'<div class="panel"><div class="panel-head">'
                f'<div class="panel-title">{xl} × Stores</div>'
                f'<div class="panel-sub">SCATTER</div></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(scatter_panel(xc, xl, col_), use_container_width=True, config=PLOT_CFG)
            st.markdown("</div>", unsafe_allow_html=True)

    row2 = st.columns(3, gap="medium")
    for (xc, xl, col_), col in zip(panels[3:], row2[:2]):
        with col:
            st.markdown(
                f'<div class="panel"><div class="panel-head">'
                f'<div class="panel-title">{xl} × Stores</div>'
                f'<div class="panel-sub">SCATTER</div></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(scatter_panel(xc, xl, col_), use_container_width=True, config=PLOT_CFG)
            st.markdown("</div>", unsafe_allow_html=True)

    with row2[2]:
        hl_row = df_demo[df_demo["COMMUNE_NAME"] == hl_name]
        if not hl_row.empty:
            h = hl_row.iloc[0]
            st.markdown(
                f"""
            <div class="panel" style="border:1px solid rgba(124,107,248,0.4);
                 box-shadow:0 0 26px rgba(124,107,248,0.14);padding:24px 24px;
                 height:240px;box-sizing:border-box;display:flex;
                 flex-direction:column;justify-content:center;">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
                <span class="badge grad">Highlight</span>
              </div>
              <div style="font-family:'Sora',sans-serif;font-size:24px;font-weight:600;
                   color:{TEXT};margin-bottom:18px;line-height:1.05;letter-spacing:-0.6px;">{hl_name}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;
                   color:{MUTED};line-height:1.95;letter-spacing:0.4px;
                   font-variant-numeric:tabular-nums;">
                POPULATION &nbsp;&nbsp; <span style="color:{TEXT};">{int(h['POPULATION']):,}</span><br>
                STORES &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{TEXT};">{int(h['STORE_COUNT'])}</span><br>
                PREDICTED &nbsp;&nbsp; <span style="color:{TEXT};">{h['PREDICTED']:.2f}</span><br>
                GAP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{CYAN_L};font-weight:700;">+{h['OPPORTUNITY']:.2f}</span><br>
                INCOME &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{TEXT};">CHF {int(h['proxy_purchasing_power_median_chf']):,}</span>
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    render_colophon()


# ══════════════════════════════════════════════════════════════════════
# 05 · GEOGRAPHIC MAP
# ══════════════════════════════════════════════════════════════════════
elif page == "◉   Geographic Map":
    render_page_bar("Geographic Map", "MIG-GE-05")

    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-title">The canton, at a <span class="grad">glance</span>.</div>
      <div class="hero-lede">Choropleth encodes the modelled opportunity gap.
        Circle pins mark existing supermarkets. The diamond pins the
        recommended target.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="toolbar"><div class="toolbar-title">▸ Map Options</div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    with m1: show_stores = st.checkbox("Show store pins", value=True)
    with m2: brand_filter = st.multiselect("Filter Brands", ["Coop", "Migros", "other"],
                                           default=["Coop", "Migros", "other"])
    with m3: map_zoom = st.slider("Zoom Level", 10, 14, 12)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("Building map…"):
        lat0 = bounds.geometry.centroid.y.mean()
        lon0 = bounds.geometry.centroid.x.mean()
        m = folium.Map(
            location=[lat0, lon0], zoom_start=map_zoom,
            tiles="cartodbdark_matter",
        )

        folium.Choropleth(
            geo_data=bounds, name="Opportunity Gap",
            data=df_f, columns=["COMMUNE_NAME", "OPPORTUNITY"],
            key_on="feature.properties.COMMUNE_NAME",
            fill_color="BuPu", fill_opacity=0.66,
            line_opacity=0.45, line_color="#070811",
            legend_name="Opportunity Gap (higher = under-served)",
            nan_fill_color="#151A2E", nan_fill_opacity=0.35,
        ).add_to(m)

        tooltip_gdf = bounds.merge(
            df_f[["COMMUNE_NAME", "OPPORTUNITY", "POPULATION", "STORE_COUNT"]],
            on="COMMUNE_NAME", how="left",
        )
        folium.GeoJson(
            tooltip_gdf,
            style_function=lambda _: {"fillOpacity": 0, "color": "#2A3050", "weight": 0.5},
            highlight_function=lambda _: {"fillColor": "#7C6BF8", "fillOpacity": 0.25,
                                          "color": "#22D3EE", "weight": 1.5},
            tooltip=folium.GeoJsonTooltip(
                fields=["COMMUNE_NAME", "POPULATION", "STORE_COUNT", "OPPORTUNITY"],
                aliases=["COMMUNE", "POPULATION", "STORES", "GAP"],
                style=(
                    "background:#0F1222;color:#F3F5FE;"
                    "font-family:JetBrains Mono,monospace;font-size:11px;"
                    "border:1px solid #7C6BF8;border-radius:8px;padding:10px;"
                ),
            ),
        ).add_to(m)

        cn = champion["COMMUNE_NAME"]
        cg = df_f[df_f["COMMUNE_NAME"] == cn]
        if not cg.empty:
            cx = cg.geometry.centroid.iloc[0].x
            cy = cg.geometry.centroid.iloc[0].y
            icon_html = (
                '<div style="background:linear-gradient(135deg,#7C6BF8 0%,#22D3EE 100%);'
                'border:2px solid #070811;width:40px;height:40px;border-radius:12px;'
                "transform:rotate(45deg);display:flex;align-items:center;justify-content:center;"
                'box-shadow:0 0 22px rgba(124,107,248,0.85),0 0 44px rgba(34,211,238,0.35);">'
                '<div style="transform:rotate(-45deg);font-family:JetBrains Mono;font-size:13px;'
                'color:#fff;font-weight:700;">★</div></div>'
            )
            popup_html = (
                f"<div style='font-family:JetBrains Mono,monospace;background:#0F1222;"
                f"color:#F3F5FE;padding:18px;border-left:3px solid #7C6BF8;border-radius:8px;"
                f"min-width:240px;font-variant-numeric:tabular-nums;'>"
                f"<b style='color:#5AE3F5;font-size:9px;letter-spacing:2px;'>★ CHAMPION TARGET</b><br><br>"
                f"<b style='font-family:Sora,sans-serif;font-size:22px;"
                f"color:#F3F5FE;font-weight:600;letter-spacing:-0.5px;'>{cn}</b><br>"
                f"<span style='color:#7A82AC;font-size:9.5px;letter-spacing:1.5px;'>CANTON OF GENEVA</span><br><br>"
                f"<span style='color:#7A82AC;'>POPULATION </span>{int(champion['POPULATION']):,}<br>"
                f"<span style='color:#7A82AC;'>STORES&nbsp;&nbsp;&nbsp;&nbsp;</span>{int(champion['STORE_COUNT'])}<br>"
                f"<span style='color:#7A82AC;'>PREDICTED&nbsp;</span>{champion['PREDICTED']:.2f}<br>"
                f"<b style='color:#5AE3F5;'>GAP&nbsp;&nbsp;&nbsp;&nbsp;+{champion['OPPORTUNITY']:.2f}</b><br><br>"
                f"<span style='color:#7A82AC;'>INCOME&nbsp;&nbsp;&nbsp;&nbsp;</span>CHF {int(champion['proxy_purchasing_power_median_chf']):,}"
                f"</div>"
            )
            folium.Marker(
                location=[cy, cx],
                popup=folium.Popup(popup_html, max_width=290),
                tooltip=f"★ CHAMPION · {cn}",
                icon=folium.DivIcon(html=icon_html, icon_size=(44, 44), icon_anchor=(22, 22)),
            ).add_to(m)

        if show_stores:
            brand_colors = {"Coop": CYAN, "Migros": MAGENTA, "other": MUTED}
            for _, row in joined.iterrows():
                brand = row.get("brand_category", "other")
                if brand not in brand_filter:
                    continue
                if row["COMMUNE_NAME"] in df_f["COMMUNE_NAME"].values:
                    bc = brand_colors.get(brand, brand_colors["other"])
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=4.5, color=BG, fill=True, fill_color=bc,
                        fill_opacity=0.9, weight=1.5,
                        tooltip=f"{brand} · {row['COMMUNE_NAME']}",
                        popup=folium.Popup(
                            f"<div style='font-family:JetBrains Mono,monospace;"
                            f"font-size:11px;background:#0F1222;border-radius:8px;"
                            f"color:#F3F5FE;padding:10px;"
                            f"border-left:3px solid {bc};'>"
                            f"<b style='color:{bc};'>{brand}</b><br>"
                            f"TYPE · {row.get('shop', '?')}<br>"
                            f"COMMUNE · {row['COMMUNE_NAME']}</div>",
                            max_width=200,
                        ),
                    ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

    # Legend chips
    st.markdown(
        f"""
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">
      <span class="chip"><span class="dot" style="background:{MAGENTA};"></span>★ Champion target</span>
      <span class="chip"><span class="dot" style="background:{CYAN};"></span>Coop</span>
      <span class="chip"><span class="dot" style="background:{MAGENTA};"></span>Migros</span>
      <span class="chip"><span class="dot" style="background:{MUTED};"></span>Other brands</span>
      <span class="chip">CHOROPLETH · OPPORTUNITY GAP</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="panel" style="padding:12px;">', unsafe_allow_html=True)
    st_folium(m, height=640, use_container_width=True, returned_objects=[])
    st.markdown("</div>", unsafe_allow_html=True)

    render_colophon()
