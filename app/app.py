"""
Migros · Location Intelligence — institutional dark dashboard.

Formal executive briefing aesthetic: matte black surfaces, champagne accent,
tabular numerics, hairline grid, monospace micro-typography.
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
# DESIGN TOKENS — matte black + champagne accent
# ──────────────────────────────────────────────────────────────────────
BG       = "#07070A"   # page
SURF     = "#0E0E12"   # primary surface
SURF_2   = "#14141A"   # raised surface
SURF_3   = "#1A1A22"   # hover / active
HAIR     = "#1C1C24"   # hairline divider
BORDER   = "#262630"   # standard border
BORDER_H = "#3A3A48"   # hover border
TEXT     = "#F5F5F7"   # primary text
TEXT_2   = "#C8C8CE"   # secondary text
MUTED    = "#86868B"   # tertiary / labels
FAINT    = "#4A4A52"   # quaternary
GOLD     = "#C9A961"   # champagne accent
GOLD_L   = "#E0C887"   # hover gold
GOLD_D   = "#8C7438"   # deep gold
POS      = "#5B9D7A"   # positive metric
NEG      = "#C26D6D"   # negative metric
DATA_BLUE = "#6B8BB5"
DATA_VIOLET = "#8B7BB0"

# Chart palette — restrained, sophisticated
PAL = [GOLD, DATA_BLUE, "#9A8A6A", DATA_VIOLET, "#7E7E88", "#A88B6F"]


# ──────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@200;300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter Tight', system-ui, sans-serif;
    color: {TEXT};
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    font-feature-settings: 'ss01', 'cv11', 'tnum';
}}
.stApp {{ background: {BG}; }}
.block-container {{ padding: 1.4rem 2.4rem 5rem; max-width: 1500px; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

/* ── SIDEBAR ────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: #050507;
    border-right: 1px solid {HAIR};
}}
[data-testid="stSidebar"] * {{ color: {TEXT}; }}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 0; }}

.brand {{
    padding: 30px 22px 24px;
    border-bottom: 1px solid {HAIR};
    position: relative;
}}
.brand::after {{
    content: '';
    position: absolute; left: 22px; bottom: -1px;
    width: 28px; height: 1px; background: {GOLD};
}}
.brand-row {{ display: flex; align-items: center; gap: 12px; }}
.brand-logo {{
    width: 36px; height: 36px;
    background: linear-gradient(135deg, {GOLD} 0%, {GOLD_D} 100%);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Inter Tight', sans-serif;
    font-size: 17px; font-weight: 700; color: {BG};
    letter-spacing: -0.5px;
    box-shadow: 0 1px 0 rgba(224,200,135,0.25) inset, 0 6px 18px rgba(201,169,97,0.18);
}}
.brand-text .name {{
    font-size: 14.5px; font-weight: 600; letter-spacing: -0.2px;
    color: {TEXT}; line-height: 1;
}}
.brand-text .sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px; color: {MUTED}; letter-spacing: 2.2px;
    text-transform: uppercase; margin-top: 5px; font-weight: 500;
}}
.brand-meta {{
    margin-top: 22px; display: grid; grid-template-columns: 1fr 1fr; gap: 14px 16px;
}}
.bm-cell .l {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; color: {FAINT}; letter-spacing: 1.6px;
    text-transform: uppercase; margin-bottom: 4px;
}}
.bm-cell .v {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: {TEXT_2}; font-weight: 500; letter-spacing: 0.4px;
}}

.nav-label {{
    padding: 24px 22px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px; letter-spacing: 2.6px; color: {FAINT};
    text-transform: uppercase; font-weight: 500;
}}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 1px; padding: 0 10px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    width: 100%; padding: 10px 14px !important; margin: 0 !important;
    border-radius: 2px; border-left: 1px solid transparent;
    cursor: pointer; font-size: 13px; font-weight: 400;
    color: {MUTED}; transition: all 0.12s ease;
    letter-spacing: 0.1px;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background: {SURF}; color: {TEXT};
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background: {SURF};
    border-left-color: {GOLD};
    color: {TEXT}; font-weight: 500;
}}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}

.side-foot {{
    margin: 26px 22px 22px;
    padding-top: 20px;
    border-top: 1px solid {HAIR};
}}
.side-foot .row {{
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: {MUTED}; letter-spacing: 1.2px;
    padding: 5px 0;
}}
.side-foot .row span:last-child {{ color: {TEXT_2}; }}
.side-foot .row.gold span:last-child {{ color: {GOLD}; }}
.status-pulse {{
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: {POS}; margin-right: 7px;
    box-shadow: 0 0 0 0 rgba(91,157,122,0.6); animation: pulse 2.4s infinite;
    vertical-align: middle;
}}
@keyframes pulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(91,157,122,0.5); }}
    70% {{ box-shadow: 0 0 0 6px rgba(91,157,122,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(91,157,122,0); }}
}}

/* ── PAGE HEADER ───────────────────────────────────────── */
.page-bar {{
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 14px; border-bottom: 1px solid {HAIR};
    margin-bottom: 26px;
}}
.crumb {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: {MUTED}; letter-spacing: 1.8px;
    text-transform: uppercase; font-weight: 500;
}}
.crumb .sep {{ color: {FAINT}; margin: 0 10px; }}
.crumb .now {{ color: {GOLD}; }}
.page-tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; color: {MUTED}; letter-spacing: 1.5px;
}}
.page-tag .ts {{ color: {TEXT_2}; }}

.hero {{
    display: grid; grid-template-columns: 1.7fr 1fr;
    gap: 48px; align-items: end; margin-bottom: 32px;
}}
.hero-title {{
    font-size: 52px; font-weight: 300; line-height: 1.02;
    letter-spacing: -1.8px; color: {TEXT};
}}
.hero-title b {{ font-weight: 600; color: {TEXT}; }}
.hero-title .gold {{ color: {GOLD}; font-weight: 500; }}
.hero-lede {{
    font-size: 13.5px; color: {TEXT_2};
    line-height: 1.7; font-weight: 300;
    border-left: 1px solid {GOLD};
    padding-left: 18px;
}}
.hero-lede b {{ color: {TEXT}; font-weight: 500; }}

.section-bar {{
    display: flex; align-items: baseline; justify-content: space-between;
    margin: 44px 0 4px;
    padding-bottom: 12px; border-bottom: 1px solid {HAIR};
}}
.section-left {{ display: flex; align-items: baseline; gap: 18px; }}
.section-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: {GOLD}; letter-spacing: 2px; font-weight: 600;
}}
.section-title {{
    font-size: 22px; font-weight: 500; letter-spacing: -0.5px; color: {TEXT};
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
    background: {SURF};
    border: 1px solid {HAIR};
    margin: 30px 0 6px;
}}
.kpi-cell {{
    padding: 24px 24px 26px;
    border-right: 1px solid {HAIR};
    position: relative;
}}
.kpi-cell:last-child {{ border-right: none; }}
.kpi-cell::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: transparent; transition: background 0.2s;
}}
.kpi-cell:hover::before {{ background: {GOLD}; }}
.kpi-cell .label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px; letter-spacing: 1.8px;
    text-transform: uppercase; color: {MUTED};
    margin-bottom: 16px; font-weight: 500;
}}
.kpi-cell .val {{
    font-size: 36px; line-height: 1; letter-spacing: -1.4px;
    color: {TEXT}; font-weight: 300;
    font-variant-numeric: tabular-nums;
}}
.kpi-cell .val.gold {{ color: {GOLD}; font-weight: 400; }}
.kpi-cell .unit {{
    font-size: 12px; color: {MUTED}; margin-left: 4px; font-weight: 400;
}}
.kpi-cell .delta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: {FAINT}; margin-top: 12px;
    letter-spacing: 1.2px; text-transform: uppercase;
}}
.kpi-cell .delta .pos {{ color: {POS}; font-weight: 600; }}
.kpi-cell .delta .neg {{ color: {NEG}; font-weight: 600; }}

/* ── RECOMMENDATION CARD (formal) ──────────────────────── */
.rec {{
    background: {SURF};
    border: 1px solid {HAIR};
    margin: 28px 0 24px;
    display: grid; grid-template-columns: 1.6fr 1fr;
}}
.rec-left {{
    padding: 36px 40px 38px;
    border-right: 1px solid {HAIR};
    position: relative;
}}
.rec-left::before {{
    content: ''; position: absolute; top: 0; left: 0; bottom: 0;
    width: 3px; background: {GOLD};
}}
.rec-eyebrow {{
    display: flex; align-items: center; gap: 10px; margin-bottom: 22px;
}}
.rec-eyebrow .tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 2.4px;
    text-transform: uppercase; color: {BG};
    background: {GOLD}; padding: 4px 9px; font-weight: 600;
}}
.rec-eyebrow .ref {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: {MUTED}; letter-spacing: 1.5px;
}}
.rec-name {{
    font-size: 56px; font-weight: 300; letter-spacing: -2.2px;
    color: {TEXT}; line-height: 1; margin-bottom: 12px;
}}
.rec-place {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: {MUTED}; letter-spacing: 1.8px;
    text-transform: uppercase;
}}
.rec-summary {{
    margin-top: 26px; padding-top: 22px;
    border-top: 1px solid {HAIR};
    font-size: 13px; color: {TEXT_2}; line-height: 1.7; font-weight: 300;
}}
.rec-summary b {{ color: {TEXT}; font-weight: 500; }}
.rec-right {{ padding: 0; }}
.rec-metric {{
    padding: 18px 28px;
    border-bottom: 1px solid {HAIR};
    display: flex; justify-content: space-between; align-items: baseline;
}}
.rec-metric:last-child {{ border-bottom: none; }}
.rec-metric .l {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 1.6px;
    text-transform: uppercase; color: {MUTED};
}}
.rec-metric .v {{
    font-size: 18px; font-weight: 400; color: {TEXT};
    font-variant-numeric: tabular-nums; letter-spacing: -0.3px;
}}
.rec-metric .v.gold {{ color: {GOLD}; font-weight: 500; }}

/* ── PROCESS FLOW (formal) ─────────────────────────────── */
.flow {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    background: {SURF};
    border: 1px solid {HAIR};
    margin: 6px 0 14px;
}}
.flow-cell {{
    padding: 26px 28px 28px;
    border-right: 1px solid {HAIR};
    position: relative;
}}
.flow-cell:last-child {{ border-right: none; }}
.flow-cell.win {{ background: linear-gradient(180deg, rgba(201,169,97,0.05) 0%, transparent 100%); }}
.flow-cell .step {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px; letter-spacing: 2px; color: {MUTED};
    text-transform: uppercase; margin-bottom: 14px; font-weight: 500;
}}
.flow-cell.win .step {{ color: {GOLD}; }}
.flow-cell .num {{
    font-size: 44px; line-height: 1; font-weight: 300;
    letter-spacing: -1.8px; color: {TEXT};
    font-variant-numeric: tabular-nums;
}}
.flow-cell.win .num {{ color: {GOLD}; font-weight: 400; }}
.flow-cell .ttl {{
    font-size: 13px; font-weight: 500; color: {TEXT};
    margin: 12px 0 4px; letter-spacing: -0.2px;
}}
.flow-cell .dsc {{
    font-size: 11.5px; color: {MUTED};
    line-height: 1.55; font-weight: 300;
}}
.flow-cell .arrow {{
    position: absolute; right: -1px; top: 36px;
    z-index: 4; font-family: 'JetBrains Mono', monospace;
    color: {FAINT}; font-size: 12px; background: {SURF};
    padding: 0 4px;
}}

/* ── PANEL ─────────────────────────────────────────────── */
.panel {{
    background: {SURF};
    border: 1px solid {HAIR};
    padding: 18px 22px 20px;
    margin-bottom: 16px;
}}
.panel-head {{
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 14px; margin-bottom: 14px;
    border-bottom: 1px solid {HAIR};
}}
.panel-title {{
    font-size: 13px; font-weight: 600; color: {TEXT};
    letter-spacing: -0.1px;
}}
.panel-sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: {MUTED}; letter-spacing: 1.4px;
    text-transform: uppercase;
}}

/* ── METHOD CARDS ──────────────────────────────────────── */
.method {{
    background: {SURF}; border: 1px solid {HAIR};
    padding: 26px 28px; height: 220px;
    position: relative; transition: border-color 0.2s;
    display: flex; flex-direction: column;
}}
.method:hover {{ border-color: {BORDER_H}; }}
.method-idx {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; color: {GOLD}; letter-spacing: 2px;
    font-weight: 600; margin-bottom: 18px;
}}
.method-title {{
    font-size: 16px; font-weight: 500; color: {TEXT};
    letter-spacing: -0.3px; margin-bottom: 10px;
}}
.method-body {{
    font-size: 12.5px; color: {MUTED};
    line-height: 1.6; font-weight: 300; flex: 1;
}}
.method-foot {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: {GOLD_D}; letter-spacing: 1.4px;
    padding-top: 14px; border-top: 1px solid {HAIR};
    text-transform: uppercase;
}}

/* ── TOOLBAR ───────────────────────────────────────────── */
.toolbar {{
    background: {SURF};
    border: 1px solid {HAIR};
    padding: 16px 22px 4px;
    margin-bottom: 22px;
}}
.toolbar-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 2.2px;
    text-transform: uppercase; color: {MUTED};
    margin-bottom: 10px; font-weight: 600;
}}

/* ── TABLE ─────────────────────────────────────────────── */
.tbl-wrap {{
    background: {SURF};
    border: 1px solid {HAIR};
    overflow: hidden;
}}
.tbl-head {{
    padding: 14px 22px;
    border-bottom: 1px solid {HAIR};
    display: flex; align-items: center; gap: 12px;
}}
table.et {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
table.et th {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 1.8px;
    text-transform: uppercase; color: {MUTED};
    padding: 11px 18px; border-bottom: 1px solid {HAIR};
    text-align: left; font-weight: 500; background: #0A0A0E;
}}
table.et td {{
    padding: 12px 18px;
    border-bottom: 1px solid {HAIR};
    color: {TEXT_2};
    font-variant-numeric: tabular-nums;
}}
table.et tr:last-child td {{ border-bottom: none; }}
table.et tr:hover td {{ background: #11111A; transition: background 0.12s; }}
table.et td.mono {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: {FAINT}; font-weight: 500;
}}
table.et td.name {{ color: {TEXT}; font-weight: 500; letter-spacing: -0.1px; }}
table.et td.gold {{ color: {GOLD}; font-weight: 600; }}
table.et tr.win td {{
    background: rgba(201,169,97,0.04) !important;
    border-left: 2px solid {GOLD};
}}
table.et tr.win td:first-child {{ padding-left: 16px; }}

/* ── BADGES & CHIPS ────────────────────────────────────── */
.badge {{
    display: inline-flex; align-items: center;
    background: transparent; color: {TEXT_2};
    border: 1px solid {BORDER};
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 600; letter-spacing: 1.6px;
    padding: 4px 10px; text-transform: uppercase;
}}
.badge.gold {{ background: {GOLD}; color: {BG}; border-color: {GOLD}; }}
.chip {{
    display: inline-flex; align-items: center;
    background: #0A0A0E; border: 1px solid {HAIR};
    color: {MUTED};
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; padding: 4px 10px;
    letter-spacing: 1px; margin-right: 6px;
}}
.chip .dot {{
    display: inline-block; width: 5px; height: 5px;
    background: {GOLD}; margin-right: 7px;
}}

/* ── INPUTS ────────────────────────────────────────────── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div {{
    background: #0A0A0E !important;
    border: 1px solid {BORDER} !important;
    border-radius: 0 !important; color: {TEXT} !important;
}}
label p, .stSlider label p, .stCheckbox label p {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important; letter-spacing: 1.5px !important;
    text-transform: uppercase !important; color: {MUTED} !important;
    font-weight: 600 !important;
}}
.stSlider [data-baseweb="slider"] > div > div > div {{
    background: {GOLD} !important;
}}
.stSlider [data-baseweb="slider"] > div > div {{ background: {BORDER} !important; }}
.stCheckbox [role="checkbox"][aria-checked="true"] {{
    background: {GOLD} !important; border-color: {GOLD} !important;
}}
.stButton > button {{
    background: transparent !important; color: {GOLD} !important;
    border: 1px solid {GOLD_D} !important; border-radius: 0 !important;
    font-family: 'JetBrains Mono', sans-serif !important;
    font-weight: 600 !important; font-size: 11px !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
}}
.stButton > button:hover {{
    background: {GOLD} !important; color: {BG} !important;
    border-color: {GOLD} !important;
}}
div[data-testid="stExpander"] {{
    background: {SURF}; border: 1px solid {HAIR};
    border-radius: 0;
}}
div[data-testid="stExpander"] summary {{ color: {TEXT_2} !important; }}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; }}
::-webkit-scrollbar-thumb:hover {{ background: {BORDER_H}; }}

/* ── COLOPHON ──────────────────────────────────────────── */
.colophon {{
    margin-top: 50px; padding-top: 22px;
    border-top: 1px solid {HAIR};
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 30px;
}}
.colo-cell {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 1.4px; color: {FAINT};
    text-transform: uppercase; line-height: 1.8;
}}
.colo-cell b {{ color: {GOLD}; font-weight: 600; display: block; margin-bottom: 6px; letter-spacing: 1.8px; }}
.colo-cell span {{ color: {TEXT_2}; }}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────
# DATA
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
        paper_bgcolor=SURF, plot_bgcolor=SURF,
        font=dict(family="Inter Tight", color=TEXT, size=11),
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
            bgcolor="#000000", font_color=TEXT, bordercolor=GOLD,
            font=dict(family="JetBrains Mono", size=11),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=HAIR,
            font=dict(size=10, color=MUTED),
        ),
    )
    base.update(kw)
    return base


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
        <div class="bm-cell"><div class="l">Build</div><div class="v">v 1.0</div></div>
      </div>
    </div>
    <div class="nav-label">Briefing</div>
    """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "nav",
        [
            "00  ·  Executive Summary",
            "01  ·  Population Pool",
            "02  ·  Composite Scoring",
            "03  ·  Regression Model",
            "04  ·  Demographic Atlas",
            "05  ·  Geographic Map",
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
        <span>Briefing N° 01 · v1.0</span></div>
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
if page == "00  ·  Executive Summary":
    cn = champion["COMMUNE_NAME"]
    render_page_bar("Executive Summary", "MIG-GE-00")

    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-title">A single optimal site for the<br>
        next <b>Migros</b> branch in <span class="gold">Geneva</span>.</div>
      <div class="hero-lede">A three-stage quantitative funnel —
        <b>population pool</b>, <b>composite scoring</b>, and an
        <b>OLS opportunity-gap model</b> — synthesises demographic,
        retail-saturation and purchasing-power signals into one
        defensible recommendation.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # KPIs
    st.markdown(
        f"""
    <div class="kpi-row">
      <div class="kpi-cell">
        <div class="label">Communes</div>
        <div class="val">{len(df_clean)}</div>
        <div class="delta">CANTON SCOPE</div>
      </div>
      <div class="kpi-cell">
        <div class="label">Stores</div>
        <div class="val">{int(df_clean['STORE_COUNT'].sum())}</div>
        <div class="delta">GEOLOCATED · OSM</div>
      </div>
      <div class="kpi-cell">
        <div class="label">Stage 1 Pool</div>
        <div class="val">20</div>
        <div class="delta">BY POPULATION</div>
      </div>
      <div class="kpi-cell">
        <div class="label">Finalists</div>
        <div class="val">5</div>
        <div class="delta">COMPOSITE SCORE</div>
      </div>
      <div class="kpi-cell">
        <div class="label">Gap</div>
        <div class="val gold">+{champion['OPPORTUNITY']:.2f}</div>
        <div class="delta"><span class="pos">▲</span> CHAMPION</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Recommendation card
    metrics_html = "".join(
        [
            f'<div class="rec-metric"><div class="l">Population</div><div class="v">{int(champion["POPULATION"]):,}</div></div>',
            f'<div class="rec-metric"><div class="l">Active Stores</div><div class="v">{int(champion["STORE_COUNT"])}</div></div>',
            f'<div class="rec-metric"><div class="l">Predicted</div><div class="v">{champion["PREDICTED"]:.2f}</div></div>',
            f'<div class="rec-metric"><div class="l">Opportunity Gap</div><div class="v gold">+{champion["OPPORTUNITY"]:.2f}</div></div>',
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

    # Process flow
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
        arr = '<div class="arrow">›</div>' if i < len(items) - 1 else ""
        cls = "flow-cell win" if win else "flow-cell"
        cells.append(
            f'<div class="{cls}"><div class="step">{step}</div>'
            f'<div class="num">{num}</div><div class="ttl">{ttl}</div>'
            f'<div class="dsc">{dsc}</div>{arr}</div>'
        )
    st.markdown(f'<div class="flow">{"".join(cells)}</div>', unsafe_allow_html=True)

    # Methodology
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
    for col, idx, ttl, body, foot in methods:
        with col:
            st.markdown(
                f"""
            <div class="method">
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
elif page == "01  ·  Population Pool":
    render_page_bar("Population Pool", "MIG-GE-01")

    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-title">Top <b>20 communes</b><br>by resident <span class="gold">population</span>.</div>
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
        bar_colors = [GOLD if i == 0 else "#3D4452" for i in range(len(df_s1))]
        fig = go.Figure(
            go.Bar(
                x=df_s1["POPULATION"], y=df_s1["COMMUNE_NAME"], orientation="h",
                marker_color=bar_colors,
                marker_line_color=SURF, marker_line_width=1,
                text=[f"{v:,}" for v in df_s1["POPULATION"]],
                textposition="outside",
                textfont=dict(color=TEXT_2, size=10),
                hovertemplate="<b>%{y}</b><br>Population · %{x:,}<extra></extra>",
            )
        )
        fig.update_layout(
            **base_layout(
                height=max(340, len(df_s1) * 32 + 60),
                xaxis=dict(gridcolor=HAIR, tickfont=dict(color=MUTED, size=9),
                           tickformat=",.0f", zerolinecolor=HAIR),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT, size=11),
                           autorange="reversed"),
                margin=dict(l=10, r=80, t=14, b=10),
            )
        )
        st.markdown(
            '<div class="panel"><div class="panel-head">'
            '<div class="panel-title">Resident Population · by commune</div>'
            '<div class="panel-sub">PERSONS · 2022</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)
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
            <span class="badge gold">Stage 1</span>
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
elif page == "02  ·  Composite Scoring":
    render_page_bar("Composite Scoring", "MIG-GE-02")

    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-title">Four dimensions, one <span class="gold">composite</span> <b>score</b>.</div>
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

    # Weight cards
    dims = [
        ("01", "Income",            w_inc, GOLD),
        ("02", "Foreign Residents", w_for, DATA_BLUE),
        ("03", "Working Age",       w_age, "#9A8A6A"),
        ("04", "Urban Density",     w_urb, DATA_VIOLET),
    ]
    cols = st.columns(4)
    for col, (idx, lbl, pct, hex_c) in zip(cols, dims):
        with col:
            st.markdown(
                f"""
            <div class="method" style="height:auto;padding:22px 24px 24px;">
              <div class="method-idx" style="color:{hex_c};">§ {idx} · {lbl.upper()}</div>
              <div style="font-size:44px;font-weight:300;color:{hex_c};
                   line-height:1;letter-spacing:-1.8px;font-variant-numeric:tabular-nums;
                   margin-top:8px;">{pct}<span style="font-size:18px;color:{MUTED};">%</span></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Charts
    pal5 = [GOLD, DATA_BLUE, "#9A8A6A", DATA_VIOLET, "#A88B6F"]
    c_a, c_b = st.columns(2, gap="medium")

    with c_a:
        fig1 = go.Figure(
            go.Bar(
                y=s2d["COMMUNE_NAME"], x=s2d["COMPOSITE_SCORE"], orientation="h",
                marker_color=pal5[: len(s2d)],
                marker_line_color=SURF, marker_line_width=1,
                text=[f"{v:.3f}" for v in s2d["COMPOSITE_SCORE"]],
                textposition="outside",
                textfont=dict(color=TEXT_2, size=10),
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
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_b:
        keys = ["SC_INC", "SC_FOR", "SC_AGE", "SC_URB"]
        labs = [f"Income · {w_inc}%", f"Foreign · {w_for}%",
                f"Age · {w_age}%", f"Urban · {w_urb}%"]
        ws = [w_inc / 100, w_for / 100, w_age / 100, w_urb / 100]
        fig2 = go.Figure()
        for d, lbl, c, w in zip(keys, labs, pal5, ws):
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
                            xanchor="left", x=0,
                            bgcolor="rgba(0,0,0,0)",
                            font=dict(size=9.5, color=MUTED)),
            )
        )
        st.markdown(
            '<div class="panel"><div class="panel-head">'
            '<div class="panel-title">Score Breakdown · stacked contributions</div>'
            '<div class="panel-sub">WEIGHT × Z-NORM</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig2, use_container_width=True)
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
          <td class="gold">{r['COMPOSITE_SCORE']:.4f}</td>
        </tr>"""
    st.markdown(
        f"""
    <div class="tbl-wrap" style="margin-top:14px;">
      <div class="tbl-head"><span class="badge gold">Stage 2</span>
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
elif page == "03  ·  Regression Model":
    render_page_bar("Regression Model", "MIG-GE-03")

    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-title">What the model <span class="gold">expected</span><br>to <b>find</b>.</div>
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

    bar_cols = [GOLD if i == 0 else "#3D4452" for i in range(len(tv))]
    c_a, c_b = st.columns(2, gap="medium")

    with c_a:
        fig1 = go.Figure(
            go.Bar(
                y=tv["COMMUNE_NAME"], x=tv["OPPORTUNITY"], orientation="h",
                marker_color=bar_cols, marker_line_color=SURF, marker_line_width=1,
                text=[f"+{v:.2f}" if v >= 0 else f"{v:.2f}" for v in tv["OPPORTUNITY"]],
                textposition="outside",
                textfont=dict(color=TEXT_2, size=10),
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
        st.plotly_chart(fig1, use_container_width=True)
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
            for i, (_, r) in enumerate(tv.iterrows()):
                c = GOLD if i == 0 else DATA_BLUE
                fig2.add_trace(
                    go.Scatter(
                        x=[r["STORE_COUNT"]], y=[r["PREDICTED"]],
                        mode="markers+text",
                        marker=dict(color=c, size=14, line=dict(color=BG, width=1.5)),
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
                    margin=dict(l=10, r=10, t=14, b=10),
                    showlegend=False,
                )
            )
            st.markdown(
                '<div class="panel"><div class="panel-head">'
                '<div class="panel-title">Actual vs Predicted</div>'
                '<div class="panel-sub">STORES · OLS</div></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig2, use_container_width=True)
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
          <td class="gold">{gap}</td>
          <td>{r['COMPOSITE_SCORE']:.4f}</td>
        </tr>"""
    st.markdown(
        f"""
    <div class="tbl-wrap" style="margin-top:14px;">
      <div class="tbl-head"><span class="badge gold">Stage 3</span>
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

    # Final formal callout
    cn = champion["COMMUNE_NAME"]
    metrics_html = "".join(
        [
            f'<div class="rec-metric"><div class="l">Population</div><div class="v">{int(champion["POPULATION"]):,}</div></div>',
            f'<div class="rec-metric"><div class="l">Active Stores</div><div class="v">{int(champion["STORE_COUNT"])}</div></div>',
            f'<div class="rec-metric"><div class="l">Predicted</div><div class="v">{champion["PREDICTED"]:.2f}</div></div>',
            f'<div class="rec-metric"><div class="l">Gap</div><div class="v gold">+{champion["OPPORTUNITY"]:.2f}</div></div>',
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
elif page == "04  ·  Demographic Atlas":
    render_page_bar("Demographic Atlas", "MIG-GE-04")

    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-title">Five lenses on store <span class="gold">demand</span>.</div>
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
        ("POPULATION", "Population", DATA_BLUE),
        ("PCT_WORKING_AGE", "Working-Age %", "#9A8A6A"),
        ("PCT_SINGLE_FAMILY", "Single-Family Housing %", DATA_VIOLET),
        ("PCT_FOREIGNERS", "Foreign Residents %", "#A88B6F"),
        ("proxy_purchasing_power_median_chf", "Median Income (CHF)", "#7E7E88"),
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
                        mode="lines", line=dict(color=GOLD, width=1.4, dash="solid"),
                        name="Trend", hoverinfo="skip",
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=df_demo[xcol], y=df_demo["STORE_COUNT"],
                mode="markers", name="Commune",
                marker=dict(color=color, size=7, opacity=0.7,
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
                    marker=dict(color=GOLD, size=16, symbol="diamond",
                                line=dict(color=BG, width=1.5)),
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
                margin=dict(l=10, r=10, t=14, b=10),
                showlegend=False,
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
            st.plotly_chart(scatter_panel(xc, xl, col_), use_container_width=True)
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
            st.plotly_chart(scatter_panel(xc, xl, col_), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with row2[2]:
        hl_row = df_demo[df_demo["COMMUNE_NAME"] == hl_name]
        if not hl_row.empty:
            h = hl_row.iloc[0]
            st.markdown(
                f"""
            <div class="panel" style="border-left:2px solid {GOLD};padding:24px 24px;
                 height:240px;box-sizing:border-box;display:flex;
                 flex-direction:column;justify-content:center;">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
                <span class="badge gold">Highlight</span>
              </div>
              <div style="font-size:24px;font-weight:500;
                   color:{TEXT};margin-bottom:18px;line-height:1.05;letter-spacing:-0.6px;">{hl_name}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;
                   color:{MUTED};line-height:1.95;letter-spacing:0.4px;
                   font-variant-numeric:tabular-nums;">
                POPULATION &nbsp;&nbsp; <span style="color:{TEXT};">{int(h['POPULATION']):,}</span><br>
                STORES &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{TEXT};">{int(h['STORE_COUNT'])}</span><br>
                PREDICTED &nbsp;&nbsp; <span style="color:{TEXT};">{h['PREDICTED']:.2f}</span><br>
                GAP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{GOLD};font-weight:700;">+{h['OPPORTUNITY']:.2f}</span><br>
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
elif page == "05  ·  Geographic Map":
    render_page_bar("Geographic Map", "MIG-GE-05")

    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-title">The canton, at a <span class="gold">glance</span>.</div>
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
            fill_color="YlOrBr", fill_opacity=0.62,
            line_opacity=0.4, line_color="#000000",
            legend_name="Opportunity Gap (higher = under-served)",
            nan_fill_color="#14141A", nan_fill_opacity=0.35,
        ).add_to(m)

        tooltip_gdf = bounds.merge(
            df_f[["COMMUNE_NAME", "OPPORTUNITY", "POPULATION", "STORE_COUNT"]],
            on="COMMUNE_NAME", how="left",
        )
        folium.GeoJson(
            tooltip_gdf,
            style_function=lambda _: {"fillOpacity": 0, "color": "#2A2A34", "weight": 0.5},
            tooltip=folium.GeoJsonTooltip(
                fields=["COMMUNE_NAME", "POPULATION", "STORE_COUNT", "OPPORTUNITY"],
                aliases=["COMMUNE", "POPULATION", "STORES", "GAP"],
                style=(
                    "background:#0E0E12;color:#F5F5F7;"
                    "font-family:JetBrains Mono,monospace;font-size:11px;"
                    "border:1px solid #C9A961;padding:10px;"
                ),
            ),
        ).add_to(m)

        cn = champion["COMMUNE_NAME"]
        cg = df_f[df_f["COMMUNE_NAME"] == cn]
        if not cg.empty:
            cx = cg.geometry.centroid.iloc[0].x
            cy = cg.geometry.centroid.iloc[0].y
            icon_html = (
                '<div style="background:linear-gradient(135deg,#E0C887 0%,#C9A961 100%);'
                'border:2px solid #07070A;width:40px;height:40px;'
                "transform:rotate(45deg);display:flex;align-items:center;justify-content:center;"
                'box-shadow:0 0 20px rgba(201,169,97,0.7),0 0 40px rgba(201,169,97,0.25);">'
                '<div style="transform:rotate(-45deg);font-family:JetBrains Mono;font-size:13px;'
                'color:#07070A;font-weight:700;">★</div></div>'
            )
            popup_html = (
                f"<div style='font-family:JetBrains Mono,monospace;background:#0E0E12;"
                f"color:#F5F5F7;padding:18px;border-left:3px solid #C9A961;"
                f"min-width:240px;font-variant-numeric:tabular-nums;'>"
                f"<b style='color:#C9A961;font-size:9px;letter-spacing:2px;'>★ CHAMPION TARGET</b><br><br>"
                f"<b style='font-family:Inter Tight,sans-serif;font-size:22px;"
                f"color:#F5F5F7;font-weight:500;letter-spacing:-0.5px;'>{cn}</b><br>"
                f"<span style='color:#86868B;font-size:9.5px;letter-spacing:1.5px;'>CANTON OF GENEVA</span><br><br>"
                f"<span style='color:#86868B;'>POPULATION </span>{int(champion['POPULATION']):,}<br>"
                f"<span style='color:#86868B;'>STORES&nbsp;&nbsp;&nbsp;&nbsp;</span>{int(champion['STORE_COUNT'])}<br>"
                f"<span style='color:#86868B;'>PREDICTED&nbsp;</span>{champion['PREDICTED']:.2f}<br>"
                f"<b style='color:#C9A961;'>GAP&nbsp;&nbsp;&nbsp;&nbsp;+{champion['OPPORTUNITY']:.2f}</b><br><br>"
                f"<span style='color:#86868B;'>INCOME&nbsp;&nbsp;&nbsp;&nbsp;</span>CHF {int(champion['proxy_purchasing_power_median_chf']):,}"
                f"</div>"
            )
            folium.Marker(
                location=[cy, cx],
                popup=folium.Popup(popup_html, max_width=290),
                tooltip=f"★ CHAMPION · {cn}",
                icon=folium.DivIcon(html=icon_html, icon_size=(44, 44), icon_anchor=(22, 22)),
            ).add_to(m)

        if show_stores:
            brand_colors = {"Coop": DATA_BLUE, "Migros": "#E07A3A", "other": "#7E7E88"}
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
                            f"font-size:11px;background:#0E0E12;"
                            f"color:#F5F5F7;padding:10px;"
                            f"border-left:3px solid {bc};'>"
                            f"<b style='color:{bc};'>{brand}</b><br>"
                            f"TYPE · {row.get('shop', '?')}<br>"
                            f"COMMUNE · {row['COMMUNE_NAME']}</div>",
                            max_width=200,
                        ),
                    ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

    st.markdown('<div class="panel" style="padding:12px;">', unsafe_allow_html=True)
    st_folium(m, height=640, use_container_width=True, returned_objects=[])
    st.markdown("</div>", unsafe_allow_html=True)

    render_colophon()
