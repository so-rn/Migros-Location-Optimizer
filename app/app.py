"""
Migros Location Intelligence — Streamlit dashboard.

Editorial executive-briefing design: warm cream surfaces, deep ink type,
Migros-orange accent, serif display heads (Fraunces) + Inter body.
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

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Migros · Location Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design tokens — editorial cream + Migros orange
CREAM   = "#F6F2EA"   # page background
PAPER   = "#FFFFFF"   # cards
SAND    = "#EFE9DC"   # secondary surfaces
INK     = "#0E1116"   # primary text
INK_2   = "#2A2F38"   # secondary text
MUTED   = "#6B7280"   # tertiary text
HAIR    = "#E3DCCC"   # hairline borders
RULE    = "#1A1D23"   # strong rule
ORANGE  = "#FF6600"   # Migros orange (primary accent)
ORANGE_D = "#CC5200"
ORANGE_L = "#FF8533"
OLIVE   = "#5B6B3E"   # secondary accent
WINE    = "#7A2E2E"   # tertiary accent (negative / wine)
GOLD    = "#B8860B"   # highlight
SUCCESS = "#1F7A4D"

PLOT_PAL = [ORANGE, OLIVE, "#3E5C76", GOLD, WINE, "#7E6B8F", "#2C7A7B"]

# ─────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {{
    font-family: 'Inter', system-ui, sans-serif;
    color: {INK};
    -webkit-font-smoothing: antialiased;
}}
.stApp {{ background: {CREAM}; }}
.block-container {{ padding: 1.6rem 2.6rem 5rem; max-width: 1480px; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

/* ── SIDEBAR ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {INK};
    border-right: 1px solid #000;
}}
[data-testid="stSidebar"] * {{ color: {CREAM}; }}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 0; }}

.brand {{
    padding: 28px 22px 22px;
    border-bottom: 1px solid #2A2F38;
}}
.brand-mark {{
    display: flex; align-items: center; gap: 12px;
}}
.brand-logo {{
    width: 38px; height: 38px;
    background: {ORANGE};
    border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Fraunces', serif; font-weight: 600;
    font-size: 22px; color: {INK}; font-style: italic;
}}
.brand-text .name {{
    font-family: 'Fraunces', serif;
    font-size: 18px; font-weight: 500; letter-spacing: -0.3px;
    color: {CREAM}; line-height: 1;
}}
.brand-text .sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #7E8794; letter-spacing: 2px;
    text-transform: uppercase; margin-top: 5px;
}}
.brand-meta {{
    margin-top: 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; color: #7E8794; letter-spacing: 1.5px;
    line-height: 1.8;
}}
.brand-meta b {{ color: {CREAM}; font-weight: 500; }}

.nav-label {{
    padding: 22px 22px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 2.6px; color: #5C6470;
    text-transform: uppercase;
}}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 1px; padding: 0 12px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    width: 100%; padding: 11px 14px !important; margin: 0 !important;
    border-radius: 4px; border-left: 2px solid transparent;
    cursor: pointer; font-size: 14px; font-weight: 400;
    color: #C2C6CC; transition: all 0.12s ease;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background: #1A1D23; color: {CREAM};
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background: #1A1D23;
    border-left-color: {ORANGE};
    color: {CREAM}; font-weight: 500;
}}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}

.side-footer {{
    margin: 24px 18px 18px;
    padding-top: 18px;
    border-top: 1px solid #2A2F38;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #5C6470; letter-spacing: 1.4px;
    line-height: 1.9;
}}
.side-footer .dot {{ color: {ORANGE}; }}

/* ── TYPOGRAPHY ──────────────────────────────────────────────── */
.eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 3.5px; color: {ORANGE};
    text-transform: uppercase; font-weight: 500;
    display: inline-block;
}}
.eyebrow .sep {{ color: {HAIR}; margin: 0 10px; }}
.eyebrow .meta {{ color: {MUTED}; }}

.hero-title {{
    font-family: 'Fraunces', serif;
    font-size: 64px; font-weight: 400; line-height: 1.02;
    letter-spacing: -2px; color: {INK};
    margin: 14px 0 0;
}}
.hero-title em {{ font-style: italic; color: {INK_2}; }}
.hero-title .accent {{
    background: linear-gradient(180deg, transparent 65%, rgba(255,102,0,0.30) 65%);
    padding: 0 2px;
}}
.hero-lede {{
    font-size: 16px; color: {INK_2};
    max-width: 640px; line-height: 1.7; margin: 22px 0 0;
    font-weight: 300;
}}

.section-head {{
    display: flex; align-items: baseline; gap: 14px;
    margin: 40px 0 6px;
}}
.section-num {{
    font-family: 'Fraunces', serif; font-style: italic;
    font-size: 22px; color: {ORANGE}; font-weight: 400;
}}
.section-title {{
    font-family: 'Fraunces', serif;
    font-size: 30px; font-weight: 400; letter-spacing: -0.6px;
    color: {INK};
}}
.section-sub {{
    font-size: 13.5px; color: {MUTED};
    line-height: 1.65; max-width: 700px;
    margin: 4px 0 22px;
}}

.divider {{
    border: none; height: 1px; background: {HAIR}; margin: 28px 0;
}}
.divider-strong {{
    border: none; border-top: 2px solid {RULE}; margin: 22px 0 28px;
}}

/* ── KPI STRIP ──────────────────────────────────────────────── */
.kpi-row {{
    display: grid; grid-template-columns: repeat(5, 1fr);
    border-top: 2px solid {RULE};
    border-bottom: 1px solid {HAIR};
    margin: 32px 0 8px;
}}
.kpi-cell {{
    padding: 22px 24px 24px;
    border-right: 1px solid {HAIR};
    position: relative;
}}
.kpi-cell:last-child {{ border-right: none; }}
.kpi-cell .label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; letter-spacing: 2px;
    text-transform: uppercase; color: {MUTED};
    margin-bottom: 14px;
}}
.kpi-cell .val {{
    font-family: 'Fraunces', serif;
    font-size: 46px; line-height: 1; letter-spacing: -1.5px;
    color: {INK}; font-weight: 400;
}}
.kpi-cell .val.accent {{ color: {ORANGE}; }}
.kpi-cell .delta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: {MUTED};
    margin-top: 10px; letter-spacing: 0.5px;
}}
.kpi-cell .delta .up {{ color: {SUCCESS}; font-weight: 600; }}

/* ── CHAMPION FEATURE ─────────────────────────────────────────── */
.feature {{
    background: {PAPER};
    border: 1px solid {HAIR};
    border-radius: 4px;
    padding: 40px 48px 44px;
    margin: 30px 0 24px;
    position: relative;
    overflow: hidden;
}}
.feature::before {{
    content: '';
    position: absolute; top: 0; left: 0; bottom: 0;
    width: 5px; background: {ORANGE};
}}
.feature-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 3px; color: {ORANGE};
    text-transform: uppercase; margin-bottom: 16px;
}}
.feature-name {{
    font-family: 'Fraunces', serif;
    font-size: 72px; font-weight: 400;
    line-height: 0.96; letter-spacing: -2.4px;
    color: {INK}; margin-bottom: 10px;
}}
.feature-name em {{ font-style: italic; color: {INK_2}; }}
.feature-place {{
    font-size: 13px; color: {MUTED};
    letter-spacing: 0.4px; margin-bottom: 34px;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
}}
.feature-grid {{
    display: grid; grid-template-columns: repeat(6, 1fr);
    gap: 0;
    border-top: 1px solid {HAIR};
}}
.feature-cell {{
    padding: 20px 18px 4px 0;
    border-right: 1px solid {HAIR};
}}
.feature-cell:last-child {{ border-right: none; padding-right: 0; }}
.feature-cell .l {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 1.8px;
    text-transform: uppercase; color: {MUTED};
    margin-bottom: 10px;
}}
.feature-cell .v {{
    font-family: 'Fraunces', serif;
    font-size: 26px; line-height: 1; letter-spacing: -0.8px;
    color: {INK}; font-weight: 400;
}}
.feature-cell .v.accent {{ color: {ORANGE}; font-weight: 500; }}

/* ── FUNNEL ─────────────────────────────────────────────────── */
.funnel {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1px; background: {HAIR};
    border: 1px solid {HAIR}; border-radius: 4px;
    overflow: hidden; margin: 6px 0 12px;
}}
.f-cell {{
    background: {PAPER};
    padding: 26px 28px 28px;
    position: relative;
}}
.f-cell.winner {{ background: linear-gradient(180deg, #FFF5EB 0%, {PAPER} 100%); }}
.f-cell .step {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 2px;
    color: {MUTED}; text-transform: uppercase;
    margin-bottom: 14px;
}}
.f-cell .num {{
    font-family: 'Fraunces', serif;
    font-size: 56px; line-height: 1; font-weight: 400;
    letter-spacing: -2px; color: {INK};
}}
.f-cell.winner .num {{ color: {ORANGE}; }}
.f-cell .ttl {{
    font-size: 13.5px; font-weight: 600; color: {INK};
    margin: 10px 0 4px;
}}
.f-cell .dsc {{
    font-size: 12px; color: {MUTED};
    line-height: 1.55;
}}
.f-cell .arrow {{
    position: absolute; right: -12px; top: 36px;
    width: 22px; height: 22px;
    background: {CREAM}; border-radius: 50%;
    border: 1px solid {HAIR};
    display: flex; align-items: center; justify-content: center;
    z-index: 4; font-size: 11px; color: {INK_2}; font-weight: 600;
}}

/* ── PANEL / CARD ───────────────────────────────────────────── */
.panel {{
    background: {PAPER};
    border: 1px solid {HAIR};
    border-radius: 4px;
    padding: 22px 26px 24px;
    margin-bottom: 18px;
}}
.panel-head {{
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 14px;
}}
.panel-title {{
    font-family: 'Fraunces', serif;
    font-size: 17px; font-weight: 500; color: {INK};
    letter-spacing: -0.3px;
}}

.toolbar {{
    background: {SAND};
    border: 1px solid {HAIR};
    border-radius: 4px;
    padding: 18px 22px 6px;
    margin-bottom: 22px;
}}
.toolbar-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; letter-spacing: 2.5px;
    text-transform: uppercase; color: {INK_2};
    margin-bottom: 10px; font-weight: 600;
}}

/* ── TABLE ──────────────────────────────────────────────────── */
.tbl-wrap {{
    background: {PAPER};
    border: 1px solid {HAIR};
    border-radius: 4px;
    overflow: hidden;
}}
.tbl-head {{
    padding: 16px 22px;
    border-bottom: 1px solid {HAIR};
    display: flex; align-items: center; gap: 12px;
}}
table.et {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
table.et th {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; letter-spacing: 1.8px;
    text-transform: uppercase; color: {MUTED};
    padding: 12px 18px; border-bottom: 1px solid {HAIR};
    text-align: left; font-weight: 500; background: {SAND};
}}
table.et td {{
    padding: 13px 18px;
    border-bottom: 1px solid {HAIR};
    color: {INK};
}}
table.et tr:last-child td {{ border-bottom: none; }}
table.et tr:hover td {{ background: #FAF6EE; transition: background 0.12s; }}
table.et td.mono {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: {MUTED}; font-weight: 500;
}}
table.et td.accent {{ color: {ORANGE}; font-weight: 600; }}
table.et td.serif {{
    font-family: 'Fraunces', serif; font-weight: 500;
    letter-spacing: -0.2px; font-size: 14.5px;
}}
table.et tr.gold td {{ background: #FFF8EA !important; }}

/* ── BADGES & CHIPS ─────────────────────────────────────────── */
.badge {{
    display: inline-flex; align-items: center;
    background: {INK}; color: {CREAM};
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; font-weight: 600; letter-spacing: 1.6px;
    padding: 4px 10px; border-radius: 2px; text-transform: uppercase;
}}
.badge.orange {{ background: {ORANGE}; color: {INK}; }}
.badge.outline {{
    background: transparent; color: {INK};
    border: 1px solid {INK};
}}
.chip {{
    display: inline-flex; align-items: center;
    background: {SAND}; border: 1px solid {HAIR};
    color: {INK_2};
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; padding: 4px 10px; border-radius: 20px;
    letter-spacing: 0.5px; margin-right: 6px;
}}
.tag-rank {{
    display: inline-block;
    font-family: 'Fraunces', serif; font-style: italic;
    font-size: 17px; color: {ORANGE}; font-weight: 500;
    margin-right: 6px;
}}

/* ── INPUTS ─────────────────────────────────────────────────── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div {{
    background: {PAPER} !important;
    border: 1px solid {HAIR} !important;
    border-radius: 3px !important; color: {INK} !important;
}}
label p, .stSlider label p, .stCheckbox label p {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9.5px !important; letter-spacing: 1.5px !important;
    text-transform: uppercase !important; color: {INK_2} !important;
    font-weight: 600 !important;
}}
.stSlider [data-baseweb="slider"] > div > div > div {{ background: {ORANGE} !important; }}
.stCheckbox [role="checkbox"][aria-checked="true"] {{
    background: {ORANGE} !important; border-color: {ORANGE} !important;
}}
.stButton > button {{
    background: {INK} !important; color: {CREAM} !important;
    border: 1px solid {INK} !important; border-radius: 3px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 13px !important;
}}
.stButton > button:hover {{
    background: {ORANGE} !important; color: {INK} !important;
    border-color: {ORANGE} !important;
}}
div[data-testid="stExpander"] {{
    background: {PAPER}; border: 1px solid {HAIR};
    border-radius: 4px;
}}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {HAIR}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {MUTED}; }}
hr {{ border: none; border-top: 1px solid {HAIR}; margin: 22px 0; }}

/* ── FOOTNOTE / RULES ───────────────────────────────────────── */
.footnote {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; letter-spacing: 1.4px;
    color: {MUTED}; text-transform: uppercase;
    padding-top: 14px; border-top: 1px solid {HAIR};
    margin-top: 38px; display: flex; justify-content: space-between;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────
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
    p = _path("geneva_communes_boundaries.geojson")
    b = gpd.read_file(p)
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
    y0 = _df["STORE_COUNT"]
    m0 = sm.OLS(y0, X0).fit()
    cooks = m0.get_influence().cooks_distance[0]
    df_f = _df.copy()
    df_f["COOKS_D"] = cooks
    df_f = df_f[df_f["COOKS_D"] <= 4 / len(df_f)].copy().reset_index(drop=True)
    Xf = sm.add_constant(df_f[["POPULATION", "proxy_purchasing_power_median_chf"]])
    yf = df_f["STORE_COUNT"]
    mf = sm.OLS(yf, Xf).fit()
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


# ─────────────────────────────────────────────────────────────────────
# PLOTLY BASE
# ─────────────────────────────────────────────────────────────────────
def base_layout(**kw):
    base = dict(
        paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(family="Inter", color=INK, size=11),
        xaxis=dict(
            gridcolor=HAIR, zerolinecolor=HAIR, linecolor=HAIR,
            tickfont=dict(color=MUTED, size=10),
        ),
        yaxis=dict(
            gridcolor=HAIR, zerolinecolor=HAIR, linecolor=HAIR,
            tickfont=dict(color=INK_2, size=11),
        ),
        margin=dict(l=10, r=10, t=46, b=10),
        hoverlabel=dict(
            bgcolor=INK, font_color=CREAM, bordercolor=INK,
            font=dict(family="JetBrains Mono", size=11),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=HAIR,
            font=dict(size=10, color=INK_2),
        ),
    )
    base.update(kw)
    return base


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
    <div class="brand">
      <div class="brand-mark">
        <div class="brand-logo">M</div>
        <div class="brand-text">
          <div class="name">Migros</div>
          <div class="sub">Location Intel</div>
        </div>
      </div>
      <div class="brand-meta">
        <b>BRIEFING N°&nbsp;01</b><br>
        Canton of Geneva · CH<br>
        FY 2022 · v1.0
      </div>
    </div>
    <div class="nav-label">Sections</div>
    """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "nav",
        [
            "Executive Briefing",
            "I · Population Funnel",
            "II · Composite Scoring",
            "III · Regression Model",
            "IV · Demographic Atlas",
            "V · Geographic Map",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        f"""
    <div class="side-footer">
      <span class="dot">●</span> PIPELINE<br>
      <span style="color:#9BA1AA;">Canton</span> &nbsp;→&nbsp; Top 20<br>
      <span style="color:#9BA1AA;">Top 20</span> &nbsp;→&nbsp; Top 5<br>
      <span style="color:#9BA1AA;">Top 5</span> &nbsp;→&nbsp; Champion ★<br><br>
      <span class="dot">●</span> SOURCES<br>
      OSM · OCS · OFS
    </div>
    """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────
try:
    stores_gdf, df_pop, df_p22 = load_data()
    bounds = load_boundaries()
    df_clean, joined = build_master(stores_gdf, df_pop, df_p22, bounds)
    top20, top5, top5_ols, champion, df_f, model = run_pipeline(df_clean)
except Exception as exc:
    st.error(f"Data load failed: {exc}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def render_eyebrow(left, right=None):
    right_html = f'<span class="sep">/</span><span class="meta">{right}</span>' if right else ""
    st.markdown(
        f'<div class="eyebrow">{left}{right_html}</div>', unsafe_allow_html=True
    )


def render_section_head(num, title, sub):
    st.markdown(
        f"""
    <div class="section-head">
      <span class="section-num">№ {num}</span>
      <span class="section-title">{title}</span>
    </div>
    <div class="section-sub">{sub}</div>
    """,
        unsafe_allow_html=True,
    )


def render_footnote(left, right):
    st.markdown(
        f'<div class="footnote"><span>{left}</span><span>{right}</span></div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════
# PAGE: EXECUTIVE BRIEFING
# ═════════════════════════════════════════════════════════════════════
if page == "Executive Briefing":
    cn = champion["COMMUNE_NAME"]

    render_eyebrow("Briefing N° 01", "Geneva Expansion · 2022")
    st.markdown(
        f"""
        <h1 class="hero-title">A single optimal site<br>for <em>Migros</em>'<br>
        <span class="accent">next branch</span> in Geneva.</h1>
        <p class="hero-lede">A three-stage quantitative funnel — population pool,
        socio-economic composite, and an OLS opportunity-gap model — synthesises
        demographic, retail-saturation and purchasing-power signals into one
        defensible recommendation.</p>
        """,
        unsafe_allow_html=True,
    )

    # KPIs
    st.markdown(
        f"""
    <div class="kpi-row">
      <div class="kpi-cell">
        <div class="label">Communes Analyzed</div>
        <div class="val">{len(df_clean)}</div>
        <div class="delta">Canton scope · ex-city</div>
      </div>
      <div class="kpi-cell">
        <div class="label">Active Stores</div>
        <div class="val">{int(df_clean['STORE_COUNT'].sum())}</div>
        <div class="delta">Geolocated · OSM</div>
      </div>
      <div class="kpi-cell">
        <div class="label">Stage I Pool</div>
        <div class="val">20</div>
        <div class="delta">By population</div>
      </div>
      <div class="kpi-cell">
        <div class="label">Stage II Finalists</div>
        <div class="val">5</div>
        <div class="delta">Composite score</div>
      </div>
      <div class="kpi-cell">
        <div class="label">Opportunity Gap</div>
        <div class="val accent">+{champion['OPPORTUNITY']:.2f}</div>
        <div class="delta"><span class="up">▲</span> champion target</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Feature recommendation
    fcells = [
        ("Population", f"{int(champion['POPULATION']):,}", ""),
        ("Stores Now", f"{int(champion['STORE_COUNT'])}", ""),
        ("Predicted", f"{champion['PREDICTED']:.2f}", ""),
        ("Gap", f"+{champion['OPPORTUNITY']:.2f}", "accent"),
        ("Income CHF", f"{int(champion['proxy_purchasing_power_median_chf']):,}", ""),
        ("Foreign %", f"{champion['PCT_FOREIGNERS']:.1f}%", ""),
    ]
    cells_html = "".join(
        f'<div class="feature-cell"><div class="l">{l}</div><div class="v {c}">{v}</div></div>'
        for l, v, c in fcells
    )
    st.markdown(
        f"""
    <div class="feature">
      <div class="feature-eyebrow">★ Recommended Target</div>
      <div class="feature-name"><em>{cn}</em></div>
      <div class="feature-place">Canton of Geneva · Switzerland</div>
      <div class="feature-grid">{cells_html}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Funnel
    render_section_head(
        "I", "Selection Funnel",
        "Four checkpoints from the full canton to a single defensible site."
    )
    items = [
        (str(len(df_clean)), "Canton", "All communes (city excluded)", False),
        ("20", "Population Pool", "Largest by residents", False),
        ("5", "Composite Cut", "Multi-factor scoring", False),
        ("1 ★", f"{cn}", "Highest opportunity gap", True),
    ]
    cells = []
    for i, (num, ttl, dsc, win) in enumerate(items):
        arr = '<div class="arrow">›</div>' if i < len(items) - 1 else ""
        cls = "f-cell winner" if win else "f-cell"
        step_lbl = ["Stage 0", "Stage I", "Stage II", "Stage III"][i]
        cells.append(
            f'<div class="{cls}"><div class="step">{step_lbl}</div>'
            f'<div class="num">{num}</div><div class="ttl">{ttl}</div>'
            f'<div class="dsc">{dsc}</div>{arr}</div>'
        )
    st.markdown(f'<div class="funnel">{"".join(cells)}</div>', unsafe_allow_html=True)

    # Methodology cards
    render_section_head(
        "II", "Methodology at a glance",
        "Each stage filters the candidate set on an independent signal — combining "
        "demand (population), affinity (socio-economic fit), and saturation (modelled gap)."
    )
    m1, m2, m3 = st.columns(3, gap="medium")
    method_cards = [
        (m1, "Population shortlist",
         "Rank by resident count. Caps the search to the 20 communes where store volume is viable.",
         "Variables · POPULATION"),
        (m2, "Composite scoring",
         "Min–max weighted index over four dimensions: income, foreign share, working-age share, urban density.",
         "Weights · 35/25/20/20"),
        (m3, "OLS opportunity gap",
         "Predicts store count from population & income. Cook's-D trimmed. Positive residual = under-served.",
         "Outliers · Cook's D ≤ 4/n"),
    ]
    for col, ttl, body, foot in method_cards:
        with col:
            st.markdown(
                f"""
            <div class="panel" style="height:200px;">
              <div class="badge outline" style="margin-bottom:14px;">METHOD</div>
              <div class="panel-title" style="margin-bottom:8px;">{ttl}</div>
              <p style="font-size:13px;color:{MUTED};line-height:1.55;margin:0 0 14px;">{body}</p>
              <div style="font-family:'JetBrains Mono',monospace;font-size:9.5px;
                   color:{ORANGE};letter-spacing:1.6px;">{foot}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    render_footnote(
        f"Migros · Location Intelligence Briefing N° 01",
        f"Canton of Geneva · {len(df_clean)} communes · Vintage 2022",
    )


# ═════════════════════════════════════════════════════════════════════
# PAGE: POPULATION FUNNEL
# ═════════════════════════════════════════════════════════════════════
elif page == "I · Population Funnel":
    render_eyebrow("Stage I", "Population Pool")
    st.markdown(
        f"""
        <h1 class="hero-title" style="font-size:48px;">Top 20 communes,<br>ranked by <em>residents</em>.</h1>
        <p class="hero-lede">The starting pool. City centre excluded — saturation
        and store ubiquity make it noise. Filters refine which subset to inspect.</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="divider-strong">', unsafe_allow_html=True)

    # Filters
    st.markdown(
        '<div class="toolbar"><div class="toolbar-title">⌖ Filter Options</div>',
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
        inc_f = st.selectbox(
            "Income Bracket",
            ["All", "< CHF 70K", "CHF 70K – 80K", "> CHF 80K"],
        )
    with f3:
        st_f = st.selectbox(
            "Store Count",
            ["All", "No stores", "1–3 stores", "4+ stores"],
        )
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
        f"<span class='chip'>{len(df_s1)} of {len(top20)} communes</span>"
        f"<span class='chip'>Sorted by population ▼</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    if df_s1.empty:
        st.warning("No communes match the current filters.")
    else:
        # Chart panel
        bar_colors = [ORANGE if i == 0 else INK_2 for i in range(len(df_s1))]
        fig = go.Figure(
            go.Bar(
                x=df_s1["POPULATION"],
                y=df_s1["COMMUNE_NAME"],
                orientation="h",
                marker_color=bar_colors,
                marker_line_color=PAPER,
                marker_line_width=1,
                text=[f"{v:,}" for v in df_s1["POPULATION"]],
                textposition="outside",
                textfont=dict(color=MUTED, size=10),
                hovertemplate="<b>%{y}</b><br>Population: %{x:,}<extra></extra>",
            )
        )
        fig.update_layout(
            **base_layout(
                height=max(320, len(df_s1) * 34 + 60),
                title=dict(
                    text="Resident population, by commune",
                    font=dict(family="Fraunces", size=16, color=INK), x=0,
                ),
                xaxis=dict(
                    gridcolor=HAIR, tickfont=dict(color=MUTED, size=9),
                    tickformat=",.0f", zerolinecolor=HAIR,
                ),
                yaxis=dict(
                    gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK, size=11),
                    autorange="reversed",
                ),
                margin=dict(l=10, r=80, t=46, b=10),
            )
        )
        st.markdown('<div class="panel" style="padding:14px 18px 4px;">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Table
        rows = ""
        for i, (_, r) in enumerate(df_s1.iterrows()):
            rank_tag = f'<span class="tag-rank">{int(r["RANK"]):02d}</span>'
            row_cls = "gold" if i == 0 else ""
            rows += f"""<tr class="{row_cls}">
              <td class="mono">{int(r['RANK']):02d}</td>
              <td class="serif">{rank_tag}{r['COMMUNE_NAME']}</td>
              <td>{int(r['POPULATION']):,}</td>
              <td>{int(r['STORE_COUNT'])}</td>
              <td>{r['PCT_FOREIGNERS']:.1f}%</td>
              <td>CHF {int(r['proxy_purchasing_power_median_chf']):,}</td>
            </tr>"""
        st.markdown(
            f"""
        <div class="tbl-wrap" style="margin-top:18px;">
          <div class="tbl-head">
            <span class="badge orange">Stage I</span>
            <span class="chip">{len(df_s1)} communes</span>
          </div>
          <table class="et">
            <thead><tr>
              <th>Rank</th><th>Commune</th><th>Population</th>
              <th>Stores</th><th>Foreign %</th><th>Median Income</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """,
            unsafe_allow_html=True,
        )

    render_footnote("Stage I · Population shortlist", "Data · OCS POPBATLOG 2022")


# ═════════════════════════════════════════════════════════════════════
# PAGE: COMPOSITE SCORING
# ═════════════════════════════════════════════════════════════════════
elif page == "II · Composite Scoring":
    render_eyebrow("Stage II", "Composite Scoring")
    st.markdown(
        f"""
        <h1 class="hero-title" style="font-size:48px;">Four dimensions,<br>one <em>score</em>.</h1>
        <p class="hero-lede">A min–max weighted index over income, foreign-resident
        share, working-age share, and urban density. Adjust the weights — the
        ranking recomputes live.</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="divider-strong">', unsafe_allow_html=True)

    st.markdown(
        '<div class="toolbar"><div class="toolbar-title">⌖ Score Weights · target sum 100%</div>',
        unsafe_allow_html=True,
    )
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        w_inc = st.slider("Income Weight %", 0, 60, 35, 5)
    with w2:
        w_for = st.slider("Foreign % Weight", 0, 60, 25, 5)
    with w3:
        w_age = st.slider("Working Age %", 0, 60, 20, 5)
    with w4:
        w_urb = st.slider("Urban Density %", 0, 60, 20, 5)
    total_w = w_inc + w_for + w_age + w_urb
    tc = SUCCESS if total_w == 100 else ORANGE_D
    st.markdown(
        f"""<div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;
        margin-top:8px;color:{tc};letter-spacing:1.2px;font-weight:600;">
        WEIGHT TOTAL: {total_w}% {"✓ BALANCED" if total_w == 100 else "← ADJUST TO 100%"}</div>""",
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
            s2d["SC_INC"] * w_inc
            + s2d["SC_FOR"] * w_for
            + s2d["SC_AGE"] * w_age
            + s2d["SC_URB"] * w_urb
        ) / 100
        s2d = s2d.sort_values("COMPOSITE_SCORE", ascending=False).reset_index(drop=True)

    # Dimension weight cards
    dims = [
        ("Income",            w_inc, ORANGE),
        ("Foreign Residents", w_for, "#3E5C76"),
        ("Working Age",       w_age, OLIVE),
        ("Urban Density",     w_urb, GOLD),
    ]
    cols = st.columns(4)
    for col, (lbl, pct, hex_c) in zip(cols, dims):
        with col:
            st.markdown(
                f"""
            <div class="panel" style="border-top:3px solid {hex_c};text-align:left;height:130px;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                   letter-spacing:1.8px;color:{MUTED};text-transform:uppercase;
                   margin-bottom:14px;">{lbl}</div>
              <div style="font-family:'Fraunces',serif;font-size:42px;color:{hex_c};
                   line-height:1;letter-spacing:-1.3px;font-weight:500;">{pct}%</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Charts
    pal5 = [ORANGE, "#3E5C76", OLIVE, GOLD, WINE]
    c_a, c_b = st.columns(2, gap="medium")

    with c_a:
        fig1 = go.Figure(
            go.Bar(
                y=s2d["COMMUNE_NAME"], x=s2d["COMPOSITE_SCORE"], orientation="h",
                marker_color=pal5[: len(s2d)], marker_line_color=PAPER, marker_line_width=1,
                text=[f"{v:.3f}" for v in s2d["COMPOSITE_SCORE"]],
                textposition="outside",
                textfont=dict(color=MUTED, size=10),
                hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
            )
        )
        fig1.update_layout(
            **base_layout(
                height=300,
                title=dict(text="Composite score, Stage II finalists",
                           font=dict(family="Fraunces", size=14, color=INK), x=0),
                xaxis=dict(gridcolor=HAIR, tickfont=dict(color=MUTED, size=9), range=[0, 1.15]),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK, size=11),
                           autorange="reversed"),
                margin=dict(l=10, r=70, t=46, b=10),
            )
        )
        st.markdown('<div class="panel" style="padding:14px 18px 4px;">', unsafe_allow_html=True)
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
                    marker_line_color=PAPER, marker_line_width=1,
                    hovertemplate=f"<b>%{{y}}</b><br>{lbl}: %{{x:.3f}}<extra></extra>",
                )
            )
        fig2.update_layout(
            **base_layout(
                barmode="stack", height=300,
                title=dict(text="Score breakdown · stacked contributions",
                           font=dict(family="Fraunces", size=14, color=INK), x=0),
                xaxis=dict(gridcolor=HAIR, tickfont=dict(color=MUTED, size=9)),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK, size=11),
                           autorange="reversed"),
                margin=dict(l=10, r=10, t=46, b=32),
                legend=dict(orientation="h", yanchor="bottom", y=1.04,
                            xanchor="left", x=0,
                            bgcolor="rgba(0,0,0,0)",
                            font=dict(size=10, color=INK_2)),
            )
        )
        st.markdown('<div class="panel" style="padding:14px 18px 4px;">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Table
    rows = ""
    for i, (_, r) in enumerate(s2d.iterrows()):
        row_cls = "gold" if i == 0 else ""
        rank_tag = f'<span class="tag-rank">{i+1:02d}</span>'
        rows += f"""<tr class="{row_cls}">
          <td class="mono">{i+1:02d}</td>
          <td class="serif">{rank_tag}{r['COMMUNE_NAME']}</td>
          <td>CHF {int(r['proxy_purchasing_power_median_chf']):,}</td>
          <td>{r['PCT_FOREIGNERS']:.1f}%</td>
          <td>{r['PCT_WORKING_AGE']:.1f}%</td>
          <td>{r['PCT_SINGLE_FAMILY']:.1f}%</td>
          <td class="accent">{r['COMPOSITE_SCORE']:.4f}</td>
        </tr>"""
    st.markdown(
        f"""
    <div class="tbl-wrap" style="margin-top:14px;">
      <div class="tbl-head"><span class="badge orange">Stage II</span>
      <span class="chip">Live weights</span></div>
      <table class="et">
        <thead><tr>
          <th>Rank</th><th>Commune</th><th>Income</th>
          <th>Foreign %</th><th>Working Age %</th>
          <th>Single-Family %</th><th>Score</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """,
        unsafe_allow_html=True,
    )

    render_footnote("Stage II · Composite scoring", "Method · min–max weighted index")


# ═════════════════════════════════════════════════════════════════════
# PAGE: REGRESSION MODEL
# ═════════════════════════════════════════════════════════════════════
elif page == "III · Regression Model":
    render_eyebrow("Stage III", "OLS Opportunity Gap")
    st.markdown(
        f"""
        <h1 class="hero-title" style="font-size:48px;">What the model<br><em>expected</em> to find.</h1>
        <p class="hero-lede">An OLS regression predicts store count from population
        and purchasing power. The residual — predicted minus actual — exposes
        markets that are demonstrably under-served.</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="divider-strong">', unsafe_allow_html=True)

    st.markdown(
        '<div class="toolbar"><div class="toolbar-title">⌖ View Options</div>',
        unsafe_allow_html=True,
    )
    o1, o2 = st.columns(2)
    with o1:
        show_neg = st.checkbox("Show negative opportunity gaps", value=True)
    with o2:
        sort_by = st.selectbox(
            "Sort by", ["Opportunity Gap ↓", "Population ↓", "Predicted ↓"],
        )
    st.markdown("</div>", unsafe_allow_html=True)

    tv = top5_ols.copy()
    if not show_neg:
        tv = tv[tv["OPPORTUNITY"] >= 0]
    if sort_by == "Population ↓":
        tv = tv.sort_values("POPULATION", ascending=False)
    elif sort_by == "Predicted ↓":
        tv = tv.sort_values("PREDICTED", ascending=False)

    bar_cols = [ORANGE if i == 0 else INK_2 for i in range(len(tv))]
    c_a, c_b = st.columns(2, gap="medium")

    with c_a:
        fig1 = go.Figure(
            go.Bar(
                y=tv["COMMUNE_NAME"], x=tv["OPPORTUNITY"], orientation="h",
                marker_color=bar_cols, marker_line_color=PAPER, marker_line_width=1,
                text=[f"+{v:.2f}" if v >= 0 else f"{v:.2f}" for v in tv["OPPORTUNITY"]],
                textposition="outside",
                textfont=dict(color=MUTED, size=10),
                hovertemplate="<b>%{y}</b><br>Gap: %{x:.2f}<extra></extra>",
            )
        )
        fig1.add_vline(x=0, line_color=INK, line_dash="dot", line_width=1)
        fig1.update_layout(
            **base_layout(
                height=300,
                title=dict(text="Opportunity gap · predicted − actual",
                           font=dict(family="Fraunces", size=14, color=INK), x=0),
                xaxis=dict(gridcolor=HAIR, tickfont=dict(color=MUTED, size=9), zerolinecolor=INK),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK, size=11),
                           autorange="reversed"),
                margin=dict(l=10, r=70, t=46, b=10),
            )
        )
        st.markdown('<div class="panel" style="padding:14px 18px 4px;">', unsafe_allow_html=True)
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
                    line=dict(color=INK_2, dash="dot", width=1),
                    name="Equilibrium", hoverinfo="skip",
                )
            )
            for i, (_, r) in enumerate(tv.iterrows()):
                c = ORANGE if i == 0 else INK_2
                fig2.add_trace(
                    go.Scatter(
                        x=[r["STORE_COUNT"]], y=[r["PREDICTED"]],
                        mode="markers+text",
                        marker=dict(color=c, size=15, line=dict(color=PAPER, width=1.5)),
                        text=[r["COMMUNE_NAME"]], textposition="top right",
                        textfont=dict(color=INK, size=10),
                        name=r["COMMUNE_NAME"],
                        hovertemplate=f"<b>{r['COMMUNE_NAME']}</b><br>"
                        f"Actual: {r['STORE_COUNT']}<br>Predicted: {r['PREDICTED']:.2f}<extra></extra>",
                    )
                )
            fig2.update_layout(
                **base_layout(
                    height=300,
                    title=dict(text="Actual vs predicted stores",
                               font=dict(family="Fraunces", size=14, color=INK), x=0),
                    xaxis=dict(title="Actual", gridcolor=HAIR,
                               tickfont=dict(color=MUTED, size=9), range=[-0.2, lim]),
                    yaxis=dict(title="Predicted", gridcolor=HAIR,
                               tickfont=dict(color=MUTED, size=9), range=[-0.2, lim]),
                    margin=dict(l=10, r=10, t=46, b=10),
                    showlegend=False,
                )
            )
            st.markdown('<div class="panel" style="padding:14px 18px 4px;">', unsafe_allow_html=True)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("⌖ OLS Model Summary"):
        Xf2 = sm.add_constant(df_f[["POPULATION", "proxy_purchasing_power_median_chf"]])
        yf2 = df_f["STORE_COUNT"]
        st.code(str(sm.OLS(yf2, Xf2).fit().summary()), language="text")

    # Table
    rows = ""
    for i, (_, r) in enumerate(tv.iterrows()):
        gap = f"+{r['OPPORTUNITY']:.2f}" if r["OPPORTUNITY"] >= 0 else f"{r['OPPORTUNITY']:.2f}"
        row_cls = "gold" if i == 0 else ""
        rank_tag = f'<span class="tag-rank">{int(r["RANK"]):02d}</span>'
        rows += f"""<tr class="{row_cls}">
          <td class="mono">{int(r['RANK']):02d}</td>
          <td class="serif">{rank_tag}{r['COMMUNE_NAME']}</td>
          <td>{int(r['POPULATION']):,}</td>
          <td>{int(r['STORE_COUNT'])}</td>
          <td>{r['PREDICTED']:.2f}</td>
          <td class="accent">{gap}</td>
          <td>{r['COMPOSITE_SCORE']:.4f}</td>
        </tr>"""
    st.markdown(
        f"""
    <div class="tbl-wrap" style="margin-top:14px;">
      <div class="tbl-head"><span class="badge orange">Stage III</span>
      <span class="chip">OLS residuals</span></div>
      <table class="et">
        <thead><tr>
          <th>Rank</th><th>Commune</th><th>Population</th><th>Stores</th>
          <th>Predicted</th><th>Opportunity</th><th>Score</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Final champion callout
    cn = champion["COMMUNE_NAME"]
    fcells = [
        ("Population", f"{int(champion['POPULATION']):,}", ""),
        ("Stores Now", f"{int(champion['STORE_COUNT'])}", ""),
        ("Predicted", f"{champion['PREDICTED']:.2f}", ""),
        ("Gap", f"+{champion['OPPORTUNITY']:.2f}", "accent"),
        ("Income CHF", f"{int(champion['proxy_purchasing_power_median_chf']):,}", ""),
        ("Foreign %", f"{champion['PCT_FOREIGNERS']:.1f}%", ""),
    ]
    cells_html = "".join(
        f'<div class="feature-cell"><div class="l">{l}</div><div class="v {c}">{v}</div></div>'
        for l, v, c in fcells
    )
    st.markdown(
        f"""
    <div class="feature" style="margin-top:28px;">
      <div class="feature-eyebrow">★ Stage III Champion</div>
      <div class="feature-name"><em>{cn}</em></div>
      <div class="feature-place">Canton of Geneva · Switzerland</div>
      <div class="feature-grid">{cells_html}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    render_footnote("Stage III · OLS opportunity model", "Trim · Cook's D ≤ 4/n")


# ═════════════════════════════════════════════════════════════════════
# PAGE: DEMOGRAPHIC ATLAS
# ═════════════════════════════════════════════════════════════════════
elif page == "IV · Demographic Atlas":
    render_eyebrow("Analytics", "Demographic Atlas")
    st.markdown(
        f"""
        <h1 class="hero-title" style="font-size:48px;">Five lenses on<br>store <em>demand</em>.</h1>
        <p class="hero-lede">Each panel pairs a demographic axis with active store
        count across the canton. The star marks the highlighted commune.</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="divider-strong">', unsafe_allow_html=True)

    st.markdown(
        '<div class="toolbar"><div class="toolbar-title">⌖ Options</div>',
        unsafe_allow_html=True,
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        min_pop = st.slider("Min Population", 0, 20000, 0, 1000)
    with d2:
        hl_opt = st.selectbox(
            "Highlight Commune",
            ["Champion"] + list(df_f["COMMUNE_NAME"].sort_values()),
        )
    with d3:
        show_tl = st.checkbox("Show trend line", value=True)
    st.markdown("</div>", unsafe_allow_html=True)

    cn = champion["COMMUNE_NAME"]
    df_demo = df_f.copy()
    if min_pop > 0:
        df_demo = df_demo[df_demo["POPULATION"] >= min_pop]
    hl_name = cn if hl_opt == "Champion" else hl_opt

    panels = [
        ("POPULATION", "Population", "#3E5C76"),
        ("PCT_WORKING_AGE", "Working-Age %", OLIVE),
        ("PCT_SINGLE_FAMILY", "Single-Family Housing %", WINE),
        ("PCT_FOREIGNERS", "Foreign Residents %", "#7E6B8F"),
        ("proxy_purchasing_power_median_chf", "Median Income (CHF)", GOLD),
    ]

    def scatter_panel(xcol, xlabel, color, height=260):
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
                        mode="lines",
                        line=dict(color=ORANGE, width=1.6, dash="solid"),
                        name="Trend", hoverinfo="skip",
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=df_demo[xcol], y=df_demo["STORE_COUNT"],
                mode="markers", name="Commune",
                marker=dict(color=color, size=8, opacity=0.65,
                            line=dict(color=PAPER, width=0.8)),
                customdata=df_demo[["COMMUNE_NAME"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>"
                + xlabel + ": %{x}<br>Stores: %{y}<extra></extra>",
            )
        )
        if not hl_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=hl_data[xcol], y=hl_data["STORE_COUNT"],
                    mode="markers", name=f"★ {hl_name}",
                    marker=dict(color=ORANGE, size=18, symbol="star",
                                line=dict(color=INK, width=1.5)),
                    hovertemplate=f"<b>★ {hl_name}</b><br>{xlabel}: %{{x}}"
                    "<br>Stores: %{y}<extra></extra>",
                )
            )
        fig.update_layout(
            **base_layout(
                height=height,
                title=dict(text=f"{xlabel} vs Stores",
                           font=dict(family="Fraunces", size=13, color=INK), x=0),
                xaxis=dict(
                    title=dict(text=xlabel, font=dict(size=9, color=MUTED)),
                    gridcolor=HAIR, tickfont=dict(color=MUTED, size=9),
                ),
                yaxis=dict(
                    title=dict(text="Stores", font=dict(size=9, color=MUTED)),
                    gridcolor=HAIR, tickfont=dict(color=MUTED, size=9),
                ),
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
            )
        )
        return fig

    row1 = st.columns(3, gap="medium")
    for (xc, xl, col_), col in zip(panels[:3], row1):
        with col:
            st.markdown('<div class="panel" style="padding:12px 16px 4px;">', unsafe_allow_html=True)
            st.plotly_chart(scatter_panel(xc, xl, col_), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    row2 = st.columns(3, gap="medium")
    for (xc, xl, col_), col in zip(panels[3:], row2[:2]):
        with col:
            st.markdown('<div class="panel" style="padding:12px 16px 4px;">', unsafe_allow_html=True)
            st.plotly_chart(scatter_panel(xc, xl, col_), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with row2[2]:
        hl_row = df_demo[df_demo["COMMUNE_NAME"] == hl_name]
        if not hl_row.empty:
            h = hl_row.iloc[0]
            st.markdown(
                f"""
            <div class="panel" style="background:linear-gradient(180deg,#FFF5EB 0%,{PAPER} 100%);
                 border-left:4px solid {ORANGE};padding:24px 24px;height:260px;box-sizing:border-box;
                 display:flex;flex-direction:column;justify-content:center;">
              <div class="badge orange" style="margin-bottom:14px;">★ HIGHLIGHT</div>
              <div style="font-family:'Fraunces',serif;font-size:30px;font-style:italic;
                   color:{INK};margin-bottom:18px;line-height:1.05;letter-spacing:-0.8px;">{hl_name}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;
                   color:{MUTED};line-height:2.0;letter-spacing:0.5px;">
                POPULATION &nbsp;&nbsp; <span style="color:{INK};font-weight:600;">{int(h['POPULATION']):,}</span><br>
                STORES &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{INK};font-weight:600;">{int(h['STORE_COUNT'])}</span><br>
                PREDICTED &nbsp;&nbsp; <span style="color:{INK};font-weight:600;">{h['PREDICTED']:.2f}</span><br>
                GAP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{ORANGE};font-weight:700;">+{h['OPPORTUNITY']:.2f}</span><br>
                INCOME &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{INK};font-weight:600;">CHF {int(h['proxy_purchasing_power_median_chf']):,}</span>
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    render_footnote("Atlas · five demographic lenses", "Trend · linear OLS")


# ═════════════════════════════════════════════════════════════════════
# PAGE: GEOGRAPHIC MAP
# ═════════════════════════════════════════════════════════════════════
elif page == "V · Geographic Map":
    render_eyebrow("Geospatial", "Market Map")
    st.markdown(
        f"""
        <h1 class="hero-title" style="font-size:48px;">The canton, at <em>a glance</em>.</h1>
        <p class="hero-lede">Choropleth encodes the modelled opportunity gap.
        Circle pins mark existing supermarkets. The star pins the recommended target.</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="divider-strong">', unsafe_allow_html=True)

    st.markdown(
        '<div class="toolbar"><div class="toolbar-title">⌖ Map Options</div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    with m1:
        show_stores = st.checkbox("Show store pins", value=True)
    with m2:
        brand_filter = st.multiselect(
            "Filter Brands", ["Coop", "Migros", "other"],
            default=["Coop", "Migros", "other"],
        )
    with m3:
        map_zoom = st.slider("Zoom Level", 10, 14, 12)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("Building map…"):
        lat0 = bounds.geometry.centroid.y.mean()
        lon0 = bounds.geometry.centroid.x.mean()
        m = folium.Map(
            location=[lat0, lon0], zoom_start=map_zoom,
            tiles="cartodbpositron",
        )

        folium.Choropleth(
            geo_data=bounds, name="Opportunity Gap",
            data=df_f, columns=["COMMUNE_NAME", "OPPORTUNITY"],
            key_on="feature.properties.COMMUNE_NAME",
            fill_color="OrRd", fill_opacity=0.72,
            line_opacity=0.5, line_color="#FFFFFF",
            legend_name="Market Opportunity Gap (higher = under-served)",
            nan_fill_color="#EFE9DC", nan_fill_opacity=0.4,
        ).add_to(m)

        tooltip_gdf = bounds.merge(
            df_f[["COMMUNE_NAME", "OPPORTUNITY", "POPULATION", "STORE_COUNT"]],
            on="COMMUNE_NAME", how="left",
        )
        folium.GeoJson(
            tooltip_gdf,
            style_function=lambda _: {
                "fillOpacity": 0, "color": "#FFFFFF", "weight": 0.4
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["COMMUNE_NAME", "POPULATION", "STORE_COUNT", "OPPORTUNITY"],
                aliases=["Commune", "Population", "Stores", "Gap"],
                style=(
                    "background:#0E1116;color:#F6F2EA;"
                    "font-family:JetBrains Mono,monospace;font-size:11px;"
                    "border:1px solid #0E1116;border-radius:3px;padding:10px;"
                ),
            ),
        ).add_to(m)

        # Champion marker
        cn = champion["COMMUNE_NAME"]
        cg = df_f[df_f["COMMUNE_NAME"] == cn]
        if not cg.empty:
            cx = cg.geometry.centroid.iloc[0].x
            cy = cg.geometry.centroid.iloc[0].y
            icon_html = (
                '<div style="background:#FF6600;border:3px solid #FFFFFF;border-radius:50%;'
                "width:44px;height:44px;display:flex;align-items:center;justify-content:center;"
                "font-size:21px;color:#0E1116;font-weight:700;"
                'box-shadow:0 0 22px rgba(255,102,0,0.65),0 0 44px rgba(255,102,0,0.25);">'
                "★</div>"
            )
            popup_html = (
                f"<div style='font-family:JetBrains Mono,monospace;background:#FFFFFF;"
                f"color:#0E1116;padding:18px;border-radius:4px;border-left:4px solid #FF6600;"
                f"min-width:240px;'>"
                f"<b style='color:#FF6600;font-size:10px;letter-spacing:2px;'>★ CHAMPION TARGET</b><br><br>"
                f"<b style='font-family:Fraunces,serif;font-style:italic;font-size:22px;"
                f"color:#0E1116;'>{cn}</b><br>"
                f"<span style='color:#6B7280;font-size:10px;letter-spacing:1px;'>CANTON OF GENEVA</span><br><br>"
                f"<span style='color:#6B7280;'>POPULATION </span>{int(champion['POPULATION']):,}<br>"
                f"<span style='color:#6B7280;'>STORES NOW </span>{int(champion['STORE_COUNT'])}<br>"
                f"<span style='color:#6B7280;'>PREDICTED&nbsp;&nbsp;</span>{champion['PREDICTED']:.2f}<br>"
                f"<b style='color:#FF6600;'>GAP +{champion['OPPORTUNITY']:.2f}</b><br><br>"
                f"<span style='color:#6B7280;'>INCOME CHF </span>{int(champion['proxy_purchasing_power_median_chf']):,}"
                f"</div>"
            )
            folium.Marker(
                location=[cy, cx],
                popup=folium.Popup(popup_html, max_width=290),
                tooltip=f"★ CHAMPION: {cn}",
                icon=folium.DivIcon(html=icon_html, icon_size=(48, 48), icon_anchor=(24, 24)),
            ).add_to(m)

        if show_stores:
            brand_colors = {"Coop": "#3E5C76", "Migros": "#FF6600", "other": "#5B6B3E"}
            for _, row in joined.iterrows():
                brand = row.get("brand_category", "other")
                if brand not in brand_filter:
                    continue
                if row["COMMUNE_NAME"] in df_f["COMMUNE_NAME"].values:
                    bc = brand_colors.get(brand, brand_colors["other"])
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=5, color="#FFFFFF", fill=True, fill_color=bc,
                        fill_opacity=0.92, weight=1.2,
                        tooltip=f"{brand} · {row['COMMUNE_NAME']}",
                        popup=folium.Popup(
                            f"<div style='font-family:JetBrains Mono,monospace;"
                            f"font-size:11px;background:#FFFFFF;"
                            f"color:#0E1116;padding:10px;border-radius:3px;"
                            f"border-left:3px solid {bc};'>"
                            f"<b style='color:{bc};'>{brand}</b><br>"
                            f"Type: {row.get('shop', '?')}<br>"
                            f"Commune: {row['COMMUNE_NAME']}</div>",
                            max_width=200,
                        ),
                    ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

    st.markdown('<div class="panel" style="padding:14px;">', unsafe_allow_html=True)
    st_folium(m, height=640, use_container_width=True, returned_objects=[])
    st.markdown("</div>", unsafe_allow_html=True)

    render_footnote("Geographic atlas", "Tiles · CARTO Positron · Boundaries · OSM")
