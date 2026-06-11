import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Migros · Site Intelligence",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN TOKENS ─────────────────────────────────────────────
BG     = '#07080D'
S1     = '#0D0F1A'
S2     = '#131728'
S3     = '#1A1E30'
TEAL   = '#00C9A7'
TEAL_L = '#00E5C8'
TEAL_D = '#007A68'
TEXT   = '#EDF1F8'
MUTED  = '#8494A9'
FAINT  = '#3A4458'
BORDER = '#1C2038'
GOLD   = '#E8B84B'
RED    = '#FF6B6B'

# ── CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
  font-family: 'DM Sans', sans-serif;
  background: {BG};
  color: {TEXT};
  -webkit-font-smoothing: antialiased;
}}
.stApp {{ background: {BG}; }}
.block-container {{ padding: 2.2rem 2.8rem 4rem; max-width: 1420px; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

/* ── SIDEBAR ─────────────────────────────────── */
[data-testid="stSidebar"] {{
  background: {S1};
  border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {TEXT}; }}

.sb-header {{
  padding: 28px 20px 22px;
  border-bottom: 1px solid {BORDER};
  margin-bottom: 4px;
}}
.sb-mark {{
  width: 42px; height: 42px;
  background: {TEAL};
  border-radius: 11px;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Instrument Serif', serif;
  font-size: 22px; color: {BG}; font-style: italic;
  box-shadow: 0 4px 20px rgba(0,201,167,0.30);
}}
.sb-title {{
  font-size: 13px; font-weight: 600; color: {TEXT};
  margin-top: 12px; letter-spacing: 0.1px;
}}
.sb-meta {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; color: {MUTED};
  letter-spacing: 1.8px; text-transform: uppercase; margin-top: 3px;
}}
.sb-section-label {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; letter-spacing: 2.5px; color: {FAINT};
  text-transform: uppercase; padding: 18px 20px 6px;
}}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; padding: 4px 10px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  width: 100%;
  padding: 10px 13px !important;
  margin: 0 !important;
  border-radius: 8px;
  border: 1px solid transparent;
  cursor: pointer;
  font-size: 13.5px; font-weight: 500;
  color: {MUTED};
  transition: all 0.15s ease;
  letter-spacing: 0.1px;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
  background: rgba(0,201,167,0.06); color: {TEXT};
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background: rgba(0,201,167,0.09);
  border-color: rgba(0,201,167,0.22);
  color: {TEAL_L}; font-weight: 600;
}}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
[data-testid="stSidebar"] .stRadio > label {{ display: none; }}

.sb-footer {{
  margin: 16px 10px 20px;
  background: {S2};
  border: 1px solid {BORDER};
  border-radius: 10px;
  padding: 14px 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; color: {MUTED}; letter-spacing: 1px;
}}
.sb-footer .ftitle {{ color: {TEAL}; letter-spacing: 2px; margin-bottom: 10px; font-weight: 600; }}
.sb-footer .frow {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
.sb-footer .frow:last-child {{ margin-bottom: 0; color: {TEAL_L}; }}

/* ── TYPOGRAPHY ──────────────────────────────── */
.display {{
  font-family: 'Instrument Serif', serif;
  font-size: 54px; font-weight: 400; line-height: 1.1;
  letter-spacing: -1.8px; color: {TEXT};
}}
.display em {{ font-style: italic; color: {TEAL}; }}
.eyebrow {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: 3.5px;
  text-transform: uppercase; color: {TEAL}; margin-bottom: 10px;
}}
.heading {{
  font-family: 'Instrument Serif', serif;
  font-size: 32px; font-weight: 400;
  letter-spacing: -0.5px; color: {TEXT}; margin-bottom: 6px;
}}
.subtext {{
  font-size: 14px; color: {MUTED};
  line-height: 1.7; max-width: 560px;
}}
.teal-rule {{
  height: 1px;
  background: linear-gradient(90deg, {TEAL}, rgba(0,201,167,0.15) 60%, transparent);
  border: none; margin: 22px 0 30px;
}}

/* ── KPI GRID ────────────────────────────────── */
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px; margin: 24px 0 32px;
}}
.kpi {{
  background: {S1}; border: 1px solid {BORDER};
  border-radius: 12px; padding: 22px 20px;
  position: relative; overflow: hidden;
  transition: border-color 0.2s, transform 0.18s;
  cursor: default;
}}
.kpi:hover {{
  border-color: rgba(0,201,167,0.28); transform: translateY(-2px);
}}
.kpi::before {{
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, {TEAL}, transparent);
  opacity: 0.5;
}}
.kpi-label {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; letter-spacing: 2px;
  text-transform: uppercase; color: {MUTED}; margin-bottom: 12px;
}}
.kpi-val {{
  font-family: 'Instrument Serif', serif;
  font-size: 40px; font-weight: 400;
  line-height: 1; letter-spacing: -1.2px; color: {TEXT};
}}
.kpi-val.teal {{ color: {TEAL}; }}
.kpi-val.gold {{ color: {GOLD}; }}
.kpi-note {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; color: {FAINT};
  margin-top: 9px; letter-spacing: 1px;
}}

/* ── CHAMPION ────────────────────────────────── */
.champion {{
  background: linear-gradient(140deg, {S2} 0%, {S1} 100%);
  border: 1px solid rgba(0,201,167,0.22);
  border-radius: 16px; padding: 48px 52px; margin: 28px 0;
  position: relative; overflow: hidden;
}}
.champion::after {{
  content: ''; position: absolute;
  top: -80px; right: -100px;
  width: 380px; height: 380px; border-radius: 50%;
  background: radial-gradient(circle, rgba(0,201,167,0.07) 0%, transparent 65%);
  pointer-events: none;
}}
.champ-eyebrow {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: 4px; color: {TEAL};
  text-transform: uppercase; margin-bottom: 14px;
}}
.champ-name {{
  font-family: 'Instrument Serif', serif;
  font-size: 68px; font-weight: 400; color: {TEXT};
  line-height: 0.95; letter-spacing: -2px; margin-bottom: 8px;
}}
.champ-place {{
  font-size: 13px; color: {MUTED}; margin-bottom: 36px; letter-spacing: 0.4px;
}}
.champ-metrics {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.cm {{
  background: rgba(255,255,255,0.025);
  border: 1px solid {BORDER}; border-radius: 10px;
  padding: 14px 18px; min-width: 112px;
  transition: border-color 0.2s, background 0.2s;
}}
.cm:hover {{
  border-color: rgba(0,201,167,0.28);
  background: rgba(0,201,167,0.04);
}}
.cm-label {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 8px; letter-spacing: 1.5px;
  text-transform: uppercase; color: {MUTED}; margin-bottom: 7px;
}}
.cm-value {{ font-size: 20px; font-weight: 700; color: {TEXT}; }}
.cm-value.teal {{ color: {TEAL}; }}
.cm-value.gold {{ color: {GOLD}; }}

/* ── FUNNEL ──────────────────────────────────── */
.funnel {{ display: flex; margin: 24px 0; gap: 0; }}
.f-step {{
  flex: 1; background: {S1};
  border: 1px solid {BORDER};
  padding: 28px 24px; position: relative;
  transition: border-color 0.2s;
}}
.f-step:first-child {{ border-radius: 12px 0 0 12px; }}
.f-step:last-child {{ border-radius: 0 12px 12px 0; }}
.f-step + .f-step {{ border-left: none; }}
.f-step.winner {{
  background: linear-gradient(135deg, rgba(0,201,167,0.07) 0%, {S1} 100%);
  border-color: rgba(0,201,167,0.28);
}}
.f-num {{
  font-family: 'Instrument Serif', serif;
  font-size: 44px; line-height: 1; color: {TEXT}; margin-bottom: 7px;
}}
.f-num.teal {{ color: {TEAL}; }}
.f-title {{ font-size: 13px; font-weight: 600; color: {TEXT}; margin-bottom: 4px; }}
.f-desc {{ font-size: 11px; color: {MUTED}; line-height: 1.45; }}
.f-arr {{
  position: absolute; right: -10px; top: 50%;
  transform: translateY(-50%);
  font-size: 18px; color: {FAINT}; z-index: 10;
}}

/* ── TABLE ───────────────────────────────────── */
.t-wrap {{
  background: {S1}; border: 1px solid {BORDER};
  border-radius: 12px; padding: 24px; overflow-x: auto;
}}
.t-header {{
  display: flex; align-items: center; gap: 10px; margin-bottom: 18px;
}}
table.et {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
table.et th {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; letter-spacing: 2px; text-transform: uppercase;
  color: {MUTED}; padding: 10px 14px;
  border-bottom: 1px solid {BORDER}; text-align: left; white-space: nowrap;
}}
table.et td {{
  padding: 11px 14px;
  border-bottom: 1px solid rgba(28,32,56,0.55); color: {TEXT};
}}
table.et tr:last-child td {{ border-bottom: none; }}
table.et tr:hover td {{
  background: rgba(0,201,167,0.035); transition: background 0.12s;
}}
.tc {{ color: {TEAL}; font-weight: 600; }}
.gc {{ color: {GOLD}; font-weight: 600; }}
.mono {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: {MUTED}; }}

/* ── UI ELEMENTS ─────────────────────────────── */
.badge {{
  display: inline-flex; align-items: center;
  background: rgba(0,201,167,0.10);
  border: 1px solid rgba(0,201,167,0.25);
  color: {TEAL_L};
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; font-weight: 600; letter-spacing: 2px;
  padding: 4px 10px; border-radius: 20px; text-transform: uppercase;
}}
.badge.gold {{
  background: rgba(232,184,75,0.10);
  border-color: rgba(232,184,75,0.25); color: {GOLD};
}}
.chip {{
  display: inline-flex; align-items: center;
  background: {S2}; border: 1px solid {BORDER};
  color: {MUTED}; font-family: 'JetBrains Mono', monospace;
  font-size: 9.5px; padding: 3px 10px;
  border-radius: 20px; letter-spacing: 0.5px; margin-right: 6px;
}}
.filter-wrap {{
  background: {S2}; border: 1px solid {BORDER};
  border-radius: 12px; padding: 20px 24px 8px; margin-bottom: 22px;
}}
.filter-title {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; letter-spacing: 2.5px;
  text-transform: uppercase; color: {MUTED}; margin-bottom: 14px;
}}

/* Inputs */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div {{
  background: {S1} !important; border: 1px solid {BORDER} !important;
  border-radius: 8px !important; color: {TEXT} !important;
}}
label p, .stSlider label p, .stCheckbox label p {{
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 9.5px !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important; color: {MUTED} !important;
}}
div[data-testid="stMetric"] {{
  background: {S1}; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px 20px;
}}
div[data-testid="stExpander"] {{
  background: {S1}; border: 1px solid {BORDER}; border-radius: 10px;
}}
.stButton > button {{
  background: transparent !important; color: {TEAL} !important;
  border: 1px solid rgba(0,201,167,0.35) !important; border-radius: 8px !important;
  font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
  font-size: 13px !important; transition: all 0.15s !important;
}}
.stButton > button:hover {{
  background: rgba(0,201,167,0.09) !important; border-color: {TEAL} !important;
}}
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {FAINT}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {TEAL_D}; }}
hr {{ border: none; border-top: 1px solid {BORDER}; margin: 22px 0; }}
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DATA_URL = 'https://raw.githubusercontent.com/so-rn/Migros-Location-Optimizer/main/data/'


def _path(rel):
    local = os.path.join(DATA_DIR, rel)
    return local if os.path.exists(local) else DATA_URL + rel


@st.cache_data(show_spinner=False)
def load_data():
    df_stores = pd.read_csv(_path('geneva_supermarkets_data_with_address.csv'))
    stores_gdf = gpd.GeoDataFrame(
        df_stores,
        geometry=gpd.points_from_xy(df_stores.longitude, df_stores.latitude),
        crs='EPSG:4326',
    )
    df_pop = pd.read_csv(_path('OCS_POPBATLOG_COMMUNE.csv'), sep=';')
    df_pop['COMMUNE'] = df_pop['COMMUNE'].str.strip()
    df_power = pd.read_csv(_path('finance/geneva_purchasing_power_proxy_all_years.csv'))
    df_p22 = df_power[df_power['year'] == 2022].copy()
    df_p22['commune'] = df_p22['commune'].str.strip()
    return stores_gdf, df_pop, df_p22


@st.cache_data(show_spinner=False)
def load_boundaries():
    local = os.path.join(DATA_DIR, 'geneva_communes_boundaries.geojson')
    path = local if os.path.exists(local) else DATA_URL + 'geneva_communes_boundaries.geojson'
    b = gpd.read_file(path)
    return b[['COMMUNE_NAME', 'geometry']].set_crs(epsg=4326, allow_override=True)


@st.cache_data(show_spinner=False)
def build_master(_stores, _pop, _power, _bounds):
    joined = gpd.sjoin(_stores, _bounds, how='inner', predicate='intersects')
    store_ct = joined.groupby('COMMUNE_NAME').size().reset_index(name='STORE_COUNT')
    df = (
        _bounds
        .merge(_pop, left_on='COMMUNE_NAME', right_on='COMMUNE', how='left')
        .merge(_power[['commune', 'proxy_purchasing_power_median_chf']],
               left_on='COMMUNE_NAME', right_on='commune', how='left')
        .merge(store_ct, on='COMMUNE_NAME', how='left')
    )
    df['STORE_COUNT']       = df['STORE_COUNT'].fillna(0)
    df['PCT_WORKING_AGE']   = (df['AGE_20_64']    / df['POPULATION']) * 100
    df['PCT_SINGLE_FAMILY'] = (df['MAISON_INDIV'] / df['BATLOG_TOT']) * 100
    df['PCT_FOREIGNERS']    = (df['POP_ETR']      / df['POPULATION']) * 100
    df_clean = df.dropna(subset=['POPULATION', 'proxy_purchasing_power_median_chf']).copy()
    df_clean = df_clean[~df_clean['COMMUNE_NAME'].isin(['Genève', 'Geneve', 'Geneva'])].reset_index(drop=True)
    return df_clean, joined


@st.cache_data(show_spinner=False)
def run_pipeline(_df):
    top20 = _df.sort_values('POPULATION', ascending=False).head(20).copy().reset_index(drop=True)
    top20['RANK'] = range(1, 21)

    def mm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0

    s2 = top20.copy()
    s2['SC_INC'] = mm(s2['proxy_purchasing_power_median_chf'])
    s2['SC_FOR'] = mm(s2['PCT_FOREIGNERS'])
    s2['SC_AGE'] = mm(s2['PCT_WORKING_AGE'])
    s2['SC_URB'] = 1 - mm(s2['PCT_SINGLE_FAMILY'])
    W = {'SC_INC': .35, 'SC_FOR': .25, 'SC_AGE': .20, 'SC_URB': .20}
    s2['COMPOSITE_SCORE'] = sum(s2[k] * v for k, v in W.items())
    s2 = s2.sort_values('COMPOSITE_SCORE', ascending=False).reset_index(drop=True)
    top5 = s2.head(5).copy()
    top5['RANK'] = range(1, 6)

    X0 = sm.add_constant(_df[['POPULATION', 'proxy_purchasing_power_median_chf']])
    y0 = _df['STORE_COUNT']
    m0 = sm.OLS(y0, X0).fit()
    cooks = m0.get_influence().cooks_distance[0]
    df_f = _df.copy()
    df_f['COOKS_D'] = cooks
    df_f = df_f[df_f['COOKS_D'] <= 4 / len(df_f)].copy().reset_index(drop=True)
    Xf = sm.add_constant(df_f[['POPULATION', 'proxy_purchasing_power_median_chf']])
    yf = df_f['STORE_COUNT']
    mf = sm.OLS(yf, Xf).fit()
    df_f['PREDICTED']   = mf.predict(Xf)
    df_f['OPPORTUNITY'] = df_f['PREDICTED'] - df_f['STORE_COUNT']

    top5_ols = (
        top5.merge(df_f[['COMMUNE_NAME', 'PREDICTED', 'OPPORTUNITY']], on='COMMUNE_NAME', how='left')
        .sort_values('OPPORTUNITY', ascending=False)
        .reset_index(drop=True)
    )
    top5_ols['RANK'] = range(1, len(top5_ols) + 1)
    champion = top5_ols.iloc[0].copy()
    return top20, top5, top5_ols, champion, df_f, mf


# ── PLOTLY BASE LAYOUT ─────────────────────────────────────────
def plot_layout(**kw):
    base = dict(
        paper_bgcolor=S1, plot_bgcolor=S1,
        font=dict(family='DM Sans', color=TEXT, size=11),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=FAINT, size=9)),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT, size=11)),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(bgcolor=S2, font_color=TEXT, bordercolor=BORDER, font_size=12),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=BORDER, font=dict(size=9, color=MUTED)),
    )
    base.update(kw)
    return base


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sb-header">
      <div class="sb-mark">M</div>
      <div class="sb-title">Location Intelligence</div>
      <div class="sb-meta">Geneva · Switzerland</div>
    </div>
    <div class="sb-section-label">Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio("nav", [
        "◎  Overview",
        "⬡  Population",
        "◈  Scoring",
        "⧖  Regression",
        "▤  Demographics",
        "⊕  Map",
    ], label_visibility="collapsed")

    st.markdown(f"""
    <div class="sb-footer">
      <div class="ftitle">PIPELINE</div>
      <div class="frow"><span>All communes</span><span style="color:{TEXT};">→ Top 20</span></div>
      <div class="frow"><span>Top 20</span><span style="color:{TEXT};">→ Top 5</span></div>
      <div class="frow"><span>Top 5</span><span>→ Champion ★</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────
with st.spinner(""):
    try:
        stores_gdf, df_pop, df_p22 = load_data()
        bounds = load_boundaries()
        df_clean, joined = build_master(stores_gdf, df_pop, df_p22, bounds)
        top20, top5, top5_ols, champion, df_f, model = run_pipeline(df_clean)
        ok = True
    except Exception as exc:
        ok, err = False, str(exc)

if not ok:
    st.error(f"Data load failed: {err}")
    st.stop()


# ─────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "◎  Overview":
    cn = champion['COMMUNE_NAME']

    st.markdown(f"""
    <div class="eyebrow">Migros · Site Intelligence</div>
    <div class="display">Geneva's optimal<br><em>expansion target</em></div>
    <p class="subtext" style="margin-top:14px;">A 3-stage quantitative funnel identifying the single best commune for a new
    Migros store in the Canton of Geneva — combining demographic, socio-economic and
    retail-saturation analysis.</p>
    <div class="teal-rule"></div>
    """, unsafe_allow_html=True)

    # KPIs
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Communes Analyzed</div>
        <div class="kpi-val">{len(df_clean)}</div>
        <div class="kpi-note">CANTON OF GENEVA</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Stores Mapped</div>
        <div class="kpi-val">{int(df_clean['STORE_COUNT'].sum())}</div>
        <div class="kpi-note">ACTIVE LOCATIONS</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Stage 1 Pool</div>
        <div class="kpi-val">20</div>
        <div class="kpi-note">BY POPULATION</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Stage 2 Finalists</div>
        <div class="kpi-val">5</div>
        <div class="kpi-note">COMPOSITE SCORE</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Opportunity Gap</div>
        <div class="kpi-val teal">+{champion['OPPORTUNITY']:.2f}</div>
        <div class="kpi-note">CHAMPION TARGET</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Champion card
    mx = ""
    for lbl, val, cls in [
        ("Population",   f"{int(champion['POPULATION']):,}", ""),
        ("Stores Now",   f"{int(champion['STORE_COUNT'])}",  ""),
        ("Predicted",    f"{champion['PREDICTED']:.2f}",     ""),
        ("Opportunity",  f"+{champion['OPPORTUNITY']:.2f}",  "teal"),
        ("Income CHF",   f"{int(champion['proxy_purchasing_power_median_chf']):,}", ""),
        ("Foreign %",    f"{champion['PCT_FOREIGNERS']:.1f}%", ""),
    ]:
        mx += f'<div class="cm"><div class="cm-label">{lbl}</div><div class="cm-value {cls}">{val}</div></div>'

    st.markdown(f"""
    <div class="champion">
      <div class="champ-eyebrow">★ Optimal Expansion Target</div>
      <div class="champ-name">{cn}</div>
      <div class="champ-place">Canton of Geneva · Switzerland</div>
      <div class="champ-metrics">{mx}</div>
    </div>
    """, unsafe_allow_html=True)

    # Funnel
    st.markdown("""
    <div class="eyebrow" style="margin-top:36px;">Analysis Pipeline</div>
    <div class="heading">3-Stage Selection Funnel</div>
    <p class="subtext" style="margin-bottom:24px;">From the full canton to a single optimal location.</p>
    """, unsafe_allow_html=True)

    items = [
        (str(len(df_clean)), "All Communes",        "Full canton · city centre excluded", False),
        ("20",               "Stage 1 · Population", "Ranked by resident population",      False),
        ("5",                "Stage 2 · Scoring",    "Composite socio-economic index",      False),
        ("1 ★",              f"Champion · {cn}",     "Highest OLS opportunity gap",         True),
    ]
    steps = []
    for i, (num, title, desc, win) in enumerate(items):
        arr = '<span class="f-arr">›</span>' if i < len(items) - 1 else ''
        cls = "f-step winner" if win else "f-step"
        nc = "f-num teal" if win else "f-num"
        steps.append(f"""
        <div class="{cls}">
          <div class="{nc}">{num}</div>
          <div class="f-title">{title}</div>
          <div class="f-desc">{desc}</div>
          {arr}
        </div>""")
    st.markdown(f'<div class="funnel">{"".join(steps)}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: POPULATION
# ─────────────────────────────────────────────────────────────
elif page == "⬡  Population":
    st.markdown("""
    <div class="eyebrow">Stage 01</div>
    <div class="heading">Top 20 Communes by Population</div>
    <p class="subtext">Starting pool drawn from all Geneva communes. City centre excluded. Ranked by resident population.</p>
    <div class="teal-rule"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="filter-wrap"><div class="filter-title">⊘ Filter Options</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        pop_range = st.slider(
            "Population Range",
            int(top20['POPULATION'].min()), int(top20['POPULATION'].max()),
            (int(top20['POPULATION'].min()), int(top20['POPULATION'].max())),
            step=500,
        )
    with fc2:
        inc_f = st.selectbox("Income Bracket", ["All", "< CHF 70K", "CHF 70K – 80K", "> CHF 80K"])
    with fc3:
        st_f = st.selectbox("Store Count", ["All", "No stores", "1–3 stores", "4+ stores"])
    st.markdown('</div>', unsafe_allow_html=True)

    df_s1 = top20[top20['POPULATION'].between(*pop_range)].copy()
    if inc_f == "< CHF 70K":
        df_s1 = df_s1[df_s1['proxy_purchasing_power_median_chf'] < 70000]
    elif inc_f == "CHF 70K – 80K":
        df_s1 = df_s1[df_s1['proxy_purchasing_power_median_chf'].between(70000, 80000)]
    elif inc_f == "> CHF 80K":
        df_s1 = df_s1[df_s1['proxy_purchasing_power_median_chf'] > 80000]
    if st_f == "No stores":
        df_s1 = df_s1[df_s1['STORE_COUNT'] == 0]
    elif st_f == "1–3 stores":
        df_s1 = df_s1[df_s1['STORE_COUNT'].between(1, 3)]
    elif st_f == "4+ stores":
        df_s1 = df_s1[df_s1['STORE_COUNT'] >= 4]

    st.markdown(f"<span class='chip'>{len(df_s1)} of {len(top20)} communes</span>", unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    if df_s1.empty:
        st.warning("No communes match the current filters.")
    else:
        bar_colors = [TEAL if i == 0 else TEAL_D for i in range(len(df_s1))]
        fig = go.Figure(go.Bar(
            x=df_s1['POPULATION'],
            y=df_s1['COMMUNE_NAME'],
            orientation='h',
            marker_color=bar_colors,
            text=[f'{v:,}' for v in df_s1['POPULATION']],
            textposition='outside',
            textfont=dict(color=MUTED, size=9),
            hovertemplate='<b>%{y}</b><br>Population: %{x:,}<extra></extra>',
        ))
        fig.update_layout(**plot_layout(
            height=max(300, len(df_s1) * 36 + 60),
            title=dict(text='Stage 1 — Communes by Resident Population', font=dict(size=13, color=TEXT), x=0),
            xaxis=dict(gridcolor=BORDER, tickfont=dict(color=FAINT, size=9), tickformat=',.0f', zerolinecolor=BORDER),
            yaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT, size=11), autorange='reversed'),
            margin=dict(l=10, r=80, t=40, b=10),
        ))
        st.plotly_chart(fig, use_container_width=True)

        rows = ""
        for i, (_, r) in enumerate(df_s1.iterrows()):
            m = {0: '🥇', 1: '🥈', 2: '🥉'}.get(i, '')
            rows += f"""<tr>
              <td class="mono">{int(r['RANK'])}</td>
              <td><b>{m} {r['COMMUNE_NAME']}</b></td>
              <td>{int(r['POPULATION']):,}</td>
              <td>{int(r['STORE_COUNT'])}</td>
              <td>{r['PCT_FOREIGNERS']:.1f}%</td>
              <td>CHF {int(r['proxy_purchasing_power_median_chf']):,}</td>
            </tr>"""
        st.markdown(f"""
        <div class="t-wrap">
          <div class="t-header">
            <span class="badge">Stage 1 Results</span>
            <span class="chip">{len(df_s1)} communes shown</span>
          </div>
          <table class="et">
            <thead><tr>
              <th>Rank</th><th>Commune</th><th>Population</th>
              <th>Stores</th><th>Foreign %</th><th>Income</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: SCORING
# ─────────────────────────────────────────────────────────────
elif page == "◈  Scoring":
    st.markdown("""
    <div class="eyebrow">Stage 02</div>
    <div class="heading">Socio-Economic Composite Scoring</div>
    <p class="subtext">Multi-factor weighted model applied to Stage 1 finalists. Adjust weights to see how rankings shift in real time.</p>
    <div class="teal-rule"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="filter-wrap"><div class="filter-title">⊘ Score Weights — must sum to 100%</div>', unsafe_allow_html=True)
    wc1, wc2, wc3, wc4 = st.columns(4)
    with wc1: w_inc = st.slider("Income Weight %",  0, 60, 35, 5)
    with wc2: w_for = st.slider("Foreign % Weight", 0, 60, 25, 5)
    with wc3: w_age = st.slider("Working Age %",    0, 60, 20, 5)
    with wc4: w_urb = st.slider("Urban Density %",  0, 60, 20, 5)
    total_w = w_inc + w_for + w_age + w_urb
    tc = TEAL if total_w == 100 else GOLD
    st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:10px;
    margin-top:8px;color:{tc};">Weight total: {total_w}% {"✓ Balanced" if total_w == 100 else "← Adjust to reach 100%"}</div>""",
    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    def mm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0

    s2d = top5.copy()
    s2d['SC_INC'] = mm(s2d['proxy_purchasing_power_median_chf'])
    s2d['SC_FOR'] = mm(s2d['PCT_FOREIGNERS'])
    s2d['SC_AGE'] = mm(s2d['PCT_WORKING_AGE'])
    s2d['SC_URB'] = 1 - mm(s2d['PCT_SINGLE_FAMILY'])
    if total_w > 0:
        s2d['COMPOSITE_SCORE'] = (
            s2d['SC_INC'] * w_inc + s2d['SC_FOR'] * w_for +
            s2d['SC_AGE'] * w_age + s2d['SC_URB'] * w_urb
        ) / 100
        s2d = s2d.sort_values('COMPOSITE_SCORE', ascending=False).reset_index(drop=True)

    # Dimension weight cards
    dims_info = [
        ("Income",             w_inc, TEAL),
        ("Foreign Residents",  w_for, '#00B8D4'),
        ("Working Age",        w_age, '#A78BFA'),
        ("Urban Density",      w_urb, '#34D399'),
    ]
    cols = st.columns(4)
    for col, (lbl, pct, hex_c) in zip(cols, dims_info):
        with col:
            st.markdown(f"""
            <div style="background:{S1};border:1px solid {BORDER};border-top:2px solid {hex_c};
                 border-radius:12px;padding:20px;text-align:center;margin-bottom:14px;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;
                   color:{MUTED};text-transform:uppercase;margin-bottom:10px;">{lbl}</div>
              <div style="font-family:'Instrument Serif',serif;font-size:38px;
                   color:{hex_c};line-height:1;letter-spacing:-1px;">{pct}%</div>
            </div>
            """, unsafe_allow_html=True)

    # Charts
    pal5 = [TEAL, '#00B8D4', '#A78BFA', '#34D399', '#F87171']
    col_a, col_b = st.columns(2)

    with col_a:
        fig1 = go.Figure(go.Bar(
            y=s2d['COMMUNE_NAME'],
            x=s2d['COMPOSITE_SCORE'],
            orientation='h',
            marker_color=pal5[:len(s2d)],
            text=[f'{v:.3f}' for v in s2d['COMPOSITE_SCORE']],
            textposition='outside',
            textfont=dict(color=MUTED, size=10),
            hovertemplate='<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>',
        ))
        fig1.update_layout(**plot_layout(
            height=290,
            title=dict(text='Composite Score', font=dict(size=13, color=TEXT), x=0),
            xaxis=dict(gridcolor=BORDER, tickfont=dict(color=FAINT, size=9), range=[0, 1.15]),
            yaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT, size=11), autorange='reversed'),
            margin=dict(l=10, r=70, t=40, b=10),
        ))
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        dim_keys   = ['SC_INC', 'SC_FOR', 'SC_AGE', 'SC_URB']
        dim_labels = [f'Income ({w_inc}%)', f'Foreign ({w_for}%)', f'Age ({w_age}%)', f'Urban ({w_urb}%)']
        dim_ws     = [w_inc / 100, w_for / 100, w_age / 100, w_urb / 100]
        fig2 = go.Figure()
        for d, lbl, c, w in zip(dim_keys, dim_labels, pal5, dim_ws):
            fig2.add_trace(go.Bar(
                name=lbl, y=s2d['COMMUNE_NAME'], x=s2d[d] * w,
                orientation='h', marker_color=c,
                hovertemplate=f'<b>%{{y}}</b><br>{lbl}: %{{x:.3f}}<extra></extra>',
            ))
        fig2.update_layout(**plot_layout(
            barmode='stack', height=290,
            title=dict(text='Score Breakdown by Dimension', font=dict(size=13, color=TEXT), x=0),
            xaxis=dict(gridcolor=BORDER, tickfont=dict(color=FAINT, size=9)),
            yaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT, size=11), autorange='reversed'),
            margin=dict(l=10, r=10, t=40, b=30),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=9, color=MUTED),
                        orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        ))
        st.plotly_chart(fig2, use_container_width=True)

    # Table
    rows = ""
    for i, (_, r) in enumerate(s2d.iterrows()):
        m = {0: '🥇', 1: '🥈', 2: '🥉'}.get(i, '')
        rows += f"""<tr>
          <td class="mono">{i+1}</td>
          <td><b>{m} {r['COMMUNE_NAME']}</b></td>
          <td>CHF {int(r['proxy_purchasing_power_median_chf']):,}</td>
          <td>{r['PCT_FOREIGNERS']:.1f}%</td>
          <td>{r['PCT_WORKING_AGE']:.1f}%</td>
          <td>{r['PCT_SINGLE_FAMILY']:.1f}%</td>
          <td class="tc"><b>{r['COMPOSITE_SCORE']:.4f}</b></td>
        </tr>"""
    st.markdown(f"""
    <div class="t-wrap" style="margin-top:14px;">
      <div class="t-header"><span class="badge">Stage 2 Results</span></div>
      <table class="et">
        <thead><tr>
          <th>Rank</th><th>Commune</th><th>Income</th>
          <th>Foreign %</th><th>Working Age %</th><th>Single-Family %</th><th>Score</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: REGRESSION
# ─────────────────────────────────────────────────────────────
elif page == "⧖  Regression":
    st.markdown("""
    <div class="eyebrow">Stage 03</div>
    <div class="heading">OLS Regression — Opportunity Gap</div>
    <p class="subtext">The model predicts store count from population and purchasing power. The gap between predicted and actual reveals under-served markets.</p>
    <div class="teal-rule"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="filter-wrap"><div class="filter-title">⊘ View Options</div>', unsafe_allow_html=True)
    oc1, oc2 = st.columns(2)
    with oc1: show_neg = st.checkbox("Show negative opportunity gaps", value=True)
    with oc2: sort_by  = st.selectbox("Sort By", ["Opportunity Gap ↓", "Population ↓", "Predicted ↓"])
    st.markdown('</div>', unsafe_allow_html=True)

    tv = top5_ols.copy()
    if not show_neg:
        tv = tv[tv['OPPORTUNITY'] >= 0]
    if sort_by == "Population ↓":
        tv = tv.sort_values('POPULATION', ascending=False)
    elif sort_by == "Predicted ↓":
        tv = tv.sort_values('PREDICTED', ascending=False)

    pal_tv = [TEAL if i == 0 else TEAL_D for i in range(len(tv))]

    col_a, col_b = st.columns(2)

    with col_a:
        fig1 = go.Figure(go.Bar(
            y=tv['COMMUNE_NAME'],
            x=tv['OPPORTUNITY'],
            orientation='h',
            marker_color=pal_tv,
            text=[f"+{v:.2f}" if v >= 0 else f"{v:.2f}" for v in tv['OPPORTUNITY']],
            textposition='outside',
            textfont=dict(color=MUTED, size=10),
            hovertemplate='<b>%{y}</b><br>Gap: %{x:.2f}<extra></extra>',
        ))
        fig1.add_vline(x=0, line_color=FAINT, line_dash='dot', line_width=1)
        fig1.update_layout(**plot_layout(
            height=290,
            title=dict(text='Opportunity Gap (Predicted − Actual)', font=dict(size=13, color=TEXT), x=0),
            xaxis=dict(gridcolor=BORDER, tickfont=dict(color=FAINT, size=9), zerolinecolor=FAINT),
            yaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT, size=11), autorange='reversed'),
            margin=dict(l=10, r=70, t=40, b=10),
        ))
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        if not tv.empty:
            all_vals = list(tv['STORE_COUNT']) + list(tv['PREDICTED'].dropna())
            lim = max(all_vals) + 1
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=[0, lim], y=[0, lim], mode='lines',
                line=dict(color=FAINT, dash='dot', width=1),
                name='Equilibrium', hoverinfo='skip',
            ))
            for i, (_, r) in enumerate(tv.iterrows()):
                c = TEAL if i == 0 else TEAL_D
                fig2.add_trace(go.Scatter(
                    x=[r['STORE_COUNT']], y=[r['PREDICTED']],
                    mode='markers+text',
                    marker=dict(color=c, size=14, line=dict(color='white', width=1.5)),
                    text=[r['COMMUNE_NAME']], textposition='top right',
                    textfont=dict(color=TEXT, size=9),
                    name=r['COMMUNE_NAME'],
                    hovertemplate=f"<b>{r['COMMUNE_NAME']}</b><br>Actual: {r['STORE_COUNT']}<br>Predicted: {r['PREDICTED']:.2f}<extra></extra>",
                ))
            fig2.update_layout(**plot_layout(
                height=290,
                title=dict(text='Actual vs Predicted Stores', font=dict(size=13, color=TEXT), x=0),
                xaxis=dict(title='Actual', gridcolor=BORDER, tickfont=dict(color=FAINT, size=9), range=[-0.2, lim]),
                yaxis=dict(title='Predicted', gridcolor=BORDER, tickfont=dict(color=FAINT, size=9), range=[-0.2, lim]),
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
            ))
            st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Model Summary — OLS Regression Statistics"):
        Xf2 = sm.add_constant(df_f[['POPULATION', 'proxy_purchasing_power_median_chf']])
        yf2 = df_f['STORE_COUNT']
        st.code(str(sm.OLS(yf2, Xf2).fit().summary()), language='text')

    # Table
    rows = ""
    for i, (_, r) in enumerate(tv.iterrows()):
        bdg = {0: '🥇 ★', 1: '🥈', 2: '🥉'}.get(i, '')
        gap = f"+{r['OPPORTUNITY']:.2f}" if r['OPPORTUNITY'] >= 0 else f"{r['OPPORTUNITY']:.2f}"
        gc = 'tc' if r['OPPORTUNITY'] == tv['OPPORTUNITY'].max() else ''
        rows += f"""<tr>
          <td class="mono">{int(r['RANK'])}</td>
          <td><b>{bdg} {r['COMMUNE_NAME']}</b></td>
          <td>{int(r['POPULATION']):,}</td>
          <td>{int(r['STORE_COUNT'])}</td>
          <td>{r['PREDICTED']:.2f}</td>
          <td class="{gc}"><b>{gap}</b></td>
          <td>{r['COMPOSITE_SCORE']:.4f}</td>
        </tr>"""
    st.markdown(f"""
    <div class="t-wrap" style="margin-top:14px;">
      <div class="t-header"><span class="badge">Stage 3 Results</span></div>
      <table class="et">
        <thead><tr>
          <th>Rank</th><th>Commune</th><th>Population</th><th>Stores</th>
          <th>Predicted</th><th>Opportunity</th><th>Score</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Champion
    cn = champion['COMMUNE_NAME']
    mx = ""
    for lbl, val, cls in [
        ("Population",  f"{int(champion['POPULATION']):,}",                        ""),
        ("Stores Now",  f"{int(champion['STORE_COUNT'])}",                         ""),
        ("Predicted",   f"{champion['PREDICTED']:.2f}",                            ""),
        ("Opportunity", f"+{champion['OPPORTUNITY']:.2f}",                         "teal"),
        ("Income CHF",  f"{int(champion['proxy_purchasing_power_median_chf']):,}", ""),
        ("Foreign %",   f"{champion['PCT_FOREIGNERS']:.1f}%",                      ""),
    ]:
        mx += f'<div class="cm"><div class="cm-label">{lbl}</div><div class="cm-value {cls}">{val}</div></div>'

    st.markdown(f"""
    <div class="champion" style="margin-top:28px;">
      <div class="champ-eyebrow">★ Stage 3 Champion</div>
      <div class="champ-name">{cn}</div>
      <div class="champ-place">Canton of Geneva · Switzerland</div>
      <div class="champ-metrics">{mx}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: DEMOGRAPHICS
# ─────────────────────────────────────────────────────────────
elif page == "▤  Demographics":
    st.markdown("""
    <div class="eyebrow">Analytics</div>
    <div class="heading">5-Factor Demographic Analysis</div>
    <p class="subtext">Regression analysis across all communes. Each panel shows the relationship between a demographic variable and store count. Champion is highlighted.</p>
    <div class="teal-rule"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="filter-wrap"><div class="filter-title">⊘ Options</div>', unsafe_allow_html=True)
    dc1, dc2, dc3 = st.columns(3)
    with dc1: min_pop = st.slider("Min Population", 0, 20000, 0, 1000)
    with dc2: hl_opt  = st.selectbox("Highlight Commune", ["Champion"] + list(df_f['COMMUNE_NAME'].sort_values()))
    with dc3: show_tl = st.checkbox("Show trend line", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    cn = champion['COMMUNE_NAME']
    df_demo = df_f.copy()
    if min_pop > 0:
        df_demo = df_demo[df_demo['POPULATION'] >= min_pop]
    hl_name = cn if hl_opt == "Champion" else hl_opt

    panels = [
        ('POPULATION',                        'Population',         '#4ECDC4'),
        ('PCT_WORKING_AGE',                   'Working-Age %',      '#45B7D1'),
        ('PCT_SINGLE_FAMILY',                 'Single-Family %',    '#A78BFA'),
        ('PCT_FOREIGNERS',                    'Foreign Residents %', '#34D399'),
        ('proxy_purchasing_power_median_chf', 'Median Income (CHF)', '#FFB347'),
    ]

    def scatter_panel(xcol, xlabel, color, height=250):
        hl_data = df_demo[df_demo['COMMUNE_NAME'] == hl_name]
        fig = go.Figure()
        if show_tl:
            from numpy.polynomial import polynomial as P
            x_all = df_demo[xcol].values
            y_all = df_demo['STORE_COUNT'].values
            mask  = ~np.isnan(x_all) & ~np.isnan(y_all)
            if mask.sum() > 2:
                c = np.polyfit(x_all[mask], y_all[mask], 1)
                x_line = np.linspace(x_all[mask].min(), x_all[mask].max(), 80)
                fig.add_trace(go.Scatter(
                    x=x_line, y=np.polyval(c, x_line),
                    mode='lines', line=dict(color=TEAL, width=1.8, dash='solid'),
                    name='Trend', hoverinfo='skip',
                ))
        fig.add_trace(go.Scatter(
            x=df_demo[xcol], y=df_demo['STORE_COUNT'],
            mode='markers', name='Commune',
            marker=dict(color=color, size=7, opacity=0.6, line=dict(color='white', width=0.5)),
            customdata=df_demo[['COMMUNE_NAME']].values,
            hovertemplate='<b>%{customdata[0]}</b><br>' + xlabel + ': %{x}<br>Stores: %{y}<extra></extra>',
        ))
        if not hl_data.empty:
            fig.add_trace(go.Scatter(
                x=hl_data[xcol], y=hl_data['STORE_COUNT'],
                mode='markers', name=f'★ {hl_name}',
                marker=dict(color=GOLD, size=16, symbol='star', line=dict(color='white', width=1.5)),
                hovertemplate=f'<b>★ {hl_name}</b><br>{xlabel}: %{{x}}<br>Stores: %{{y}}<extra></extra>',
            ))
        fig.update_layout(**plot_layout(
            height=height,
            title=dict(text=f'{xlabel} vs Stores', font=dict(size=12, color=TEXT), x=0),
            xaxis=dict(title=dict(text=xlabel, font=dict(size=9, color=MUTED)),
                       gridcolor=BORDER, tickfont=dict(color=FAINT, size=8)),
            yaxis=dict(title=dict(text='Stores', font=dict(size=9, color=MUTED)),
                       gridcolor=BORDER, tickfont=dict(color=FAINT, size=8)),
            margin=dict(l=10, r=10, t=36, b=10),
            showlegend=False,
        ))
        return fig

    row1 = st.columns(3)
    for (xcol, xlabel, color), col in zip(panels[:3], row1):
        with col:
            st.plotly_chart(scatter_panel(xcol, xlabel, color), use_container_width=True)

    row2 = st.columns([1, 1, 1])
    for (xcol, xlabel, color), col in zip(panels[3:], row2[:2]):
        with col:
            st.plotly_chart(scatter_panel(xcol, xlabel, color), use_container_width=True)

    with row2[2]:
        hl_row = df_demo[df_demo['COMMUNE_NAME'] == hl_name]
        if not hl_row.empty:
            h = hl_row.iloc[0]
            st.markdown(f"""
            <div style="background:{S2};border:1px solid rgba(0,201,167,0.2);border-radius:12px;
                 padding:28px 24px;height:250px;box-sizing:border-box;display:flex;
                 flex-direction:column;justify-content:center;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2.5px;
                   color:{TEAL};text-transform:uppercase;margin-bottom:12px;">★ Highlight</div>
              <div style="font-family:'Instrument Serif',serif;font-size:22px;
                   color:{TEXT};margin-bottom:16px;line-height:1.2;">{hl_name}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;
                   color:{MUTED};line-height:2.1;">
                POPULATION &nbsp;&nbsp; <span style="color:{TEXT};">{int(h['POPULATION']):,}</span><br>
                STORES &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{TEXT};">{int(h['STORE_COUNT'])}</span><br>
                PREDICTED &nbsp;&nbsp; <span style="color:{TEXT};">{h['PREDICTED']:.2f}</span><br>
                GAP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{TEAL};font-weight:600;">+{h['OPPORTUNITY']:.2f}</span><br>
                INCOME &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{TEXT};">CHF {int(h['proxy_purchasing_power_median_chf']):,}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: MAP
# ─────────────────────────────────────────────────────────────
elif page == "⊕  Map":
    st.markdown("""
    <div class="eyebrow">Geospatial</div>
    <div class="heading">Interactive Market Map</div>
    <p class="subtext">Choropleth shows the opportunity gap per commune. Circle pins mark existing stores. The star marks the champion expansion target.</p>
    <div class="teal-rule"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="filter-wrap"><div class="filter-title">⊘ Map Options</div>', unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    with mc1: show_stores  = st.checkbox("Show store pins", value=True)
    with mc2: brand_filter = st.multiselect("Filter Brands", ["Coop", "Migros", "other"], default=["Coop", "Migros", "other"])
    with mc3: map_zoom     = st.slider("Zoom Level", 10, 14, 12)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Building map…"):
        lat0 = bounds.geometry.centroid.y.mean()
        lon0 = bounds.geometry.centroid.x.mean()
        m = folium.Map(location=[lat0, lon0], zoom_start=map_zoom, tiles='cartodbdark_matter')

        folium.Choropleth(
            geo_data=bounds, name='Opportunity Gap',
            data=df_f, columns=['COMMUNE_NAME', 'OPPORTUNITY'],
            key_on='feature.properties.COMMUNE_NAME',
            fill_color='YlGn', fill_opacity=0.68,
            line_opacity=0.28, line_color='#ffffff',
            legend_name='Market Opportunity Gap (higher = under-served)',
            nan_fill_color='#1a1a2e', nan_fill_opacity=0.4,
        ).add_to(m)

        tooltip_gdf = bounds.merge(
            df_f[['COMMUNE_NAME', 'OPPORTUNITY', 'POPULATION', 'STORE_COUNT']],
            on='COMMUNE_NAME', how='left',
        )
        folium.GeoJson(
            tooltip_gdf,
            style_function=lambda _: {'fillOpacity': 0, 'color': '#ffffff', 'weight': 0.3},
            tooltip=folium.GeoJsonTooltip(
                fields=['COMMUNE_NAME', 'POPULATION', 'STORE_COUNT', 'OPPORTUNITY'],
                aliases=['Commune', 'Population', 'Stores', 'Gap'],
                style='background:#0d0f1a;color:#edf1f8;font-family:monospace;font-size:12px;'
                      'border:1px solid #1c2038;border-radius:8px;padding:10px;',
            ),
        ).add_to(m)

        # Champion marker
        cn = champion['COMMUNE_NAME']
        cg = df_f[df_f['COMMUNE_NAME'] == cn]
        if not cg.empty:
            cx = cg.geometry.centroid.iloc[0].x
            cy = cg.geometry.centroid.iloc[0].y
            icon_html = (
                '<div style="background:#00C9A7;border:3px solid #fff;border-radius:50%;'
                'width:42px;height:42px;display:flex;align-items:center;justify-content:center;'
                'font-size:20px;box-shadow:0 0 22px #00C9A7,0 0 44px rgba(0,201,167,0.4);">'
                '★</div>'
            )
            popup_html = (
                f"<div style='font-family:monospace;background:#07080d;color:#edf1f8;"
                f"padding:18px;border-radius:10px;border:2px solid #00C9A7;min-width:220px;'>"
                f"<b style='color:#00C9A7;font-size:11px;letter-spacing:1.5px;'>CHAMPION TARGET</b><br><br>"
                f"<b style='font-size:16px;color:#fff;'>{cn}</b><br>"
                f"<span style='color:#8494a9;font-size:11px;'>Canton of Geneva</span><br><br>"
                f"<span style='color:#8494a9;'>Population </span>{int(champion['POPULATION']):,}<br>"
                f"<span style='color:#8494a9;'>Stores now </span>{int(champion['STORE_COUNT'])}<br>"
                f"<span style='color:#8494a9;'>Predicted &nbsp;</span>{champion['PREDICTED']:.2f}<br>"
                f"<b style='color:#00C9A7;'>Gap: +{champion['OPPORTUNITY']:.2f}</b><br><br>"
                f"<span style='color:#8494a9;'>Income CHF </span>{int(champion['proxy_purchasing_power_median_chf']):,}"
                f"</div>"
            )
            folium.Marker(
                location=[cy, cx],
                popup=folium.Popup(popup_html, max_width=270),
                tooltip=f'★ CHAMPION: {cn}',
                icon=folium.DivIcon(html=icon_html, icon_size=(46, 46), icon_anchor=(23, 23)),
            ).add_to(m)

        if show_stores:
            brand_colors = {'Coop': '#FFD700', 'Migros': '#FF4444', 'other': '#00C9A7'}
            for _, row in joined.iterrows():
                brand = row.get('brand_category', 'other')
                if brand not in brand_filter:
                    continue
                if row['COMMUNE_NAME'] in df_f['COMMUNE_NAME'].values:
                    bc = brand_colors.get(brand, brand_colors['other'])
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5, color=bc, fill=True, fill_color=bc,
                        fill_opacity=0.80, weight=1.5,
                        tooltip=f"{brand} · {row['COMMUNE_NAME']}",
                        popup=folium.Popup(
                            f"<div style='font-family:monospace;font-size:11px;background:#07080d;"
                            f"color:#edf1f8;padding:10px;border-radius:6px;'>"
                            f"<b style='color:{bc};'>{brand}</b><br>"
                            f"Type: {row.get('shop', '?')}<br>"
                            f"Commune: {row['COMMUNE_NAME']}</div>",
                            max_width=180,
                        ),
                    ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

    st_folium(m, height=650, use_container_width=True, returned_objects=[])
