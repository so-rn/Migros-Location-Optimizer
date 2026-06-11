import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import os
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Migros Location Intelligence",
    page_icon="🟠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── BRAND PALETTE ─────────────────────────────────────────────
MIGROS_ORANGE = '#FF6B00'
ORANGE_LIGHT  = '#FF8A3D'
MIGROS_TEAL   = '#4FB3E8'
BG_DARK       = '#0B0F19'
CARD_BG       = '#121826'
TEXT_LIGHT    = '#EDF1F7'
TEXT_MUTED    = '#94A3B8'
BORDER        = '#1E293B'

# ─── GLOBAL CSS ────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

  /* ── Base ──────────────────────────────────────────────── */
  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {BG_DARK};
    color: {TEXT_LIGHT};
    -webkit-font-smoothing: antialiased;
  }}
  .stApp {{
    background: radial-gradient(ellipse 80% 50% at 15% 0%, rgba(255,107,0,0.05) 0%, transparent 60%),
                radial-gradient(ellipse 70% 50% at 90% 100%, rgba(79,179,232,0.04) 0%, transparent 60%),
                {BG_DARK};
  }}
  .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1380px; }}

  /* Hide default Streamlit chrome */
  #MainMenu, footer {{ visibility: hidden; }}
  header[data-testid="stHeader"] {{ background: transparent; }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: #2A3648; border-radius: 4px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {MIGROS_ORANGE}; }}

  /* ── Sidebar ───────────────────────────────────────────── */
  [data-testid="stSidebar"] {{
    background: #0D121D;
    border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] * {{ color: {TEXT_LIGHT}; }}

  /* Sidebar nav: radio styled as menu pills */
  [data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; }}
  [data-testid="stSidebar"] [role="radiogroup"] label {{
    width: 100%;
    padding: 9px 14px !important;
    margin: 0 !important;
    border-radius: 10px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
    font-size: 13.5px;
  }}
  [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background: rgba(255,255,255,0.04);
  }}
  [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background: rgba(255,107,0,0.10);
    border-color: rgba(255,107,0,0.35);
  }}
  [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{
    color: {ORANGE_LIGHT} !important;
    font-weight: 600;
  }}
  /* Hide the radio dot so it reads as a nav menu */
  [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
  [data-testid="stSidebar"] .stRadio > label {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2.5px !important;
    color: {TEXT_MUTED} !important;
    text-transform: uppercase !important;
    margin-bottom: 6px;
  }}

  /* ── Top header bar ────────────────────────────────────── */
  .top-bar {{
    background: linear-gradient(135deg, #121826 0%, #151C2C 60%, #18202F 100%);
    border: 1px solid {BORDER};
    padding: 20px 28px;
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 32px;
    border-radius: 16px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
  }}
  .top-bar::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {MIGROS_ORANGE}, {ORANGE_LIGHT} 30%, transparent 70%);
  }}
  .top-bar .logo {{
    background: linear-gradient(135deg, {MIGROS_ORANGE}, #E55A00);
    color: white;
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 20px;
    padding: 10px 17px;
    border-radius: 12px;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 18px rgba(255,107,0,0.35);
  }}
  .top-bar .title {{
    font-family: 'Sora', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: {TEXT_LIGHT};
    letter-spacing: 0.2px;
  }}
  .top-bar .subtitle {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    color: {TEXT_MUTED};
    margin-top: 4px;
    letter-spacing: 0.8px;
  }}
  .top-bar .live-dot {{
    width: 7px; height: 7px;
    background: #34D399;
    border-radius: 50%;
    display: inline-block;
    margin-right: 7px;
    animation: pulse 2.4s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.35; }}
  }}

  /* ── KPI cards ─────────────────────────────────────────── */
  .kpi-grid {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 4px 0 28px; }}
  .kpi-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 20px 22px;
    min-width: 150px;
    flex: 1;
    position: relative;
    overflow: hidden;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    cursor: default;
  }}
  .kpi-card:hover {{
    transform: translateY(-2px);
    border-color: rgba(255,107,0,0.45);
    box-shadow: 0 10px 28px rgba(0,0,0,0.35);
  }}
  .kpi-card .kpi-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    color: {TEXT_MUTED};
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-bottom: 10px;
  }}
  .kpi-card .kpi-value {{
    font-family: 'Sora', sans-serif;
    font-size: 30px;
    font-weight: 700;
    color: {TEXT_LIGHT};
    line-height: 1.1;
  }}
  .kpi-card .kpi-accent {{ color: {MIGROS_ORANGE}; }}
  .kpi-card .kpi-sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px;
    color: {TEXT_MUTED};
    margin-top: 7px;
    letter-spacing: 1px;
  }}

  /* ── Section headings ──────────────────────────────────── */
  .stage-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,107,0,0.10);
    color: {ORANGE_LIGHT};
    border: 1px solid rgba(255,107,0,0.30);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    padding: 4px 13px;
    border-radius: 20px;
    letter-spacing: 2px;
    margin-bottom: 10px;
    text-transform: uppercase;
  }}
  .section-title {{
    font-family: 'Sora', sans-serif;
    font-size: 23px;
    font-weight: 700;
    color: {TEXT_LIGHT};
    margin-bottom: 6px;
    line-height: 1.3;
    letter-spacing: -0.2px;
  }}
  .section-sub {{
    font-size: 13.5px;
    color: {TEXT_MUTED};
    margin-bottom: 26px;
    line-height: 1.5;
  }}

  /* ── Champion card ─────────────────────────────────────── */
  .champion-card {{
    background: linear-gradient(160deg, #1A1208 0%, #16110C 45%, {CARD_BG} 100%);
    border: 1px solid rgba(255,107,0,0.40);
    border-radius: 18px;
    padding: 40px 36px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
    margin: 28px 0;
    position: relative;
    overflow: hidden;
  }}
  .champion-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, {MIGROS_ORANGE}, transparent);
  }}
  .champion-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    color: {ORANGE_LIGHT};
    font-weight: 700;
    letter-spacing: 4px;
    margin-bottom: 16px;
    text-transform: uppercase;
  }}
  .champion-name {{
    font-family: 'Sora', sans-serif;
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
  }}
  .champion-loc {{
    font-size: 13px;
    color: {TEXT_MUTED};
    margin-bottom: 30px;
    letter-spacing: 0.4px;
  }}
  .champion-stats {{
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .champion-stat {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 14px 20px;
    min-width: 116px;
    transition: background 0.2s, border-color 0.2s, transform 0.2s;
  }}
  .champion-stat:hover {{
    background: rgba(255,107,0,0.07);
    border-color: rgba(255,107,0,0.30);
    transform: translateY(-2px);
  }}
  .champion-stat-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px;
    color: {TEXT_MUTED};
    letter-spacing: 1.6px;
    margin-bottom: 8px;
    text-transform: uppercase;
  }}
  .champion-stat-value {{
    font-family: 'Sora', sans-serif;
    font-size: 19px;
    font-weight: 700;
  }}

  /* ── Funnel viz ────────────────────────────────────────── */
  .funnel-step {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-left: 3px solid {MIGROS_ORANGE};
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 20px;
    transition: transform 0.18s, border-color 0.18s, box-shadow 0.18s;
  }}
  .funnel-step:hover {{
    transform: translateX(5px);
    border-color: rgba(255,107,0,0.35);
    box-shadow: 0 6px 22px rgba(0,0,0,0.30);
  }}
  .funnel-num {{
    font-family: 'Sora', sans-serif;
    font-size: 25px;
    font-weight: 700;
    color: {MIGROS_ORANGE};
    min-width: 64px;
  }}
  .funnel-title {{
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    font-size: 14.5px;
  }}
  .funnel-sub {{
    font-size: 12px;
    color: {TEXT_MUTED};
    margin-top: 4px;
    line-height: 1.4;
  }}

  /* ── Table ─────────────────────────────────────────────── */
  .intel-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .intel-table th {{
    background: rgba(255,255,255,0.02);
    padding: 11px 16px;
    text-align: left;
    color: {TEXT_MUTED};
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px;
    font-weight: 600;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(255,107,0,0.25);
  }}
  .intel-table td {{
    padding: 11px 16px;
    border-bottom: 1px solid {BORDER};
    color: {TEXT_LIGHT};
    font-size: 13.5px;
  }}
  .intel-table tr:last-child td {{ border-bottom: none; }}
  .intel-table tr:hover td {{ background: rgba(255,107,0,0.045); transition: background 0.15s; }}

  /* ── Buttons ───────────────────────────────────────────── */
  .stButton > button {{
    background: linear-gradient(135deg, {MIGROS_ORANGE}, #E55A00) !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 26px !important;
    letter-spacing: 0.4px !important;
    font-size: 13px !important;
    transition: all 0.18s !important;
    box-shadow: 0 4px 14px rgba(255,107,0,0.25) !important;
  }}
  .stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255,107,0,0.40) !important;
  }}

  /* ── Inputs ────────────────────────────────────────────── */
  div[data-testid="stSelectbox"] > div > div,
  div[data-testid="stMultiSelect"] > div > div {{
    background: {CARD_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT_LIGHT} !important;
  }}
  div[data-testid="stMetric"] {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 16px 20px;
  }}
  label, .stSelectbox label, .stSlider label, .stMultiSelect label, .stCheckbox label p {{
    color: {TEXT_MUTED} !important;
    font-size: 12px !important;
    font-weight: 500 !important;
  }}
  .stSelectbox > label p, .stSlider > label p, .stMultiSelect > label p {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 1.6px !important;
    text-transform: uppercase !important;
  }}

  div[data-testid="stExpander"] {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
  }}

  /* ── Filter panel ──────────────────────────────────────── */
  .filter-panel {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 20px 24px 8px;
    margin-bottom: 22px;
  }}
  .filter-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    color: {TEXT_MUTED};
    letter-spacing: 2.2px;
    text-transform: uppercase;
    margin-bottom: 12px;
  }}

  /* Horizontal divider */
  hr {{ border: none; border-top: 1px solid {BORDER}; margin: 20px 0; }}

  /* Tooltip chip */
  .chip {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(255,255,255,0.03);
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 4px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: {TEXT_MUTED};
    margin-right: 6px;
    letter-spacing: 0.5px;
  }}

  /* Charts */
  .stPlotlyChart, .stPyplot, [data-testid="stImage"] {{ border-radius: 14px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ──────────────────────────────────────────────
ACCENT_PURPLE = '#A78BFA'
ACCENT_GREEN  = '#34D399'
ACCENT_RED    = '#F87171'
ACCENT_GOLD   = '#FBBF24'
GRID          = 'rgba(148,163,184,0.10)'

_axis = dict(
    gridcolor=GRID, zerolinecolor='rgba(148,163,184,0.22)',
    linecolor='rgba(148,163,184,0.18)',
    tickfont=dict(color=TEXT_MUTED, size=11.5),
    title=dict(font=dict(color=TEXT_MUTED, size=12)),
)
pio.templates['migros_dark'] = go.layout.Template(layout=go.Layout(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color=TEXT_LIGHT, size=13),
    title=dict(font=dict(family='Sora, sans-serif', size=16, color=TEXT_LIGHT),
               x=0.01, xanchor='left'),
    xaxis=_axis, yaxis=_axis,
    legend=dict(bgcolor='rgba(18,24,38,0.85)', bordercolor=BORDER, borderwidth=1,
                font=dict(size=11.5, color=TEXT_LIGHT)),
    hoverlabel=dict(bgcolor='#1A2232', bordercolor='rgba(255,107,0,0.55)',
                    font=dict(family='Inter, sans-serif', size=12.5, color=TEXT_LIGHT)),
    margin=dict(l=12, r=12, t=56, b=12),
    colorway=[MIGROS_ORANGE, MIGROS_TEAL, ACCENT_PURPLE, ACCENT_GREEN, ACCENT_RED, ACCENT_GOLD],
))
pio.templates.default = 'migros_dark'
PLOTLY_CFG = {'displayModeBar': False}


def show_chart(fig, height=None):
    if height:
        fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

# ─── DATA LOADING (CACHED) ─────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DATA_URL = 'https://raw.githubusercontent.com/so-rn/Migros-Location-Optimizer/main/data/'


def _data_path(rel):
    """Prefer the file bundled in the repo; fall back to the GitHub raw URL."""
    local = os.path.join(DATA_DIR, rel)
    return local if os.path.exists(local) else DATA_URL + rel


@st.cache_data(show_spinner=False)
def load_data():
    URL_STORES = _data_path('geneva_supermarkets_data_with_address.csv')
    URL_POP    = _data_path('OCS_POPBATLOG_COMMUNE.csv')
    URL_POWER  = _data_path('finance/geneva_purchasing_power_proxy_all_years.csv')
    df_stores = pd.read_csv(URL_STORES)
    stores_gdf = gpd.GeoDataFrame(
        df_stores,
        geometry=gpd.points_from_xy(df_stores.longitude, df_stores.latitude),
        crs='EPSG:4326'
    )
    df_pop = pd.read_csv(URL_POP, sep=';')
    df_pop['COMMUNE'] = df_pop['COMMUNE'].str.strip()
    df_power = pd.read_csv(URL_POWER)
    df_power_2022 = df_power[df_power['year'] == 2022].copy()
    df_power_2022['commune'] = df_power_2022['commune'].str.strip()
    return stores_gdf, df_pop, df_power_2022

@st.cache_data(show_spinner=False)
def load_boundaries():
    """Commune boundaries from the bundled GeoJSON — instant startup, no live
    OSM/Nominatim call (which used to hang the whole app on Streamlit Cloud)."""
    boundaries = gpd.read_file(_data_path('geneva_communes_boundaries.geojson'))
    return boundaries[['COMMUNE_NAME', 'geometry']].set_crs(epsg=4326, allow_override=True)

@st.cache_data(show_spinner=False)
def build_master(_stores_gdf, _df_pop, _df_power_2022, _boundaries):
    joined = gpd.sjoin(_stores_gdf, _boundaries, how='inner', predicate='intersects')
    # A store sitting exactly on a shared border matches two polygons — keep one
    joined = joined[~joined.index.duplicated(keep='first')]
    store_counts = joined.groupby('COMMUNE_NAME').size().reset_index(name='STORE_COUNT')
    df_master = (
        _boundaries
        .merge(_df_pop, left_on='COMMUNE_NAME', right_on='COMMUNE', how='left')
        .merge(_df_power_2022[['commune', 'proxy_purchasing_power_median_chf',
                               'proxy_spread_iqr_chf', 'proxy_index_canton_100']],
               left_on='COMMUNE_NAME', right_on='commune', how='left')
        .merge(store_counts, on='COMMUNE_NAME', how='left')
    )
    df_master['STORE_COUNT'] = df_master['STORE_COUNT'].fillna(0)

    # ── Engineered demographic features ───────────────────────
    pop = df_master['POPULATION']
    df_master['PCT_WORKING_AGE']      = (df_master['AGE_20_64']    / pop) * 100
    df_master['PCT_SENIORS']          = (df_master['AGE_65_PLUS']  / pop) * 100
    df_master['PCT_YOUTH']            = (df_master['AGE_0_19']     / pop) * 100
    df_master['PCT_FOREIGNERS']       = (df_master['POP_ETR']      / pop) * 100
    df_master['PCT_SINGLE_FAMILY']    = (df_master['MAISON_INDIV'] / df_master['BATLOG_TOT']) * 100
    df_master['PCT_APARTMENT_BLOCKS'] = (df_master['BATLOG_10P']   / df_master['BATLOG_TOT']) * 100
    df_master['DWELLINGS']            = df_master['LOG_TOTAL']
    # Density from official area (SHAPE.AREA is in m²) → people per km²
    df_master['POP_DENSITY']          = pop / (df_master['SHAPE.AREA'] / 1e6)
    df_master['INCOME_IQR']           = df_master['proxy_spread_iqr_chf']
    # Current retail saturation
    df_master['STORES_PER_10K']       = (df_master['STORE_COUNT'] / pop) * 10000

    df_clean = df_master.dropna(subset=['POPULATION', 'proxy_purchasing_power_median_chf']).copy()
    df_clean = df_clean[~df_clean['COMMUNE_NAME'].isin(['Genève', 'Geneve', 'Geneva'])].reset_index(drop=True)
    return df_clean, joined


# ── Rich feature set shared by the ML model ────────────────────
ML_FEATURES = [
    'POPULATION', 'POP_DENSITY', 'PCT_WORKING_AGE', 'PCT_SENIORS',
    'PCT_FOREIGNERS', 'PCT_SINGLE_FAMILY', 'PCT_APARTMENT_BLOCKS',
    'DWELLINGS', 'proxy_purchasing_power_median_chf', 'INCOME_IQR',
]
ML_FEATURE_LABELS = {
    'POPULATION': 'Population', 'POP_DENSITY': 'Population density',
    'PCT_WORKING_AGE': 'Working-age %', 'PCT_SENIORS': 'Seniors %',
    'PCT_FOREIGNERS': 'Foreign residents %', 'PCT_SINGLE_FAMILY': 'Single-family %',
    'PCT_APARTMENT_BLOCKS': 'Apartment-block %', 'DWELLINGS': 'Dwellings',
    'proxy_purchasing_power_median_chf': 'Purchasing power', 'INCOME_IQR': 'Income spread (IQR)',
}


@st.cache_data(show_spinner=False)
def run_pipeline(_df_clean):
    df = _df_clean.copy()

    # ═══ STAGE 1 — population shortlist ════════════════════════
    top20 = df.sort_values('POPULATION', ascending=False).head(20).copy().reset_index(drop=True)
    top20['RANK'] = range(1, len(top20) + 1)

    # ═══ STAGE 2 — socio-economic composite score ══════════════
    def minmax(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0.0
    s2 = top20.copy()
    s2['SCORE_INCOME']  = minmax(s2['proxy_purchasing_power_median_chf'])
    s2['SCORE_FOREIGN'] = minmax(s2['PCT_FOREIGNERS'])
    s2['SCORE_AGE']     = minmax(s2['PCT_WORKING_AGE'])
    s2['SCORE_URBAN']   = 1 - minmax(s2['PCT_SINGLE_FAMILY'])
    WEIGHTS = {'SCORE_INCOME': 0.35, 'SCORE_FOREIGN': 0.25, 'SCORE_AGE': 0.20, 'SCORE_URBAN': 0.20}
    s2['COMPOSITE_SCORE'] = sum(s2[k] * w for k, w in WEIGHTS.items())
    s2 = s2.sort_values('COMPOSITE_SCORE', ascending=False).reset_index(drop=True)
    top5 = s2.head(5).copy()
    top5['RANK'] = range(1, len(top5) + 1)

    # ═══ STAGE 3a — OLS baseline (interpretable) ═══════════════
    ols_feats = ['POPULATION', 'proxy_purchasing_power_median_chf']
    X_init = sm.add_constant(df[ols_feats])
    init_model = sm.OLS(df['STORE_COUNT'], X_init).fit()
    cooks_d = init_model.get_influence().cooks_distance[0]
    df_f = df.copy()
    df_f['COOKS_D'] = cooks_d
    df_filtered = df_f[df_f['COOKS_D'] <= 4 / len(df_f)].copy().reset_index(drop=True)

    X_final = sm.add_constant(df_filtered[ols_feats])
    final_model = sm.OLS(df_filtered['STORE_COUNT'], X_final).fit()
    ols_pred = final_model.get_prediction(X_final)
    df_filtered['PREDICTED_STORES']     = ols_pred.predicted_mean
    df_filtered['OLS_PRED']             = ols_pred.predicted_mean
    ci = ols_pred.conf_int(obs=True)
    df_filtered['OLS_PRED_LO']          = ci[:, 0]
    df_filtered['OLS_PRED_HI']          = ci[:, 1]
    df_filtered['OPPORTUNITY_SCORE']    = df_filtered['PREDICTED_STORES'] - df_filtered['STORE_COUNT']
    df_filtered['OLS_RESID']            = df_filtered['STORE_COUNT'] - df_filtered['PREDICTED_STORES']

    # ═══ STAGE 3b — Gradient Boosting (richer, non-linear) ═════
    feat = [f for f in ML_FEATURES if f in df_filtered.columns]
    Xml = df_filtered[feat].apply(lambda c: c.fillna(c.median()))
    yml = df_filtered['STORE_COUNT']

    gbr = GradientBoostingRegressor(n_estimators=250, max_depth=2,
                                    learning_rate=0.05, subsample=0.9, random_state=42)
    # Honest out-of-sample predictions via K-fold CV (small-sample friendly)
    k = min(5, max(2, len(df_filtered) // 6))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_pred = cross_val_predict(gbr, Xml, yml, cv=kf)
    cv_r2   = r2_score(yml, cv_pred)
    cv_rmse = float(np.sqrt(mean_squared_error(yml, cv_pred)))
    cv_mae  = float(mean_absolute_error(yml, cv_pred))

    gbr.fit(Xml, yml)
    df_filtered['ML_PRED']           = gbr.predict(Xml)
    df_filtered['ML_CV_PRED']        = cv_pred
    df_filtered['ML_OPPORTUNITY']    = df_filtered['ML_PRED'] - df_filtered['STORE_COUNT']
    df_filtered['ML_RESID']          = df_filtered['STORE_COUNT'] - df_filtered['ML_PRED']

    importances = (pd.DataFrame({'feature': feat, 'importance': gbr.feature_importances_})
                   .sort_values('importance', ascending=False).reset_index(drop=True))

    ols_r2   = float(final_model.rsquared)
    ols_pred_in = final_model.predict(X_final)
    ols_rmse = float(np.sqrt(mean_squared_error(df_filtered['STORE_COUNT'], ols_pred_in)))

    # ═══ Blended opportunity (OLS + ML) → champion ═════════════
    df_filtered['BLENDED_OPPORTUNITY'] = (
        df_filtered['OPPORTUNITY_SCORE'] + df_filtered['ML_OPPORTUNITY']) / 2

    model_cols = ['COMMUNE_NAME', 'PREDICTED_STORES', 'OPPORTUNITY_SCORE',
                  'OLS_PRED_LO', 'OLS_PRED_HI', 'ML_PRED', 'ML_OPPORTUNITY',
                  'BLENDED_OPPORTUNITY']
    top5_ols = (
        top5
        .merge(df_filtered[model_cols], on='COMMUNE_NAME', how='left')
        .sort_values('BLENDED_OPPORTUNITY', ascending=False)
        .reset_index(drop=True)
    )
    top5_ols['RANK'] = range(1, len(top5_ols) + 1)
    champion = top5_ols.iloc[0].copy()

    diagnostics = {
        'ols_r2': ols_r2, 'ols_rmse': ols_rmse,
        'cv_r2': cv_r2, 'cv_rmse': cv_rmse, 'cv_mae': cv_mae,
        'importances': importances, 'k_folds': k,
        'n_train': len(df_filtered), 'n_outliers': len(df) - len(df_filtered),
        'features': feat, 'ols_summary': str(final_model.summary()),
    }
    return top20, top5, top5_ols, champion, df_filtered, final_model, diagnostics


# ─── GEOSPATIAL HELPERS (lat/lon, no routing dependency) ───────
def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km."""
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner=False)
def catchment_analysis(_stores_df, radius_m=800):
    """For every store: neighbours, own-brand overlap and a saturation index."""
    s = _stores_df.copy().reset_index(drop=True)
    lat = s['latitude'].to_numpy()
    lon = s['longitude'].to_numpy()
    brand = s['brand_category'].fillna('Other').to_numpy()
    r_km = radius_m / 1000.0
    n = len(s)
    neigh_all, neigh_same, nearest = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        d = _haversine_km(lat[i], lon[i], lat, lon)
        within = (d <= r_km) & (np.arange(n) != i)
        neigh_all[i] = within.sum()
        neigh_same[i] = (within & (brand == brand[i])).sum()
        other = d[np.arange(n) != i]
        nearest[i] = other.min() if len(other) else np.nan
    s['NEIGHBORS'] = neigh_all
    s['SAME_BRAND_NEAR'] = neigh_same
    s['NEAREST_KM'] = nearest
    # Saturation: more neighbours + closer competitor ⇒ higher cannibalisation risk
    s['SATURATION'] = neigh_all + neigh_same * 0.5
    return s


@st.cache_data(show_spinner=False)
def build_site_grid(_boundaries, _stores_df, _df_models, n_side=70, target_communes=None):
    """Grid of candidate sites scored by demand (commune) vs supply gap (distance)."""
    from shapely.geometry import Point
    minx, miny, maxx, maxy = _boundaries.total_bounds
    xs = np.linspace(minx, maxx, n_side)
    ys = np.linspace(miny, maxy, n_side)
    gx, gy = np.meshgrid(xs, ys)
    pts = gpd.GeoDataFrame(
        {'lon': gx.ravel(), 'lat': gy.ravel()},
        geometry=[Point(x, y) for x, y in zip(gx.ravel(), gy.ravel())],
        crs='EPSG:4326',
    )
    # Keep only points inside the canton, tagged with their commune
    inside = gpd.sjoin(pts, _boundaries[['COMMUNE_NAME', 'geometry']],
                       how='inner', predicate='within').drop(columns='index_right')
    if target_communes:
        inside = inside[inside['COMMUNE_NAME'].isin(target_communes)]
    inside = inside.reset_index(drop=True)
    if inside.empty:
        return inside

    # Distance to nearest existing store (supply gap)
    slat = _stores_df['latitude'].to_numpy()
    slon = _stores_df['longitude'].to_numpy()
    dist = np.array([_haversine_km(la, lo, slat, slon).min()
                     for la, lo in zip(inside['lat'], inside['lon'])])
    inside['DIST_NEAREST_KM'] = dist

    # Demand from the host commune
    demand = _df_models[['COMMUNE_NAME', 'POP_DENSITY', 'proxy_purchasing_power_median_chf',
                         'BLENDED_OPPORTUNITY']].copy()
    inside = inside.merge(demand, on='COMMUNE_NAME', how='left')

    def mm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng and rng > 0 else s * 0.0
    inside['SUPPLY_GAP']   = mm(inside['DIST_NEAREST_KM'])
    inside['DEMAND_DENS']  = mm(inside['POP_DENSITY'])
    inside['DEMAND_INCOME'] = mm(inside['proxy_purchasing_power_median_chf'])
    inside['DEMAND_OPP']   = mm(inside['BLENDED_OPPORTUNITY'].clip(lower=0))
    inside['SITE_SCORE'] = (0.40 * inside['SUPPLY_GAP'] +
                            0.25 * inside['DEMAND_DENS'] +
                            0.15 * inside['DEMAND_INCOME'] +
                            0.20 * inside['DEMAND_OPP'])
    return inside.sort_values('SITE_SCORE', ascending=False).reset_index(drop=True)


# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 8px 8px;">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;">
        <div style="background:linear-gradient(135deg,{MIGROS_ORANGE},#E55A00);color:white;font-family:'Sora',sans-serif;
             font-weight:800;font-size:16px;padding:9px 14px;border-radius:11px;letter-spacing:0.5px;
             box-shadow:0 4px 16px rgba(255,107,0,0.30);">M</div>
        <div>
          <div style="font-family:'Sora',sans-serif;font-size:13px;font-weight:700;color:{TEXT_LIGHT};line-height:1.35;">Location<br>Intelligence</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:{TEXT_MUTED};margin-top:4px;letter-spacing:1.5px;">GENEVA · CH</div>
        </div>
      </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,{BORDER},transparent);margin:0 4px 18px;"></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        ["🏠  Overview", "📊  Stage 1 — Population", "🧮  Stage 2 — Scoring",
         "📈  Stage 3 — OLS vs ML", "🤖  Model Lab", "📡  Catchment & Overlap",
         "🎯  Site Finder", "📋  Demographic Dashboard", "🗺️  Interactive Map"],
        label_visibility="visible"
    )

    st.markdown(f"""
    <div style="height:1px;background:linear-gradient(90deg,transparent,{BORDER},transparent);margin:22px 4px 16px;"></div>
    <div style="background:rgba(255,255,255,0.02);border:1px solid {BORDER};border-radius:12px;
         padding:14px 16px;margin:0 4px;font-family:'JetBrains Mono',monospace;font-size:9.5px;
         color:{TEXT_MUTED};letter-spacing:1px;">
      <div style="margin-bottom:11px;color:{ORANGE_LIGHT};letter-spacing:2.5px;font-weight:700;">PIPELINE</div>
      <div style="display:flex;justify-content:space-between;margin-bottom:7px;">
        <span>All communes</span><span style="color:{TEXT_LIGHT};">→ Top 20</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:7px;">
        <span>Top 20</span><span style="color:{TEXT_LIGHT};">→ Top 5</span>
      </div>
      <div style="display:flex;justify-content:space-between;">
        <span>Top 5</span><span style="color:{ORANGE_LIGHT};">→ Champion ★</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── LOAD DATA ─────────────────────────────────────────────────
with st.spinner("Loading data & fitting models…"):
    try:
        stores_gdf, df_pop, df_power_2022 = load_data()
        boundaries = load_boundaries()
        df_clean, joined_stores = build_master(stores_gdf, df_pop, df_power_2022, boundaries)
        top20, top5, top5_ols, champion, df_filtered, final_model, diag = run_pipeline(df_clean)
        data_ok = True
    except Exception as e:
        data_ok = False
        err_msg = str(e)

if not data_ok:
    st.error(f"Failed to load data: {err_msg}")
    st.stop()

# ─── HEADER ────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-bar">
  <div class="logo">M</div>
  <div>
    <div class="title">Migros Location Intelligence</div>
    <div class="subtitle"><span class="live-dot"></span>3-STAGE FUNNEL · CANTON OF GENEVA · OLS REGRESSION</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "🏠  Overview":

    champ_name = champion['COMMUNE_NAME']

    # KPI row
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Total Communes</div>
        <div class="kpi-value">{len(df_clean)}</div>
        <div class="kpi-sub">GENEVA AREA</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Stores Mapped</div>
        <div class="kpi-value">{int(df_clean['STORE_COUNT'].sum())}</div>
        <div class="kpi-sub">ACTIVE LOCATIONS</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Stage 1</div>
        <div class="kpi-value">20</div>
        <div class="kpi-sub">BY POPULATION</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Stage 2</div>
        <div class="kpi-value">5</div>
        <div class="kpi-sub">COMPOSITE SCORE</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Opportunity Gap</div>
        <div class="kpi-value kpi-accent">+{champion['OPPORTUNITY_SCORE']:.2f}</div>
        <div class="kpi-sub">CHAMPION TARGET</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Champion card
    stats_html = ""
    for lbl, val, col in [
        ('POPULATION',      f"{int(champion['POPULATION']):,}",                              TEXT_LIGHT),
        ('STORES NOW',      f"{int(champion['STORE_COUNT'])}",                               TEXT_LIGHT),
        ('PREDICTED',       f"{champion['PREDICTED_STORES']:.2f}",                           TEXT_LIGHT),
        ('OPPORTUNITY GAP', f"+{champion['OPPORTUNITY_SCORE']:.2f}",                         MIGROS_ORANGE),
        ('INCOME CHF',      f"{int(champion['proxy_purchasing_power_median_chf']):,}",        TEXT_LIGHT),
        ('FOREIGN %',       f"{champion['PCT_FOREIGNERS']:.1f}%",                            MIGROS_TEAL),
    ]:
        stats_html += f"""
        <div class="champion-stat">
          <div class="champion-stat-label">{lbl}</div>
          <div class="champion-stat-value" style="color:{col};">{val}</div>
        </div>"""

    st.markdown(f"""
    <div class="champion-card">
      <div class="champion-label">🏆 MIGROS OPTIMAL EXPANSION TARGET</div>
      <div class="champion-name">{champ_name}</div>
      <div class="champion-loc">Canton of Geneva · Switzerland</div>
      <div class="champion-stats">{stats_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline funnel
    st.markdown(f"""
    <div class="stage-badge">◈ PIPELINE</div>
    <div class="section-title">3-Stage Funnel Summary</div>
    <div class="section-sub">Each stage narrows the search for the optimal Migros expansion location</div>
    """, unsafe_allow_html=True)

    funnel_data = [
        (str(len(df_clean)), "All Communes",      "Starting universe · City centre excluded"),
        ("20",               "Stage 1 → Top 20",  "Filtered by Resident Population"),
        ("5",                "Stage 2 → Top 5",   "Socio-Economic Composite Score (Income 35% · Foreign 25% · Age 20% · Urban 20%)"),
        ("1 ★",              "Stage 3 → Champion", f"OLS Opportunity Gap · Winner: {champ_name}"),
    ]
    for (num, title, sub), width in zip(funnel_data, ["100%", "86%", "72%", "58%"]):
        st.markdown(f"""
        <div class="funnel-step" style="width:{width};">
          <div class="funnel-num">{num}</div>
          <div>
            <div class="funnel-title">{title}</div>
            <div class="funnel-sub">{sub}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: STAGE 1
# ═══════════════════════════════════════════════════════════════
elif page == "📊  Stage 1 — Population":

    st.markdown(f"""
    <div class="stage-badge">◈ STAGE 1</div>
    <div class="section-title">Top 20 Communes by Population</div>
    <div class="section-sub">Starting pool: all communes · City centre excluded · Ranked by resident population</div>
    """, unsafe_allow_html=True)

    # ── Filters ────────────────────────────────────────────────
    st.markdown(f'<div class="filter-panel"><div class="filter-title">⚙ Filters</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        pop_min = int(top20['POPULATION'].min())
        pop_max = int(top20['POPULATION'].max())
        pop_range = st.slider("Population Range", pop_min, pop_max, (pop_min, pop_max), step=500)
    with fc2:
        income_opts = ["All", "< CHF 70,000", "CHF 70,000–80,000", "> CHF 80,000"]
        income_filter = st.selectbox("Income Bracket", income_opts)
    with fc3:
        store_opts = ["All", "0 stores", "1–3 stores", "4+ stores"]
        store_filter = st.selectbox("Current Store Count", store_opts)
    st.markdown('</div>', unsafe_allow_html=True)

    # Apply filters
    top20_f = top20[
        (top20['POPULATION'] >= pop_range[0]) &
        (top20['POPULATION'] <= pop_range[1])
    ].copy()
    if income_filter == "< CHF 70,000":
        top20_f = top20_f[top20_f['proxy_purchasing_power_median_chf'] < 70000]
    elif income_filter == "CHF 70,000–80,000":
        top20_f = top20_f[(top20_f['proxy_purchasing_power_median_chf'] >= 70000) & (top20_f['proxy_purchasing_power_median_chf'] <= 80000)]
    elif income_filter == "> CHF 80,000":
        top20_f = top20_f[top20_f['proxy_purchasing_power_median_chf'] > 80000]
    if store_filter == "0 stores":
        top20_f = top20_f[top20_f['STORE_COUNT'] == 0]
    elif store_filter == "1–3 stores":
        top20_f = top20_f[(top20_f['STORE_COUNT'] >= 1) & (top20_f['STORE_COUNT'] <= 3)]
    elif store_filter == "4+ stores":
        top20_f = top20_f[top20_f['STORE_COUNT'] >= 4]

    st.markdown(f"<span class='chip'>Showing {len(top20_f)} of {len(top20)} communes</span>", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if top20_f.empty:
        st.warning("No communes match the current filters.")
    else:
        d = top20_f.sort_values('POPULATION', ascending=True)
        colors = [MIGROS_ORANGE if n == d['POPULATION'].idxmax() else 'rgba(79,179,232,0.75)'
                  for n in d.index]
        fig = go.Figure(go.Bar(
            x=d['POPULATION'], y=d['COMMUNE_NAME'], orientation='h',
            marker=dict(color=colors, cornerradius=6),
            text=[f"{v:,}" for v in d['POPULATION']],
            textposition='outside', textfont=dict(size=11, color=TEXT_MUTED),
            customdata=np.stack([d['STORE_COUNT'], d['PCT_FOREIGNERS'],
                                 d['proxy_purchasing_power_median_chf']], axis=-1),
            hovertemplate='<b>%{y}</b><br>Population %{x:,}<br>'
                          'Stores %{customdata[0]:.0f} · Foreign %{customdata[1]:.1f}%%<br>'
                          'Income CHF %{customdata[2]:,.0f}<extra></extra>',
        ))
        fig.update_layout(
            title='Top Communes by Population',
            xaxis=dict(title='Resident population', tickformat='~s'),
            yaxis=dict(title=None), bargap=0.32,
            height=max(380, len(d) * 34 + 110),
        )
        fig.update_xaxes(range=[0, d['POPULATION'].max() * 1.14])
        show_chart(fig)

        st.markdown(f"<div style='height:20px'></div>", unsafe_allow_html=True)
        display_cols = top20_f[['RANK', 'COMMUNE_NAME', 'POPULATION', 'STORE_COUNT',
                               'PCT_FOREIGNERS', 'proxy_purchasing_power_median_chf']].rename(columns={
            'COMMUNE_NAME': 'Commune', 'POPULATION': 'Population', 'STORE_COUNT': 'Stores',
            'PCT_FOREIGNERS': 'Foreign %', 'proxy_purchasing_power_median_chf': 'Income (CHF)'
        })
        rows_html = ""
        for i, (_, row) in enumerate(display_cols.iterrows()):
            badge = {0: '🥇', 1: '🥈', 2: '🥉'}.get(i, '')
            rows_html += f"""<tr>
              <td>{int(row['RANK'])}</td>
              <td>{badge} {row['Commune']}</td>
              <td>{int(row['Population']):,}</td>
              <td>{int(row['Stores'])}</td>
              <td>{row['Foreign %']:.1f}%</td>
              <td>CHF {int(row['Income (CHF)']):,}</td>
            </tr>"""
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:22px;overflow-x:auto;">
          <div style="margin-bottom:14px;display:flex;align-items:center;gap:10px;">
            <span class="stage-badge">STAGE 1 RESULTS</span>
            <span style="font-family:'Inter',sans-serif;font-size:14px;font-weight:500;color:{TEXT_MUTED};">{len(top20_f)} communes shown</span>
          </div>
          <table class="intel-table">
            <thead><tr>
              <th>RANK</th><th>COMMUNE</th><th>POPULATION</th><th>STORES</th><th>FOREIGN %</th><th>INCOME</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: STAGE 2
# ═══════════════════════════════════════════════════════════════
elif page == "🧮  Stage 2 — Scoring":

    st.markdown(f"""
    <div class="stage-badge">◈ STAGE 2</div>
    <div class="section-title">Socio-Economic Composite Scoring</div>
    <div class="section-sub">Income 35% · Foreign Residents 25% · Working-Age 20% · Urban Density 20%</div>
    """, unsafe_allow_html=True)

    # ── Interactive weight sliders ──────────────────────────────
    st.markdown(f'<div class="filter-panel"><div class="filter-title">⚙ Adjust Score Weights (must sum to 100%)</div>', unsafe_allow_html=True)
    wc1, wc2, wc3, wc4 = st.columns(4)
    with wc1:
        w_income  = st.slider("Income Weight %",   0, 60, 35, step=5)
    with wc2:
        w_foreign = st.slider("Foreign % Weight",  0, 60, 25, step=5)
    with wc3:
        w_age     = st.slider("Working Age %",     0, 60, 20, step=5)
    with wc4:
        w_urban   = st.slider("Urban Density %",   0, 60, 20, step=5)

    total_w = w_income + w_foreign + w_age + w_urban
    st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:10px;margin-top:8px;">
      Weight total: <span style="color:{'#22c55e' if total_w==100 else MIGROS_ORANGE};font-weight:700;">{total_w}%</span>
      {"✓ Balanced" if total_w==100 else "  ← Adjust to reach 100%"}
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Recalculate scores with custom weights
    def minmax(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0.0

    s2_data = top5.copy()
    s2_data['SCORE_INCOME']  = minmax(s2_data['proxy_purchasing_power_median_chf'])
    s2_data['SCORE_FOREIGN'] = minmax(s2_data['PCT_FOREIGNERS'])
    s2_data['SCORE_AGE']     = minmax(s2_data['PCT_WORKING_AGE'])
    s2_data['SCORE_URBAN']   = 1 - minmax(s2_data['PCT_SINGLE_FAMILY'])

    if total_w > 0:
        CUSTOM_WEIGHTS = {
            'SCORE_INCOME':  w_income  / 100,
            'SCORE_FOREIGN': w_foreign / 100,
            'SCORE_AGE':     w_age     / 100,
            'SCORE_URBAN':   w_urban   / 100,
        }
        s2_data['COMPOSITE_SCORE'] = sum(s2_data[k] * v for k, v in CUSTOM_WEIGHTS.items())
        s2_data = s2_data.sort_values('COMPOSITE_SCORE', ascending=False).reset_index(drop=True)

    # Weight cards
    weights = [('Income', w_income, MIGROS_ORANGE), ('Foreign Residents', w_foreign, MIGROS_TEAL),
               ('Working Age', w_age, '#A78BFA'), ('Urban Density', w_urban, '#34D399')]
    cols = st.columns(4)
    for col, (label, pct, color) in zip(cols, weights):
        with col:
            st.markdown(f"""
            <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:12px;
                 padding:18px 16px;text-align:center;border-top:2px solid {color};">
              <div style="font-family:'JetBrains Mono',monospace;font-size:9.5px;color:{TEXT_MUTED};
                   margin-bottom:8px;letter-spacing:1.6px;text-transform:uppercase;">{label}</div>
              <div style="font-family:'Sora',sans-serif;font-size:27px;font-weight:700;color:{color};">{pct}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    dims        = ['SCORE_INCOME', 'SCORE_FOREIGN', 'SCORE_AGE', 'SCORE_URBAN']
    dim_labels  = [f'Income ({w_income}%)', f'Foreign % ({w_foreign}%)', f'Working Age ({w_age}%)', f'Urban Density ({w_urban}%)']
    dim_colors  = [MIGROS_ORANGE, MIGROS_TEAL, '#A78BFA', '#34D399']
    dim_weights_list = [w_income/100, w_foreign/100, w_age/100, w_urban/100]
    c1, c2 = st.columns(2)

    with c1:
        d = s2_data.sort_values('COMPOSITE_SCORE', ascending=True)
        colors = [MIGROS_ORANGE if v == d['COMPOSITE_SCORE'].max()
                  else 'rgba(79,179,232,0.7)' for v in d['COMPOSITE_SCORE']]
        fig = go.Figure(go.Bar(
            x=d['COMPOSITE_SCORE'], y=d['COMMUNE_NAME'], orientation='h',
            marker=dict(color=colors, cornerradius=6),
            text=[f"{v:.3f}" for v in d['COMPOSITE_SCORE']],
            textposition='outside', textfont=dict(size=11.5, color=TEXT_LIGHT),
            hovertemplate='<b>%{y}</b><br>Composite score %{x:.4f}<extra></extra>',
        ))
        fig.update_layout(title='Composite Score', bargap=0.35,
                          xaxis=dict(title='Score (0–1)',
                                     range=[0, max(0.01, d['COMPOSITE_SCORE'].max()) * 1.22]),
                          yaxis=dict(title=None))
        show_chart(fig, height=400)

    with c2:
        fig = go.Figure()
        for dim, lbl, col, w in zip(dims, dim_labels, dim_colors, dim_weights_list):
            fig.add_bar(name=lbl, x=s2_data['COMMUNE_NAME'], y=s2_data[dim] * w,
                        marker=dict(color=col, cornerradius=4),
                        hovertemplate='<b>%{x}</b><br>' + lbl + ': %{y:.3f}<extra></extra>')
        fig.update_layout(title='Breakdown by Dimension', barmode='stack', bargap=0.42,
                          yaxis=dict(title='Weighted score'), xaxis=dict(title=None),
                          legend=dict(orientation='h', y=1.06, x=0))
        show_chart(fig, height=400)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    rows_html = ""
    for i, (_, row) in enumerate(s2_data.iterrows()):
        badge = {0: '🥇', 1: '🥈', 2: '🥉'}.get(i, '')
        rows_html += f"""<tr>
          <td>{i+1}</td>
          <td>{badge} {row['COMMUNE_NAME']}</td>
          <td>CHF {int(row['proxy_purchasing_power_median_chf']):,}</td>
          <td>{row['PCT_FOREIGNERS']:.1f}%</td>
          <td>{row['PCT_WORKING_AGE']:.1f}%</td>
          <td>{row['PCT_SINGLE_FAMILY']:.1f}%</td>
          <td><b style="color:{MIGROS_ORANGE};">{row['COMPOSITE_SCORE']:.4f}</b></td>
        </tr>"""
    st.markdown(f"""
    <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:22px;overflow-x:auto;">
      <div style="margin-bottom:14px;">
        <span class="stage-badge">STAGE 2 RESULTS</span>
      </div>
      <table class="intel-table">
        <thead><tr>
          <th>RANK</th><th>COMMUNE</th><th>INCOME</th><th>FOREIGN %</th>
          <th>WORKING AGE %</th><th>SINGLE-FAMILY %</th><th>COMPOSITE SCORE</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: STAGE 3
# ═══════════════════════════════════════════════════════════════
elif page == "📈  Stage 3 — OLS vs ML":

    st.markdown(f"""
    <div class="stage-badge">◈ STAGE 3</div>
    <div class="section-title">Opportunity Gap — OLS vs Machine Learning</div>
    <div class="section-sub">Two models estimate the "expected" store count; the gap vs reality flags
    under-served markets. The champion is ranked on the <b>blended</b> gap of both models.</div>
    """, unsafe_allow_html=True)

    # ── Model-agreement comparison (OLS · ML · Blended) ─────────
    comp = top5_ols.sort_values('BLENDED_OPPORTUNITY', ascending=True).reset_index(drop=True)
    fig = go.Figure()
    for col, label, color in [('OPPORTUNITY_SCORE', 'OLS gap', MIGROS_TEAL),
                              ('ML_OPPORTUNITY', 'ML gap', ACCENT_PURPLE),
                              ('BLENDED_OPPORTUNITY', 'Blended', MIGROS_ORANGE)]:
        fig.add_bar(name=label, y=comp['COMMUNE_NAME'], x=comp[col], orientation='h',
                    marker=dict(color=color, cornerradius=5),
                    hovertemplate='<b>%{y}</b><br>' + label + ': %{x:+.2f}<extra></extra>')
    fig.add_vline(x=0, line=dict(color='rgba(148,163,184,0.35)', dash='dash', width=1))
    fig.update_layout(title='Model Agreement on Under-supply', barmode='group',
                      bargap=0.30, bargroupgap=0.08,
                      xaxis=dict(title='Opportunity gap (predicted − actual stores)'),
                      yaxis=dict(title=None),
                      legend=dict(orientation='h', y=1.08, x=0))
    show_chart(fig, height=420)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Filters ────────────────────────────────────────────────
    st.markdown(f'<div class="filter-panel"><div class="filter-title">⚙ Filters</div>', unsafe_allow_html=True)
    fc1, fc2 = st.columns(2)
    with fc1:
        show_negative = st.checkbox("Show negative opportunity gaps", value=True)
    with fc2:
        sort_by = st.selectbox("Sort By", ["Opportunity Gap ↓", "Population ↓", "Predicted Stores ↓"])
    st.markdown('</div>', unsafe_allow_html=True)

    top5_view = top5_ols.copy()
    if not show_negative:
        top5_view = top5_view[top5_view['OPPORTUNITY_SCORE'] >= 0]
    if sort_by == "Population ↓":
        top5_view = top5_view.sort_values('POPULATION', ascending=False)
    elif sort_by == "Predicted Stores ↓":
        top5_view = top5_view.sort_values('PREDICTED_STORES', ascending=False)

    c1, c2 = st.columns(2)

    with c1:
        d = top5_view.sort_values('OPPORTUNITY_SCORE', ascending=True)
        colors = [MIGROS_ORANGE if v == d['OPPORTUNITY_SCORE'].max()
                  else 'rgba(79,179,232,0.7)' for v in d['OPPORTUNITY_SCORE']]
        fig = go.Figure(go.Bar(
            x=d['OPPORTUNITY_SCORE'], y=d['COMMUNE_NAME'], orientation='h',
            marker=dict(color=colors, cornerradius=6),
            text=[f"{v:+.2f}" for v in d['OPPORTUNITY_SCORE']],
            textposition='outside', textfont=dict(size=11.5, color=TEXT_LIGHT),
            hovertemplate='<b>%{y}</b><br>OLS gap %{x:+.2f}<extra></extra>',
        ))
        fig.add_vline(x=0, line=dict(color='rgba(148,163,184,0.35)', dash='dash', width=1))
        fig.update_layout(title='OLS Opportunity Gap', bargap=0.35,
                          xaxis=dict(title='Predicted − actual stores'),
                          yaxis=dict(title=None))
        show_chart(fig, height=400)

    with c2:
        if not top5_view.empty:
            vals = list(top5_view['STORE_COUNT']) + list(top5_view['PREDICTED_STORES'].dropna())
            lim_lo, lim_hi = max(0, min(vals) - 0.4), max(vals) + 0.9
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[lim_lo, lim_hi], y=[lim_lo, lim_hi], mode='lines',
                line=dict(color='rgba(148,163,184,0.4)', dash='dash', width=1.4),
                name='Equilibrium', hoverinfo='skip'))
            best = top5_view['BLENDED_OPPORTUNITY'].idxmax()
            err = (top5_view['OLS_PRED_HI'] - top5_view['PREDICTED_STORES']).abs()
            fig.add_trace(go.Scatter(
                x=top5_view['STORE_COUNT'], y=top5_view['PREDICTED_STORES'],
                mode='markers+text', text=top5_view['COMMUNE_NAME'],
                textposition='middle right', textfont=dict(size=11, color=TEXT_MUTED),
                error_y=dict(type='data', array=err, visible=True,
                             color='rgba(148,163,184,0.35)', thickness=1, width=4),
                marker=dict(
                    size=16, line=dict(color='white', width=1.2),
                    color=[MIGROS_ORANGE if i == best else MIGROS_TEAL
                           for i in top5_view.index],
                    symbol=['star' if i == best else 'circle' for i in top5_view.index]),
                name='Commune',
                hovertemplate='<b>%{text}</b><br>Actual %{x:.0f} · Predicted %{y:.2f}'
                              '<extra></extra>'))
            fig.add_annotation(x=lim_lo + (lim_hi-lim_lo)*0.04, y=lim_hi - (lim_hi-lim_lo)*0.06,
                               text='<i>Above line = under-served</i>', showarrow=False,
                               font=dict(size=11, color=MIGROS_ORANGE), xanchor='left')
            fig.update_layout(title='Actual vs Predicted (with 95% CI)',
                              xaxis=dict(title='Actual stores', range=[lim_lo, lim_hi]),
                              yaxis=dict(title='OLS-predicted stores', range=[lim_lo, lim_hi]),
                              showlegend=False)
            show_chart(fig, height=400)

    with st.expander("📋 View OLS Regression Summary"):
        st.code(diag['ols_summary'], language='text')

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    def _gap(v):
        return f"+{v:.2f}" if v >= 0 else f"{v:.2f}"
    rows_html = ""
    for i, (_, row) in enumerate(top5_view.iterrows()):
        badge = {0: '🥇 ★', 1: '🥈', 2: '🥉'}.get(i, '')
        bl_color = MIGROS_ORANGE if row['BLENDED_OPPORTUNITY'] == top5_view['BLENDED_OPPORTUNITY'].max() else TEXT_LIGHT
        rows_html += f"""<tr>
          <td>{int(row['RANK'])}</td>
          <td>{badge} {row['COMMUNE_NAME']}</td>
          <td>{int(row['STORE_COUNT'])}</td>
          <td>{row['PREDICTED_STORES']:.2f}</td>
          <td>{row['ML_PRED']:.2f}</td>
          <td style="color:{MIGROS_TEAL};">{_gap(row['OPPORTUNITY_SCORE'])}</td>
          <td style="color:#A78BFA;">{_gap(row['ML_OPPORTUNITY'])}</td>
          <td><b style="color:{bl_color};">{_gap(row['BLENDED_OPPORTUNITY'])}</b></td>
        </tr>"""
    st.markdown(f"""
    <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:22px;overflow-x:auto;">
      <div style="margin-bottom:14px;">
        <span class="stage-badge">STAGE 3 RESULTS</span>
      </div>
      <table class="intel-table">
        <thead><tr>
          <th>RANK</th><th>COMMUNE</th><th>STORES NOW</th>
          <th>OLS PRED</th><th>ML PRED</th><th>OLS GAP</th><th>ML GAP</th><th>BLENDED GAP</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Champion card
    champ_name = champion['COMMUNE_NAME']
    stats_html = ""
    for lbl, val, col in [
        ('STORES NOW',     f"{int(champion['STORE_COUNT'])}",                          TEXT_LIGHT),
        ('OLS PRED',       f"{champion['PREDICTED_STORES']:.2f}",                      MIGROS_TEAL),
        ('ML PRED',        f"{champion['ML_PRED']:.2f}",                               '#A78BFA'),
        ('BLENDED GAP',    f"+{champion['BLENDED_OPPORTUNITY']:.2f}",                  MIGROS_ORANGE),
        ('INCOME CHF',     f"{int(champion['proxy_purchasing_power_median_chf']):,}",   TEXT_LIGHT),
        ('FOREIGN %',      f"{champion['PCT_FOREIGNERS']:.1f}%",                       MIGROS_TEAL),
    ]:
        stats_html += f"""
        <div class="champion-stat">
          <div class="champion-stat-label">{lbl}</div>
          <div class="champion-stat-value" style="color:{col};">{val}</div>
        </div>"""

    st.markdown(f"""
    <div class="champion-card">
      <div class="champion-label">🏆 MIGROS OPTIMAL EXPANSION TARGET</div>
      <div class="champion-name">{champ_name}</div>
      <div class="champion-loc">Canton of Geneva · Switzerland</div>
      <div class="champion-stats">{stats_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: MODEL LAB  (diagnostics + ML internals)
# ═══════════════════════════════════════════════════════════════
elif page == "🤖  Model Lab":

    st.markdown(f"""
    <div class="stage-badge">◈ MODEL LAB</div>
    <div class="section-title">Model Diagnostics & Drivers</div>
    <div class="section-sub">Honest out-of-sample performance (K-fold CV), what drives the
    Gradient-Boosting prediction, and residual checks for both models.</div>
    """, unsafe_allow_html=True)

    # ── Performance KPIs ───────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">OLS · R²</div>
        <div class="kpi-value">{diag['ols_r2']:.3f}</div><div class="kpi-sub">IN-SAMPLE FIT</div></div>
      <div class="kpi-card"><div class="kpi-label">OLS · RMSE</div>
        <div class="kpi-value">{diag['ols_rmse']:.2f}</div><div class="kpi-sub">STORES</div></div>
      <div class="kpi-card"><div class="kpi-label">ML · CV R²</div>
        <div class="kpi-value kpi-accent">{diag['cv_r2']:.3f}</div><div class="kpi-sub">{diag['k_folds']}-FOLD OUT-OF-SAMPLE</div></div>
      <div class="kpi-card"><div class="kpi-label">ML · CV RMSE</div>
        <div class="kpi-value">{diag['cv_rmse']:.2f}</div><div class="kpi-sub">STORES</div></div>
      <div class="kpi-card"><div class="kpi-label">ML · CV MAE</div>
        <div class="kpi-value">{diag['cv_mae']:.2f}</div><div class="kpi-sub">STORES</div></div>
      <div class="kpi-card"><div class="kpi-label">Training Set</div>
        <div class="kpi-value">{diag['n_train']}</div><div class="kpi-sub">{diag['n_outliers']} OUTLIERS REMOVED</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature importance ─────────────────────────────────────
    imp = diag['importances'].copy()
    imp['label'] = imp['feature'].map(ML_FEATURE_LABELS).fillna(imp['feature'])
    imp = imp.sort_values('importance', ascending=True)
    frac = imp['importance'] / max(imp['importance'].max(), 1e-9)
    fig = go.Figure(go.Bar(
        x=imp['importance'], y=imp['label'], orientation='h',
        marker=dict(cornerradius=6,
                    color=[f'rgba(255,107,0,{0.35 + 0.65*f:.2f})' for f in frac]),
        text=[f"{v:.2f}" for v in imp['importance']],
        textposition='outside', textfont=dict(size=11, color=TEXT_MUTED),
        hovertemplate='<b>%{y}</b><br>Importance %{x:.3f}<extra></extra>',
    ))
    fig.update_layout(title='What Drives the ML Store-Demand Model',
                      xaxis=dict(title='Relative importance',
                                 range=[0, imp['importance'].max() * 1.15]),
                      yaxis=dict(title=None), bargap=0.32,
                      height=max(360, len(imp) * 36 + 110))
    show_chart(fig)

    # ── Residual diagnostics (2×2 interactive grid) ─────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    names = df_filtered['COMMUNE_NAME']
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.09,
                        subplot_titles=('OLS — Fitted vs Residual', 'ML — Fitted vs Residual',
                                        'OLS Residuals — Normal Q-Q',
                                        'ML — Actual vs Out-of-Sample'))
    _pt = '<b>%{customdata}</b><br>x %{x:.2f} · y %{y:.2f}<extra></extra>'

    fig.add_trace(go.Scatter(x=df_filtered['PREDICTED_STORES'], y=df_filtered['OLS_RESID'],
                             mode='markers', customdata=names, hovertemplate=_pt,
                             marker=dict(color=MIGROS_TEAL, size=9, opacity=0.85,
                                         line=dict(color='white', width=0.6))), 1, 1)
    fig.add_hline(y=0, line=dict(color=MIGROS_ORANGE, dash='dash', width=1.2), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_filtered['ML_PRED'], y=df_filtered['ML_RESID'],
                             mode='markers', customdata=names, hovertemplate=_pt,
                             marker=dict(color=ACCENT_PURPLE, size=9, opacity=0.85,
                                         line=dict(color='white', width=0.6))), 1, 2)
    fig.add_hline(y=0, line=dict(color=MIGROS_ORANGE, dash='dash', width=1.2), row=1, col=2)

    (qq_x, qq_y), (qq_s, qq_i, _) = stats.probplot(df_filtered['OLS_RESID'].dropna(), dist='norm')
    fig.add_trace(go.Scatter(x=qq_x, y=qq_y, mode='markers',
                             marker=dict(color=MIGROS_TEAL, size=8, opacity=0.85,
                                         line=dict(color='white', width=0.5)),
                             hovertemplate='Theoretical %{x:.2f}<br>Sample %{y:.2f}<extra></extra>'), 2, 1)
    fig.add_trace(go.Scatter(x=qq_x, y=qq_s * qq_x + qq_i, mode='lines',
                             line=dict(color=MIGROS_ORANGE, width=1.6), hoverinfo='skip'), 2, 1)

    mx = max(df_filtered['STORE_COUNT'].max(), df_filtered['ML_CV_PRED'].max()) + 0.5
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode='lines', hoverinfo='skip',
                             line=dict(color='rgba(148,163,184,0.4)', dash='dash', width=1.3)), 2, 2)
    fig.add_trace(go.Scatter(x=df_filtered['STORE_COUNT'], y=df_filtered['ML_CV_PRED'],
                             mode='markers', customdata=names, hovertemplate=_pt,
                             marker=dict(color=MIGROS_ORANGE, size=9, opacity=0.9,
                                         line=dict(color='white', width=0.6))), 2, 2)

    fig.update_layout(showlegend=False, height=760, title='Residual Diagnostics')
    fig.update_annotations(font=dict(family='Sora, sans-serif', size=13.5, color=TEXT_LIGHT))
    for r, c, xt, yt in [(1, 1, 'OLS fitted', 'Residual'), (1, 2, 'ML fitted', 'Residual'),
                         (2, 1, 'Theoretical quantiles', 'Sample quantiles'),
                         (2, 2, 'Actual stores', 'CV prediction')]:
        fig.update_xaxes(title_text=xt, row=r, col=c)
        fig.update_yaxes(title_text=yt, row=r, col=c)
    show_chart(fig)

    with st.expander("📋 View full OLS regression summary"):
        st.code(diag['ols_summary'], language='text')


# ═══════════════════════════════════════════════════════════════
# PAGE: CATCHMENT & OVERLAP  (trade areas + cannibalisation)
# ═══════════════════════════════════════════════════════════════
elif page == "📡  Catchment & Overlap":

    st.markdown(f"""
    <div class="stage-badge">◈ GEOSPATIAL</div>
    <div class="section-title">Catchment & Cannibalisation</div>
    <div class="section-sub">Trade-area overlap between existing stores. High saturation = crowded
    micro-markets; isolated stores reveal genuine white space.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="filter-panel"><div class="filter-title">⚙ Catchment Options</div>', unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        radius_m = st.slider("Catchment radius (m)", 400, 2000, 800, step=100)
    with cc2:
        brands_all = sorted(joined_stores['brand_category'].fillna('Other').unique().tolist())
        brand_sel = st.multiselect("Brands", brands_all, default=brands_all)
    st.markdown('</div>', unsafe_allow_html=True)

    sdf = joined_stores.copy()
    sdf['brand_category'] = sdf['brand_category'].fillna('Other')
    sdf = sdf[sdf['brand_category'].isin(brand_sel)]
    cat = catchment_analysis(sdf, radius_m)

    overlap_pct = 100 * (cat['SAME_BRAND_NEAR'] > 0).mean() if len(cat) else 0
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">Stores</div>
        <div class="kpi-value">{len(cat)}</div><div class="kpi-sub">IN SELECTION</div></div>
      <div class="kpi-card"><div class="kpi-label">Avg Neighbours</div>
        <div class="kpi-value">{cat['NEIGHBORS'].mean():.1f}</div><div class="kpi-sub">WITHIN {radius_m} M</div></div>
      <div class="kpi-card"><div class="kpi-label">Same-Brand Overlap</div>
        <div class="kpi-value kpi-accent">{overlap_pct:.0f}%</div><div class="kpi-sub">SELF-CANNIBALISING</div></div>
      <div class="kpi-card"><div class="kpi-label">Median Nearest</div>
        <div class="kpi-value">{cat['NEAREST_KM'].median():.2f}</div><div class="kpi-sub">KM TO NEXT STORE</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Charts: nearest-distance distribution + saturation by brand
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Histogram(
            x=cat['NEAREST_KM'].dropna(), nbinsx=18,
            marker=dict(color='rgba(79,179,232,0.75)',
                        line=dict(color=BG_DARK, width=1)),
            hovertemplate='%{x} km<br>%{y} stores<extra></extra>'))
        med = cat['NEAREST_KM'].median()
        fig.add_vline(x=med, line=dict(color=MIGROS_ORANGE, dash='dash', width=1.5),
                      annotation_text=f' median {med:.2f} km',
                      annotation_font=dict(size=11, color=MIGROS_ORANGE))
        fig.update_layout(title='Isolation of Existing Stores', bargap=0.06,
                          xaxis=dict(title='Distance to nearest other store (km)'),
                          yaxis=dict(title='Stores'))
        show_chart(fig, height=380)
    with c2:
        by_brand = cat.groupby('brand_category')['SATURATION'].mean().sort_values(ascending=False)
        brand_cols = {'Migros': MIGROS_ORANGE, 'Coop': ACCENT_GOLD, 'Denner': ACCENT_RED,
                      'Aldi': MIGROS_TEAL, 'Lidl': ACCENT_GREEN}
        fig = go.Figure(go.Bar(
            x=by_brand.index, y=by_brand.values,
            marker=dict(cornerradius=6,
                        color=[brand_cols.get(b, '#9CA3AF') for b in by_brand.index]),
            text=[f"{v:.1f}" for v in by_brand.values], textposition='outside',
            textfont=dict(size=11.5, color=TEXT_MUTED),
            hovertemplate='<b>%{x}</b><br>Mean saturation %{y:.2f}<extra></extra>'))
        fig.update_layout(title='Average Saturation by Brand', bargap=0.45,
                          yaxis=dict(title='Mean saturation index',
                                     range=[0, by_brand.max() * 1.18]),
                          xaxis=dict(title=None))
        show_chart(fig, height=380)

    # Map: stores coloured by saturation + catchment rings for Migros
    with st.spinner("Rendering catchment map…"):
        m_lat = sdf['latitude'].mean()
        m_lon = sdf['longitude'].mean()
        cmap = folium.Map(location=[m_lat, m_lon], zoom_start=12, tiles='cartodbdark_matter')
        sat_max = max(cat['SATURATION'].max(), 1)
        for _, r in cat.iterrows():
            frac = r['SATURATION'] / sat_max
            color = f'#{int(255):02x}{int(180*(1-frac)):02x}{int(60*(1-frac)):02x}'
            if r['brand_category'] == 'Migros':
                folium.Circle(location=[r['latitude'], r['longitude']], radius=radius_m,
                              color=MIGROS_ORANGE, weight=1, fill=True,
                              fill_color=MIGROS_ORANGE, fill_opacity=0.05).add_to(cmap)
            folium.CircleMarker(
                location=[r['latitude'], r['longitude']],
                radius=4 + 7 * frac, color=color, fill=True, fill_color=color, fill_opacity=0.85, weight=1,
                tooltip=(f"{r['brand_category']} · {r['COMMUNE_NAME']}<br>"
                         f"Neighbours: {int(r['NEIGHBORS'])} · Nearest: {r['NEAREST_KM']:.2f} km"),
            ).add_to(cmap)
        st_folium(cmap, height=560, use_container_width=True, returned_objects=[])

    # Most cannibalised stores
    worst = cat.sort_values('SATURATION', ascending=False).head(10)
    rows_html = ""
    for _, r in worst.iterrows():
        rows_html += f"""<tr>
          <td>{r['brand_category']}</td><td>{r['COMMUNE_NAME']}</td>
          <td>{int(r['NEIGHBORS'])}</td><td>{int(r['SAME_BRAND_NEAR'])}</td>
          <td>{r['NEAREST_KM']:.2f} km</td>
          <td><b style="color:{MIGROS_ORANGE};">{r['SATURATION']:.1f}</b></td>
        </tr>"""
    st.markdown(f"""
    <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:22px;overflow-x:auto;">
      <div style="margin-bottom:14px;"><span class="stage-badge">MOST SATURATED STORES</span></div>
      <table class="intel-table">
        <thead><tr><th>BRAND</th><th>COMMUNE</th><th>NEIGHBOURS</th>
          <th>SAME-BRAND</th><th>NEAREST</th><th>SATURATION</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: SITE FINDER  (grid-level candidate scoring)
# ═══════════════════════════════════════════════════════════════
elif page == "🎯  Site Finder":

    st.markdown(f"""
    <div class="stage-badge">◈ GEOSPATIAL</div>
    <div class="section-title">Site Finder — Point-Level Recommendations</div>
    <div class="section-sub">Goes beyond communes: a grid of candidate locations across the canton,
    each scored on supply gap (distance to nearest store) and local demand (density · income · model gap).</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="filter-panel"><div class="filter-title">⚙ Search Options</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        focus = st.selectbox("Search area", ["Whole canton", "Top-5 opportunity communes"])
    with sc2:
        grid_res = st.select_slider("Grid resolution", options=[50, 70, 90, 110], value=70)
    with sc3:
        n_top = st.slider("Sites to highlight", 5, 20, 10)
    st.markdown('</div>', unsafe_allow_html=True)

    target = tuple(top5_ols['COMMUNE_NAME'].tolist()) if focus.startswith("Top-5") else None
    with st.spinner("Scoring candidate sites…"):
        grid = build_site_grid(boundaries, stores_gdf, df_filtered, grid_res, target)

    if grid.empty:
        st.warning("No candidate sites found for this selection.")
    else:
        best = grid.iloc[0]
        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi-card"><div class="kpi-label">Candidate Cells</div>
            <div class="kpi-value">{len(grid):,}</div><div class="kpi-sub">SCORED</div></div>
          <div class="kpi-card"><div class="kpi-label">Best Site Score</div>
            <div class="kpi-value kpi-accent">{best['SITE_SCORE']:.3f}</div><div class="kpi-sub">0–1 SCALE</div></div>
          <div class="kpi-card"><div class="kpi-label">Best Commune</div>
            <div class="kpi-value" style="font-size:20px;">{best['COMMUNE_NAME']}</div><div class="kpi-sub">HOST</div></div>
          <div class="kpi-card"><div class="kpi-label">Supply Gap</div>
            <div class="kpi-value">{best['DIST_NEAREST_KM']:.2f}</div><div class="kpi-sub">KM TO NEAREST STORE</div></div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Rendering site map…"):
            smap = folium.Map(location=[grid['lat'].mean(), grid['lon'].mean()],
                              zoom_start=12, tiles='cartodbdark_matter')
            folium.GeoJson(boundaries, style_function=lambda _: {
                'fillOpacity': 0, 'color': '#ffffff', 'weight': 0.4}).add_to(smap)
            # Score cloud (cap for performance)
            cloud = grid.head(600)
            smax = cloud['SITE_SCORE'].max() or 1
            for _, r in cloud.iterrows():
                frac = r['SITE_SCORE'] / smax
                col = f'#{int(255):02x}{int(200*(1-frac)):02x}{int(40):02x}'
                folium.CircleMarker(location=[r['lat'], r['lon']], radius=3,
                                    color=col, fill=True, fill_color=col,
                                    fill_opacity=0.45, weight=0).add_to(smap)
            # Existing stores (context)
            for _, r in stores_gdf.iterrows():
                folium.CircleMarker(location=[r['latitude'], r['longitude']], radius=3,
                                    color='#4FB3E8', fill=True, fill_color='#4FB3E8',
                                    fill_opacity=0.9, weight=0,
                                    tooltip=f"{r.get('brand_category','?')}").add_to(smap)
            # Top recommended sites
            for i, (_, r) in enumerate(grid.head(n_top).iterrows()):
                folium.Marker(
                    location=[r['lat'], r['lon']],
                    tooltip=f"#{i+1} · {r['COMMUNE_NAME']} · score {r['SITE_SCORE']:.3f}",
                    icon=folium.DivIcon(html=(
                        '<div style="background:#FF6B00;border:2px solid #fff;border-radius:50%;'
                        'width:26px;height:26px;display:flex;align-items:center;justify-content:center;'
                        f'color:#fff;font-size:11px;font-weight:700;">{i+1}</div>'),
                        icon_size=(26, 26), icon_anchor=(13, 13)),
                ).add_to(smap)
            st_folium(smap, height=580, use_container_width=True, returned_objects=[])

        rows_html = ""
        for i, (_, r) in enumerate(grid.head(n_top).iterrows()):
            rows_html += f"""<tr>
              <td>{i+1}</td><td>{r['COMMUNE_NAME']}</td>
              <td>{r['lat']:.4f}, {r['lon']:.4f}</td>
              <td>{r['DIST_NEAREST_KM']:.2f} km</td>
              <td>{int(r['POP_DENSITY']):,}</td>
              <td><b style="color:{MIGROS_ORANGE};">{r['SITE_SCORE']:.3f}</b></td>
            </tr>"""
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:22px;overflow-x:auto;">
          <div style="margin-bottom:14px;"><span class="stage-badge">TOP CANDIDATE SITES</span></div>
          <table class="intel-table">
            <thead><tr><th>#</th><th>COMMUNE</th><th>LAT, LON</th>
              <th>NEAREST STORE</th><th>DENSITY /KM²</th><th>SITE SCORE</th></tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: DEMOGRAPHIC DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif page == "📋  Demographic Dashboard":

    st.markdown(f"""
    <div class="stage-badge">◈ ANALYTICS</div>
    <div class="section-title">5-Factor Socio-Demographic Dashboard</div>
    <div class="section-sub">Regression analysis across all communes · Champion highlighted with ★</div>
    """, unsafe_allow_html=True)

    # ── Filters ────────────────────────────────────────────────
    st.markdown(f'<div class="filter-panel"><div class="filter-title">⚙ Filters</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        min_pop_demo = st.slider("Min Population", 0, 20000, 0, step=1000)
    with dc2:
        highlight_top = st.selectbox("Highlight Commune", ["Champion Only"] + list(df_filtered['COMMUNE_NAME'].sort_values()))
    st.markdown('</div>', unsafe_allow_html=True)

    champ_name = champion['COMMUNE_NAME']
    df_demo = df_filtered.copy()
    if min_pop_demo > 0:
        df_demo = df_demo[df_demo['POPULATION'] >= min_pop_demo]

    highlight_name = champ_name if highlight_top == "Champion Only" else highlight_top

    panels = [
        ('POPULATION',                        MIGROS_TEAL,   'Population',            'Resident population', '~s'),
        ('proxy_purchasing_power_median_chf', ACCENT_GOLD,   'Purchasing Power',      'Median income (CHF)', '~s'),
        ('PCT_FOREIGNERS',                    ACCENT_PURPLE, 'Foreign Residents',     'Foreign residents (%)', '.0f'),
        ('PCT_WORKING_AGE',                   ACCENT_GREEN,  'Working-Age Share',     'Working-age (%)', '.0f'),
        ('PCT_SINGLE_FAMILY',                 ACCENT_RED,    'Single-Family Housing', 'Single-family (%)', '.0f'),
        ('POP_DENSITY',                       '#60A5FA',     'Population Density',    'People per km²', '~s'),
    ]

    def factor_panel(xcol, color, title, xlab, tickfmt):
        d = df_demo.dropna(subset=[xcol])
        fig = go.Figure()
        # OLS trendline
        if len(d) > 2 and d[xcol].nunique() > 1:
            k, b0 = np.polyfit(d[xcol], d['STORE_COUNT'], 1)
            xs = np.linspace(d[xcol].min(), d[xcol].max(), 50)
            fig.add_trace(go.Scatter(x=xs, y=k * xs + b0, mode='lines', hoverinfo='skip',
                                     line=dict(color='rgba(255,107,0,0.85)', width=2)))
        fig.add_trace(go.Scatter(
            x=d[xcol], y=d['STORE_COUNT'], mode='markers', customdata=d['COMMUNE_NAME'],
            marker=dict(color=color, size=10, opacity=0.78,
                        line=dict(color='white', width=0.6)),
            hovertemplate='<b>%{customdata}</b><br>' + xlab + ' %{x:,.1f}<br>'
                          'Stores %{y:.0f}<extra></extra>'))
        h = d[d['COMMUNE_NAME'] == highlight_name]
        if not h.empty:
            fig.add_trace(go.Scatter(
                x=h[xcol], y=h['STORE_COUNT'], mode='markers', customdata=h['COMMUNE_NAME'],
                marker=dict(color=MIGROS_ORANGE, size=20, symbol='star',
                            line=dict(color='white', width=1.4)),
                hovertemplate='<b>★ %{customdata}</b><br>' + xlab + ' %{x:,.1f}<br>'
                              'Stores %{y:.0f}<extra></extra>'))
        fig.update_layout(title=title, showlegend=False,
                          xaxis=dict(title=xlab, tickformat=tickfmt),
                          yaxis=dict(title='Supermarkets'))
        return fig

    for i in range(0, len(panels), 2):
        cols = st.columns(2)
        for col, panel in zip(cols, panels[i:i+2]):
            with col:
                show_chart(factor_panel(*panel), height=360)

    # Highlight summary card
    h_data = df_demo[df_demo['COMMUNE_NAME'] == highlight_name]
    if not h_data.empty:
        h = h_data.iloc[0]
        stats_html = ""
        for lbl, val, col in [
            ('Population',  f"{int(h['POPULATION']):,}",                          TEXT_LIGHT),
            ('Stores',      f"{int(h['STORE_COUNT'])}",                           TEXT_LIGHT),
            ('OLS Pred',    f"{h['PREDICTED_STORES']:.2f}",                       MIGROS_TEAL),
            ('Gap',         f"{h['OPPORTUNITY_SCORE']:+.2f}",                     MIGROS_ORANGE),
            ('Income CHF',  f"{int(h['proxy_purchasing_power_median_chf']):,}",    TEXT_LIGHT),
            ('Foreign %',   f"{h['PCT_FOREIGNERS']:.1f}%",                        ACCENT_PURPLE),
        ]:
            stats_html += f"""
            <div class="champion-stat">
              <div class="champion-stat-label">{lbl}</div>
              <div class="champion-stat-value" style="color:{col};font-size:17px;">{val}</div>
            </div>"""
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid rgba(255,107,0,0.35);border-radius:16px;
             padding:24px;margin-top:6px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:{ORANGE_LIGHT};
               letter-spacing:2.5px;text-transform:uppercase;margin-bottom:14px;">★ Highlighted — {highlight_name}</div>
          <div class="champion-stats" style="justify-content:flex-start;">{stats_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: INTERACTIVE MAP
# ═══════════════════════════════════════════════════════════════
elif page == "🗺️  Interactive Map":

    st.markdown(f"""
    <div class="stage-badge">◈ MAP</div>
    <div class="section-title">Interactive Folium Map</div>
    <div class="section-sub">Choropleth: Opportunity Gap · Pins: Existing stores · ★ Champion location</div>
    """, unsafe_allow_html=True)

    # ── Map Filters ─────────────────────────────────────────────
    st.markdown(f'<div class="filter-panel"><div class="filter-title">⚙ Map Options</div>', unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        show_stores = st.checkbox("Show store pins", value=True)
    with mc2:
        all_brands = sorted(joined_stores['brand_category'].fillna('Other').unique().tolist())
        brand_filter = st.multiselect("Filter Brands", all_brands, default=all_brands)
    with mc3:
        map_zoom = st.slider("Initial Zoom", 10, 14, 12)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Building map…"):
        map_lat = boundaries.geometry.centroid.y.mean()
        map_lon = boundaries.geometry.centroid.x.mean()
        geomap = folium.Map(location=[map_lat, map_lon], zoom_start=map_zoom, tiles='cartodbdark_matter')

        folium.Choropleth(
            geo_data=boundaries,
            name='Opportunity Score',
            data=df_filtered,
            columns=['COMMUNE_NAME', 'OPPORTUNITY_SCORE'],
            key_on='feature.properties.COMMUNE_NAME',
            fill_color='YlOrRd',
            fill_opacity=0.72,
            line_opacity=0.35,
            line_color='#ffffff',
            legend_name='Market Opportunity Gap  (higher = more under-served)',
            nan_fill_color='#2d2d2d',
            nan_fill_opacity=0.4,
        ).add_to(geomap)

        tooltip_gdf = boundaries.merge(
            df_filtered[['COMMUNE_NAME', 'OPPORTUNITY_SCORE', 'POPULATION', 'STORE_COUNT']],
            on='COMMUNE_NAME', how='left'
        )
        folium.GeoJson(
            tooltip_gdf,
            style_function=lambda _: {'fillOpacity': 0, 'color': '#ffffff', 'weight': 0.4},
            tooltip=folium.GeoJsonTooltip(
                fields=['COMMUNE_NAME', 'POPULATION', 'STORE_COUNT', 'OPPORTUNITY_SCORE'],
                aliases=['Commune', 'Population', 'Current Stores', 'Opportunity Gap'],
                localize=True,
                style='background:#1a1a2e;color:#e6edf3;font-family:monospace;font-size:12px;',
            ),
            name='Commune Details',
        ).add_to(geomap)

        champ_name = champion['COMMUNE_NAME']
        champ_geo = df_filtered[df_filtered['COMMUNE_NAME'] == champ_name]
        if not champ_geo.empty:
            cx = champ_geo.geometry.centroid.iloc[0].x
            cy = champ_geo.geometry.centroid.iloc[0].y
            icon_html = (
                '<div style="background:#FF6B00;border:3px solid #fff;border-radius:50%;'
                'width:40px;height:40px;display:flex;align-items:center;justify-content:center;'
                'font-size:22px;box-shadow:0 0 20px #FF6B00,0 0 40px rgba(255,107,0,0.5);">&#9733;</div>'
            )
            popup_content = (
                f"<div style='font-family:monospace;background:#1a0800;color:#e6edf3;"
                f"padding:16px;border-radius:10px;border:2px solid #FF6B00;min-width:210px;'>"
                f"<b style='color:#FF6B00;font-size:13px;'>CHAMPION LOCATION</b><br><br>"
                f"<b style='font-size:16px;color:#fff;'>{champ_name}</b><br>"
                f"<span style='color:#aaa;font-size:11px;'>Canton of Geneva</span><br><br>"
                f"<span style='color:#8b949e;'>Population </span> {int(champion['POPULATION']):,}<br>"
                f"<span style='color:#8b949e;'>Stores now </span> {int(champion['STORE_COUNT'])}<br>"
                f"<span style='color:#8b949e;'>Predicted &nbsp;</span> {champion['PREDICTED_STORES']:.2f}<br>"
                f"<b style='color:#FF6B00;'>Opportunity Gap: +{champion['OPPORTUNITY_SCORE']:.2f}</b><br><br>"
                f"<span style='color:#8b949e;'>Income CHF </span> {int(champion['proxy_purchasing_power_median_chf']):,}<br>"
                f"<span style='color:#8b949e;'>Foreign %&nbsp; </span> {champion['PCT_FOREIGNERS']:.1f}%"
                f"</div>"
            )
            folium.Marker(
                location=[cy, cx],
                popup=folium.Popup(popup_content, max_width=270),
                tooltip=f'CHAMPION: {champ_name}  —  Click for details',
                icon=folium.DivIcon(html=icon_html, icon_size=(44, 44), icon_anchor=(22, 22)),
            ).add_to(geomap)

        if show_stores:
            brand_colors = {'Coop': '#FFD700', 'Migros': '#FF3B30', 'Denner': '#FF8A3D',
                            'Aldi': '#4FB3E8', 'Lidl': '#34D399', 'Other': '#9CA3AF'}
            for _, row in joined_stores.iterrows():
                brand = row.get('brand_category', 'Other')
                if pd.isna(brand):
                    brand = 'Other'
                if brand not in brand_filter:
                    continue
                if row['COMMUNE_NAME'] in df_filtered['COMMUNE_NAME'].values:
                    bc = brand_colors.get(brand, brand_colors['Other'])
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5,
                        popup=folium.Popup(
                            f"<div style='font-family:monospace;font-size:11px;'>"
                            f"<b style='color:{bc};'>{brand}</b><br>"
                            f"Type: {row.get('shop', '?')}<br>"
                            f"Commune: {row['COMMUNE_NAME']}</div>",
                            max_width=180),
                        tooltip=f"{brand} - {row['COMMUNE_NAME']}",
                        color=bc, fill=True, fill_color=bc, fill_opacity=0.80, weight=1.5,
                    ).add_to(geomap)

        folium.LayerControl(collapsed=False).add_to(geomap)

    st_folium(geomap, height=650, use_container_width=True, returned_objects=[])
    st.markdown("<div style='height:1px'></div>", unsafe_allow_html=True)
