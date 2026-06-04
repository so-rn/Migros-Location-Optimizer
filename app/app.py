import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium
from streamlit_folium import st_folium
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
MIGROS_ORANGE = '#FF6600'
MIGROS_TEAL   = '#0077B6'
BG_DARK       = '#0A0E17'
CARD_BG       = '#111827'
CARD_GLASS    = 'rgba(17,24,39,0.7)'
TEXT_LIGHT    = '#F0F6FC'
TEXT_MUTED    = '#7D8DA1'
BORDER        = '#1E2D3D'
GLOW          = 'rgba(255,102,0,0.18)'

# ─── GLOBAL CSS ────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Unbounded:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {BG_DARK};
    color: {TEXT_LIGHT};
  }}
  .stApp {{
    background: radial-gradient(ellipse at 20% 10%, rgba(255,102,0,0.07) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(0,119,182,0.06) 0%, transparent 50%),
                {BG_DARK};
  }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
  ::-webkit-scrollbar-track {{ background: {BG_DARK}; }}
  ::-webkit-scrollbar-thumb {{ background: {MIGROS_ORANGE}; border-radius: 4px; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #080c14 0%, #0d1624 100%);
    border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] * {{ color: {TEXT_LIGHT} !important; }}

  /* Top header bar */
  .top-bar {{
    background: linear-gradient(90deg, rgba(10,14,23,0.95) 0%, rgba(26,8,0,0.85) 50%, rgba(10,14,23,0.95) 100%);
    border-bottom: 1px solid rgba(255,102,0,0.4);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 20px 36px;
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 36px;
    border-radius: 0 0 20px 20px;
    box-shadow: 0 4px 40px rgba(255,102,0,0.12);
    position: relative;
    overflow: hidden;
  }}
  .top-bar::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, {MIGROS_ORANGE}, transparent);
    opacity: 0.6;
  }}
  .top-bar .logo {{
    background: {MIGROS_ORANGE};
    color: white;
    font-family: 'Unbounded', sans-serif;
    font-weight: 900;
    font-size: 20px;
    padding: 10px 16px;
    border-radius: 12px;
    letter-spacing: 1px;
    box-shadow: 0 0 30px rgba(255,102,0,0.5), 0 0 60px rgba(255,102,0,0.2);
  }}
  .top-bar .title {{
    font-family: 'Unbounded', sans-serif;
    font-size: 17px;
    font-weight: 700;
    color: {TEXT_LIGHT};
    letter-spacing: 0.5px;
  }}
  .top-bar .subtitle {{
    font-size: 11px;
    color: {TEXT_MUTED};
    font-family: 'Space Mono', monospace;
    margin-top: 3px;
  }}
  .top-bar .live-dot {{
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 10px #22c55e;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.5; transform: scale(1.3); }}
  }}

  /* KPI cards */
  .kpi-grid {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 20px 0 28px; }}
  .kpi-card {{
    background: {CARD_GLASS};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 20px 24px;
    min-width: 140px;
    flex: 1;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    cursor: default;
  }}
  .kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(255,102,0,0.15);
    border-color: rgba(255,102,0,0.4);
  }}
  .kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, {MIGROS_ORANGE}, transparent);
  }}
  .kpi-card .kpi-label {{
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: {TEXT_MUTED};
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
  }}
  .kpi-card .kpi-value {{
    font-family: 'Unbounded', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: {TEXT_LIGHT};
  }}
  .kpi-card .kpi-accent {{ color: {MIGROS_ORANGE}; }}
  .kpi-card .kpi-sub {{
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: {TEXT_MUTED};
    margin-top: 4px;
    letter-spacing: 0.5px;
  }}

  /* Stage badge */
  .stage-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,102,0,0.12);
    color: {MIGROS_ORANGE};
    border: 1px solid rgba(255,102,0,0.3);
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 2px;
    margin-bottom: 8px;
    text-transform: uppercase;
  }}
  .section-title {{
    font-family: 'Unbounded', sans-serif;
    font-size: 21px;
    font-weight: 700;
    color: {TEXT_LIGHT};
    margin-bottom: 4px;
    line-height: 1.3;
  }}
  .section-sub {{
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: {TEXT_MUTED};
    margin-bottom: 24px;
    letter-spacing: 0.3px;
  }}

  /* Champion card */
  .champion-card {{
    background: linear-gradient(135deg,rgba(26,8,0,0.9),rgba(42,18,0,0.85),rgba(26,8,0,0.9));
    border: 1px solid rgba(255,102,0,0.5);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    box-shadow: 0 0 80px rgba(255,102,0,0.2), inset 0 0 80px rgba(255,102,0,0.03);
    margin: 28px 0;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
  }}
  .champion-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, {MIGROS_ORANGE}, transparent);
  }}
  .champion-card::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,102,0,0.4), transparent);
  }}
  .champion-label {{
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: {MIGROS_ORANGE};
    font-weight: 700;
    letter-spacing: 4px;
    margin-bottom: 14px;
    text-transform: uppercase;
  }}
  .champion-name {{
    font-family: 'Unbounded', sans-serif;
    font-size: 40px;
    font-weight: 900;
    color: white;
    margin-bottom: 6px;
    text-shadow: 0 0 40px rgba(255,102,0,0.4);
  }}
  .champion-loc {{
    font-size: 12px;
    color: {TEXT_MUTED};
    margin-bottom: 32px;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
  }}
  .champion-stats {{
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .champion-stat {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 20px;
    min-width: 110px;
    transition: background 0.2s, border-color 0.2s;
  }}
  .champion-stat:hover {{
    background: rgba(255,102,0,0.08);
    border-color: rgba(255,102,0,0.3);
  }}
  .champion-stat-label {{
    font-family: 'Space Mono', monospace;
    font-size: 8px;
    color: {TEXT_MUTED};
    letter-spacing: 1.5px;
    margin-bottom: 8px;
    text-transform: uppercase;
  }}
  .champion-stat-value {{
    font-family: 'Unbounded', sans-serif;
    font-size: 19px;
    font-weight: 700;
  }}

  /* Funnel viz */
  .funnel-step {{
    background: {CARD_GLASS};
    border: 1px solid {BORDER};
    border-left: 3px solid {MIGROS_ORANGE};
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 18px;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
  }}
  .funnel-step:hover {{
    transform: translateX(4px);
    box-shadow: -4px 0 20px rgba(255,102,0,0.15);
  }}
  .funnel-num {{
    font-family: 'Unbounded', monospace;
    font-size: 26px;
    font-weight: 700;
    color: {MIGROS_ORANGE};
    min-width: 64px;
    text-shadow: 0 0 20px rgba(255,102,0,0.4);
  }}

  /* Table */
  .intel-table {{ width: 100%; border-collapse: collapse; font-family: 'Space Mono', monospace; font-size: 11px; }}
  .intel-table th {{
    background: rgba(255,102,0,0.06);
    padding: 11px 16px;
    text-align: left;
    color: {TEXT_MUTED};
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(255,102,0,0.25);
  }}
  .intel-table td {{ padding: 10px 16px; border-bottom: 1px solid {BORDER}; color: {TEXT_LIGHT}; font-family: 'DM Sans', sans-serif; font-size: 13px; }}
  .intel-table tr:nth-child(even) td {{ background: rgba(255,255,255,0.015); }}
  .intel-table tr:hover td {{ background: rgba(255,102,0,0.05); transition: background 0.15s; }}

  /* Glass button */
  .glass-btn {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,102,0,0.1);
    border: 1px solid rgba(255,102,0,0.35);
    color: {MIGROS_ORANGE};
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 10px 20px;
    border-radius: 8px;
    cursor: pointer;
    text-transform: uppercase;
    transition: all 0.2s;
    backdrop-filter: blur(10px);
  }}
  .glass-btn:hover {{
    background: rgba(255,102,0,0.2);
    border-color: {MIGROS_ORANGE};
    box-shadow: 0 0 20px rgba(255,102,0,0.25);
    transform: translateY(-1px);
  }}

  /* Matplotlib dark bg */
  .stPlotlyChart, .stPyplot {{ border-radius: 14px; overflow: hidden; }}

  /* Streamlit widgets — glass buttons */
  .stButton > button {{
    background: rgba(255,102,0,0.1) !important;
    color: {MIGROS_ORANGE} !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255,102,0,0.4) !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    letter-spacing: 1.5px !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    backdrop-filter: blur(10px) !important;
  }}
  .stButton > button:hover {{
    background: rgba(255,102,0,0.22) !important;
    border-color: {MIGROS_ORANGE} !important;
    box-shadow: 0 0 24px rgba(255,102,0,0.3) !important;
    transform: translateY(-2px) !important;
  }}

  /* Selectbox / slider */
  div[data-testid="stSelectbox"] > div > div {{
    background: {CARD_GLASS} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT_LIGHT} !important;
    backdrop-filter: blur(10px) !important;
  }}
  .stSlider [data-testid="stSlider"] {{
    accent-color: {MIGROS_ORANGE};
  }}

  div[data-testid="stMetric"] {{
    background: {CARD_GLASS};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 16px 20px;
    backdrop-filter: blur(10px);
  }}
  label, .stSelectbox label, .stSlider label, .stMultiSelect label {{
    color: {TEXT_MUTED} !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
  }}

  div[data-testid="stExpander"] {{
    background: {CARD_GLASS};
    border: 1px solid {BORDER};
    border-radius: 12px;
    backdrop-filter: blur(10px);
  }}

  /* Filter panel */
  .filter-panel {{
    background: {CARD_GLASS};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 24px;
    backdrop-filter: blur(12px);
  }}
  .filter-title {{
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: {TEXT_MUTED};
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 14px;
  }}

  /* Sidebar radio */
  [data-testid="stSidebar"] .stRadio > label {{
    font-family: 'Space Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 2px !important;
    color: {TEXT_MUTED} !important;
    text-transform: uppercase !important;
  }}

  /* Sidebar nav items */
  [data-testid="stSidebar"] div[data-testid="stRadio"] label {{
    padding: 8px 0 !important;
    cursor: pointer !important;
  }}

  /* Horizontal divider */
  hr {{ border: none; border-top: 1px solid {BORDER}; margin: 20px 0; }}

  /* Tooltip chip */
  .chip {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(255,255,255,0.04);
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: {TEXT_MUTED};
    margin-right: 6px;
    letter-spacing: 0.5px;
  }}
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB STYLE ──────────────────────────────────────────
sns.set_theme(style='darkgrid')
plt.rcParams.update({
    'figure.facecolor': BG_DARK, 'axes.facecolor': '#0d1624',
    'axes.edgecolor': BORDER, 'axes.labelcolor': TEXT_LIGHT,
    'xtick.color': TEXT_MUTED, 'ytick.color': TEXT_MUTED,
    'text.color': TEXT_LIGHT, 'grid.color': '#1a2535',
    'grid.linewidth': 0.5, 'font.family': 'DejaVu Sans',
    'axes.titlecolor': TEXT_LIGHT, 'axes.titleweight': 'bold',
    'axes.titlesize': 13, 'legend.facecolor': CARD_BG,
    'legend.edgecolor': BORDER, 'legend.labelcolor': TEXT_LIGHT,
})

# ─── DATA LOADING (CACHED) ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    BASE_URL   = 'https://raw.githubusercontent.com/so-rn/Migros-Location-Optimizer/main/data/'
    URL_STORES = BASE_URL + 'geneva_supermarkets_data_with_address.csv'
    URL_POP    = BASE_URL + 'OCS_POPBATLOG_COMMUNE.csv'
    URL_POWER  = BASE_URL + 'finance/geneva_purchasing_power_proxy_all_years.csv'
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
    boundaries = ox.features_from_place(
        'Canton of Geneva, Switzerland', tags={'admin_level': '8'}
    )
    boundaries = (
        boundaries[boundaries.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        .reset_index()[['name', 'geometry']]
        .rename(columns={'name': 'COMMUNE_NAME'})
        .to_crs(epsg=4326)
    )
    return boundaries

@st.cache_data(show_spinner=False)
def build_master(_stores_gdf, _df_pop, _df_power_2022, _boundaries):
    joined = gpd.sjoin(_stores_gdf, _boundaries, how='inner', predicate='intersects')
    store_counts = joined.groupby('COMMUNE_NAME').size().reset_index(name='STORE_COUNT')
    df_master = (
        _boundaries
        .merge(_df_pop, left_on='COMMUNE_NAME', right_on='COMMUNE', how='left')
        .merge(_df_power_2022[['commune', 'proxy_purchasing_power_median_chf']],
               left_on='COMMUNE_NAME', right_on='commune', how='left')
        .merge(store_counts, on='COMMUNE_NAME', how='left')
    )
    df_master['STORE_COUNT']       = df_master['STORE_COUNT'].fillna(0)
    df_master['PCT_WORKING_AGE']   = (df_master['AGE_20_64']    / df_master['POPULATION']) * 100
    df_master['PCT_SINGLE_FAMILY'] = (df_master['MAISON_INDIV'] / df_master['BATLOG_TOT']) * 100
    df_master['PCT_FOREIGNERS']    = (df_master['POP_ETR']      / df_master['POPULATION']) * 100
    df_clean = df_master.dropna(subset=['POPULATION', 'proxy_purchasing_power_median_chf']).copy()
    df_clean = df_clean[~df_clean['COMMUNE_NAME'].isin(['Genève', 'Geneve', 'Geneva'])].reset_index(drop=True)
    return df_clean, joined

@st.cache_data(show_spinner=False)
def run_pipeline(_df_clean):
    top20 = _df_clean.sort_values('POPULATION', ascending=False).head(20).copy().reset_index(drop=True)
    top20['RANK'] = range(1, 21)

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
    top5['RANK'] = range(1, 6)

    X_init = sm.add_constant(_df_clean[['POPULATION', 'proxy_purchasing_power_median_chf']])
    y_init = _df_clean['STORE_COUNT']
    init_model = sm.OLS(y_init, X_init).fit()
    cooks_d = init_model.get_influence().cooks_distance[0]
    df_f = _df_clean.copy()
    df_f['COOKS_D'] = cooks_d
    df_filtered = df_f[df_f['COOKS_D'] <= 4 / len(df_f)].copy().reset_index(drop=True)
    X_final = sm.add_constant(df_filtered[['POPULATION', 'proxy_purchasing_power_median_chf']])
    y_final = df_filtered['STORE_COUNT']
    final_model = sm.OLS(y_final, X_final).fit()
    df_filtered['PREDICTED_STORES'] = final_model.predict(X_final)
    df_filtered['OPPORTUNITY_SCORE'] = df_filtered['PREDICTED_STORES'] - df_filtered['STORE_COUNT']

    top5_ols = (
        top5
        .merge(df_filtered[['COMMUNE_NAME', 'PREDICTED_STORES', 'OPPORTUNITY_SCORE']],
               on='COMMUNE_NAME', how='left')
        .sort_values('OPPORTUNITY_SCORE', ascending=False)
        .reset_index(drop=True)
    )
    top5_ols['RANK'] = range(1, len(top5_ols) + 1)
    champion = top5_ols.iloc[0].copy()
    return top20, top5, top5_ols, champion, df_filtered, final_model

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:24px 12px 12px;">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
        <div style="background:{MIGROS_ORANGE};color:white;font-family:'Unbounded',sans-serif;
             font-weight:900;font-size:16px;padding:9px 14px;border-radius:10px;letter-spacing:1px;
             box-shadow:0 0 24px rgba(255,102,0,0.45);">M</div>
        <div>
          <div style="font-family:'Unbounded',sans-serif;font-size:12px;font-weight:700;color:{TEXT_LIGHT};line-height:1.4;">Location<br>Intelligence</div>
          <div style="font-family:'Space Mono',monospace;font-size:9px;color:{TEXT_MUTED};margin-top:3px;letter-spacing:1px;">GENEVA · CH</div>
        </div>
      </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,{BORDER},transparent);margin:0 8px 20px;"></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        ["🏠  Overview", "📊  Stage 1 — Population", "🧮  Stage 2 — Scoring",
         "📈  Stage 3 — OLS Model", "📋  Demographic Dashboard", "🗺️  Interactive Map"],
        label_visibility="visible"
    )

    st.markdown(f"""
    <div style="height:1px;background:linear-gradient(90deg,transparent,{BORDER},transparent);margin:20px 8px 16px;"></div>
    <div style="font-family:'Space Mono',monospace;font-size:9px;color:{TEXT_MUTED};padding:0 12px;letter-spacing:1px;">
      <div style="margin-bottom:10px;color:{MIGROS_ORANGE};letter-spacing:2px;">PIPELINE</div>
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
        <span>All communes</span><span style="color:{TEXT_LIGHT};">→ Top 20</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
        <span>Top 20</span><span style="color:{TEXT_LIGHT};">→ Top 5</span>
      </div>
      <div style="display:flex;justify-content:space-between;">
        <span>Top 5</span><span style="color:{MIGROS_ORANGE};">→ Champion ★</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── LOAD DATA ─────────────────────────────────────────────────
with st.spinner("Initialising intelligence engine…"):
    try:
        stores_gdf, df_pop, df_power_2022 = load_data()
        boundaries = load_boundaries()
        df_clean, joined_stores = build_master(stores_gdf, df_pop, df_power_2022, boundaries)
        top20, top5, top5_ols, champion, df_filtered, final_model = run_pipeline(df_clean)
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
    <div class="title">MIGROS Location Intelligence Engine</div>
    <div class="subtitle"><span class="live-dot"></span>STRATEGIC 3-STAGE FUNNEL · CANTON OF GENEVA · OLS REGRESSION</div>
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
    for num, title, sub in funnel_data:
        st.markdown(f"""
        <div class="funnel-step">
          <div class="funnel-num">{num}</div>
          <div>
            <div style="font-family:'Unbounded',sans-serif;font-weight:700;font-size:14px;">{title}</div>
            <div style="font-family:'Space Mono',monospace;font-size:10px;color:{TEXT_MUTED};margin-top:4px;">{sub}</div>
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
        fig, ax = plt.subplots(figsize=(13, max(5, len(top20_f) * 0.42)))
        colors = [MIGROS_ORANGE if i == 0 else MIGROS_TEAL for i in range(len(top20_f))]
        bars = ax.barh(top20_f['COMMUNE_NAME'], top20_f['POPULATION'], color=colors, edgecolor='none', height=0.65)
        ax.invert_yaxis()
        ax.set_xlabel('Resident Population', fontsize=11)
        ax.set_title('Stage 1  —  Top Communes by Population')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
        for bar, val in zip(bars, top20_f['POPULATION']):
            ax.text(val + 50, bar.get_y() + bar.get_height() / 2,
                    f'{val:,}', va='center', ha='left', fontsize=8, color=TEXT_MUTED)
        legend_patches = [
            mpatches.Patch(color=MIGROS_ORANGE, label='#1 Largest'),
            mpatches.Patch(color=MIGROS_TEAL,   label='Top 20'),
        ]
        ax.legend(handles=legend_patches, loc='lower right')
        plt.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close()

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
        <div style="background:{BG_DARK};border:1px solid {BORDER};border-radius:14px;padding:22px;overflow-x:auto;">
          <div style="margin-bottom:14px;display:flex;align-items:center;gap:10px;">
            <span class="stage-badge">STAGE 1 RESULTS</span>
            <span style="font-family:'DM Sans',sans-serif;font-size:14px;font-weight:500;color:{TEXT_MUTED};">{len(top20_f)} communes shown</span>
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
    st.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:10px;margin-top:8px;">
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
               ('Working Age', w_age, '#9B59B6'), ('Urban Density', w_urban, '#27AE60')]
    cols = st.columns(4)
    for col, (label, pct, color) in zip(cols, weights):
        with col:
            st.markdown(f"""
            <div style="background:{CARD_GLASS};border:1px solid {BORDER};border-radius:12px;
                 padding:16px;text-align:center;border-top:2px solid {color};
                 backdrop-filter:blur(10px);transition:all 0.2s;">
              <div style="font-family:'Space Mono',monospace;font-size:9px;color:{TEXT_MUTED};
                   margin-bottom:8px;letter-spacing:1.5px;text-transform:uppercase;">{label}</div>
              <div style="font-family:'Unbounded',sans-serif;font-size:26px;font-weight:700;color:{color};">{pct}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    dims        = ['SCORE_INCOME', 'SCORE_FOREIGN', 'SCORE_AGE', 'SCORE_URBAN']
    dim_labels  = [f'Income ({w_income}%)', f'Foreign % ({w_foreign}%)', f'Working Age ({w_age}%)', f'Urban Density ({w_urban}%)']
    dim_colors  = [MIGROS_ORANGE, MIGROS_TEAL, '#9B59B6', '#27AE60']
    dim_weights_list = [w_income/100, w_foreign/100, w_age/100, w_urban/100]
    pal5 = [MIGROS_ORANGE, MIGROS_TEAL, '#9B59B6', '#27AE60', '#E74C3C']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Stage 2  —  Socio-Economic Multi-Factor Analysis', fontsize=14, y=1.02)

    ax1 = axes[0]
    bars = ax1.barh(s2_data['COMMUNE_NAME'], s2_data['COMPOSITE_SCORE'],
                    color=pal5[:len(s2_data)], height=0.6, edgecolor='none')
    ax1.invert_yaxis()
    ax1.set_xlabel('Composite Score (0–1)', fontsize=10)
    ax1.set_title('Composite Socio-Economic Score')
    for bar, sc in zip(bars, s2_data['COMPOSITE_SCORE']):
        ax1.text(sc + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{sc:.3f}', va='center', fontsize=10, color=TEXT_LIGHT)

    ax2 = axes[1]
    x = np.arange(len(s2_data))
    bottom = np.zeros(len(s2_data))
    for dim, lbl, col, w in zip(dims, dim_labels, dim_colors, dim_weights_list):
        vals = s2_data[dim].values * w
        ax2.bar(x, vals, bottom=bottom, label=lbl, color=col, edgecolor='none', width=0.55, alpha=0.9)
        bottom += vals
    ax2.set_xticks(x)
    ax2.set_xticklabels(s2_data['COMMUNE_NAME'], rotation=22, ha='right', fontsize=9)
    ax2.set_ylabel('Weighted Score')
    ax2.set_title('Score Breakdown by Dimension')
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close()

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
    <div style="background:{BG_DARK};border:1px solid {BORDER};border-radius:14px;padding:22px;overflow-x:auto;">
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
elif page == "📈  Stage 3 — OLS Model":

    st.markdown(f"""
    <div class="stage-badge">◈ STAGE 3</div>
    <div class="section-title">OLS Regression — Opportunity Gap</div>
    <div class="section-sub">Gap = Predicted Stores − Current Stores · Highest gap = most under-served market</div>
    """, unsafe_allow_html=True)

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

    pal3 = [MIGROS_ORANGE if i == 0 else MIGROS_TEAL for i in range(len(top5_view))]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Stage 3  —  OLS Regression: Market Opportunity Analysis', fontsize=14, y=1.02)

    ax1 = axes[0]
    ax1.barh(top5_view['COMMUNE_NAME'], top5_view['OPPORTUNITY_SCORE'],
             color=pal3, height=0.6, edgecolor='none')
    ax1.invert_yaxis()
    ax1.set_xlabel('Opportunity Gap (Predicted − Actual Stores)', fontsize=10)
    ax1.set_title('Opportunity Gap by Commune')
    ax1.axvline(0, color='#444', linewidth=1, linestyle='--', alpha=0.6)
    for i, (_, row) in enumerate(top5_view.iterrows()):
        sign = '+' if row['OPPORTUNITY_SCORE'] >= 0 else ''
        ax1.text(row['OPPORTUNITY_SCORE'] + 0.015, i,
                 f"{sign}{row['OPPORTUNITY_SCORE']:.2f}", va='center', ha='left',
                 fontsize=10, color=TEXT_LIGHT)

    ax2 = axes[1]
    if not top5_view.empty:
        all_vals = list(top5_view['STORE_COUNT']) + list(top5_view['PREDICTED_STORES'].dropna())
        lim_max = max(all_vals) + 0.9
        lim_min = max(0, min(all_vals) - 0.3)
        ax2.plot([lim_min, lim_max], [lim_min, lim_max], color='#555', linestyle='--', lw=1.2,
                 label='Perfect equilibrium', zorder=1)
        for i, (_, row) in enumerate(top5_view.iterrows()):
            c = MIGROS_ORANGE if i == 0 else MIGROS_TEAL
            ax2.scatter(row['STORE_COUNT'], row['PREDICTED_STORES'],
                        color=c, s=220, zorder=5, edgecolors='white', lw=1)
            ax2.annotate(row['COMMUNE_NAME'], (row['STORE_COUNT'], row['PREDICTED_STORES']),
                         textcoords='offset points', xytext=(9, 4), fontsize=8.5, color=TEXT_LIGHT)
        ax2.set_xlim(lim_min, lim_max)
        ax2.set_ylim(lim_min, lim_max)
        ax2.set_xlabel('Actual Stores')
        ax2.set_ylabel('Model-Predicted Stores')
        ax2.set_title('Actual vs Predicted Stores')
        ax2.legend(fontsize=8)
        ax2.text(lim_min + 0.05, lim_max - 0.3, 'Above line = under-served',
                 fontsize=8, color=MIGROS_ORANGE, style='italic')

    plt.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close()

    with st.expander("📋 View OLS Regression Summary"):
        X_f = sm.add_constant(df_filtered[['POPULATION', 'proxy_purchasing_power_median_chf']])
        y_f = df_filtered['STORE_COUNT']
        model = sm.OLS(y_f, X_f).fit()
        st.code(str(model.summary()), language='text')

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    rows_html = ""
    for i, (_, row) in enumerate(top5_view.iterrows()):
        badge = {0: '🥇 ★', 1: '🥈', 2: '🥉'}.get(i, '')
        gap_color = MIGROS_ORANGE if row['OPPORTUNITY_SCORE'] == top5_view['OPPORTUNITY_SCORE'].max() else TEXT_LIGHT
        gap_val = f"+{row['OPPORTUNITY_SCORE']:.2f}" if row['OPPORTUNITY_SCORE'] >= 0 else f"{row['OPPORTUNITY_SCORE']:.2f}"
        rows_html += f"""<tr>
          <td>{int(row['RANK'])}</td>
          <td>{badge} {row['COMMUNE_NAME']}</td>
          <td>{int(row['POPULATION']):,}</td>
          <td>{int(row['STORE_COUNT'])}</td>
          <td>{row['PREDICTED_STORES']:.2f}</td>
          <td><b style="color:{gap_color};">{gap_val}</b></td>
          <td>{row['COMPOSITE_SCORE']:.4f}</td>
        </tr>"""
    st.markdown(f"""
    <div style="background:{BG_DARK};border:1px solid {BORDER};border-radius:14px;padding:22px;overflow-x:auto;">
      <div style="margin-bottom:14px;">
        <span class="stage-badge">STAGE 3 RESULTS</span>
      </div>
      <table class="intel-table">
        <thead><tr>
          <th>RANK</th><th>COMMUNE</th><th>POPULATION</th><th>STORES NOW</th>
          <th>PREDICTED</th><th>GAP (OPPORTUNITY)</th><th>SOCIO SCORE</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Champion card
    champ_name = champion['COMMUNE_NAME']
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
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        exclude_outliers = st.checkbox("Exclude outliers (Cook's D)", value=False)
    with dc2:
        min_pop_demo = st.slider("Min Population", 0, 20000, 0, step=1000)
    with dc3:
        highlight_top = st.selectbox("Highlight Commune", ["Champion Only"] + list(df_filtered['COMMUNE_NAME'].sort_values()))
    st.markdown('</div>', unsafe_allow_html=True)

    champ_name = champion['COMMUNE_NAME']
    df_demo = df_filtered.copy()
    if min_pop_demo > 0:
        df_demo = df_demo[df_demo['POPULATION'] >= min_pop_demo]

    highlight_name = champ_name if highlight_top == "Champion Only" else highlight_top

    panels = [
        ('POPULATION',                       '#4ECDC4', 'Population vs Supermarket Count',     'Resident Population'),
        ('PCT_WORKING_AGE',                  '#45B7D1', 'Working-Age Share vs Supermarkets',   'Working-Age (%)'),
        ('PCT_SINGLE_FAMILY',                '#96CEB4', 'Single-Family Housing vs Supermarkets','Single-Family (%)'),
        ('PCT_FOREIGNERS',                   '#DDA0DD', 'Foreign-Resident Share vs Supermarkets','Foreign Residents (%)'),
        ('proxy_purchasing_power_median_chf','#FFB347', 'Purchasing Power vs Supermarkets',    'Median Income (CHF)'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Geneva Retail Analytics  —  5-Factor Socio-Demographic Dashboard', fontsize=14, y=1.01)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]

    for (r, c), (xcol, color, title, xlab) in zip(positions, panels):
        ax = axes[r][c]
        sns.regplot(ax=ax, x=xcol, y='STORE_COUNT', data=df_demo, color=color,
                    scatter_kws={'alpha': 0.65, 's': 45, 'zorder': 3},
                    line_kws={'color': MIGROS_ORANGE, 'lw': 2.2, 'zorder': 2})
        h_row = df_demo[df_demo['COMMUNE_NAME'] == highlight_name]
        if not h_row.empty:
            ax.scatter(h_row[xcol], h_row['STORE_COUNT'],
                       color=MIGROS_ORANGE, s=220, zorder=10, edgecolors='white', lw=1.5,
                       marker='*', label=f'★ {highlight_name}')
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel('Number of Supermarkets', fontsize=9)

    ax_box = axes[2][1]
    ax_box.set_facecolor(BG_DARK)
    ax_box.axis('off')
    for sp in ax_box.spines.values():
        sp.set_visible(True)
        sp.set_color(MIGROS_ORANGE)
        sp.set_linewidth(1)

    h_data = df_demo[df_demo['COMMUNE_NAME'] == highlight_name]
    if not h_data.empty:
        h = h_data.iloc[0]
        entries = [
            ('HIGHLIGHTED',   highlight_name,                                              MIGROS_ORANGE, 14, 'bold'),
            ('Population',    f"{int(h['POPULATION']):,}",                                TEXT_LIGHT,    11, 'normal'),
            ('Stores',        f"{int(h['STORE_COUNT'])}",                                 TEXT_LIGHT,    11, 'normal'),
            ('Predicted',     f"{h['PREDICTED_STORES']:.2f}",                             TEXT_LIGHT,    11, 'normal'),
            ('Opportunity',   f"+{h['OPPORTUNITY_SCORE']:.2f}",                           MIGROS_ORANGE, 13, 'bold'),
            ('Income (CHF)',  f"{int(h['proxy_purchasing_power_median_chf']):,}",          TEXT_LIGHT,    11, 'normal'),
            ('Foreign %',     f"{h['PCT_FOREIGNERS']:.1f}%",                              MIGROS_TEAL,   11, 'normal'),
        ]
        y_pos = 0.90
        for lbl, val, col, sz, wt in entries:
            ax_box.text(0.07, y_pos, f'{lbl}:', fontsize=9, color=TEXT_MUTED, transform=ax_box.transAxes, va='top')
            ax_box.text(0.55, y_pos, val, fontsize=sz, color=col, fontweight=wt, transform=ax_box.transAxes, va='top')
            y_pos -= 0.12

    plt.tight_layout(pad=1.8)
    st.pyplot(fig)
    plt.close()


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
        brand_filter = st.multiselect("Filter Brands", ["Coop", "Migros", "other"], default=["Coop", "Migros", "other"])
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
                '<div style="background:#FF6600;border:3px solid #fff;border-radius:50%;'
                'width:40px;height:40px;display:flex;align-items:center;justify-content:center;'
                'font-size:22px;box-shadow:0 0 20px #FF6600,0 0 40px rgba(255,102,0,0.5);">&#9733;</div>'
            )
            popup_content = (
                f"<div style='font-family:monospace;background:#1a0800;color:#e6edf3;"
                f"padding:16px;border-radius:10px;border:2px solid #FF6600;min-width:210px;'>"
                f"<b style='color:#FF6600;font-size:13px;'>CHAMPION LOCATION</b><br><br>"
                f"<b style='font-size:16px;color:#fff;'>{champ_name}</b><br>"
                f"<span style='color:#aaa;font-size:11px;'>Canton of Geneva</span><br><br>"
                f"<span style='color:#8b949e;'>Population </span> {int(champion['POPULATION']):,}<br>"
                f"<span style='color:#8b949e;'>Stores now </span> {int(champion['STORE_COUNT'])}<br>"
                f"<span style='color:#8b949e;'>Predicted &nbsp;</span> {champion['PREDICTED_STORES']:.2f}<br>"
                f"<b style='color:#FF6600;'>Opportunity Gap: +{champion['OPPORTUNITY_SCORE']:.2f}</b><br><br>"
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
            brand_colors = {'Coop': '#FFD700', 'Migros': '#FF0000', 'other': '#3498db'}
            for _, row in joined_stores.iterrows():
                brand = row.get('brand_category', 'other')
                if brand not in brand_filter:
                    continue
                if row['COMMUNE_NAME'] in df_filtered['COMMUNE_NAME'].values:
                    bc = brand_colors.get(brand, brand_colors['other'])
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
