import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
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

# ─── MATPLOTLIB STYLE ──────────────────────────────────────────
sns.set_theme(style='darkgrid')
plt.rcParams.update({
    'figure.facecolor': BG_DARK, 'axes.facecolor': '#101522',
    'axes.edgecolor': BORDER, 'axes.labelcolor': TEXT_LIGHT,
    'xtick.color': TEXT_MUTED, 'ytick.color': TEXT_MUTED,
    'text.color': TEXT_LIGHT, 'grid.color': '#1B2433',
    'grid.linewidth': 0.6, 'font.family': 'DejaVu Sans',
    'axes.titlecolor': TEXT_LIGHT, 'axes.titleweight': 'bold',
    'axes.titlesize': 13, 'axes.titlepad': 14,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.facecolor': CARD_BG, 'legend.edgecolor': BORDER,
    'legend.labelcolor': TEXT_LIGHT, 'legend.framealpha': 0.9,
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
with st.spinner("Initialising intelligence engine…"):
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
    pal5 = [MIGROS_ORANGE, MIGROS_TEAL, '#A78BFA', '#34D399', '#F87171']

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
    comp = top5_ols.sort_values('BLENDED_OPPORTUNITY', ascending=False).reset_index(drop=True)
    figc, axc = plt.subplots(figsize=(13, 5.2))
    yb = np.arange(len(comp))
    h = 0.26
    axc.barh(yb - h, comp['OPPORTUNITY_SCORE'], height=h, color=MIGROS_TEAL, label='OLS gap')
    axc.barh(yb,      comp['ML_OPPORTUNITY'],   height=h, color='#A78BFA', label='ML gap')
    axc.barh(yb + h,  comp['BLENDED_OPPORTUNITY'], height=h, color=MIGROS_ORANGE, label='Blended')
    axc.set_yticks(yb)
    axc.set_yticklabels(comp['COMMUNE_NAME'])
    axc.invert_yaxis()
    axc.axvline(0, color='#444', lw=1, ls='--', alpha=0.6)
    axc.set_xlabel('Opportunity Gap  (predicted − actual stores)')
    axc.set_title('Model Agreement on Under-supply')
    axc.legend(loc='lower right', fontsize=9)
    plt.tight_layout(pad=1.2)
    st.pyplot(figc)
    plt.close()

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
    fig, ax = plt.subplots(figsize=(13, max(4, len(imp) * 0.42)))
    ax.barh(imp['label'], imp['importance'], color=MIGROS_ORANGE, height=0.62, edgecolor='none')
    ax.invert_yaxis()
    ax.set_xlabel('Relative importance')
    ax.set_title('What Drives the ML Store-Demand Model')
    for y, v in enumerate(imp['importance']):
        ax.text(v + 0.005, y, f'{v:.2f}', va='center', fontsize=9, color=TEXT_MUTED)
    plt.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close()

    # ── Residual diagnostics ───────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Residual Diagnostics', fontsize=14, y=1.0)

    a = axes[0][0]
    a.scatter(df_filtered['PREDICTED_STORES'], df_filtered['OLS_RESID'],
              color=MIGROS_TEAL, s=55, alpha=0.8, edgecolors='white', lw=0.5)
    a.axhline(0, color=MIGROS_ORANGE, ls='--', lw=1.2)
    a.set_xlabel('OLS fitted'); a.set_ylabel('Residual'); a.set_title('OLS — Fitted vs Residual')

    b = axes[0][1]
    b.scatter(df_filtered['ML_PRED'], df_filtered['ML_RESID'],
              color='#A78BFA', s=55, alpha=0.8, edgecolors='white', lw=0.5)
    b.axhline(0, color=MIGROS_ORANGE, ls='--', lw=1.2)
    b.set_xlabel('ML fitted'); b.set_ylabel('Residual'); b.set_title('ML — Fitted vs Residual')

    c = axes[1][0]
    stats.probplot(df_filtered['OLS_RESID'], dist='norm', plot=c)
    c.get_lines()[0].set_color(MIGROS_TEAL); c.get_lines()[0].set_markeredgecolor('white')
    c.get_lines()[1].set_color(MIGROS_ORANGE)
    c.set_title('OLS Residuals — Normal Q-Q')

    d = axes[1][1]
    mx = max(df_filtered['STORE_COUNT'].max(), df_filtered['ML_CV_PRED'].max()) + 0.5
    d.plot([0, mx], [0, mx], color='#555', ls='--', lw=1.2, label='Perfect')
    d.scatter(df_filtered['STORE_COUNT'], df_filtered['ML_CV_PRED'],
              color=MIGROS_ORANGE, s=60, alpha=0.85, edgecolors='white', lw=0.6)
    d.set_xlabel('Actual stores'); d.set_ylabel('ML cross-validated prediction')
    d.set_title('ML — Actual vs Out-of-Sample Prediction'); d.legend(fontsize=8)

    plt.tight_layout(pad=1.6)
    st.pyplot(fig)
    plt.close()

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
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(cat['NEAREST_KM'].dropna(), bins=18, color=MIGROS_TEAL, edgecolor=BG_DARK)
    axes[0].set_xlabel('Distance to nearest other store (km)')
    axes[0].set_ylabel('Stores'); axes[0].set_title('Isolation of Existing Stores')
    by_brand = cat.groupby('brand_category')['SATURATION'].mean().sort_values(ascending=False)
    axes[1].bar(by_brand.index, by_brand.values, color=MIGROS_ORANGE, edgecolor='none')
    axes[1].set_ylabel('Mean saturation index'); axes[1].set_title('Average Saturation by Brand')
    axes[1].tick_params(axis='x', rotation=20)
    plt.tight_layout(pad=1.4)
    st.pyplot(fig)
    plt.close()

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
