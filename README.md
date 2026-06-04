# Migros Location Optimizer

A data-science project that identifies the best locations for opening new **Migros**
branches in the **Canton of Geneva** using geospatial analytics and statistical modelling.

The project combines three data sources — existing supermarket locations, commune-level
demographics, and a purchasing-power proxy — to score and rank communes by their
attractiveness for a new store, and presents the result in an interactive Streamlit
dashboard ("Migros Location Intelligence").

## Project structure

```
Migros-Location-Optimizer/
├── app/
│   └── app.py            # Streamlit dashboard (the deliverable)
├── notebooks/            # Exploratory analysis by team members
│   ├── 01_get_data_Varantorn.ipynb        # Supermarket scraping / data collection
│   ├── 02_population_Alexandros.ipynb      # Geneva population processing
│   ├── migros_project.ipynb                # Core analysis
│   └── migros_project_extension.ipynb      # Extended analysis
├── data/
│   ├── geneva_supermarkets_data_with_address.csv
│   ├── OCS_POPBATLOG_COMMUNE.csv           # Commune demographics (OCS, ; separated)
│   ├── switzerland.geojson
│   └── finance/
│       ├── geneva_purchasing_power_proxy_all_years.csv
│       └── T_20_02_8_21.xlsx
├── requirements.txt      # Python dependencies
└── packages.txt          # System packages (for Streamlit Cloud deploy)
```

## Data sources

| File | Description |
|------|-------------|
| `data/geneva_supermarkets_data_with_address.csv` | Existing supermarkets in Geneva (lat/lon + address) |
| `data/OCS_POPBATLOG_COMMUNE.csv` | Commune-level population, age, housing & nationality stats |
| `data/finance/geneva_purchasing_power_proxy_all_years.csv` | Purchasing-power proxy per commune per year |
| `switzerland.geojson` | Country boundary geometry |

Commune boundaries are fetched live from OpenStreetMap via **OSMnx**
(`Canton of Geneva, Switzerland`, `admin_level=8`).

> **Note:** `app/app.py` loads the CSVs from this repo's `data/` folder over GitHub raw
> URLs, so the dashboard runs without the files being present locally — but the files
> must stay committed to the repo for the deployed app to work.

## Running locally

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. launch the dashboard
streamlit run app/app.py
```

On Debian/Ubuntu (or Streamlit Cloud) the system packages in `packages.txt`
(`libspatialindex-dev`, `libgeos-dev`) are required for `geopandas` / `osmnx`.

## Method (high level)

1. **Load & join** supermarket, demographic and purchasing-power data onto Geneva
   commune boundaries.
2. **Engineer features** — store count, % working-age, % single-family housing,
   % foreigners, purchasing power.
3. **Rank** the top communes by population and score them with a min–max weighted model.
4. **Visualise** the recommendations on interactive Folium maps and charts.
