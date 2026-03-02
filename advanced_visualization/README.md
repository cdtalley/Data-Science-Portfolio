# Advanced Visualization & Geospatial Toolkit

Standalone toolkit for **advanced data visualization** and **geospatial visualization**, aimed at real-estate–grade and analytics use cases (choropleths, heatmaps, network maps, 3D, proportional symbols, catchment-style analysis).

**App stack: Dash (Plotly)** — callback-based (no full-script rerun), production-ready, native Plotly + dash-leaflet for maps. Avoids Streamlit’s rerun model and gives finer control over layout and state.

## Stack Overview

| Use case | Primary tool | Notes |
|----------|--------------|--------|
| **Interactive web maps** (markers, popups, tiles) | **Folium** | Leaflet-based; outputs HTML; no API key. |
| **Styled web maps** (Mapbox, 3D, custom styling) | **Plotly + Mapbox** | Best polish; free Mapbox token for high limits. |
| **Vector layers + basemaps** (GeoDataFrames, shapefiles) | **GeoPandas + contextily** | Static publication-quality; OSM/Carto basemaps. |
| **Large-scale / GPU-style** (hexbins, heatmaps, arcs, 3D) | **Pydeck** | deck.gl; handles large point sets. |
| **Choropleths & classification** | **GeoPandas + mapclassify** | Jenks, quantiles, etc. |
| **Advanced non-map viz** | **Plotly** | Treemap, sunburst, parallel coords, 3D scatter. |

Optional additions:

- **Kepler.gl**: Very large datasets, time-series, trips; often used via export or Jupyter integration.
- **Leafmap**: Unified wrapper (Folium, ipyleaflet, Plotly, Pydeck, Kepler); good for quick exploration.

## Setup

From repo root:

```bash
pip install -r requirements-core.txt
pip install -r advanced_visualization/requirements-viz.txt
```

For Plotly Mapbox (optional but recommended): create a free [Mapbox](https://www.mapbox.com/) account and set:

```bash
set MAPBOX_ACCESS_TOKEN=your_token   # Windows
export MAPBOX_ACCESS_TOKEN=your_token  # Linux/macOS
```

Or in Python: `os.environ["MAPBOX_ACCESS_TOKEN"] = "your_token"`.

## Contents

- **`app.py`** – **Dash app** (callback-based; no full-script rerun). Leaflet map (dash-leaflet) with clustered points, sidebar filters, and linked Plotly scatter + treemap. Run: `python -m advanced_visualization.app` from repo root; open http://localhost:8050.
- **`requirements-viz.txt`** – Python dependencies for this toolkit.

## Run the app

From repo root:

```bash
pip install -r requirements-core.txt
pip install -r advanced_visualization/requirements-viz.txt
python -m advanced_visualization.app
```

Then open **http://localhost:8050**. Use the sidebar to filter by region and value range; the map and charts update via callbacks.

## When to Use What

- **Real estate / site selection**: Choropleths (value by area), heatmaps (density), buffers/catchments (GeoPandas), proportional symbols (Folium/Plotly).
- **Transport / networks**: Line layers (GeoPandas or Pydeck), arc layers (Pydeck), node–edge on basemap.
- **Very large point data**: Pydeck hexbin/heatmap or Kepler.gl; consider Datashader for raster.
- **Presentations / reports**: Plotly (interactive) or GeoPandas+contextily (static PNG/SVG).
- **Dashboards / apps**: **Dash** (callback-based, production-ready); embed Leaflet (dash-leaflet), Plotly, Pydeck.

All examples use built-in or synthetic data; swap in your GeoDataFrames or DataFrames with `geometry` / `lat`/`lon` as needed.
