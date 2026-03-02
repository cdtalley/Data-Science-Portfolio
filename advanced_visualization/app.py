"""
Advanced Geospatial & Data Visualization — Dash app.
Dark, command-center style: Bootstrap theme + custom CSS + dark map + Plotly dark.
Run: python -m advanced_visualization.app   (from repo root)
"""
from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html

# Dark theme: CYBORG = terminal/spy aesthetic; custom.css in assets/ refines it
THEME = dbc.themes.CYBORG
FONT_LINK = "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap"

# Demo data
def _demo_points(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lat = 40.0 + rng.uniform(-1.2, 1.2, n)
    lon = -74.0 + rng.uniform(-1.5, 1.5, n)
    value = rng.exponential(5, n)
    region = rng.choice(["North", "South", "Central", "Coast"], n)
    return pd.DataFrame({"lat": lat, "lon": lon, "value": value, "region": region})


def _geo_json_from_df(df: pd.DataFrame) -> dict:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": {"value": round(row["value"], 2), "region": row["region"]},
        }
        for _, row in df.iterrows()
    ]
    return {"type": "FeatureCollection", "features": features}


# Plotly dark template — single place so all figures match
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(13,17,23,0.95)",
    plot_bgcolor="rgba(13,17,23,0.9)",
    font=dict(family="JetBrains Mono, SF Mono, monospace", color="#e6edf3", size=11),
    title_font=dict(size=14, color="#00d4ff"),
    margin=dict(l=0, r=0, t=32, b=0),
    hoverlabel=dict(bgcolor="#0d1117", bordercolor="#00d4ff", font_size=11),
    colorway=["#00d4ff", "#58a6ff", "#79c0ff", "#a5d6ff"],
)


app = dash.Dash(
    __name__,
    title="Geospatial Command",
    suppress_callback_exceptions=True,
    external_stylesheets=[THEME, FONT_LINK],
)
server = app.server

df_demo = _demo_points()
geo_json = _geo_json_from_df(df_demo)

# Dark map tiles (CartoDB Dark)
DARK_TILES = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
TILE_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>'

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("GEOSPATIAL COMMAND", className="fw-bold", style={"letterSpacing": "0.2em", "color": "#00d4ff"}),
            dbc.NavbarToggler(id="navbar-toggler"),
        ],
        fluid=True,
    ),
    dark=True,
    className="mb-0",
)

sidebar = html.Div(
    [
        html.H5("FILTERS", className="text-uppercase mb-3", style={"color": "#8b949e", "letterSpacing": "0.15em", "fontSize": "11px"}),
        html.Label("Region", className="mb-1"),
        dcc.Dropdown(
            id="region-filter",
            options=[{"label": "All", "value": "All"}] + [{"label": r, "value": r} for r in df_demo["region"].unique()],
            value="All",
            clearable=False,
            className="mb-3",
        ),
        html.Label("Value range", className="mb-1"),
        dcc.RangeSlider(
            id="value-range",
            min=float(df_demo["value"].min()),
            max=float(df_demo["value"].max()),
            step=0.5,
            value=[float(df_demo["value"].min()), float(df_demo["value"].max())],
            tooltip={"placement": "bottom", "template": "{value}"},
            className="mb-2",
        ),
        html.Div(id="value-summary", className="small"),
    ],
    className="av-sidebar p-4",
    style={"width": "240px", "height": "100vh", "overflowY": "auto"},
)

map_component = html.Div(
    dl.Map(
        [
            dl.TileLayer(url=DARK_TILES, attribution=TILE_ATTRIBUTION, maxZoom=19),
            dl.GeoJSON(
                id="geojson-layer",
                data=geo_json,
                cluster=True,
                zoomToBoundsOnClick=True,
                superClusterOptions={"radius": 120},
            ),
        ],
        id="map",
        style={"width": "100%", "height": "100%", "minHeight": "380px"},
        center=(40.0, -74.0),
        zoom=8,
    ),
    className="av-map-wrapper",
    style={"height": "380px"},
)

app.layout = html.Div(
    [
        html.Link(rel="stylesheet", href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"),
        navbar,
        html.Div(
            [
                sidebar,
                html.Div(
                    [
                        dbc.Container(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Card([dbc.CardBody(map_component)], className="av-card border-0 mb-3"),
                                            md=12,
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id="plotly-scatter"))], className="av-card border-0"), md=6),
                                        dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id="plotly-treemap"))], className="av-card border-0"), md=6),
                                    ],
                                ),
                            ],
                            fluid=True,
                            className="py-4",
                        ),
                    ],
                    style={"flex": 1, "minWidth": 0, "overflow": "auto"},
                ),
            ],
            style={"display": "flex", "height": "calc(100vh - 56px)"},
        ),
    ],
    style={"fontFamily": "JetBrains Mono, SF Mono, Consolas, monospace"},
)


@app.callback(
    Output("value-summary", "children"),
    Input("value-range", "value"),
)
def _update_value_summary(value_range: list[float] | None) -> str:
    if not value_range or len(value_range) != 2:
        return ""
    return f">> {value_range[0]:.1f} – {value_range[1]:.1f}"


@app.callback(
    [Output("geojson-layer", "data"), Output("plotly-scatter", "figure"), Output("plotly-treemap", "figure")],
    [Input("region-filter", "value"), Input("value-range", "value")],
)
def _update_map_and_charts(region: str, value_range: list[float] | None) -> tuple[dict, dict, dict]:
    f = df_demo.copy()
    if region and region != "All":
        f = f[f["region"] == region]
    if value_range and len(value_range) == 2:
        f = f[(f["value"] >= value_range[0]) & (f["value"] <= value_range[1])]
    geo = _geo_json_from_df(f)

    scatter = px.scatter(
        f,
        x="lon",
        y="lat",
        size="value",
        color="region",
        hover_data=["value", "region"],
        title="Points by location",
    )
    scatter.update_layout(**PLOTLY_DARK, height=260)

    treemap = px.treemap(
        f,
        path=["region"],
        values=f["value"].round(2),
        color="value",
        color_continuous_scale=["#0d1117", "#00d4ff"],
        title="Value by region",
    )
    treemap.update_layout(**PLOTLY_DARK, height=260)

    return geo, scatter, treemap


if __name__ == "__main__":
    app.run(debug=True, port=8050)
