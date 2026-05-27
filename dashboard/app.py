"""
finbert-signals — redesigned dashboard.
Ticker pills at top, 52-week sentiment heatmap, entity bar chart,
score decomposition, VIX regime scatter, transcript scrubber,
alert threshold builder. Port 8052.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
import requests as req_lib
import yfinance as yf

import dash
from dash import dcc, html, Input, Output, State, dash_table, no_update, ALL, ctx
import dash_bootstrap_components as dbc

from fetchers.edgar import fetch_and_store as edgar_fetch
from fetchers.news import fetch_and_store as news_fetch
from fetchers.prices import fetch_and_store as price_fetch
from nlp.pipeline import score_filings, score_articles
from signals.aligner import get_aligned, get_daily_sentiment
from signals.generator import generate_signals
from signals.correlation import analyze as correlate
from backtest.engine import run as backtest_run
from backtest.metrics import compute as backtest_metrics
from backtest.ic import (
    load_signals_with_horizons,
    compute_ic_series,
    compute_rolling_ic,
    compute_pnl_attribution as ic_pnl_attribution,
)
from fetchers.prices import compute_and_store_horizons
from db import get_conn, migrate
from tickers import get_sp500_tickers, ticker_name_map

# ── Constants ──────────────────────────────────────────────────────────────────

_POS_COLOR  = "#2d6a4f"
_NEG_COLOR  = "#c0392b"
_NEUT_COLOR = "#adb5bd"
_SIG_COLOR  = {"LONG": _POS_COLOR, "SHORT": _NEG_COLOR, "HOLD": _NEUT_COLOR}

_REGIME_PALETTE = {
    "Q1": "#4a90d9", "Q2": "#27ae60",
    "Q3": "#f39c12", "Q4": "#e74c3c", "Unknown": _NEUT_COLOR,
}
_SEC_COLORS  = {"financial": "#0d6efd", "management": "#6f42c1",
                "guidance": "#fd7e14", "disclaimer": _NEUT_COLOR, "overview": "#6c757d"}
_SEC_SYMBOLS = {"financial": "diamond", "management": "star",
                "guidance": "triangle-up", "disclaimer": "square", "overview": "circle"}

TICKERS   = get_sp500_tickers()
_NAME_MAP = ticker_name_map()

QUICK_TICKERS = ["NVDA", "MSFT", "AMZN", "META", "TSLA", "AAPL", "GOOGL", "JPM"]

migrate()   # ensure schema is up to date

# ── Theme helpers ──────────────────────────────────────────────────────────────

def _bg(dark):      return "#1e2124" if dark else "white"
def _plot_bg(dark): return "#282b30" if dark else "#fafafa"
def _grid(dark):    return "#3d4045" if dark else "#eeeeee"
def _zero(dark):    return "#4a5568" if dark else "#dddddd"
def _font(dark):    return "#dee2e6" if dark else "#1a1a1a"

def _base_layout(dark):
    return dict(
        paper_bgcolor=_bg(dark), plot_bgcolor=_plot_bg(dark),
        font={"color": _font(dark), "family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"},
        margin=dict(l=0, r=0, t=10, b=0),
    )

def _empty_fig(msg="No data", dark=False):
    fig = go.Figure()
    fig.update_layout(
        **_base_layout(dark),
        annotations=[{"text": msg, "showarrow": False,
                      "font": {"size": 13, "color": "#adb5bd" if dark else "#6c757d"},
                      "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}],
    )
    return fig

def _err(label, detail, dark):
    ef = _empty_fig(label, dark)
    alert = dbc.Alert([html.Strong(label + ". "), detail],
                      color="danger", dismissable=True, className="mb-0")
    # 19 outputs total: 10 metric values, heatmap, sent, equity, decomp,
    # decomp-labels, table, source, status, signals-store
    return ("—", "", "—", "", "—", "", "—", "", "—", "",
            ef, ef, ef, ef, "", no_update, "", label, no_update)


# ── Layout helpers ─────────────────────────────────────────────────────────────

def _metric_card(label, val_id, sub_id):
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.Span(label, className="label-xs"),
            html.Div("—", id=val_id, className="metric-big"),
            html.Div("", id=sub_id, className="metric-sub"),
        ]), className="fin-card h-100"),
        width=True,
    )


def _section_card(title, *children, id_suffix=""):
    return dbc.Card(dbc.CardBody([
        html.Span(title, className="label-xs"),
        *children,
    ]), className="fin-card h-100")


# ── App ────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
app.title = "finbert-signals"
server = app.server

app.layout = dbc.Container([
    # Hidden stores
    dcc.Store(id="dark-store",    data=False),
    dcc.Store(id="signals-store"),
    dcc.Store(id="scrubber-store"),
    dcc.Store(id="ticker-store",  data="AAPL"),
    dcc.Download(id="csv-dl"),

    # ── Navbar ─────────────────────────────────────────────────────────────────
    dbc.Navbar(dbc.Container([
        dbc.NavbarBrand("finbert-signals", className="fw-bold fs-5 me-3"),
        html.Span("FinBERT · SEC EDGAR · NewsAPI",
                  className="text-muted small d-none d-md-inline"),
        dbc.Nav(dbc.NavItem(
            dbc.Switch(id="dark-toggle", value=False, className="mb-0 ms-3",
                       label=html.Span("🌙 Dark", style={"fontSize": "0.82rem"})),
        ), className="ms-auto d-flex align-items-center"),
    ], fluid=True), color="white", className="mb-3 border-bottom"),

    # ── Ticker row ─────────────────────────────────────────────────────────────
    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Span("Active Ticker", className="label-xs"),
                html.Div([
                    # Quick-access pills
                    *[
                        html.Button(t,
                            id={"type": "ticker-pill", "ticker": t},
                            className="ticker-pill ticker-pill-active" if t == "AAPL" else "ticker-pill",
                            n_clicks=0)
                        for t in QUICK_TICKERS
                    ],
                    # S&P 500 dropdown for any ticker
                    dcc.Dropdown(
                        id="custom-ticker-dd",
                        options=TICKERS,
                        placeholder="+ Any S&P 500 ticker…",
                        searchable=True, clearable=True,
                        style={"width": "220px", "display": "inline-block",
                               "verticalAlign": "middle", "fontSize": "0.82rem"},
                        className="ms-1",
                    ),
                    # Company name (hidden input, auto-filled)
                    dcc.Input(id="company-input", type="hidden", value="Apple"),
                    # Run & export
                    html.Button("Run Analysis", id="run-btn",
                                className="run-btn-primary ms-3"),
                    html.Button("⬇ CSV", id="export-btn",
                                className="ticker-pill ms-2",
                                style={"fontWeight": "500"}),
                ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "gap": "2px"}),
            ], width=10),
            dbc.Col([
                html.Div(id="source-label", className="text-muted text-end",
                         style={"fontSize": "0.75rem", "paddingTop": "22px"}),
            ], width=2),
        ]),
        html.Div(id="status-msg", className="text-muted mt-1",
                 style={"fontSize": "0.75rem"}),
    ]), className="fin-card mb-3"),

    # ── All chart content inside a single fullscreen loader ────────────────────
    dcc.Loading(
        type="circle",
        fullscreen=True,
        overlay_style={"visibility": "visible", "opacity": 0.8,
                        "backgroundColor": "rgba(0,0,0,0.4)"},
        children=[
            # Metric cards
            dbc.Row([
                _metric_card("FinBERT Score",     "metric-score",  "metric-score-sub"),
                _metric_card("Next-Day Alpha",     "metric-alpha",  "metric-alpha-sub"),
                _metric_card("Signal Precision",   "metric-prec",   "metric-prec-sub"),
                _metric_card("Sharpe Ratio",       "metric-sharpe", "metric-sharpe-sub"),
                _metric_card("Signals (n)",        "metric-n",      "metric-n-sub"),
            ], className="mb-3 g-2"),

            # Row 1: 52-week heatmap + entity bar
            dbc.Row([
                dbc.Col(_section_card(
                    "Sentiment Heatmap — 52 Weeks",
                    dcc.Graph(id="heatmap-chart", style={"height": "160px"},
                              config={"displayModeBar": False}),
                    html.Div([
                        html.Span(className="legend-swatch",
                                  style={"background": _NEG_COLOR}),
                        html.Span("Bearish", style={"fontSize": "0.7rem", "color": "#6c757d"}),
                        html.Span(className="legend-swatch ms-2",
                                  style={"background": "#f0f0f0", "border": "1px solid #ddd"}),
                        html.Span("Neutral", style={"fontSize": "0.7rem", "color": "#6c757d"}),
                        html.Span(className="legend-swatch ms-2",
                                  style={"background": _POS_COLOR}),
                        html.Span("Bullish", style={"fontSize": "0.7rem", "color": "#6c757d"}),
                        html.Span("  ○  Earnings filing", style={"fontSize": "0.7rem", "color": "#6c757d"}),
                    ], className="heatmap-legend mt-1"),
                ), width=7),
                dbc.Col(_section_card(
                    "Entity-Level Sentiment",
                    dcc.Graph(id="entity-bar-chart", style={"height": "180px"},
                              config={"displayModeBar": False}),
                    html.Div([
                        dbc.Button("Drill into sentences ↗", id="entity-detail-btn",
                                   size="sm", outline=True, color="secondary",
                                   className="mt-1", style={"fontSize": "0.75rem"}),
                    ]),
                    html.Div(id="entity-source-label",
                             className="text-muted mt-1", style={"fontSize": "0.72rem"}),
                ), width=5),
            ], className="mb-3 g-2"),

            # Row 2: Regime scatter + Transcript scrubber
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Span("Sentiment Shift vs Next-Day Return",
                                          className="label-xs"), width=9),
                        dbc.Col(dbc.Switch(id="regime-toggle", value=True,
                                          label=html.Span("VIX overlay",
                                                          style={"fontSize": "0.75rem"}),
                                          className="mb-0"), width=3, className="text-end"),
                    ], className="align-items-center mb-1"),
                    dcc.Graph(id="regime-scatter", style={"height": "270px"},
                              config={"displayModeBar": False}),
                ]), className="fin-card"), width=7),

                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Span("Transcript Timeline", className="label-xs"),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id="filing-dd", placeholder="Select 8-K filing…",
                                             clearable=False, style={"fontSize": "0.82rem"})),
                        dbc.Col(html.Small(id="scrubber-meta", className="text-muted"),
                                width="auto", className="d-flex align-items-center"),
                    ], className="mb-2"),
                    dcc.Loading(type="dot", children=[
                        dcc.Graph(id="scrubber-chart", style={"height": "200px"},
                                  config={"displayModeBar": False}),
                    ]),
                    html.Div(id="scrubber-text", className="mt-2 p-2",
                             style={"fontSize": "0.82rem", "lineHeight": "1.6",
                                    "borderRadius": "6px", "border": "1px solid #eee",
                                    "minHeight": "48px"}),
                ]), className="fin-card"), width=5),
            ], className="mb-3 g-2"),

            # Score decomposition
            dbc.Card(dbc.CardBody([
                html.Span("Score Decomposition", className="label-xs"),
                dcc.Graph(id="score-decomp", style={"height": "48px"},
                          config={"displayModeBar": False}),
                html.Div(id="decomp-labels", className="text-muted mt-1",
                         style={"fontSize": "0.75rem"}),
            ]), className="fin-card mb-3"),

            # Sentiment + equity charts (existing)
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Span("Daily Sentiment Score & Price", className="label-xs"),
                    dcc.Graph(id="sentiment-chart", style={"height": "280px"},
                              config={"displayModeBar": False}),
                ]), className="fin-card"), width=7),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Span("Equity Curve vs Buy & Hold", className="label-xs"),
                    dcc.Graph(id="equity-chart", style={"height": "280px"},
                              config={"displayModeBar": False}),
                ]), className="fin-card"), width=5),
            ], className="mb-3 g-2"),

            # Signal table
            dbc.Card(dbc.CardBody([
                html.Span("Signal Detail Table — most recent 50", className="label-xs"),
                html.Div(id="signal-table"),
            ]), className="fin-card mb-3"),

            # Alert threshold builder
            dbc.Card(dbc.CardBody([
                html.Span("Alert Threshold Builder", className="label-xs"),
                html.P("Drag thresholds to explore precision / recall without re-running analysis.",
                       className="text-muted mb-3", style={"fontSize": "0.8rem"}),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("LONG threshold (score ≥)",
                                  style={"fontSize": "0.78rem", "color": _POS_COLOR}),
                        dcc.Slider(id="long-slider", min=0, max=1, step=0.01, value=0.1,
                                   marks={0: "0", 0.25: ".25", 0.5: ".50", 0.75: ".75", 1: "1"},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ], width=6),
                    dbc.Col([
                        dbc.Label("SHORT threshold (score ≤)",
                                  style={"fontSize": "0.78rem", "color": _NEG_COLOR}),
                        dcc.Slider(id="short-slider", min=-1, max=0, step=0.01, value=-0.1,
                                   marks={-1: "-1", -0.75: "-.75", -0.5: "-.50", -0.25: "-.25", 0: "0"},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ], width=6),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="threshold-hist", style={"height": "220px"},
                                     config={"displayModeBar": False}), width=8),
                    dbc.Col(html.Div(id="threshold-metrics"), width=4,
                            className="d-flex align-items-center"),
                ]),
            ]), className="fin-card mb-4"),
        ],
    ),

    # ── Panel 5 — Alpha Decay & PnL Attribution ────────────────────────────────
    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Span("Alpha Decay & PnL Attribution", className="label-xs"),
                html.P(
                    "IC decay shows the horizon at which the FinBERT signal loses statistical "
                    "edge. PnL attribution decomposes gross returns into signal alpha, "
                    "timing slippage, and execution drag. Run Analysis first to populate data.",
                    className="text-muted mb-0",
                    style={"fontSize": "0.78rem"},
                ),
            ], width=10),
            dbc.Col(
                dbc.Badge(id="ic-data-badge", color="secondary",
                          style={"fontSize": "0.72rem"}),
                width=2, className="text-end d-flex align-items-center justify-content-end",
            ),
        ], className="mb-3 align-items-start"),

        html.Div(id="ic-panel-alert"),

        dcc.Loading(type="dot", children=[
            dbc.Row([
                # Chart A — IC Decay Curve
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Span("Chart A — IC Decay Curve", className="label-xs"),
                    dcc.Graph(id="ic-decay-fig",
                              style={"height": "300px"},
                              config={"displayModeBar": False}),
                ]), className="fin-card"), width=5),

                # Chart B — PnL Attribution Waterfall
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Span("Chart B — PnL Attribution", className="label-xs"),
                    dcc.Graph(id="ic-waterfall-fig",
                              style={"height": "300px"},
                              config={"displayModeBar": False}),
                ]), className="fin-card"), width=3),

                # Chart C — Rolling IC
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Span("Chart C — Rolling IC (60-trade window, 1d horizon)",
                              className="label-xs"),
                    dcc.Graph(id="ic-rolling-fig",
                              style={"height": "300px"},
                              config={"displayModeBar": False}),
                ]), className="fin-card"), width=4),
            ], className="g-2"),
        ]),
    ]), className="fin-card mb-4"),

], fluid=True)


# ── Callbacks ──────────────────────────────────────────────────────────────────

# Dark mode
app.clientside_callback(
    "function(dark){document.documentElement.setAttribute('data-bs-theme',dark?'dark':'light');return dark;}",
    Output("dark-store", "data"),
    Input("dark-toggle", "value"),
)

# CSV export
@app.callback(
    Output("csv-dl", "data"),
    Input("export-btn", "n_clicks"),
    State("signals-store", "data"),
    prevent_initial_call=True,
)
def export_csv(_, store_json):
    if not store_json:
        return no_update
    df = pd.read_json(io.StringIO(store_json), orient="split")
    return dcc.send_data_frame(df.to_csv, f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)


# Ticker pill click → update store + pill styles
@app.callback(
    [Output("ticker-store", "data"),
     Output({"type": "ticker-pill", "ticker": ALL}, "className"),
     Output("company-input", "value")],
    [Input({"type": "ticker-pill", "ticker": ALL}, "n_clicks"),
     Input("custom-ticker-dd", "value")],
    prevent_initial_call=True,
)
def select_ticker(pill_clicks, custom_ticker):
    triggered = ctx.triggered_id

    if isinstance(triggered, dict):          # a pill was clicked
        ticker = triggered["ticker"]
    elif custom_ticker:                      # dropdown selection
        ticker = custom_ticker.upper()
    else:
        raise dash.exceptions.PreventUpdate

    classes = [
        "ticker-pill ticker-pill-active" if t == ticker else "ticker-pill"
        for t in QUICK_TICKERS
    ]
    company = _NAME_MAP.get(ticker, ticker)
    return ticker, classes, company


# Filing dropdown population
@app.callback(
    Output("filing-dd", "options"),
    [Input("ticker-store", "data"),
     Input("signals-store", "data")],
)
def populate_filing_options(ticker, _store):
    if not ticker:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filing_date, form_type FROM filings WHERE ticker=? ORDER BY filing_date DESC",
            (ticker.upper(),),
        ).fetchall()
    return [{"label": f"{r['form_type']} — {r['filing_date']}", "value": r["id"]} for r in rows]


# ── Main analysis callback ─────────────────────────────────────────────────────

@app.callback(
    [
        Output("metric-score",      "children"),
        Output("metric-score-sub",  "children"),
        Output("metric-alpha",      "children"),
        Output("metric-alpha-sub",  "children"),
        Output("metric-prec",       "children"),
        Output("metric-prec-sub",   "children"),
        Output("metric-sharpe",     "children"),
        Output("metric-sharpe-sub", "children"),
        Output("metric-n",          "children"),
        Output("metric-n-sub",      "children"),
        Output("heatmap-chart",     "figure"),
        Output("sentiment-chart",   "figure"),
        Output("equity-chart",      "figure"),
        Output("score-decomp",      "figure"),
        Output("decomp-labels",     "children"),
        Output("signal-table",      "children"),
        Output("source-label",      "children"),
        Output("status-msg",        "children"),
        Output("signals-store",     "data"),
    ],
    [Input("run-btn",      "n_clicks"),
     Input("ticker-store", "data")],
    [State("company-input",  "value"),
     State("dark-toggle",    "value"),
     State("long-slider",    "value"),
     State("short-slider",   "value")],
    prevent_initial_call=True,
)
def run_analysis(_n, ticker, company, dark, long_t_state, short_t_state):
    if not ticker:
        raise dash.exceptions.PreventUpdate

    dark    = bool(dark)
    ticker  = ticker.upper()
    long_t  = float(long_t_state  or 0.1)
    short_t = float(short_t_state or -0.1)
    today         = datetime.now().strftime("%Y-%m-%d")
    from_date     = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")
    recent_cutoff = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    inc_from      = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

    try:
        with get_conn() as conn:
            n_filings      = conn.execute("SELECT COUNT(*) FROM filings WHERE ticker=?", (ticker,)).fetchone()[0]
            n_all_articles = conn.execute("SELECT COUNT(*) FROM articles WHERE ticker=?", (ticker,)).fetchone()[0]
            n_new_articles = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE ticker=? AND published_at >= ?",
                (ticker, recent_cutoff)).fetchone()[0]
            n_new_prices = conn.execute(
                "SELECT COUNT(*) FROM prices WHERE ticker=? AND date >= ?",
                (ticker, recent_cutoff)).fetchone()[0]

        if not n_filings:
            edgar_fetch(ticker, max_count=10)
        if n_all_articles == 0:
            news_fetch(ticker, company_name=company or ticker, from_date=from_date, to_date=today)
        if not n_new_articles:
            news_fetch(ticker, company_name=company or ticker, from_date=inc_from, to_date=today)
        if not n_new_prices:
            price_fetch(ticker, start=from_date if n_all_articles == 0 else inc_from, end=today)

        score_filings(ticker)
        score_articles(ticker)

        raw_df   = get_aligned(ticker)
        daily_df = get_daily_sentiment(ticker)

        if raw_df.empty:
            ef = _empty_fig("No aligned data — run again to complete price fetch", dark)
            return ("—",) * 10 + (ef, ef, ef, ef, "", no_update,
                                  f"SEC EDGAR + NewsAPI · {ticker}", "No data.", None, "")

        raw_df   = generate_signals(raw_df,   long_threshold=long_t, short_threshold=short_t)
        daily_df = generate_signals(daily_df, long_threshold=long_t, short_threshold=short_t)

        # VIX regime join
        try:
            _end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            _vix = yf.download("^VIX", start=raw_df["price_date"].min(),
                               end=_end, auto_adjust=True, progress=False)
            if isinstance(_vix.columns, pd.MultiIndex):
                _vix.columns = _vix.columns.get_level_values(0)
            _vix = _vix[["Close"]].rename(columns={"Close": "vix"})
            _vix["_vd"] = _vix.index.strftime("%Y-%m-%d")
            raw_df = raw_df.merge(_vix[["_vd", "vix"]], left_on="price_date",
                                  right_on="_vd", how="left").drop(columns=["_vd"], errors="ignore")
            _q = raw_df["vix"].quantile([0.25, 0.50, 0.75])
            raw_df["vix_regime"] = raw_df["vix"].apply(lambda v:
                f"Q1 — Low (VIX ≤{_q[0.25]:.0f})"  if pd.notna(v) and v <= _q[0.25] else
                f"Q2 — Moderate (VIX ≤{_q[0.50]:.0f})" if pd.notna(v) and v <= _q[0.50] else
                f"Q3 — Elevated (VIX ≤{_q[0.75]:.0f})" if pd.notna(v) and v <= _q[0.75] else
                f"Q4 — High (VIX >{_q[0.75]:.0f})" if pd.notna(v) else "Unknown"
            )
        except Exception:
            raw_df["vix"] = float("nan")
            raw_df["vix_regime"] = "Unknown"

        bt_df   = backtest_run(raw_df)
        metrics = backtest_metrics(bt_df)
        corr    = correlate(raw_df)

        with get_conn() as conn:
            price_df = pd.read_sql_query(
                "SELECT date, close FROM prices WHERE ticker=? ORDER BY date",
                conn, params=(ticker,))

    except req_lib.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response else "?"
        detail = ("NewsAPI rate limit (100 req/day). Try after midnight UTC."
                  if code == 429 else f"HTTP {code} from news API.")
        return _err("Fetch failed", detail, dark)
    except ValueError as exc:
        return _err("Ticker not found", str(exc), dark)
    except Exception as exc:
        return _err("Error", f"{type(exc).__name__}: {exc}", dark)

    # ── 52-week heatmap ───────────────────────────────────────────────────────
    heatmap_fig = _build_heatmap(ticker, dark)

    # ── Sentiment + price chart ───────────────────────────────────────────────
    fig_sent = make_subplots(specs=[[{"secondary_y": True}]])
    bar_colors = [_SIG_COLOR.get(str(s), _NEUT_COLOR) for s in daily_df["signal"]]
    fig_sent.add_trace(go.Bar(x=daily_df["event_date"], y=daily_df["composite_score"],
                              marker_color=bar_colors, name="Score", opacity=0.85),
                       secondary_y=False)
    if not price_df.empty:
        d_min, d_max = daily_df["event_date"].min(), daily_df["event_date"].max()
        ps = price_df[(price_df["date"] >= d_min) & (price_df["date"] <= d_max)]
        fig_sent.add_trace(go.Scatter(x=ps["date"], y=ps["close"], name="Close",
                                      line=dict(color="#0d6efd", width=1.8)),
                           secondary_y=True)
    fig_sent.add_hline(y=long_t,  line_dash="dot", line_color=_POS_COLOR, line_width=0.8)
    fig_sent.add_hline(y=short_t, line_dash="dot", line_color=_NEG_COLOR, line_width=0.8)
    fig_sent.update_layout(**_base_layout(dark), hovermode="x unified", bargap=0.2,
                           legend=dict(orientation="h", y=1.12, x=0))
    fig_sent.update_yaxes(title_text="Score",    gridcolor=_grid(dark), secondary_y=False)
    fig_sent.update_yaxes(title_text="Close ($)", showgrid=False, secondary_y=True)

    # ── Equity curve ──────────────────────────────────────────────────────────
    fig_eq = go.Figure()
    if not bt_df.empty:
        fig_eq.add_trace(go.Scatter(x=bt_df["event_date"], y=bt_df["portfolio_value"],
                                    name="Strategy", line=dict(color="#0d6efd", width=2),
                                    fill="tozeroy", fillcolor="rgba(13,110,253,0.07)"))
        fig_eq.add_trace(go.Scatter(x=bt_df["event_date"], y=bt_df["bnh_value"],
                                    name="Buy & Hold", line=dict(color="#6c757d", width=1.5, dash="dash")))
    fig_eq.update_layout(**_base_layout(dark), hovermode="x unified",
                         legend=dict(orientation="h", y=1.12, x=0),
                         yaxis=dict(title="Portfolio ($)", gridcolor=_grid(dark)))

    # ── Score decomposition ───────────────────────────────────────────────────
    avg_pos = float(raw_df["positive"].mean())
    avg_neg = float(raw_df["negative"].mean())
    avg_neu = float(raw_df["neutral"].mean())

    fig_decomp = go.Figure()
    fig_decomp.add_trace(go.Bar(x=[avg_pos], y=[""], orientation="h",
                                 marker_color=_POS_COLOR, name="Positive",
                                 hovertemplate=f"Positive: {avg_pos:.2f}<extra></extra>"))
    fig_decomp.add_trace(go.Bar(x=[avg_neg], y=[""], orientation="h",
                                 marker_color=_NEG_COLOR, name="Negative",
                                 hovertemplate=f"Negative: {avg_neg:.2f}<extra></extra>"))
    fig_decomp.add_trace(go.Bar(x=[avg_neu], y=[""], orientation="h",
                                 marker_color="#e0e0e0", name="Neutral",
                                 hovertemplate=f"Neutral: {avg_neu:.2f}<extra></extra>"))
    fig_decomp.update_layout(**{**_base_layout(dark), "margin": dict(l=0, r=0, t=0, b=0)},
                              barmode="stack", showlegend=False,
                              xaxis=dict(visible=False, range=[0, 1]),
                              yaxis=dict(visible=False))
    decomp_labels = html.Span([
        html.Span(f"+pos {avg_pos:.2f}", style={"color": _POS_COLOR, "fontWeight": "600",
                                                 "marginRight": "12px"}),
        html.Span(f"−neg {avg_neg:.2f}", style={"color": _NEG_COLOR, "fontWeight": "600",
                                                  "marginRight": "12px"}),
        html.Span(f"∅ neu {avg_neu:.2f}", style={"color": "#6c757d"}),
    ])

    # ── Signal table ──────────────────────────────────────────────────────────
    tbl_bg, tbl_fg = (_bg(dark), "#dee2e6") if dark else ("#fafafa", "#1a1a1a")
    tbl = raw_df[["event_date", "source", "composite_score", "signal", "next_day_return"]].copy()
    tbl = tbl.sort_values("event_date", ascending=False).head(50)
    tbl["composite_score"] = tbl["composite_score"].round(4)
    tbl["next_day_return"]  = (tbl["next_day_return"] * 100).round(2)
    tbl.columns = ["Date", "Source", "Score", "Signal", "Next-Day Ret (%)"]
    signal_tbl = dash_table.DataTable(
        data=tbl.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tbl.columns],
        style_data_conditional=[
            {"if": {"filter_query": '{Signal} = "LONG"'},  "backgroundColor": "rgba(45,106,79,0.10)"},
            {"if": {"filter_query": '{Signal} = "SHORT"'}, "backgroundColor": "rgba(192,57,43,0.10)"},
            {"if": {"column_id": "Next-Day Ret (%)", "filter_query": "{Next-Day Ret (%)} > 0"},
             "color": _POS_COLOR, "fontWeight": "600"},
            {"if": {"column_id": "Next-Day Ret (%)", "filter_query": "{Next-Day Ret (%)} < 0"},
             "color": _NEG_COLOR, "fontWeight": "600"},
        ],
        style_cell={"textAlign": "left", "fontSize": "0.82rem", "padding": "6px 12px",
                    "backgroundColor": tbl_bg, "color": tbl_fg, "border": "0",
                    "fontFamily": "system-ui"},
        style_header={"fontWeight": "600", "backgroundColor": tbl_bg, "color": tbl_fg,
                      "fontSize": "0.78rem", "border": "0",
                      "textTransform": "uppercase", "letterSpacing": "0.04em"},
        page_size=15, sort_action="native", filter_action="native",
    )

    # ── Metric values ─────────────────────────────────────────────────────────
    composite_avg = float(raw_df["composite_score"].mean())
    sentiment_label = html.Span("Bullish" if composite_avg > 0.05 else
                                "Bearish" if composite_avg < -0.05 else "Neutral",
                                className="badge-bullish" if composite_avg > 0.05 else
                                          "badge-bearish" if composite_avg < -0.05 else "")
    score_val = [f"{composite_avg:+.2f}", sentiment_label]

    # Next-day alpha: avg return when score > 0.1
    high_score = raw_df[raw_df["composite_score"] > 0.1]["next_day_return"].dropna()
    alpha_val  = f"{high_score.mean()*100:+.2f}%" if len(high_score) else "—"
    alpha_sub  = f"Avg return when score > 0.1 (n={len(high_score)})"

    # Signal precision (hit rate of active trades)
    active = bt_df[bt_df["position"] != 0]
    if len(active):
        hits = (((active["position"] == 1) & (active["next_day_return"] > 0)) |
                ((active["position"] == -1) & (active["next_day_return"] < 0)))
        prec_val = f"{hits.mean()*100:.1f}%"
    else:
        prec_val = "—"
    prec_sub = f"Rolling {len(active)}-signal hit rate"

    sharpe = metrics.get("sharpe_ratio")
    sharpe_val = f"{sharpe:.2f}" if sharpe is not None else "—"
    sharpe_sub = f"Max DD {metrics.get('max_drawdown', 0)*100:+.1f}%"

    n_sigs = str(len(raw_df))
    corr_r = corr.get("pearson_r", "n/a")
    corr_p = corr.get("pearson_p", "n/a")

    refreshed = "news refreshed" if not n_new_articles else "news up to date"
    source_label = f"SEC EDGAR · NewsAPI · {ticker}"
    status = f"Pearson r={corr_r} (p={corr_p}) · {len(raw_df)} signals · {refreshed} · {today}"

    return (
        score_val,    f"vs last analysis",
        alpha_val,    alpha_sub,
        prec_val,     prec_sub,
        sharpe_val,   sharpe_sub,
        n_sigs,       f"{len(daily_df)} trading days",
        heatmap_fig,
        fig_sent, fig_eq,
        fig_decomp, decomp_labels,
        signal_tbl,
        source_label, status,
        raw_df.to_json(orient="split", date_format="iso"),
    )


# ── 52-week heatmap builder ────────────────────────────────────────────────────

def _build_heatmap(ticker: str, dark: bool) -> go.Figure:
    """GitHub-style 52-week daily sentiment calendar."""
    end_date   = datetime.now().date()
    start_date = end_date - timedelta(weeks=52)

    with get_conn() as conn:
        df = pd.read_sql_query(
            """SELECT event_date, AVG(composite_score) as score
               FROM sentiment_scores WHERE ticker=? AND event_date >= ?
               GROUP BY event_date ORDER BY event_date""",
            conn, params=(ticker.upper(), start_date.strftime("%Y-%m-%d")),
        )
        filing_dates = {r[0] for r in conn.execute(
            "SELECT DISTINCT filing_date FROM filings WHERE ticker=? AND filing_date >= ?",
            (ticker.upper(), start_date.strftime("%Y-%m-%d")),
        ).fetchall()}

    score_map = {} if df.empty else dict(zip(df["event_date"], df["score"]))

    # Align to Monday
    monday = start_date - timedelta(days=start_date.weekday())

    x_vals, y_vals, colors, hover_texts = [], [], [], []
    x_earn, y_earn = [], []

    for week in range(52):
        for dow in range(5):
            d = monday + timedelta(weeks=week, days=dow)
            d_str = d.strftime("%Y-%m-%d")
            score = score_map.get(d_str)
            if score is not None:
                x_vals.append(week)
                y_vals.append(4 - dow)
                colors.append(float(score))
                hover_texts.append(f"{d_str}<br>Score: {score:+.3f}")
            if d_str in filing_dates:
                x_earn.append(week)
                y_earn.append(4 - dow)

    fig = go.Figure()

    if x_vals:
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="markers",
            marker=dict(symbol="square", size=10, color=colors,
                        colorscale=[[0, _NEG_COLOR], [0.5, "#f0f0f0"], [1, _POS_COLOR]],
                        cmin=-0.5, cmax=0.5, showscale=False,
                        line=dict(color=_bg(dark), width=1.5)),
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts, showlegend=False,
        ))

    if x_earn:
        fig.add_trace(go.Scatter(
            x=x_earn, y=y_earn, mode="markers",
            marker=dict(symbol="circle-open", size=14,
                        color="#333", line=dict(width=2)),
            name="Earnings filing", hoverinfo="skip",
        ))

    # Month annotations
    seen_month = None
    for week in range(52):
        d = monday + timedelta(weeks=week)
        if d.month != seen_month:
            seen_month = d.month
            fig.add_annotation(x=week, y=5.2, text=d.strftime("%b"),
                               showarrow=False, font=dict(size=9, color="#6c757d"),
                               xanchor="left")

    fig.update_layout(
        **{**_base_layout(dark), "margin": dict(l=30, r=0, t=20, b=0)},
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.5, 52.5]),
        yaxis=dict(tickvals=[0, 1, 2, 3, 4],
                   ticktext=["Fri", "Thu", "Wed", "Tue", "Mon"],
                   tickfont=dict(size=8, color="#6c757d"),
                   showgrid=False, zeroline=False),
    )
    return fig


# ── Entity bar chart callback ──────────────────────────────────────────────────

@app.callback(
    [Output("entity-bar-chart",   "figure"),
     Output("entity-source-label", "children")],
    [Input("scrubber-store", "data"),
     Input("signals-store",  "data")],
    State("dark-store", "data"),
)
def update_entity_bar(scrubber_json, signals_json, dark):
    from nlp.entity_scorer import score_entities
    dark = bool(dark)

    text, source_label = "", ""

    triggered = ctx.triggered_id
    if triggered == "scrubber-store" and scrubber_json:
        text = " ".join(c["text"] for c in scrubber_json)
        source_label = "source: active filing"

    if not text and signals_json:
        try:
            df = pd.read_json(io.StringIO(signals_json), orient="split")
            tickers = df["ticker"].unique().tolist() if "ticker" in df.columns else []
            if tickers:
                with get_conn() as conn:
                    rows = conn.execute(
                        f"SELECT title, description, content FROM articles WHERE ticker IN "
                        f"({','.join('?'*len(tickers))}) AND relevant=1 ORDER BY published_at DESC LIMIT 50",
                        tickers,
                    ).fetchall()
                text = " ".join(
                    " ".join(filter(None, [r["title"], r["description"], r["content"]]))
                    for r in rows
                )
                source_label = f"source: {len(rows)} news articles"
        except Exception:
            pass

    if not text:
        return _empty_fig("Run Analysis to load entity data", dark), source_label

    try:
        results = score_entities(text, min_mentions=1, max_entities=30)
    except Exception as exc:
        return _empty_fig(f"Error: {exc}", dark), source_label

    if not results:
        return _empty_fig("No entities found", dark), source_label

    top = sorted(results.items(), key=lambda x: abs(x[1]["composite_score"]), reverse=True)[:10]
    top = sorted(top, key=lambda x: x[1]["composite_score"])

    names  = [e[0] for e in top]
    scores = [e[1]["composite_score"] for e in top]
    colors = [_POS_COLOR if s > 0.05 else (_NEG_COLOR if s < -0.05 else _NEUT_COLOR)
              for s in scores]

    fig = go.Figure(go.Bar(
        y=names, x=scores, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{s:+.2f}" for s in scores], textposition="outside",
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=_zero(dark), line_width=1)
    fig.update_layout(
        **{**_base_layout(dark), "margin": dict(l=0, r=40, t=0, b=0)},
        xaxis=dict(range=[-1.1, 1.1], showgrid=True, gridcolor=_grid(dark), zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        showlegend=False,
    )
    return fig, source_label


# ── VIX regime scatter ─────────────────────────────────────────────────────────

@app.callback(
    Output("regime-scatter", "figure"),
    [Input("signals-store", "data"),
     Input("regime-toggle",  "value")],
    State("dark-store", "data"),
)
def update_regime_scatter(store_json, show_regime, dark):
    dark = bool(dark)
    if not store_json:
        return _empty_fig("Run Analysis first", dark)
    df = pd.read_json(io.StringIO(store_json), orient="split")
    df = df.dropna(subset=["composite_score", "next_day_return"])
    if df.empty:
        return _empty_fig("No signals", dark)

    x_all = df["composite_score"].astype(float)
    y_all = df["next_day_return"].astype(float)
    fig = go.Figure()
    has_vix = "vix_regime" in df.columns and df["vix_regime"].notna().any()

    if show_regime and has_vix:
        for qkey in ["Q1", "Q2", "Q3", "Q4"]:
            sub = df[df["vix_regime"].str.startswith(qkey, na=False)]
            if sub.empty:
                continue
            label = sub["vix_regime"].iloc[0]
            color = _REGIME_PALETTE[qkey]
            x = sub["composite_score"].astype(float)
            y = sub["next_day_return"].astype(float)
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=label,
                                     marker=dict(color=color, size=8, opacity=0.75,
                                                 line=dict(color="white", width=0.5)),
                                     hovertemplate=f"<b>{label}</b><br>Score:%{{x:.3f}}<br>Ret:%{{y:.2%}}<extra></extra>"))
            if len(sub) >= 3:
                slope, intercept, r_val, _, _ = scipy_stats.linregress(x, y)
                xl = np.linspace(float(x.min()), float(x.max()), 60)
                fig.add_trace(go.Scatter(x=xl, y=slope*xl+intercept, mode="lines",
                                         showlegend=False, line=dict(color=color, dash="dash", width=1.8),
                                         hovertemplate=f"{qkey} r={r_val:.2f}<extra></extra>"))
        if len(df) >= 3:
            slope, intercept, r_val, _, _ = scipy_stats.linregress(x_all, y_all)
            xl = np.linspace(float(x_all.min()), float(x_all.max()), 80)
            fig.add_trace(go.Scatter(x=xl, y=slope*xl+intercept, mode="lines",
                                     name=f"Overall (r={r_val:.2f})",
                                     line=dict(color=_zero(dark), dash="dot", width=1.2)))
    else:
        for sig in ["LONG", "HOLD", "SHORT"]:
            sub = df[df["signal"] == sig] if "signal" in df.columns else df
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["composite_score"], y=sub["next_day_return"],
                                         mode="markers", name=sig,
                                         marker=dict(color=_SIG_COLOR.get(sig, _NEUT_COLOR),
                                                     size=7, opacity=0.65)))
        if len(df) >= 3:
            slope, intercept, r_val, _, _ = scipy_stats.linregress(x_all, y_all)
            xl = np.linspace(float(x_all.min()), float(x_all.max()), 80)
            fig.add_trace(go.Scatter(x=xl, y=slope*xl+intercept, mode="lines",
                                     name=f"Trend (r={r_val:.2f})",
                                     line=dict(color=_zero(dark), dash="dash", width=1.5)))

    fig.add_hline(y=0, line_dash="dot", line_color=_zero(dark), opacity=0.4)
    fig.add_vline(x=0, line_dash="dot", line_color=_zero(dark), opacity=0.4)
    fig.update_layout(**_base_layout(dark), hovermode="closest",
                      legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
                      xaxis=dict(title="Composite Sentiment Score", gridcolor=_grid(dark)),
                      yaxis=dict(title="Next-Day Return", gridcolor=_grid(dark), tickformat=".1%"))
    return fig


# ── Transcript scrubber ────────────────────────────────────────────────────────

@app.callback(
    [Output("scrubber-chart", "figure"),
     Output("scrubber-store", "data"),
     Output("scrubber-meta",  "children")],
    Input("filing-dd",  "value"),
    State("dark-store", "data"),
    prevent_initial_call=True,
)
def build_scrubber(filing_id, dark):
    from nlp.segmenter import segment, score_chunks as seg_score
    dark = bool(dark)
    if not filing_id:
        return _empty_fig("Select a filing above", dark), None, ""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT ticker, filing_date, form_type, raw_text FROM filings WHERE id=?",
            (filing_id,)).fetchone()
    if not row or not row["raw_text"]:
        return _empty_fig("Filing has no text", dark), None, ""
    chunks = segment(row["raw_text"])
    if not chunks:
        return _empty_fig("Could not segment text", dark), None, ""
    scored = seg_score(chunks)
    x_vals = [c["index"] for c in scored]
    y_vals = [c["composite_score"] for c in scored]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=[max(0, v) for v in y_vals], fill="tozeroy",
                              fillcolor="rgba(45,106,79,0.12)", line=dict(color="rgba(0,0,0,0)"),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x_vals, y=[min(0, v) for v in y_vals], fill="tozeroy",
                              fillcolor="rgba(192,57,43,0.12)", line=dict(color="rgba(0,0,0,0)"),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines",
                              line=dict(color="rgba(120,120,120,0.25)", width=1.2),
                              showlegend=False, hoverinfo="skip"))
    for sec, color in _SEC_COLORS.items():
        sub = [c for c in scored if c["section"] == sec]
        if not sub:
            continue
        fig.add_trace(go.Scatter(
            x=[c["index"] for c in sub], y=[c["composite_score"] for c in sub],
            mode="markers", name=sec.capitalize(),
            marker=dict(color=color, symbol=_SEC_SYMBOLS[sec], size=12,
                        line=dict(color="white", width=1)),
            customdata=[[c["index"], c["section"], c["text"], c["composite_score"]] for c in sub],
            hovertemplate="<b>%{customdata[1]}</b>  %{customdata[3]:+.3f}<br>%{customdata[2]:.100}…<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=_zero(dark), opacity=0.4)
    fig.update_layout(**_base_layout(dark), hovermode="closest", clickmode="event",
                      legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9)),
                      xaxis=dict(title="← Document start · chunk · end →",
                                 gridcolor=_grid(dark), tickmode="linear", dtick=1),
                      yaxis=dict(title="Sentiment", range=[-1.1, 1.1],
                                 gridcolor=_grid(dark), tickformat="+.2f"))
    store = [{"index": c["index"], "section": c["section"], "text": c["text"],
              "positive": c["positive"], "negative": c["negative"],
              "neutral": c["neutral"], "composite_score": c["composite_score"]}
             for c in scored]
    meta = f"{row['form_type']} · {row['filing_date']} · {len(scored)} segments · {row['ticker']}"
    return fig, store, meta


@app.callback(
    Output("scrubber-text", "children"),
    Input("scrubber-chart",  "clickData"),
    State("scrubber-store",  "data"),
    prevent_initial_call=True,
)
def show_segment_text(click_data, store):
    if not click_data or not store:
        raise dash.exceptions.PreventUpdate
    points = click_data.get("points", [])
    if not points or "customdata" not in points[0]:
        raise dash.exceptions.PreventUpdate
    chunk = next((c for c in store if c["index"] == points[0]["customdata"][0]), None)
    if not chunk:
        raise dash.exceptions.PreventUpdate
    score = chunk["composite_score"]
    sec   = chunk["section"]
    color = _SEC_COLORS.get(sec, "#6c757d")
    tone  = (_POS_COLOR, "Positive") if score > 0.05 else ((_NEG_COLOR, "Negative") if score < -0.05 else ("#6c757d", "Neutral"))
    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Badge(sec.upper(), style={"backgroundColor": color, "color": "white",
                                                   "fontSize": "0.68rem"}), width="auto"),
            dbc.Col(dbc.Badge(f"{tone[1]}  {score:+.3f}",
                              style={"backgroundColor": tone[0], "color": "white",
                                     "fontSize": "0.68rem"}), width="auto"),
            dbc.Col(html.Span(f"pos={chunk['positive']:.2f} neg={chunk['negative']:.2f} "
                              f"neu={chunk['neutral']:.2f}",
                              className="text-muted", style={"fontSize": "0.72rem"}),
                    className="ms-2"),
        ], className="mb-2 g-1 align-items-center"),
        html.P(chunk["text"], className="mb-0", style={"fontSize": "0.84rem", "lineHeight": "1.65"}),
    ])


# ── Threshold builder ──────────────────────────────────────────────────────────

@app.callback(
    [Output("threshold-hist",    "figure"),
     Output("threshold-metrics", "children")],
    [Input("long-slider",  "value"),
     Input("short-slider", "value")],
    [State("signals-store", "data"),
     State("dark-store",    "data")],
)
def update_threshold_builder(long_t, short_t, store_json, dark):
    dark  = bool(dark)
    long_t  = float(long_t  or  0.1)
    short_t = float(short_t or -0.1)
    if not store_json:
        return _empty_fig("Run Analysis first", dark), html.P("No data", className="text-muted small")
    df = pd.read_json(io.StringIO(store_json), orient="split")
    df = df.dropna(subset=["composite_score", "next_day_return"])
    if df.empty:
        return _empty_fig("No signals", dark), ""

    scores  = df["composite_score"].astype(float)
    returns = df["next_day_return"].astype(float)
    long_m  = scores >= long_t
    short_m = scores <= short_t
    n_long, n_short, n_hold = int(long_m.sum()), int(short_m.sum()), int((~long_m & ~short_m).sum())
    n_total = len(df)

    def _prec(mask, cond): return float(cond[mask].mean()) if mask.sum() > 0 else None
    def _rec(mask, cond):
        d = cond.sum(); return float((mask & cond).sum() / d) if d > 0 else None
    def _f1(p, r): return 2*p*r/(p+r) if p and r and (p+r) else None

    pos_ret, neg_ret = returns > 0, returns < 0
    lp, lr = _prec(long_m, pos_ret), _rec(long_m, pos_ret)
    sp, sr = _prec(short_m, neg_ret), _rec(short_m, neg_ret)

    bins = np.linspace(-1.0, 1.0, 41)
    mids = (bins[:-1] + bins[1:]) / 2
    hist, _ = np.histogram(scores, bins=bins)
    bar_colors = [_POS_COLOR if m >= long_t else (_NEG_COLOR if m <= short_t else "#d0d0d0")
                  for m in mids]

    fig = go.Figure(go.Bar(x=mids, y=hist, width=(bins[1]-bins[0])*0.9,
                            marker_color=bar_colors, marker_line_width=0,
                            hovertemplate="Score: %{x:.2f}<br>n=%{y}<extra></extra>"))
    fig.add_vline(x=long_t,  line_dash="dash", line_color=_POS_COLOR, line_width=2)
    fig.add_vline(x=short_t, line_dash="dash", line_color=_NEG_COLOR, line_width=2)
    fig.add_vrect(x0=-1.05, x1=short_t, fillcolor=f"rgba(192,57,43,0.07)",  line_width=0)
    fig.add_vrect(x0=long_t, x1=1.05,   fillcolor=f"rgba(45,106,79,0.07)",  line_width=0)
    y_max = max(hist) if hist.max() > 0 else 1
    fig.add_annotation(x=long_t+0.02, y=y_max, text=f"LONG ≥{long_t:.2f}",
                       showarrow=False, font=dict(size=9, color=_POS_COLOR), xanchor="left")
    fig.add_annotation(x=short_t-0.02, y=y_max, text=f"SHORT ≤{short_t:.2f}",
                       showarrow=False, font=dict(size=9, color=_NEG_COLOR), xanchor="right")
    fig.update_layout(**_base_layout(dark), bargap=0.05, showlegend=False,
                      xaxis=dict(title="Composite Score", range=[-1.05, 1.05], gridcolor=_grid(dark)),
                      yaxis=dict(title="# Signals", gridcolor=_grid(dark)))

    def _row(label, n, color, p, r, f1):
        pct = f"{n/n_total*100:.0f}%" if n_total else "—"
        return html.Tr([
            html.Td(html.Span(label, style={"color": color, "fontWeight": "600",
                                             "fontSize": "0.8rem"})),
            html.Td(f"{n} ({pct})", style={"fontSize": "0.8rem"}),
            html.Td(f"{p*100:.0f}%" if p else "—",
                    style={"fontSize": "0.8rem",
                           "color": _POS_COLOR if p and p > 0.5 else _NEG_COLOR}),
            html.Td(f"{r*100:.0f}%" if r else "—", style={"fontSize": "0.8rem"}),
            html.Td(f"{f1*100:.0f}%" if f1 else "—",
                    style={"fontSize": "0.8rem", "fontWeight": "600"}),
        ])

    panel = html.Div([
        html.P(f"{n_total} signals", className="text-muted mb-2",
               style={"fontSize": "0.78rem"}),
        html.Table([
            html.Thead(html.Tr([html.Th(h, style={"fontSize": "0.72rem"})
                                for h in ["Signal", "n", "Prec", "Recall", "F1"]])),
            html.Tbody([
                _row("LONG",  n_long,  _POS_COLOR, lp, lr, _f1(lp, lr)),
                _row("SHORT", n_short, _NEG_COLOR, sp, sr, _f1(sp, sr)),
                _row("HOLD",  n_hold,  _NEUT_COLOR, None, None, None),
            ]),
        ], className="table table-sm table-borderless mb-0"),
        html.Hr(className="my-2"),
        html.P("Precision: hit rate of signals fired. "
               "Recall: fraction of correct moves captured.",
               className="text-muted mb-0", style={"fontSize": "0.7rem"}),
    ])
    return fig, panel


# ── Panel 5 callback — Alpha Decay & PnL Attribution ──────────────────────────

@app.callback(
    Output("ic-decay-fig",    "figure"),
    Output("ic-waterfall-fig","figure"),
    Output("ic-rolling-fig",  "figure"),
    Output("ic-panel-alert",  "children"),
    Output("ic-data-badge",   "children"),
    Input("signals-store",    "data"),
    State("ticker-store",     "data"),
    State("dark-store",       "data"),
)
def update_ic_panel(signals_json, ticker, dark):
    """Render Alpha Decay & PnL Attribution charts when signals-store is updated.

    Triggered automatically after run_analysis() completes for any ticker.
    Reads multi-horizon returns from the prices table; calls
    compute_and_store_horizons() first to ensure the columns are populated.
    Gracefully shows a 'not enough data' message for tickers with < 30 signals.
    """
    dark = bool(dark)
    ef   = _empty_fig("Run Analysis to load data", dark)

    if not signals_json or not ticker:
        return ef, ef, ef, None, "—"

    ticker = ticker.upper()

    try:
        # Ensure multi-horizon return columns are populated for this ticker
        compute_and_store_horizons(ticker)

        df = load_signals_with_horizons(ticker)
    except Exception as exc:
        msg = dbc.Alert(f"IC panel error: {exc}", color="danger",
                        dismissable=True, className="mb-2")
        return ef, ef, ef, msg, "Error"

    # Graceful low-data path
    if df.empty or len(df) < 30:
        n = 0 if df.empty else len(df)
        msg = dbc.Alert(
            [html.Strong(f"{ticker}: "),
             f"Only {n} aligned signals found. At least 30 are needed for reliable "
             "IC estimates. Run Analysis on more dates or choose a different ticker."],
            color="warning", dismissable=True, className="mb-2",
        )
        return ef, ef, ef, msg, f"{n} signals"

    ic_tbl  = compute_ic_series(df)
    roll_ic = compute_rolling_ic(df, horizon=1, window=60)
    attr    = ic_pnl_attribution(df)

    # ── Chart A — IC Decay Curve ───────────────────────────────────────────────
    horizons   = ic_tbl.index.tolist()
    ic_vals    = ic_tbl["ic"].tolist()
    ic_se      = ic_tbl["ic_se"].tolist()
    sig_mask   = ic_tbl["significant"].tolist()
    node_cols  = [_POS_COLOR if s else _NEG_COLOR for s in sig_mask]

    first_insig = next(
        (h for h, s in zip(horizons, sig_mask)
         if not s and not np.isnan(ic_tbl.loc[h, "ic"])),
        None,
    )

    fig_decay = make_subplots(specs=[[{"secondary_y": True}]])

    # 95% CI band
    fig_decay.add_trace(go.Scatter(
        x=horizons + horizons[::-1],
        y=[v + e for v, e in zip(ic_vals, ic_se)] +
          [v - e for v, e in zip(ic_vals[::-1], ic_se[::-1])],
        fill="toself",
        fillcolor="rgba(13,110,253,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1 SE", hoverinfo="skip",
    ), secondary_y=False)

    fig_decay.add_trace(go.Scatter(
        x=horizons, y=ic_vals,
        mode="lines+markers", name="IC (Spearman)",
        line=dict(color="#0d6efd", width=2.5),
        marker=dict(size=9, color=node_cols),
        customdata=np.column_stack([
            ic_tbl["t_stat"].tolist(),
            ic_tbl["p_value"].tolist(),
            ic_tbl["n_trades"].tolist(),
        ]),
        hovertemplate=(
            "<b>%{x}d horizon</b><br>"
            "IC: %{y:.4f}<br>"
            "t-stat: %{customdata[0]:.2f} | p: %{customdata[1]:.4f}<br>"
            "N: %{customdata[2]}<extra></extra>"
        ),
    ), secondary_y=False)

    # Cumulative |IC| on right axis
    cum_ic = list(np.nancumsum(np.abs(ic_vals)))
    fig_decay.add_trace(go.Scatter(
        x=horizons, y=cum_ic,
        mode="lines+markers", name="Cumul. |IC|",
        line=dict(color=_NEUT_COLOR, width=1.5, dash="dot"),
        marker=dict(size=5, symbol="diamond"),
    ), secondary_y=True)

    fig_decay.add_hline(y=0, line_dash="dot", line_color=_NEUT_COLOR, line_width=1)

    if first_insig is not None:
        fig_decay.add_vline(
            x=first_insig, line_dash="dash", line_color=_NEG_COLOR, line_width=1.5,
            annotation_text="signal loses significance here",
            annotation_position="top right",
            annotation_font=dict(size=10, color=_NEG_COLOR),
        )

    fig_decay.update_layout(
        **{**_base_layout(dark), "margin": dict(l=40, r=40, t=30, b=40)},
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig_decay.update_xaxes(title_text="Holding horizon (days)",
                           gridcolor=_grid(dark), tickvals=horizons)
    fig_decay.update_yaxes(title_text="IC", gridcolor=_grid(dark), secondary_y=False)
    fig_decay.update_yaxes(title_text="Cumul. |IC|", showgrid=False, secondary_y=True)

    # ── Chart B — PnL Attribution Waterfall ────────────────────────────────────
    labels  = ["Gross Signal\nAlpha", "Timing\nSlippage", "Execution\nDrag", "Net PnL"]
    y_vals  = [attr["signal_alpha"], attr["timing_slippage"], attr["execution_drag"], 0]
    net_bps = attr["net_pnl"]

    fig_wfall = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=labels,
        y=y_vals,
        textposition="outside",
        text=[f"{v:+.0f}" if v != 0 else f"{net_bps:+.0f}" for v in y_vals],
        connector=dict(line=dict(color="#555", width=1)),
        increasing=dict(marker=dict(color=_POS_COLOR)),
        decreasing=dict(marker=dict(color=_NEG_COLOR)),
        totals=dict(marker=dict(color="#0d6efd")),
    ))
    fig_wfall.update_layout(
        **{**_base_layout(dark), "margin": dict(l=10, r=10, t=30, b=50)},
        showlegend=False,
        yaxis=dict(title="Basis points", gridcolor=_grid(dark)),
    )

    # ── Chart C — Rolling IC ────────────────────────────────────────────────────
    dates = pd.DatetimeIndex(roll_ic.index)
    ic_s  = roll_ic.values.astype(float)

    fig_roll = make_subplots(specs=[[{"secondary_y": True}]])

    # Green/red fill
    pos_ic = np.where(ic_s >= 0, ic_s, 0.0)
    neg_ic = np.where(ic_s <  0, ic_s, 0.0)

    fig_roll.add_trace(go.Scatter(
        x=dates, y=pos_ic, fill="tozeroy",
        fillcolor="rgba(45,106,79,0.20)", line=dict(color="rgba(0,0,0,0)"),
        name="IC > 0", hoverinfo="skip",
    ), secondary_y=False)
    fig_roll.add_trace(go.Scatter(
        x=dates, y=neg_ic, fill="tozeroy",
        fillcolor="rgba(192,57,43,0.20)", line=dict(color="rgba(0,0,0,0)"),
        name="IC < 0 (decay)", hoverinfo="skip",
    ), secondary_y=False)
    fig_roll.add_trace(go.Scatter(
        x=dates, y=ic_s,
        mode="lines", name="Rolling IC",
        line=dict(color="#0d6efd", width=2),
    ), secondary_y=False)

    # Rolling hit rate (right axis)
    col1 = "next_day_return"
    active_df = df[df["signal_type"] != "HOLD"].sort_values("signal_date")
    hr_window  = 60
    hit_rates  = []
    hr_dates   = []
    for i in range(len(active_df)):
        s     = max(0, i - hr_window + 1)
        chunk = active_df.iloc[s : i + 1]
        if len(chunk) >= 5:
            correct = (
                ((chunk["signal_type"] == "LONG")  & (chunk[col1].fillna(0) > 0)) |
                ((chunk["signal_type"] == "SHORT") & (chunk[col1].fillna(0) < 0))
            )
            hit_rates.append(float(correct.mean()) * 100)
            hr_dates.append(chunk["signal_date"].iloc[-1])

    if hit_rates:
        fig_roll.add_trace(go.Scatter(
            x=hr_dates, y=hit_rates,
            mode="lines", name="Hit Rate %",
            line=dict(color=_NEUT_COLOR, width=1.5, dash="dash"),
        ), secondary_y=True)
        fig_roll.add_hline(y=50, line_dash="dot", line_color=_NEUT_COLOR,
                           line_width=0.8, secondary_y=True)

    fig_roll.add_hline(y=0, line_dash="dot", line_color=_NEUT_COLOR, line_width=0.8)
    fig_roll.update_layout(
        **{**_base_layout(dark), "margin": dict(l=40, r=40, t=30, b=40)},
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig_roll.update_xaxes(title_text="Signal date", gridcolor=_grid(dark))
    fig_roll.update_yaxes(title_text="Rolling IC", gridcolor=_grid(dark), secondary_y=False)
    fig_roll.update_yaxes(title_text="Hit rate (%)", showgrid=False, secondary_y=True)

    badge = f"{len(df)} signals · {attr['n_trades']} active"
    return fig_decay, fig_wfall, fig_roll, None, badge


if __name__ == "__main__":
    app.run(debug=True, port=8052)
