"""
Sentiment Trading Signal Dashboard — port 8052.

UI features: dark/light mode toggle, fullscreen loading spinner,
CSV export, friendly error messages.
"""

from __future__ import annotations

import io
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
from dash import dcc, html, Input, Output, State, dash_table, no_update
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
from db import get_conn
from tickers import get_sp500_tickers, ticker_name_map

# ── Constants ──────────────────────────────────────────────────────────────────

_SIG_COLOR = {"LONG": "#28a745", "SHORT": "#dc3545", "HOLD": "#adb5bd"}
TICKERS    = get_sp500_tickers()
_NAME_MAP  = ticker_name_map()

# ── Theme helpers ──────────────────────────────────────────────────────────────

def _chart_style(dark: bool) -> dict:
    return dict(
        paper_bgcolor="#1e2124" if dark else "white",
        plot_bgcolor ="#282b30" if dark else "#f8f9fa",
        font={"color": "#dee2e6" if dark else "#212529"},
    )

def _grid_color(dark: bool) -> str:
    return "#3d4045" if dark else "#e9ecef"

def _zero_color(dark: bool) -> str:
    return "#4a5568" if dark else "#dee2e6"

def _empty_fig(msg: str = "No data", dark: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_chart_style(dark),
        margin=dict(l=0, r=0, t=10, b=0),
        annotations=[{
            "text": msg, "showarrow": False,
            "font": {"size": 13, "color": "#adb5bd" if dark else "#6c757d"},
            "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
        }],
    )
    return fig

def _err(label: str, detail: str, dark: bool):
    """Return the 11 callback outputs for an error state."""
    ef    = _empty_fig(label, dark)
    alert = dbc.Alert(
        [html.Strong(label + ". "), detail],
        color="danger", dismissable=True, className="mb-0",
    )
    return "—", "—", "—", "—", "0", ef, ef, ef, alert, detail, no_update


# ── Layout helpers ─────────────────────────────────────────────────────────────

def _metric_card(label: str, value_id: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.P(label, className="text-muted mb-1",
                   style={"fontSize": "0.72rem", "textTransform": "uppercase",
                          "letterSpacing": "0.06em"}),
            html.H5("—", id=value_id, className="mb-0 fw-bold"),
        ]), className="text-center shadow-sm border-0 h-100"),
        width=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

sidebar = dbc.Card(dbc.CardBody([
    html.H6("Ticker", className="fw-bold mb-1"),
    dcc.Dropdown(
        id="ticker-dd", options=TICKERS, value="AAPL",
        searchable=True, clearable=False, className="mb-3",
    ),

    html.H6("Company Name", className="fw-bold mb-1"),
    dbc.Input(
        id="company-input", value="Apple", type="text",
        placeholder="for NewsAPI query", className="mb-3",
    ),

    html.H6("Data Source", className="fw-bold mb-1"),
    dbc.RadioItems(
        id="source-filter",
        options=[
            {"label": "All Sources", "value": "all"},
            {"label": "EDGAR Only",  "value": "edgar"},
            {"label": "News Only",   "value": "news"},
        ],
        value="all", className="mb-3",
    ),

    html.Hr(),
    html.H6("Signal Thresholds", className="fw-bold mb-1"),
    html.P("Long if score ≥", className="text-muted mb-1", style={"fontSize": "0.8rem"}),
    dbc.Input(
        id="long-thresh", type="number", value=0.1, step=0.05, min=0, max=1, className="mb-2",
    ),
    html.P("Short if score ≤", className="text-muted mb-1", style={"fontSize": "0.8rem"}),
    dbc.Input(
        id="short-thresh", type="number", value=-0.1, step=0.05, min=-1, max=0, className="mb-3",
    ),

    dbc.Button("Run Analysis", id="run-btn", color="primary", className="w-100 mb-2"),
    dbc.Button(
        [html.I(className="me-1"), "Export CSV"],
        id="export-btn", color="secondary", outline=True, className="w-100 mb-3",
    ),
    dcc.Download(id="csv-dl"),

    html.Small(
        "First run fetches 30d of data. "
        "Backtest is event-driven: one paper trade per sentiment signal.",
        className="text-muted",
    ),
    html.Div(id="status-msg", className="text-muted mt-2", style={"fontSize": "0.78rem"}),
]), className="shadow-sm border-0 sidebar-card", style={"top": "72px"})


# ── App ────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
app.title = "Sentiment Trading Signal"
server = app.server

app.layout = dbc.Container([
    # Hidden state stores
    dcc.Store(id="dark-store",    data=False),
    dcc.Store(id="signals-store"),
    dcc.Store(id="scrubber-store"),

    # ── Navbar ────────────────────────────────────────────────────────────────
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Sentiment Trading Signal", className="fw-bold fs-5"),
            html.Span(
                "FinBERT · SEC EDGAR · NewsAPI",
                className="text-muted small ms-2 d-none d-md-inline",
            ),
            dbc.Nav(
                dbc.NavItem(
                    dbc.Switch(
                        id="dark-toggle", value=False, className="mb-0 ms-3",
                        label=html.Span("🌙 Dark", style={"fontSize": "0.85rem"}),
                    )
                ),
                className="ms-auto d-flex align-items-center",
            ),
        ], fluid=True),
        color="light", className="mb-4 shadow-sm border-bottom",
    ),

    # ── Main grid ─────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(sidebar, width=3),

        dbc.Col([
            # Metric cards
            dbc.Row([
                _metric_card("Total Return",  "metric-return"),
                _metric_card("Sharpe Ratio",  "metric-sharpe"),
                _metric_card("Max Drawdown",  "metric-drawdown"),
                _metric_card("Hit Rate",      "metric-hitrate"),
                _metric_card("Signals (n)",   "metric-n"),
            ], className="mb-3 g-2"),

            # Fullscreen spinner wraps all chart outputs
            dcc.Loading(
                type="circle",
                fullscreen=True,
                overlay_style={
                    "visibility": "visible",
                    "opacity": 0.85,
                    "backgroundColor": "rgba(0,0,0,0.45)",
                },
                children=[
                    # Sentiment + price timeline
                    dbc.Card(dbc.CardBody([
                        html.H6("Daily Sentiment Score & Price", className="fw-bold mb-1"),
                        dcc.Graph(
                            id="sentiment-chart", style={"height": "310px"},
                            config={"displayModeBar": False},
                        ),
                    ]), className="shadow-sm border-0 mb-3"),

                    # Equity curve + scatter
                    dbc.Row([
                        dbc.Col(dbc.Card(dbc.CardBody([
                            html.H6("Equity Curve vs Buy & Hold", className="fw-bold mb-1"),
                            dcc.Graph(
                                id="equity-chart", style={"height": "270px"},
                                config={"displayModeBar": False},
                            ),
                        ]), className="shadow-sm border-0"), width=7),

                        dbc.Col(dbc.Card(dbc.CardBody([
                            html.H6("Sentiment vs Next-Day Return", className="fw-bold mb-1"),
                            dcc.Graph(
                                id="scatter-chart", style={"height": "270px"},
                                config={"displayModeBar": False},
                            ),
                        ]), className="shadow-sm border-0"), width=5),
                    ], className="mb-3 g-2"),

                    # Signal table
                    dbc.Card(dbc.CardBody([
                        html.H6("Signal Detail Table — most recent 50", className="fw-bold mb-2"),
                        html.Div(id="signal-table"),
                    ]), className="shadow-sm border-0 mb-3"),

                    # ── Regime Scatter Overlay ────────────────────────────────
                    dbc.Card(dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("Sentiment × Volatility Regime",
                                        className="fw-bold mb-0"),
                                html.P(
                                    "Dots colored by VIX quartile at the time of the signal. "
                                    "Steeper slope in Q4 (red) means sentiment predicts returns "
                                    "more reliably during high-volatility regimes.",
                                    className="text-muted mb-0",
                                    style={"fontSize": "0.82rem"},
                                ),
                            ], width=10),
                            dbc.Col(
                                dbc.Switch(id="regime-toggle", label="VIX overlay",
                                           value=True, className="mb-0 mt-1"),
                                width=2, className="text-end",
                            ),
                        ], className="mb-2 align-items-center"),
                        dcc.Graph(id="regime-scatter", style={"height": "300px"},
                                  config={"displayModeBar": False}),
                    ]), className="shadow-sm border-0 mb-3"),

                    # ── Alert Threshold Builder ───────────────────────────────
                    dbc.Card(dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("Alert Threshold Builder", className="fw-bold mb-0"),
                                html.P(
                                    "Drag thresholds to explore the precision / recall trade-off "
                                    "on historical signals — without re-running the full analysis.",
                                    className="text-muted mb-3",
                                    style={"fontSize": "0.82rem"},
                                ),
                            ]),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("LONG threshold (score ≥)",
                                          style={"fontSize": "0.82rem", "color": "#28a745"}),
                                dcc.Slider(
                                    id="long-slider", min=0, max=1, step=0.01, value=0.1,
                                    marks={0: "0", 0.25: ".25", 0.5: ".50", 0.75: ".75", 1: "1"},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                    className="mb-2",
                                ),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("SHORT threshold (score ≤)",
                                          style={"fontSize": "0.82rem", "color": "#dc3545"}),
                                dcc.Slider(
                                    id="short-slider", min=-1, max=0, step=0.01, value=-0.1,
                                    marks={-1: "-1", -0.75: "-.75", -0.5: "-.50", -0.25: "-.25", 0: "0"},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                    className="mb-2",
                                ),
                            ], width=6),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(
                                dcc.Graph(id="threshold-hist", style={"height": "240px"},
                                          config={"displayModeBar": False}),
                                width=8,
                            ),
                            dbc.Col(html.Div(id="threshold-metrics"), width=4,
                                    className="d-flex align-items-center"),
                        ]),
                    ]), className="shadow-sm border-0 mb-4"),

                    # ── Transcript / Filing Scrubber ──────────────────────────
                    dbc.Card(dbc.CardBody([
                        html.H6("Transcript / Filing Scrubber", className="fw-bold mb-0"),
                        html.P(
                            "Select an 8-K filing to view sentence-level sentiment along the "
                            "document. Sections: financial (◆), management (★), "
                            "guidance (▲), disclaimer (■), overview (●). "
                            "Click any point to read that passage.",
                            className="text-muted mb-3",
                            style={"fontSize": "0.82rem"},
                        ),
                        dbc.Row([
                            dbc.Col(
                                dcc.Dropdown(
                                    id="filing-dd",
                                    placeholder="Select an 8-K filing…",
                                    clearable=False,
                                    className="mb-2",
                                ),
                                width=8,
                            ),
                            dbc.Col(
                                html.Div(id="scrubber-meta",
                                         className="text-muted small mt-1"),
                                width=4,
                            ),
                        ]),
                        dcc.Loading(type="circle", children=[
                            dcc.Graph(
                                id="scrubber-chart", style={"height": "260px"},
                                config={"displayModeBar": False},
                            ),
                        ]),
                        html.Div(
                            id="scrubber-text",
                            className="mt-3 p-3",
                            style={
                                "borderRadius": "8px",
                                "fontSize": "0.87rem",
                                "lineHeight": "1.65",
                                "minHeight": "72px",
                                "border": "1px solid rgba(0,0,0,0.08)",
                            },
                        ),
                    ]), className="shadow-sm border-0 mb-4"),

                    # ── Entity Tag Cloud ──────────────────────────────────────
                    dbc.Card(dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("Entity-Level Sentiment Drill-down",
                                        className="fw-bold mb-0"),
                                html.P(
                                    "Named entities sized by mention frequency, colored by "
                                    "sentiment weight. Hover for the sentence that drove the score. "
                                    "Updates from the active filing (scrubber) or news articles.",
                                    className="text-muted mb-0",
                                    style={"fontSize": "0.82rem"},
                                ),
                            ], width=10),
                            dbc.Col(
                                html.Small(id="entity-source-label",
                                           className="text-muted d-block text-end mt-1"),
                                width=2,
                            ),
                        ], className="mb-3 align-items-start"),
                        dcc.Loading(type="dot", children=[
                            html.Div(id="entity-cloud",
                                     style={"minHeight": "60px", "lineHeight": "2.4"}),
                        ]),
                        html.Hr(className="my-2"),
                        html.Div(id="entity-table"),
                    ]), className="shadow-sm border-0 mb-4"),
                ],
            ),
        ], width=9),
    ]),
], fluid=True)


# ── Callbacks ──────────────────────────────────────────────────────────────────

# Dark mode: toggle Bootstrap 5's native data-bs-theme on the root <html> element.
# This updates all dbc components automatically; Plotly charts are updated via State.
app.clientside_callback(
    """
    function(dark) {
        document.documentElement.setAttribute(
            'data-bs-theme', dark ? 'dark' : 'light'
        );
        return dark;
    }
    """,
    Output("dark-store", "data"),
    Input("dark-toggle", "value"),
)


@app.callback(
    Output("company-input", "value"),
    Input("ticker-dd", "value"),
    prevent_initial_call=True,
)
def autofill_company(ticker: str) -> str:
    return _NAME_MAP.get(ticker, ticker) if ticker else ""


@app.callback(
    Output("csv-dl", "data"),
    Input("export-btn", "n_clicks"),
    State("signals-store", "data"),
    prevent_initial_call=True,
)
def export_csv(_, store_json: str | None):
    if not store_json:
        return no_update
    df    = pd.read_json(io.StringIO(store_json), orient="split")
    fname = f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    return dcc.send_data_frame(df.to_csv, fname, index=False)


@app.callback(
    [
        Output("metric-return",   "children"),
        Output("metric-sharpe",   "children"),
        Output("metric-drawdown", "children"),
        Output("metric-hitrate",  "children"),
        Output("metric-n",        "children"),
        Output("sentiment-chart", "figure"),
        Output("equity-chart",    "figure"),
        Output("scatter-chart",   "figure"),
        Output("signal-table",    "children"),
        Output("status-msg",      "children"),
        Output("signals-store",   "data"),
    ],
    [Input("run-btn", "n_clicks")],
    [
        State("ticker-dd",     "value"),
        State("company-input", "value"),
        State("source-filter", "value"),
        State("long-thresh",   "value"),
        State("short-thresh",  "value"),
        State("dark-store",    "data"),
    ],
    prevent_initial_call=True,
)
def run_analysis(_n_clicks, ticker, company, source, long_thresh, short_thresh, dark):
    if not ticker:
        raise dash.exceptions.PreventUpdate

    dark    = bool(dark)
    ticker  = ticker.upper()
    src     = None if source == "all" else source
    long_t  = float(long_thresh  if long_thresh  is not None else 0.1)
    short_t = float(short_thresh if short_thresh is not None else -0.1)
    today         = datetime.now().strftime("%Y-%m-%d")
    from_date     = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")
    recent_cutoff = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    inc_from      = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

    try:
        # ── Fetch / refresh ───────────────────────────────────────────────────
        with get_conn() as conn:
            n_filings      = conn.execute("SELECT COUNT(*) FROM filings WHERE ticker=?", (ticker,)).fetchone()[0]
            n_all_articles = conn.execute("SELECT COUNT(*) FROM articles WHERE ticker=?", (ticker,)).fetchone()[0]
            n_new_articles = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE ticker=? AND published_at >= ?",
                (ticker, recent_cutoff),
            ).fetchone()[0]
            n_new_prices = conn.execute(
                "SELECT COUNT(*) FROM prices WHERE ticker=? AND date >= ?",
                (ticker, recent_cutoff),
            ).fetchone()[0]

        # EDGAR: once per ticker (slow due to SEC rate limits)
        if not n_filings:
            edgar_fetch(ticker, max_count=10)

        # News: full 30d on first run; 3d incremental on every subsequent run
        if n_all_articles == 0:
            news_fetch(ticker, company_name=company or ticker, from_date=from_date, to_date=today)
        if not n_new_articles:
            news_fetch(ticker, company_name=company or ticker, from_date=inc_from, to_date=today)

        # Prices: same incremental pattern
        if not n_new_prices:
            start = from_date if n_all_articles == 0 else inc_from
            price_fetch(ticker, start=start, end=today)

        # ── Score ─────────────────────────────────────────────────────────────
        score_filings(ticker)
        score_articles(ticker)

        # ── Align + signal ────────────────────────────────────────────────────
        raw_df   = get_aligned(ticker, src)
        daily_df = get_daily_sentiment(ticker, src)

        if raw_df.empty:
            return _err(
                "No aligned data",
                f"No news/filings for {ticker} overlap with price history in the DB. "
                "Click Run Analysis once more — prices will be fetched on this request.",
                dark,
            )

        raw_df   = generate_signals(raw_df,   long_threshold=long_t, short_threshold=short_t)
        daily_df = generate_signals(daily_df, long_threshold=long_t, short_threshold=short_t)

        # ── VIX regime classification ─────────────────────────────────────────
        # Fetch VIX for the signal date window and join so the regime scatter
        # callback can read it directly from signals-store without re-fetching.
        try:
            _end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            _vix = yf.download("^VIX", start=raw_df["price_date"].min(),
                               end=_end, auto_adjust=True, progress=False)
            if isinstance(_vix.columns, pd.MultiIndex):
                _vix.columns = _vix.columns.get_level_values(0)
            _vix = _vix[["Close"]].rename(columns={"Close": "vix"})
            _vix["_vdate"] = _vix.index.strftime("%Y-%m-%d")
            raw_df = raw_df.merge(_vix[["_vdate", "vix"]],
                                  left_on="price_date", right_on="_vdate",
                                  how="left").drop(columns=["_vdate"], errors="ignore")
            _q = raw_df["vix"].quantile([0.25, 0.50, 0.75])
            def _regime(v):
                if pd.isna(v): return "Unknown"
                if v <= _q[0.25]: return f"Q1 — Low (VIX ≤{_q[0.25]:.0f})"
                if v <= _q[0.50]: return f"Q2 — Moderate (VIX ≤{_q[0.50]:.0f})"
                if v <= _q[0.75]: return f"Q3 — Elevated (VIX ≤{_q[0.75]:.0f})"
                return f"Q4 — High (VIX >{_q[0.75]:.0f})"
            raw_df["vix_regime"] = raw_df["vix"].apply(_regime)
        except Exception:
            raw_df["vix"]        = float("nan")
            raw_df["vix_regime"] = "Unknown"

        # ── Backtest + correlation ────────────────────────────────────────────
        bt_df   = backtest_run(raw_df)
        metrics = backtest_metrics(bt_df)
        corr    = correlate(raw_df)

        with get_conn() as conn:
            price_df = pd.read_sql_query(
                "SELECT date, close FROM prices WHERE ticker=? ORDER BY date",
                conn, params=(ticker,),
            )

    # ── Friendly error messages ───────────────────────────────────────────────
    except req_lib.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response else "?"
        if code == 429:
            detail = (
                "NewsAPI free tier allows 100 requests/day. "
                "Limit reached — try again after midnight UTC, or upgrade at newsapi.org."
            )
        elif code == 401:
            detail = "NewsAPI key rejected. Open .env and check that NEWSAPI_KEY is set correctly."
        else:
            detail = f"HTTP {code} error from news API. Check your network connection."
        return _err("News fetch failed", detail, dark)

    except ValueError as exc:
        raw = str(exc)
        detail = (
            f"'{ticker}' was not found in the SEC EDGAR company list. "
            "Verify it is a valid US-listed ticker symbol."
            if "not found" in raw else raw
        )
        return _err("Ticker not found", detail, dark)

    except Exception as exc:
        return _err("Unexpected error", f"{type(exc).__name__}: {exc}", dark)

    # ── Build figures ─────────────────────────────────────────────────────────
    cs   = _chart_style(dark)
    grid = _grid_color(dark)
    zero = _zero_color(dark)
    layout_base = dict(**cs, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                       legend=dict(orientation="h", y=1.12, x=0))

    # Sentiment + price (dual axis)
    fig_sent = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sent.add_trace(go.Bar(
        x=daily_df["event_date"],
        y=daily_df["composite_score"],
        marker_color=[_SIG_COLOR.get(str(s), "#adb5bd") for s in daily_df["signal"]],
        name="Avg Score", opacity=0.82,
    ), secondary_y=False)
    if not price_df.empty:
        d_min, d_max = daily_df["event_date"].min(), daily_df["event_date"].max()
        ps = price_df[(price_df["date"] >= d_min) & (price_df["date"] <= d_max)]
        fig_sent.add_trace(go.Scatter(
            x=ps["date"], y=ps["close"],
            name="Close", line=dict(color="#0d6efd", width=1.8),
        ), secondary_y=True)
    fig_sent.add_hline(y=0,       line_dash="dash", line_color=zero,      line_width=0.8)
    fig_sent.add_hline(y=long_t,  line_dash="dot",  line_color="#28a745", line_width=0.8)
    fig_sent.add_hline(y=short_t, line_dash="dot",  line_color="#dc3545", line_width=0.8)
    fig_sent.update_layout(**layout_base, bargap=0.2)
    fig_sent.update_yaxes(title_text="Score",    secondary_y=False, gridcolor=grid)
    fig_sent.update_yaxes(title_text="Close ($)", secondary_y=True,  showgrid=False)

    # Equity curve
    fig_eq = go.Figure()
    if not bt_df.empty:
        fig_eq.add_trace(go.Scatter(
            x=bt_df["event_date"], y=bt_df["portfolio_value"],
            name="Strategy", line=dict(color="#0d6efd", width=2),
            fill="tozeroy", fillcolor="rgba(13,110,253,0.07)",
        ))
        fig_eq.add_trace(go.Scatter(
            x=bt_df["event_date"], y=bt_df["bnh_value"],
            name="Buy & Hold", line=dict(color="#6c757d", width=1.5, dash="dash"),
        ))
    fig_eq.update_layout(**layout_base, yaxis=dict(title="Portfolio ($)", gridcolor=grid))

    # Correlation scatter
    fig_sc = go.Figure()
    clean  = raw_df.dropna(subset=["composite_score", "next_day_return"])
    for sig in ["LONG", "HOLD", "SHORT"]:
        sub = clean[clean["signal"] == sig]
        if not sub.empty:
            fig_sc.add_trace(go.Scatter(
                x=sub["composite_score"], y=sub["next_day_return"],
                mode="markers", name=sig,
                marker=dict(color=_SIG_COLOR[sig], size=7, opacity=0.65),
            ))
    if len(clean) >= 3:
        slope, intercept, r_val, _, _ = scipy_stats.linregress(
            clean["composite_score"].astype(float),
            clean["next_day_return"].astype(float),
        )
        x_lin = np.linspace(float(clean["composite_score"].min()),
                            float(clean["composite_score"].max()), 80)
        fig_sc.add_trace(go.Scatter(
            x=x_lin, y=slope * x_lin + intercept, mode="lines",
            name=f"Trend (r={r_val:.2f})",
            line=dict(color="#dee2e6" if dark else "#212529", dash="dash", width=1.5),
        ))
    fig_sc.add_hline(y=0, line_dash="dot", line_color=zero, opacity=0.4)
    fig_sc.add_vline(x=0, line_dash="dot", line_color=zero, opacity=0.4)
    fig_sc.update_layout(
        **cs, margin=dict(l=0, r=0, t=10, b=0), hovermode="closest",
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(title="Score", gridcolor=grid),
        yaxis=dict(title="Next-Day Return", gridcolor=grid),
    )

    # Signal table
    tbl_bg   = "#2c3034" if dark else "#f8f9fa"
    tbl_text = "#dee2e6" if dark else "#212529"
    tbl = raw_df[["event_date", "source", "composite_score", "signal", "next_day_return"]].copy()
    tbl = tbl.sort_values("event_date", ascending=False).head(50)
    tbl["composite_score"] = tbl["composite_score"].round(4)
    tbl["next_day_return"] = (tbl["next_day_return"] * 100).round(2)
    tbl.columns = ["Date", "Source", "Score", "Signal", "Next-Day Ret (%)"]

    signal_tbl = dash_table.DataTable(
        data=tbl.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tbl.columns],
        style_data_conditional=[
            {"if": {"filter_query": '{Signal} = "LONG"'},
             "backgroundColor": "rgba(40,167,69,0.13)"},
            {"if": {"filter_query": '{Signal} = "SHORT"'},
             "backgroundColor": "rgba(220,53,69,0.13)"},
            {"if": {"column_id": "Next-Day Ret (%)", "filter_query": "{Next-Day Ret (%)} > 0"},
             "color": "#28a745", "fontWeight": "600"},
            {"if": {"column_id": "Next-Day Ret (%)", "filter_query": "{Next-Day Ret (%)} < 0"},
             "color": "#dc3545", "fontWeight": "600"},
        ],
        style_cell={
            "textAlign": "left", "fontSize": "0.83rem", "padding": "6px 12px",
            "backgroundColor": tbl_bg, "color": tbl_text, "border": "0",
            "fontFamily": "system-ui, sans-serif",
        },
        style_header={
            "fontWeight": "bold", "backgroundColor": tbl_bg, "color": tbl_text,
            "fontSize": "0.83rem", "border": "0",
        },
        page_size=15, sort_action="native", filter_action="native",
    )

    # Metric values
    def _pct(v): return f"{v*100:+.1f}%" if v is not None else "—"
    def _f2(v):  return f"{v:.2f}"       if v is not None else "—"
    def _p0(v):  return f"{v*100:.1f}%"  if v is not None else "—"

    refreshed = "news refreshed" if not n_new_articles else "up to date"
    pr = corr.get("pearson_r", "n/a")
    pp = corr.get("pearson_p", "n/a")
    status = (
        f"r={pr} (p={pp}) · {len(raw_df)} signals · "
        f"{len(daily_df)} days · news {refreshed} · {today}"
    )

    return (
        _pct(metrics.get("total_return")),
        _f2(metrics.get("sharpe_ratio")),
        _pct(metrics.get("max_drawdown")),
        _p0(metrics.get("hit_rate")),
        str(len(raw_df)),
        fig_sent, fig_eq, fig_sc, signal_tbl, status,
        raw_df.to_json(orient="split", date_format="iso"),
    )


# ── Regime Scatter callback ────────────────────────────────────────────────────
# Fires whenever new signal data lands (signals-store) OR the overlay toggle changes.
# VIX + regime columns are pre-joined in run_analysis so no extra fetch needed here.

_REGIME_PALETTE = {
    "Q1": "#4a90d9",   # cool blue   — low vol
    "Q2": "#27ae60",   # green       — moderate
    "Q3": "#f39c12",   # amber       — elevated
    "Q4": "#e74c3c",   # red         — high vol
    "Unknown": "#adb5bd",
}


@app.callback(
    Output("regime-scatter", "figure"),
    [Input("signals-store",  "data"),
     Input("regime-toggle",  "value")],
    State("dark-store", "data"),
)
def update_regime_scatter(store_json, show_regime, dark):
    dark = bool(dark)
    cs   = _chart_style(dark)
    grid = _grid_color(dark)
    zero = _zero_color(dark)

    if not store_json:
        return _empty_fig("Run Analysis first to load signal data", dark)

    df = pd.read_json(io.StringIO(store_json), orient="split")
    df = df.dropna(subset=["composite_score", "next_day_return"])
    if df.empty:
        return _empty_fig("No aligned signals", dark)

    fig = go.Figure()
    x_all = df["composite_score"].astype(float)
    y_all = df["next_day_return"].astype(float)

    has_vix = "vix_regime" in df.columns and df["vix_regime"].notna().any()

    if show_regime and has_vix:
        # ── Regime mode: one trace + trend line per VIX quartile ─────────────
        for qkey in ["Q1", "Q2", "Q3", "Q4"]:
            sub = df[df["vix_regime"].str.startswith(qkey, na=False)]
            if sub.empty:
                continue
            label = sub["vix_regime"].iloc[0]
            color = _REGIME_PALETTE[qkey]
            x = sub["composite_score"].astype(float)
            y = sub["next_day_return"].astype(float)

            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers",
                name=label,
                marker=dict(color=color, size=8, opacity=0.72,
                            line=dict(color="white", width=0.5)),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Score: %{x:.3f}<br>Return: %{y:.2%}<extra></extra>"
                ),
            ))

            if len(sub) >= 3:
                slope, intercept, r_val, _, _ = scipy_stats.linregress(x, y)
                x_lo, x_hi = float(x.min()), float(x.max())
                x_line = np.linspace(x_lo, x_hi, 60)
                fig.add_trace(go.Scatter(
                    x=x_line, y=slope * x_line + intercept,
                    mode="lines", showlegend=False,
                    line=dict(color=color, dash="dash", width=1.8),
                    hovertemplate=f"{qkey} trend r={r_val:.2f}<extra></extra>",
                ))

        # Overall trend (thin, dark)
        if len(df) >= 3:
            slope, intercept, r_val, _, _ = scipy_stats.linregress(x_all, y_all)
            x_line = np.linspace(float(x_all.min()), float(x_all.max()), 80)
            fig.add_trace(go.Scatter(
                x=x_line, y=slope * x_line + intercept,
                mode="lines", name=f"Overall (r={r_val:.2f})",
                line=dict(color="#dee2e6" if dark else "#212529",
                          dash="dot", width=1.2),
            ))

    else:
        # ── Signal mode: standard LONG/HOLD/SHORT coloring ────────────────────
        for sig in ["LONG", "HOLD", "SHORT"]:
            sub = df[df["signal"] == sig] if "signal" in df.columns else df
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["composite_score"], y=sub["next_day_return"],
                mode="markers", name=sig,
                marker=dict(color=_SIG_COLOR.get(sig, "#adb5bd"), size=7, opacity=0.65),
            ))

        if len(df) >= 3:
            slope, intercept, r_val, _, _ = scipy_stats.linregress(x_all, y_all)
            x_line = np.linspace(float(x_all.min()), float(x_all.max()), 80)
            fig.add_trace(go.Scatter(
                x=x_line, y=slope * x_line + intercept,
                mode="lines", name=f"Trend (r={r_val:.2f})",
                line=dict(color="#dee2e6" if dark else "#212529",
                          dash="dash", width=1.5),
            ))

    fig.add_hline(y=0, line_dash="dot", line_color=zero, opacity=0.4)
    fig.add_vline(x=0, line_dash="dot", line_color=zero, opacity=0.4)
    fig.update_layout(
        **cs,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="closest",
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
        xaxis=dict(title="Composite Sentiment Score", gridcolor=grid),
        yaxis=dict(title="Next-Day Return",           gridcolor=grid,
                   tickformat=".1%"),
    )
    return fig


# ── Alert Threshold Builder callback ──────────────────────────────────────────

@app.callback(
    [Output("threshold-hist",     "figure"),
     Output("threshold-metrics",  "children")],
    [Input("long-slider",  "value"),
     Input("short-slider", "value")],
    [State("signals-store", "data"),
     State("dark-store",    "data")],
)
def update_threshold_builder(long_t, short_t, store_json, dark):
    dark = bool(dark)
    cs   = _chart_style(dark)
    grid = _grid_color(dark)

    if not store_json:
        return _empty_fig("Run Analysis first to load signal data", dark), html.P(
            "No data — click Run Analysis.", className="text-muted small"
        )

    df = pd.read_json(io.StringIO(store_json), orient="split")
    df = df.dropna(subset=["composite_score", "next_day_return"])

    if df.empty:
        return _empty_fig("No aligned signals", dark), html.P("No data.", className="text-muted small")

    long_t  = float(long_t  or  0.1)
    short_t = float(short_t or -0.1)

    scores = df["composite_score"].astype(float)
    returns = df["next_day_return"].astype(float)

    long_mask  = scores >= long_t
    short_mask = scores <= short_t
    hold_mask  = ~long_mask & ~short_mask

    n_long  = int(long_mask.sum())
    n_short = int(short_mask.sum())
    n_hold  = int(hold_mask.sum())
    n_total = len(df)

    def _precision(mask, positive_condition):
        return float((positive_condition[mask]).mean()) if mask.sum() > 0 else None

    def _recall(mask, positive_condition):
        denom = positive_condition.sum()
        return float((mask & positive_condition).sum() / denom) if denom > 0 else None

    pos_ret = returns > 0
    neg_ret = returns < 0

    lp = _precision(long_mask,  pos_ret)
    lr = _recall(long_mask,     pos_ret)
    sp = _precision(short_mask, neg_ret)
    sr = _recall(short_mask,    neg_ret)

    def _f1(p, r):
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    lf1 = _f1(lp, lr)
    sf1 = _f1(sp, sr)

    # ── Histogram figure ──────────────────────────────────────────────────────
    bins  = np.linspace(-1.0, 1.0, 41)
    width = bins[1] - bins[0]
    mids  = (bins[:-1] + bins[1:]) / 2
    hist, _ = np.histogram(scores, bins=bins)

    bar_colors = []
    for m in mids:
        if m >= long_t:
            bar_colors.append("#28a745")
        elif m <= short_t:
            bar_colors.append("#dc3545")
        else:
            bar_colors.append("#adb5bd")

    fig = go.Figure(go.Bar(
        x=mids, y=hist,
        width=width * 0.9,
        marker_color=bar_colors,
        marker_line_width=0,
        hovertemplate="Score: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))

    # Threshold lines
    fig.add_vline(x=long_t,  line_dash="dash", line_color="#28a745", line_width=2)
    fig.add_vline(x=short_t, line_dash="dash", line_color="#dc3545", line_width=2)

    # Zone shading
    fig.add_vrect(x0=-1.05, x1=short_t, fillcolor="rgba(220,53,69,0.08)",  line_width=0)
    fig.add_vrect(x0=long_t, x1=1.05,   fillcolor="rgba(40,167,69,0.08)",  line_width=0)

    # Annotations
    y_max = max(hist) if hist.max() > 0 else 1
    fig.add_annotation(x=long_t  + 0.02, y=y_max, text=f"LONG ≥{long_t:.2f}",
                       showarrow=False, font=dict(size=10, color="#28a745"), xanchor="left")
    fig.add_annotation(x=short_t - 0.02, y=y_max, text=f"SHORT ≤{short_t:.2f}",
                       showarrow=False, font=dict(size=10, color="#dc3545"), xanchor="right")

    fig.update_layout(
        **cs,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        bargap=0.05,
        xaxis=dict(title="Composite Score", range=[-1.05, 1.05], gridcolor=grid),
        yaxis=dict(title="# Signals",       gridcolor=grid),
    )

    # ── Precision / Recall metrics panel ─────────────────────────────────────
    def _signal_row(label, n, color, prec, rec, f1):
        pct = f"{n/n_total*100:.0f}%" if n_total else "—"
        return html.Tr([
            html.Td(html.Span(label, style={"color": color, "fontWeight": "600",
                                             "fontSize": "0.82rem"})),
            html.Td(f"{n} ({pct})", style={"fontSize": "0.82rem"}),
            html.Td(f"{prec*100:.0f}%" if prec is not None else "—",
                    style={"fontSize": "0.82rem", "color": "#28a745" if prec and prec > 0.5 else "#dc3545"}),
            html.Td(f"{rec*100:.0f}%"  if rec  is not None else "—",
                    style={"fontSize": "0.82rem"}),
            html.Td(f"{f1*100:.0f}%"   if f1   is not None else "—",
                    style={"fontSize": "0.82rem", "fontWeight": "600"}),
        ])

    metrics_panel = html.Div([
        html.P(f"{n_total} historical signals",
               className="text-muted mb-2", style={"fontSize": "0.8rem"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Signal", style={"fontSize": "0.75rem"}),
                html.Th("n",      style={"fontSize": "0.75rem"}),
                html.Th("Prec",   style={"fontSize": "0.75rem"}),
                html.Th("Recall", style={"fontSize": "0.75rem"}),
                html.Th("F1",     style={"fontSize": "0.75rem"}),
            ])),
            html.Tbody([
                _signal_row("LONG",  n_long,  "#28a745", lp, lr, lf1),
                _signal_row("SHORT", n_short, "#dc3545", sp, sr, sf1),
                _signal_row("HOLD",  n_hold,  "#adb5bd", None, None, None),
            ]),
        ], className="table table-sm table-borderless mb-0"),
        html.Hr(className="my-2"),
        html.P(
            "Precision: hit rate of fired signals. "
            "Recall: fraction of correct moves captured.",
            className="text-muted mb-0",
            style={"fontSize": "0.72rem"},
        ),
    ])

    return fig, metrics_panel


# ── Transcript Scrubber callbacks ─────────────────────────────────────────────

_SEC_COLORS = {
    "financial":  "#0d6efd",
    "management": "#6f42c1",
    "guidance":   "#fd7e14",
    "disclaimer": "#adb5bd",
    "overview":   "#6c757d",
}
_SEC_SYMBOLS = {
    "financial":  "diamond",
    "management": "star",
    "guidance":   "triangle-up",
    "disclaimer": "square",
    "overview":   "circle",
}


@app.callback(
    Output("filing-dd", "options"),
    [Input("ticker-dd",     "value"),
     Input("signals-store", "data")],   # refresh after Run Analysis fetches new filings
)
def populate_filing_options(ticker, _store):
    if not ticker:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, filing_date, form_type
               FROM filings WHERE ticker=?
               ORDER BY filing_date DESC""",
            (ticker.upper(),),
        ).fetchall()
    return [
        {"label": f"{r['form_type']} — {r['filing_date']}", "value": r["id"]}
        for r in rows
    ]


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
    cs   = _chart_style(dark)
    grid = _grid_color(dark)
    zero = _zero_color(dark)

    if not filing_id:
        return _empty_fig("Select a filing above", dark), None, ""

    with get_conn() as conn:
        row = conn.execute(
            "SELECT ticker, filing_date, form_type, raw_text FROM filings WHERE id=?",
            (filing_id,),
        ).fetchone()

    if not row or not row["raw_text"]:
        return _empty_fig("Filing has no text content", dark), None, ""

    chunks = segment(row["raw_text"])
    if not chunks:
        return _empty_fig("Could not extract text segments", dark), None, ""

    scored = seg_score(chunks)   # batched FinBERT — ~1–2 s on MPS

    x_vals = [c["index"] for c in scored]
    y_vals = [c["composite_score"] for c in scored]

    fig = go.Figure()

    # Area fills
    fig.add_trace(go.Scatter(
        x=x_vals, y=[max(0.0, v) for v in y_vals],
        fill="tozeroy", fillcolor="rgba(40,167,69,0.12)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=[min(0.0, v) for v in y_vals],
        fill="tozeroy", fillcolor="rgba(220,53,69,0.12)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False, hoverinfo="skip",
    ))
    # Connecting spine
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="lines",
        line=dict(color="rgba(120,120,120,0.25)", width=1.2),
        showlegend=False, hoverinfo="skip",
    ))
    # Points per section type
    for sec, color in _SEC_COLORS.items():
        sub = [c for c in scored if c["section"] == sec]
        if not sub:
            continue
        fig.add_trace(go.Scatter(
            x=[c["index"] for c in sub],
            y=[c["composite_score"] for c in sub],
            mode="markers",
            name=sec.capitalize(),
            marker=dict(color=color, symbol=_SEC_SYMBOLS[sec],
                        size=13, line=dict(color="white", width=1)),
            customdata=[[c["index"], c["section"],
                         c["text"], c["composite_score"]] for c in sub],
            hovertemplate=(
                "<b>%{customdata[1]}</b>  score=%{customdata[3]:+.3f}<br>"
                "%{customdata[2]:.120}…<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color=zero, opacity=0.4)
    fig.update_layout(
        **cs,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="closest",
        clickmode="event",
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
        xaxis=dict(title="← Document start  ·  chunk index  ·  end →",
                   gridcolor=grid, tickmode="linear", dtick=1),
        yaxis=dict(title="Sentiment", range=[-1.1, 1.1],
                   gridcolor=grid, tickformat="+.2f"),
    )

    store = [
        {"index": c["index"], "section": c["section"], "text": c["text"],
         "positive": c["positive"], "negative": c["negative"],
         "neutral": c["neutral"], "composite_score": c["composite_score"]}
        for c in scored
    ]
    meta = (f"{row['form_type']} · {row['filing_date']} · "
            f"{len(scored)} segments · {row['ticker']}")

    return fig, store, meta


@app.callback(
    Output("scrubber-text", "children"),
    Input("scrubber-chart", "clickData"),
    State("scrubber-store", "data"),
    prevent_initial_call=True,
)
def show_segment_text(click_data, store):
    if not click_data or not store:
        raise dash.exceptions.PreventUpdate

    # customdata[0] is the chunk index stored in every point trace
    points = click_data.get("points", [])
    if not points or "customdata" not in points[0]:
        raise dash.exceptions.PreventUpdate

    chunk_index = points[0]["customdata"][0]
    chunk = next((c for c in store if c["index"] == chunk_index), None)
    if not chunk:
        raise dash.exceptions.PreventUpdate

    sec   = chunk["section"]
    score = chunk["composite_score"]
    color = _SEC_COLORS.get(sec, "#6c757d")

    if score > 0.05:
        tone_color, tone_label = "#28a745", "Positive"
    elif score < -0.05:
        tone_color, tone_label = "#dc3545", "Negative"
    else:
        tone_color, tone_label = "#6c757d", "Neutral"

    return html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Badge(sec.upper(), color="light",
                          style={"backgroundColor": color, "color": "white",
                                 "fontSize": "0.72rem"}),
                width="auto",
            ),
            dbc.Col(
                dbc.Badge(
                    f"{tone_label}  {score:+.3f}",
                    color="light",
                    style={"backgroundColor": tone_color, "color": "white",
                           "fontSize": "0.72rem"},
                ),
                width="auto",
            ),
            dbc.Col(
                html.Span(
                    f"▪ pos={chunk['positive']:.2f}  "
                    f"neg={chunk['negative']:.2f}  "
                    f"neu={chunk['neutral']:.2f}",
                    className="text-muted",
                    style={"fontSize": "0.75rem"},
                ),
                className="ms-2",
            ),
        ], className="mb-2 align-items-center g-1"),
        html.P(chunk["text"], className="mb-0",
               style={"fontSize": "0.87rem", "lineHeight": "1.65"}),
    ])


# ── Entity Tag Cloud callback ──────────────────────────────────────────────────

@app.callback(
    [Output("entity-cloud",        "children"),
     Output("entity-table",        "children"),
     Output("entity-source-label", "children")],
    [Input("scrubber-store", "data"),
     Input("signals-store",  "data")],
    State("dark-store", "data"),
)
def update_entity_cloud(scrubber_json, signals_json, dark):
    from nlp.entity_scorer import score_entities
    from dash import ctx

    dark = bool(dark)

    # Prefer whichever store just fired
    triggered = ctx.triggered_id if ctx.triggered_id else "signals-store"

    text = ""
    source_label = ""

    if triggered == "scrubber-store" and scrubber_json:
        # Build text from all scored chunks in the current filing
        chunks = scrubber_json
        text   = " ".join(c["text"] for c in chunks)
        source_label = "source: active filing"

    if not text and signals_json:
        # Reconstruct text from news article signals in signals-store
        try:
            df = pd.read_json(io.StringIO(signals_json), orient="split")
            # signals-store holds composite scores, not raw text —
            # pull raw article text from DB for the ticker
            tickers = df["ticker"].unique().tolist() if "ticker" in df.columns else []
            if tickers:
                with get_conn() as conn:
                    rows = conn.execute(
                        f"""SELECT title, description, content FROM articles
                            WHERE ticker IN ({','.join('?' * len(tickers))})
                              AND relevant = 1
                            ORDER BY published_at DESC LIMIT 50""",
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
        return (
            html.P("Run Analysis or select a filing to load entity data.",
                   className="text-muted small"),
            "", source_label,
        )

    try:
        results = score_entities(text, min_mentions=1, max_entities=35)
    except Exception as exc:
        return (
            html.P(f"Entity scoring failed: {exc}", className="text-danger small"),
            "", source_label,
        )

    if not results:
        return (
            html.P("No entities found in this text.", className="text-muted small"),
            "", source_label,
        )

    # ── Build tag cloud ───────────────────────────────────────────────────────
    sorted_ents = sorted(
        results.items(),
        key=lambda x: abs(x[1]["composite_score"]),
        reverse=True,
    )[:30]

    max_mentions = max(v["mentions"] for _, v in sorted_ents) or 1

    tags = []
    for entity, data in sorted_ents:
        score    = data["composite_score"]
        mentions = data["mentions"]
        example  = data["example"][:180]

        # Color by sentiment (intensity scales with score magnitude)
        alpha = 0.25 + min(abs(score), 1.0) * 0.65
        if score > 0.05:
            bg   = f"rgba(40,167,69,{alpha:.2f})"
            fg   = "#0a3d1a" if not dark else "#a3ffc1"
        elif score < -0.05:
            bg   = f"rgba(220,53,69,{alpha:.2f})"
            fg   = "#4a0010" if not dark else "#ffb3bc"
        else:
            bg   = "rgba(108,117,125,0.18)"
            fg   = "#343a40" if not dark else "#dee2e6"

        # Font size scales with mention count (0.78 → 1.30 rem)
        fz = 0.78 + (mentions / max_mentions) * 0.52

        tags.append(html.Span(
            f"{entity}  {score:+.2f}",
            title=f"[{data['type']}] n={mentions} — '{example}'",
            style={
                "backgroundColor": bg,
                "color": fg,
                "fontSize":   f"{fz:.2f}rem",
                "fontWeight": "500",
                "padding":    "3px 10px",
                "borderRadius": "20px",
                "margin":     "3px",
                "display":    "inline-block",
                "cursor":     "help",
                "border":     f"1px solid {fg}30",
                "transition": "transform 0.1s",
            },
        ))

    # ── Detail table (top 12 by absolute score) ───────────────────────────────
    top12 = sorted_ents[:12]
    tbl_rows = []
    for entity, data in top12:
        score = data["composite_score"]
        color = "#28a745" if score > 0.05 else ("#dc3545" if score < -0.05 else "#6c757d")
        tbl_rows.append(html.Tr([
            html.Td(entity,          style={"fontSize": "0.82rem", "fontWeight": "500"}),
            html.Td(data["type"],    style={"fontSize": "0.82rem", "color": "#6c757d"}),
            html.Td(str(data["mentions"]), style={"fontSize": "0.82rem", "textAlign": "center"}),
            html.Td(f"{score:+.3f}", style={"fontSize": "0.82rem", "color": color,
                                             "fontWeight": "600", "textAlign": "right"}),
            html.Td(data["example"][:90] + "…",
                    style={"fontSize": "0.78rem", "color": "#6c757d",
                           "maxWidth": "300px", "overflow": "hidden"}),
        ]))

    detail_tbl = html.Table([
        html.Thead(html.Tr([
            html.Th("Entity",   style={"fontSize": "0.75rem"}),
            html.Th("Type",     style={"fontSize": "0.75rem"}),
            html.Th("n",        style={"fontSize": "0.75rem", "textAlign": "center"}),
            html.Th("Score",    style={"fontSize": "0.75rem", "textAlign": "right"}),
            html.Th("Context",  style={"fontSize": "0.75rem"}),
        ])),
        html.Tbody(tbl_rows),
    ], className="table table-sm table-borderless mb-0")

    return tags, detail_tbl, source_label


if __name__ == "__main__":
    app.run(debug=True, port=8052)
