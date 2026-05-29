"""
dashboard.py
------------
Interactive Plotly Dash dashboard for the Churn & Revenue Recovery project.

Run: python dashboards/dashboard.py
Then open: http://127.0.0.1:8050
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc


# ── Load data ──────────────────────────────────────────────────────────────────
def load_recovery_plan() -> pd.DataFrame:
    path = "data/processed/recovery_plan.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "recovery_plan.csv not found. Run `python main.py` first."
        )
    return pd.read_csv(path)


def load_customer_data() -> pd.DataFrame:
    path = "data/processed/customer_data.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "customer_data.csv not found. Run `python data/generate_data.py` first."
        )
    return pd.read_csv(path)


# ── App init ───────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Churn & Revenue Recovery Dashboard",
)

TIER_COLORS = {"High": "#EF4444", "Medium": "#F97316", "Low": "#22C55E"}


# ── Layout helpers ─────────────────────────────────────────────────────────────
def kpi_card(title: str, value: str, color: str = "#3B82F6") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.85rem"}),
            html.H4(value, style={"color": color, "fontWeight": "bold"}),
        ]),
        className="mb-3 shadow-sm",
        style={"borderTop": f"4px solid {color}"},
    )


# ── Layout ─────────────────────────────────────────────────────────────────────
def build_layout(recovery_df: pd.DataFrame, customer_df: pd.DataFrame) -> html.Div:
    total_at_risk    = recovery_df["recovery_score"].sum()
    total_projected  = recovery_df["projected_recovery"].sum()
    high_count       = (recovery_df["tier"] == "High").sum()
    avg_churn_prob   = recovery_df["churn_proba"].mean()

    tier_options = [{"label": t, "value": t} for t in ["All", "High", "Medium", "Low"]]

    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col(html.H2("📉 Customer Churn & Revenue Recovery",
                            className="text-white mt-3 mb-0"), width=9),
            dbc.Col(html.P("Powered by Random Forest + Statistical Analysis",
                           className="text-muted mt-4"), width=3),
        ]),
        html.Hr(style={"borderColor": "#374151"}),

        # KPI Cards
        dbc.Row([
            dbc.Col(kpi_card("Total Revenue at Risk", f"₹{total_at_risk:,.0f}", "#EF4444"), md=3),
            dbc.Col(kpi_card("Projected Recovery (15% lift)", f"₹{total_projected:,.0f}", "#22C55E"), md=3),
            dbc.Col(kpi_card("High-Risk Customers", f"{high_count:,}", "#F97316"), md=3),
            dbc.Col(kpi_card("Avg Churn Probability", f"{avg_churn_prob:.1%}", "#8B5CF6"), md=3),
        ]),

        # Charts Row 1
        dbc.Row([
            dbc.Col(dcc.Graph(id="tier-pie"), md=4),
            dbc.Col(dcc.Graph(id="recovery-bar"), md=4),
            dbc.Col(dcc.Graph(id="churn-proba-hist"), md=4),
        ]),

        # Charts Row 2 — customer data
        dbc.Row([
            dbc.Col(dcc.Graph(id="churn-by-contract"), md=4),
            dbc.Col(dcc.Graph(id="clv-vs-churn"), md=4),
            dbc.Col(dcc.Graph(id="tenure-churn"), md=4),
        ]),

        # Controls + Table
        dbc.Row([
            dbc.Col([
                html.Label("Filter by Tier:", className="text-white"),
                dcc.Dropdown(
                    id="tier-filter",
                    options=tier_options,
                    value="All",
                    clearable=False,
                    style={"color": "#111"},
                ),
                html.Br(),
                html.Label("Min Churn Probability:", className="text-white"),
                dcc.Slider(
                    id="prob-slider",
                    min=0, max=1, step=0.05, value=0.5,
                    marks={i/10: f"{i/10:.0%}" for i in range(0, 11, 2)},
                ),
            ], md=3),
            dbc.Col(
                dash_table.DataTable(
                    id="customer-table",
                    columns=[
                        {"name": "Customer ID",          "id": "customer_id"},
                        {"name": "Churn Prob",           "id": "churn_proba"},
                        {"name": "CLV (₹)",              "id": "clv"},
                        {"name": "Recovery Score",       "id": "recovery_score"},
                        {"name": "Tier",                 "id": "tier"},
                        {"name": "Offer",                "id": "offer"},
                        {"name": "Discount %",           "id": "discount_pct"},
                        {"name": "Projected Recovery ₹", "id": "projected_recovery"},
                    ],
                    page_size=12,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#1F2937", "color": "white", "fontWeight": "bold"},
                    style_cell={"backgroundColor": "#111827", "color": "#D1D5DB", "fontSize": "12px"},
                    style_data_conditional=[
                        {"if": {"filter_query": '{tier} = "High"',   "column_id": "tier"}, "color": "#EF4444"},
                        {"if": {"filter_query": '{tier} = "Medium"', "column_id": "tier"}, "color": "#F97316"},
                        {"if": {"filter_query": '{tier} = "Low"',    "column_id": "tier"}, "color": "#22C55E"},
                    ],
                ), md=9
            ),
        ], className="mt-3"),

        # Hidden stores
        dcc.Store(id="recovery-store", data=recovery_df.to_dict("records")),
        dcc.Store(id="customer-store", data=customer_df.to_dict("records")),

        html.Footer(
            "Gauri Singhal | Customer Churn & Revenue Recovery Model",
            className="text-center text-muted mt-5 mb-3",
            style={"fontSize": "0.8rem"},
        ),
    ], fluid=True, style={"backgroundColor": "#0F172A", "minHeight": "100vh"})


# ── Callbacks ──────────────────────────────────────────────────────────────────
def register_callbacks(app: Dash) -> None:

    @app.callback(
        [
            Output("tier-pie", "figure"),
            Output("recovery-bar", "figure"),
            Output("churn-proba-hist", "figure"),
            Output("customer-table", "data"),
        ],
        [
            Input("tier-filter", "value"),
            Input("prob-slider", "value"),
            Input("recovery-store", "data"),
        ],
    )
    def update_recovery_charts(selected_tier, min_prob, records):
        df = pd.DataFrame(records)
        filtered = df.copy()
        if selected_tier != "All":
            filtered = filtered[filtered["tier"] == selected_tier]
        filtered = filtered[filtered["churn_proba"] >= min_prob]

        # Pie
        tier_counts = df.groupby("tier")["recovery_score"].sum().reset_index()
        pie = px.pie(
            tier_counts, values="recovery_score", names="tier",
            color="tier", color_discrete_map=TIER_COLORS,
            title="Revenue at Risk by Tier",
            template="plotly_dark",
        )

        # Bar
        tier_bar = filtered.groupby("tier")[["recovery_score", "projected_recovery"]].sum().reset_index()
        bar = px.bar(
            tier_bar, x="tier", y=["recovery_score", "projected_recovery"],
            barmode="group", title="At-Risk vs Projected Recovery",
            color_discrete_sequence=["#EF4444", "#22C55E"],
            template="plotly_dark",
        )

        # Histogram
        hist = px.histogram(
            filtered, x="churn_proba", color="tier",
            color_discrete_map=TIER_COLORS, nbins=40,
            title="Churn Probability Distribution",
            template="plotly_dark",
        )

        return pie, bar, hist, filtered.to_dict("records")

    @app.callback(
        [
            Output("churn-by-contract", "figure"),
            Output("clv-vs-churn", "figure"),
            Output("tenure-churn", "figure"),
        ],
        Input("customer-store", "data"),
    )
    def update_customer_charts(records):
        df = pd.DataFrame(records)
        df["churn_label"] = df["churn"].map({0: "Retained", 1: "Churned"})

        # Churn by contract
        rates = df.groupby("contract_type")["churn"].mean().reset_index()
        fig1 = px.bar(
            rates, x="contract_type", y="churn", text_auto=".1%",
            title="Churn Rate by Contract Type",
            color="churn", color_continuous_scale="RdYlGn_r",
            template="plotly_dark",
        )
        fig1.update_layout(coloraxis_showscale=False)

        # CLV vs Churn box
        fig2 = px.box(
            df, x="churn_label", y="clv",
            color="churn_label",
            color_discrete_map={"Retained": "#22C55E", "Churned": "#EF4444"},
            title="CLV Distribution by Churn Status",
            template="plotly_dark",
        )

        # Tenure density
        fig3 = px.histogram(
            df, x="tenure_months", color="churn_label",
            barmode="overlay", opacity=0.6, nbins=50,
            color_discrete_map={"Retained": "#22C55E", "Churned": "#EF4444"},
            title="Tenure Distribution by Churn Status",
            template="plotly_dark",
        )

        return fig1, fig2, fig3


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    recovery_df  = load_recovery_plan()
    customer_df  = load_customer_data()
    app.layout   = build_layout(recovery_df, customer_df)
    register_callbacks(app)
    print("🚀 Dashboard running at http://127.0.0.1:8050")
    app.run(debug=True)
