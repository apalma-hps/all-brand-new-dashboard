from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from forecast_engine import (
    ForecastConfig,
    build_daily_product_fact,
    build_forecast_for_metric,
    build_product_base_facts,
    build_projection_table,
    load_catalogo_operativo,
    load_sales_data,
)


st.set_page_config(
    page_title="Proyeccion por Producto – Marcas HP",
    page_icon="📈",
    layout="wide",
)


def hp_altair_theme():
    return {
        "config": {
            "background": "rgba(0,0,0,0)",
            "view": {"stroke": "transparent"},
            "axis": {
                "labelColor": "#1B1D22",
                "titleColor": "#1B1D22",
                "gridColor": "#E7ECEF",
            },
            "legend": {
                "labelColor": "#1B1D22",
                "titleColor": "#1B1D22",
            },
            "range": {
                "category": [
                    "#1B1D22",
                    "#0F766E",
                    "#14B8A6",
                    "#7AD9CF",
                    "#A7F0E3",
                    "#6F7277",
                ]
            },
        }
    }


alt.themes.register("hp_forecast_theme", hp_altair_theme)
alt.themes.enable("hp_forecast_theme")


st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 85% 10%, #DDF7F1 0, #F7FAFB 35%),
            radial-gradient(circle at 10% 80%, #EAF7F5 0, #F7FAFB 35%),
            #F7FAFB;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.88);
        backdrop-filter: blur(14px);
        border-right: 1px solid rgba(148,163,184,0.25);
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.94);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 14px 36px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.22);
        text-align: center;
    }

    .panel-card {
        background: rgba(255,255,255,0.96);
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
        border: 1px solid rgba(148,163,184,0.20);
        margin-bottom: 1rem;
    }

    .panel-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6F7277;
        margin-bottom: 0.6rem;
        font-weight: 700;
    }

    .small-muted {
        color: #6F7277;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


LOGO_URL = "https://raw.githubusercontent.com/apalma-hps/Dashboard-Ventas-HP/main/logo_hp.png"
BASE_DIR = Path(__file__).resolve().parents[1]


def fmt_money(x):
    if x is None or pd.isna(x):
        return "—"
    return f"${x:,.0f}"


def fmt_num(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.0f}"


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:,.1f}%"


@st.cache_data(ttl=600)
def load_forecast_inputs():
    sales = load_sales_data()
    catalog, source = load_catalogo_operativo(BASE_DIR)
    facts = build_product_base_facts(sales, catalog)
    daily_fact = build_daily_product_fact(facts)
    return sales, catalog, source, facts, daily_fact


sales_df, catalog_df, catalog_source, product_facts, daily_product_fact = load_forecast_inputs()

st.sidebar.markdown("### Actualizacion")
if st.sidebar.button("Actualizar proyeccion"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"Ultima vista: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

if daily_product_fact.empty:
    st.error("No hay suficiente data diaria por producto para proyectar.")
    st.stop()

st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:1.25rem;margin-bottom:1.4rem;">
        <div style="
            width:110px;height:110px;border-radius:55px;border:4px solid #A7F0E3;
            display:flex;align-items:center;justify-content:center;background:#FFFFFF;
            box-shadow:0 18px 45px rgba(15,23,42,0.10);">
            <img src="{LOGO_URL}" style="width:70%;height:70%;border-radius:50%;" />
        </div>
        <div>
            <h1 style="margin:0;">Proyeccion de Ventas por Producto</h1>
            <p style="margin:0.35rem 0 0 0;color:#6F7277;">
            Forecast diario con agregacion a semana y mes, usando demanda historica por producto base.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Fuente catálogo: {catalog_source}. La proyeccion en dinero usa productos base y venta realizada por linea; complementos no se valorizan como producto independiente."
)

restaurants_all = sorted(daily_product_fact["restaurante"].dropna().unique().tolist())
selected_restaurants = st.sidebar.multiselect(
    "Restaurantes",
    options=restaurants_all,
    default=restaurants_all,
)

filtered_daily = daily_product_fact.copy()
if selected_restaurants:
    filtered_daily = filtered_daily[filtered_daily["restaurante"].isin(selected_restaurants)].copy()

recent_cutoff = filtered_daily["fecha"].max() - pd.Timedelta(days=90)
top_products = (
    filtered_daily[filtered_daily["fecha"] >= recent_cutoff]
    .groupby("producto", as_index=False)
    .agg(sales=("sales", "sum"))
    .sort_values("sales", ascending=False)
)
product_options = top_products["producto"].tolist() or sorted(filtered_daily["producto"].dropna().unique().tolist())
default_products = product_options[:5]

selected_products = st.sidebar.multiselect(
    "Productos",
    options=product_options,
    default=default_products,
)

granularity = st.sidebar.radio("Granularidad", ["Día", "Semana", "Mes"], index=1, horizontal=True)
metric_view = st.sidebar.radio("Vista principal", ["Cantidad", "Venta"], index=0, horizontal=True)
lookback_days = st.sidebar.slider("Historial visible", min_value=30, max_value=180, value=90, step=15)

if not selected_products:
    st.info("Selecciona al menos un producto para proyectar.")
    st.stop()

config = ForecastConfig(daily_horizon=90)

actual_qty, forecast_qty = build_forecast_for_metric(
    daily_product_fact,
    "qty",
    selected_restaurants,
    selected_products,
    config,
)
actual_sales, forecast_sales = build_forecast_for_metric(
    daily_product_fact,
    "sales",
    selected_restaurants,
    selected_products,
    config,
)

if actual_qty.empty or forecast_qty.empty:
    st.warning("No se pudo construir la proyeccion con los filtros actuales.")
    st.stop()

qty_daily_total = forecast_qty.groupby("fecha", as_index=False)["valor"].sum()
sales_daily_total = forecast_sales.groupby("fecha", as_index=False)["valor"].sum()

next_7_cutoff = qty_daily_total["fecha"].min() + pd.Timedelta(days=6)
next_30_cutoff = qty_daily_total["fecha"].min() + pd.Timedelta(days=29)

qty_next_7 = qty_daily_total.loc[qty_daily_total["fecha"] <= next_7_cutoff, "valor"].sum()
qty_next_30 = qty_daily_total.loc[qty_daily_total["fecha"] <= next_30_cutoff, "valor"].sum()
sales_next_7 = sales_daily_total.loc[sales_daily_total["fecha"] <= next_7_cutoff, "valor"].sum()
sales_next_30 = sales_daily_total.loc[sales_daily_total["fecha"] <= next_30_cutoff, "valor"].sum()

diag_qty = forecast_qty.groupby("producto", as_index=False)["wape"].mean().rename(columns={"wape": "wape_qty"})
diag_sales = forecast_sales.groupby("producto", as_index=False)["wape"].mean().rename(columns={"wape": "wape_sales"})
diagnostics = diag_qty.merge(diag_sales, on="producto", how="outer")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Cantidad proyectada 7 dias", fmt_num(qty_next_7))
k2.metric("Venta proyectada 7 dias", fmt_money(sales_next_7))
k3.metric("Cantidad proyectada 30 dias", fmt_num(qty_next_30))
k4.metric("Venta proyectada 30 dias", fmt_money(sales_next_30))

history_floor = max(actual_qty["fecha"].max() - pd.Timedelta(days=lookback_days), actual_qty["fecha"].min())

chart_actual = actual_qty if metric_view == "Cantidad" else actual_sales
chart_forecast = forecast_qty if metric_view == "Cantidad" else forecast_sales
chart_actual = chart_actual[chart_actual["fecha"] >= history_floor].copy()

projection_table = build_projection_table(chart_actual, chart_forecast, granularity)

left, right = st.columns([1.7, 1])

with left:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel-title">Proyeccion {metric_view.lower()} · {granularity.lower()}</div>',
        unsafe_allow_html=True,
    )

    chart_df = projection_table.copy()
    if chart_df.empty:
        st.info("No hay datos para graficar.")
    else:
        order_periods = (
            chart_df[["periodo", "periodo_label"]]
            .drop_duplicates()
            .sort_values("periodo")["periodo_label"]
            .tolist()
        )

        base = alt.Chart(chart_df)
        if len(selected_products) == 1 and "lower" in chart_df.columns and "upper" in chart_df.columns:
            band = (
                base.transform_filter(alt.datum.tipo == "forecast")
                .mark_area(opacity=0.10)
                .encode(
                    x=alt.X("periodo_label:N", sort=order_periods, title=""),
                    y=alt.Y("lower:Q", title=metric_view),
                    y2="upper:Q",
                    color=alt.Color("producto:N", legend=None),
                )
            )
        else:
            band = None

        lines = (
            base.mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("periodo_label:N", sort=order_periods, title=""),
                y=alt.Y("valor:Q", title=metric_view),
                color=alt.Color("producto:N", title="Producto"),
                strokeDash=alt.StrokeDash("tipo:N", title="Serie"),
                tooltip=[
                    alt.Tooltip("producto:N", title="Producto"),
                    alt.Tooltip("tipo:N", title="Serie"),
                    alt.Tooltip("periodo_label:N", title="Periodo"),
                    alt.Tooltip("valor:Q", title=metric_view, format=",.2f"),
                ],
            )
            .properties(height=380)
        )

        st.altair_chart(alt.layer(*(layer for layer in [band, lines] if layer is not None)), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Diagnostico del modelo</div>', unsafe_allow_html=True)

    if diagnostics.empty:
        st.info("Sin diagnosticos disponibles.")
    else:
        diag_view = diagnostics.copy()
        diag_view["WAPE qty"] = diag_view["wape_qty"].map(fmt_pct)
        diag_view["WAPE venta"] = diag_view["wape_sales"].map(fmt_pct)
        st.dataframe(
            diag_view[["producto", "WAPE qty", "WAPE venta"]],
            use_container_width=True,
            hide_index=True,
        )

    st.caption("Modelo base: mezcla de promedio por dia de semana, promedio reciente de 7 dias y promedio de 28 dias, con ajuste ligero por tendencia.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("## Tabla de proyeccion")

table_view = projection_table.copy()
if table_view.empty:
    st.warning("No hay tabla de proyeccion con los filtros actuales.")
else:
    table_view["Valor"] = table_view["valor"].map(fmt_num if metric_view == "Cantidad" else fmt_money)
    if metric_view == "Cantidad":
        table_view["Rango"] = table_view.apply(lambda row: f"{row['lower']:,.0f} - {row['upper']:,.0f}" if pd.notna(row["lower"]) else "—", axis=1)
    else:
        table_view["Rango"] = table_view.apply(lambda row: f"{fmt_money(row['lower'])} - {fmt_money(row['upper'])}" if pd.notna(row["lower"]) else "—", axis=1)
    table_view["WAPE"] = table_view["wape"].map(fmt_pct)

    st.dataframe(
        table_view[["periodo_label", "producto", "tipo", "Valor", "Rango", "WAPE"]],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("## Supuestos")
st.markdown(
    """
    - La proyeccion se construye sobre productos base, no sobre complementos.
    - Semana y mes se agregan desde el forecast diario, para mantener coherencia entre niveles.
    - Cuando falta precio por linea, el dinero se completa con precio mediano historico del producto.
    - La precision sera mejor en productos con rotacion estable y peor en productos intermitentes o muy promocionales.
    """
)
