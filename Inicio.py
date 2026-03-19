import re
import unicodedata
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalog_engine import compile_catalog


st.set_page_config(
    page_title="Operations Control Center – HP",
    page_icon="📊",
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


alt.themes.register("hp_theme", hp_altair_theme)
alt.themes.enable("hp_theme")


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

    div[data-testid="stMetricLabel"] {
        color: #6F7277;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetricValue"] {
        color: #1B1D22;
        font-weight: 700;
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

    .stDataFrame {
        background: rgba(255,255,255,0.96);
        border-radius: 18px;
        padding: 0.35rem 0.35rem 0.8rem 0.35rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
        border: 1px solid rgba(148,163,184,0.20);
    }

    .alert-box {
        background: rgba(255,255,255,0.96);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        border-left: 5px solid #14B8A6;
        margin-bottom: 0.8rem;
        border-top: 1px solid rgba(148,163,184,0.12);
        border-right: 1px solid rgba(148,163,184,0.12);
        border-bottom: 1px solid rgba(148,163,184,0.12);
    }

    .alert-box.red { border-left-color: #DC2626; }
    .alert-box.yellow { border-left-color: #D97706; }
    .alert-box.green { border-left-color: #059669; }

    .small-muted {
        color: #6F7277;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSLIeswEs8OILxZmVMwObbli0Zpbbqx7g7h6ZC5Fwm0PCjlZEFy66L9Xpha6ROW3loFCIRiWvEnLRHS/pub?output=csv"
CATALOGO_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQtKQGyCaerGAedhlpzaXlr-ycmm1t08a6lUtg-_3f7yWtJhLkQ6vn0TlI89l0FGVxOUy1Cwj5ykliB/pub?output=csv"
LOGO_URL = "https://raw.githubusercontent.com/apalma-hps/Dashboard-Ventas-HP/main/logo_hp.png"

BASE_DIR = Path(__file__).resolve().parent
CATALOGO_OPERATIVO_PATH = BASE_DIR / "data" / "catalogo_operativo.csv"

COL_RESTAURANTE = "Restaurante"
COL_FECHA = "Fecha"
COL_SUBTOTAL = "Subtotal"
COL_TOTAL = "Total"
COL_DESCUENTOS = "Descuentos"
COL_ESTADO = "Estado"
COL_TIPO = "Tipo"
COL_FOLIO = "Folio"
COL_DETALLE = "Detalle Items"
COL_VENTAS = "ventas_efectivas"

PRICE_RE = re.compile(r"\(\s*\$?\s*(-?[\d\.,]+)\s*\)")
QTY_RE = re.compile(r"\s+[xX]\s*([\d\.]+)\s*$")


def clean_money_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "—": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def fmt_money(x):
    if x is None or pd.isna(x):
        return "—"
    return f"${x:,.0f}"


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:,.1f}%"


def fmt_pp(x):
    if x is None or pd.isna(x):
        return "—"
    sign = "+" if x > 0 else ""
    return f"{sign}{x * 100:,.1f}pp"


def safe_pct_change(current, previous):
    if previous is None or pd.isna(previous) or previous == 0:
        return np.nan
    if current is None or pd.isna(current):
        return np.nan
    return (current - previous) / previous


def detect_tax_column(df_: pd.DataFrame) -> str | None:
    candidates = ["Impuestos", "IVA", "Tax", "Taxes", "Impuesto", "VAT"]
    for column in candidates:
        if column in df_.columns:
            return column
    return None


def get_void_mask(df_: pd.DataFrame, col_estado: str) -> pd.Series:
    if col_estado in df_.columns:
        return df_[col_estado].astype(str).str.strip().str.lower().eq("void")
    return pd.Series(False, index=df_.index)


def is_delivery(val):
    try:
        text = str(val).strip().lower()
    except Exception:
        return False
    return "delivery" in text or "takeout" in text


def norm_key(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    text = re.sub(r"[^\w\s\/\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\-+", "-", text).strip()
    return text


def split_top_level(texto: str, separator: str = "|") -> list[str]:
    if not isinstance(texto, str):
        return []

    partes = []
    buffer = []
    bracket_depth = 0

    for char in texto:
        if char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth > 0:
            bracket_depth -= 1

        if char == separator and bracket_depth == 0:
            parte = "".join(buffer).strip()
            if parte:
                partes.append(parte)
            buffer = []
            continue

        buffer.append(char)

    parte = "".join(buffer).strip()
    if parte:
        partes.append(parte)
    return partes


def parse_qty(qty_text: str) -> float:
    try:
        qty = float(qty_text)
    except (TypeError, ValueError):
        return 1.0
    if qty <= 0:
        return 1.0
    if not qty.is_integer():
        return 1.0 if qty < 1 else float(int(round(qty)))
    return float(int(qty))


def parse_base_item(raw: str) -> tuple[str, float]:
    text = str(raw).strip()
    if not text:
        return "", 0.0

    match_price = PRICE_RE.search(text)
    if match_price:
        text = text[:match_price.start()].strip()

    qty = 1.0
    match_qty = QTY_RE.search(text)
    if match_qty:
        qty = parse_qty(match_qty.group(1))
        text = text[:match_qty.start()].strip()

    return text.strip(), qty


def to_monday(date_value: pd.Timestamp) -> pd.Timestamp:
    date_value = pd.to_datetime(date_value).normalize()
    return date_value - pd.Timedelta(days=date_value.weekday())


def get_current_and_previous_week_ranges(max_date: pd.Timestamp):
    current_start = to_monday(max_date)
    current_end = current_start + pd.Timedelta(days=6)
    previous_start = current_start - pd.Timedelta(days=7)
    previous_end = current_start - pd.Timedelta(days=1)
    return current_start, current_end, previous_start, previous_end


def filtrar_periodo(df: pd.DataFrame, inicio: pd.Timestamp, fin: pd.Timestamp, restaurantes: list[str]) -> pd.DataFrame:
    filtered = df[(df[COL_FECHA] >= inicio) & (df[COL_FECHA] <= fin)].copy()
    if restaurantes:
        filtered = filtered[filtered[COL_RESTAURANTE].isin(restaurantes)].copy()
    return filtered


def calcular_metricas(df_periodo: pd.DataFrame) -> dict:
    if df_periodo.empty:
        return {
            "ventas": 0.0,
            "tickets": 0,
            "ticket_promedio": 0.0,
            "ventas_delivery": 0.0,
            "pct_delivery": 0.0,
            "cancelados": 0,
        }

    is_void = get_void_mask(df_periodo, COL_ESTADO)
    df_valid = df_periodo.loc[~is_void].copy()

    ventas = float(df_valid[COL_VENTAS].sum())
    tickets = int(df_valid[COL_FOLIO].nunique())
    ticket_promedio = ventas / tickets if tickets > 0 else 0.0
    ventas_delivery = float(df_valid.loc[df_valid[COL_TIPO].map(is_delivery), COL_VENTAS].sum())
    pct_delivery = ventas_delivery / ventas if ventas > 0 else 0.0
    cancelados = int(df_periodo.loc[is_void, COL_FOLIO].nunique())

    return {
        "ventas": ventas,
        "tickets": tickets,
        "ticket_promedio": ticket_promedio,
        "ventas_delivery": ventas_delivery,
        "pct_delivery": pct_delivery,
        "cancelados": cancelados,
    }


def build_daily_sales(df_periodo: pd.DataFrame, label: str, week_start: pd.Timestamp) -> pd.DataFrame:
    days = pd.date_range(week_start, week_start + pd.Timedelta(days=6), freq="D")
    base = pd.DataFrame({"dia_fecha": days})

    if df_periodo.empty:
        base["ventas"] = 0.0
        base["tickets"] = 0
        base["periodo"] = label
        base["dia_nombre"] = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        base["ticket_promedio"] = 0.0
        return base

    is_void = get_void_mask(df_periodo, COL_ESTADO)
    df_valid = df_periodo.loc[~is_void].copy()
    df_valid["dia_fecha"] = df_valid[COL_FECHA].dt.normalize()

    sales = (
        df_valid.groupby("dia_fecha", as_index=False)
        .agg(ventas=(COL_VENTAS, "sum"), tickets=(COL_FOLIO, "nunique"))
    )

    out = base.merge(sales, on="dia_fecha", how="left")
    out["ventas"] = out["ventas"].fillna(0.0)
    out["tickets"] = out["tickets"].fillna(0).astype(int)
    out["periodo"] = label
    out["dia_nombre"] = out["dia_fecha"].dt.day_name().map(
        {
            "Monday": "Lun",
            "Tuesday": "Mar",
            "Wednesday": "Mié",
            "Thursday": "Jue",
            "Friday": "Vie",
            "Saturday": "Sáb",
            "Sunday": "Dom",
        }
    )
    out["ticket_promedio"] = np.where(out["tickets"] > 0, out["ventas"] / out["tickets"], 0.0)
    return out


def build_restaurant_performance(df_cur: pd.DataFrame, df_prev: pd.DataFrame) -> pd.DataFrame:
    cur_void = get_void_mask(df_cur, COL_ESTADO)
    prev_void = get_void_mask(df_prev, COL_ESTADO)

    cur = (
        df_cur.loc[~cur_void]
        .groupby(COL_RESTAURANTE, as_index=False)
        .agg(ventas_actual=(COL_VENTAS, "sum"), tickets_actual=(COL_FOLIO, "nunique"))
    )
    prev = (
        df_prev.loc[~prev_void]
        .groupby(COL_RESTAURANTE, as_index=False)
        .agg(ventas_previa=(COL_VENTAS, "sum"), tickets_previa=(COL_FOLIO, "nunique"))
    )

    out = cur.merge(prev, on=COL_RESTAURANTE, how="outer").fillna(0)
    if out.empty:
        return out

    out["ticket_actual"] = np.where(out["tickets_actual"] > 0, out["ventas_actual"] / out["tickets_actual"], 0.0)
    out["ticket_previa"] = np.where(out["tickets_previa"] > 0, out["ventas_previa"] / out["tickets_previa"], 0.0)
    out["delta_ventas"] = out.apply(lambda row: safe_pct_change(row["ventas_actual"], row["ventas_previa"]), axis=1)
    out["delta_ticket"] = out.apply(lambda row: safe_pct_change(row["ticket_actual"], row["ticket_previa"]), axis=1)

    def tag_status(delta_ventas, delta_ticket):
        if pd.notna(delta_ventas) and delta_ventas <= -0.05:
            return "Caída ventas"
        if pd.notna(delta_ticket) and delta_ticket <= -0.05:
            return "Cae ticket"
        if pd.notna(delta_ventas) and delta_ventas >= 0.05:
            return "Crecimiento"
        return "Estable"

    out["status"] = out.apply(lambda row: tag_status(row["delta_ventas"], row["delta_ticket"]), axis=1)
    return out.sort_values("ventas_actual", ascending=False)


def detect_daily_anomalies(df_daily_current: pd.DataFrame) -> pd.DataFrame:
    values = df_daily_current["ventas"].astype(float)
    out = df_daily_current.copy()

    if len(values) < 3 or values.std(ddof=0) == 0:
        out["zscore"] = np.nan
        out["anomaly"] = False
        return out

    mean = values.mean()
    std = values.std(ddof=0)
    out["zscore"] = (out["ventas"] - mean) / std
    out["anomaly"] = out["zscore"].abs() >= 1.5
    return out


def load_catalogo_operativo() -> tuple[pd.DataFrame | None, str]:
    if CATALOGO_OPERATIVO_PATH.exists():
        try:
            catalog = pd.read_csv(CATALOGO_OPERATIVO_PATH)
            return catalog, f"Local compilado: {CATALOGO_OPERATIVO_PATH.name}"
        except Exception:
            pass

    try:
        raw_catalog = pd.read_csv(CATALOGO_URL)
    except Exception:
        return None, "Sin catálogo"

    compiled, _ = compile_catalog(raw_catalog)
    return compiled, "Compilado al vuelo desde Google Sheets"


def build_catalog_lookups(catalogo: pd.DataFrame) -> tuple[dict, dict]:
    if catalogo is None or catalogo.empty:
        return {}, {}

    exact_lookup = {}
    canonical_lookup = {}

    for row in catalogo.to_dict("records"):
        concepto_key = row.get("concepto_key", "")
        tipo_key = row.get("tipo_concepto_key", "")
        canon_key = row.get("canon_key", "")
        include_in_count = bool(row.get("include_in_count", True))

        if concepto_key and tipo_key:
            exact_lookup[(concepto_key, tipo_key)] = {
                "concepto_canonico": str(row.get("concepto_canonico", "") or "").strip(),
                "include_in_count": include_in_count,
            }

        if canon_key and canon_key not in canonical_lookup:
            canonical_lookup[canon_key] = {
                "concepto_canonico": str(row.get("concepto_canonico", "") or "").strip(),
                "include_in_count": include_in_count,
            }

    return exact_lookup, canonical_lookup


@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    df_ = pd.read_csv(DATA_URL)
    df_.columns = [col.strip() for col in df_.columns]

    required = [COL_RESTAURANTE, COL_FECHA, COL_TOTAL]
    missing = [column for column in required if column not in df_.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df_[COL_FECHA] = pd.to_datetime(df_[COL_FECHA], errors="coerce", dayfirst=True)
    df_[COL_TOTAL] = clean_money_series(df_[COL_TOTAL]).fillna(0.0)

    if COL_SUBTOTAL not in df_.columns:
        df_[COL_SUBTOTAL] = df_[COL_TOTAL]
    else:
        df_[COL_SUBTOTAL] = clean_money_series(df_[COL_SUBTOTAL]).fillna(0.0)

    if COL_DESCUENTOS not in df_.columns:
        df_[COL_DESCUENTOS] = 0.0
    else:
        df_[COL_DESCUENTOS] = clean_money_series(df_[COL_DESCUENTOS]).fillna(0.0)

    tax_col = detect_tax_column(df_)
    if tax_col is not None:
        df_["_impuestos"] = clean_money_series(df_[tax_col]).fillna(0.0)
    else:
        df_["_impuestos"] = (df_[COL_TOTAL] - df_[COL_SUBTOTAL]).clip(lower=0.0)

    is_void = get_void_mask(df_, COL_ESTADO)
    df_["_calc_sti_d"] = (df_[COL_SUBTOTAL] + df_["_impuestos"] - df_[COL_DESCUENTOS]).fillna(0.0)
    df_["_ventas_brutas_regla"] = np.where(
        df_["_calc_sti_d"] > df_[COL_TOTAL],
        df_[COL_TOTAL],
        df_["_calc_sti_d"],
    )
    df_[COL_VENTAS] = np.where(
        is_void,
        0.0,
        pd.Series(df_["_ventas_brutas_regla"], index=df_.index).clip(lower=0.0),
    )

    if COL_FOLIO not in df_.columns:
        df_[COL_FOLIO] = np.arange(len(df_)).astype(str)
    else:
        df_[COL_FOLIO] = df_[COL_FOLIO].astype(str)

    if COL_TIPO not in df_.columns:
        df_[COL_TIPO] = ""

    if COL_DETALLE not in df_.columns:
        df_[COL_DETALLE] = ""

    return df_


@st.cache_data(ttl=600)
def build_items_flat(df: pd.DataFrame, catalogo: pd.DataFrame | None) -> pd.DataFrame:
    g = df.copy()
    g = g[g[COL_FECHA].notna()].copy()
    exact_lookup, canonical_lookup = build_catalog_lookups(catalogo)

    rows = []
    for _, row in g.iterrows():
        detalle = row.get(COL_DETALLE, "")
        for producto in split_top_level(detalle):
            base_text = producto.split("[", 1)[0].strip()
            item_raw, qty = parse_base_item(base_text)
            if not item_raw or qty <= 0:
                continue

            item_key = norm_key(item_raw)
            tipo_key = "base"
            meta = exact_lookup.get((item_key, tipo_key), canonical_lookup.get(item_key))
            include_in_count = meta.get("include_in_count", True) if isinstance(meta, dict) else True
            if not include_in_count:
                continue

            item = meta.get("concepto_canonico", item_raw) if isinstance(meta, dict) else item_raw

            rows.append(
                {
                    "Fecha": row[COL_FECHA],
                    "Restaurante": row.get(COL_RESTAURANTE, None),
                    "Estado": row.get(COL_ESTADO, None),
                    "Folio": row.get(COL_FOLIO, None),
                    "item": item,
                    "qty": qty,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Fecha", "Restaurante", "Estado", "Folio", "item", "qty"])

    return pd.DataFrame(rows)


def build_product_mix(items_cur: pd.DataFrame, items_prev: pd.DataFrame) -> pd.DataFrame:
    cur_valid = items_cur.loc[~get_void_mask(items_cur, "Estado")] if not items_cur.empty else items_cur
    prev_valid = items_prev.loc[~get_void_mask(items_prev, "Estado")] if not items_prev.empty else items_prev

    cur_mix = (
        cur_valid.groupby("item", as_index=False).agg(qty_actual=("qty", "sum"))
        if not cur_valid.empty
        else pd.DataFrame(columns=["item", "qty_actual"])
    )
    prev_mix = (
        prev_valid.groupby("item", as_index=False).agg(qty_previa=("qty", "sum"))
        if not prev_valid.empty
        else pd.DataFrame(columns=["item", "qty_previa"])
    )

    mix = cur_mix.merge(prev_mix, on="item", how="outer").fillna(0)
    if mix.empty:
        return mix

    total_cur = float(mix["qty_actual"].sum())
    total_prev = float(mix["qty_previa"].sum())
    mix["mix_actual"] = mix["qty_actual"] / total_cur if total_cur > 0 else 0.0
    mix["mix_previa"] = mix["qty_previa"] / total_prev if total_prev > 0 else 0.0
    mix["delta_mix_pp"] = (mix["mix_actual"] - mix["mix_previa"]) * 100
    mix["delta_qty"] = mix.apply(lambda row: safe_pct_change(row["qty_actual"], row["qty_previa"]), axis=1)
    return mix.sort_values("qty_actual", ascending=False)


st.sidebar.markdown("### Actualización")
if st.sidebar.button("Actualizar data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"Última vista: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

try:
    df = load_data()
except Exception as exc:
    st.error(f"No se pudo cargar la base: {exc}")
    st.stop()

if df[COL_FECHA].dropna().empty:
    st.error("No hay fechas válidas en la base.")
    st.stop()

catalogo_operativo, catalog_source = load_catalogo_operativo()
if catalogo_operativo is not None and not catalogo_operativo.empty and "compile_status" in catalogo_operativo.columns:
    catalogo_operativo = catalogo_operativo[catalogo_operativo["compile_status"].eq("ok")].copy()

items_flat = build_items_flat(df, catalogo_operativo)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filtros")

restaurants_all = sorted(df[COL_RESTAURANTE].dropna().unique().tolist())
selected_restaurants = st.sidebar.multiselect(
    "Restaurantes",
    options=restaurants_all,
    default=restaurants_all,
)

max_fecha_data = df[COL_FECHA].max().normalize()
fecha_referencia = st.sidebar.date_input(
    "Semana de referencia",
    value=max_fecha_data.date(),
    help="Selecciona cualquier fecha. El dashboard tomará el lunes de esa semana como semana actual.",
)

fecha_referencia_ts = pd.to_datetime(fecha_referencia)
cur_start, cur_end, prev_start, prev_end = get_current_and_previous_week_ranges(fecha_referencia_ts)

st.sidebar.markdown("### Semana operativa")
st.sidebar.caption(f"Semana actual: {cur_start.strftime('%d/%m/%Y')} → {cur_end.strftime('%d/%m/%Y')}")
st.sidebar.caption(f"Semana previa: {prev_start.strftime('%d/%m/%Y')} → {prev_end.strftime('%d/%m/%Y')}")
st.sidebar.caption(f"Fecha máxima detectada en base: {max_fecha_data.strftime('%d/%m/%Y')}")

col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown(
        f"""
        <div style="
            width:120px;height:120px;border-radius:60px;
            border:4px solid #A7F0E3;display:flex;align-items:center;justify-content:center;
            background:#FFFFFF;box-shadow:0 18px 45px rgba(15,23,42,0.10);">
            <img src="{LOGO_URL}" style="width:70%;height:70%;border-radius:50%;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_title:
    st.markdown(
        f"""
        <h1 style="margin-bottom:0;">Operations Control Center</h1>
        <p style="color:#6F7277;font-size:0.98rem;margin-top:0.25rem;">
        Vista ejecutiva semanal · ventas · ticket promedio · restaurantes · mix de producto
        </p>
        <p style="color:#6F7277;font-size:0.86rem;margin-top:0.4rem;">
        Semana seleccionada: {cur_start.strftime('%d/%m/%Y')} → {cur_end.strftime('%d/%m/%Y')} ·
        Fuente catálogo: {catalog_source}
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

df_cur = filtrar_periodo(df, cur_start, cur_end, selected_restaurants)
df_prev = filtrar_periodo(df, prev_start, prev_end, selected_restaurants)

items_cur = filtrar_periodo(items_flat, cur_start, cur_end, selected_restaurants) if not items_flat.empty else items_flat
items_prev = filtrar_periodo(items_flat, prev_start, prev_end, selected_restaurants) if not items_flat.empty else items_flat

m_cur = calcular_metricas(df_cur)
m_prev = calcular_metricas(df_prev)

delta_ventas = safe_pct_change(m_cur["ventas"], m_prev["ventas"])
delta_tickets = safe_pct_change(m_cur["tickets"], m_prev["tickets"])
delta_ticket = safe_pct_change(m_cur["ticket_promedio"], m_prev["ticket_promedio"])
delta_delivery_pp = m_cur["pct_delivery"] - m_prev["pct_delivery"] if pd.notna(m_cur["pct_delivery"]) and pd.notna(m_prev["pct_delivery"]) else np.nan
delta_cancelados = safe_pct_change(m_cur["cancelados"], m_prev["cancelados"])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ventas semana", fmt_money(m_cur["ventas"]), fmt_pct(delta_ventas))
k2.metric("Tickets", f"{m_cur['tickets']:,.0f}", fmt_pct(delta_tickets))
k3.metric("Ticket promedio", fmt_money(m_cur["ticket_promedio"]), fmt_pct(delta_ticket))
k4.metric("Delivery mix", fmt_pct(m_cur["pct_delivery"]), fmt_pp(delta_delivery_pp))
k5.metric("Cancelados", f"{m_cur['cancelados']:,.0f}", fmt_pct(delta_cancelados))

daily_cur = build_daily_sales(df_cur, "Semana actual", cur_start)
daily_prev = build_daily_sales(df_prev, "Semana previa", prev_start)
daily_prev_plot = daily_prev.copy()
daily_prev_plot["dia_fecha"] = daily_cur["dia_fecha"].values
daily_plot = pd.concat([daily_cur, daily_prev_plot], ignore_index=True)
daily_anom = detect_daily_anomalies(daily_cur)
restaurant_perf = build_restaurant_performance(df_cur, df_prev)
product_mix = build_product_mix(items_cur, items_prev)

left, right = st.columns([1.65, 1])

with left:
    st.markdown('<div class="panel-title">Sales Velocity · Ventas por día</div>', unsafe_allow_html=True)

    line = (
        alt.Chart(daily_plot)
        .mark_line(strokeWidth=3, point=True)
        .encode(
            x=alt.X("dia_nombre:N", sort=["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"], title=""),
            y=alt.Y("ventas:Q", title="Ventas"),
            color=alt.Color("periodo:N", title="Periodo"),
            tooltip=[
                alt.Tooltip("periodo:N", title="Periodo"),
                alt.Tooltip("dia_nombre:N", title="Día"),
                alt.Tooltip("ventas:Q", title="Ventas", format=",.0f"),
                alt.Tooltip("tickets:Q", title="Tickets"),
                alt.Tooltip("ticket_promedio:Q", title="Ticket promedio", format=",.2f"),
            ],
        )
        .properties(height=340)
    )

    st.altair_chart(line, use_container_width=True)

with right:
    st.markdown('<div class="panel-title">Anomalías semanales</div>', unsafe_allow_html=True)

    anomaly_rows = daily_anom[daily_anom["anomaly"]].copy()
    if anomaly_rows.empty:
        st.success("No se detectaron anomalías relevantes en ventas diarias.")
    else:
        for _, row in anomaly_rows.iterrows():
            css = "red" if row["zscore"] < 0 else "green"
            title = "Caída atípica" if row["zscore"] < 0 else "Pico atípico"
            st.markdown(
                f"""
                <div class="alert-box {css}">
                    <strong>{title}</strong><br>
                    Día: {row['dia_nombre']}<br>
                    Ventas: {fmt_money(row['ventas'])}<br>
                    Z-score: {row['zscore']:.2f}
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("## Alertas operativas")

alerts = []
if pd.notna(delta_ventas) and delta_ventas <= -0.05:
    alerts.append(("red", f"Ventas semanales cayendo {fmt_pct(delta_ventas)} vs semana previa."))
if pd.notna(delta_ticket) and delta_ticket <= -0.05:
    alerts.append(("yellow", f"Ticket promedio cayendo {fmt_pct(delta_ticket)} vs semana previa."))

if not restaurant_perf.empty:
    worst_sales = restaurant_perf.sort_values("delta_ventas", ascending=True).head(1)
    if not worst_sales.empty and pd.notna(worst_sales.iloc[0]["delta_ventas"]) and worst_sales.iloc[0]["delta_ventas"] <= -0.05:
        alerts.append(
            (
                "red",
                f"{worst_sales.iloc[0][COL_RESTAURANTE]} presenta la peor caída en ventas: {fmt_pct(worst_sales.iloc[0]['delta_ventas'])}.",
            )
        )

    worst_ticket = restaurant_perf.sort_values("delta_ticket", ascending=True).head(1)
    if not worst_ticket.empty and pd.notna(worst_ticket.iloc[0]["delta_ticket"]) and worst_ticket.iloc[0]["delta_ticket"] <= -0.05:
        alerts.append(
            (
                "yellow",
                f"{worst_ticket.iloc[0][COL_RESTAURANTE]} presenta la mayor caída en ticket promedio: {fmt_pct(worst_ticket.iloc[0]['delta_ticket'])}.",
            )
        )

if not product_mix.empty:
    top_up = product_mix.sort_values("delta_mix_pp", ascending=False).head(1)
    top_down = product_mix.sort_values("delta_mix_pp", ascending=True).head(1)
    if not top_up.empty and top_up.iloc[0]["delta_mix_pp"] > 1:
        alerts.append(("green", f"Producto ganando mix: {top_up.iloc[0]['item']} ({top_up.iloc[0]['delta_mix_pp']:+.1f}pp)."))
    if not top_down.empty and top_down.iloc[0]["delta_mix_pp"] < -1:
        alerts.append(("yellow", f"Producto perdiendo mix: {top_down.iloc[0]['item']} ({top_down.iloc[0]['delta_mix_pp']:+.1f}pp)."))

if not alerts:
    st.info("Sin alertas críticas. La operación luce estable contra la semana previa.")
else:
    for css, message in alerts:
        st.markdown(
            f"""
            <div class="alert-box {css}">
                {message}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("## Performance por restaurante")

if restaurant_perf.empty:
    st.warning("No hay datos de restaurantes para los filtros actuales.")
else:
    rest_show = restaurant_perf.copy()
    rest_show["Ventas actual"] = rest_show["ventas_actual"].map(fmt_money)
    rest_show["Ventas previa"] = rest_show["ventas_previa"].map(fmt_money)
    rest_show["Δ Ventas"] = rest_show["delta_ventas"].map(fmt_pct)
    rest_show["Ticket actual"] = rest_show["ticket_actual"].map(fmt_money)
    rest_show["Ticket previo"] = rest_show["ticket_previa"].map(fmt_money)
    rest_show["Δ Ticket"] = rest_show["delta_ticket"].map(fmt_pct)

    st.dataframe(
        rest_show[
            [
                COL_RESTAURANTE,
                "Ventas actual",
                "Ventas previa",
                "Δ Ventas",
                "Ticket actual",
                "Ticket previo",
                "Δ Ticket",
                "status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("## Heatmap de ventas por restaurante y día")

if df_cur.empty:
    st.warning("No hay datos en la semana actual para el heatmap.")
else:
    heat = df_cur.loc[~get_void_mask(df_cur, COL_ESTADO)].copy()
    heat["dia_nombre"] = heat[COL_FECHA].dt.day_name().map(
        {
            "Monday": "Lun",
            "Tuesday": "Mar",
            "Wednesday": "Mié",
            "Thursday": "Jue",
            "Friday": "Vie",
            "Saturday": "Sáb",
            "Sunday": "Dom",
        }
    )

    heat = heat.groupby([COL_RESTAURANTE, "dia_nombre"], as_index=False).agg(ventas=(COL_VENTAS, "sum"))

    heatmap = (
        alt.Chart(heat)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("dia_nombre:N", sort=["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"], title=""),
            y=alt.Y(f"{COL_RESTAURANTE}:N", sort="-x", title=""),
            color=alt.Color("ventas:Q", title="Ventas"),
            tooltip=[
                alt.Tooltip(f"{COL_RESTAURANTE}:N", title="Restaurante"),
                alt.Tooltip("dia_nombre:N", title="Día"),
                alt.Tooltip("ventas:Q", title="Ventas", format=",.0f"),
            ],
        )
        .properties(height=max(180, 40 * max(1, heat[COL_RESTAURANTE].nunique())))
    )

    st.altair_chart(heatmap, use_container_width=True)

st.markdown("## Mix de producto resumido")

if product_mix.empty:
    st.warning("No se pudo construir el mix de productos con los datos actuales.")
else:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Productos ganando participación")
        top_gain = product_mix.sort_values("delta_mix_pp", ascending=False).head(10).copy()
        top_gain["Mix actual"] = top_gain["mix_actual"].map(fmt_pct)
        top_gain["Mix previo"] = top_gain["mix_previa"].map(fmt_pct)
        top_gain["Δ Mix"] = top_gain["delta_mix_pp"].map(lambda x: f"{x:+.1f}pp")
        st.dataframe(
            top_gain[["item", "qty_actual", "qty_previa", "Mix actual", "Mix previo", "Δ Mix"]],
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.markdown("### Productos perdiendo participación")
        top_loss = product_mix.sort_values("delta_mix_pp", ascending=True).head(10).copy()
        top_loss["Mix actual"] = top_loss["mix_actual"].map(fmt_pct)
        top_loss["Mix previo"] = top_loss["mix_previa"].map(fmt_pct)
        top_loss["Δ Mix"] = top_loss["delta_mix_pp"].map(lambda x: f"{x:+.1f}pp")
        st.dataframe(
            top_loss[["item", "qty_actual", "qty_previa", "Mix actual", "Mix previo", "Δ Mix"]],
            use_container_width=True,
            hide_index=True,
        )

    top_products_chart = (
        alt.Chart(product_mix.head(15))
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("item:N", sort="-x", title=""),
            x=alt.X("qty_actual:Q", title="Cantidad actual"),
            tooltip=[
                alt.Tooltip("item:N", title="Producto"),
                alt.Tooltip("qty_actual:Q", title="Qty actual"),
                alt.Tooltip("qty_previa:Q", title="Qty previa"),
                alt.Tooltip("delta_mix_pp:Q", title="Δ mix pp", format=".2f"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(top_products_chart, use_container_width=True)

st.markdown("---")
st.markdown(
    f"""
    <div class="small-muted">
        Base: Google Sheets · Cálculo principal: ventas efectivas · Semana operativa: lunes a domingo ·
        Fecha máxima detectada: {max_fecha_data.strftime('%d/%m/%Y')}
    </div>
    """,
    unsafe_allow_html=True,
)
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

    div[data-testid="stMetricLabel"] {
        color: #6F7277;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetricValue"] {
        color: #1B1D22;
        font-weight: 700;
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

    .stDataFrame {
        background: rgba(255,255,255,0.96);
        border-radius: 18px;
        padding: 0.35rem 0.35rem 0.8rem 0.35rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
        border: 1px solid rgba(148,163,184,0.20);
    }

    .alert-box {
        background: rgba(255,255,255,0.96);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        border-left: 5px solid #14B8A6;
        margin-bottom: 0.8rem;
        border-top: 1px solid rgba(148,163,184,0.12);
        border-right: 1px solid rgba(148,163,184,0.12);
        border-bottom: 1px solid rgba(148,163,184,0.12);
    }

    .alert-box.red { border-left-color: #DC2626; }
    .alert-box.yellow { border-left-color: #D97706; }
    .alert-box.green { border-left-color: #059669; }

    .small-muted {
        color: #6F7277;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSLIeswEs8OILxZmVMwObbli0Zpbbqx7g7h6ZC5Fwm0PCjlZEFy66L9Xpha6ROW3loFCIRiWvEnLRHS/pub?output=csv"
CATALOGO_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQtKQGyCaerGAedhlpzaXlr-ycmm1t08a6lUtg-_3f7yWtJhLkQ6vn0TlI89l0FGVxOUy1Cwj5ykliB/pub?output=csv"
LOGO_URL = "https://raw.githubusercontent.com/apalma-hps/Dashboard-Ventas-HP/main/logo_hp.png"

BASE_DIR = Path(__file__).resolve().parent
CATALOGO_OPERATIVO_PATH = BASE_DIR / "data" / "catalogo_operativo.csv"

COL_RESTAURANTE = "Restaurante"
COL_FECHA = "Fecha"
COL_SUBTOTAL = "Subtotal"
COL_TOTAL = "Total"
COL_DESCUENTOS = "Descuentos"
COL_ESTADO = "Estado"
COL_TIPO = "Tipo"
COL_FOLIO = "Folio"
COL_DETALLE = "Detalle Items"
COL_VENTAS = "ventas_efectivas"

PRICE_RE = re.compile(r"\(\s*\$?\s*(-?[\d\.,]+)\s*\)")
QTY_RE = re.compile(r"\s+[xX]\s*([\d\.]+)\s*$")


def clean_money_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "—": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def fmt_money(x):
    if x is None or pd.isna(x):
        return "—"
    return f"${x:,.0f}"


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:,.1f}%"


def fmt_pp(x):
    if x is None or pd.isna(x):
        return "—"
    sign = "+" if x > 0 else ""
    return f"{sign}{x * 100:,.1f}pp"


def safe_pct_change(current, previous):
    if previous is None or pd.isna(previous) or previous == 0:
        return np.nan
    if current is None or pd.isna(current):
        return np.nan
    return (current - previous) / previous


def detect_tax_column(df_: pd.DataFrame) -> str | None:
    candidates = ["Impuestos", "IVA", "Tax", "Taxes", "Impuesto", "VAT"]
    for column in candidates:
        if column in df_.columns:
            return column
    return None


def get_void_mask(df_: pd.DataFrame, col_estado: str) -> pd.Series:
    if col_estado in df_.columns:
        return df_[col_estado].astype(str).str.strip().str.lower().eq("void")
    return pd.Series(False, index=df_.index)


def is_delivery(val):
    try:
        text = str(val).strip().lower()
    except Exception:
        return False
    return "delivery" in text or "takeout" in text


def norm_key(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    text = re.sub(r"[^\w\s\/\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\-+", "-", text).strip()
    return text


def split_top_level(texto: str, separator: str = "|") -> list[str]:
    if not isinstance(texto, str):
        return []

    partes = []
    buffer = []
    bracket_depth = 0

    for char in texto:
        if char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth > 0:
            bracket_depth -= 1

        if char == separator and bracket_depth == 0:
            parte = "".join(buffer).strip()
            if parte:
                partes.append(parte)
            buffer = []
            continue

        buffer.append(char)

    parte = "".join(buffer).strip()
    if parte:
        partes.append(parte)
    return partes


def parse_qty(qty_text: str) -> float:
    try:
        qty = float(qty_text)
    except (TypeError, ValueError):
        return 1.0
    if qty <= 0:
        return 1.0
    if not qty.is_integer():
        return 1.0 if qty < 1 else float(int(round(qty)))
    return float(int(qty))


def parse_base_item(raw: str) -> tuple[str, float]:
    text = str(raw).strip()
    if not text:
        return "", 0.0

    match_price = PRICE_RE.search(text)
    if match_price:
        text = text[:match_price.start()].strip()

    qty = 1.0
    match_qty = QTY_RE.search(text)
    if match_qty:
        qty = parse_qty(match_qty.group(1))
        text = text[:match_qty.start()].strip()

    return text.strip(), qty


def to_monday(date_value: pd.Timestamp) -> pd.Timestamp:
    date_value = pd.to_datetime(date_value).normalize()
    return date_value - pd.Timedelta(days=date_value.weekday())


def get_current_and_previous_week_ranges(max_date: pd.Timestamp):
    current_start = to_monday(max_date)
    current_end = current_start + pd.Timedelta(days=6)
    previous_start = current_start - pd.Timedelta(days=7)
    previous_end = current_start - pd.Timedelta(days=1)
    return current_start, current_end, previous_start, previous_end


def filtrar_periodo(df: pd.DataFrame, inicio: pd.Timestamp, fin: pd.Timestamp, restaurantes: list[str]) -> pd.DataFrame:
    filtered = df[(df[COL_FECHA] >= inicio) & (df[COL_FECHA] <= fin)].copy()
    if restaurantes:
        filtered = filtered[filtered[COL_RESTAURANTE].isin(restaurantes)].copy()
    return filtered


def calcular_metricas(df_periodo: pd.DataFrame) -> dict:
    if df_periodo.empty:
        return {
            "ventas": 0.0,
            "tickets": 0,
            "ticket_promedio": 0.0,
            "ventas_delivery": 0.0,
            "pct_delivery": 0.0,
            "cancelados": 0,
        }

    is_void = get_void_mask(df_periodo, COL_ESTADO)
    df_valid = df_periodo.loc[~is_void].copy()

    ventas = float(df_valid[COL_VENTAS].sum())
    tickets = int(df_valid[COL_FOLIO].nunique())
    ticket_promedio = ventas / tickets if tickets > 0 else 0.0
    ventas_delivery = float(df_valid.loc[df_valid[COL_TIPO].map(is_delivery), COL_VENTAS].sum())
    pct_delivery = ventas_delivery / ventas if ventas > 0 else 0.0
    cancelados = int(df_periodo.loc[is_void, COL_FOLIO].nunique())

    return {
        "ventas": ventas,
        "tickets": tickets,
        "ticket_promedio": ticket_promedio,
        "ventas_delivery": ventas_delivery,
        "pct_delivery": pct_delivery,
        "cancelados": cancelados,
    }


def build_daily_sales(df_periodo: pd.DataFrame, label: str, week_start: pd.Timestamp) -> pd.DataFrame:
    days = pd.date_range(week_start, week_start + pd.Timedelta(days=6), freq="D")
    base = pd.DataFrame({"dia_fecha": days})

    if df_periodo.empty:
        base["ventas"] = 0.0
        base["tickets"] = 0
        base["periodo"] = label
        base["dia_nombre"] = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        base["ticket_promedio"] = 0.0
        return base

    is_void = get_void_mask(df_periodo, COL_ESTADO)
    df_valid = df_periodo.loc[~is_void].copy()
    df_valid["dia_fecha"] = df_valid[COL_FECHA].dt.normalize()

    sales = (
        df_valid.groupby("dia_fecha", as_index=False)
        .agg(ventas=(COL_VENTAS, "sum"), tickets=(COL_FOLIO, "nunique"))
    )

    out = base.merge(sales, on="dia_fecha", how="left")
    out["ventas"] = out["ventas"].fillna(0.0)
    out["tickets"] = out["tickets"].fillna(0).astype(int)
    out["periodo"] = label
    out["dia_nombre"] = out["dia_fecha"].dt.day_name().map(
        {
            "Monday": "Lun",
            "Tuesday": "Mar",
            "Wednesday": "Mié",
            "Thursday": "Jue",
            "Friday": "Vie",
            "Saturday": "Sáb",
            "Sunday": "Dom",
        }
    )
    out["ticket_promedio"] = np.where(out["tickets"] > 0, out["ventas"] / out["tickets"], 0.0)
    return out


def build_restaurant_performance(df_cur: pd.DataFrame, df_prev: pd.DataFrame) -> pd.DataFrame:
    cur_void = get_void_mask(df_cur, COL_ESTADO)
    prev_void = get_void_mask(df_prev, COL_ESTADO)

    cur = (
        df_cur.loc[~cur_void]
        .groupby(COL_RESTAURANTE, as_index=False)
        .agg(ventas_actual=(COL_VENTAS, "sum"), tickets_actual=(COL_FOLIO, "nunique"))
    )
    prev = (
        df_prev.loc[~prev_void]
        .groupby(COL_RESTAURANTE, as_index=False)
        .agg(ventas_previa=(COL_VENTAS, "sum"), tickets_previa=(COL_FOLIO, "nunique"))
    )

    out = cur.merge(prev, on=COL_RESTAURANTE, how="outer").fillna(0)
    if out.empty:
        return out

    out["ticket_actual"] = np.where(out["tickets_actual"] > 0, out["ventas_actual"] / out["tickets_actual"], 0.0)
    out["ticket_previa"] = np.where(out["tickets_previa"] > 0, out["ventas_previa"] / out["tickets_previa"], 0.0)
    out["delta_ventas"] = out.apply(lambda row: safe_pct_change(row["ventas_actual"], row["ventas_previa"]), axis=1)
    out["delta_ticket"] = out.apply(lambda row: safe_pct_change(row["ticket_actual"], row["ticket_previa"]), axis=1)

    def tag_status(delta_ventas, delta_ticket):
        if pd.notna(delta_ventas) and delta_ventas <= -0.05:
            return "Caída ventas"
        if pd.notna(delta_ticket) and delta_ticket <= -0.05:
            return "Cae ticket"
        if pd.notna(delta_ventas) and delta_ventas >= 0.05:
            return "Crecimiento"
        return "Estable"

    out["status"] = out.apply(lambda row: tag_status(row["delta_ventas"], row["delta_ticket"]), axis=1)
    return out.sort_values("ventas_actual", ascending=False)


def detect_daily_anomalies(df_daily_current: pd.DataFrame) -> pd.DataFrame:
    values = df_daily_current["ventas"].astype(float)
    out = df_daily_current.copy()

    if len(values) < 3 or values.std(ddof=0) == 0:
        out["zscore"] = np.nan
        out["anomaly"] = False
        return out

    mean = values.mean()
    std = values.std(ddof=0)
    out["zscore"] = (out["ventas"] - mean) / std
    out["anomaly"] = out["zscore"].abs() >= 1.5
    return out


def load_catalogo_operativo() -> tuple[pd.DataFrame | None, str]:
    if CATALOGO_OPERATIVO_PATH.exists():
        try:
            catalog = pd.read_csv(CATALOGO_OPERATIVO_PATH)
            return catalog, f"Local compilado: {CATALOGO_OPERATIVO_PATH.name}"
        except Exception:
            pass

    try:
        raw_catalog = pd.read_csv(CATALOGO_URL)
    except Exception:
        return None, "Sin catálogo"

    compiled, _ = compile_catalog(raw_catalog)
    return compiled, "Compilado al vuelo desde Google Sheets"


def build_catalog_lookups(catalogo: pd.DataFrame) -> tuple[dict, dict]:
    if catalogo is None or catalogo.empty:
        return {}, {}

    exact_lookup = {}
    canonical_lookup = {}

    for row in catalogo.to_dict("records"):
        concepto_key = row.get("concepto_key", "")
        tipo_key = row.get("tipo_concepto_key", "")
        canon_key = row.get("canon_key", "")
        include_in_count = bool(row.get("include_in_count", True))

        if concepto_key and tipo_key:
            exact_lookup[(concepto_key, tipo_key)] = {
                "concepto_canonico": str(row.get("concepto_canonico", "") or "").strip(),
                "include_in_count": include_in_count,
            }

        if canon_key and canon_key not in canonical_lookup:
            canonical_lookup[canon_key] = {
                "concepto_canonico": str(row.get("concepto_canonico", "") or "").strip(),
                "include_in_count": include_in_count,
            }

    return exact_lookup, canonical_lookup


@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    df_ = pd.read_csv(DATA_URL)
    df_.columns = [col.strip() for col in df_.columns]

    required = [COL_RESTAURANTE, COL_FECHA, COL_TOTAL]
    missing = [column for column in required if column not in df_.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df_[COL_FECHA] = pd.to_datetime(df_[COL_FECHA], errors="coerce", dayfirst=True)
    df_[COL_TOTAL] = clean_money_series(df_[COL_TOTAL]).fillna(0.0)

    if COL_SUBTOTAL not in df_.columns:
        df_[COL_SUBTOTAL] = df_[COL_TOTAL]
    else:
        df_[COL_SUBTOTAL] = clean_money_series(df_[COL_SUBTOTAL]).fillna(0.0)

    if COL_DESCUENTOS not in df_.columns:
        df_[COL_DESCUENTOS] = 0.0
    else:
        df_[COL_DESCUENTOS] = clean_money_series(df_[COL_DESCUENTOS]).fillna(0.0)

    tax_col = detect_tax_column(df_)
    if tax_col is not None:
        df_["_impuestos"] = clean_money_series(df_[tax_col]).fillna(0.0)
    else:
        df_["_impuestos"] = (df_[COL_TOTAL] - df_[COL_SUBTOTAL]).clip(lower=0.0)

    is_void = get_void_mask(df_, COL_ESTADO)
    df_["_calc_sti_d"] = (df_[COL_SUBTOTAL] + df_["_impuestos"] - df_[COL_DESCUENTOS]).fillna(0.0)
    df_["_ventas_brutas_regla"] = np.where(
        df_["_calc_sti_d"] > df_[COL_TOTAL],
        df_[COL_TOTAL],
        df_["_calc_sti_d"],
    )
    df_[COL_VENTAS] = np.where(
        is_void,
        0.0,
        pd.Series(df_["_ventas_brutas_regla"], index=df_.index).clip(lower=0.0),
    )

    if COL_FOLIO not in df_.columns:
        df_[COL_FOLIO] = np.arange(len(df_)).astype(str)
    else:
        df_[COL_FOLIO] = df_[COL_FOLIO].astype(str)

    if COL_TIPO not in df_.columns:
        df_[COL_TIPO] = ""

    if COL_DETALLE not in df_.columns:
        df_[COL_DETALLE] = ""

    return df_


@st.cache_data(ttl=600)
def build_items_flat(df: pd.DataFrame, catalogo: pd.DataFrame | None) -> pd.DataFrame:
    g = df.copy()
    g = g[g[COL_FECHA].notna()].copy()
    exact_lookup, canonical_lookup = build_catalog_lookups(catalogo)

    rows = []
    for _, row in g.iterrows():
        detalle = row.get(COL_DETALLE, "")
        for producto in split_top_level(detalle):
            base_text = producto.split("[", 1)[0].strip()
            item_raw, qty = parse_base_item(base_text)
            if not item_raw or qty <= 0:
                continue

            item_key = norm_key(item_raw)
            tipo_key = "base"
            meta = exact_lookup.get((item_key, tipo_key), canonical_lookup.get(item_key))
            include_in_count = meta.get("include_in_count", True) if isinstance(meta, dict) else True
            if not include_in_count:
                continue

            item = meta.get("concepto_canonico", item_raw) if isinstance(meta, dict) else item_raw

            rows.append(
                {
                    "Fecha": row[COL_FECHA],
                    "Restaurante": row.get(COL_RESTAURANTE, None),
                    "Estado": row.get(COL_ESTADO, None),
                    "Folio": row.get(COL_FOLIO, None),
                    "item": item,
                    "qty": qty,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Fecha", "Restaurante", "Estado", "Folio", "item", "qty"])

    return pd.DataFrame(rows)


def build_product_mix(items_cur: pd.DataFrame, items_prev: pd.DataFrame) -> pd.DataFrame:
    cur_valid = items_cur.loc[~get_void_mask(items_cur, "Estado")] if not items_cur.empty else items_cur
    prev_valid = items_prev.loc[~get_void_mask(items_prev, "Estado")] if not items_prev.empty else items_prev

    cur_mix = (
        cur_valid.groupby("item", as_index=False).agg(qty_actual=("qty", "sum"))
        if not cur_valid.empty
        else pd.DataFrame(columns=["item", "qty_actual"])
    )
    prev_mix = (
        prev_valid.groupby("item", as_index=False).agg(qty_previa=("qty", "sum"))
        if not prev_valid.empty
        else pd.DataFrame(columns=["item", "qty_previa"])
    )

    mix = cur_mix.merge(prev_mix, on="item", how="outer").fillna(0)
    if mix.empty:
        return mix

    total_cur = float(mix["qty_actual"].sum())
    total_prev = float(mix["qty_previa"].sum())
    mix["mix_actual"] = mix["qty_actual"] / total_cur if total_cur > 0 else 0.0
    mix["mix_previa"] = mix["qty_previa"] / total_prev if total_prev > 0 else 0.0
    mix["delta_mix_pp"] = (mix["mix_actual"] - mix["mix_previa"]) * 100
    mix["delta_qty"] = mix.apply(lambda row: safe_pct_change(row["qty_actual"], row["qty_previa"]), axis=1)
    return mix.sort_values("qty_actual", ascending=False)


st.sidebar.markdown("### Actualización")
if st.sidebar.button("Actualizar data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"Última vista: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

try:
    df = load_data()
except Exception as exc:
    st.error(f"No se pudo cargar la base: {exc}")
    st.stop()

if df[COL_FECHA].dropna().empty:
    st.error("No hay fechas válidas en la base.")
    st.stop()

catalogo_operativo, catalog_source = load_catalogo_operativo()
if catalogo_operativo is not None and not catalogo_operativo.empty and "compile_status" in catalogo_operativo.columns:
    catalogo_operativo = catalogo_operativo[catalogo_operativo["compile_status"].eq("ok")].copy()

items_flat = build_items_flat(df, catalogo_operativo)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filtros")

restaurants_all = sorted(df[COL_RESTAURANTE].dropna().unique().tolist())
selected_restaurants = st.sidebar.multiselect(
    "Restaurantes",
    options=restaurants_all,
    default=restaurants_all,
)

max_fecha_data = df[COL_FECHA].max().normalize()
cur_start, cur_end, prev_start, prev_end = get_current_and_previous_week_ranges(max_fecha_data)

st.sidebar.markdown("### Semana operativa")
st.sidebar.caption(f"Semana actual: {cur_start.strftime('%d/%m/%Y')} → {cur_end.strftime('%d/%m/%Y')}")
st.sidebar.caption(f"Semana previa: {prev_start.strftime('%d/%m/%Y')} → {prev_end.strftime('%d/%m/%Y')}")

col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown(
        f"""
        <div style="
            width:120px;height:120px;border-radius:60px;
            border:4px solid #A7F0E3;display:flex;align-items:center;justify-content:center;
            background:#FFFFFF;box-shadow:0 18px 45px rgba(15,23,42,0.10);">
            <img src="{LOGO_URL}" style="width:70%;height:70%;border-radius:50%;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_title:
    st.markdown(
        f"""
        <h1 style="margin-bottom:0;">Operations Control Center</h1>
        <p style="color:#6F7277;font-size:0.98rem;margin-top:0.25rem;">
        Vista ejecutiva semanal · ventas · ticket promedio · restaurantes · mix de producto
        </p>
        <p style="color:#6F7277;font-size:0.86rem;margin-top:0.4rem;">
        Semana analizada: {cur_start.strftime('%d/%m/%Y')} → {cur_end.strftime('%d/%m/%Y')} ·
        Fuente catálogo: {catalog_source}
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

df_cur = filtrar_periodo(df, cur_start, cur_end, selected_restaurants)
df_prev = filtrar_periodo(df, prev_start, prev_end, selected_restaurants)

items_cur = filtrar_periodo(items_flat, cur_start, cur_end, selected_restaurants) if not items_flat.empty else items_flat
items_prev = filtrar_periodo(items_flat, prev_start, prev_end, selected_restaurants) if not items_flat.empty else items_flat

m_cur = calcular_metricas(df_cur)
m_prev = calcular_metricas(df_prev)

delta_ventas = safe_pct_change(m_cur["ventas"], m_prev["ventas"])
delta_tickets = safe_pct_change(m_cur["tickets"], m_prev["tickets"])
delta_ticket = safe_pct_change(m_cur["ticket_promedio"], m_prev["ticket_promedio"])
delta_delivery_pp = m_cur["pct_delivery"] - m_prev["pct_delivery"] if pd.notna(m_cur["pct_delivery"]) and pd.notna(m_prev["pct_delivery"]) else np.nan
delta_cancelados = safe_pct_change(m_cur["cancelados"], m_prev["cancelados"])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ventas semana", fmt_money(m_cur["ventas"]), fmt_pct(delta_ventas))
k2.metric("Tickets", f"{m_cur['tickets']:,.0f}", fmt_pct(delta_tickets))
k3.metric("Ticket promedio", fmt_money(m_cur["ticket_promedio"]), fmt_pct(delta_ticket))
k4.metric("Delivery mix", fmt_pct(m_cur["pct_delivery"]), fmt_pp(delta_delivery_pp))
k5.metric("Cancelados", f"{m_cur['cancelados']:,.0f}", fmt_pct(delta_cancelados))

daily_cur = build_daily_sales(df_cur, "Semana actual", cur_start)
daily_prev = build_daily_sales(df_prev, "Semana previa", prev_start)
daily_prev_plot = daily_prev.copy()
daily_prev_plot["dia_fecha"] = daily_cur["dia_fecha"].values
daily_plot = pd.concat([daily_cur, daily_prev_plot], ignore_index=True)
daily_anom = detect_daily_anomalies(daily_cur)
restaurant_perf = build_restaurant_performance(df_cur, df_prev)
product_mix = build_product_mix(items_cur, items_prev)

left, right = st.columns([1.65, 1])

with left:
    st.markdown('<div class="panel-title">Sales Velocity · Ventas por día</div>', unsafe_allow_html=True)

    line = (
        alt.Chart(daily_plot)
        .mark_line(strokeWidth=3, point=True)
        .encode(
            x=alt.X("dia_nombre:N", sort=["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"], title=""),
            y=alt.Y("ventas:Q", title="Ventas"),
            color=alt.Color("periodo:N", title="Periodo"),
            tooltip=[
                alt.Tooltip("periodo:N", title="Periodo"),
                alt.Tooltip("dia_nombre:N", title="Día"),
                alt.Tooltip("ventas:Q", title="Ventas", format=",.0f"),
                alt.Tooltip("tickets:Q", title="Tickets"),
                alt.Tooltip("ticket_promedio:Q", title="Ticket promedio", format=",.2f"),
            ],
        )
        .properties(height=340)
    )

    st.altair_chart(line, use_container_width=True)

with right:
    st.markdown('<div class="panel-title">Anomalías semanales</div>', unsafe_allow_html=True)

    anomaly_rows = daily_anom[daily_anom["anomaly"]].copy()
    if anomaly_rows.empty:
        st.success("No se detectaron anomalías relevantes en ventas diarias.")
    else:
        for _, row in anomaly_rows.iterrows():
            css = "red" if row["zscore"] < 0 else "green"
            title = "Caída atípica" if row["zscore"] < 0 else "Pico atípico"
            st.markdown(
                f"""
                <div class="alert-box {css}">
                    <strong>{title}</strong><br>
                    Día: {row['dia_nombre']}<br>
                    Ventas: {fmt_money(row['ventas'])}<br>
                    Z-score: {row['zscore']:.2f}
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("## Alertas operativas")

alerts = []
if pd.notna(delta_ventas) and delta_ventas <= -0.05:
    alerts.append(("red", f"Ventas semanales cayendo {fmt_pct(delta_ventas)} vs semana previa."))
if pd.notna(delta_ticket) and delta_ticket <= -0.05:
    alerts.append(("yellow", f"Ticket promedio cayendo {fmt_pct(delta_ticket)} vs semana previa."))

if not restaurant_perf.empty:
    worst_sales = restaurant_perf.sort_values("delta_ventas", ascending=True).head(1)
    if not worst_sales.empty and pd.notna(worst_sales.iloc[0]["delta_ventas"]) and worst_sales.iloc[0]["delta_ventas"] <= -0.05:
        alerts.append(
            (
                "red",
                f"{worst_sales.iloc[0][COL_RESTAURANTE]} presenta la peor caída en ventas: {fmt_pct(worst_sales.iloc[0]['delta_ventas'])}.",
            )
        )

    worst_ticket = restaurant_perf.sort_values("delta_ticket", ascending=True).head(1)
    if not worst_ticket.empty and pd.notna(worst_ticket.iloc[0]["delta_ticket"]) and worst_ticket.iloc[0]["delta_ticket"] <= -0.05:
        alerts.append(
            (
                "yellow",
                f"{worst_ticket.iloc[0][COL_RESTAURANTE]} presenta la mayor caída en ticket promedio: {fmt_pct(worst_ticket.iloc[0]['delta_ticket'])}.",
            )
        )

if not product_mix.empty:
    top_up = product_mix.sort_values("delta_mix_pp", ascending=False).head(1)
    top_down = product_mix.sort_values("delta_mix_pp", ascending=True).head(1)
    if not top_up.empty and top_up.iloc[0]["delta_mix_pp"] > 1:
        alerts.append(("green", f"Producto ganando mix: {top_up.iloc[0]['item']} ({top_up.iloc[0]['delta_mix_pp']:+.1f}pp)."))
    if not top_down.empty and top_down.iloc[0]["delta_mix_pp"] < -1:
        alerts.append(("yellow", f"Producto perdiendo mix: {top_down.iloc[0]['item']} ({top_down.iloc[0]['delta_mix_pp']:+.1f}pp)."))

if not alerts:
    st.info("Sin alertas críticas. La operación luce estable contra la semana previa.")
else:
    for css, message in alerts:
        st.markdown(
            f"""
            <div class="alert-box {css}">
                {message}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("## Performance por restaurante")

if restaurant_perf.empty:
    st.warning("No hay datos de restaurantes para los filtros actuales.")
else:
    rest_show = restaurant_perf.copy()
    rest_show["Ventas actual"] = rest_show["ventas_actual"].map(fmt_money)
    rest_show["Ventas previa"] = rest_show["ventas_previa"].map(fmt_money)
    rest_show["Δ Ventas"] = rest_show["delta_ventas"].map(fmt_pct)
    rest_show["Ticket actual"] = rest_show["ticket_actual"].map(fmt_money)
    rest_show["Ticket previo"] = rest_show["ticket_previa"].map(fmt_money)
    rest_show["Δ Ticket"] = rest_show["delta_ticket"].map(fmt_pct)

    st.dataframe(
        rest_show[
            [
                COL_RESTAURANTE,
                "Ventas actual",
                "Ventas previa",
                "Δ Ventas",
                "Ticket actual",
                "Ticket previo",
                "Δ Ticket",
                "status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("## Heatmap de ventas por restaurante y día")

if df_cur.empty:
    st.warning("No hay datos en la semana actual para el heatmap.")
else:
    heat = df_cur.loc[~get_void_mask(df_cur, COL_ESTADO)].copy()
    heat["dia_nombre"] = heat[COL_FECHA].dt.day_name().map(
        {
            "Monday": "Lun",
            "Tuesday": "Mar",
            "Wednesday": "Mié",
            "Thursday": "Jue",
            "Friday": "Vie",
            "Saturday": "Sáb",
            "Sunday": "Dom",
        }
    )

    heat = heat.groupby([COL_RESTAURANTE, "dia_nombre"], as_index=False).agg(ventas=(COL_VENTAS, "sum"))

    heatmap = (
        alt.Chart(heat)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("dia_nombre:N", sort=["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"], title=""),
            y=alt.Y(f"{COL_RESTAURANTE}:N", sort="-x", title=""),
            color=alt.Color("ventas:Q", title="Ventas"),
            tooltip=[
                alt.Tooltip(f"{COL_RESTAURANTE}:N", title="Restaurante"),
                alt.Tooltip("dia_nombre:N", title="Día"),
                alt.Tooltip("ventas:Q", title="Ventas", format=",.0f"),
            ],
        )
        .properties(height=max(180, 40 * max(1, heat[COL_RESTAURANTE].nunique())))
    )

    st.altair_chart(heatmap, use_container_width=True)

st.markdown("## Mix de producto resumido")

if product_mix.empty:
    st.warning("No se pudo construir el mix de productos con los datos actuales.")
else:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Productos ganando participación")
        top_gain = product_mix.sort_values("delta_mix_pp", ascending=False).head(10).copy()
        top_gain["Mix actual"] = top_gain["mix_actual"].map(fmt_pct)
        top_gain["Mix previo"] = top_gain["mix_previa"].map(fmt_pct)
        top_gain["Δ Mix"] = top_gain["delta_mix_pp"].map(lambda x: f"{x:+.1f}pp")
        st.dataframe(
            top_gain[["item", "qty_actual", "qty_previa", "Mix actual", "Mix previo", "Δ Mix"]],
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.markdown("### Productos perdiendo participación")
        top_loss = product_mix.sort_values("delta_mix_pp", ascending=True).head(10).copy()
        top_loss["Mix actual"] = top_loss["mix_actual"].map(fmt_pct)
        top_loss["Mix previo"] = top_loss["mix_previa"].map(fmt_pct)
        top_loss["Δ Mix"] = top_loss["delta_mix_pp"].map(lambda x: f"{x:+.1f}pp")
        st.dataframe(
            top_loss[["item", "qty_actual", "qty_previa", "Mix actual", "Mix previo", "Δ Mix"]],
            use_container_width=True,
            hide_index=True,
        )

    top_products_chart = (
        alt.Chart(product_mix.head(15))
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("item:N", sort="-x", title=""),
            x=alt.X("qty_actual:Q", title="Cantidad actual"),
            tooltip=[
                alt.Tooltip("item:N", title="Producto"),
                alt.Tooltip("qty_actual:Q", title="Qty actual"),
                alt.Tooltip("qty_previa:Q", title="Qty previa"),
                alt.Tooltip("delta_mix_pp:Q", title="Δ mix pp", format=".2f"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(top_products_chart, use_container_width=True)

st.markdown("---")
st.markdown(
    f"""
    <div class="small-muted">
        Base: Google Sheets · Cálculo principal: ventas efectivas · Semana operativa: lunes a domingo ·
        Fecha máxima detectada: {max_fecha_data.strftime('%d/%m/%Y')}
    </div>
    """,
    unsafe_allow_html=True,
)
