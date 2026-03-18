# pages/04_Ventas_por_Producto.py

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
from pathlib import Path
import re
import unicodedata

from catalog_engine import compile_catalog

# ============= CONFIG BÁSICA =============
st.set_page_config(
    page_title="Ventas por Producto – Marcas HP",
    page_icon="🧾",
    layout="wide",
)

# ===== Tema de Altair =====
def byf_altair_theme():
    return {
        "config": {
            "background": "rgba(0,0,0,0)",
            "view": {"stroke": "transparent"},
            "axis": {"labelColor": "#1B1D22", "titleColor": "#1B1D22"},
            "legend": {"labelColor": "#1B1D22", "titleColor": "#1B1D22"},
            "range": {
                "category": [
                    "#1B1D22",
                    "#7AD9CF",
                    "#A7F0E3",
                    "#B8EDEA",
                    "#6F7277",
                    "#37D2A3",
                ],
            },
        }
    }

alt.themes.register("byf_theme", byf_altair_theme)
alt.themes.enable("byf_theme")

# ===== Estilos =====
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 80% 10%, #A7F0E3 0, #F5F8F9 40%),
                    radial-gradient(circle at 0% 80%, #B8EDEA 0, #F5F8F9 45%),
                    #F5F8F9;
    }
    [data-testid="stHeader"] { background: transparent; }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .stDataFrame {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 18px;
        padding: 0.3rem 0.3rem 0.8rem 0.3rem;
        box-shadow:0 14px 32px rgba(15, 23, 42, 0.12);
        border:1px solid rgba(148, 163, 184, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# URLs
# =========================================================
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSLIeswEs8OILxZmVMwObbli0Zpbbqx7g7h6ZC5Fwm0PCjlZEFy66L9Xpha6ROW3loFCIRiWvEnLRHS/pub?output=csv"
CATALOGO_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQtKQGyCaerGAedhlpzaXlr-ycmm1t08a6lUtg-_3f7yWtJhLkQ6vn0TlI89l0FGVxOUy1Cwj5ykliB/pub?output=csv"
BASE_DIR = Path(__file__).resolve().parents[1]
CATALOGO_OPERATIVO_PATH = BASE_DIR / "data" / "catalogo_operativo.csv"
CATALOGO_ISSUES_PATH = BASE_DIR / "data" / "catalogo_issues.csv"
CATALOGO_REVIEW_QUEUE_PATH = BASE_DIR / "data" / "catalog_review_queue.csv"

# =========================================================
# Helpers
# =========================================================
def is_delivery(val):
    try:
        return "delivery" in str(val).lower()
    except Exception:
        return False

def agregar_periodo(df_src: pd.DataFrame, gran: str, col_fecha: str) -> pd.DataFrame:
    g = df_src.copy()
    if gran == "Día":
        g["periodo"] = g[col_fecha].dt.to_period("D").dt.to_timestamp()
    elif gran == "Semana":
        g["periodo"] = g[col_fecha].dt.to_period("W-MON").apply(lambda r: r.start_time)
    else:  # Mes
        g["periodo"] = g[col_fecha].dt.to_period("M").dt.to_timestamp()
    return g

def clean_money_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "—": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def norm_key(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )
    s = re.sub(r"[^\w\s\/\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\-+", "-", s).strip()
    return s

def get_void_mask(df_: pd.DataFrame, col_estado: str) -> pd.Series:
    if col_estado in df_.columns:
        return df_[col_estado].astype(str).str.strip().str.lower().eq("void")
    return pd.Series(False, index=df_.index)

# =========================================================
# Parser (idéntico al tuyo)
# =========================================================
PRICE_RE = re.compile(r"\(\s*\$?\s*(-?[\d\.,]+)\s*\)")
QTY_RE = re.compile(r"\s+[xX]\s*([\d\.]+)\s*$")

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

def extract_bracket_groups(texto: str) -> list[str]:
    if not isinstance(texto, str):
        return []

    grupos = []
    buffer = []
    bracket_depth = 0

    for char in texto:
        if char == "[":
            if bracket_depth == 0:
                buffer = []
            else:
                buffer.append(char)
            bracket_depth += 1
            continue

        if char == "]" and bracket_depth > 0:
            bracket_depth -= 1
            if bracket_depth == 0:
                grupo = "".join(buffer).strip()
                if grupo:
                    grupos.append(grupo)
                buffer = []
                continue

        if bracket_depth > 0:
            buffer.append(char)

    return grupos

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


def build_catalog_lookups(catalogo: pd.DataFrame) -> tuple[dict, dict]:
    if catalogo is None or catalogo.empty:
        return {}, {}

    exact_lookup = {}
    canonical_lookup = {}

    for row in catalogo.to_dict("records"):
        concepto_key = row.get("concepto_key", "")
        tipo_key = row.get("tipo_concepto_key", "")
        canon_key = row.get("canon_key", "")
        es_remap = bool(row.get("es_remap", False))
        include_in_count = bool(row.get("include_in_count", True))

        if concepto_key and tipo_key:
            exact_lookup[(concepto_key, tipo_key)] = {
                "concepto_canonico": str(row.get("concepto_canonico", "") or "").strip(),
                "es_remap": es_remap,
                "include_in_count": include_in_count,
                "rule_action": str(row.get("rule_action", "") or "").strip(),
            }

        if canon_key and canon_key not in canonical_lookup:
            canonical_lookup[canon_key] = {
                "concepto_canonico": str(row.get("concepto_canonico", "") or "").strip(),
                "include_in_count": include_in_count,
                "rule_action": "canonical_fallback",
            }

    return exact_lookup, canonical_lookup

def _parse_base_item(raw: str):
    txt = raw.strip()
    if not txt:
        return "", 0.0, None, np.nan

    precio_linea = None
    m_precio = PRICE_RE.search(txt)
    if m_precio:
        num = m_precio.group(1).replace(",", "")
        try:
            precio_linea = float(num)
        except ValueError:
            precio_linea = None
        txt = txt[:m_precio.start()].strip()

    qty = 1.0
    qty_raw = 1.0
    m_qty = QTY_RE.search(txt)
    if m_qty:
        qty_raw = pd.to_numeric(m_qty.group(1), errors="coerce")
        qty = parse_qty(m_qty.group(1))
        txt = txt[:m_qty.start()].strip()

    nombre = txt.strip()
    if not nombre:
        return "", 0.0, None, qty_raw

    precio_unit = None
    if precio_linea is not None and qty > 0:
        precio_unit = precio_linea / qty

    return nombre, qty, precio_unit, qty_raw

def parse_detalle_items_base_y_complementos(texto: str):
    registros = []
    if not isinstance(texto, str) or not texto.strip():
        return registros

    productos_principales = split_top_level(texto)

    for producto in productos_principales:
        base_texto = producto.split("[", 1)[0].strip()
        complementos_texto = extract_bracket_groups(producto)

        nombre_base, qty_base, precio_unit, qty_original = _parse_base_item(base_texto)
        if nombre_base and qty_base > 0:
            registros.append({
                "item": nombre_base,
                "qty": qty_base,
                "qty_original": qty_original,
                "precio_unitario": precio_unit,
                "tipo_concepto": "base",
            })

        if not complementos_texto or qty_base <= 0:
            continue

        for grupo in complementos_texto:
            complementos = [c.strip() for c in grupo.split(",") if c.strip()]
            for comp in complementos:
                comp_limpio = comp.lstrip("+").strip()
                if comp_limpio:
                    registros.append({
                        "item": comp_limpio,
                        "qty": qty_base,
                        "qty_original": qty_original,
                        "precio_unitario": None,
                        "tipo_concepto": "complemento",
                    })

    return registros

# =========================================================
# Carga de datos y catálogo
# =========================================================
@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    df_ = pd.read_csv(DATA_URL)
    df_.columns = [c.strip() for c in df_.columns]
    return df_

@st.cache_data(ttl=600)
def load_catalogo() -> tuple[pd.DataFrame | None, pd.DataFrame, str]:
    if CATALOGO_OPERATIVO_PATH.exists():
        try:
            catalogo = pd.read_csv(CATALOGO_OPERATIVO_PATH)
            issues = (
                pd.read_csv(CATALOGO_ISSUES_PATH)
                if CATALOGO_ISSUES_PATH.exists()
                else pd.DataFrame()
            )
            return catalogo, issues, f"Local compilado: {CATALOGO_OPERATIVO_PATH.name}"
        except Exception as e:
            st.warning(f"No se pudo leer el catálogo operativo local: {e}")

    try:
        cat = pd.read_csv(CATALOGO_URL)
    except Exception as e:
        st.warning(f"No se pudo cargar el catálogo: {e}")
        return None, pd.DataFrame(), "Sin catálogo"

    compiled, issues = compile_catalog(cat)
    return compiled, issues, "Compilado al vuelo desde Google Sheets"


@st.cache_data(ttl=600)
def load_review_queue() -> pd.DataFrame:
    if not CATALOGO_REVIEW_QUEUE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CATALOGO_REVIEW_QUEUE_PATH)

# =========================================================
# Construcción de tabla flat con canonización (cache)
# =========================================================
@st.cache_data(ttl=600)
def build_items_flat(df: pd.DataFrame, catalogo: pd.DataFrame | None) -> pd.DataFrame:
    COL_CC = "Restaurante"
    COL_ESTADO = "Estado"
    COL_FECHA = "Fecha"
    COL_FOLIO = "Folio"
    COL_DETALLE = "Detalle Items"

    g = df.copy()
    g[COL_FECHA] = pd.to_datetime(g[COL_FECHA], errors="coerce", dayfirst=True)
    g = g[g[COL_FECHA].notna()].copy()

    # Flatten por fila (ticket)
    rows = []
    for _, r in g.iterrows():
        detalle = r.get(COL_DETALLE, "")
        regs = parse_detalle_items_base_y_complementos(detalle)
        if not regs:
            continue
        for it in regs:
            rows.append({
                "Fecha": r[COL_FECHA],
                "Restaurante": r.get(COL_CC, None),
                "Estado": r.get(COL_ESTADO, None),
                "Folio": r.get(COL_FOLIO, None),
                "tipo_concepto": it["tipo_concepto"],
                "item_raw": it["item"],
                "qty": it["qty"],
                "qty_original": it["qty_original"],
            })

    if not rows:
        return pd.DataFrame(columns=["Fecha","Restaurante","Estado","Folio","tipo_concepto","item_raw","qty","qty_original","item","mapping_status"])

    flat = pd.DataFrame(rows)
    flat["item_key"] = flat["item_raw"].map(norm_key)
    flat["tipo_concepto_key"] = flat["tipo_concepto"].astype(str).str.strip().str.lower()

    if catalogo is None or catalogo.empty:
        flat["item"] = flat["item_raw"].astype(str)
        flat["mapping_status"] = "Sin mapear"
        return flat

    exact_lookup, canonical_lookup = build_catalog_lookups(catalogo)
    j = flat.copy()
    j["lookup_key"] = list(zip(j["item_key"], j["tipo_concepto_key"]))
    j["_catalog_meta"] = j["lookup_key"].map(exact_lookup)
    missing_mask = j["_catalog_meta"].isna()
    j.loc[missing_mask, "_catalog_meta"] = j.loc[missing_mask, "item_key"].map(canonical_lookup)
    j["concepto_canonico"] = j["_catalog_meta"].map(
        lambda meta: meta.get("concepto_canonico") if isinstance(meta, dict) else None
    )
    j["include_in_count"] = j["_catalog_meta"].map(
        lambda meta: meta.get("include_in_count", True) if isinstance(meta, dict) else True
    )
    j["rule_action"] = j["_catalog_meta"].map(
        lambda meta: meta.get("rule_action", "") if isinstance(meta, dict) else ""
    )

    j["item"] = j["concepto_canonico"].fillna(j["item_raw"]).astype(str).str.strip()
    j["mapping_status"] = np.where(j["_catalog_meta"].notna(), "Mapeado", "Sin mapear")
    j = j[j["include_in_count"].fillna(True)].copy()

    keep = ["Fecha","Restaurante","Estado","Folio","tipo_concepto","item_raw","item","qty","qty_original","mapping_status"]
    return j[keep]

# =========================================================
# UI
# =========================================================
st.sidebar.markdown("### Actualización")
if st.sidebar.button("🔄 Actualizar data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"Última vista: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

df = load_data()
catalogo, catalogo_issues, catalogo_source = load_catalogo()
review_queue = load_review_queue()
catalogo_ok = (
    catalogo[catalogo["compile_status"].eq("ok")].copy()
    if catalogo is not None and not catalogo.empty and "compile_status" in catalogo.columns
    else catalogo
)
flat = build_items_flat(df, catalogo_ok)

st.markdown("## Ventas por Producto (conteo)")

if flat.empty:
    st.info("No hay items para analizar (revisa Detalle Items / rango de fechas).")
    st.stop()

# Filtros generales
min_f = flat["Fecha"].min().date()
max_f = flat["Fecha"].max().date()

c1, c2, c3 = st.columns([2,2,2])
with c1:
    rango = st.date_input("Rango de fechas", value=(min_f, max_f))
with c2:
    granularidad = st.radio("Periodo", ["Día","Semana","Mes"], index=2, horizontal=True)
with c3:
    incluir_void = st.toggle("Incluir VOID en conteo", value=False)

if isinstance(rango, (list, tuple)) and len(rango) == 2:
    f_ini = pd.to_datetime(rango[0])
    f_fin = pd.to_datetime(rango[1])
else:
    f_ini = pd.to_datetime(rango)
    f_fin = pd.to_datetime(rango)

mask = (flat["Fecha"].dt.date >= f_ini.date()) & (flat["Fecha"].dt.date <= f_fin.date())
flat_f = flat[mask].copy()

if not incluir_void:
    is_void = flat_f["Estado"].astype(str).str.strip().str.lower().eq("void")
    flat_f = flat_f[~is_void].copy()

# Filtro restaurante (opcional)
rests = sorted([x for x in flat_f["Restaurante"].dropna().unique().tolist()])
sel_rests = st.multiselect("Restaurantes", options=rests, default=rests)
if sel_rests:
    flat_f = flat_f[flat_f["Restaurante"].isin(sel_rests)].copy()

with st.expander("Diagnóstico de catálogo", expanded=False):
    st.caption(catalogo_source)

    if catalogo is None or catalogo.empty:
        st.warning("No hay catálogo disponible.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reglas compiladas", f"{len(catalogo):,}")
        c2.metric("Reglas listas", f"{len(catalogo_ok) if catalogo_ok is not None else 0:,}")
        c3.metric("En revisión", f"{len(catalogo_issues):,}")
        c4.metric(
            "Canónicos",
            f"{catalogo_ok['concepto_canonico'].nunique():,}" if catalogo_ok is not None and not catalogo_ok.empty else "0",
        )

        if not catalogo_issues.empty:
            st.caption("Reglas del catálogo que necesitan revisión")
            issue_cols = [
                col for col in [
                    "concepto",
                    "tipo_concepto",
                    "concepto_canonico",
                    "rule_action",
                    "issue_reason",
                    "duplicate_count",
                    "source_rows",
                ]
                if col in catalogo_issues.columns
            ]
            st.dataframe(
                catalogo_issues[issue_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )

        if not review_queue.empty:
            st.caption("Conceptos detectados en ventas que aún no están resueltos en el catálogo")
            q1, q2 = st.columns(2)
            q1.metric("Pendientes", f"{len(review_queue):,}")
            q2.metric(
                "Revisión manual",
                f"{int(review_queue['needs_manual_review'].fillna(False).sum()):,}"
                if "needs_manual_review" in review_queue.columns
                else "0",
            )

            queue_cols = [
                col for col in [
                    "item_raw",
                    "tipo_concepto",
                    "qty_total",
                    "tickets",
                    "suggested_canonical",
                    "suggestion_score",
                    "needs_manual_review",
                ]
                if col in review_queue.columns
            ]
            st.dataframe(
                review_queue[queue_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )

with st.expander("Diagnóstico de lectura", expanded=False):
    sin_mapear = flat_f[flat_f["mapping_status"].eq("Sin mapear")].copy()
    qty_anomala = flat_f[
        flat_f["qty_original"].notna() & ~np.isclose(flat_f["qty_original"] % 1, 0.0)
    ].copy()

    d1, d2, d3 = st.columns(3)
    d1.metric("Conceptos leídos", f"{len(flat_f):,}")
    d2.metric("Productos canónicos", f"{flat_f['item'].nunique():,}")
    d3.metric("Sin mapear", f"{sin_mapear['item_raw'].nunique():,}")

    if not sin_mapear.empty:
        st.caption("Conceptos sin mapear detectados en el rango actual")
        st.dataframe(
            sin_mapear.groupby("item_raw", as_index=False)
            .agg(conteo=("qty", "sum"))
            .sort_values("conteo", ascending=False)
            .head(25),
            use_container_width=True,
            hide_index=True,
        )

    if not qty_anomala.empty:
        st.caption("Cantidades atípicas detectadas y normalizadas a 1")
        st.dataframe(
            qty_anomala[["Fecha", "Restaurante", "Folio", "item_raw", "qty_original", "qty"]]
            .sort_values("Fecha", ascending=False)
            .head(25),
            use_container_width=True,
            hide_index=True,
        )

# Selector de productos (canónicos)
items = sorted([x for x in flat_f["item"].dropna().unique().tolist()])
default_pick = items[:1] if items else []
sel_items = st.multiselect("Productos (canónicos)", options=items, default=default_pick)

if not sel_items:
    st.info("Selecciona al menos 1 producto.")
    st.stop()

# Preparar serie
flat_f = flat_f[flat_f["item"].isin(sel_items)].copy()
flat_f = agregar_periodo(flat_f, granularidad, "Fecha")

serie = (
    flat_f.groupby(["periodo","item"], as_index=False)
    .agg(conteo=("qty","sum"))
    .sort_values(["periodo","item"])
)

st.markdown("### Conteo en el tiempo")

serie = serie.sort_values("periodo").copy()

if granularidad == "Día":
    serie["periodo_str"] = pd.to_datetime(serie["periodo"]).dt.strftime("%d %b %Y")
elif granularidad == "Semana":
    serie["periodo_str"] = pd.to_datetime(serie["periodo"]).dt.strftime("Sem %d %b %Y")
else:
    serie["periodo_str"] = pd.to_datetime(serie["periodo"]).dt.strftime("%b %Y")

orden_periodos = (
    serie[["periodo", "periodo_str"]]
    .drop_duplicates()
    .sort_values("periodo")["periodo_str"]
    .tolist()
)

base = (
    alt.Chart(serie)
    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    .encode(
        x=alt.X(
            "periodo_str:N",
            title=None,
            sort=orden_periodos,
            axis=alt.Axis(labelAngle=-45, labelLimit=140, ticks=False),
        ),
        y=alt.Y("conteo:Q", title="Unidades", scale=alt.Scale(zero=True)),
        tooltip=[
            alt.Tooltip("periodo_str:N", title=granularidad),
            alt.Tooltip("item:N", title="Producto"),
            alt.Tooltip("conteo:Q", title="Unidades", format=",.0f"),
        ],
    )
    .properties(height=140)  # ✅ el height va aquí
)

ch = (
    base.facet(
        row=alt.Row("item:N", title=None, header=alt.Header(labelAngle=0, labelPadding=6))
    )
    .configure_view(stroke=None)
    .configure_facet(spacing=8)
)

st.altair_chart(ch, use_container_width=True)


# Tabla pivote (opcional)
st.markdown("### Tabla (periodo × producto)")
pivot = serie.pivot_table(index="periodo", columns="item", values="conteo", aggfunc="sum", fill_value=0).reset_index()
st.dataframe(pivot, use_container_width=True)

# Top restaurantes para el/los productos (útil cuando seleccionas varios)
st.markdown("### Top restaurantes (en el rango)")
top_rest = (
    flat_f.groupby(["Restaurante","item"], as_index=False)
    .agg(conteo=("qty","sum"))
    .sort_values(["conteo"], ascending=False)
)
st.dataframe(top_rest, use_container_width=True)
