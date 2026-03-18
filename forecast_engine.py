from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from catalog_engine import compile_catalog, norm_key


DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSLIeswEs8OILxZmVMwObbli0Zpbbqx7g7h6ZC5Fwm0PCjlZEFy66L9Xpha6ROW3loFCIRiWvEnLRHS/pub?output=csv"
CATALOGO_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQtKQGyCaerGAedhlpzaXlr-ycmm1t08a6lUtg-_3f7yWtJhLkQ6vn0TlI89l0FGVxOUy1Cwj5ykliB/pub?output=csv"

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


@dataclass(frozen=True)
class ForecastConfig:
    daily_horizon: int = 90
    recent_window: int = 28
    short_window: int = 7
    weekday_lookback: int = 8
    trend_window: int = 14
    holdout_days: int = 28


def clean_money_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    series = series.astype(str).str.strip()
    series = series.replace({"": np.nan, "—": np.nan, "nan": np.nan, "None": np.nan})
    series = series.str.replace(r"[\$,]", "", regex=True)
    series = series.str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(series, errors="coerce")


def detect_tax_column(df_: pd.DataFrame) -> str | None:
    candidates = ["Impuestos", "IVA", "Tax", "Taxes", "Impuesto", "VAT"]
    for column in candidates:
        if column in df_.columns:
            return column
    return None


def get_void_mask(df_: pd.DataFrame, col_estado: str = COL_ESTADO) -> pd.Series:
    if col_estado in df_.columns:
        return df_[col_estado].astype(str).str.strip().str.lower().eq("void")
    return pd.Series(False, index=df_.index)


def split_top_level(texto: str, separator: str = "|") -> list[str]:
    if not isinstance(texto, str):
        return []

    parts = []
    buffer = []
    bracket_depth = 0

    for char in texto:
        if char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth > 0:
            bracket_depth -= 1

        if char == separator and bracket_depth == 0:
            part = "".join(buffer).strip()
            if part:
                parts.append(part)
            buffer = []
            continue

        buffer.append(char)

    part = "".join(buffer).strip()
    if part:
        parts.append(part)
    return parts


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


def parse_base_item_with_price(raw: str) -> tuple[str, float, float | None, float | None]:
    text = str(raw).strip()
    if not text:
        return "", 0.0, None, None

    line_price = None
    match_price = PRICE_RE.search(text)
    if match_price:
        value = match_price.group(1).replace(",", "")
        try:
            line_price = float(value)
        except ValueError:
            line_price = None
        text = text[:match_price.start()].strip()

    qty = 1.0
    match_qty = QTY_RE.search(text)
    if match_qty:
        qty = parse_qty(match_qty.group(1))
        text = text[:match_qty.start()].strip()

    name = text.strip()
    if not name:
        return "", 0.0, None, line_price

    unit_price = None
    if line_price is not None and qty > 0:
        unit_price = line_price / qty

    return name, qty, unit_price, line_price


def load_catalogo_operativo(base_dir: Path) -> tuple[pd.DataFrame | None, str]:
    local_path = base_dir / "data" / "catalogo_operativo.csv"
    if local_path.exists():
        try:
            catalog = pd.read_csv(local_path)
            if "compile_status" in catalog.columns:
                catalog = catalog[catalog["compile_status"].eq("ok")].copy()
            return catalog, f"Local compilado: {local_path.name}"
        except Exception:
            pass

    try:
        raw_catalog = pd.read_csv(CATALOGO_URL)
    except Exception:
        return None, "Sin catálogo"

    compiled, _ = compile_catalog(raw_catalog)
    return compiled[compiled["compile_status"].eq("ok")].copy(), "Compilado al vuelo desde Google Sheets"


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


def load_sales_data() -> pd.DataFrame:
    df_ = pd.read_csv(DATA_URL)
    df_.columns = [column.strip() for column in df_.columns]

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

    is_void = get_void_mask(df_)
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

    if COL_DETALLE not in df_.columns:
        df_[COL_DETALLE] = ""

    return df_


def build_product_base_facts(df: pd.DataFrame, catalogo: pd.DataFrame | None) -> pd.DataFrame:
    exact_lookup, canonical_lookup = build_catalog_lookups(catalogo)

    rows = []
    base_df = df[df[COL_FECHA].notna()].copy()

    for _, row in base_df.iterrows():
        detalle = row.get(COL_DETALLE, "")
        if not isinstance(detalle, str) or not detalle.strip():
            continue

        for product in split_top_level(detalle):
            base_text = product.split("[", 1)[0].strip()
            item_raw, qty, unit_price, line_sales = parse_base_item_with_price(base_text)
            if not item_raw or qty <= 0:
                continue

            item_key = norm_key(item_raw)
            meta = exact_lookup.get((item_key, "base"), canonical_lookup.get(item_key))
            include_in_count = meta.get("include_in_count", True) if isinstance(meta, dict) else True
            if not include_in_count:
                continue

            item = meta.get("concepto_canonico", item_raw) if isinstance(meta, dict) else item_raw

            rows.append(
                {
                    "fecha": row[COL_FECHA].normalize(),
                    "restaurante": row.get(COL_RESTAURANTE, None),
                    "folio": row.get(COL_FOLIO, None),
                    "estado": row.get(COL_ESTADO, None),
                    "producto_raw": item_raw,
                    "producto": item,
                    "qty": qty,
                    "line_sales": line_sales,
                    "unit_price": unit_price,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["fecha", "restaurante", "folio", "estado", "producto_raw", "producto", "qty", "line_sales", "unit_price"]
        )

    facts = pd.DataFrame(rows)
    non_void = ~get_void_mask(facts, "estado")
    facts = facts.loc[non_void].copy()

    median_prices = (
        facts.dropna(subset=["unit_price"])
        .groupby(["restaurante", "producto"], as_index=False)
        .agg(median_unit_price=("unit_price", "median"))
    )
    global_prices = (
        facts.dropna(subset=["unit_price"])
        .groupby("producto", as_index=False)
        .agg(global_unit_price=("unit_price", "median"))
    )

    facts = facts.merge(median_prices, on=["restaurante", "producto"], how="left")
    facts = facts.merge(global_prices, on="producto", how="left")
    facts["unit_price_filled"] = facts["unit_price"].fillna(facts["median_unit_price"]).fillna(facts["global_unit_price"])
    facts["sales"] = facts["line_sales"].fillna(facts["qty"] * facts["unit_price_filled"]).fillna(0.0)
    return facts


def build_daily_product_fact(facts: pd.DataFrame) -> pd.DataFrame:
    if facts.empty:
        return pd.DataFrame(columns=["fecha", "restaurante", "producto", "qty", "sales", "tickets", "unit_price_realized"])

    daily = (
        facts.groupby(["fecha", "restaurante", "producto"], as_index=False)
        .agg(
            qty=("qty", "sum"),
            sales=("sales", "sum"),
            tickets=("folio", "nunique"),
        )
    )
    daily["unit_price_realized"] = np.where(daily["qty"] > 0, daily["sales"] / daily["qty"], np.nan)
    return daily


def _blend_forecast_value(series: pd.Series, target_date: pd.Timestamp, config: ForecastConfig) -> float:
    weekday_history = series[series.index.dayofweek == target_date.dayofweek].tail(config.weekday_lookback)
    recent_short = series.tail(config.short_window)
    recent_long = series.tail(config.recent_window)

    components = []
    weights = []
    if not weekday_history.empty:
        components.append(float(weekday_history.mean()))
        weights.append(0.5)
    if not recent_short.empty:
        components.append(float(recent_short.mean()))
        weights.append(0.3)
    if not recent_long.empty:
        components.append(float(recent_long.mean()))
        weights.append(0.2)

    if not components:
        return 0.0

    base = float(np.average(components, weights=weights))

    recent_trend = series.tail(config.trend_window)
    previous_trend = series.tail(config.trend_window * 2).head(config.trend_window)
    if not recent_trend.empty and not previous_trend.empty and previous_trend.mean() > 0:
        trend_rate = (recent_trend.mean() - previous_trend.mean()) / previous_trend.mean()
        trend_factor = 1 + np.clip(trend_rate, -0.25, 0.25) * 0.35
    else:
        trend_factor = 1.0

    return max(base * trend_factor, 0.0)


def forecast_series(history: pd.Series, horizon: int, config: ForecastConfig) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["fecha", "forecast"])

    history = history.sort_index().asfreq("D", fill_value=0.0)
    working = history.astype(float).copy()
    last_date = working.index.max()
    rows = []

    for step in range(1, horizon + 1):
        target_date = last_date + pd.Timedelta(days=step)
        forecast = _blend_forecast_value(working, target_date, config)
        rows.append({"fecha": target_date, "forecast": forecast})
        working.loc[target_date] = forecast

    return pd.DataFrame(rows)


def backtest_series(history: pd.Series, config: ForecastConfig) -> tuple[pd.DataFrame, dict]:
    if history.empty:
        return pd.DataFrame(columns=["fecha", "actual", "forecast"]), {"wape": np.nan, "mae": np.nan}

    history = history.sort_index().asfreq("D", fill_value=0.0)
    holdout_days = min(config.holdout_days, max(7, len(history) // 5))
    if len(history) <= holdout_days + 7:
        return pd.DataFrame(columns=["fecha", "actual", "forecast"]), {"wape": np.nan, "mae": np.nan}

    train = history.iloc[:-holdout_days].copy()
    actual = history.iloc[-holdout_days:].copy()
    forecast_df = forecast_series(train, holdout_days, config)
    forecast_df = forecast_df.set_index("fecha").reindex(actual.index).fillna(0.0).reset_index()
    forecast_df.columns = ["fecha", "forecast"]
    backtest = forecast_df.copy()
    backtest["actual"] = actual.values
    backtest["abs_error"] = (backtest["actual"] - backtest["forecast"]).abs()

    denominator = backtest["actual"].abs().sum()
    wape = backtest["abs_error"].sum() / denominator if denominator > 0 else np.nan
    mae = backtest["abs_error"].mean() if not backtest.empty else np.nan
    return backtest[["fecha", "actual", "forecast"]], {"wape": wape, "mae": mae}


def build_forecast_for_metric(
    daily_fact: pd.DataFrame,
    metric: str,
    restaurants: list[str],
    products: list[str],
    config: ForecastConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = daily_fact.copy()
    if restaurants:
        filtered = filtered[filtered["restaurante"].isin(restaurants)].copy()
    if products:
        filtered = filtered[filtered["producto"].isin(products)].copy()

    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame()

    series_df = (
        filtered.groupby(["fecha", "producto"], as_index=False)[metric]
        .sum()
        .sort_values(["producto", "fecha"])
    )

    actual_rows = []
    forecast_rows = []

    for product, group in series_df.groupby("producto"):
        history = group.set_index("fecha")[metric].sort_index()
        history = history.asfreq("D", fill_value=0.0)
        for date_value, actual_value in history.items():
            actual_rows.append(
                {
                    "fecha": date_value,
                    "producto": product,
                    "tipo": "actual",
                    "valor": float(actual_value),
                }
            )

        backtest_df, diagnostics = backtest_series(history, config)
        forecast_df = forecast_series(history, config.daily_horizon, config)
        mae = diagnostics.get("mae", np.nan)
        interval = 1.28 * mae if pd.notna(mae) else np.nan

        for _, row in forecast_df.iterrows():
            lower = max(row["forecast"] - interval, 0.0) if pd.notna(interval) else np.nan
            upper = row["forecast"] + interval if pd.notna(interval) else np.nan
            forecast_rows.append(
                {
                    "fecha": row["fecha"],
                    "producto": product,
                    "tipo": "forecast",
                    "valor": float(row["forecast"]),
                    "lower": lower,
                    "upper": upper,
                    "wape": diagnostics.get("wape", np.nan),
                    "mae": diagnostics.get("mae", np.nan),
                }
            )

    return pd.DataFrame(actual_rows), pd.DataFrame(forecast_rows)


def aggregate_projection(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["fecha"] = pd.to_datetime(out["fecha"])

    if granularity == "Día":
        out["periodo"] = out["fecha"].dt.to_period("D").dt.to_timestamp()
        out["periodo_label"] = out["periodo"].dt.strftime("%d %b %Y")
    elif granularity == "Semana":
        out["periodo"] = out["fecha"].dt.to_period("W-MON").apply(lambda period: period.start_time)
        out["periodo_label"] = out["periodo"].dt.strftime("Sem %d %b %Y")
    else:
        out["periodo"] = out["fecha"].dt.to_period("M").dt.to_timestamp()
        out["periodo_label"] = out["periodo"].dt.strftime("%b %Y")

    group_cols = ["periodo", "periodo_label", "producto", "tipo"]
    agg = (
        out.groupby(group_cols, as_index=False)
        .agg(
            valor=("valor", "sum"),
            lower=("lower", "sum") if "lower" in out.columns else ("valor", "sum"),
            upper=("upper", "sum") if "upper" in out.columns else ("valor", "sum"),
            wape=("wape", "mean") if "wape" in out.columns else ("valor", "mean"),
        )
    )
    return agg.sort_values(["periodo", "producto", "tipo"])


def build_projection_table(actual: pd.DataFrame, forecast: pd.DataFrame, granularity: str) -> pd.DataFrame:
    actual_agg = aggregate_projection(actual, granularity)
    forecast_agg = aggregate_projection(forecast, granularity)
    combined = pd.concat([actual_agg, forecast_agg], ignore_index=True)
    if combined.empty:
        return combined

    return combined[
        ["periodo", "periodo_label", "producto", "tipo", "valor", "lower", "upper", "wape"]
    ].sort_values(["periodo", "producto", "tipo"])
