#!/usr/bin/env python3

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog_engine import norm_key


DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSLIeswEs8OILxZmVMwObbli0Zpbbqx7g7h6ZC5Fwm0PCjlZEFy66L9Xpha6ROW3loFCIRiWvEnLRHS/pub?output=csv"
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


def parse_base_item(raw: str) -> tuple[str, float]:
    texto = str(raw).strip()
    if not texto:
        return "", 0.0

    match_price = PRICE_RE.search(texto)
    if match_price:
        texto = texto[:match_price.start()].strip()

    qty = 1.0
    match_qty = QTY_RE.search(texto)
    if match_qty:
        qty = parse_qty(match_qty.group(1))
        texto = texto[:match_qty.start()].strip()

    nombre = texto.strip()
    return nombre, qty


def parse_detail_items(texto: str) -> list[dict]:
    registros = []
    if not isinstance(texto, str) or not texto.strip():
        return registros

    for producto in split_top_level(texto):
        base_texto = producto.split("[", 1)[0].strip()
        complementos = extract_bracket_groups(producto)

        nombre_base, qty_base = parse_base_item(base_texto)
        if nombre_base and qty_base > 0:
            registros.append({
                "item_raw": nombre_base,
                "tipo_concepto": "base",
                "qty": qty_base,
            })

        if not complementos or qty_base <= 0:
            continue

        for grupo in complementos:
            for comp in [c.strip() for c in grupo.split(",") if c.strip()]:
                comp_limpio = comp.lstrip("+").strip()
                if comp_limpio:
                    registros.append({
                        "item_raw": comp_limpio,
                        "tipo_concepto": "complemento",
                        "qty": qty_base,
                    })

    return registros


def get_void_mask(df: pd.DataFrame, column: str = "Estado") -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str).str.strip().str.lower().eq("void")


def build_sales_items(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    if "Fecha" in g.columns:
        g["Fecha"] = pd.to_datetime(g["Fecha"], errors="coerce", dayfirst=True)
    rows = []

    for _, row in g.iterrows():
        regs = parse_detail_items(row.get("Detalle Items", ""))
        for reg in regs:
            rows.append({
                "Fecha": row.get("Fecha"),
                "Restaurante": row.get("Restaurante"),
                "Estado": row.get("Estado"),
                "Folio": row.get("Folio"),
                "item_raw": reg["item_raw"],
                "tipo_concepto": reg["tipo_concepto"],
                "qty": reg["qty"],
                "item_key": norm_key(reg["item_raw"]),
                "tipo_concepto_key": norm_key(reg["tipo_concepto"]),
            })

    return pd.DataFrame(rows)


def score_suggestion(item_key: str, candidate_key: str) -> float:
    return SequenceMatcher(None, item_key, candidate_key).ratio()


def suggest_best_match(item_key: str, tipo_key: str, catalogo: pd.DataFrame) -> tuple[str, float]:
    same_type = catalogo[catalogo["tipo_concepto_key"] == tipo_key].copy()
    if same_type.empty:
        same_type = catalogo.copy()

    same_type = same_type.drop_duplicates(subset=["canon_key"])
    if same_type.empty:
        return "", 0.0

    same_type["score"] = same_type["canon_key"].map(lambda candidate: score_suggestion(item_key, candidate))
    best = same_type.sort_values(["score", "conteo_total"], ascending=[False, False]).head(1)
    if best.empty:
        return "", 0.0

    row = best.iloc[0]
    return (
        str(row.get("concepto_canonico", "") or ""),
        float(row.get("score", 0.0) or 0.0),
    )


def build_known_keys(catalog: pd.DataFrame) -> tuple[set, set]:
    exact_keys = set(zip(catalog["concepto_key"], catalog["tipo_concepto_key"]))
    canonical_keys = set(catalog["canon_key"].dropna().astype(str).str.strip())
    return exact_keys, canonical_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera una cola automática de conceptos por revisar.")
    parser.add_argument(
        "--sales-source",
        default=DATA_URL,
        help="Ruta o URL del CSV de ventas.",
    )
    parser.add_argument(
        "--catalog-source",
        default=str(ROOT / "data" / "catalogo_operativo.csv"),
        help="Ruta del catálogo operativo compilado.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "catalog_review_queue.csv"),
        help="Ruta del CSV de salida.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sales = pd.read_csv(args.sales_source)
    catalog = pd.read_csv(args.catalog_source)
    catalog = catalog[catalog["compile_status"].eq("ok")].copy()

    items = build_sales_items(sales)
    if items.empty:
        print("No se encontraron items en Detalle Items.")
        return 0

    items = items[~get_void_mask(items)].copy()
    exact_keys, canonical_keys = build_known_keys(catalog)
    queue = items[
        ~items.apply(
            lambda row: (
                (row["item_key"], row["tipo_concepto_key"]) in exact_keys
                or row["item_key"] in canonical_keys
            ),
            axis=1,
        )
    ].copy()

    if queue.empty:
        print("No se encontraron conceptos pendientes por clasificar.")
        return 0

    summary = (
        queue.groupby(["item_raw", "item_key", "tipo_concepto", "tipo_concepto_key"], as_index=False)
        .agg(
            qty_total=("qty", "sum"),
            tickets=("Folio", "nunique"),
            restaurantes=("Restaurante", "nunique"),
            primera_fecha=("Fecha", "min"),
            ultima_fecha=("Fecha", "max"),
        )
        .sort_values(["qty_total", "tickets"], ascending=[False, False])
    )

    suggested = summary.apply(
        lambda row: suggest_best_match(row["item_key"], row["tipo_concepto_key"], catalog),
        axis=1,
        result_type="expand",
    )
    suggested.columns = ["suggested_canonical", "suggestion_score"]
    summary = pd.concat([summary, suggested], axis=1)
    summary["needs_manual_review"] = summary["suggestion_score"] < 0.86

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"Conceptos por revisar: {len(summary):,} -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
