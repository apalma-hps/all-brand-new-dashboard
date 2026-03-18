from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


COUNT_AS_RE = re.compile(r"(?i)^\s*contar(?:\s+como)?\s+(.+?)\s*$")


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


def _first_non_empty(series: pd.Series, default: str = "") -> str:
    for value in series:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _join_unique(series: pd.Series) -> str:
    values = []
    for value in series:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in values:
            values.append(text)
    return " | ".join(values)


def _parse_rule(category_raw: str) -> dict:
    category_raw = str(category_raw or "").strip()
    if not category_raw:
        return {
            "rule_action": "review",
            "include_in_count": True,
            "es_remap": False,
            "concepto_canonico": "",
            "issue_reason": "missing_category",
        }

    if norm_key(category_raw) == "no contar":
        return {
            "rule_action": "exclude",
            "include_in_count": False,
            "es_remap": False,
            "concepto_canonico": "",
            "issue_reason": "",
        }

    match = COUNT_AS_RE.match(category_raw)
    if match:
        canonical_name = re.sub(r"\s+", " ", match.group(1)).strip()
        if not canonical_name:
            return {
                "rule_action": "review",
                "include_in_count": True,
                "es_remap": False,
                "concepto_canonico": "",
                "issue_reason": "invalid_count_as_rule",
            }
        return {
            "rule_action": "count_as",
            "include_in_count": True,
            "es_remap": True,
            "concepto_canonico": canonical_name,
            "issue_reason": "",
        }

    return {
        "rule_action": "direct",
        "include_in_count": True,
        "es_remap": False,
        "concepto_canonico": "",
        "issue_reason": "",
    }


def compile_catalog(raw_catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat = raw_catalog.copy()
    cat.columns = [str(col).strip() for col in cat.columns]

    for column in ["concepto", "tipo_concepto", "Categoria", "conteo_total"]:
        if column not in cat.columns:
            cat[column] = ""

    cat["source_row"] = np.arange(2, len(cat) + 2)
    cat["concepto"] = cat["concepto"].astype(str).str.strip()
    cat["tipo_concepto"] = cat["tipo_concepto"].astype(str).str.strip().str.lower()
    cat["Categoria_raw"] = cat["Categoria"].astype(str).str.strip()
    cat["conteo_total_num"] = pd.to_numeric(cat["conteo_total"], errors="coerce").fillna(0)
    cat["concepto_key"] = cat["concepto"].map(norm_key)
    cat["tipo_concepto_key"] = cat["tipo_concepto"].map(norm_key)

    parsed = cat["Categoria_raw"].map(_parse_rule).apply(pd.Series)
    cat = pd.concat([cat, parsed], axis=1)
    cat["concepto_canonico"] = np.where(
        cat["concepto_canonico"].astype(str).str.strip().eq(""),
        cat["concepto"],
        cat["concepto_canonico"],
    )
    cat["canon_key"] = cat["concepto_canonico"].map(norm_key)

    cat = cat.sort_values(
        by=["conteo_total_num", "source_row"],
        ascending=[False, True],
    )

    compiled = (
        cat.groupby(["concepto_key", "tipo_concepto_key"], dropna=False, as_index=False)
        .agg(
            concepto=("concepto", _first_non_empty),
            tipo_concepto=("tipo_concepto", _first_non_empty),
            conteo_total=("conteo_total_num", "max"),
            Categoria_raw=("Categoria_raw", _first_non_empty),
            categoria_fuente_all=("Categoria_raw", _join_unique),
            rule_action=("rule_action", _first_non_empty),
            include_in_count=("include_in_count", "max"),
            es_remap=("es_remap", "max"),
            concepto_canonico=("concepto_canonico", _first_non_empty),
            canon_key=("canon_key", _first_non_empty),
            issue_reason=("issue_reason", _first_non_empty),
            source_rows=("source_row", lambda s: ",".join(str(int(v)) for v in sorted(s.tolist()))),
            duplicate_count=("source_row", "size"),
        )
    )

    compiled["compile_status"] = "ok"

    compiled.loc[compiled["concepto_key"].eq(""), "compile_status"] = "review"
    compiled.loc[compiled["concepto_key"].eq(""), "issue_reason"] = compiled["issue_reason"].mask(
        compiled["concepto_key"].eq(""),
        "missing_concept",
    )

    missing_type_mask = compiled["tipo_concepto_key"].eq("")
    compiled.loc[missing_type_mask, "compile_status"] = "review"
    compiled.loc[missing_type_mask & compiled["issue_reason"].eq(""), "issue_reason"] = "missing_type"

    unresolved_remap_mask = (
        compiled["rule_action"].eq("count_as")
        & compiled["canon_key"].astype(str).str.strip().eq("")
    )
    compiled.loc[unresolved_remap_mask, "compile_status"] = "review"
    compiled.loc[unresolved_remap_mask & compiled["issue_reason"].eq(""), "issue_reason"] = "missing_remap_target"

    duplicate_mask = compiled["duplicate_count"] > 1
    conflict_mask = duplicate_mask & compiled["categoria_fuente_all"].str.contains(r"\|", regex=True)
    compiled.loc[conflict_mask & compiled["issue_reason"].eq(""), "issue_reason"] = "duplicate_conflict"
    compiled.loc[conflict_mask, "compile_status"] = "review"

    compiled["include_in_count"] = compiled["include_in_count"].fillna(False).astype(bool)
    compiled["es_remap"] = compiled["es_remap"].fillna(False).astype(bool)
    compiled["conteo_total"] = compiled["conteo_total"].fillna(0).astype(float)

    compiled = compiled[
        [
            "concepto",
            "tipo_concepto",
            "conteo_total",
            "Categoria_raw",
            "rule_action",
            "include_in_count",
            "es_remap",
            "concepto_canonico",
            "concepto_key",
            "tipo_concepto_key",
            "canon_key",
            "compile_status",
            "issue_reason",
            "duplicate_count",
            "source_rows",
            "categoria_fuente_all",
        ]
    ].sort_values(["compile_status", "conteo_total", "concepto"], ascending=[True, False, True])

    issues = compiled[
        compiled["compile_status"].ne("ok") | compiled["duplicate_count"].gt(1)
    ].copy()

    return compiled.reset_index(drop=True), issues.reset_index(drop=True)


def load_and_compile_catalog(source: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = pd.read_csv(source)
    return compile_catalog(catalog)
