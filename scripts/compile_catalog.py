#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog_engine import compile_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compila el catálogo editable a un catálogo operativo.")
    parser.add_argument(
        "--source",
        required=True,
        help="Ruta o URL del catálogo fuente en CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "catalogo_operativo.csv"),
        help="Ruta del CSV compilado de salida.",
    )
    parser.add_argument(
        "--issues-output",
        default=str(ROOT / "data" / "catalogo_issues.csv"),
        help="Ruta del CSV con issues de compilación.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_catalog = pd.read_csv(args.source)
    compiled, issues = compile_catalog(raw_catalog)

    output_path = Path(args.output)
    issues_path = Path(args.issues_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    issues_path.parent.mkdir(parents=True, exist_ok=True)

    compiled.to_csv(output_path, index=False)
    issues.to_csv(issues_path, index=False)

    print(f"Catalogo compilado: {len(compiled):,} filas -> {output_path}")
    print(f"Issues detectados: {len(issues):,} filas -> {issues_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
