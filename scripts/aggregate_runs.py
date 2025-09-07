"""
Aggregate results/runs.csv and print best configs by key metrics.

Usage:
  python scripts/aggregate_runs.py --csv results/runs.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def best_by(rows: List[Dict[str, str]], key: str, better: str = "min") -> Tuple[Dict[str, str], float]:
    best_row: Dict[str, str] | None = None
    best_val: float | None = None
    for row in rows:
        try:
            v = float(row.get(key, "nan"))
        except (ValueError, TypeError):
            continue
        if v != v:  # NaN
            continue
        if best_val is None:
            best_row, best_val = row, v
            continue
        if (better == "min" and v < best_val) or (better == "max" and v > best_val):
            best_row, best_val = row, v
    return best_row or {}, best_val or 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate bench results")
    ap.add_argument("--csv", default="results/runs.csv")
    args = ap.parse_args()

    rows = parse_rows(args.csv)
    if not rows:
        print("No rows in CSV.")
        return

    # Filter median rows for summary
    med = [r for r in rows if r.get("notes", "").endswith("|median")]
    if not med:
        med = rows

    best_ft, v_ft = best_by(med, "first_token_ms", better="min")
    best_tps, v_tps = best_by(med, "tokens_per_s", better="max")
    best_mem, v_mem = best_by(med, "peak_vram_alloc_mb", better="min")

    def fmt(r: Dict[str, str]) -> str:
        return f"{r.get('model')} | {r.get('quant')} | {r.get('runtime')} | attn={r.get('attn')} | bs={r.get('batch_size')}"

    print("Best first_token_ms:")
    print(f"  {v_ft:.2f} ms -> {fmt(best_ft)}")
    print("Best tokens_per_s:")
    print(f"  {v_tps:.2f} tok/s -> {fmt(best_tps)}")
    print("Best peak_vram_alloc_mb:")
    print(f"  {v_mem:.2f} MB -> {fmt(best_mem)}")


if __name__ == "__main__":
    main()

