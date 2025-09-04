"""
Aggregate JSONL logs and output simple metrics (P50/P95, json_valid rate, etc.).
"""
import argparse
import glob
import json
import statistics
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize JSONL logs (latency/tokens/sec/json_valid)")
    ap.add_argument("--logs", required=True, help="Glob for JSONL files (e.g., runs/*.jsonl)")
    args = ap.parse_args()

    files = glob.glob(args.logs)
    if not files:
        print("No files matched.")
        return

    latencies = []
    tps_vals = []
    json_valid_vals = []
    for fp in files:
        for line in Path(fp).read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                if "latency_ms" in obj:
                    latencies.append(float(obj["latency_ms"]))
                if "tokens_per_sec" in obj:
                    tps_vals.append(float(obj["tokens_per_sec"]))
                if "json_valid" in obj and obj["json_valid"] is not None:
                    json_valid_vals.append(bool(obj["json_valid"]))
            except Exception:
                continue

    def pct(values, p):
        if not values:
            return None
        return statistics.quantiles(values, n=100)[p-1] if len(values) >= 100 else sorted(values)[int((p/100)*len(values))-1]

    print("Files:", len(files))
    if latencies:
        print("Latency ms:", "P50=", round(pct(latencies,50) or 0,2), "P95=", round(pct(latencies,95) or 0,2))
    if tps_vals:
        print("Tokens/sec:", "avg=", round(sum(tps_vals)/len(tps_vals),2))
    if json_valid_vals:
        print("JSON valid rate:", f"{round(100*sum(1 for v in json_valid_vals if v)/len(json_valid_vals),2)}%")


if __name__ == "__main__":
    main()

