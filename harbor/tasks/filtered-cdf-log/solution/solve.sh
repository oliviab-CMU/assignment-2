#!/bin/bash
# Reference solution for filtered-cdf-log: filter, ECDF, log axis.
# Writes plot.py to disk and executes it, so the delivered figure is the one
# the script renders (the verifier re-executes it).
set -euo pipefail

cat > /app/plot.py <<'PY'
"""Empirical CDF of request latencies above 50 ms, on a log x axis."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

latencies = []
with open("latencies.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        latencies.append(float(row["latency_ms"]))

kept = sorted(v for v in latencies if v > 50.0)   # keep only > 50 ms
n = len(kept)
cdf = [(i + 1) / n for i in range(n)]             # fraction <= each value

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(kept, cdf, color="blue", linewidth=2)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.5)
ax.set_xscale("log")
ax.set_xlim(50, 1000)
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Cumulative fraction")
ax.set_title("CDF of Request Latencies above 50ms")

# smallest kept latency whose cumulative fraction reaches 0.5
median = kept[next(i for i, v in enumerate(cdf) if v >= 0.5)]
ax.annotate("median", xy=(median, 0.5), xytext=(median * 1.1, 0.35), color="black")

fig.tight_layout()
fig.savefig("figure.png", dpi=100)

with open("plotted_values.json", "w") as handle:
    json.dump(
        {
            "kept_count": n,
            "median_ms": median,
            "f_100": sum(1 for v in kept if v <= 100) / n,
            "f_200": sum(1 for v in kept if v <= 200) / n,
            "f_500": sum(1 for v in kept if v <= 500) / n,
        },
        handle,
        indent=2,
    )
PY

cd /app && python plot.py