#!/bin/bash
# MUTANT (filtered-cdf-log): drops the ">50 ms" filter and plots the CDF of
# ALL 300 latencies, not the 260 that exceed 50 ms. SCORES 0.0.
#
# Caught by the hand-derived literals, measured: test_s2_cdf_line_matches_key
# (the line has 300 points, not 260, and every checkpoint value is wrong),
# test_s5_sidecar_matches_key_and_chart (kept_count 300 vs 260, median_ms
# wrong, every f_* fraction wrong). The cheap route the assignment describes:
# the plotted_values.json sidecar plus literals in the test file means any
# mutant that changes a number fails. This is the ordinary case -- a wrong
# answer, computed honestly, that any verifier which checks the NUMBERS
# catches and a verifier which only checks "a valid PNG exists" rewards.
#
# To run it:
#   rm -rf /tmp/try && cp -r harbor/tasks/filtered-cdf-log /tmp/try
#   cp harbor/tasks/filtered-cdf-log/mutants/no_filter/solve.sh /tmp/try/solution/solve.sh
#   uv run harbor run -p /tmp/try -a oracle --job-name mutant-no_filter
set -euo pipefail

cat > /app/plot.py <<'PY'
"""CDF of ALL latencies -- the >50 ms filter is missing."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

latencies = []
with open("latencies.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        latencies.append(float(row["latency_ms"]))          # BUG: no >50 filter

latencies.sort()
n = len(latencies)
cdf = [(i + 1) / n for i in range(n)]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(latencies, cdf, color="blue", linewidth=2)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.5)
ax.set_xscale("log")
ax.set_xlim(50, 1000)
median = latencies[0]
for i in range(n):
    if cdf[i] >= 0.5:
        median = latencies[i]
        break
ax.annotate("median", xy=(median, 0.5))
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Cumulative fraction")
ax.set_title("CDF of Request Latencies above 50ms")
fig.tight_layout()
fig.savefig("figure.png", dpi=100)

stats = {
    "kept_count": n,
    "median_ms": median,
    "f_100": sum(1 for v in latencies if v <= 100) / n,
    "f_200": sum(1 for v in latencies if v <= 200) / n,
    "f_500": sum(1 for v in latencies if v <= 500) / n,
}
with open("plotted_values.json", "w") as handle:
    json.dump(stats, handle, indent=2)
PY

cd /app && python plot.py