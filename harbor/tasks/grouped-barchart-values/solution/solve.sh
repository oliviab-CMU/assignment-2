#!/bin/bash
# Reference solution for grouped-barchart-values: grouped bars, one group per
# product, Q1/Q2 side by side, value printed above each bar.
# Writes plot.py to disk and executes it, so the delivered figure is the one
# the script renders (the verifier re-executes it).
set -euo pipefail

cat > /app/plot.py <<'PY'
"""Grouped bar chart of product sales by quarter, values printed above bars."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRODUCTS = ["alpha", "beta", "gamma", "delta", "epsilon"]
QUARTERS = ["Q1", "Q2"]
COLOURS = {"Q1": "#1f77b4", "Q2": "#ff7f0e"}

sales = {p: {q: 0.0 for q in QUARTERS} for p in PRODUCTS}
with open("product_sales.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        sales[row["product"]][row["quarter"]] = float(row["sales"])

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(PRODUCTS))
width = 0.35
bars = {}
for qi, quarter in enumerate(QUARTERS):
    values = [sales[p][quarter] for p in PRODUCTS]
    bars[quarter] = ax.bar(x + (qi - 0.5) * width, values, width,
                           color=COLOURS[quarter], label=quarter)

for quarter in QUARTERS:
    for bar in bars[quarter]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                f"{bar.get_height():.0f}", ha="center", va="bottom",
                color="black", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(PRODUCTS)
ax.set_xlabel("Product")
ax.set_ylabel("Sales")
ax.set_title("Product Sales by Quarter")
ax.set_ylim(0, 320)  # headroom so the upper-right legend clears the tallest bar (250)
ax.legend(loc="upper right")

fig.tight_layout()
fig.savefig("figure.png", dpi=100)

with open("plotted_values.json", "w") as handle:
    json.dump(sales, handle, indent=2)
PY

cd /app && python plot.py