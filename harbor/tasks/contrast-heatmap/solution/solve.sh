#!/bin/bash
# Reference solution for contrast-heatmap: 6x6 correlation matrix as a
# coolwarm heatmap, every cell annotated with its two-decimal value.
# Writes plot.py to disk and executes it, so the delivered figure is the one
# the script renders (the verifier re-executes it).
set -euo pipefail

cat > /app/plot.py <<'PY'
"""Correlation matrix heatmap with per-cell value annotations."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARS = ["var1", "var2", "var3", "var4", "var5", "var6"]

matrix = {}
with open("correlation.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        matrix[row[""]] = {v: float(row[v]) for v in VARS}

values = np.array([[matrix[r][c] for c in VARS] for r in VARS])

fig, ax = plt.subplots(figsize=(8, 6))
image = ax.imshow(values, cmap="coolwarm", vmin=-1.0, vmax=1.0)

ax.set_xticks(range(len(VARS)))
ax.set_xticklabels(VARS, rotation=45)
ax.set_yticks(range(len(VARS)))
ax.set_yticklabels(VARS)
ax.set_xlabel("Variable")
ax.set_ylabel("Variable")
ax.set_title("Correlation Matrix")

for i in range(len(VARS)):
    for j in range(len(VARS)):
        ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                color="black", fontsize=12)

colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label("Correlation")

fig.tight_layout()
fig.savefig("figure.png", dpi=100)

with open("plotted_values.json", "w") as handle:
    json.dump({r: {c: matrix[r][c] for c in VARS} for r in VARS}, handle, indent=2)
PY

cd /app && python plot.py